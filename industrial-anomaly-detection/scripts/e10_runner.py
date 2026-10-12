#!/usr/bin/env python
"""E10 - Local-Manifold Calibrated Scoring (scoring-only).

Frozen: input image / backbone / feature extraction / memory bank / coreset are untouched.
Only the patch-level score aggregation changes. S0 replicates the frozen PatchCore image score
exactly (via `PatchcoreModel.compute_anomaly_score`, which uses num_neighbors=9 softmin
reweighting). S1-S4 use max-pooling over the corresponding patch-level score (standard
PatchCore image aggregation).

Round1: Bird/Car/Cube x seed0 x view120. Baseline = E8 Original P00 (reused, 0 GPU).
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

import e4d1_m2ad_loader as LD  # noqa: E402
import e7_common as E7  # noqa: E402
import e8_common as E8  # noqa: E402
import e8_fetch as EF  # noqa: E402
import experiment1b_defect_sensitivity as e1b  # noqa: E402

OUT = ROOT / "results" / "e10"
RAW = OUT / "raw"
EPS = 1e-8

METHODS = ["S0", "S1", "S2", "S3", "S4", "S5"]
METHOD_DESC = {
    "S0": "Original (frozen softmin-reweighted max d1)",
    "S1": "kNN3 mean",
    "S2": "kNN5 trimmed (drop min/max, mean middle3)",
    "S3": "local density ratio k=3",
    "S4": "local density ratio k=5",
    "S5": "normal ECDF calibration of S0",
}


# ------------------------------------------------------------------ fit (frozen PatchCore)
def fit_patchcore(bank: list, dev, tag: str, seed: int = 0):
    from experiment5a_model import FAlphaDualLayerPatchcore
    from anomalib.engine import Engine
    # seed RNG before fit so the greedy coreset matches E8/E9's frozen Original (seed 0)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    dm = LD.make_folder_datamodule([r["image_path"] for r in bank], batch_size=16,
                                   num_workers=0, seed=E7.SEED, name=f"e10_{tag}")
    model = FAlphaDualLayerPatchcore(backbone="wide_resnet50_2", layers=["layer2", "layer3"],
                                     coreset_sampling_ratio=0.1, num_neighbors=9,
                                     alpha_l2=0.0, alpha_l3=0.0, visualizer=False)
    engine = Engine(enable_progress_bar=False, logger=False, barebones=True,
                    num_sanity_val_steps=0, limit_val_batches=0,
                    default_root_dir=str(OUT / "fit" / tag))
    engine.fit(model=model, datamodule=dm)
    tm = model.model
    tm.eval()
    e1b._move_model_to_device(tm, dev)
    return tm


# ------------------------------------------------------------------ embedding (exact forward steps)
def extract_embedding(tm, img_tensor, dev):
    inp = e1b.preprocess_for_model(img_tensor, dev).unsqueeze(0)   # (1,3,256,256) normalized
    inp = inp.type(tm.memory_bank.dtype)
    with torch.no_grad():
        features = tm.feature_extractor(inp)
    features = {layer: tm.feature_pooler(f) for layer, f in features.items()}
    emb = tm.generate_embedding(features)                          # (1, C, H, W)
    return tm.reshape_embedding(emb)                               # (P, C)


# ------------------------------------------------------------------ rho (local normal scale, normal-only)
def compute_rho(tm):
    dist, _ = tm.nearest_neighbors(tm.memory_bank, 6)   # (N,6); index 0 is self (dist 0)
    rho3 = dist[:, 1:4].mean(1)                          # 3 nearest OTHER patches
    rho5 = dist[:, 1:6].mean(1)                          # 5 nearest OTHER patches
    return rho3, rho5


# ------------------------------------------------------------------ per-image scoring (S0-S4)
def score_patches(tm, rho3, rho5, emb):
    d1, loc1 = tm.nearest_neighbors(emb, 1)              # (P,1)
    s0 = float(tm.compute_anomaly_score(d1.reshape(1, -1), loc1.reshape(1, -1), emb).item())

    d3, _ = tm.nearest_neighbors(emb, 3)                 # (P,3)
    s1 = float(d3.mean(1).max().item())

    d5, _ = tm.nearest_neighbors(emb, 5)                 # (P,5)
    s2 = float(d5[:, 1:4].mean(1).max().item())

    raw = d1            # (P,) -- nearest_neighbors(...,1) returns 1D via .min(1)
    mstar = loc1         # (P,)
    s3 = float((raw / (rho3[mstar] + EPS)).max().item())
    s4 = float((raw / (rho5[mstar] + EPS)).max().item())
    return {"S0": s0, "S1": s1, "S2": s2, "S3": s3, "S4": s4}

# ------------------------------------------------------------------ run one (category, seed) -> all 6 methods from ONE fit
def run_unit(category: str, seed: int, dev) -> dict:
    scope = EF.build_scope(category)
    bank = E8.bank_for(scope["pool"], seed)
    bank_r = [dict(r, image_path=str(E8.cache_path("l0_256", r))) for r in bank]
    test = [r for r in scope["test"] if r["view"] == "120"]
    # S5 ECDF reference = un-sampled bank-pool images (normal train, held out from the 300 bank,
    # already cached). val specimens (2) are not cached for Car/Cube, so they are NOT used here.
    bank_ids = {r["image_path"] for r in bank}
    val = [r for r in scope["pool"] if r["image_path"] not in bank_ids]

    tm = fit_patchcore(bank_r, dev, f"{category}_s{seed}", seed=seed)
    rho3, rho5 = compute_rho(tm)

    def img_path(r):
        return E8.cache_path("l0_256", r)

    # normal calibration scores (S5 ECDF reference) — val specimens, normal-only, held out
    normal_scores = []
    for r in val:
        emb = extract_embedding(tm, e1b.load_image_as_tensor(img_path(r)), dev)
        normal_scores.append(score_patches(tm, rho3, rho5, emb)["S0"])
    normal_sorted = np.sort(np.asarray(normal_scores, dtype=np.float64))

    def ecdf(x):
        return float(np.searchsorted(normal_sorted, x, side="right") / len(normal_sorted))

    rows = {m: [] for m in METHODS}
    for r in test:
        emb = extract_embedding(tm, e1b.load_image_as_tensor(img_path(r)), dev)
        s = score_patches(tm, rho3, rho5, emb)
        s["S5"] = ecdf(s["S0"])
        for m in METHODS:
            rows[m].append({"specimen": r["specimen"], "kind": r["kind"],
                            "illumination": r["illumination"], "score": s[m]})

    per_method = {}
    for m in METHODS:
        per_method[m] = E7.evaluate(rows[m], want_pixel=False)
        per_method[m]["method"] = m
        per_method[m]["n_normal_calib"] = len(normal_scores)

    return {"status": "OK", "category": category, "seed": seed, "view": "120",
            "bank_size": len(bank), "test_size": len(test), "coreset_size": int(tm.memory_bank.shape[0]),
            "per_method": per_method, "rows": rows}


# ------------------------------------------------------------------ baseline
def baseline(category: str) -> dict:
    p = ROOT / "results" / "e8" / "raw" / f"P00_{category}_seed0.json"
    d = json.loads(p.read_text())["per_view"]["120"]
    return {"auroc": float(d["image_auroc"]), "dprime": float(d["d_prime"])}


# ------------------------------------------------------------------ smoke (incl. S0 equivalence)
def smoke(dev) -> int:
    print("[E10 SMOKE] Bird/seed0 small subset + S0 equivalence vs frozen predict_one", flush=True)
    scope = EF.build_scope("Bird")
    bank = E8.bank_for(scope["pool"], 0)
    bank_r = [dict(r, image_path=str(E8.cache_path("l0_256", r))) for r in bank]
    tm = fit_patchcore(bank_r, dev, "smoke", seed=0)
    rho3, rho5 = compute_rho(tm)
    test = [r for r in scope["test"] if r["view"] == "120"]
    good = [r for r in test if r["kind"] == "Good"][:6]
    ng = [r for r in test if r["kind"] == "NG"][:6]
    sub = good + ng

    max_s0_diff = 0.0
    for r in sub:
        img = e1b.load_image_as_tensor(E8.cache_path("l0_256", r))
        emb = extract_embedding(tm, img, dev)
        s = score_patches(tm, rho3, rho5, emb)
        ref, _ = e1b.predict_one(tm, img, dev)   # frozen forward
        d = abs(s["S0"] - float(ref))
        max_s0_diff = max(max_s0_diff, d)
        fin = all(np.isfinite(v) for v in s.values())
        print(f"  {r['kind']:4s} {r['specimen']:8s} S0={s['S0']:.5f} ref={ref:.5f} diff={d:.2e} "
              f"S1={s['S1']:.5f} S2={s['S2']:.5f} S3={s['S3']:.5f} S4={s['S4']:.5f} finite={fin}",
              flush=True)
    print(f"[E10 SMOKE] max |S0 - frozen predict_one| = {max_s0_diff:.3e} "
          f"({'PASS' if max_s0_diff <= 1e-6 else 'FAIL'})", flush=True)
    # rho sanity
    ok_rho = bool(torch.isfinite(rho3).all() and torch.isfinite(rho5).all()
                  and (rho3 > 0).all() and (rho5 > 0).all())
    print(f"[E10 SMOKE] rho3/rho5 finite & >0 = {ok_rho}  (mean rho3={rho3.mean().item():.4f}, "
          f"rho5={rho5.mean().item():.4f})", flush=True)
    return 0 if (max_s0_diff <= 1e-6 and ok_rho) else 2


# ------------------------------------------------------------------ round1
def round1(dev) -> int:
    units = [(cat, 0) for cat in ("Bird", "Car", "Cube")]
    tracker = E8.EtaTracker(len(units), "E10-Round1")
    tracker.stage_announce("Round 1 (3 cats x seed0 x view120, 6 methods per fit)", len(units) * 4)
    RAW.mkdir(parents=True, exist_ok=True)
    for cat, seed in units:
        up = RAW / f"{cat}_seed{seed}.json"
        if up.exists():
            print(f"[E10 R1] SKIP {cat}/s{seed}", flush=True)
            continue
        name = f"{cat}/s{seed}"
        tracker.start_unit()
        print(f"\n[E10 R1] START {name} (fit once -> 6 methods)", flush=True)
        try:
            out = run_unit(cat, seed, dev)
        except Exception as exc:
            import traceback
            traceback.print_exc()
            out = {"status": "FAILED", "reason": f"{type(exc).__name__}: {exc}"}
        if out["status"] == "OK":
            base = baseline(cat)
            out["baseline"] = base
            for m in METHODS:
                mm = out["per_method"][m]
                mm["d_auroc"] = mm["image_auroc"] - base["auroc"]
                mm["d_dprime"] = mm["d_prime"] - base["dprime"]
            E8.save_json(up, out)
            tracker.finish_unit(name, extra=f"fit-coreset={out['coreset_size']}")
        else:
            E8.save_json(up, out)
            tracker.finish_unit(name, extra=f"FAILED {out.get('reason','')[:50]}")
        tracker.heartbeat_if_due()
    print("\n[E10 R1] done", flush=True)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="smoke", help="smoke|round1")
    args = ap.parse_args()
    assert torch.cuda.is_available()
    dev = torch.device("cuda")
    print(f"[E10] GPU {torch.cuda.get_device_properties(0).name} | stage={args.stage} | "
          f"disk {E8.free_gb():.1f} GB", flush=True)
    if args.stage == "smoke":
        return smoke(dev)
    if args.stage == "round1":
        return round1(dev)
    print("unknown stage", args.stage)
    return 3


if __name__ == "__main__":
    raise SystemExit(main())
