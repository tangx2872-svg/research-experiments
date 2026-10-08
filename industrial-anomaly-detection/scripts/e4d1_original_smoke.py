#!/usr/bin/env python
"""E4-D1 — M²AD Bird x seed0 x Original PatchCore GPU smoke.

Protocol frozen in `docs/E4_D1_PROTOCOL.md` BEFORE any GPU run.
Runs ONLY Original (alpha_l2 = alpha_l3 = 0). No B2. No X6c. No extra seeds/categories.
Writes to `results/e4_d1_smoke/` only (output isolation, sanity S8).

Usage:
  HF_ENDPOINT=https://hf-mirror.com python scripts/e4d1_original_smoke.py
"""
from __future__ import annotations

import csv
import json
import random
import resource
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

import experiment1b_defect_sensitivity as e1b  # noqa: E402
from experiment1h_runner import image_auroc, pixel_auroc_from_maps  # noqa: E402
from experiment5a_h_runner import compute_aupro  # noqa: E402
from experiment5a_model import FAlphaDualLayerPatchcore, FAlphaDualLayerPatchcoreModel  # noqa: E402
import e4d1_m2ad_loader as LD  # noqa: E402

from anomalib.engine import Engine  # noqa: E402
from anomalib.models.image.patchcore.torch_model import PatchcoreModel  # noqa: E402

OUT = ROOT / "results" / "e4_d1_smoke"
SEED = 0
BATCH = 16
PROGRESS_EVERY = 200


# --------------------------------------------------------------------------- utils
def ram_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0     # KB -> MB


def vram() -> tuple[float, float]:
    return (torch.cuda.memory_allocated() / 2**20, torch.cuda.memory_reserved() / 2**20)


def dprime(good: np.ndarray, defect: np.ndarray) -> float:
    """Exp16 formula (pooled, no defect-type stratification)."""
    if len(good) < 2 or len(defect) < 2:
        return float("nan")
    pooled = float(np.sqrt((defect.std(ddof=1) ** 2 + good.std(ddof=1) ** 2) / 2.0))
    return float((defect.mean() - good.mean()) / pooled) if pooled > 0 else float("nan")


def progress(done: int, total: int, t0: float, label: str, t_stage0: float | None = None) -> None:
    el = time.time() - t0
    rate = done / el if el > 0 else 0.0
    remain = (total - done) / rate if rate > 0 else float("inf")
    eta = time.strftime("%H:%M:%S", time.localtime(time.time() + remain)) if np.isfinite(remain) else "n/a"
    a, r = vram()
    print(f"  [{label}] {done:5d}/{total:<5d} {100.0*done/max(1,total):5.1f}%  "
          f"elapsed={time.strftime('%H:%M:%S', time.gmtime(el))}  ETA={eta} "
          f"(remain {remain/60:.1f} min)  {rate:6.1f} img/s  "
          f"VRAM alloc={a:7.1f}MB reserved={r:7.1f}MB  RAM={ram_mb():7.1f}MB", flush=True)


# --------------------------------------------------------------------------- sanity
def sanity_pre(bank: list[dict], val: list[dict], score: list[dict], all_recs: list[dict]) -> dict:
    dev = torch.device("cpu")
    res: dict[str, dict] = {}

    # S1 metadata / files
    res["S1_metadata_files"] = {"pass": len(all_recs) == 12000,
                                "detail": f"records with existing files = {len(all_recs)} (expect 12000)"}

    # S2 alpha=0 equivalence vs plain PatchcoreModel
    x = torch.randn(1, 3, 256, 256)
    base = PatchcoreModel(backbone="wide_resnet50_2", layers=["layer2", "layer3"],
                          pre_trained=True, num_neighbors=9)
    dual = FAlphaDualLayerPatchcoreModel(backbone="wide_resnet50_2", layers=["layer2", "layer3"],
                                         pre_trained=True, num_neighbors=9, alpha_l2=0.0, alpha_l3=0.0)
    base.eval(); dual.eval()
    with torch.no_grad():
        feats = base.feature_extractor(x)
        diff = float((base.generate_embedding(feats) - dual.generate_embedding(feats)).abs().max().item())
    res["S2_alpha0_equivalence"] = {"pass": diff < 1e-6, "detail": f"max|delta| = {diff:.3e}"}

    # S3 counts
    ok = (len(bank), len(val), len(score)) == (1200, 240, 8400)
    res["S3_counts"] = {"pass": ok, "detail": f"bank={len(bank)} val={len(val)} scoring={len(score)}"}

    # S4 view x illumination cartesian completeness per specimen
    from collections import defaultdict
    per = defaultdict(set)
    for r in score:
        per[r["specimen"]].add((r["view"], r["illumination"]))
    incomplete = {s: len(v) for s, v in per.items() if len(v) != 120}
    res["S4_cartesian"] = {"pass": not incomplete,
                           "detail": f"{len(per)} specimens; incomplete={incomplete if incomplete else 'none'}"}

    # S5 mask alignment (sample)
    rnd = random.Random(SEED)
    cand = [r for r in score if r["mask_path"]]
    picked = rnd.sample(cand, 10)
    from PIL import Image
    bad = []
    for r in picked:
        with Image.open(Path(r["image_path"])) as im, Image.open(Path(r["mask_path"])) as mk:
            if im.size != mk.size:
                bad.append(r["image_path"])
    res["S5_mask_alignment"] = {"pass": not bad, "detail": f"10 sampled masks; size mismatch={bad}"}

    # S7 value domains
    ill = sorted({r["illumination"] for r in score}); vw = sorted({r["view"] for r in score})
    res["S7_domains"] = {"pass": ill == LD.ILLUMINATIONS and len(vw) == 12,
                         "detail": f"illuminations={ill} views(n)={len(vw)}"}

    # S8 output isolation: no pre-existing smoke artifact; nothing written outside OUT
    pre = [p.name for p in (OUT / "per_image.csv", OUT / "info.json", OUT / "sanity.json") if p.exists()]
    res["S8_output_isolation"] = {"pass": not pre,
                                  "detail": f"OUT={OUT} pre-existing smoke artifacts={pre or 'none'}"}

    # S9 no B2 / X6c — RUNTIME check (not source-text matching)
    fusion_mods = sorted({m for m in sys.modules
                          if any(t in m for t in ("experiment16", "experiment14_model",
                                                  "experiment14_candidates", "experiment11_candidates"))})
    alpha_ok = (abs(dual.alpha_l2) == 0.0 and abs(dual.alpha_l3) == 0.0)
    res["S9_no_b2_x6c"] = {"pass": alpha_ok and not fusion_mods,
                           "detail": f"model alpha=({dual.alpha_l2}, {dual.alpha_l3}); "
                                     f"fusion modules loaded={fusion_mods or 'none'}"}
    return res


# --------------------------------------------------------------------------- main
def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    protected = [OUT / n for n in ("per_image.csv", "info.json", "sanity.json")]
    if any(p.exists() for p in protected):
        print(f"[ABORT] existing smoke artifacts found (refuse to overwrite): "
              f"{[str(p) for p in protected if p.exists()]}")
        return 2

    assert torch.cuda.is_available()
    dev = torch.device("cuda")
    props = torch.cuda.get_device_properties(0)
    print(f"[gpu] {props.name}  {props.total_memory/2**30:.1f} GiB", flush=True)

    all_recs = LD.load_records()
    est = LD.estimate_scope(all_recs)
    bank = sorted(LD.select(all_recs, set(est["bank_specimens"])), key=lambda r: r["image_path"])
    val = sorted(LD.select(all_recs, set(est["val_specimens"])), key=lambda r: r["image_path"])
    score = sorted([r for r in LD.select(all_recs, views=None) if r["split"] == "test"],
                   key=lambda r: r["image_path"])
    print(f"[scope] bank={len(bank)} val={len(val)} scoring={len(score)}", flush=True)

    san = sanity_pre(bank, val, score, all_recs)
    for k, v in san.items():
        print(f"  sanity {k}: {'PASS' if v['pass'] else 'FAIL'}  {v['detail']}", flush=True)
    failed = [k for k, v in san.items() if not v["pass"]]
    (OUT / "sanity_pre.json").write_text(json.dumps(san, indent=2))
    if failed:
        print(f"[ABORT] sanity FAIL: {failed}")
        return 3

    stages: dict[str, float] = {}
    T_ALL = time.time()

    # ---------------- fit
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    dm = LD.make_folder_datamodule([r["image_path"] for r in bank],
                                   val_paths=[r["image_path"] for r in val],
                                   batch_size=BATCH, num_workers=0, seed=SEED, name="e4d1")
    stages["datamodule_setup_s"] = time.time() - t0
    print(f"[fit] train_data={len(dm.train_data)}  setup={stages['datamodule_setup_s']:.2f}s", flush=True)

    model = FAlphaDualLayerPatchcore(
        backbone="wide_resnet50_2", layers=["layer2", "layer3"],
        coreset_sampling_ratio=0.1, num_neighbors=9,
        alpha_l2=0.0, alpha_l3=0.0, visualizer=False)
    engine = Engine(enable_progress_bar=False, logger=False, barebones=True,
                    num_sanity_val_steps=0, limit_val_batches=0,
                    default_root_dir=str(OUT / "fit"))
    t0 = time.time()
    engine.fit(model=model, datamodule=dm)
    stages["fit_s"] = time.time() - t0
    tm = model.model; tm.eval()
    coreset = int(tm.memory_bank.shape[0])
    stages["peak_vram_after_fit_mb"] = round(torch.cuda.max_memory_allocated() / 2**20, 1)
    print(f"[fit] done {stages['fit_s']:.2f}s coreset={coreset} "
          f"peak_alloc={stages['peak_vram_after_fit_mb']}MB", flush=True)
    e1b._move_model_to_device(tm, dev)

    # ---------------- tau_val
    t0 = time.time()
    val_scores = []
    for i, r in enumerate(val):
        s, _ = e1b.predict_one(tm, e1b.load_image_as_tensor(Path(r["image_path"])), dev)
        val_scores.append(s)
        if (i + 1) % 60 == 0:
            progress(i + 1, len(val), t0, "val")
    stages["val_scoring_s"] = time.time() - t0
    tau_val = float(max(val_scores))
    print(f"[val] tau_val={tau_val:.6f} mean={np.mean(val_scores):.6f} "
          f"std={np.std(val_scores):.6f}  {stages['val_scoring_s']:.1f}s", flush=True)

    # ---------------- scoring
    t0 = time.time()
    rows, good_maps, defect_maps, defect_masks = [], [], [], []
    for i, r in enumerate(score, 1):
        s, amap = e1b.predict_one(tm, e1b.load_image_as_tensor(Path(r["image_path"])), dev)
        if not np.isfinite(s):
            raise FloatingPointError(f"non-finite score: {r['image_path']}")
        rows.append({
            "split": r["split"], "specimen": r["specimen"], "kind": r["kind"],
            "object_anomaly": r["object_anomaly"], "image_anomaly": r["image_anomaly"],
            "detectable": r["detectable"], "defect_types": r["defect_types"],
            "view": r["view"], "illumination": r["illumination"],
            "image_path": r["image_path"], "mask_path": r["mask_path"], "score": s,
        })
        if r["object_anomaly"] == 0:
            good_maps.append(amap)
        else:
            defect_maps.append(amap)
            defect_masks.append(e1b.load_mask(Path(r["mask_path"])) if r["mask_path"]
                                else np.zeros((1, 1), dtype=np.uint8))
        if i % PROGRESS_EVERY == 0 or i == len(score):
            progress(i, len(score), t0, "score")
    stages["scoring_s"] = time.time() - t0
    stages["peak_vram_alloc_mb"] = round(torch.cuda.max_memory_allocated() / 2**20, 1)
    stages["peak_vram_reserved_mb"] = round(torch.cuda.max_memory_reserved() / 2**20, 1)
    stages["peak_ram_mb"] = round(ram_mb(), 1)

    # S6 finiteness
    finite = all(np.isfinite(r["score"]) for r in rows)
    san["S6_scores_finite"] = {"pass": bool(finite), "detail": f"n={len(rows)}"}

    # ---------------- metrics
    t0 = time.time()
    s_good = np.array([r["score"] for r in rows if r["object_anomaly"] == 0])
    s_def = np.array([r["score"] for r in rows if r["object_anomaly"] == 1])
    overall = {"image_auroc": float(image_auroc(s_good, s_def)), "d_prime": dprime(s_good, s_def),
               "n_good": len(s_good), "n_defect": len(s_def)}

    ill_rows = []
    for il in LD.ILLUMINATIONS:
        g = np.array([r["score"] for r in rows if r["illumination"] == il and r["object_anomaly"] == 0])
        d = np.array([r["score"] for r in rows if r["illumination"] == il and r["object_anomaly"] == 1])
        ill_rows.append({"illumination": il, "n_good": len(g), "n_defect": len(d),
                         "image_auroc": float(image_auroc(g, d)), "d_prime": dprime(g, d),
                         "good_mean": float(g.mean()), "defect_mean": float(d.mean())})

    view_rows = []
    for vw in LD.VIEWS:
        g = np.array([r["score"] for r in rows if r["view"] == vw and r["object_anomaly"] == 0])
        d = np.array([r["score"] for r in rows if r["view"] == vw and r["object_anomaly"] == 1])
        view_rows.append({"view": vw, "n_good": len(g), "n_defect": len(d),
                          "image_auroc": float(image_auroc(g, d)), "d_prime": dprime(g, d),
                          "good_mean": float(g.mean()), "defect_mean": float(d.mean())})

    # cross-illumination dispersion per (specimen, view)
    from collections import defaultdict
    grp = defaultdict(list)
    for r in rows:
        grp[(r["specimen"], r["view"], r["kind"])].append(r["score"])
    disp = defaultdict(list)
    disp_rows = []
    for (spec, vw, kind), sc in sorted(grp.items()):
        if len(sc) != 10:
            continue
        a = np.array(sc)
        std = float(a.std(ddof=1))
        disp[kind].append(std)
        disp_rows.append({"specimen": spec, "view": vw, "kind": kind, "n_illum": len(sc),
                          "score_std": std, "score_range": float(a.max() - a.min()),
                          "score_mean": float(a.mean())})
    disp_summary = {}
    for kind in ("Good", "NG"):
        arr = np.array(disp[kind])
        if len(arr):
            disp_summary[kind] = {"n_pairs": int(len(arr)), "mean_std": float(arr.mean()),
                                  "median_std": float(np.median(arr)), "max_std": float(arr.max())}

    px = float("nan"); aupro_v = float("nan")
    try:
        px = float(pixel_auroc_from_maps(good_maps, defect_maps, defect_masks))
    except Exception as exc:  # noqa: BLE001
        print(f"[warn] pixel_auroc failed: {exc}")
    try:
        aupro_v = float(compute_aupro(good_maps, defect_maps, defect_masks))
    except Exception as exc:  # noqa: BLE001
        print(f"[warn] aupro failed: {exc}")
    overall["pixel_auroc"] = px
    overall["aupro"] = aupro_v
    stages["analysis_s"] = time.time() - t0

    # ---------------- persistence
    for name, data in (("per_image.csv", rows), ("illumination_metrics.csv", ill_rows),
                       ("view_metrics.csv", view_rows), ("dispersion_pairs.csv", disp_rows)):
        with open(OUT / name, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(data[0].keys())); w.writeheader(); w.writerows(data)

    stages["total_s"] = time.time() - T_ALL
    info = {
        "status": "OK", "category": LD.CATEGORY, "seed": SEED, "method": "Original (alpha_l2=0, alpha_l3=0)",
        "coreset_size": coreset, "tau_val": tau_val,
        "n_bank": len(bank), "n_val": len(val), "n_scoring": len(rows),
        "bank_specimens": est["bank_specimens"], "val_specimens": est["val_specimens"],
        "stages_seconds": {k: round(v, 2) for k, v in stages.items()},
        "gpu": {"name": props.name, "total_MiB": round(props.total_memory / 2**20, 1)},
        "overall": overall, "dispersion_summary": disp_summary,
        "images_per_s_scoring": round(len(rows) / stages["scoring_s"], 3),
        "illumination_auroc_min": min(r["image_auroc"] for r in ill_rows),
        "illumination_auroc_max": max(r["image_auroc"] for r in ill_rows),
        "view_auroc_min": min(r["image_auroc"] for r in view_rows),
        "view_auroc_max": max(r["image_auroc"] for r in view_rows),
    }
    (OUT / "info.json").write_text(json.dumps(info, indent=2))
    (OUT / "sanity.json").write_text(json.dumps(san, indent=2))

    print("\n===== E4-D1 ORIGINAL SMOKE SUMMARY =====", flush=True)
    print(json.dumps({k: info[k] for k in ("status", "coreset_size", "tau_val", "overall",
                                           "dispersion_summary", "images_per_s_scoring",
                                           "illumination_auroc_min", "illumination_auroc_max",
                                           "view_auroc_min", "view_auroc_max")}, indent=2), flush=True)
    print("\n[stages]", json.dumps(info["stages_seconds"], indent=2), flush=True)
    print(f"[written] {OUT}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
