#!/usr/bin/env python
"""E24 - photometric-augmentation sensitivity audit of frozen R1-ILA (M2AD only).

IMPORTANT: E24 is a POST-HOC explanatory audit. It never re-defines, re-selects or
re-tunes the frozen method used in E17/E22/E25, and it never looks at RobustAD
target/test results. Any "better" setting found here is reported as a candidate
configuration only; the frozen definitions stay unchanged.

Three stages (all on M2AD, official metadata labels, view120, 300-bank):
  strength  : S05..S30 x Bird/Car/Cube x seed 0                 (6 levels)
  type      : A0..A6  x Bird/Car/Cube x seed 0                 (7 subsets)
  stability : chosen strengths x Bird/Car/Cube x seeds 0/1/2

Injection mechanism (mirrors the E17 BETA injection precedent):
  e13a_screen.AUG_LEVELS / AUG_FAMILIES / AUG_LIST are module globals read at call
  time by this runner; `e13a_screen.prep_image(img, fam, lv)` consumes the explicit
  (family, level) pair. No frozen file is edited. BETA / lr / steps / patience /
  batch_size are asserted unchanged against results/e17/final_method.json.

Reused imports only: e4d1_m2ad_loader, e8_common, e11_p2_runner (route/scoring/metrics),
e13a_screen (model/prep/adapters), e13b_screen.train_r1 (frozen training loop).

CLI:
  python scripts/e24_runner.py --stage strength --strengths S05,S10,S15,S20,S25,S30
  python scripts/e24_runner.py --stage type --strength S15 --types A0,A1,A2,A3,A4,A5,A6
  python scripts/e24_runner.py --stage stability --strengths S10,S15 --seeds 0,1,2
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402

import e4d1_m2ad_loader as LD  # noqa: E402
import e8_common as E8  # noqa: E402
import e13a_screen as E13A  # noqa: E402
from e13a_screen import (load_dinov2_local, freeze_backbone, LinearAdapter,  # noqa: E402
                         adapted_feat_adapter, prep_image, forward_frozen_batch,
                         l2norm, D, N_PATCH)
from e13b_screen import train_r1  # noqa: E402
from e11_p2_runner import raw_test_map, official_metrics, route, nn_image_scores  # noqa: E402

OUTDIR = ROOT / "results" / "e24"
CELLDIR = OUTDIR / "cells"
PARTIAL = OUTDIR / "e24_partial.json"
FINAL_METHOD = ROOT / "results" / "e17" / "final_method.json"

STRENGTHS = {
    "S05": [0.95, 1.05],
    "S10": [0.90, 1.10],
    "S15": [0.85, 1.15],          # = the frozen E17 setting
    "S20": [0.80, 1.20],
    "S25": [0.75, 1.25],
    "S30": [0.70, 1.30],
}
TYPES = {
    "A0": ["brightness"],
    "A1": ["contrast"],
    "A2": ["gamma"],
    "A3": ["brightness", "contrast"],
    "A4": ["brightness", "gamma"],
    "A5": ["contrast", "gamma"],
    "A6": ["brightness", "contrast", "gamma"],
}
CATS = ["Bird", "Car", "Cube"]
TRAIN_N = 240
VAL_N = 60
EXTRACT_BS = 8
AUG_BS = 6
TOL = 1e-6


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def hms(s: float) -> str:
    s = int(max(0, s))
    return f"{s // 3600}h{s % 3600 // 60}m{s % 60}s"


def write_json_atomic(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1, default=float)
    os.replace(tmp, path)


def free_gb() -> float:
    st = os.statvfs(str(ROOT))
    return st.f_bavail * st.f_frsize / 2 ** 30


# ------------------------------------------------------------------ frozen guards
def assert_frozen_hyperparams() -> None:
    fm = json.loads(FINAL_METHOD.read_text())["parameters"]
    checks = {
        "BETA": (E13A.BETA, fm["BETA"]),
        "LR_ADAPTER": (E13A.LR_ADAPTER, fm["LR_ADAPTER"]),
        "MAX_STEPS": (E13A.MAX_STEPS, fm["MAX_STEPS"]),
        "PATIENCE_STEPS": (E13A.PATIENCE_STEPS, fm["PATIENCE_STEPS"]),
        "EVAL_EVERY": (E13A.EVAL_EVERY, fm["EVAL_EVERY"]),
        "BATCH_IMGS": (E13A.BATCH_IMGS, fm["BATCH_IMGS"]),
        "TRAIN_N": (TRAIN_N, fm["TRAIN_N"]),
        "VAL_N": (VAL_N, fm["VAL_N"]),
    }
    bad = {k: v for k, v in checks.items() if v[0] != v[1]}
    if bad:
        raise RuntimeError(f"frozen hyperparameter mismatch: {bad}")
    if (D, N_PATCH) != (768, 1024):
        raise RuntimeError(f"D/N_PATCH changed: {D}/{N_PATCH}")
    print(f"[E24] frozen params intact: BETA={E13A.BETA} lr={E13A.LR_ADAPTER} "
          f"steps={E13A.MAX_STEPS}/{E13A.PATIENCE_STEPS} batch={E13A.BATCH_IMGS} "
          f"bank={E8.BANK_SIZE} train/val={TRAIN_N}/{VAL_N}", flush=True)


def apply_aug_config(families: list[str], levels: list[float]) -> list[tuple]:
    """Inject the augmentation family/strength under test (module-global pattern)."""
    E13A.AUG_LEVELS = list(levels)
    E13A.AUG_FAMILIES = list(families)
    E13A.AUG_LIST = [(f, lv) for f in families for lv in levels]
    expect = [(f, lv) for f in families for lv in levels]
    if list(E13A.AUG_LIST) != expect:
        raise RuntimeError(f"AUG_LIST injection failed: {E13A.AUG_LIST} != {expect}")
    return list(E13A.AUG_LIST)


# ------------------------------------------------------------------ features
def frozen_feats(model, paths: list[str], dev) -> np.ndarray:
    out = np.empty((len(paths), N_PATCH, D), dtype="float32")
    for i0 in range(0, len(paths), EXTRACT_BS):
        chunk = paths[i0:i0 + EXTRACT_BS]
        tens = np.stack([prep_image(Image.open(p).convert("RGB")) for p in chunk])
        out[i0:i0 + len(chunk)] = forward_frozen_batch(model, tens, dev)
    return out


def aug_feats(model, paths: list[str], aug_list: list[tuple], dev) -> np.ndarray:
    K = len(aug_list)
    out = np.empty((len(paths), K, N_PATCH, D), dtype="float32")
    for i0 in range(0, len(paths), AUG_BS):
        chunk = paths[i0:i0 + AUG_BS]
        tens = np.empty((len(chunk) * K, 3, 448, 448), dtype="float32")
        for j, p in enumerate(chunk):
            img = Image.open(p).convert("RGB")
            for k, (fam, lv) in enumerate(aug_list):
                tens[j * K + k] = prep_image(img, fam, lv)
        f = forward_frozen_batch(model, tens, dev)
        out[i0:i0 + len(chunk)] = f.reshape(len(chunk), K, N_PATCH, D)
    return out


# ------------------------------------------------------------------ one cell
def run_cell(cat: str, seed: int, families: list[str], levels: list[float],
             label: str, dev, smoke: bool = False) -> dict:
    t0 = time.time()
    torch.cuda.reset_peak_memory_stats()
    aug_list = apply_aug_config(families, levels)
    recs = LD.load_records(category=cat, require_exists=False)
    est = LD.estimate_scope(recs)
    pool = sorted(LD.select(recs, set(est["bank_specimens"])), key=lambda r: r["image_path"])
    bank = E8.bank_for(pool, seed)
    test = [r for r in LD.select(recs, views=["120"]) if r["split"] == "test"]
    bank_r, test_r = route(bank, "l0_448"), route(test, "l0_448")
    rel = lambda r: str(Path(r["image_path"]).relative_to(LD.DATA_ROOT))
    rawmap = raw_test_map(cat)
    missing = [r["image_path"] for r in bank_r + test_r if not Path(r["image_path"]).exists()]
    if missing:
        raise RuntimeError(f"{cat}|{seed}: {len(missing)} cache files missing")

    if smoke:
        bank_r, bank = bank_r[:40], bank[:40]
        test_r, test = test_r[:40], test[:40]

    model = load_dinov2_local(dev).to(dev)
    freeze_backbone(model)
    bank_paths = [r["image_path"] for r in bank_r]
    test_paths = [r["image_path"] for r in test_r]
    bank_f = frozen_feats(model, bank_paths, dev)
    test_f = frozen_feats(model, test_paths, dev)
    bank_aug = aug_feats(model, bank_paths, aug_list, dev)

    b0 = bank_f.reshape(-1, D).astype("float32")
    b0 /= np.maximum(np.linalg.norm(b0, axis=1, keepdims=True), 1e-12)
    t0f = test_f.reshape(-1, D).astype("float32")
    t0f /= np.maximum(np.linalg.norm(t0f, axis=1, keepdims=True), 1e-12)
    s0 = nn_image_scores(b0, t0f, dev)
    met0 = official_metrics(E13A._rows(test, s0, rawmap, rel))

    train_n = 32 if smoke else TRAIN_N
    val_n = 16 if smoke else VAL_N
    tr, tr_aug = bank_f[:train_n], bank_aug[:train_n]
    va, va_aug = bank_f[train_n:train_n + val_n], bank_aug[train_n:train_n + val_n]
    torch.manual_seed(seed)
    adapter = LinearAdapter().to(dev)
    n_params = sum(p.numel() for p in adapter.parameters())
    if n_params != 589824:
        raise RuntimeError(f"trainable params {n_params} != 589824")
    trep = train_r1(adapter, tr, tr_aug, va, va_aug, dev, seed)
    bank_af = adapted_feat_adapter(adapter, bank_f, dev)
    test_af = adapted_feat_adapter(adapter, test_f, dev)
    bn = l2norm(torch.from_numpy(bank_af).to(dev)).cpu().numpy().astype("float32")
    tn = l2norm(torch.from_numpy(test_af).to(dev)).cpu().numpy().astype("float32")
    s1 = nn_image_scores(bn, tn, dev)
    met1 = official_metrics(E13A._rows(test, s1, rawmap, rel))

    out = {"category": cat, "seed": seed, "view": "120", "label": label,
           "families": list(families), "levels": list(levels),
           "n_aug_variants": len(aug_list),
           "n_bank": len(bank), "n_test": len(test),
           "n_ok": met0["n_ok"], "n_ng": met0["n_ng"], "n_filtered": met0["n_filtered"],
           "f0": {"auroc": met0["auroc"], "ap": met0["ap"]},
           "r1": {"auroc": met1["auroc"], "ap": met1["ap"]},
           "d_auroc": met1["auroc"] - met0["auroc"],
           "d_ap": met1["ap"] - met0["ap"],
           "steps_used": trep["steps_used"], "best_val": trep["best_val"],
           "loss_finite": trep["loss_finite"],
           "pairwise_cos": trep["after"]["pairwise_cos"],
           "trainable_params": n_params,
           "bank_hash": hashlib.sha256("\n".join(r["image_path"] for r in bank).encode()).hexdigest(),
           "runtime_s": round(time.time() - t0, 1),
           "peak_gpu_mb": round(torch.cuda.max_memory_allocated() / 2 ** 20, 1),
           "scores_f0": [float(x) for x in s0], "scores_r1": [float(x) for x in s1],
           "labels": [int(m["image_anomaly"]) for m in
                      (rawmap[rel(r)] for r in test)],
           "files": [rel(r) for r in test],
           "timestamp": now_iso(), "status": "OK"}
    del model, bank_f, test_f, bank_aug, bank_af, test_af, adapter
    torch.cuda.empty_cache()
    return out


# ------------------------------------------------------------------ summaries
def cell_path(stage: str, label: str, cat: str, seed: int) -> Path:
    return CELLDIR / f"{stage}_{label}_{cat}_s{seed}.json"


def refresh_csv(stage: str) -> None:
    rows = []
    for p in sorted(CELLDIR.glob(f"{stage}_*.json")):
        c = json.loads(p.read_text())
        if c.get("status") != "OK":
            continue
        rows.append({"stage": stage, "label": c["label"], "category": c["category"],
                     "seed": c["seed"], "n_aug": c["n_aug_variants"],
                     "n_ok": c["n_ok"], "n_ng": c["n_ng"],
                     "auroc_f0": c["f0"]["auroc"], "auroc_r1": c["r1"]["auroc"],
                     "d_auroc": c["d_auroc"], "ap_f0": c["f0"]["ap"],
                     "ap_r1": c["r1"]["ap"], "d_ap": c["d_ap"],
                     "steps_used": c["steps_used"], "runtime_s": c["runtime_s"],
                     "peak_gpu_mb": c["peak_gpu_mb"]})
    rows.sort(key=lambda r: (r["label"], r["category"], r["seed"]))
    name = {"strength": "strength_round1.csv", "type": "type_ablation.csv",
            "stability": "strength_round2.csv"}[stage]
    path = OUTDIR / name
    if not rows:
        return
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.6f}" if isinstance(v, float) and k.startswith(
                ("auroc", "ap", "d_")) else v) for k, v in r.items()})
    print(f"[E24] wrote {path} ({len(rows)} rows)", flush=True)


def cell_ok(p: Path) -> bool:
    if not p.exists():
        return False
    try:
        return json.loads(p.read_text()).get("status") == "OK"
    except json.JSONDecodeError:
        return False


# ------------------------------------------------------------------ main
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["strength", "type", "stability"])
    ap.add_argument("--strengths", default="S05,S10,S15,S20,S25,S30")
    ap.add_argument("--types", default="A0,A1,A2,A3,A4,A5,A6")
    ap.add_argument("--single-strength", default="S15")
    ap.add_argument("--categories", default=",".join(CATS))
    ap.add_argument("--seeds", default="0")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    dev = torch.device("cuda")
    OUTDIR.mkdir(parents=True, exist_ok=True)
    CELLDIR.mkdir(parents=True, exist_ok=True)
    assert_frozen_hyperparams()
    cats = [c.strip() for c in args.categories.split(",") if c.strip()]
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    print(f"[E24] stage={args.stage} cats={cats} seeds={seeds} smoke={args.smoke} | "
          f"disk {free_gb():.1f} GB", flush=True)

    if args.stage == "strength":
        combos = [(lab, TYPES["A6"], STRENGTHS[lab]) for lab in
                  [s.strip() for s in args.strengths.split(",") if s.strip()]]
    elif args.stage == "type":
        combos = [(lab, TYPES[lab], STRENGTHS[args.single_strength]) for lab in
                  [s.strip() for s in args.types.split(",") if s.strip()]]
    else:
        combos = [(lab, TYPES["A6"], STRENGTHS[lab]) for lab in
                  [s.strip() for s in args.strengths.split(",") if s.strip()]]

    todo = [(lab, fam, lev, cat, seed) for (lab, fam, lev) in combos
            for seed in seeds for cat in cats]
    total = len(todo)
    done = 0
    t_start = time.time()
    for i, (lab, fam, lev, cat, seed) in enumerate(todo, 1):
        p = cell_path(args.stage, lab, cat, seed)
        if cell_ok(p) and not args.smoke:
            print(f"[E24] SKIP {p.name} (done)", flush=True)
            done += 1
            continue
        print(f"\n[E24] === {args.stage} {lab} {cat} seed{seed} ({i}/{total}) "
              f"fam={fam} lev={lev} ===", flush=True)
        try:
            cell = run_cell(cat, seed, fam, lev, lab, dev, smoke=args.smoke)
        except Exception as exc:  # noqa: BLE001
            import traceback
            traceback.print_exc()
            print(f"!! [E24] {lab}/{cat}/s{seed} FAILED: {type(exc).__name__}: {exc}",
                  flush=True)
            return 2
        if args.smoke:
            print(f"[E24 smoke] auroc {cell['f0']['auroc']:.4f} -> {cell['r1']['auroc']:.4f} "
                  f"steps={cell['steps_used']} shape_ok={cell['n_aug'] == len(fam) * len(lev)}",
                  flush=True)
            return 0
        cell["stage"] = args.stage
        write_json_atomic(p, cell)
        refresh_csv(args.stage)
        done += 1
        el = time.time() - t_start
        eta = el / max(1, done) * (total - done)
        print(f"[E24] {lab} {cat} s{seed}: F0 {cell['f0']['auroc']:.5f} -> "
              f"R1 {cell['r1']['auroc']:.5f} (d={cell['d_auroc']:+.5f}) | "
              f"dAP={cell['d_ap']:+.5f} steps={cell['steps_used']} "
              f"| elapsed={hms(el)} ETA={hms(eta)}", flush=True)
    refresh_csv(args.stage)
    print(f"\n[E24] stage {args.stage} finished in {hms(time.time() - t_start)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
