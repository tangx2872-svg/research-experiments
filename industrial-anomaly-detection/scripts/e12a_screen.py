#!/usr/bin/env python
"""E12-A - Subspace Scoring Screening (frozen DINOv2-B, PCA/subspace residual).

Baseline S0 = frozen DINOv2-B + 1-NN-max (REUSED from E11-P5, not recomputed).
Candidates = PCA residual with tau in {0.95, 0.99, 0.995} (S1/S2/S3), each with
  AGG-1 MAX (image score = max patch residual) and
  AGG-2 TOP-REGION (SubspaceAD TVaR: mean of top-1% patch residuals).

Feature source = DINOv2-B final-layer patch tokens (get_intermediate_layers n=1
norm=True), 448 input, 1024 patches/image, 768-d. PCA is fit on RAW (LayerNorm'd)
mean-centred features (NO L2 normalisation), per SubspaceAD official definition.

Checkpoint: results/e12a/e12a_partial.json (per-cell metrics + per-image scores),
resume skips completed cells. Baseline reproduction is a hard gate per category.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torchvision.transforms as T  # noqa: E402
from PIL import Image  # noqa: E402

import e4d1_m2ad_loader as LD  # noqa: E402
import e8_common as E8  # noqa: E402
from e11_p2_runner import (load_dinov2, raw_test_map, official_metrics, route,  # noqa: E402
                           nn_image_scores, BACKBONES)

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
N_PATCH = 1024
TAUS = [0.95, 0.99, 0.995]          # S1, S2, S3
AGG_MODES = {"max": "AGG-1", "subspace": "AGG-2"}
TOP_K = max(1, int(np.ceil(0.01 * N_PATCH)))   # = 11

OUTDIR = ROOT / "results" / "e12a"
PARTIAL = OUTDIR / "e12a_partial.json"
CSV_PATH = OUTDIR / "e12a_screening.csv"

# S0 baseline reused from E11-P5 (frozen, exact from CSV).
BASELINE_CSV = ROOT / "results" / "e11" / "p5_screening.csv"


def _prep(img: Image.Image, model, dev):
    tf = T.Compose([T.Resize(448, interpolation=T.InterpolationMode.BICUBIC, antialias=True),
                    T.ToTensor(), T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)])
    t = tf(img)
    ps = model.patch_size
    return t[:, :t.shape[1] - t.shape[1] % ps, :t.shape[2] - t.shape[2] % ps].to(dev)


def extract_patches(model, path: str, dev) -> np.ndarray:
    """-> (1024, 768) float32 patch tokens (norm=True, final layer, no CLS)."""
    img = Image.open(path).convert("RGB")
    with torch.inference_mode():
        t = _prep(img, model, dev)
        r = model.get_intermediate_layers(t.unsqueeze(0), n=1, norm=True)[0][0]
    return r.float().cpu().numpy()


def load_p5_baseline() -> dict:
    """Read S0 (baseline) exact auroc/ap per category from E11-P5 CSV."""
    out = {}
    with open(BASELINE_CSV) as f:
        for row in csv.DictReader(f):
            if row["module"] == "baseline":
                out[row["category"]] = {"auroc": float(row["auroc"]), "ap": float(row["ap"])}
    assert set(out) == {"Bird", "Car", "Cube"}, f"baseline CSV missing categories: {set(out)}"
    return out


# ------------------------------------------------------------------ PCA
def fit_pca(bank_patches: np.ndarray, tau: float):
    """Deterministic covariance PCA. Returns (mu float32 (768,), C float32 (768,r), r, explained)."""
    mu = bank_patches.mean(axis=0, dtype=np.float64)
    X = bank_patches.astype(np.float64)
    X -= mu
    cov = X.T @ X / X.shape[0]
    eigvals, eigvecs = np.linalg.eigh(cov)            # ascending
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]
    total = eigvals.sum()
    cum = np.cumsum(eigvals)
    r = int(np.searchsorted(cum, tau * total, side="left") + 1)
    r = max(1, min(r, X.shape[1]))
    return mu.astype(np.float32), eigvecs[:, :r].astype(np.float32), r


def residual_scores(test_patches: np.ndarray, mu, C) -> np.ndarray:
    """S(x) = ||x - (mu + C C^T (x-mu))||_2^2, chunked. -> (M,) float64."""
    M = test_patches.shape[0]
    out = np.empty(M, dtype=np.float64)
    CH = 204800  # 200 images
    for i in range(0, M, CH):
        Xc = test_patches[i:i + CH].astype(np.float64) - mu
        proj = Xc @ C
        recon = proj @ C.T
        resid = Xc - recon
        out[i:i + CH] = (resid * resid).sum(axis=1)
    return out


# ------------------------------------------------------------------ aggregation
def image_scores(patch_scores: np.ndarray, mode: str) -> np.ndarray:
    """patch_scores (n_img*1024,) -> per-image score (n_img,)."""
    n_img = patch_scores.shape[0] // N_PATCH
    M = patch_scores.reshape(n_img, N_PATCH)
    if mode == "max":
        return M.max(axis=1)
    k = TOP_K
    top = np.partition(M, -k, axis=1)[:, -k:]
    return top.mean(axis=1)


# ------------------------------------------------------------------ run
def run_category(category: str, dev, smoke: bool) -> dict:
    recs = LD.load_records(category=category, require_exists=False)
    est = LD.estimate_scope(recs)
    bank_specs = set(est["bank_specimens"])
    bank = E8.bank_for(sorted(LD.select(recs, bank_specs), key=lambda r: r["image_path"]), 0)
    test = [r for r in LD.select(recs, views=["120"]) if r["split"] == "test"]

    if smoke:
        good = [r for r in test if r["kind"] == "Good"][:12]
        ng = [r for r in test if r["kind"] == "NG"][:12]
        test = good + ng
        bank = bank[:20]

    bank_r = route(bank, "l0_448")
    test_r = route(test, "l0_448")
    model = load_dinov2(BACKBONES["dinov2b"]["hub"], dev)

    t0 = time.time()
    bank_p = np.concatenate([extract_patches(model, r["image_path"], dev) for r in bank_r], axis=0)
    test_p = np.concatenate([extract_patches(model, r["image_path"], dev) for r in test_r], axis=0)
    extract_s = time.time() - t0

    rel = lambda r: str(Path(r["image_path"]).relative_to(LD.DATA_ROOT))
    rawmap = raw_test_map(category)

    # --- labels (identical across configs)
    labels = []
    for r in test:
        m = rawmap[rel(r)]
        labels.append({"specimen": r["specimen"], "kind": r["kind"],
                       "illumination": r["illumination"], "view": "120",
                       "image_anomaly": int(m["image_anomaly"]), "detectable": m["detectable"]})

    def metrics_from(scores):
        rows = [dict(l, score=float(s)) for l, s in zip(labels, scores)]
        return official_metrics(rows)

    # --- baseline reproduction sanity (internal gate, not a re-run of S0)
    bank_n = bank_p / np.maximum(np.linalg.norm(bank_p, axis=1, keepdims=True), 1e-12)
    test_n = test_p / np.maximum(np.linalg.norm(test_p, axis=1, keepdims=True), 1e-12)
    t0 = time.time()
    base_scores = nn_image_scores(bank_n.astype("float32"), test_n.astype("float32"), dev)
    base_s = time.time() - t0
    base_met = metrics_from(base_scores)

    out = {"category": category, "smoke": smoke, "extract_s": round(extract_s, 1),
           "n_bank": len(bank), "n_test": len(test),
           "baseline_reprod": {"auroc": base_met["auroc"], "ap": base_met["ap"],
                               "n_ok": base_met["n_ok"], "n_ng": base_met["n_ng"],
                               "score_s": round(base_s, 1)},
           "configs": {}}

    # --- PCA candidates
    for tau in TAUS:
        key_s = {0.95: "S1", 0.99: "S2", 0.995: "S3"}[tau]
        t0 = time.time()
        mu, C, r = fit_pca(bank_p, tau)
        patch_scores = residual_scores(test_p, mu, C)
        for mode, agg_name in AGG_MODES.items():
            img = image_scores(patch_scores, mode)
            met = metrics_from(img)
            out["configs"][f"{key_s}-{agg_name}"] = {
                "tau": tau, "r": r, "aggregation": mode,
                "metrics": met, "score_s": round(time.time() - t0, 1)}
        print(f"    {key_s} r={r} max={out['configs'][key_s+'-AGG-1']['metrics']['auroc']:.5f} "
              f"subspace={out['configs'][key_s+'-AGG-2']['metrics']['auroc']:.5f}", flush=True)

    del bank_p, test_p, bank_n, test_n
    torch.cuda.empty_cache()
    return out


def cell_rows(cat, base, out):
    rows = []
    s0 = base[cat]
    rows.append([cat, "120", "S0", f"{s0['auroc']:.5f}", f"{s0['ap']:.5f}",
                 "+0.00000", "+0.00000", "", "reused_p5"])
    for key, cfg in out["configs"].items():
        m = cfg["metrics"]
        rows.append([cat, "120", key, f"{m['auroc']:.5f}", f"{m['ap']:.5f}",
                     f"{m['auroc']-s0['auroc']:+.5f}", f"{m['ap']-s0['ap']:+.5f}",
                     cfg["r"], "OK"])
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--categories", default="Bird,Car,Cube")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    dev = torch.device("cuda")
    cats = [c.strip() for c in args.categories.split(",") if c.strip()]
    OUTDIR.mkdir(parents=True, exist_ok=True)

    base = load_p5_baseline()
    print(f"[E12-A] categories={cats} smoke={args.smoke} | disk {E8.free_gb():.1f} GB | "
          f"GPU {E8.gpu_util()}", flush=True)
    print(f"[E12-A] S0 reuse (P5): Bird {base['Bird']['auroc']:.5f} / "
          f"Car {base['Car']['auroc']:.5f} / Cube {base['Cube']['auroc']:.5f}", flush=True)

    # resume
    partial = {"experiment": "E12-A", "cells": {}, "baseline": base, "sanity": {}}
    if PARTIAL.exists():
        try:
            prev = json.loads(PARTIAL.read_text())
            partial.update(prev)
        except json.JSONDecodeError:
            pass

    t_start = time.time()
    for ci, cat in enumerate(cats):
        done_keys = {k for k in partial.get("cells", {}) if k.startswith(f"{cat}|")}
        need = [f"{cat}|S1-AGG-1", f"{cat}|S1-AGG-2", f"{cat}|S2-AGG-1",
                f"{cat}|S2-AGG-2", f"{cat}|S3-AGG-1", f"{cat}|S3-AGG-2"]
        if not args.smoke and all(k in done_keys for k in need):
            print(f"[E12-A] SKIP {cat} (all cells done)", flush=True)
            continue

        print(f"\n[E12-A] RUN {cat}", flush=True)
        try:
            out = run_category(cat, dev, args.smoke)
        except Exception as exc:
            import traceback
            traceback.print_exc()
            print(f"[E12-A] {cat} FAILED: {type(exc).__name__}: {exc}", flush=True)
            continue

        # baseline reproduction gate (skip on smoke)
        if not args.smoke:
            b = out["baseline_reprod"]
            s0 = base[cat]
            d_au = b["auroc"] - s0["auroc"]
            print(f"[E12-A] {cat} baseline reprod: auroc={b['auroc']:.5f} (P5 {s0['auroc']:.5f}, "
                  f"d={d_au:+.6f}) ap={b['ap']:.5f} (P5 {s0['ap']:.5f})", flush=True)
            if abs(d_au) > 1e-4:
                print(f"[E12-A] FATAL: {cat} baseline did NOT reproduce (d={d_au:+.6f}); "
                      f"feature source mismatch - STOPPING", flush=True)
                return 2

        if not args.smoke:
            partial["sanity"][cat] = out["baseline_reprod"]
            for key, cfg in out["configs"].items():
                partial.setdefault("cells", {})[f"{cat}|{key}"] = {
                    "auroc": cfg["metrics"]["auroc"], "ap": cfg["metrics"]["ap"],
                    "tau": cfg["tau"], "r": cfg["r"], "aggregation": cfg["aggregation"],
                    "score_s": cfg["score_s"]}
            # write atomically
            tmp = PARTIAL.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(partial, indent=2, default=float))
            tmp.replace(PARTIAL)

        el = time.time() - t_start
        done = sum(1 for k in partial.get("cells", {}) if k.endswith(("-AGG-1", "-AGG-2")))
        total = len(cats) * 6
        per = el / max(1, ci + 1)
        eta = per * (len(cats) - (ci + 1))
        print(f"[E12-A] {ci+1}/{len(cats)} cats | cells done={done}/{total} | "
              f"elapsed={el/60:.1f}m | ETA={eta/60:.1f}m | disk {E8.free_gb():.1f} GB", flush=True)

    if args.smoke:
        print(f"\n[E12-A] SMOKE OK (no checkpoint/CSV written)", flush=True)
        return 0

    # ---- write screening CSV
    w = csv.writer(open(CSV_PATH, "w", newline=""))
    w.writerow(["category", "view", "config", "auroc", "ap", "d_auroc", "d_ap", "r", "status"])
    for cat in cats:
        s0 = base[cat]
        w.writerow([cat, "120", "S0", f"{s0['auroc']:.5f}", f"{s0['ap']:.5f}",
                    "+0.00000", "+0.00000", "", "reused_p5"])
        for key in ["S1-AGG-1", "S1-AGG-2", "S2-AGG-1", "S2-AGG-2", "S3-AGG-1", "S3-AGG-2"]:
            c = partial.get("cells", {}).get(f"{cat}|{key}")
            if not c:
                w.writerow([cat, "120", key, "", "", "", "", "", "MISSING"])
                continue
            w.writerow([cat, "120", key, f"{c['auroc']:.5f}", f"{c['ap']:.5f}",
                        f"{c['auroc']-s0['auroc']:+.5f}", f"{c['ap']-s0['ap']:+.5f}", c["r"], "OK"])
    print(f"\n[E12-A] done -> {CSV_PATH}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
