#!/usr/bin/env python
"""E15 - EfficientAD teacher-student feasibility + mini-screen.

Runs the OFFICIAL EfficientAD (third_party/EfficientAD/efficientad.py) as a subprocess
on an M2AD symlink farm (train/good + test/{good,anomaly}), then parses the saved anomaly
maps to get image-level scores, and evaluates AUROC/AP with the official
image_anomaly + detectable protocol (matching E11-E14).

Frozen protocol: results/e15/e15_protocol.json
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np  # noqa: E402
import tifffile  # noqa: E402

import e4d1_m2ad_loader as LD  # noqa: E402
import e8_common as E8  # noqa: E402
from e11_p2_runner import raw_test_map, official_metrics, route  # noqa: E402

OUTDIR = ROOT / "results" / "e15"
PARTIAL = OUTDIR / "e15_partial.json"
CSV = OUTDIR / "e15_metrics.csv"
FARM_ROOT = ROOT / "data" / "m2ad" / "e15_farm"
EAD_DIR = ROOT / "third_party" / "EfficientAD"

F0 = {"Bird": [0.74725, 0.78863], "Car": [0.78552, 0.84207], "Cube": [0.85410, 0.91686]}


def load_scope(category):
    recs = LD.load_records(category=category, require_exists=False)
    est = LD.estimate_scope(recs)
    bank_specs = set(est["bank_specimens"])
    bank = E8.bank_for(sorted(LD.select(recs, bank_specs), key=lambda r: r["image_path"]), 0)
    test = [r for r in LD.select(recs, views=["120"]) if r["split"] == "test"]
    return bank, test


def build_farm(category):
    """M2AD -> MVTec-style symlink farm. Returns filename -> (is_anomaly)."""
    bank, test = load_scope(category)
    rawmap = raw_test_map(category)
    rel = lambda r: str(Path(r["image_path"]).relative_to(LD.DATA_ROOT))

    train_dir = FARM_ROOT / category / "train" / "good"
    good_dir = FARM_ROOT / category / "test" / "good"
    anom_dir = FARM_ROOT / category / "test" / "anomaly"
    for d in (train_dir, good_dir, anom_dir):
        d.mkdir(parents=True, exist_ok=True)

    # bank -> train/good (use l0_256 cache); idempotent (skip if link exists)
    for i, r in enumerate(route(bank, "l0_256")):
        src = Path(r["image_path"])
        dst = train_dir / f"{i:06d}_{src.name}"
        if not dst.exists():
            dst.symlink_to(src.resolve())

    # test -> good/anomaly (filtered by detectable, official image_anomaly)
    filename_to_anomaly = {}
    gi = ai = 0
    for r in test:
        m = rawmap[rel(r)]
        if m["detectable"] == False:
            continue
        src = Path(route([r], "l0_256")[0]["image_path"])
        if m["image_anomaly"] == 1:
            dst = anom_dir / f"{ai:06d}_{src.name}"
            ai += 1
            filename_to_anomaly[dst.name] = 1
        else:
            dst = good_dir / f"{gi:06d}_{src.name}"
            gi += 1
            filename_to_anomaly[dst.name] = 0
        if not dst.exists():
            dst.symlink_to(src.resolve())
    return filename_to_anomaly


def run_efficientad(category, train_steps, output_dir):
    cmd = [sys.executable, "efficientad.py", "-d", "mvtec_ad", "-s", category,
           "-a", str(FARM_ROOT), "-m", "small", "-i", "none",
           "-o", str(output_dir), "-t", str(train_steps)]
    env = dict(os.environ, HF_ENDPOINT="https://hf-mirror.com")
    t0 = time.time()
    proc = subprocess.run(cmd, cwd=str(EAD_DIR), env=env, capture_output=True, text=True)
    return proc, time.time() - t0


def parse_scores(category, output_dir, filename_to_anomaly):
    """Parse tiff anomaly maps -> (scores, labels), matching official image_anomaly."""
    base = Path(output_dir) / "anomaly_maps" / "mvtec_ad" / category / "test"
    scores, labels = [], []
    for cls in ("good", "anomaly"):
        d = base / cls
        if not d.exists():
            continue
        for tiff in sorted(d.glob("*.tiff")):
            scores.append(float(np.max(tifffile.imread(tiff))))
            labels.append(0 if cls == "good" else 1)
    return scores, labels


def eval_scores(scores, labels):
    from sklearn.metrics import roc_auc_score, average_precision_score
    y = np.array(labels, dtype=int)
    s = np.array(scores, dtype=float)
    if len(set(y.tolist())) < 2:
        return {"auroc": float("nan"), "ap": float("nan"), "n_ok": int((y == 0).sum()),
                "n_ng": int((y == 1).sum())}
    return {"auroc": float(roc_auc_score(y, s)), "ap": float(average_precision_score(y, s)),
            "n_ok": int((y == 0).sum()), "n_ng": int((y == 1).sum())}


def run_category(category, train_steps):
    outdir = OUTDIR / "efficientad_output" / f"{category}_s{train_steps}"
    outdir.mkdir(parents=True, exist_ok=True)
    filename_to_anomaly = build_farm(category)
    proc, wall = run_efficientad(category, train_steps, outdir)
    if proc.returncode != 0:
        return {"status": "FAILED", "stderr": proc.stderr[-1500:]}
    scores, labels = parse_scores(category, outdir, filename_to_anomaly)
    met = eval_scores(scores, labels)
    met["wall_s"] = round(wall, 1)
    met["n_images"] = len(scores)
    return {"status": "OK", "metrics": met, "stderr_tail": proc.stderr[-500:]}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--categories", default="Bird,Car,Cube")
    ap.add_argument("--train-steps", type=int, default=70000)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    cats = [c.strip() for c in args.categories.split(",") if c.strip()]
    OUTDIR.mkdir(parents=True, exist_ok=True)
    steps = 1000 if args.smoke else args.train_steps

    print(f"[E15] cats={cats} smoke={args.smoke} train_steps={steps} | disk {E8.free_gb():.1f} GB",
          flush=True)
    print(f"[E15] F0 (DINOv2-B 1-NN-max): " + " ".join(f"{c}={F0[c][0]:.5f}" for c in cats),
          flush=True)

    partial = {"experiment": "E15", "cells": {}}
    if PARTIAL.exists():
        try:
            partial.update(json.loads(PARTIAL.read_text()))
        except json.JSONDecodeError:
            pass

    t_start = time.time()
    last = None
    done_cnt = 0
    for cat in cats:
        key = cat
        if not args.smoke and key in partial.get("cells", {}):
            print(f"[E15] SKIP {cat} (done)", flush=True)
            done_cnt += 1
            continue
        print(f"\n[E15] RUN {cat}", flush=True)
        out = run_category(cat, steps)
        last = out
        if out["status"] == "FAILED":
            print(f"[E15] {cat} FAILED: {out['stderr'][-400:]}", flush=True)
            if args.smoke:
                return 1
            continue
        m = out["metrics"]
        d_au = m["auroc"] - F0[cat][0]
        d_ap = m["ap"] - F0[cat][1]
        if not args.smoke:
            partial["cells"][cat] = {"auroc": m["auroc"], "ap": m["ap"],
                                     "wall_s": m["wall_s"], "n_images": m["n_images"],
                                     "n_ok": m["n_ok"], "n_ng": m["n_ng"]}
            tmp = PARTIAL.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(partial, indent=2, default=float))
            tmp.replace(PARTIAL)
        done_cnt += 1
        el = time.time() - t_start
        per = el / max(1, done_cnt)
        eta = per * (len(cats) - done_cnt)
        print(f"[E15] PROGRESS {done_cnt}/{len(cats)} | elapsed={el/60:.1f}m | ETA={eta/60:.1f}m | "
              f"{cat} AUROC={m['auroc']:.5f} dAUROC={d_au:+.5f} AP={m['ap']:.5f} "
              f"dAP={d_ap:+.5f} wall={m['wall_s']:.0f}s n={m['n_images']}", flush=True)

    if args.smoke:
        ok = last is not None and last["status"] == "OK" and \
             np.isfinite(last["metrics"]["auroc"])
        print(f"\n[E15] SMOKE {'PASS' if ok else 'FAIL'}", flush=True)
        return 0 if ok else 1

    write_csv(partial)
    print(f"\n[E15] done -> {CSV}", flush=True)
    return 0


def write_csv(partial):
    w = csv.writer(open(CSV, "w", newline=""))
    w.writerow(["category", "view", "family", "auroc", "ap", "d_auroc", "d_ap", "status"])
    for cat in ("Bird", "Car", "Cube"):
        w.writerow([cat, "120", "F0", f"{F0[cat][0]:.5f}", f"{F0[cat][1]:.5f}",
                    "+0.00000", "+0.00000", "reused_e13"])
        cell = partial.get("cells", {}).get(cat)
        if not cell:
            w.writerow([cat, "120", "EfficientAD", "", "", "", "", "MISSING"])
            continue
        w.writerow([cat, "120", "EfficientAD", f"{cell['auroc']:.5f}", f"{cell['ap']:.5f}",
                    f"{cell['auroc']-F0[cat][0]:+.5f}", f"{cell['ap']-F0[cat][1]:+.5f}", "OK"])


if __name__ == "__main__":
    raise SystemExit(main())
