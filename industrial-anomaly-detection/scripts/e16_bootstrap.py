#!/usr/bin/env python
"""E16 paired bootstrap of dAUROC / dAP on the OFFICIAL detectable-filtered test set."""
from __future__ import annotations
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np  # noqa: E402
from sklearn.metrics import roc_auc_score, average_precision_score  # noqa: E402

import e4d1_m2ad_loader as LD  # noqa: E402
from e11_p2_runner import raw_test_map  # noqa: E402

CATS = ["Bird", "Car", "Cube"]
SEEDS = [0, 1, 2]


def detectable_mask(cat):
    recs = LD.load_records(category=cat, require_exists=False)
    test = [r for r in LD.select(recs, views=["120"]) if r["split"] == "test"]
    rawmap = raw_test_map(cat)
    rel = lambda r: str(Path(r["image_path"]).relative_to(LD.DATA_ROOT))
    return np.array([rawmap[rel(r)]["detectable"] != False for r in test])


def main() -> int:
    cells = json.loads((ROOT / "results" / "e16" / "e16_partial.json").read_text())["cells"]
    rng = np.random.default_rng(0)
    out = {}
    for cat in CATS:
        m = detectable_mask(cat)
        for s in SEEDS:
            c = cells[f"{cat}|{s}"]
            y = np.array(c["labels"])[m]
            s0 = np.array(c["test_scores_f0"])[m]
            s1 = np.array(c["test_scores_r1"])[m]
            n = len(y)
            da, dp = [], []
            for _ in range(3000):
                i = rng.integers(0, n, n)
                if len(set(y[i].tolist())) < 2:
                    continue
                da.append(roc_auc_score(y[i], s1[i]) - roc_auc_score(y[i], s0[i]))
                dp.append(average_precision_score(y[i], s1[i]) - average_precision_score(y[i], s0[i]))
            da, dp = np.array(da), np.array(dp)
            obs_a = roc_auc_score(y, s1) - roc_auc_score(y, s0)
            obs_p = average_precision_score(y, s1) - average_precision_score(y, s0)
            out[f"{cat}|{s}"] = {
                "n_detectable": int(n), "n_ok": int((y == 0).sum()), "n_ng": int((y == 1).sum()),
                "d_auroc": float(obs_a),
                "d_auroc_ci": [float(np.percentile(da, 2.5)), float(np.percentile(da, 97.5))],
                "p_auroc_gt0": float((da > 0).mean()),
                "d_ap": float(obs_p),
                "d_ap_ci": [float(np.percentile(dp, 2.5)), float(np.percentile(dp, 97.5))],
                "p_ap_gt0": float((dp > 0).mean())}
    bc = [out[k] for k in out if k.startswith("Bird") or k.startswith("Car")]
    summary = {
        "bird_car_mean_d_auroc": float(np.mean([v["d_auroc"] for v in bc])),
        "bird_car_min_d_auroc": float(np.min([v["d_auroc"] for v in bc])),
        "bird_car_positive": int(sum(1 for v in bc if v["d_auroc"] > 0)),
        "bird_car_mean_d_ap": float(np.mean([v["d_ap"] for v in bc])),
        "cube_mean_d_auroc": float(np.mean([out[f"Cube|{s}"]["d_auroc"] for s in SEEDS])),
        "cells_with_ci_excluding_zero": [k for k, v in out.items()
                                         if v["d_auroc_ci"][0] > 0 or v["d_auroc_ci"][1] < 0],
    }
    (ROOT / "results" / "e16" / "bootstrap.json").write_text(
        json.dumps({"cells": out, "summary": summary}, indent=2, default=float))
    print(json.dumps(summary, indent=2, default=float))
    for k, v in out.items():
        print(f"{k:9s} n={v['n_detectable']} dAUROC={v['d_auroc']:+.5f} "
              f"CI[{v['d_auroc_ci'][0]:+.5f},{v['d_auroc_ci'][1]:+.5f}] P>0={v['p_auroc_gt0']:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
