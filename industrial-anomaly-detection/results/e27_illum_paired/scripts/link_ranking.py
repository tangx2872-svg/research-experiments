#!/usr/bin/env python
"""E27-B: 把"跨光照得分变化"与"排序指标变化"并列，检查是否存在
'跨光照变化减小、但排序退化' 的类别（全部列出，不做选择性报告）。"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parents[1]
CELLS = ROOT / "results" / "e22" / "cells"
CATS = ["Bird", "Car", "Cube", "Dice", "Doll", "Holder", "Motor", "Ring", "Teapot", "Tube"]
SEEDS = [0, 1, 2]
DEV = {"Bird", "Car", "Cube"}


def main() -> int:
    rank = {}
    lvl = {}
    for cat in CATS:
        ds = []
        for seed in SEEDS:
            c = json.loads((CELLS / f"{cat}_seed{seed}.json").read_text())
            y = np.array(c["labels"], int)
            det = np.array([x is not False for x in c["detectable_raw"]], bool)
            s0 = np.array(c["scores_f0"], float)[det]
            s1 = np.array(c["scores_r1"], float)[det]
            yy = y[det]
            ds.append((roc_auc_score(yy, s1) - roc_auc_score(yy, s0),
                       average_precision_score(yy, s1) - average_precision_score(yy, s0)))
        rank[cat] = (float(np.mean([d[0] for d in ds])), float(np.mean([d[1] for d in ds])))

    for r in csv.DictReader(open(OUT / "illum_level_stats.csv")):
        key = (r["category"], int(r["seed"]), r["subset"], r["method"])
        lvl[key] = (float(r["mean"]), float(r["sd"]))

    rows = []
    for cat in CATS:
        seedavg = [r for r in csv.DictReader(open(OUT / "illum_sd_category_seedavg.csv"))
                   if r["category"] == cat][0]
        dsd = float(seedavg["mean_delta_sd"])
        dau, dap = rank[cat]
        n_mean = np.mean([lvl[(cat, s, "normal", "R1")][0] - lvl[(cat, s, "normal", "F0")][0]
                          for s in SEEDS])
        n_sd = np.mean([lvl[(cat, s, "normal", "R1")][1] - lvl[(cat, s, "normal", "F0")][1]
                        for s in SEEDS])
        a_mean = np.mean([lvl[(cat, s, "anomaly", "R1")][0] - lvl[(cat, s, "anomaly", "F0")][0]
                          for s in SEEDS])
        a_sd = np.mean([lvl[(cat, s, "anomaly", "R1")][1] - lvl[(cat, s, "anomaly", "F0")][1]
                        for s in SEEDS])
        rows.append({
            "category": cat, "group": "dev" if cat in DEV else "ext",
            "delta_sd_normal_across_illum": dsd,
            "delta_auroc": dau, "delta_ap": dap,
            "delta_normal_mean": float(n_mean), "delta_normal_sd": float(n_sd),
            "delta_anomaly_mean": float(a_mean), "delta_anomaly_sd": float(a_sd),
            "sd_decreased": dsd < 0,
            "auroc_decreased": dau < 0,
            "pattern": ("SD_down_AUROC_up" if dsd < 0 and dau > 0 else
                        "SD_down_AUROC_down" if dsd < 0 else
                        "SD_up_AUROC_down" if dau < 0 else "both_up"),
        })

    with open(OUT / "illum_vs_ranking.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{r[k]:.6f}" if isinstance(r[k], float) else r[k]) for k in rows[0]})

    print(f"[e27b] illum_vs_ranking.csv: {len(rows)} rows")
    print(f"{'cat':8s} {'ΔSD':>9s} {'ΔAUROC':>9s} {'ΔAP':>9s} {'ΔN_mean':>9s} {'ΔN_sd':>9s} "
          f"{'ΔA_mean':>9s} {'ΔA_sd':>9s}  pattern")
    for r in rows:
        print(f"{r['category']:8s} {r['delta_sd_normal_across_illum']:+9.5f} "
              f"{r['delta_auroc']:+9.5f} {r['delta_ap']:+9.5f} "
              f"{r['delta_normal_mean']:+9.5f} {r['delta_normal_sd']:+9.5f} "
              f"{r['delta_anomaly_mean']:+9.5f} {r['delta_anomaly_sd']:+9.5f}  {r['pattern']}")
    m = np.mean([r["delta_sd_normal_across_illum"] for r in rows])
    print(f"\nmacro ΔSD={m:+.6f}  "
          f"n_SD_down={sum(1 for r in rows if r['sd_decreased'])}/10  "
          f"n_AUROC_down={sum(1 for r in rows if r['auroc_decreased'])}/10  "
          f"n_SD_down_AUROC_up={sum(1 for r in rows if r['pattern']=='SD_down_AUROC_up')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
