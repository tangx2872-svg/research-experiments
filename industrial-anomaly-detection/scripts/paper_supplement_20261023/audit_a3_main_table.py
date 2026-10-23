#!/usr/bin/env python
"""A3: 从冻结 E22 逐 seed cell 独立复算主表统计量，并对账 paper_main_table.csv。

复算项：
- 每类 3-seed 的 d_auroc / d_ap 均值、ddof=1 样本标准差
- 10 类等权宏平均（先每类 3-seed 均值，再 10 类等权）
- median ΔAUROC
- 正收益类别数 / 正收益 seed 数 / 三 seed 全正类别数
- Doll 的 d_auroc 样本 std（ddof=1，未舍入值），核对 0.004638
- 区分"平均指标"（对每类先平均 3-seed 的 delta）vs "先平均分数再算指标"（不混用）

不修改冻结产物。
"""
from __future__ import annotations
import json
from pathlib import Path
from collections import defaultdict

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
CELLDIR = ROOT / "results" / "e22" / "cells"
OUTDIR = ROOT / "results" / "paper_supplement_20261023"

CATS = ["Bird", "Car", "Cube", "Dice", "Doll", "Holder", "Motor", "Ring", "Teapot", "Tube"]
SEEDS = [0, 1, 2]


def main():
    # 读所有 cell
    delta = defaultdict(dict)   # cat -> {seed: (d_auroc, d_ap)}
    for cat in CATS:
        for seed in SEEDS:
            cp = CELLDIR / f"{cat}_seed{seed}.json"
            c = json.loads(cp.read_text())
            delta[cat][seed] = (c["delta_auroc"], c["delta_ap"])

    # --- 每类复算 ---
    rows = []
    for cat in CATS:
        d_auroc = [delta[cat][s][0] for s in SEEDS]
        d_ap = [delta[cat][s][1] for s in SEEDS]
        rows.append({
            "category": cat,
            "d_auroc_mean": float(np.mean(d_auroc)),
            "d_auroc_std_ddof1": float(np.std(d_auroc, ddof=1)),
            "d_auroc_min": float(np.min(d_auroc)),
            "d_auroc_max": float(np.max(d_auroc)),
            "pos_seeds_auroc": int(sum(1 for x in d_auroc if x > 0)),
            "d_ap_mean": float(np.mean(d_ap)),
            "d_ap_std_ddof1": float(np.std(d_ap, ddof=1)),
            "pos_seeds_ap": int(sum(1 for x in d_ap if x > 0)),
        })

    # --- 宏平均 ---
    macro_auroc = float(np.mean([r["d_auroc_mean"] for r in rows]))
    macro_ap = float(np.mean([r["d_ap_mean"] for r in rows]))
    median_auroc = float(np.median([r["d_auroc_mean"] for r in rows]))
    n_pos_cat_auroc = sum(1 for r in rows if r["d_auroc_mean"] > 0)
    n_pos_cat_ap = sum(1 for r in rows if r["d_ap_mean"] > 0)
    n_all3pos_auroc = sum(1 for r in rows if r["pos_seeds_auroc"] == 3)
    n_all3pos_ap = sum(1 for r in rows if r["pos_seeds_ap"] == 3)

    # --- Doll std 精确核对 ---
    doll = [delta["Doll"][s][0] for s in SEEDS]
    doll_std = float(np.std(doll, ddof=1))
    doll_mean = float(np.mean(doll))

    result = {
        "per_category": rows,
        "macro": {
            "macro_d_auroc_mean": macro_auroc,
            "macro_d_ap_mean": macro_ap,
            "median_d_auroc": median_auroc,
            "n_pos_cat_auroc": n_pos_cat_auroc,
            "n_pos_cat_ap": n_pos_cat_ap,
            "n_all3pos_auroc": n_all3pos_auroc,
            "n_all3pos_ap": n_all3pos_ap,
        },
        "doll": {
            "per_seed_d_auroc": doll,
            "d_auroc_mean": doll_mean,
            "d_auroc_std_ddof1": doll_std,
            "matches_0_004638": abs(doll_std - 0.004638) < 5e-7,
        },
    }

    # --- 与 E22 全精度主表 (main_table_10cat_3seed.csv) 对账 ---
    # 注意：paper_main_table.csv 是 5 位舍入值，不作为全精度对账对象；
    # 权威全精度来源是 results/e22/main_table_10cat_3seed.csv。
    import csv
    mt = {}
    with open(ROOT / "results/e22/main_table_10cat_3seed.csv", newline="") as f:
        for row in csv.DictReader(f):
            mt[row["category"]] = row

    checks = []
    for r in rows:
        cat = r["category"]
        frozen_mean = float(mt[cat]["d_auroc_mean"])
        frozen_std = float(mt[cat]["d_auroc_std"])
        checks.append({
            "category": cat,
            "mean_match": abs(r["d_auroc_mean"] - frozen_mean) < 1e-6,
            "std_match": abs(r["d_auroc_std_ddof1"] - frozen_std) < 1e-6,
            "frozen_mean": frozen_mean, "frozen_std": frozen_std,
        })

    # paper_main_table.csv（5 位舍入）一致性（宽松核对，容差 1e-4，因舍入）
    pmt = {}
    with open(ROOT / "results/paper_ready/paper_main_table.csv", newline="") as f:
        for row in csv.DictReader(f):
            pmt[row["Category"]] = row
    pmt_checks = []
    for r in rows:
        cat = r["category"]
        if cat not in pmt:
            pmt_checks.append({"category": cat, "rounded_match": False,
                               "reason": "missing in paper_main_table.csv"})
            continue
        frozen_mean = float(pmt[cat]["dAUROC_mean"])
        frozen_std = float(pmt[cat]["dAUROC_std"])
        pmt_checks.append({
            "category": cat,
            "rounded_mean_match": abs(r["d_auroc_mean"] - frozen_mean) < 5e-5,
            "rounded_std_match": abs(r["d_auroc_std_ddof1"] - frozen_std) < 5e-5,
        })

    result["reconciliation"] = {
        "per_category_vs_full_precision": checks,
        "all_mean_match": all(c["mean_match"] for c in checks),
        "all_std_match": all(c["std_match"] for c in checks),
        "per_category_vs_rounded_paper_table": pmt_checks,
        "all_rounded_match": all(c.get("rounded_mean_match", False) and
                                 c.get("rounded_std_match", False) for c in pmt_checks),
    }

    out = OUTDIR / "audit_a3_main_table_reconciliation.json"
    with open(out, "w") as f:
        json.dump(result, f, indent=1, ensure_ascii=False)

    print("=== A3 主表复算 ===")
    print(f"macro ΔAUROC = {macro_auroc:.6f}  (声称 +0.00760)")
    print(f"median ΔAUROC = {median_auroc:.6f}  (声称 +0.01080)")
    print(f"macro ΔAP = {macro_ap:.6f}  (声称 +0.00777)")
    print(f"正收益类别(ΔAUROC) = {n_pos_cat_auroc}/10  (声称 9/10)")
    print(f"三 seed 全正(ΔAUROC) = {n_all3pos_auroc}/10  (声称 7/10)")
    print(f"Doll: mean={doll_mean:.6f} std(ddof=1)={doll_std:.6f} 匹配0.004638={abs(doll_std-0.004638)<5e-7}")
    print(f"对账: 均值全匹配={result['reconciliation']['all_mean_match']}  标准差全匹配={result['reconciliation']['all_std_match']}")
    print()
    for r in rows:
        print(f"  {r['category']:8s} mean={r['d_auroc_mean']:+.6f} std={r['d_auroc_std_ddof1']:.6f} pos={r['pos_seeds_auroc']}/3")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
