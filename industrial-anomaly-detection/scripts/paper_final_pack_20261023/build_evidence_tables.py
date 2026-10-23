#!/usr/bin/env python
"""投稿前收尾：从冻结产物汇总 ①消融 ②光照证据 两类表（只读，不重算分数）。

输出（全部写到 results/paper_final_pack_20261023/）：
  ablation_summary.csv         F0 / C0(β=0.1) / C1(β=0) / R2(preserve-only) 逐类对照
  illumination_evidence.csv    合成光度扰动 vs 真实光照，分组列清
  illumination_definitions.md  每个量的定义、单位、覆盖范围、来源文件

所有数值来自冻结文件；本脚本不产生新指标，只做选取、平均与格式化。
"""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "paper_final_pack_20261023"
E17 = ROOT / "results" / "e17"
E21 = ROOT / "results" / "e21"
E22 = ROOT / "results" / "e22"
E23 = ROOT / "results" / "e23"
E13A = ROOT / "results" / "e13a"

CATS = ["Bird", "Car", "Cube", "Dice", "Doll", "Holder", "Motor", "Ring", "Teapot", "Tube"]
DEV = ["Bird", "Car", "Cube"]


def rd(p: Path):
    with open(p) as f:
        return list(csv.DictReader(f))


def f(x, default=float("nan")):
    try:
        return float(x)
    except (TypeError, ValueError):
        return default


# --------------------------------------------------------------------------- 消融
def ablation() -> None:
    rows = rd(E17 / "summary.csv")
    by = {(r["category"], r["config"]): r for r in rows}
    ref = json.load(open(E13A / "e13a_partial.json"))
    r2 = {k.split("|")[0]: v for k, v in ref["cells"].items() if k.endswith("|R2")}
    out = []
    for c in DEV:
        f0 = ref["ref"][c]
        c0, c1 = by[(c, "C0")], by[(c, "C1")]
        out.append({
            "category": c,
            "F0_auroc": f(f0["auroc"]), "F0_ap": f(f0["ap"]),
            "C0_fullR1_auroc": f(c0["AUROC"]), "C0_fullR1_ap": f(c0["AP"]),
            "C0_minus_F0_auroc": f(c0["dAUROC_vs_F0"]), "C0_minus_F0_ap": f(c0["dAP_vs_F0"]),
            "C1_beta0_auroc": f(c1["AUROC"]), "C1_beta0_ap": f(c1["AP"]),
            "C1_minus_C0_auroc": f(c1["dAUROC_vs_C0"]), "C1_minus_C0_ap": f(c1["dAP_vs_C0"]),
            "C1_collapsed": c1["collapsed"],
            "R2_preserve_only_auroc": r2[c]["auroc"], "R2_preserve_only_ap": r2[c]["ap"],
            "R2_minus_F0_auroc": r2[c]["auroc"] - f(f0["auroc"]),
            "R2_minus_F0_ap": r2[c]["ap"] - f(f0["ap"]),
        })
    with open(OUT / "ablation_summary.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(out)
    print(f"[ablation] {len(out)} rows -> {OUT/'ablation_summary.csv'}")


# --------------------------------------------------------------------------- 光照
def cells(pattern: str):
    for p in sorted((E22 / "cells").glob(pattern)):
        yield json.loads(p.read_text())


def illumination() -> None:
    rows = []

    # A. 合成光度扰动（训练同款 6 变体：brightness/contrast/gamma x{0.85,1.15}）
    agg = defaultdict(lambda: defaultdict(list))
    for c in cells("*.json"):
        for meth, key in (("F0", "f0"), ("R1", "r1")):
            for m in ("robustness_feat", "robustness_score"):
                agg[(c["category"], meth, m)]["v"].append(c[key][m])
            for m in ("D_illum_shift",):
                agg[(c["category"], meth, m)]["v"].append(c["diagnostics"][f"{m}_{meth}"])
            agg[(c["category"], meth, "B_illum_group_centroid_dist")]["v"].append(
                c["diagnostics"][f"B_illum_group_centroid_dist_{meth}"])
            agg[(c["category"], meth, "A_disp_pairwise_cos")]["v"].append(
                c["diagnostics"][f"A_disp_pairwise_cos_{meth}"])
    for (cat, meth, m), d in sorted(agg.items()):
        arr = np.array(d["v"])
        rows.append({"group": "synthetic_photometric", "metric": m, "dataset": "M2AD view120",
                     "category": cat, "method": meth, "n": arr.size,
                     "value_mean": float(arr.mean()), "value_std": float(arr.std(ddof=1)),
                     "source": "results/e22/cells/*.json",
                     "definition_key": f"{m}_{meth}"})

    # B. 真实光照（M²AD 的 10 个 illumination 分组）
    for r in rd(E22 / "illumination_summary.csv"):
        for meth in ("F0", "R1"):
            rows.append({"group": "real_illumination_M2AD", "metric": "good_score_spread_across_illum_groups",
                         "dataset": "M2AD view120", "category": r["category"], "method": meth,
                         "n": 3, "value_mean": f(r[f"spread_{meth}_mean"]),
                         "value_std": f(r[f"spread_{meth}_std"]),
                         "source": "results/e22/illumination_summary.csv",
                         "definition_key": "max_minus_min of good_mean over 10 illumination groups"})
        rows.append({"group": "real_illumination_M2AD", "metric": "spread_ratio_R1_over_F0",
                     "dataset": "M2AD view120", "category": r["category"], "method": "R1/F0",
                     "n": 3, "value_mean": f(r["spread_ratio_R1_over_F0"]), "value_std": float("nan"),
                     "source": "results/e22/illumination_summary.csv",
                     "definition_key": "ratio of the two spreads above"})

    # C. 外部真实域偏移（RobustAD 条件组，含光照相关组）
    e25 = defaultdict(list)
    for p in sorted((ROOT / "results" / "e25" / "cells").glob("*.json")):
        c = json.loads(p.read_text())
        if not c["valid"]:
            continue
        e25[(c["domain"], c["condition_group"])].append(c["d_auroc"])
    for (dom, grp), vals in sorted(e25.items()):
        arr = np.array(vals)
        rows.append({"group": "external_real_shift_RobustAD", "metric": "d_auroc_by_condition_group",
                     "dataset": f"RobustAD {dom}", "category": dom, "method": f"{grp}",
                     "n": arr.size, "value_mean": float(arr.mean()),
                     "value_std": float(arr.std(ddof=1)) if arr.size > 1 else float("nan"),
                     "source": "results/e25/cells/*.json",
                     "definition_key": "mean dAUROC over seeds, valid cells only"})

    # D. 多视角（不是光照，必须单独标注）
    for r in rd(E21 / "analysis_cells.csv"):
        if r["seed"] != "0":
            continue
        rows.append({"group": "viewpoint_M2AD", "metric": "d_auroc_by_view",
                     "dataset": "M2AD", "category": r["category"], "method": f"view{r['view']}",
                     "n": 1, "value_mean": f(r["d_auroc"]), "value_std": float("nan"),
                     "source": "results/e21/analysis_cells.csv",
                     "definition_key": "dAUROC for a viewpoint change, NOT an illumination change"})

    with open(OUT / "illumination_evidence.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["group", "metric", "dataset", "category", "method",
                                           "n", "value_mean", "value_std", "source",
                                           "definition_key"], lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    print(f"[illumination] {len(rows)} rows -> {OUT/'illumination_evidence.csv'}")

    # 供 REPORT 引用的紧凑摘要
    summ = defaultdict(dict)
    for r in rows:
        if r["group"] == "synthetic_photometric" and r["metric"] in ("robustness_feat", "robustness_score"):
            summ[(r["category"], r["metric"])][r["method"]] = r["value_mean"]
    with open(OUT / "illumination_synthetic_summary.csv", "w", newline="") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(["category", "metric", "F0_mean_over_3seeds", "R1_mean_over_3seeds", "delta"])
        for (cat, m), d in sorted(summ.items()):
            w.writerow([cat, m, f"{d['F0']:.6f}", f"{d['R1']:.6f}", f"{d['R1']-d['F0']:+.6f}"])
    print(f"[illumination] synthetic summary -> {OUT/'illumination_synthetic_summary.csv'}")


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    ablation()
    illumination()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
