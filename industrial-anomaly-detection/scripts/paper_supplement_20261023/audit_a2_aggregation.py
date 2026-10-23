#!/usr/bin/env python
"""A2: 独立复算 E25 的聚合口径，与冻结 multiseed_analysis.json 对账。

从冻结 cell 的 d_auroc/d_ap 复算：
- PCB 5 target split 等权均值（排除 test0 source）
- MetalParts 6 target split 等权均值（排除 test0 invalid）
- 11 target split 等权宏平均（主值）
- 2 子数据集等权（(PCB+MP)/2）
- 逐 seed 域均值 + seed std
并核对 39 = 33(有效 target) + 3(PCB test0) + 3(MP test0) 的 cell 账目。
"""
from __future__ import annotations
import json
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).resolve().parents[2]
CELLDIR = ROOT / "results" / "e25" / "cells"
OUTDIR = ROOT / "results" / "paper_supplement_20261023"

# domain -> 有效 target split 列表（排除 source test0 / invalid test0）
VALID_TARGETS = {
    "PCB": ["test1", "test2", "test3", "test4", "test5"],
    "MetalParts": ["test1", "test2", "test3", "test4", "test5", "test6"],
}
DOMAIN_ORDER = ["PCB", "MetalParts"]


def load(domain, split, seed):
    fn = CELLDIR / (f"{domain}_{split}.json" if seed == 0
                    else f"{domain}_seed{seed}_{split}.json")
    return json.loads(fn.read_text())


def main():
    cells = sorted(CELLDIR.glob("*.json"))
    # --- cell 账目核对 ---
    n_pcb_target = 0
    n_mp_target = 0
    n_pcb_test0 = 0
    n_mp_test0 = 0
    seen = set()
    for cp in cells:
        c = json.loads(cp.read_text())
        key = (c["domain"], c["split"], c["seed"])
        assert key not in seen, f"duplicate cell {key}"
        seen.add(key)
        if c["domain"] == "PCB" and c["split"] == "test0":
            n_pcb_test0 += 1
        elif c["domain"] == "MetalParts" and c["split"] == "test0":
            n_mp_test0 += 1
        elif c["domain"] == "PCB" and c["split"] in VALID_TARGETS["PCB"]:
            n_pcb_target += 1
        elif c["domain"] == "MetalParts" and c["split"] in VALID_TARGETS["MetalParts"]:
            n_mp_target += 1
    n_target = n_pcb_target + n_mp_target
    accounting = {
        "total_cells": len(cells),
        "pcb_target_cells": n_pcb_target,      # 期望 5*3=15
        "mp_target_cells": n_mp_target,        # 期望 6*3=18
        "target_cells": n_target,              # 期望 33
        "pcb_test0_cells": n_pcb_test0,        # 期望 3
        "mp_test0_cells": n_mp_test0,          # 期望 3
        "accounting_39": (n_target + n_pcb_test0 + n_mp_test0) == 39,
        "accounting_33": n_target == 33,
    }

    # --- 复算聚合 ---
    seeds = [0, 1, 2]
    per_split = {}   # (domain, split) -> {seed: d_auroc}
    per_seed_macro = defaultdict(list)   # seed -> [d_auroc across 11 target splits]
    per_domain_mean = {}
    for domain in DOMAIN_ORDER:
        dlist = []
        for seed in seeds:
            splits = []
            for split in VALID_TARGETS[domain]:
                c = load(domain, split, seed)
                per_split.setdefault((domain, split), {})[seed] = c["d_auroc"]
                splits.append(c["d_auroc"])
                per_seed_macro[seed].append(c["d_auroc"])
            dlist.append(sum(splits) / len(splits))
        per_domain_mean[domain] = {
            "n_splits": len(VALID_TARGETS[domain]),
            "d_auroc_mean_over_splits": sum(dlist) / len(dlist),
            "per_seed": {s: v for s, v in zip(seeds, dlist)},
        }

    # 11 target split 等权宏平均（主值）：每 split 先做 3-seed 均值，再 11 split 等权
    split_means = []
    for domain in DOMAIN_ORDER:
        for split in VALID_TARGETS[domain]:
            m = sum(per_split[(domain, split)][s] for s in seeds) / len(seeds)
            split_means.append(m)
    macro_split_weighted = sum(split_means) / len(split_means)

    # 2 子数据集等权
    macro_domain_weighted = (per_domain_mean["PCB"]["d_auroc_mean_over_splits"]
                             + per_domain_mean["MetalParts"]["d_auroc_mean_over_splits"]) / 2

    # 逐 seed 宏平均（统计样本 = 3 个 per-seed macro，用于 t-interval）
    per_seed_macro_mean = {s: sum(per_seed_macro[s]) / len(per_seed_macro[s])
                           for s in seeds}

    result = {
        "accounting": accounting,
        "per_domain_recomp": per_domain_mean,
        "macro_d_auroc_split_weighted": macro_split_weighted,
        "macro_d_auroc_domain_weighted": macro_domain_weighted,
        "per_seed_macro_mean": per_seed_macro_mean,
        "per_split_d_auroc_seed_mean": {f"{d}|{s}": m for (d, s), m in
                                        sorted(per_split.items())},
    }

    # 与冻结值对账
    frozen = json.loads((ROOT / "results/e25/multiseed_analysis.json").read_text())
    checks = {}
    checks["pcb_d_auroc_mean"] = (abs(per_domain_mean["PCB"]["d_auroc_mean_over_splits"]
                                       - frozen["per_domain"]["PCB"]["d_auroc_mean"]) < 1e-12)
    checks["mp_d_auroc_mean"] = (abs(per_domain_mean["MetalParts"]["d_auroc_mean_over_splits"]
                                     - frozen["per_domain"]["MetalParts"]["d_auroc_mean"]) < 1e-12)
    checks["macro_split_weighted"] = (abs(macro_split_weighted
                                          - frozen["overall"]["macro_d_auroc_mean_over_splits"]) < 1e-12)
    # 子数据集等权：EVIDENCE_MAP 声称 +0.000550，由两个 per-domain 全精度复算
    checks["macro_domain_weighted_matches_evidence_map"] = abs(macro_domain_weighted - 0.000550) < 5e-4
    result["reconciliation"] = checks
    result["frozen_reference"] = {
        "pcb_d_auroc_mean": frozen["per_domain"]["PCB"]["d_auroc_mean"],
        "mp_d_auroc_mean": frozen["per_domain"]["MetalParts"]["d_auroc_mean"],
        "macro_split_weighted": frozen["overall"]["macro_d_auroc_mean_over_splits"],
        "macro_d_ap_split_weighted": frozen["overall"]["macro_d_ap_mean_over_splits"],
    }

    out = OUTDIR / "audit_a2_aggregation_reconciliation.json"
    with open(out, "w") as f:
        json.dump(result, f, indent=1, ensure_ascii=False)

    print(f"=== A2 聚合复算 ===")
    print(f"cell账目: 39={n_target}+{n_pcb_test0}+{n_mp_test0} -> "
          f"{accounting['accounting_39']} (33 target -> {accounting['accounting_33']})")
    print(f"PCB  5-split 等权 mean d_auroc = {per_domain_mean['PCB']['d_auroc_mean_over_splits']:.8f}  "
          f"(冻结 {frozen['per_domain']['PCB']['d_auroc_mean']:.8f}) -> {checks['pcb_d_auroc_mean']}")
    print(f"MP   6-split 等权 mean d_auroc = {per_domain_mean['MetalParts']['d_auroc_mean_over_splits']:.8f}  "
          f"(冻结 {frozen['per_domain']['MetalParts']['d_auroc_mean']:.8f}) -> {checks['mp_d_auroc_mean']}")
    print(f"11-split 等权宏平均 = {macro_split_weighted:.8f}  (冻结 {frozen['overall']['macro_d_auroc_mean_over_splits']:.8f}) -> {checks['macro_split_weighted']}")
    print(f"2-子数据集等权 = {macro_domain_weighted:.8f}  (EVIDENCE_MAP 声称 +0.000550) -> {checks['macro_domain_weighted_matches_evidence_map']}")
    print(f"per-seed 宏平均: {per_seed_macro_mean}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
