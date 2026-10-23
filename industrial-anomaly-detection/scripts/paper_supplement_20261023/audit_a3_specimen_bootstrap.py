#!/usr/bin/env python
"""A3 补充：工件级（specimen-level）配对 cluster bootstrap。

估计对象：以 specimen 为 cluster 重采样时，每类的 ΔAUROC = AUROC(R1) - AUROC(F0) 的
bootstrap 分布。同一 specimen 的全部 illumination/view 条件作为一个整体被重采样；
F0 与 R1 使用完全相同的重采样索引（paired）。

与既有 image-level bootstrap 的区别（必须标注）：
- 既有 (results/e22/bootstrap_e22.json)：以单张测试图（image）为 resample 单元。
- 本补充：以 specimen（工件）为 resample 单元，specimen 内所有图一起进入。
- 两者估计的对象不同：一个是 image 级抽样不确定性，一个是 specimen 级抽样不确定性。

规则：
- 固定 rng_seed=20261023，n_boot=3000。
- 单类（仅一种 label）重采样时：若重采样后仅剩单一 label，该次 AUROC 无法定义，
  记 NaN 并记录次数（不得静默丢弃）。
- 只使用冻结 cell 的 scores_f0/scores_r1/labels/specimen；label 只用 image_anomaly。
"""
from __future__ import annotations
import json
from pathlib import Path
from collections import defaultdict

import numpy as np
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
CELLDIR = ROOT / "results" / "e22" / "cells"
OUTDIR = ROOT / "results" / "paper_supplement_20261023"

CATS = ["Bird", "Car", "Cube", "Dice", "Doll", "Holder", "Motor", "Ring", "Teapot", "Tube"]
SEEDS = [0, 1, 2]
N_BOOT = 3000
RNG_SEED = 20261023


def specimen_cluster_bootstrap(scores_f0, scores_r1, labels, specimen, n_boot, seed):
    """以 specimen 为 cluster 的 paired bootstrap，返回 ΔAUROC 分布 + 退化次数。

    labels 必须与 specimen 一一对应；同一 specimen 的 label 应一致（已由上游保证）。
    """
    labels = np.asarray(labels, dtype=int)
    s0 = np.asarray(scores_f0, dtype=float)
    s1 = np.asarray(scores_r1, dtype=float)

    # 建立 specimen -> 索引
    spec_idx = defaultdict(list)
    for i, s in enumerate(specimen):
        spec_idx[s].append(i)
    spec_keys = sorted(spec_idx.keys())
    n_spec = len(spec_keys)
    # 每个 specimen 的 label（取该 specimen 内所有图的众数，应为单一 label）
    spec_label = {}
    for s, idxs in spec_idx.items():
        labs = labels[idxs]
        assert len(set(labs.tolist())) == 1, f"specimen {s} has mixed labels"
        spec_label[s] = int(labs[0])

    rng = np.random.default_rng(seed)
    dist = np.empty(n_boot, dtype=float)
    n_degenerate = 0
    for b in range(n_boot):
        # 以 specimen 为单位有放回重采样
        chosen = rng.integers(0, n_spec, size=n_spec)
        chosen_keys = [spec_keys[c] for c in chosen]
        # 收集这些 specimen 下的所有图索引
        idxs = []
        for k in chosen_keys:
            idxs.extend(spec_idx[k])
        idxs = np.asarray(idxs, dtype=int)
        if len(idxs) == 0:
            dist[b] = np.nan
            n_degenerate += 1
            continue
        yb = labels[idxs]
        if len(set(yb.tolist())) != 2:
            dist[b] = np.nan
            n_degenerate += 1
            continue
        a0 = roc_auc_score(yb, s0[idxs])
        a1 = roc_auc_score(yb, s1[idxs])
        dist[b] = a1 - a0

    valid = dist[~np.isnan(dist)]
    return {
        "n_specimen": n_spec,
        "n_boot_requested": n_boot,
        "n_boot_effective": int((~np.isnan(dist)).sum()),
        "n_degenerate": n_degenerate,
        "rng_seed": seed,
        "d_auroc_mean": float(np.nanmean(dist)),
        "d_auroc_ci_2p5": float(np.nanpercentile(valid, 2.5)),
        "d_auroc_ci_97p5": float(np.nanpercentile(valid, 97.5)),
        "p_d_auroc_gt0": float((valid > 0).mean()),
    }


def main():
    out = {"note": "specimen-level paired cluster bootstrap (new supplementary analysis, "
                   "estimand = specimen-level resampling uncertainty of dAUROC)",
           "rng_seed": RNG_SEED, "n_boot": N_BOOT, "per_category": {}}
    for cat in CATS:
        # 对每类，3 seed 的逐图分数拼接后做 cluster bootstrap（specimen 跨 seed 独立，
        # 但每个 cell 的 specimen 编号是同一套工件；这里对每个 seed 单独做，再取平均）
        per_seed = {}
        for seed in SEEDS:
            c = json.loads((CELLDIR / f"{cat}_seed{seed}.json").read_text())
            per_seed[str(seed)] = specimen_cluster_bootstrap(
                c["scores_f0"], c["scores_r1"], c["labels"], c["specimen"],
                N_BOOT, RNG_SEED + seed)
        # 汇总 3 seed 的均值与 CI（取各 seed CI 端点的均值，仅描述性）
        means = [per_seed[str(s)]["d_auroc_mean"] for s in SEEDS]
        ci_lo = np.mean([per_seed[str(s)]["d_auroc_ci_2p5"] for s in SEEDS])
        ci_hi = np.mean([per_seed[str(s)]["d_auroc_ci_97p5"] for s in SEEDS])
        p_gt0 = np.mean([per_seed[str(s)]["p_d_auroc_gt0"] for s in SEEDS])
        out["per_category"][cat] = {
            "per_seed": per_seed,
            "mean_of_seed_means": float(np.mean(means)),
            "mean_ci_2p5": float(ci_lo),
            "mean_ci_97p5": float(ci_hi),
            "mean_p_d_auroc_gt0": float(p_gt0),
        }

    with open(OUTDIR / "audit_a3_specimen_cluster_bootstrap.json", "w") as f:
        json.dump(out, f, indent=1, ensure_ascii=False)

    print("=== A3 补充：specimen 级 cluster bootstrap ===")
    print(f"rng_seed={RNG_SEED} n_boot={N_BOOT}")
    for cat in CATS:
        r = out["per_category"][cat]
        n_spec = r["per_seed"]["0"]["n_specimen"]
        n_degen = sum(r["per_seed"][str(s)]["n_degenerate"] for s in SEEDS)
        print(f"  {cat:8s} n_specimen={n_spec:3d} mean_dAUROC={r['mean_of_seed_means']:+.5f} "
              f"CI=[{r['mean_ci_2p5']:+.5f},{r['mean_ci_97p5']:+.5f}] "
              f"P(>0)={r['mean_p_d_auroc_gt0']:.3f} degen_total={n_degen}")
    print(f"wrote {OUTDIR / 'audit_a3_specimen_cluster_bootstrap.json'}")


if __name__ == "__main__":
    main()
