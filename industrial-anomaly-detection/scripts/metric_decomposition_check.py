"""Experiment 1D — metric decomposition sanity check。

目的：解释 Δz（sample-level）与 d'（group-level）为何方向可能相反。
不是更换 outcome，也不是找显著性，而是把 α-IN 对 good distribution /
defect mean / defect variance 的影响拆开看清楚。

对每个 (seed, defect_type, alpha) 输出：
  mu_good, std_good, mu_defect, std_defect, mean_gap, mean_sample_z, pooled_std, dprime
重点比较 alpha=0 vs alpha=1。

口径（与冻结口径一致）：
  mean_gap      = mu_defect - mu_good
  mean_sample_z = mean[(score_defect - mu_good) / std_good]
  pooled_std    = sqrt((std_defect^2 + std_good^2) / 2)
  dprime        = mean_gap / pooled_std

输出：
  results/experiment_1d/summary/metric_decomposition.csv
  results/experiment_1d/figures/metric_decomposition.png
"""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_1C = PROJECT_ROOT / "results" / "experiment_1c"
OUT = PROJECT_ROOT / "results" / "experiment_1d"
SUMMARY = OUT / "summary"
FIGURES = OUT / "figures"

ALPHAS = [0.0, 0.25, 0.5, 0.75, 1.0]
DEFECT_TYPES = ["broken_large", "broken_small", "contamination"]
SEEDS = [0, 1, 2]


def load_by_seed():
    good = {s: defaultdict(list) for s in SEEDS}
    defect = {s: defaultdict(lambda: defaultdict(list)) for s in SEEDS}
    for s in SEEDS:
        with open(RESULTS_1C / f"seed_{s}" / "raw" / "all_sample_scores.csv", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                if r["split"] != "test":
                    continue
                a = float(r["alpha"])
                sc = float(r["anomaly_score"])
                if r["defect_type"] == "good":
                    good[s][a].append(sc)
                else:
                    defect[s][r["defect_type"]][a].append(sc)
    return good, defect


def main() -> None:
    SUMMARY.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)

    good, defect = load_by_seed()

    rows = []
    for s in SEEDS:
        for dt in DEFECT_TYPES:
            for a in ALPHAS:
                g = np.array(good[s][a])
                d = np.array(defect[s][dt][a])
                mu_g = float(g.mean())
                sd_g = float(g.std(ddof=1))
                mu_d = float(d.mean())
                sd_d = float(d.std(ddof=1))
                mean_gap = mu_d - mu_g
                z_vals = (d - mu_g) / sd_g if sd_g > 0 else np.full_like(d, np.nan)
                mean_sample_z = float(np.nanmean(z_vals))
                pooled_std = float(np.sqrt((sd_d ** 2 + sd_g ** 2) / 2))
                dprime = mean_gap / pooled_std if pooled_std > 0 else float("nan")
                rows.append({
                    "seed": s, "defect_type": dt, "alpha": a,
                    "mu_good": mu_g, "std_good": sd_g,
                    "mu_defect": mu_d, "std_defect": sd_d,
                    "mean_gap": mean_gap, "mean_sample_z": mean_sample_z,
                    "pooled_std": pooled_std, "dprime": dprime,
                })

    fieldnames = ["seed", "defect_type", "alpha", "mu_good", "std_good",
                  "mu_defect", "std_defect", "mean_gap", "mean_sample_z",
                  "pooled_std", "dprime"]
    with open(SUMMARY / "metric_decomposition.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    # ---- 控制台输出 alpha=0 vs 1 对比 ----
    print("=== metric decomposition: alpha=0 -> alpha=1 (per seed, per type) ===")
    print(f"{'seed':>4} {'type':16s} {'Δmean_gap':>10} {'Δstd_good':>10} "
          f"{'Δstd_defect':>12} {'Δmean_z':>9} {'Δdprime':>9}")
    for s in SEEDS:
        for dt in DEFECT_TYPES:
            r0 = next(r for r in rows if r["seed"] == s and r["defect_type"] == dt and r["alpha"] == 0.0)
            r1 = next(r for r in rows if r["seed"] == s and r["defect_type"] == dt and r["alpha"] == 1.0)
            print(f"{s:>4} {dt:16s} {r1['mean_gap']-r0['mean_gap']:>+10.3f} "
                  f"{r1['std_good']-r0['std_good']:>+10.3f} "
                  f"{r1['std_defect']-r0['std_defect']:>+12.3f} "
                  f"{r1['mean_sample_z']-r0['mean_sample_z']:>+9.3f} "
                  f"{r1['dprime']-r0['dprime']:>+9.3f}")

    # ---- 图：三类 defect 从 alpha=0 -> 1 各量变化 ----
    metrics = ["mean_gap", "std_good", "std_defect", "mean_sample_z", "dprime"]
    metric_labels = ["mean gap (μD-μG)", "good std (σG)", "defect std (σD)",
                     "mean sample z", "d' (pooled)"]
    colors = {"broken_large": "tab:red", "broken_small": "tab:orange", "contamination": "tab:blue"}

    fig, axes = plt.subplots(1, len(metrics), figsize=(len(metrics) * 3.2, 4.2))
    for ax, m, mlab in zip(axes, metrics, metric_labels):
        # 三 seed 平均后的曲线（alpha 0 -> 1）
        for dt in DEFECT_TYPES:
            vals = []
            for a in ALPHAS:
                v = [r[m] for r in rows if r["defect_type"] == dt and r["alpha"] == a]
                vals.append(float(np.mean(v)))
            ax.plot(ALPHAS, vals, "o-", color=colors[dt], label=dt, markersize=4)
        ax.set_xlabel("alpha")
        ax.set_ylabel(mlab)
        ax.set_title(mlab)
        ax.grid(alpha=0.3)
    axes[-1].legend(fontsize=8)
    fig.suptitle("Metric decomposition: how α-IN changes each quantity (3-seed mean)",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(FIGURES / "metric_decomposition.png", dpi=150)
    plt.close(fig)

    # ---- 回答诊断问题所需的关键汇总 ----
    print("\n=== 诊断汇总（3-seed mean，alpha=0 -> 1 变化量）===")
    for dt in DEFECT_TYPES:
        vals = {}
        for m in metrics:
            d0 = np.mean([r[m] for r in rows if r["defect_type"] == dt and r["alpha"] == 0.0])
            d1 = np.mean([r[m] for r in rows if r["defect_type"] == dt and r["alpha"] == 1.0])
            vals[m] = d1 - d0
        print(f"  {dt:16s} Δmean_gap={vals['mean_gap']:+.3f} Δstd_good={vals['std_good']:+.3f} "
              f"Δstd_defect={vals['std_defect']:+.3f} Δmean_z={vals['mean_sample_z']:+.3f} "
              f"Δdprime={vals['dprime']:+.3f}")

    print("\n[done] metric decomposition check 完成")
    print(f"[done] csv={SUMMARY / 'metric_decomposition.csv'}")
    print(f"[done] fig={FIGURES / 'metric_decomposition.png'}")


if __name__ == "__main__":
    main()
