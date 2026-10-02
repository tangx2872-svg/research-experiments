"""Experiment 1C 分析脚本：跨 seed 稳定性验证。

读取 results/experiment_1c/seed_{0,1,2}/raw/all_sample_scores.csv，
计算每个 seed × alpha × defect 的 d' 分离度，回答：

  1. 三类 defect 的 d' 曲线在不同 seed 下是否一致（图1，seed 用不同线型）？
  2. Δd' = d'(α=1) - d'(α=0) 在每个 defect 上的跨 seed mean/std（图2）。
  3. Continue / Stop 判断标准输出。

d' 定义（与 1B 一致）：
  d' = (mean_defect - mean_normal) / sqrt((var_defect + var_normal) / 2)

输出到 results/experiment_1c/ 下 summary/ 与 figures/。
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_ROOT = PROJECT_ROOT / "results" / "experiment_1c"
SUMMARY_DIR = RESULTS_ROOT / "summary"
FIGURES_DIR = RESULTS_ROOT / "figures"

ALPHAS = [0.0, 0.25, 0.5, 0.75, 1.0]
DEFECT_TYPES = ["broken_large", "broken_small", "contamination"]
SEEDS = [0, 1, 2]

# 线型/标记：seed 用不同线型区分（协议要求）
SEED_STYLES = {
    0: dict(color="tab:blue", linestyle="-", marker="o"),
    1: dict(color="tab:green", linestyle="--", marker="s"),
    2: dict(color="tab:purple", linestyle=":", marker="^"),
}
DEFECT_COLORS = {
    "broken_large": "tab:red",
    "broken_small": "tab:orange",
    "contamination": "tab:blue",
}


def load_raw(seed: int) -> list[dict]:
    path = RESULTS_ROOT / f"seed_{seed}" / "raw" / "all_sample_scores.csv"
    rows = []
    with open(path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            r["alpha"] = float(r["alpha"])
            r["anomaly_score"] = float(r["anomaly_score"])
            rows.append(r)
    return rows


def compute_dprime(seed: int) -> dict[float, dict[str, float]]:
    """返回 {alpha: {defect_type: d'}}。"""
    rows = load_raw(seed)
    test_rows = [r for r in rows if r["split"] == "test"]

    good_by_alpha: dict[float, list[float]] = defaultdict(list)
    defect_by_alpha: dict[float, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for r in test_rows:
        if r["defect_type"] == "good":
            good_by_alpha[r["alpha"]].append(r["anomaly_score"])
        else:
            defect_by_alpha[r["alpha"]][r["defect_type"]].append(r["anomaly_score"])

    dprime_table: dict[float, dict[str, float]] = {}
    for a in ALPHAS:
        good = np.array(good_by_alpha[a])
        row = {}
        for dt in DEFECT_TYPES:
            d = np.array(defect_by_alpha[a][dt])
            pooled_std = np.sqrt((good.std() ** 2 + d.std() ** 2) / 2)
            row[dt] = float((d.mean() - good.mean()) / pooled_std) if pooled_std > 0 else float("nan")
        dprime_table[a] = row
    return dprime_table


def main() -> None:
    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # 检查 3 个 seed 是否都跑完
    missing = [s for s in SEEDS if not (RESULTS_ROOT / f"seed_{s}" / "raw" / "all_sample_scores.csv").exists()]
    if missing:
        print(f"[error] 缺少 seed 结果：{missing}，请先运行 experiment1c_multiseed.py")
        return

    # ---- 计算每个 seed 的 d' ----
    dprime_by_seed: dict[int, dict[float, dict[str, float]]] = {}
    for s in SEEDS:
        dprime_by_seed[s] = compute_dprime(s)

    # ---- 打印 d' 表格 ----
    print("=== d' 分离度（每个 seed × alpha） ===")
    for s in SEEDS:
        print(f"\n--- seed={s} ---")
        print(f"{'alpha':>6} | {'broken_large':>13} | {'broken_small':>13} | {'contamination':>13}")
        for a in ALPHAS:
            row = dprime_by_seed[s][a]
            print(f"{a:>6} | {row['broken_large']:>13.3f} | {row['broken_small']:>13.3f} | {row['contamination']:>13.3f}")

    # ---- Δd' = d'(α=1) - d'(α=0)，跨 seed mean/std ----
    print("\n=== Δd' = d'(α=1) - d'(α=0) 跨 seed 统计 ===")
    delta_by_seed: dict[int, dict[str, float]] = {}
    for s in SEEDS:
        delta_by_seed[s] = {
            dt: dprime_by_seed[s][1.0][dt] - dprime_by_seed[s][0.0][dt]
            for dt in DEFECT_TYPES
        }

    for dt in DEFECT_TYPES:
        vals = [delta_by_seed[s][dt] for s in SEEDS]
        # 协议口径：sample std（ddof=1）
        mean = float(np.mean(vals))
        std = float(np.std(vals, ddof=1))
        print(f"  {dt:16s}: seed0={vals[0]:+.3f}, seed1={vals[1]:+.3f}, seed2={vals[2]:+.3f}"
              f"  -> mean={mean:+.3f}, std={std:.3f}")

    # ---- 保存 Δd' 统计到 CSV ----
    with open(SUMMARY_DIR / "delta_dprime_stability.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["defect_type", "seed0", "seed1", "seed2", "mean", "std"])
        for dt in DEFECT_TYPES:
            vals = [delta_by_seed[s][dt] for s in SEEDS]
            w.writerow([dt, vals[0], vals[1], vals[2], float(np.mean(vals)), float(np.std(vals, ddof=1))])

    # ---- 保存完整 d' 表（供复现） ----
    with open(SUMMARY_DIR / "dprime_all_seeds.json", "w", encoding="utf-8") as f:
        json.dump(
            {str(s): {str(a): dprime_by_seed[s][a] for a in ALPHAS} for s in SEEDS},
            f, indent=2, ensure_ascii=False,
        )

    # ---- 图1：三类 defect 的 d' 曲线，seed 用不同线型 ----
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)
    for idx, dt in enumerate(DEFECT_TYPES):
        ax = axes[idx]
        for s in SEEDS:
            y = [dprime_by_seed[s][a][dt] for a in ALPHAS]
            ax.plot(ALPHAS, y, label=f"seed={s}", **SEED_STYLES[s])
        ax.set_title(dt)
        ax.set_xlabel("alpha")
        ax.set_ylabel("d'" if idx == 0 else "")
        ax.grid(alpha=0.3)
        ax.legend()
    fig.suptitle("Figure 1: d' vs alpha (per defect, per seed)")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "dprime_curves_per_seed.png", dpi=150)
    plt.close(fig)

    # ---- 图2：seed 稳定性（Δd' 的 bar + error bar） ----
    fig, ax = plt.subplots(figsize=(8, 5))
    means = []
    stds = []
    for dt in DEFECT_TYPES:
        vals = [delta_by_seed[s][dt] for s in SEEDS]
        means.append(float(np.mean(vals)))
        stds.append(float(np.std(vals, ddof=1)))
    xs = np.arange(len(DEFECT_TYPES))
    ax.bar(xs, means, yerr=stds, capsize=6, color=[DEFECT_COLORS[dt] for dt in DEFECT_TYPES], alpha=0.8)
    # 每个 seed 的散点叠加
    for s in SEEDS:
        ax.scatter(
            xs + (s - 1) * 0.15,
            [delta_by_seed[s][dt] for dt in DEFECT_TYPES],
            label=f"seed={s}", zorder=3, **{k: v for k, v in SEED_STYLES[s].items() if k != "linestyle"},
        )
    ax.axhline(0, color="gray", linewidth=0.8)
    ax.set_xticks(xs)
    ax.set_xticklabels(DEFECT_TYPES)
    ax.set_ylabel("Δd' = d'(α=1) - d'(α=0)")
    ax.set_title("Figure 2: seed stability of Δd' (mean ± std)")
    ax.legend()
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "delta_dprime_stability.png", dpi=150)
    plt.close(fig)

    # ---- Continue / Stop 判断标准 ----
    print("\n=== Continue / Stop 判断 ===")
    # 判断：broken_large 三个 seed 是否全部明显下降（Δd' < 0 且 |Δd'| 较大）
    large_deltas = [delta_by_seed[s]["broken_large"] for s in SEEDS]
    small_deltas = [delta_by_seed[s]["broken_small"] for s in SEEDS]
    cont_deltas = [delta_by_seed[s]["contamination"] for s in SEEDS]

    large_all_down = all(d < -1.0 for d in large_deltas)   # 全部明显下降（阈值 -1.0）
    small_small = all(abs(d) < 2.0 for d in small_deltas)   # 变化较小
    cont_diff = not all(d < -1.0 for d in cont_deltas)      # 方向不同或不下降
    std_small = float(np.std(large_deltas, ddof=1)) < 1.0           # std 较小（sample std）

    print(f"  broken_large  Δd' = {[f'{d:+.2f}' for d in large_deltas]}  -> 全部明显下降: {large_all_down}")
    print(f"  broken_small  Δd' = {[f'{d:+.2f}' for d in small_deltas]}  -> 变化较小: {small_small}")
    print(f"  contamination Δd' = {[f'{d:+.2f}' for d in cont_deltas]}  -> 方向不同/不下降: {cont_diff}")
    print(f"  broken_large std = {np.std(large_deltas, ddof=1):.3f}  -> std 较小: {std_small}")

    if large_all_down and small_small and cont_diff and std_small:
        verdict = "CONTINUE → 进入 Experiment 1D (Size Confound Analysis)"
    elif large_all_down and std_small:
        verdict = "CONTINUE（主效应稳定；small/cont 的精细分化需结合图1人工复核）"
    else:
        verdict = "STOP / RETHINK → 现象可能主要来自 coreset/feature 随机性，需重新设计"

    print(f"\n  >>> 判定：{verdict}")

    # 保存判定
    with open(SUMMARY_DIR / "verdict.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "delta_dprime_by_seed": {str(s): delta_by_seed[s] for s in SEEDS},
                "large_all_down": large_all_down,
                "small_small": small_small,
                "cont_diff": cont_diff,
                "std_small": std_small,
                "verdict": verdict,
            },
            f, indent=2, ensure_ascii=False,
        )

    print(f"\n[done] 分析完成，summary={SUMMARY_DIR} figures={FIGURES_DIR}")


if __name__ == "__main__":
    main()
