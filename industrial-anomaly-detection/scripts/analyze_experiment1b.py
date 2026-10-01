"""Experiment 1B 分析脚本：读取 raw CSV + area + anomaly maps，计算指标并生成图表。

回答任务 Q1-Q7：
  Q1 alpha=0 是否复现 baseline
  Q2 三类 defect 是否表现不同检测能力变化曲线
  Q3 差异是否多数样本共同支持（vs 少数 outlier）
  Q4 缺陷面积是否解释差异
  Q5 good vs defect score distribution 如何变化（是否只是 scale 改变）
  Q6 heatmap 是否与 quantitative metric 一致
  Q7 结论属于 A/B/C/D 哪类

输出到 results/experiment_1b/summary/ 与 figures/。
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
RESULTS_ROOT = PROJECT_ROOT / "results" / "experiment_1b"
RAW_DIR = RESULTS_ROOT / "raw"
SUMMARY_DIR = RESULTS_ROOT / "summary"
FIGURES_DIR = RESULTS_ROOT / "figures"
HEATMAPS_DIR = RESULTS_ROOT / "heatmaps"

ALPHAS = [0.0, 0.25, 0.5, 0.75, 1.0]
DEFECT_TYPES = ["broken_large", "broken_small", "contamination"]

DATA_ROOT = PROJECT_ROOT / "data" / "mvtec_ad"


def load_raw() -> list[dict]:
    rows = []
    with open(RAW_DIR / "all_sample_scores.csv", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            r["alpha"] = float(r["alpha"])
            r["anomaly_score"] = float(r["anomaly_score"])
            r["is_anomaly"] = int(r["is_anomaly"])
            r["gt_label"] = int(r["gt_label"])
            rows.append(r)
    return rows


def load_thresholds() -> dict[float, float]:
    with open(SUMMARY_DIR / "thresholds.json", encoding="utf-8") as f:
        d = json.load(f)
    return {float(k): v for k, v in d.items()}


def load_area() -> dict[str, float]:
    area = {}
    with open(SUMMARY_DIR / "area_analysis.csv", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            area[r["image_path"]] = float(r["defect_area_ratio"])
    return area


def auroc(y_true: list[int], y_score: list[float]) -> float:
    """计算 AUROC（用 rank 方法，避免依赖 sklearn）。"""
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    n_pos = (y_true == 1).sum()
    n_neg = (y_true == 0).sum()
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    # 对 score 排序，正样本 rank 之和
    order = np.argsort(y_score)
    ranks = np.empty_like(order)
    ranks[order] = np.arange(1, len(y_score) + 1)
    rank_sum_pos = ranks[y_true == 1].sum()
    auroc_val = (rank_sum_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
    return float(auroc_val)


def main() -> None:
    rows = load_raw()
    thresholds = load_thresholds()
    area = load_area()

    # 按 alpha + defect_type 分组（test split）
    test_rows = [r for r in rows if r["split"] == "test"]
    val_rows = [r for r in rows if r["split"] == "validation"]

    # good 分数（test）
    good_by_alpha: dict[float, list[float]] = defaultdict(list)
    defect_by_alpha: dict[float, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for r in test_rows:
        if r["defect_type"] == "good":
            good_by_alpha[r["alpha"]].append(r["anomaly_score"])
        else:
            defect_by_alpha[r["alpha"]][r["defect_type"]].append(r["anomaly_score"])

    # ---- defect-wise AUROC ----
    print("=== Defect-wise AUROC (test/good vs defect) ===")
    print(f"{'alpha':>6} | {'broken_large':>13} | {'broken_small':>13} | {'contamination':>13}")
    auroc_table = {}
    for a in ALPHAS:
        good = good_by_alpha[a]
        row_auroc = {}
        for dt in DEFECT_TYPES:
            d = defect_by_alpha[a][dt]
            y_true = [0] * len(good) + [1] * len(d)
            y_score = good + d
            row_auroc[dt] = auroc(y_true, y_score)
        auroc_table[a] = row_auroc
        print(f"{a:>6} | {row_auroc['broken_large']:>13.4f} | {row_auroc['broken_small']:>13.4f} | {row_auroc['contamination']:>13.4f}")

    # ---- defect-wise Recall + FPR ----
    print("\n=== Defect-wise Recall (threshold tau_alpha) + test/good FPR ===")
    print(f"{'alpha':>6} | {'tau':>10} | {'FPR':>8} | {'large_R':>8} | {'small_R':>8} | {'cont_R':>8}")
    recall_table = {}
    fpr_table = {}
    for a in ALPHAS:
        tau = thresholds[a]
        good = good_by_alpha[a]
        fpr = sum(1 for s in good if s > tau) / len(good)
        fpr_table[a] = fpr
        row_rec = {}
        for dt in DEFECT_TYPES:
            d = defect_by_alpha[a][dt]
            rec = sum(1 for s in d if s > tau) / len(d)
            row_rec[dt] = rec
        recall_table[a] = row_rec
        print(f"{a:>6} | {tau:>10.4f} | {fpr:>8.4f} | {row_rec['broken_large']:>8.4f} | {row_rec['broken_small']:>8.4f} | {row_rec['contamination']:>8.4f}")

    # 保存 defect-wise metrics CSV
    with open(SUMMARY_DIR / "defect_wise_metrics.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["alpha", "threshold", "test_good_fpr",
                    "auroc_broken_large", "auroc_broken_small", "auroc_contamination",
                    "recall_broken_large", "recall_broken_small", "recall_contamination"])
        for a in ALPHAS:
            w.writerow([a, thresholds[a], fpr_table[a],
                        auroc_table[a]["broken_large"], auroc_table[a]["broken_small"], auroc_table[a]["contamination"],
                        recall_table[a]["broken_large"], recall_table[a]["broken_small"], recall_table[a]["contamination"]])

    # ---- Figure 1: AUROC vs alpha ----
    fig, ax = plt.subplots(figsize=(7, 5))
    for dt, color in zip(DEFECT_TYPES, ["tab:red", "tab:orange", "tab:blue"]):
        y = [auroc_table[a][dt] for a in ALPHAS]
        ax.plot(ALPHAS, y, marker="o", label=dt, color=color)
    ax.set_xlabel("alpha")
    ax.set_ylabel("image-level AUROC (vs test/good)")
    ax.set_title("Figure 1: Defect-wise AUROC vs alpha")
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_ylim(0, 1.05)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "auroc_vs_alpha.png", dpi=150)
    plt.close(fig)

    # ---- Figure 2: Recall vs alpha ----
    fig, ax = plt.subplots(figsize=(7, 5))
    for dt, color in zip(DEFECT_TYPES, ["tab:red", "tab:orange", "tab:blue"]):
        y = [recall_table[a][dt] for a in ALPHAS]
        ax.plot(ALPHAS, y, marker="o", label=dt, color=color)
    ax.set_xlabel("alpha")
    ax.set_ylabel("Recall (threshold tau_alpha)")
    ax.set_title("Figure 2: Defect-wise Recall vs alpha")
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_ylim(0, 1.05)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "recall_vs_alpha.png", dpi=150)
    plt.close(fig)

    # ---- Figure 3: FPR vs alpha ----
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(ALPHAS, [fpr_table[a] for a in ALPHAS], marker="o", color="tab:green")
    ax.set_xlabel("alpha")
    ax.set_ylabel("test/good FPR")
    ax.set_title("Figure 3: test/good FPR vs alpha")
    ax.grid(alpha=0.3)
    ax.set_ylim(0, 1.05)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fpr_vs_alpha.png", dpi=150)
    plt.close(fig)

    # ---- Figure 4: score distribution (good + 3 defects) vs alpha ----
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    for idx, a in enumerate(ALPHAS):
        ax = axes[idx // 3][idx % 3]
        data = [good_by_alpha[a]] + [defect_by_alpha[a][dt] for dt in DEFECT_TYPES]
        labels = ["good"] + DEFECT_TYPES
        ax.boxplot(data, tick_labels=labels, showfliers=False)
        ax.axhline(thresholds[a], color="red", linestyle="--", linewidth=1, label=f"tau={thresholds[a]:.1f}")
        ax.set_title(f"alpha={a}")
        ax.tick_params(axis="x", rotation=20)
        ax.legend(fontsize=7)
    axes[1][2].axis("off")
    fig.suptitle("Figure 4: anomaly score distribution vs alpha")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "score_distribution.png", dpi=150)
    plt.close(fig)

    # ---- d' 分离度（同 α 内 good vs defect 的标准化分离度，不受阈值影响）----
    print("\n=== Defect-normal 分离度 d'（同 α 内） ===")
    print(f"{'alpha':>6} | {'large_dp':>10} | {'small_dp':>10} | {'cont_dp':>10}")
    dprime_table = {}
    for a in ALPHAS:
        good = np.array(good_by_alpha[a])
        row_dp = {}
        for dt in DEFECT_TYPES:
            d = np.array(defect_by_alpha[a][dt])
            pooled_std = np.sqrt((good.std() ** 2 + d.std() ** 2) / 2)
            if pooled_std == 0:
                row_dp[dt] = float("nan")
            else:
                row_dp[dt] = float((d.mean() - good.mean()) / pooled_std)
        dprime_table[a] = row_dp
        print(f"{a:>6} | {row_dp['broken_large']:>10.3f} | {row_dp['broken_small']:>10.3f} | {row_dp['contamination']:>10.3f}")

    # 保存 d' 到 CSV（追加到 defect_wise_metrics）
    with open(SUMMARY_DIR / "defect_wise_metrics.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["alpha", "threshold", "test_good_fpr",
                    "auroc_broken_large", "auroc_broken_small", "auroc_contamination",
                    "recall_broken_large", "recall_broken_small", "recall_contamination",
                    "dprime_broken_large", "dprime_broken_small", "dprime_contamination"])
        for a in ALPHAS:
            w.writerow([a, thresholds[a], fpr_table[a],
                        auroc_table[a]["broken_large"], auroc_table[a]["broken_small"], auroc_table[a]["contamination"],
                        recall_table[a]["broken_large"], recall_table[a]["broken_small"], recall_table[a]["contamination"],
                        dprime_table[a]["broken_large"], dprime_table[a]["broken_small"], dprime_table[a]["contamination"]])

    # ---- Figure: d' vs alpha（补充图，最关键）----
    fig, ax = plt.subplots(figsize=(7, 5))
    for dt, color in zip(DEFECT_TYPES, ["tab:red", "tab:orange", "tab:blue"]):
        y = [dprime_table[a][dt] for a in ALPHAS]
        ax.plot(ALPHAS, y, marker="o", label=dt, color=color)
    ax.set_xlabel("alpha")
    ax.set_ylabel("d' (good vs defect separation, same-alpha)")
    ax.set_title("Defect-normal separation d' vs alpha")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "dprime_vs_alpha.png", dpi=150)
    plt.close(fig)

    # ---- Figure 5: individual trajectories ----
    # 按 sample 追踪跨 alpha 的 score
    sample_scores: dict[str, dict[float, float]] = defaultdict(dict)
    sample_type: dict[str, str] = {}
    for r in test_rows:
        sample_scores[r["image_path"]][r["alpha"]] = r["anomaly_score"]
        sample_type[r["image_path"]] = r["defect_type"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharex=True)
    for idx, dt in enumerate(DEFECT_TYPES):
        ax = axes[idx]
        for path, scores in sample_scores.items():
            if sample_type[path] != dt:
                continue
            ys = [scores.get(a, float("nan")) for a in ALPHAS]
            ax.plot(ALPHAS, ys, marker=".", alpha=0.5, linewidth=0.8)
        # 类别均值
        mean_ys = [np.nanmean([sample_scores[p].get(a, np.nan) for p in sample_scores if sample_type[p] == dt]) for a in ALPHAS]
        ax.plot(ALPHAS, mean_ys, color="black", linewidth=2.5, label="mean")
        ax.set_title(f"{dt} (n={sum(1 for p in sample_type if sample_type[p]==dt)})")
        ax.set_xlabel("alpha")
        ax.set_ylabel("anomaly score")
        ax.legend()
        ax.grid(alpha=0.3)
    fig.suptitle("Figure 5: individual sample trajectories")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "individual_trajectories.png", dpi=150)
    plt.close(fig)

    # ---- 计算每样本的 alpha 响应（score 从 α=0 到 α=1 的变化）----
    print("\n=== 每样本 score 变化 (alpha=0 -> alpha=1) ===")
    for dt in DEFECT_TYPES:
        deltas = []
        for path, scores in sample_scores.items():
            if sample_type[path] == dt and 0.0 in scores and 1.0 in scores:
                deltas.append(scores[1.0] - scores[0.0])
        if deltas:
            deltas = np.array(deltas)
            n_up = (deltas > 0).sum()
            print(f"  {dt:16s}: n={len(deltas)}, 上升 {n_up}/{len(deltas)}, "
                  f"mean_delta={deltas.mean():+.3f}, median_delta={np.median(deltas):+.3f}")

    # ---- Figure 6: area vs response ----
    # 每 defect 样本的 delta vs area
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharex=False)
    area_delta: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for path, scores in sample_scores.items():
        dt = sample_type.get(path)
        if dt in DEFECT_TYPES and 0.0 in scores and 1.0 in scores and path in area:
            area_delta[dt].append((area[path], scores[1.0] - scores[0.0]))
    for idx, dt in enumerate(DEFECT_TYPES):
        ax = axes[idx]
        pts = area_delta[dt]
        if pts:
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            ax.scatter(xs, ys, alpha=0.7)
            # 相关系数
            if len(xs) > 2:
                corr = np.corrcoef(xs, ys)[0, 1]
                ax.set_title(f"{dt}\ncorr={corr:.3f}")
            else:
                ax.set_title(dt)
        ax.set_xlabel("defect_area_ratio")
        ax.set_ylabel("score delta (α1 - α0)")
        ax.axhline(0, color="gray", linewidth=0.5)
        ax.grid(alpha=0.3)
    fig.suptitle("Figure 6: defect area vs α-response (size confound check)")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "area_vs_response.png", dpi=150)
    plt.close(fig)

    # ---- 保存 area_delta 分析 ----
    with open(SUMMARY_DIR / "area_delta_analysis.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["image_path", "defect_type", "defect_area_ratio", "score_delta_alpha0_to_1"])
        for dt in DEFECT_TYPES:
            for path, scores in sample_scores.items():
                if sample_type[path] == dt and 0.0 in scores and 1.0 in scores:
                    w.writerow([path, dt, area.get(path, float("nan")), scores[1.0] - scores[0.0]])

    print("\n[done] 分析完成")
    print(f"[done] 图表目录 = {FIGURES_DIR}")
    print(f"[done] summary 目录 = {SUMMARY_DIR}")


if __name__ == "__main__":
    main()
