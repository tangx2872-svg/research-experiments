"""F_alpha 实验分析：alpha=0 vs alpha=1 的 paired anomaly-score 分析。

只读取已有 results/falpha_experiment/bottle/alpha_{0,1}/ 下的
predictions.jsonl / metrics.json，不重新训练、不修改原始结果。

输出（写入归档目录下 reruns/analysis/，保留历史分析不变）：
  - paired_scores.csv          每张测试图的 paired score
  - stats_summary.json         分 defect_type 的统计 + Wilcoxon
  - fig1_mean_score.png        defect_type × mean anomaly score (alpha0 vs alpha1)
  - fig2_delta_boxplot.png     delta_score 分布（boxplot）
  - fig3_paired_slope.png      每 defect_type 的 alpha0→alpha1 变化（子图）
"""

from __future__ import annotations

import json
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # 无显示环境
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy import stats  # noqa: E402

ROOT = Path(__file__).resolve().parents[1] / "results" / "falpha_experiment" / "bottle"
OUT = Path(__file__).resolve().parents[1] / "reruns" / "analysis"

# defect type 顺序（good 放最后，异常类放前面便于对比）
DEFECT_ORDER = ["broken_large", "broken_small", "contamination", "good"]


def load_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def paired_records(d0: dict, d1: dict) -> list[dict]:
    """按 image_path 配对，返回带 delta_score 的记录列表。"""
    records = []
    for path in sorted(d0.keys()):
        r0, r1 = d0[path], d1[path]
        s0 = r0["anomaly_score"]
        s1 = r1["anomaly_score"]
        records.append(
            {
                "image_path": path,
                "defect_type": r0["defect_type"],
                "gt_label": r0["gt_label"],
                "score_alpha0": s0,
                "score_alpha1": s1,
                "delta_score": s1 - s0,
            }
        )
    return records


def group_stats(records: list[dict], defect: str) -> dict:
    rows = [r for r in records if r["defect_type"] == defect]
    a0 = np.array([r["score_alpha0"] for r in rows])
    a1 = np.array([r["score_alpha1"] for r in rows])
    d = a1 - a0
    n_down = int((d < 0).sum())
    n_up = int((d > 0).sum())
    n_tied = int((d == 0).sum())

    # Wilcoxon signed-rank（配对小样本），仅在有非零差异时计算
    wilcoxon = None
    r_effect = None
    dd = d[d != 0]
    if len(dd) >= 2:
        try:
            w = stats.wilcoxon(a0, a1, alternative="two-sided", zero_method="wilcox")
            wilcoxon = {"statistic": float(w.statistic), "pvalue": float(w.pvalue)}
        except Exception as exc:  # noqa: BLE001
            wilcoxon = {"error": str(exc)}
        # rank-biserial correlation（方向性 effect size，范围 [-1,1]，>0 表示 alpha1 更高）
        ranks = stats.rankdata(np.abs(dd))
        w_pos = float(ranks[dd > 0].sum())
        w_neg = float(ranks[dd < 0].sum())
        if w_pos + w_neg > 0:
            r_effect = (w_pos - w_neg) / (w_pos + w_neg)

    return {
        "defect_type": defect,
        "count": len(rows),
        "mean_score_alpha0": float(a0.mean()),
        "mean_score_alpha1": float(a1.mean()),
        "mean_delta": float(d.mean()),
        "median_delta": float(np.median(d)),
        "std_delta": float(d.std(ddof=1)) if len(d) > 1 else 0.0,
        "min_delta": float(d.min()),
        "max_delta": float(d.max()),
        "n_down": n_down,
        "n_up": n_up,
        "n_tied": n_tied,
        "prop_down": round(n_down / len(d), 4) if len(d) else None,
        "prop_up": round(n_up / len(d), 4) if len(d) else None,
        "wilcoxon": wilcoxon,
        "rank_biserial_r": round(r_effect, 4) if r_effect is not None else None,
    }


def fig1_mean_score(all_stats: dict, out: Path) -> None:
    defects = [d for d in DEFECT_ORDER if d in all_stats]
    a0 = [all_stats[d]["mean_score_alpha0"] for d in defects]
    a1 = [all_stats[d]["mean_score_alpha1"] for d in defects]
    x = np.arange(len(defects))
    w = 0.36
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar(x - w / 2, a0, w, label="alpha=0", color="#4C72B0")
    ax.bar(x + w / 2, a1, w, label="alpha=1", color="#DD8452")
    ax.set_xticks(x)
    ax.set_xticklabels(defects)
    ax.set_ylabel("mean anomaly score")
    ax.set_title("Defect type x mean anomaly score (bottle)")
    ax.set_ylim(0, 1)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)


def fig2_delta_boxplot(records: list[dict], out: Path) -> None:
    defects = DEFECT_ORDER
    data = [
        [r["delta_score"] for r in records if r["defect_type"] == d]
        for d in defects
    ]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    bp = ax.boxplot(data, tick_labels=defects, patch_artist=True)
    colors = ["#DD8452", "#55A868", "#C44E52", "#8172B3"]
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.7)
    ax.axhline(0, color="black", lw=0.8, ls="--")
    ax.set_ylabel("delta_score = score(alpha=1) - score(alpha=0)")
    ax.set_title("delta_score distribution per defect type (bottle)")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)


def fig3_paired_slope(records: list[dict], out: Path) -> None:
    defects = [d for d in DEFECT_ORDER if any(r["defect_type"] == d for r in records)]
    fig, axes = plt.subplots(1, len(defects), figsize=(4 * len(defects), 4), sharey=False)
    if len(defects) == 1:
        axes = [axes]
    for ax, d in zip(axes, defects):
        rows = [r for r in records if r["defect_type"] == d]
        a0 = [r["score_alpha0"] for r in rows]
        a1 = [r["score_alpha1"] for r in rows]
        for x, y in zip(a0, a1):
            c = "#DD8452" if y > x else "#4C72B0"
            ax.plot([0, 1], [x, y], color=c, alpha=0.55, lw=1)
            ax.scatter([0, 1], [x, y], color=c, s=14, alpha=0.9)
        lo = min(min(a0), min(a1))
        hi = max(max(a0), max(a1))
        pad = (hi - lo) * 0.08 + 1e-6
        ax.set_ylim(lo - pad, hi + pad)
        ax.set_xlim(-0.3, 1.3)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["alpha=0", "alpha=1"])
        ax.set_title(f"{d} (n={len(rows)})")
        ax.grid(axis="y", alpha=0.3)
        if ax is axes[0]:
            ax.set_ylabel("anomaly score")
    fig.suptitle("Paired anomaly score change (alpha=0 -> alpha=1)", y=1.02)
    fig.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)


def top_samples(records: list[dict], defect: str, k: int = 5) -> list[dict]:
    rows = [r for r in records if r["defect_type"] == defect]
    rows_sorted = sorted(rows, key=lambda r: r["delta_score"])
    most_neg = rows_sorted[:k]
    most_pos = rows_sorted[-k:][::-1]
    return most_neg, most_pos


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    a0 = load_jsonl(ROOT / "alpha_0" / "predictions.jsonl")
    a1 = load_jsonl(ROOT / "alpha_1" / "predictions.jsonl")

    d0 = {r["image_path"]: r for r in a0}
    d1 = {r["image_path"]: r for r in a1}

    # ---- 1. 配对性检查 ----
    print("=" * 70)
    print("[1] 数据配对性检查")
    print(f"  alpha=0 记录数: {len(a0)}")
    print(f"  alpha=1 记录数: {len(a1)}")
    print(f"  alpha=0 唯一 image_path 数: {len(d0)}")
    print(f"  alpha=1 唯一 image_path 数: {len(d1)}")
    same_keys = set(d0.keys()) == set(d1.keys())
    print(f"  image_path 集合完全一致: {same_keys}")
    if not same_keys:
        only0 = set(d0) - set(d1)
        only1 = set(d1) - set(d0)
        print(f"    仅在 alpha=0: {len(only0)} 张")
        print(f"    仅在 alpha=1: {len(only1)} 张")
        raise SystemExit("配对失败：两个实验结果无法一一对应，停止分析。")

    mismatch_defect = 0
    mismatch_gt = 0
    missing_score = 0
    for p in d0:
        if d0[p]["defect_type"] != d1[p]["defect_type"]:
            mismatch_defect += 1
        if d0[p]["gt_label"] != d1[p]["gt_label"]:
            mismatch_gt += 1
        if d0[p].get("anomaly_score") is None or d1[p].get("anomaly_score") is None:
            missing_score += 1
    print(f"  defect_type 不一致: {mismatch_defect}")
    print(f"  gt_label 不一致: {mismatch_gt}")
    print(f"  anomaly_score 缺失: {missing_score}")

    # 各 defect_type 数量
    counts = defaultdict(int)
    for p in d0:
        counts[d0[p]["defect_type"]] += 1
    print("  各 defect_type 数量:")
    for d in DEFECT_ORDER:
        print(f"    {d}: {counts.get(d, 0)}")

    # ---- 2. paired delta ----
    records = paired_records(d0, d1)
    all_stats = {}
    for d in DEFECT_ORDER:
        if counts.get(d, 0):
            all_stats[d] = group_stats(records, d)

    print("\n[2] 分 defect_type 的 paired delta 统计")
    header = ["defect", "n", "mean_a0", "mean_a1", "mean_delta", "median_delta", "std_delta", "min", "max", "down%", "up%"]
    print("  " + " | ".join(f"{h:>11}" for h in header))
    for d in DEFECT_ORDER:
        if d not in all_stats:
            continue
        s = all_stats[d]
        row = [
            d, s["count"], s["mean_score_alpha0"], s["mean_score_alpha1"],
            s["mean_delta"], s["median_delta"], s["std_delta"],
            s["min_delta"], s["max_delta"],
            s["prop_down"], s["prop_up"],
        ]
        print("  " + " | ".join(f"{v:>11}" if not isinstance(v, str) else f"{v:>11}" for v in row))

    # ---- 3. Wilcoxon ----
    print("\n[3] Wilcoxon signed-rank (探索性)")
    for d in DEFECT_ORDER:
        if d not in all_stats:
            continue
        s = all_stats[d]
        w = s["wilcoxon"] or {}
        if "error" in w:
            print(f"  {d}: Wilcoxon 不适用 ({w['error']})")
        else:
            print(
                f"  {d}: n={s['count']}, median_delta={s['median_delta']:.4f}, "
                f"p={w.get('pvalue', float('nan')):.4f}, "
                f"rank_biserial_r={s['rank_biserial_r']}"
            )

    # ---- 4. 保存 CSV + JSON ----
    with open(OUT / "paired_scores.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["image_path", "defect_type", "gt_label",
                        "score_alpha0", "score_alpha1", "delta_score"],
        )
        writer.writeheader()
        for r in records:
            writer.writerow(r)

    stats_payload = {
        "paired": True,
        "n_test_images": len(records),
        "counts": dict(counts),
        "defects": all_stats,
    }
    with open(OUT / "stats_summary.json", "w", encoding="utf-8") as f:
        json.dump(stats_payload, f, indent=2, ensure_ascii=False)

    # ---- 5. 典型样本 ----
    print("\n[4] 最典型样本（每类 delta 最负/最正 top5）")
    typical = {}
    for d in DEFECT_ORDER:
        if d not in all_stats:
            continue
        neg, pos = top_samples(records, d, k=5)
        typical[d] = {"most_negative": neg, "most_positive": pos}
        print(f"\n  [{d}] 下降最多 (delta 最负):")
        for r in neg:
            print(f"    {Path(r['image_path']).name:>8}  a0={r['score_alpha0']:.4f}  a1={r['score_alpha1']:.4f}  delta={r['delta_score']:+.4f}")
        print(f"  [{d}] 上升最多 (delta 最正):")
        for r in pos:
            print(f"    {Path(r['image_path']).name:>8}  a0={r['score_alpha0']:.4f}  a1={r['score_alpha1']:.4f}  delta={r['delta_score']:+.4f}")

    # ---- 6. 图表 ----
    fig1_mean_score(all_stats, OUT / "fig1_mean_score.png")
    fig2_delta_boxplot(records, OUT / "fig2_delta_boxplot.png")
    fig3_paired_slope(records, OUT / "fig3_paired_slope.png")

    print("\n[5] 输出文件:")
    for p in sorted(OUT.iterdir()):
        print(f"  {p.relative_to(OUT.parent.parent.parent)}")


if __name__ == "__main__":
    main()
