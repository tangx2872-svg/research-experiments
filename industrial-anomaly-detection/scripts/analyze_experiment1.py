"""Experiment 1 结果分析：Synthetic Illumination × α-IN Mechanism Screening。

只读取 results/experiment1_illumination_tradeoff/raw_results.csv，不重新训练。

回答任务第 16 节的六个问题：
  Q1 synthetic photometric shift 是否让 normal anomaly score 上升？
  Q2 α 增大后 photometric sensitivity 是否下降？
  Q3 α 增大后真实 defect sensitivity 是否下降？
  Q4 broken_large / broken_small / contamination 趋势是否不同？
  Q5 是否出现 robustness↑ + defect sensitivity↓ 的稳定趋势？
  Q6 趋势是否由极少数样本造成？

输出（results/experiment1_illumination_tradeoff/）：
  summary_results.csv    分 defect_type / illumination condition × alpha 汇总
  figures/fig1_illumination_robustness.png
  figures/fig2_defect_sensitivity.png
  figures/fig3_tradeoff.png（如数据允许）
  figures/fig4_sample_trajectories.png
  analysis_summary.json   Q1-Q6 的定量回答依据
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

RESULTS_ROOT = Path(__file__).resolve().parents[1] / "results" / "experiment1_illumination_tradeoff"
RAW_CSV = RESULTS_ROOT / "raw_results.csv"
FIGURES_DIR = RESULTS_ROOT / "figures"

ALPHAS = [0.0, 0.25, 0.5, 0.75, 1.0]
DEFECT_TYPES = ["broken_large", "broken_small", "contamination"]

# illumination condition 展示顺序
COND_ORDER = [
    ("original", "identity", 1.0),
    ("brightness_0.7", "brightness", 0.7),
    ("brightness_1.3", "brightness", 1.3),
    ("gamma_0.7", "gamma", 0.7),
    ("gamma_1.3", "gamma", 1.3),
]


def load_raw() -> list[dict]:
    rows = []
    with open(RAW_CSV, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            r["alpha"] = float(r["alpha"])
            r["pred_score"] = float(r["pred_score"])
            r["illumination_level"] = float(r["illumination_level"])
            r["is_anomaly"] = int(r["is_anomaly"])
            rows.append(r)
    return rows


def cond_key(r: dict) -> str:
    """返回可读的 illumination condition 名。"""
    if r["illumination_type"] == "identity":
        return "original"
    return f"{r['illumination_type']}_{r['illumination_level']:g}"


# ---------------------------------------------------------------------------
# 汇总
# ---------------------------------------------------------------------------
def build_summary(rows: list[dict]) -> list[dict]:
    # normal side: 按 alpha × condition 汇总
    summary = []
    normal = [r for r in rows if r["gt_type"] == "good"]
    for alpha in ALPHAS:
        for cond_name, _, _ in COND_ORDER:
            scores = [
                r["pred_score"]
                for r in normal
                if r["alpha"] == alpha and cond_key(r) == cond_name
            ]
            if not scores:
                continue
            arr = np.array(scores)
            summary.append(
                {
                    "side": "normal",
                    "alpha": alpha,
                    "condition": cond_name,
                    "n": len(arr),
                    "mean_score": float(arr.mean()),
                    "median_score": float(np.median(arr)),
                    "std_score": float(arr.std(ddof=1)) if len(arr) > 1 else 0.0,
                    "min_score": float(arr.min()),
                    "max_score": float(arr.max()),
                }
            )

    # defect side: 按 alpha × defect_type 汇总
    defect = [r for r in rows if r["gt_type"] in DEFECT_TYPES]
    for alpha in ALPHAS:
        for dt in DEFECT_TYPES:
            scores = [r["pred_score"] for r in defect if r["alpha"] == alpha and r["gt_type"] == dt]
            arr = np.array(scores)
            summary.append(
                {
                    "side": "defect",
                    "alpha": alpha,
                    "condition": dt,
                    "n": len(arr),
                    "mean_score": float(arr.mean()),
                    "median_score": float(np.median(arr)),
                    "std_score": float(arr.std(ddof=1)) if len(arr) > 1 else 0.0,
                    "min_score": float(arr.min()),
                    "max_score": float(arr.max()),
                }
            )
    return summary


# ---------------------------------------------------------------------------
# 关键指标
# ---------------------------------------------------------------------------
def normal_delta_vs_original(rows: list[dict]) -> dict:
    """每张 normal 图从 original 到各 perturbation 的 score change。

    返回 {condition: {alpha: mean_delta}}，其中 delta = score_perturbed - score_original。
    """
    normal = [r for r in rows if r["gt_type"] == "good"]
    # 按 (image_path, alpha) 索引 original score
    orig = {}
    for r in normal:
        if cond_key(r) == "original":
            orig[(r["image_path"], r["alpha"])] = r["pred_score"]

    result: dict[str, dict[float, list[float]]] = defaultdict(lambda: defaultdict(list))
    for r in normal:
        ck = cond_key(r)
        if ck == "original":
            continue
        key = (r["image_path"], r["alpha"])
        if key in orig:
            delta = r["pred_score"] - orig[key]
            result[ck][r["alpha"]].append(delta)
    # 转成 mean
    out: dict[str, dict[float, float]] = {}
    for ck, by_alpha in result.items():
        out[ck] = {a: float(np.mean(v)) for a, v in by_alpha.items()}
    return out


def defect_recall(rows: list[dict], threshold: float) -> dict:
    """每个 alpha 下各 defect type 的 recall（score >= threshold 判为异常）。"""
    defect = [r for r in rows if r["gt_type"] in DEFECT_TYPES]
    out: dict[str, dict[float, float]] = {}
    for dt in DEFECT_TYPES:
        out[dt] = {}
        for alpha in ALPHAS:
            scores = [r["pred_score"] for r in defect if r["alpha"] == alpha and r["gt_type"] == dt]
            if not scores:
                continue
            detected = sum(1 for s in scores if s >= threshold)
            out[dt][alpha] = detected / len(scores)
    return out


# ---------------------------------------------------------------------------
# 图
# ---------------------------------------------------------------------------
def fig1_illumination_robustness(rows: list[dict], out: Path) -> None:
    """α vs synthetic illumination robustness：normal 图 score change（相对 original）。"""
    delta = normal_delta_vs_original(rows)
    fig, ax = plt.subplots(figsize=(7, 5))
    colors = {"brightness_0.7": "#4C72B0", "brightness_1.3": "#55A868",
              "gamma_0.7": "#C44E52", "gamma_1.3": "#8172B3"}
    for ck in ["brightness_0.7", "brightness_1.3", "gamma_0.7", "gamma_1.3"]:
        if ck not in delta:
            continue
        xs = sorted(delta[ck].keys())
        ys = [delta[ck][x] for x in xs]
        ax.plot(xs, ys, marker="o", label=ck, color=colors.get(ck))
    ax.axhline(0, color="black", lw=0.8, ls="--")
    ax.set_xlabel("alpha")
    ax.set_ylabel("mean score change vs original (normal images)")
    ax.set_title("Fig1: alpha vs synthetic illumination robustness")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)


def fig2_defect_sensitivity(rows: list[dict], out: Path) -> None:
    """α vs defect sensitivity：各 defect type 的 mean anomaly score。"""
    defect = [r for r in rows if r["gt_type"] in DEFECT_TYPES]
    colors = {"broken_large": "#DD8452", "broken_small": "#55A868", "contamination": "#C44E52"}
    fig, ax = plt.subplots(figsize=(7, 5))
    for dt in DEFECT_TYPES:
        xs, ys = [], []
        for alpha in ALPHAS:
            scores = [r["pred_score"] for r in defect if r["alpha"] == alpha and r["gt_type"] == dt]
            if scores:
                xs.append(alpha)
                ys.append(float(np.mean(scores)))
        ax.plot(xs, ys, marker="o", label=dt, color=colors[dt])
    ax.set_xlabel("alpha")
    ax.set_ylabel("mean anomaly score (real defect, original)")
    ax.set_title("Fig2: alpha vs defect sensitivity")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)


def fig3_tradeoff(rows: list[dict], out: Path) -> None:
    """Robustness–Sensitivity trade-off：每个 alpha 一个点。

    x = illumination robustness（负的 normal perturbation delta 均值，越大越 robust）
    y = defect sensitivity（负的 defect mean score 相对 alpha=0 的下降，越大越敏感）
    """
    delta = normal_delta_vs_original(rows)
    defect = [r for r in rows if r["gt_type"] in DEFECT_TYPES]

    # x: 所有 perturbation 的 delta 均值（越小越 robust），取负号
    x_vals, y_vals, labels = [], [], []
    for alpha in ALPHAS:
        ds = [delta[ck][alpha] for ck in delta if alpha in delta[ck]]
        if not ds:
            continue
        x = -np.mean(ds)  # 越大 = 越 robust（perturbation 带来更小 score 上升）
        # y: defect 相对 alpha=0 的 mean score 变化，取负号后越大=越敏感
        ref = None
        cur = []
        for r in defect:
            if r["alpha"] == 0.0:
                pass
        # 计算每个 alpha 的 defect 总 mean，相对 alpha=0
        def mean_at(a):
            s = [r["pred_score"] for r in defect if r["alpha"] == a]
            return float(np.mean(s)) if s else None
        m0 = mean_at(0.0)
        ma = mean_at(alpha)
        if m0 is None or ma is None:
            continue
        y = - (ma - m0)  # 越大 = 越敏感（score 下降越多）
        x_vals.append(x)
        y_vals.append(y)
        labels.append(f"a={alpha:g}")

    if len(x_vals) < 2:
        print("  [skip] fig3_tradeoff 数据不足")
        return

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    ax.scatter(x_vals, y_vals, s=80)
    for x, y, lab in zip(x_vals, y_vals, labels):
        ax.annotate(lab, (x, y), textcoords="offset points", xytext=(6, 4))
    ax.set_xlabel("illumination robustness ↑ (neg. normal perturbation delta)")
    ax.set_ylabel("defect sensitivity ↑ (neg. defect score drop)")
    ax.set_title("Fig3: robustness–sensitivity trade-off")
    ax.axhline(0, color="gray", lw=0.6, ls="--")
    ax.axvline(0, color="gray", lw=0.6, ls="--")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)


def fig4_sample_trajectories(rows: list[dict], out: Path) -> None:
    """individual sample trajectory：每张 normal / defect 图 alpha 0→1 的 score 变化。"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # normal（original condition）
    normal = [r for r in rows if r["gt_type"] == "good" and cond_key(r) == "original"]
    ax = axes[0]
    for path in sorted({r["image_path"] for r in normal}):
        pts = sorted([r for r in normal if r["image_path"] == path], key=lambda r: r["alpha"])
        xs = [p["alpha"] for p in pts]
        ys = [p["pred_score"] for p in pts]
        ax.plot(xs, ys, marker=".", alpha=0.5, lw=1)
    ax.set_xlabel("alpha")
    ax.set_ylabel("anomaly score")
    ax.set_title("normal images (original) trajectory")
    ax.grid(alpha=0.3)

    # defect
    defect = [r for r in rows if r["gt_type"] in DEFECT_TYPES]
    ax = axes[1]
    colors = {"broken_large": "#DD8452", "broken_small": "#55A868", "contamination": "#C44E52"}
    for dt in DEFECT_TYPES:
        for path in sorted({r["image_path"] for r in defect if r["gt_type"] == dt}):
            pts = sorted([r for r in defect if r["image_path"] == path], key=lambda r: r["alpha"])
            xs = [p["alpha"] for p in pts]
            ys = [p["pred_score"] for p in pts]
            ax.plot(xs, ys, marker=".", alpha=0.35, lw=1, color=colors[dt])
    ax.set_xlabel("alpha")
    ax.set_ylabel("anomaly score")
    ax.set_title("defect images trajectory (color by type)")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Q1-Q6 定量回答
# ---------------------------------------------------------------------------
def answer_questions(rows: list[dict], threshold: float) -> dict:
    normal = [r for r in rows if r["gt_type"] == "good"]
    defect = [r for r in rows if r["gt_type"] in DEFECT_TYPES]

    # Q1: 各 alpha 下 normal original vs perturbation 的 mean score
    q1 = {}
    for alpha in ALPHAS:
        orig = np.mean([r["pred_score"] for r in normal if r["alpha"] == alpha and cond_key(r) == "original"])
        per_cond = {}
        for cond_name, _, _ in COND_ORDER:
            if cond_name == "original":
                continue
            s = [r["pred_score"] for r in normal if r["alpha"] == alpha and cond_key(r) == cond_name]
            per_cond[cond_name] = float(np.mean(s)) if s else None
        q1[f"alpha_{alpha:g}"] = {"original_mean": float(orig), "perturbed_means": per_cond}

    # Q2: normal perturbation delta 随 alpha
    delta = normal_delta_vs_original(rows)
    q2 = {ck: {f"a{a:g}": v for a, v in bya.items()} for ck, bya in delta.items()}

    # Q3/Q4: defect mean score 随 alpha，分 type
    q3q4 = {}
    for dt in DEFECT_TYPES:
        q3q4[dt] = {}
        for alpha in ALPHAS:
            s = [r["pred_score"] for r in defect if r["alpha"] == alpha and r["gt_type"] == dt]
            q3q4[dt][f"a{alpha:g}"] = float(np.mean(s)) if s else None

    # recall（用 alpha=0 normal max 作为阈值）
    recall = defect_recall(rows, threshold)

    return {
        "threshold": threshold,
        "Q1_normal_perturbed_means": q1,
        "Q2_normal_delta_vs_original": q2,
        "Q3_Q4_defect_mean_scores": q3q4,
        "defect_recall_by_alpha": recall,
    }


def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    rows = load_raw()
    print(f"总记录数: {len(rows)}")

    # 阈值：alpha=0 normal original max
    normal_a0_orig = [
        r["pred_score"]
        for r in rows
        if r["gt_type"] == "good" and r["alpha"] == 0.0 and cond_key(r) == "original"
    ]
    threshold = float(max(normal_a0_orig))
    print(f"阈值(alpha=0 normal original max) = {threshold:.6f}")

    # 汇总
    summary = build_summary(rows)
    with open(RESULTS_ROOT / "summary_results.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["side", "alpha", "condition", "n", "mean_score",
                           "median_score", "std_score", "min_score", "max_score"]
        )
        writer.writeheader()
        writer.writerows(summary)

    # 图
    fig1_illumination_robustness(rows, FIGURES_DIR / "fig1_illumination_robustness.png")
    fig2_defect_sensitivity(rows, FIGURES_DIR / "fig2_defect_sensitivity.png")
    fig3_tradeoff(rows, FIGURES_DIR / "fig3_tradeoff.png")
    fig4_sample_trajectories(rows, FIGURES_DIR / "fig4_sample_trajectories.png")

    # Q1-Q6
    answers = answer_questions(rows, threshold)
    with open(RESULTS_ROOT / "analysis_summary.json", "w", encoding="utf-8") as f:
        json.dump(answers, f, indent=2, ensure_ascii=False)

    # 打印关键结论
    print("\n=== Q1: normal original vs perturbation mean score (per alpha) ===")
    for a, d in answers["Q1_normal_perturbed_means"].items():
        print(f"  {a}: original={d['original_mean']:.3f}, " +
              ", ".join(f"{k}={v:.3f}" for k, v in d["perturbed_means"].items() if v is not None))

    print("\n=== Q2: normal delta vs original (per condition, per alpha) ===")
    for ck, bya in answers["Q2_normal_delta_vs_original"].items():
        print(f"  {ck}: " + ", ".join(f"{a}={v:+.3f}" for a, v in bya.items()))

    print("\n=== Q3/Q4: defect mean score (per type, per alpha) ===")
    for dt, bya in answers["Q3_Q4_defect_mean_scores"].items():
        print(f"  {dt}: " + ", ".join(f"{a}={v:.3f}" for a, v in bya.items()))

    print("\n=== defect recall (threshold = alpha0 normal max) ===")
    for dt, bya in answers["defect_recall_by_alpha"].items():
        print(f"  {dt}: " + ", ".join(f"{a}={v:.2%}" for a, v in bya.items()))

    print(f"\n输出目录: {RESULTS_ROOT}")


if __name__ == "__main__":
    main()
