"""Experiment 1E Phase 14-15：Figures + Cross-Category Area Control。

依赖 experiment1e_analysis.py 生成的 analysis/ 目录（signatures/summary/trajectory）。

生成：
  Figure 1: 每 category 的 mean_gap/defect_std/mean_z/d' vs α
  Figure 2: Defect Response Signature Heatmap（raw + column-standardized）
  Figure 3: Mean–Variance Response Map（X=Δmean_gap, Y=Δdefect_std, size=median area）
  Figure 4: Area-response plots（log_area vs 4 个 response 维度）

Phase 15 area control:
  Model A: response ~ log(area)
  Model B: response ~ log(area) + category
  Model C: response ~ log(area) + category + defect identity
  对 Δz / Δdefect_std / Δd' 分别报告 R² / adjusted R² / ΔR²。
"""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT_ROOT = PROJECT_ROOT / "results" / "experiment_1e"
ANALYSIS_DIR = OUT_ROOT / "analysis"
FIGURES_DIR = OUT_ROOT / "figures"

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

CATEGORIES = ["bottle", "grid", "cable", "screw", "hazelnut"]
# summary CSV 里 response 维度列（跨 seed 均值）
DIMS = ["delta_mean_gap_mean", "delta_defect_std_mean", "delta_z_mean", "delta_dprime_mean"]
# 中国股市红涨绿跌惯例不适用，此处用科学配色（连续 colormap）
CATEGORY_COLORS = {
    "bottle": "#e74c3c", "grid": "#3498db", "cable": "#2ecc71",
    "screw": "#f39c12", "hazelnut": "#9b59b6",
}


def load_summary() -> list[dict]:
    p = ANALYSIS_DIR / "defect_response_signatures_seed_summary.csv"
    rows = list(csv.DictReader(open(p, encoding="utf-8")))
    for r in rows:
        for k in list(r.keys()):
            if k not in ("category", "defect_type"):
                try:
                    r[k] = float(r[k])
                except (ValueError, TypeError):
                    pass
    return rows


def load_trajectory() -> list[dict]:
    p = ANALYSIS_DIR / "alpha_trajectory.csv"
    rows = list(csv.DictReader(open(p, encoding="utf-8")))
    for r in rows:
        r["alpha"] = float(r["alpha"])
        r["seed"] = int(r["seed"])
        for k in ["mean_gap", "defect_std", "mean_z", "d_prime"]:
            r[k] = float(r[k])
    return rows


def figure1(summary_rows, traj_rows) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    traj_metric_map = {"mean_gap": "mean_gap", "defect_std": "defect_std", "mean_z": "mean_z", "d_prime": "d_prime"}
    fig, axes = plt.subplots(len(CATEGORIES), 4, figsize=(20, 3.2 * len(CATEGORIES)), sharex=True)
    alphas = [0.0, 0.25, 0.5, 0.75, 1.0]
    for ci, cat in enumerate(CATEGORIES):
        cat_traj = [t for t in traj_rows if t["category"] == cat]
        defect_types = sorted(set(t["defect_type"] for t in cat_traj))
        for di, (label, key) in enumerate(traj_metric_map.items()):
            ax = axes[ci, di]
            for dt in defect_types:
                dt_traj = [t for t in cat_traj if t["defect_type"] == dt]
                # 3 seeds 平均
                ys = []
                for a in alphas:
                    vals = [t[key] for t in dt_traj if t["alpha"] == a]
                    ys.append(np.mean(vals) if vals else np.nan)
                ax.plot(alphas, ys, marker="o", ms=3, lw=1.2, label=dt)
            ax.set_title(f"{cat} — {label}", fontsize=9)
            ax.grid(alpha=0.3)
            if di == 0:
                ax.set_ylabel(cat, fontsize=10, rotation=0, labelpad=25)
            if ci == len(CATEGORIES) - 1:
                ax.set_xlabel("α")
            if ci == 0 and di == 3:
                ax.legend(fontsize=6, ncol=1, loc="best")
    fig.suptitle("Experiment 1E Figure 1: Full α Trajectory (mean over 3 seeds)", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    fig.savefig(FIGURES_DIR / "figure1_alpha_trajectories.png", dpi=130)
    plt.close(fig)
    print("[fig1] saved")


def figure2(summary_rows) -> None:
    # Heatmap: rows=25 defect types, cols=4 dims
    all_dt = [f"{r['category']}/{r['defect_type']}" for r in summary_rows]
    dims_disp = ["Δmean_gap", "Δdefect_std", "Δz", "Δd'"]
    dims_key = DIMS
    mat = np.array([[r[d] for d in dims_key] for r in summary_rows], dtype=float)

    fig, axes = plt.subplots(1, 2, figsize=(14, 0.35 * len(all_dt) + 2))
    # raw
    vmax = np.nanmax(np.abs(mat))
    im0 = axes[0].imshow(mat, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    axes[0].set_xticks(range(len(dims_disp))); axes[0].set_xticklabels(dims_disp, fontsize=8)
    axes[0].set_yticks(range(len(all_dt))); axes[0].set_yticklabels(all_dt, fontsize=6)
    axes[0].set_title("Raw signature")
    fig.colorbar(im0, ax=axes[0])
    # column-standardized
    mat_std = (mat - np.nanmean(mat, axis=0)) / (np.nanstd(mat, axis=0) + 1e-9)
    vmax2 = np.nanmax(np.abs(mat_std))
    im1 = axes[1].imshow(mat_std, aspect="auto", cmap="RdBu_r", vmin=-vmax2, vmax=vmax2)
    axes[1].set_xticks(range(len(dims_disp))); axes[1].set_xticklabels(dims_disp, fontsize=8)
    axes[1].set_yticks(range(len(all_dt))); axes[1].set_yticklabels(all_dt, fontsize=6)
    axes[1].set_title("Column-standardized")
    fig.colorbar(im1, ax=axes[1])
    fig.suptitle("Experiment 1E Figure 2: Defect Response Signature Heatmap", fontsize=13)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "figure2_response_heatmap.png", dpi=130)
    plt.close(fig)
    print("[fig2] saved")


def figure3(summary_rows) -> None:
    fig, ax = plt.subplots(figsize=(10, 8))
    for r in summary_rows:
        cat = r["category"]
        x = r["delta_mean_gap_mean"]
        y = r["delta_defect_std_mean"]
        size = max(10, r["median_area_ratio"] * 1500)
        ax.scatter(x, y, s=size, c=CATEGORY_COLORS[cat], alpha=0.75,
                   edgecolors="black", linewidths=0.4, label=cat)
        ax.annotate(f"{r['defect_type']}", (x, y), fontsize=6,
                    textcoords="offset points", xytext=(3, 3), alpha=0.8)
    # 去重 legend
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), fontsize=8)
    ax.axhline(0, color="gray", lw=0.5, ls="--")
    ax.axvline(0, color="gray", lw=0.5, ls="--")
    ax.set_xlabel("Δmean_gap (α=1 − α=0)")
    ax.set_ylabel("Δdefect_std (α=1 − α=0)")
    ax.set_title("Experiment 1E Figure 3: Mean–Variance Response Map\n(point size ∝ median area ratio)")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "figure3_mean_variance_map.png", dpi=130)
    plt.close(fig)
    print("[fig3] saved")


def figure4(summary_rows) -> None:
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    dim_labels = {"delta_mean_gap_mean": "Δmean_gap", "delta_defect_std_mean": "Δdefect_std",
                  "delta_z_mean": "Δz", "delta_dprime_mean": "Δd'"}
    for di, dim in enumerate(DIMS):
        ax = axes[di]
        for r in summary_rows:
            if math.isnan(r["median_area_ratio"]) or r["median_area_ratio"] <= 0:
                continue
            x = math.log(r["median_area_ratio"])
            ax.scatter(x, r[dim], c=CATEGORY_COLORS[r["category"]], alpha=0.75,
                       s=35, edgecolors="black", linewidths=0.3)
        ax.set_xlabel("log(area_ratio)")
        ax.set_ylabel(dim_labels[dim])
        ax.set_title(dim_labels[dim])
        ax.grid(alpha=0.3)
    handles = [plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=CATEGORY_COLORS[c], markersize=8, label=c) for c in CATEGORIES]
    fig.legend(handles=handles, loc="lower center", ncol=5, fontsize=8)
    fig.suptitle("Experiment 1E Figure 4: Area–Response Plots", fontsize=13)
    fig.tight_layout(rect=[0, 0.05, 1, 0.95])
    fig.savefig(FIGURES_DIR / "figure4_area_response.png", dpi=130)
    plt.close(fig)
    print("[fig4] saved")


def ols(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, float, float]:
    """普通最小二乘，返回 (coef, r2, adj_r2)。X 含截距列。"""
    n, p = X.shape
    beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    yhat = X @ beta
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p) if n > p else float("nan")
    return beta, r2, adj_r2


def area_control(summary_rows) -> None:
    """Phase 15：Model A/B/C 对 Δz / Δdefect_std / Δd' 分别报告 R²/adjR²/ΔR²。"""
    results = {}
    # 构建设计矩阵
    cats = sorted(set(r["category"] for r in summary_rows))
    cat2idx = {c: i for i, c in enumerate(cats)}
    # defect identity：category x defect_type 唯一 id
    def_ids = sorted(set((r["category"], r["defect_type"]) for r in summary_rows))
    def2idx = {d: i for i, d in enumerate(def_ids)}

    n = len(summary_rows)
    log_area = np.array([math.log(r["median_area_ratio"]) if r["median_area_ratio"] > 0 else 0.0 for r in summary_rows])

    for dim in ["delta_z_mean", "delta_defect_std_mean", "delta_dprime_mean"]:
        y = np.array([r[dim] for r in summary_rows])
        # Model A: response ~ log(area)  -> 截距 + log_area
        XA = np.column_stack([np.ones(n), log_area])
        # Model B: + category（one-hot，drop first）
        cat_onehot = np.zeros((n, len(cats) - 1))
        for i, r in enumerate(summary_rows):
            ci = cat2idx[r["category"]]
            if ci > 0:
                cat_onehot[i, ci - 1] = 1.0
        XB = np.column_stack([XA, cat_onehot])
        # Model C: + defect identity（one-hot，drop first）
        def_onehot = np.zeros((n, len(def_ids) - 1))
        for i, r in enumerate(summary_rows):
            di = def2idx[(r["category"], r["defect_type"])]
            if di > 0:
                def_onehot[i, di - 1] = 1.0
        XC = np.column_stack([XB, def_onehot])

        _, r2a, adj_a = ols(XA, y)
        _, r2b, adj_b = ols(XB, y)
        _, r2c, adj_c = ols(XC, y)

        results[dim] = {
            "Model_A_r2": r2a, "Model_A_adj_r2": adj_a,
            "Model_B_r2": r2b, "Model_B_adj_r2": adj_b,
            "Model_C_r2": r2c, "Model_C_adj_r2": adj_c,
            "delta_r2_area_to_B": r2b - r2a,
            "delta_r2_B_to_C": r2c - r2b,
        }
        print(f"[area-control] {dim}: R² A={r2a:.4f} B={r2b:.4f} C={r2c:.4f} | "
              f"ΔR²(A→B)={r2b-r2a:.4f} ΔR²(B→C)={r2c-r2b:.4f}")

    with open(ANALYSIS_DIR / "area_control_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)


def main() -> None:
    summary_rows = load_summary()
    traj_rows = load_trajectory()
    print(f"summary={len(summary_rows)} trajectory={len(traj_rows)}")
    figure1(summary_rows, traj_rows)
    figure2(summary_rows)
    figure3(summary_rows)
    figure4(summary_rows)
    area_control(summary_rows)
    print(f"[done] 输出目录 = {FIGURES_DIR} / {ANALYSIS_DIR}")


if __name__ == "__main__":
    main()
