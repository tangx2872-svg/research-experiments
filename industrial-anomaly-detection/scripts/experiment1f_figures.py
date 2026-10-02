"""
Experiment 1F — 核心图
Figure 1: 11 attributes × Δdefect_std 散点矩阵（category 区分标记）
Figure 2: 4D response PCA 地图（点=defect type，标注 shrink family）
"""

import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ANALYSIS = ROOT / "results" / "experiment_1f" / "analysis"
FIG_DIR = ROOT / "results" / "experiment_1f" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

CATS = ["bottle", "grid", "cable", "screw", "hazelnut"]
CAT_COLOR = {"bottle": "#1f77b4", "grid": "#d62728", "cable": "#2ca02c",
             "screw": "#9467bd", "hazelnut": "#ff7f0e"}
CAT_MARKER = {"bottle": "o", "grid": "s", "cable": "^", "screw": "D", "hazelnut": "v"}

ATTRS = [
    ("log_area_ratio", "size"),
    ("intensity_contrast", "contrast"),
    ("lab_contrast", "contrast"),
    ("local_variance_ratio", "contrast"),
    ("gradient_mean_defect", "frequency"),
    ("gradient_contrast", "frequency"),
    ("laplacian_energy", "frequency"),
    ("hf_energy_ratio", "frequency"),
    ("compactness", "morphology"),
    ("elongation", "morphology"),
]

PRIMARY = "delta_defect_std"


def fig1(rows: list[dict]) -> None:
    n = len(ATTRS)
    ncols = 4
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 3.6 * nrows))
    axes = axes.ravel()
    y = np.array([float(r[PRIMARY]) for r in rows])

    for i, (a, fam) in enumerate(ATTRS):
        ax = axes[i]
        x = np.array([float(r[f"{a}_median"]) for r in rows])
        keep = ~(np.isnan(x) | np.isnan(y))
        for r, xv, yv in zip([r for r, k in zip(rows, keep) if k],
                             x[keep], y[keep]):
            c = r["category"]
            ax.scatter(xv, yv, c=CAT_COLOR[c], marker=CAT_MARKER[c], s=55,
                       edgecolors="black", linewidths=0.4, alpha=0.85, zorder=3)
        # 零线
        ax.axhline(0, color="gray", lw=0.8, ls="--", alpha=0.6)
        ax.set_title(f"{a}\n[{fam}]", fontsize=10)
        ax.grid(alpha=0.25)

    for j in range(n, len(axes)):
        axes[j].axis("off")
    # 统一图例
    handles = [plt.Line2D([0], [0], color=CAT_COLOR[c], marker=CAT_MARKER[c], ls="",
                          markeredgecolor="black", markersize=8, label=c) for c in CATS]
    fig.legend(handles=handles, ncol=5, loc="lower center", fontsize=10, frameon=False)
    fig.suptitle("Experiment 1F — Visual Attributes vs Δdefect_std (primary outcome, n=25 defect types)",
                 fontsize=13)
    fig.tight_layout(rect=[0, 0.04, 1, 0.97])
    fig.savefig(FIG_DIR / "figure1_attribute_scatter.png", dpi=150)
    plt.close(fig)
    print("figure1 saved")


def fig2(rows: list[dict]) -> None:
    pca = list(csv.DictReader(open(ANALYSIS / "a5_pca_response.csv", encoding="utf-8")))
    fig, ax = plt.subplots(figsize=(10.5, 8))

    shrink_family = {("bottle", "contamination"), ("cable", "bent_wire"), ("hazelnut", "print")}
    for r in pca:
        c = r["category"]
        key = (r["category"], r["defect_type"])
        x, yv = float(r["pc1"]), float(r["pc2"])
        is_shrink = key in shrink_family
        ax.scatter(x, yv, c=CAT_COLOR[c], marker=CAT_MARKER[c],
                   s=180 if is_shrink else 70,
                   edgecolors="red" if is_shrink else "black",
                   linewidths=2.2 if is_shrink else 0.5,
                   alpha=0.95, zorder=4 if is_shrink else 3)
        ax.annotate(f"{c}/{r['defect_type']}", (x, yv), fontsize=7.5,
                    xytext=(4, 4), textcoords="offset points", alpha=0.85)

    ax.axhline(0, color="gray", lw=0.8, ls="--", alpha=0.6)
    ax.axvline(0, color="gray", lw=0.8, ls="--", alpha=0.6)
    ax.set_xlabel("PC1 (49.7% var) — variance-shrink side →  variance-expand side")
    ax.set_ylabel("PC2 (32.6% var)")
    ax.set_title("Experiment 1F — 4D Response PCA (exploratory)\n"
                 "red halo = variance-shrink family (contamination / bent_wire / print); "
                 "no pre-registered attribute explains this axis")
    handles = [plt.Line2D([0], [0], color=CAT_COLOR[c], marker=CAT_MARKER[c], ls="",
                          markeredgecolor="black", markersize=8, label=c) for c in CATS]
    ax.legend(handles=handles, loc="best", fontsize=9, frameon=True)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "figure2_response_pca.png", dpi=150)
    plt.close(fig)
    print("figure2 saved")


def main() -> None:
    rows = list(csv.DictReader(open(ANALYSIS / "merged_attributes_responses.csv", encoding="utf-8")))
    fig1(rows)
    fig2(rows)


if __name__ == "__main__":
    main()
