#!/usr/bin/env python
"""E21 figures (Round-1 results only; HOLD => Fig5 (3-seed) is NOT produced).

Data source: results/e21/analysis_cells.csv, analysis_category_pooled.csv,
analysis_view_pooled.csv, analysis_overall.csv (single source of truth = analysis CSVs).

Outputs (PNG + PDF): results/e21/figures/
  fig1_auroc_by_view.*      F0 vs R1-ILA AUROC for each category x view
  fig2_dauroc_heatmap.*     category x view dAUROC heatmap
  fig3_dauroc_by_view.*     per-view dAUROC points (3 categories) + mean line
  fig4_pooled_auroc.*       4-view pooled F0 vs R1-ILA (per category + overall macro)
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUTDIR = ROOT / "results" / "e21"
FIGDIR = OUTDIR / "figures"
VIEWS = ["090", "120", "240", "330"]
CATS = ["Bird", "Car", "Cube"]
C_F0, C_R1 = "#4C72B0", "#DD8452"
C_CAT = {"Bird": "#4C72B0", "Car": "#DD8452", "Cube": "#55A868"}


def load_csv(name):
    with open(OUTDIR / name) as f:
        return list(csv.DictReader(f))


def fnum(x):
    return float(x) if x not in ("", None) else float("nan")


def save(fig, name):
    FIGDIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGDIR / f"{name}.png", dpi=200, bbox_inches="tight")
    fig.savefig(FIGDIR / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote figures/{name}.png + .pdf")


def main():
    cells = load_csv("analysis_cells.csv")
    cat_pool = load_csv("analysis_category_pooled.csv")
    overall = load_csv("analysis_overall.csv")
    seed = 0
    cells = [r for r in cells if int(r["seed"]) == seed]
    cat_pool = [r for r in cat_pool if int(r["seed"]) == seed]
    overall = [r for r in overall if int(r["seed"]) == seed]
    ov = overall[0] if overall else None

    # ---- Fig 1: F0 vs R1 AUROC, 3 cats x 4 views
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), sharey=True)
    for ax, cat in zip(axes, CATS):
        rows = {r["view"]: r for r in cells if r["category"] == cat}
        x = np.arange(len(VIEWS))
        f0 = [fnum(rows[v]["auroc_f0"]) for v in VIEWS]
        r1 = [fnum(rows[v]["auroc_r1"]) for v in VIEWS]
        ax.bar(x - 0.19, f0, 0.36, label="F0 (frozen)", color=C_F0)
        ax.bar(x + 0.19, r1, 0.36, label="R1-ILA", color=C_R1)
        for xi, (a, b) in enumerate(zip(f0, r1)):
            ax.text(xi + 0.19, b + 0.004, f"{b - a:+.3f}", ha="center", fontsize=7.5)
        ax.set_xticks(x)
        ax.set_xticklabels(VIEWS)
        ax.set_title(cat)
        ax.grid(axis="y", alpha=0.3)
        ax.set_ylim(0.6, min(1.0, max(f0 + r1) + 0.07))
    axes[0].set_ylabel("AUROC")
    axes[0].legend(loc="lower right", fontsize=8)
    fig.suptitle("E21 · F0 vs R1-ILA AUROC across M²AD views (seed0)", y=1.02)
    save(fig, "fig1_auroc_by_view")

    # ---- Fig 2: heatmap dAUROC
    mat = np.full((3, 4), np.nan)
    for r in cells:
        mat[CATS.index(r["category"]), VIEWS.index(r["view"])] = fnum(r["d_auroc"])
    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    lim = max(0.03, float(np.nanmax(np.abs(mat))))
    im = ax.imshow(mat, cmap="RdBu_r", vmin=-lim, vmax=lim, aspect="auto")
    for i in range(3):
        for j in range(4):
            ax.text(j, i, f"{mat[i, j]:+.4f}", ha="center", va="center", fontsize=9)
    ax.set_xticks(range(4)); ax.set_xticklabels(VIEWS)
    ax.set_yticks(range(3)); ax.set_yticklabels(CATS)
    ax.set_xlabel("view"); ax.set_ylabel("category")
    ax.set_title("E21 · ΔAUROC (R1-ILA − F0), seed0")
    fig.colorbar(im, ax=ax, label="ΔAUROC")
    save(fig, "fig2_dauroc_heatmap")

    # ---- Fig 3: per-view dAUROC points + mean
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    for cat in CATS:
        ys = [fnum(r["d_auroc"]) for v in VIEWS for r in cells if r["category"] == cat and r["view"] == v]
        ax.scatter(range(4), ys, label=cat, color=C_CAT[cat], s=42, zorder=3)
    means = [np.mean([fnum(r["d_auroc"]) for r in cells if r["view"] == v]) for v in VIEWS]
    ax.plot(range(4), means, color="black", marker="D", lw=1.4, ms=5, label="3-cat mean")
    ax.axhline(0, color="gray", lw=1, ls="--")
    ax.set_xticks(range(4)); ax.set_xticklabels(VIEWS)
    ax.set_xlabel("view"); ax.set_ylabel("ΔAUROC")
    ax.set_title("E21 · ΔAUROC by view (seed0)")
    ax.grid(alpha=0.3); ax.legend(fontsize=8)
    save(fig, "fig3_dauroc_by_view")

    # ---- Fig 4: 4-view pooled F0 vs R1
    x = np.arange(3 + (1 if ov else 0))
    f0 = [fnum(r["pooled_auroc_f0"]) for r in cat_pool]
    r1 = [fnum(r["pooled_auroc_r1"]) for r in cat_pool]
    labs = list(CATS)
    if ov:
        f0.append(float(np.mean(f0))); r1.append(float(np.mean(r1))); labs.append("Overall(macro)")
    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    ax.bar(x - 0.19, f0, 0.36, label="F0 (frozen)", color=C_F0)
    ax.bar(x + 0.19, r1, 0.36, label="R1-ILA", color=C_R1)
    for xi, (a, b) in enumerate(zip(f0, r1)):
        ax.text(xi + 0.19, b + 0.003, f"{b - a:+.4f}", ha="center", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(labs)
    ax.set_ylabel("4-view pooled AUROC")
    ax.set_title("E21 · 4-view pooled AUROC (seed0)")
    ax.grid(axis="y", alpha=0.3); ax.legend(loc="lower right", fontsize=8)
    ax.set_ylim(0.6, min(1.0, max(f0 + r1) + 0.06))
    save(fig, "fig4_pooled_auroc")

    print(f"[E21 figures] done. HOLD => fig5 (3-seed) intentionally NOT produced.")


if __name__ == "__main__":
    main()
