#!/usr/bin/env python
"""E13-B2 analysis: Cube bank-sensitivity diagnosis + ceiling-effect check (CPU-only).

Combines Cube seed0-7 (E13-A seed0 + E13-B seed1/2 + E13-B2 seed3-7) and
Bird/Car seed0-2 (E13-B). Stage A verdict on Cube determines the outcome.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "e13b2"
FIGS = OUT / "figures"

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

CATS = ["Bird", "Car", "Cube"]


def load_cells():
    """-> dict[(cat, seed)] = {r0_auroc, r0_ap, r1_auroc, r1_ap, d_auroc, d_ap}"""
    cells = {}
    # seed0 from E13-A
    with open(ROOT / "results" / "e13a" / "e13a_screening.csv") as f:
        for r in csv.DictReader(f):
            if r["config"] not in ("R0", "R1"):
                continue
            cat = r["category"]
            if r["config"] == "R0":
                cells[(cat, 0)] = {"r0_auroc": float(r["auroc"]), "r0_ap": float(r["ap"])}
            else:
                c = cells[(cat, 0)]
                c["r1_auroc"] = float(r["auroc"]); c["r1_ap"] = float(r["ap"])
                c["d_auroc"] = float(r["d_auroc"]); c["d_ap"] = float(r["d_ap"])
    # seed1/2 from E13-B
    with open(ROOT / "results" / "e13b" / "e13b_results.csv") as f:
        for r in csv.DictReader(f):
            cat = r["category"]; s = int(r["seed"])
            cells[(cat, s)] = {"r0_auroc": float(r["r0_auroc"]), "r0_ap": float(r["r0_ap"]),
                               "r1_auroc": float(r["r1_auroc"]), "r1_ap": float(r["r1_ap"]),
                               "d_auroc": float(r["d_auroc"]), "d_ap": float(r["d_ap"])}
    # seed3-7 from E13-B2
    with open(OUT / "e13b2_results.csv") as f:
        for r in csv.DictReader(f):
            cat = r["category"]; s = int(r["seed"])
            cells[(cat, s)] = {"r0_auroc": float(r["r0_auroc"]), "r0_ap": float(r["r0_ap"]),
                               "r1_auroc": float(r["r1_auroc"]), "r1_ap": float(r["r1_ap"]),
                               "d_auroc": float(r["d_auroc"]), "d_ap": float(r["d_ap"])}
    return cells


def cat_stats(cells, cat, seeds):
    d_au = [cells[(cat, s)]["d_auroc"] for s in seeds]
    d_ap = [cells[(cat, s)]["d_ap"] for s in seeds]
    r0 = [cells[(cat, s)]["r0_auroc"] for s in seeds]
    return {"mean_d_auroc": float(np.mean(d_au)), "std_d_auroc": float(np.std(d_au, ddof=1)),
            "min_d_auroc": float(np.min(d_au)), "max_d_auroc": float(np.max(d_au)),
            "mean_d_ap": float(np.mean(d_ap)), "positive_auroc": int(sum(1 for x in d_au if x > 0)),
            "n_seeds": len(seeds), "r0_min": float(np.min(r0)), "r0_max": float(np.max(r0)),
            "seeds_d_auroc": {s: cells[(cat, s)]["d_auroc"] for s in seeds},
            "seeds_r0": {s: cells[(cat, s)]["r0_auroc"] for s in seeds}}


def pearson(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    return float(np.corrcoef(x, y)[0, 1])


def main() -> int:
    FIGS.mkdir(parents=True, exist_ok=True)
    cells = load_cells()

    cube_seeds = [0, 1, 2, 3, 4, 5, 6, 7]
    other_seeds = [0, 1, 2]
    cube = cat_stats(cells, "Cube", cube_seeds)
    bird = cat_stats(cells, "Bird", other_seeds)
    car = cat_stats(cells, "Car", other_seeds)

    # overall (all available cells)
    all_d_au = [cells[(c, s)]["d_auroc"] for c in CATS
                for s in (cube_seeds if c == "Cube" else other_seeds)]
    all_d_ap = [cells[(c, s)]["d_ap"] for c in CATS
                for s in (cube_seeds if c == "Cube" else other_seeds)]

    # correlation: R0 vs dAUROC (within Cube + overall)
    cube_r0 = [cells[("Cube", s)]["r0_auroc"] for s in cube_seeds]
    cube_d = [cells[("Cube", s)]["d_auroc"] for s in cube_seeds]
    cube_pearson = pearson(cube_r0, cube_d)
    cube_spearman = float(spearmanr(cube_r0, cube_d)[0])
    all_r0 = [cells[(c, s)]["r0_auroc"] for c in CATS
              for s in (cube_seeds if c == "Cube" else other_seeds)]
    all_d = all_d_au
    overall_pearson = pearson(all_r0, all_d)

    # Stage A verdict
    if (cube["mean_d_auroc"] >= 0.005 and cube["positive_auroc"] >= 6
            and cube["mean_d_ap"] >= 0):
        stage_a = "STABLE"
    elif cube["mean_d_auroc"] >= 0 and cube["positive_auroc"] >= 4:
        stage_a = "BORDERLINE"
    else:
        stage_a = "UNSTABLE"

    result = {
        "experiment": "E13-B2", "stage_a_verdict": stage_a,
        "cube": cube, "bird": bird, "car": car,
        "overall_mean_d_auroc": float(np.mean(all_d_au)),
        "overall_mean_d_ap": float(np.mean(all_d_ap)),
        "overall_positive_auroc": int(sum(1 for x in all_d_au if x > 0)),
        "overall_n_cells": len(all_d_au),
        "cube_pearson_r0_vs_dauroc": cube_pearson,
        "cube_spearman_r0_vs_dauroc": cube_spearman,
        "overall_pearson_r0_vs_dauroc": overall_pearson,
    }
    (OUT / "e13b2_summary.json").write_text(json.dumps(result, indent=2, default=float))

    # ---- figures
    # fig1 delta by seed
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for c in CATS:
        seeds = cube_seeds if c == "Cube" else other_seeds
        d = [cells[(c, s)]["d_auroc"] for s in seeds]
        ax.plot(seeds, d, "-o", label=c)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xlabel("seed")
    ax.set_ylabel("delta AUROC vs R0")
    ax.set_title("E13-B2 delta AUROC by seed")
    ax.legend()
    fig.tight_layout(); fig.savefig(FIGS / "fig1_delta_by_seed.png", dpi=150); plt.close(fig)

    # fig2 per-category mean +- std
    fig, ax = plt.subplots(figsize=(6, 4.5))
    means = [bird["mean_d_auroc"], car["mean_d_auroc"], cube["mean_d_auroc"]]
    stds = [bird["std_d_auroc"], car["std_d_auroc"], cube["std_d_auroc"]]
    ax.bar(CATS, means, yerr=stds, capsize=5, color=["#2c7fb8", "#d95f02", "#999999"])
    ax.axhline(0, color="black", lw=0.8)
    ax.set_ylabel("mean delta AUROC")
    ax.set_title("E13-B2 per-category mean +/- std")
    fig.tight_layout(); fig.savefig(FIGS / "fig2_per_category_mean_std.png", dpi=150); plt.close(fig)

    # fig3 R0 vs delta scatter
    fig, ax = plt.subplots(figsize=(6.5, 5))
    colors = {"Bird": "#2c7fb8", "Car": "#d95f02", "Cube": "#999999"}
    for c in CATS:
        seeds = cube_seeds if c == "Cube" else other_seeds
        r0 = [cells[(c, s)]["r0_auroc"] for s in seeds]
        d = [cells[(c, s)]["d_auroc"] for s in seeds]
        ax.scatter(r0, d, label=c, color=colors[c], s=60)
    z = np.polyfit(all_r0, all_d, 1)
    xs = np.linspace(min(all_r0), max(all_r0), 20)
    ax.plot(xs, np.polyval(z, xs), "k--", lw=0.8)
    ax.axhline(0, color="grey", lw=0.6)
    ax.set_xlabel("R0 AUROC (frozen baseline)")
    ax.set_ylabel("delta AUROC")
    ax.set_title(f"R0 vs delta (Pearson r={overall_pearson:+.2f}) — ceiling effect")
    ax.legend()
    fig.tight_layout(); fig.savefig(FIGS / "fig3_r0_vs_delta_scatter.png", dpi=150); plt.close(fig)

    # fig4 positive ratio
    fig, ax = plt.subplots(figsize=(6, 4))
    ratios = [bird["positive_auroc"] / bird["n_seeds"],
              car["positive_auroc"] / car["n_seeds"],
              cube["positive_auroc"] / cube["n_seeds"]]
    ax.bar(CATS, ratios, color=["#2c7fb8", "#d95f02", "#999999"])
    ax.set_ylim(0, 1)
    ax.set_ylabel("positive AUROC ratio")
    ax.set_title(f"E13-B2 positive ratio (Cube {cube['positive_auroc']}/{cube['n_seeds']})")
    fig.tight_layout(); fig.savefig(FIGS / "fig4_positive_ratio.png", dpi=150); plt.close(fig)

    # fig5 heatmap
    fig, ax = plt.subplots(figsize=(6.5, 4))
    all_seeds = sorted({s for c in CATS for s in (cube_seeds if c == "Cube" else other_seeds)})
    mat = np.full((len(CATS), len(all_seeds)), np.nan)
    for i, c in enumerate(CATS):
        for j, s in enumerate(all_seeds):
            if (c, s) in cells:
                mat[i, j] = cells[(c, s)]["d_auroc"]
    im = ax.imshow(mat, cmap="RdBu_r", vmin=-0.02, vmax=0.04, aspect="auto")
    ax.set_xticks(range(len(all_seeds))); ax.set_xticklabels([f"s{s}" for s in all_seeds])
    ax.set_yticks(range(len(CATS))); ax.set_yticklabels(CATS)
    for i in range(len(CATS)):
        for j in range(len(all_seeds)):
            if not np.isnan(mat[i, j]):
                ax.text(j, i, f"{mat[i, j]:+.3f}", ha="center", va="center", fontsize=8)
    ax.set_title("E13-B2 dAUROC heatmap (grey = not run)")
    fig.colorbar(im, ax=ax, label="dAUROC")
    fig.tight_layout(); fig.savefig(FIGS / "fig5_seed_heatmap.png", dpi=150); plt.close(fig)

    print(json.dumps(result, indent=2, default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
