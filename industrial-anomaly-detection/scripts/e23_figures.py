#!/usr/bin/env python
"""E23 figures (Fig 1-5), PNG + PDF, from results/e23 audit outputs + frozen E22 artifacts.

  Fig1  Doll per-image F0 vs R1 score scatter (good / anomaly), one panel per seed
  Fig2  per-image delta_score distributions, good vs anomaly (one panel per seed)
  Fig3  Doll illumination-level dAUROC / dAP (10 groups x 3 seeds)
  Fig4  10-category illumination shift vs mean dAUROC, Doll highlighted (n = 10, exploratory)
  Fig5  10-category mean dAUROC with cross-seed spread, Doll highlighted

Read-only: no experiment is re-run; only stored scores/tables are plotted.
"""
from __future__ import annotations

import csv
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
E22 = ROOT / "results" / "e22"
OUT = Path(os.environ.get("E23_OUT", str(ROOT / "results" / "e23")))
FIG = OUT / "figures"
SEEDS = [0, 1, 2]

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

plt.rcParams.update({"figure.dpi": 120, "savefig.bbox": "tight", "font.size": 9,
                     "axes.grid": True, "grid.alpha": 0.3, "axes.axisbelow": True})
C_F0, C_R1 = "#4C72B0", "#C44E52"
C_GOOD, C_ANOM = "#55A868", "#C44E52"
C_DOLL = "#C44E52"


def save(fig, name: str) -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(FIG / f"{name}.{ext}")
    plt.close(fig)
    print(f"[E23 figures] wrote {name}.png/.pdf", flush=True)


def load_doll_cells() -> dict:
    return {s: json.loads((E22 / "cells" / f"Doll_seed{s}.json").read_text()) for s in SEEDS}


def main() -> int:
    cells = load_doll_cells()
    per_img = list(csv.DictReader(open(OUT / "doll_per_image_delta.csv")))
    ill = list(csv.DictReader(open(OUT / "doll_illumination_breakdown.csv")))
    comp = list(csv.DictReader(open(OUT / "category_comparison.csv")))
    vd = json.loads((OUT / "e23_verdict.json").read_text())

    # ---------------- Fig 1: per-image F0 vs R1 scatter ----------------
    fig, axes = plt.subplots(1, 3, figsize=(11.0, 3.6), sharex=True, sharey=True)
    for ax, s in zip(axes, SEEDS):
        rows = [r for r in per_img if int(r["seed"]) == s and r["detectable"] == "True"]
        gx = [float(r["score_f0"]) for r in rows if int(r["label_official"]) == 0]
        gy = [float(r["score_r1"]) for r in rows if int(r["label_official"]) == 0]
        ax_ = [float(r["score_f0"]) for r in rows if int(r["label_official"]) == 1]
        ay = [float(r["score_r1"]) for r in rows if int(r["label_official"]) == 1]
        lo = min(min(gx), min(gy), min(ax_), min(ay)) * 0.98
        hi = max(max(gx), max(gy), max(ax_), max(ay)) * 1.02
        ax.plot([lo, hi], [lo, hi], color="0.6", lw=0.8, ls="--", zorder=1)
        ax.scatter(gx, gy, s=5, c=C_GOOD, alpha=0.45, lw=0, label="good", zorder=2)
        ax.scatter(ax_, ay, s=5, c=C_ANOM, alpha=0.35, lw=0, label="anomaly", zorder=3)
        c = cells[s]
        ax.set_title(f"seed{s}: AUROC {c['f0']['auroc']:.4f} -> {c['r1']['auroc']:.4f} "
                     f"(Δ {c['delta_auroc']:+.4f})", fontsize=8.5)
        ax.set_xlabel("F0 score")
        if ax is axes[0]:
            ax.set_ylabel("R1 score")
            ax.legend(fontsize=7, loc="lower right")
    fig.suptitle("E23 Fig1 — Doll per-image scores: F0 vs R1 (view120, detectable-filtered)", fontsize=9)
    save(fig, "fig1_doll_f0_vs_r1_scatter")

    # ---------------- Fig 2: delta_score distributions ----------------
    fig, axes = plt.subplots(1, 3, figsize=(11.0, 3.4), sharex=True)
    for ax, s in zip(axes, SEEDS):
        rows = [r for r in per_img if int(r["seed"]) == s and r["detectable"] == "True"]
        gd = [float(r["delta_score"]) for r in rows if int(r["label_official"]) == 0]
        ad = [float(r["delta_score"]) for r in rows if int(r["label_official"]) == 1]
        bins = np.linspace(min(min(gd), min(ad)), max(max(gd), max(ad)), 45)
        ax.hist(gd, bins=bins, color=C_GOOD, alpha=0.55, label=f"good (n={len(gd)})")
        ax.hist(ad, bins=bins, color=C_ANOM, alpha=0.45, label=f"anomaly (n={len(ad)})")
        ax.axvline(np.mean(gd), color=C_GOOD, lw=1.1, ls="--")
        ax.axvline(np.mean(ad), color=C_ANOM, lw=1.1, ls="--")
        ax.set_title(f"seed{s}: mean Δ good {np.mean(gd):+.4f} / anomaly {np.mean(ad):+.4f}",
                     fontsize=8.5)
        ax.set_xlabel("Δscore = R1 − F0")
        if ax is axes[0]:
            ax.set_ylabel("# images")
            ax.legend(fontsize=7)
    fig.suptitle("E23 Fig2 — Doll per-image Δscore distributions (dashed = class mean)", fontsize=9)
    save(fig, "fig2_doll_delta_score_distributions")

    # ---------------- Fig 3: illumination-level dAUROC / dAP ----------------
    groups = sorted({r["illumination"] for r in ill if r["illumination"] != "ALL"})
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 3.6), sharex=True)
    for ax, key, lbl in ((axes[0], "d_auroc", "ΔAUROC"), (axes[1], "d_ap", "ΔAP")):
        x = np.arange(len(groups))
        vals = []
        for s in SEEDS:
            y = [float(next(r[key] for r in ill
                            if int(r["seed"]) == s and r["illumination"] == gr))
                 for gr in groups]
            vals.append(y)
            ax.plot(x, y, marker="o", ms=3, lw=1.0, label=f"seed{s}")
        ax.axhline(0.0, color="0.4", lw=0.9)
        ax.set_xticks(x)
        ax.set_xticklabels(groups)
        ax.set_xlabel("illumination group")
        ax.set_ylabel(lbl)
        ax.legend(fontsize=7)
    fig.suptitle("E23 Fig3 — Doll illumination-level ΔAUROC / ΔAP (per seed; ALL-group values are in the CSV)",
                 fontsize=9)
    save(fig, "fig3_doll_illumination_delta")

    # ---------------- Fig 4: shift vs gain (10 categories) ----------------
    fig, ax = plt.subplots(figsize=(5.4, 4.0))
    for r in comp:
        is_doll = r["category"] == "Doll"
        ax.scatter(float(r["D_shift_F0"]), float(r["d_auroc_mean"]),
                   s=64 if is_doll else 26, c=C_DOLL if is_doll else "#4C72B0",
                   edgecolor="k" if is_doll else "none", lw=0.7, zorder=3 if is_doll else 2)
        ax.annotate(r["category"], (float(r["D_shift_F0"]), float(r["d_auroc_mean"])),
                    textcoords="offset points", xytext=(4, 4),
                    fontsize=7.5, weight="bold" if is_doll else "normal")
    xs = np.array([float(r["D_shift_F0"]) for r in comp])
    ys = np.array([float(r["d_auroc_mean"]) for r in comp])
    slope, intercept = np.polyfit(xs, ys, 1)
    xr = np.linspace(xs.min(), xs.max(), 50)
    ax.plot(xr, slope * xr + intercept, color="0.5", lw=0.9, ls="--")
    ax.axhline(0.0, color="0.4", lw=0.8)
    s = vd["shift_gain"]
    ax.set_xlabel("illumination shift D (F0, seed mean)")
    ax.set_ylabel("mean ΔAUROC (3 seeds)")
    ax.set_title(f"E23 Fig4 — shift vs gain, n=10 (Pearson r={s['full_n10']['pearson_r']:+.3f}, "
                 f"p={s['full_n10']['pearson_p']:.3f}; exploratory, Doll highlighted)", fontsize=8.5)
    save(fig, "fig4_shift_vs_gain_10cat_doll_highlighted")

    # ---------------- Fig 5: 10-category dAUROC with seed spread ----------------
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    order = sorted(comp, key=lambda r: float(r["d_auroc_mean"]))
    x = np.arange(len(order))
    means = [float(r["d_auroc_mean"]) for r in order]
    stds = [float(r["d_auroc_std"]) for r in order]
    cols = [C_DOLL if r["category"] == "Doll" else "#4C72B0" for r in order]
    ax.bar(x, means, yerr=stds, color=cols, width=0.62, capsize=2.5,
           error_kw={"lw": 0.9, "ecolor": "0.35"})
    ax.axhline(0.0, color="0.3", lw=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels([r["category"] for r in order], rotation=35, ha="right")
    ax.set_ylabel("ΔAUROC (mean ± cross-seed std)")
    ax.set_title("E23 Fig5 — 10-category ΔAUROC; Doll = failure boundary (E22, view120)",
                 fontsize=9)
    save(fig, "fig5_delta_auroc_10cat_doll_highlighted")

    print("[E23 figures] done", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
