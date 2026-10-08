#!/usr/bin/env python
"""E4-X - three quick screening figures (docs/E4_X_PROTOCOL.md section 19)."""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "e4_x"
FIG = OUT / "figures"
COLOR = {"Original": "#444444", "B2": "#1f77b4", "X6c": "#d62728"}

summary = list(csv.DictReader(open(OUT / "summary.csv")))
ill = list(csv.DictReader(open(OUT / "illumination_metrics.csv")))
per = {m: list(csv.DictReader(open(OUT / f"per_image_{m.lower()}.csv"))) for m in ("Original", "B2", "X6c")}


def fig1():
    fig, ax = plt.subplots(figsize=(5.6, 4.4))
    for r in summary:
        m = r["method"]
        ax.scatter(float(r["R_all"]), float(r["image_auroc"]), s=150, c=COLOR[m],
                   edgecolor="k", zorder=3, label=m)
        ax.annotate(m, (float(r["R_all"]), float(r["image_auroc"])),
                    textcoords="offset points", xytext=(9, 6), fontsize=10)
    o = next(r for r in summary if r["method"] == "Original")
    ax.axvline(float(o["R_all"]) * 0.95, ls="--", c="green", lw=1, alpha=.7)
    ax.axhline(float(o["image_auroc"]) - 0.03, ls="--", c="green", lw=1, alpha=.7)
    ax.annotate("GO region (upper-left)", xy=(ax.get_xlim()[0], 0), xytext=(0.02, 0.06),
                textcoords="axes fraction", fontsize=9, color="green")
    ax.set_xlabel("R_all  (illumination dispersion, lower = more robust)")
    ax.set_ylabel("Image AUROC  (higher = better detection)")
    ax.set_title("E4-X  robustness-detection plane\nM2AD Bird, seed 0, view 120, 10 illuminations", fontsize=10)
    ax.grid(alpha=.3)
    fig.tight_layout()
    fig.savefig(FIG / "fig1_detection_vs_dispersion.png", dpi=140)
    plt.close(fig)


def fig2():
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    x = np.arange(10)
    for m in ("Original", "B2", "X6c"):
        y = [float(r["image_auroc"]) for r in ill if r["method"] == m]
        ax.plot(x, y, "o-", color=COLOR[m], label=m)
    ax.set_xticks(x)
    ax.set_xticklabels([f"I{r['illumination']}" for r in ill if r["method"] == "Original"])
    ax.set_ylabel("Image AUROC")
    ax.set_ylim(0.65, 0.90)
    ax.set_title("E4-X  per-illumination image AUROC (frozen view 120)", fontsize=10)
    ax.grid(alpha=.3); ax.legend()
    fig.tight_layout()
    fig.savefig(FIG / "fig2_illumination_auroc.png", dpi=140)
    plt.close(fig)


def fig3():
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.8))
    for ax, kind in zip(axes, ("Good", "NG")):
        data, labels = [], []
        for m in ("Original", "B2", "X6c"):
            rows = per[m]
            g = np.array([float(r["score"]) for r in rows if r["kind"] == "Good"])
            mu, sd = g.mean(), g.std(ddof=1)
            grp = {}
            for r in rows:
                if r["kind"] != kind:
                    continue
                grp.setdefault(r["specimen"], []).append(float(r["score"]))
            vals = [((np.array(v) - mu) / sd).std(ddof=1) for v in grp.values() if len(v) == 10]
            data.append(vals); labels.append(m)
        bp = ax.boxplot(data, tick_labels=labels, patch_artist=True, widths=.6)
        for patch, m in zip(bp["boxes"], labels):
            patch.set_facecolor(COLOR[m]); patch.set_alpha(.45)
        ax.set_ylabel("per-specimen std across 10 illuminations (z)")
        ax.set_title(f"{kind} specimens", fontsize=10)
        ax.grid(alpha=.3, axis="y")
    fig.suptitle("E4-X  paired illumination dispersion (view 120)", fontsize=11)
    fig.tight_layout()
    fig.savefig(FIG / "fig3_paired_dispersion.png", dpi=140)
    plt.close(fig)


if __name__ == "__main__":
    FIG.mkdir(parents=True, exist_ok=True)
    fig1(); fig2(); fig3()
    print("[figs] written to", FIG)
