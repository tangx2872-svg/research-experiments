"""Experiment 13-P — Figure 1/2/3（CPU-only，读 analysis CSV）。"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "results" / "experiment_13p"
ANA, FIG = EXP / "analysis", EXP / "figures"
CATS = ["bottle", "cable", "hazelnut", "screw", "grid"]
LAYERS = ["layer2", "layer3"]
KGRID = [1, 2, 4, 8, 16, 32]


def rd(name):
    return list(csv.DictReader(open(ANA / name)))


def main() -> None:
    A, B, C, D = rd("A_low_dimensionality.csv"), rd("B_augmentation_stability.csv"), \
        rd("C_cross_category.csv"), rd("D_seed_stability.csv")
    fig, axes = plt.subplots(1, 2, figsize=(12.6, 4.7))
    for ax, layer in zip(axes, LAYERS):
        for cat in CATS:
            r = next((x for x in A if x["definition"] == "PRIMARY_centered" and x["category"] == cat
                      and x["layer"] == layer and x["seed"] == "0"), None)
            if r is None:
                continue
            ax.plot(KGRID, [float(r[f"EV@{k}"]) for k in KGRID], "-o", ms=4, label=cat)
        ax.axhline(0.70, ls="--", c="#C44E52", lw=1.2, label="CASE A threshold EV@8 >= 0.70")
        ax.set_xlabel("K (number of PCs)"); ax.set_ylabel("cumulative explained variance")
        ax.set_title(layer, fontsize=11); ax.set_xticks(KGRID); ax.grid(alpha=0.3)
        ax.set_ylim(0, 1.0)
    axes[0].legend(fontsize=7.5)
    fig.suptitle("Exp13-P Fig1 — Cumulative explained variance of illumination paired-differences "
                 "(train/good only, seed0, PRIMARY centered)", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(FIG / "fig1_ev_spectrum.png", dpi=170, bbox_inches="tight"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    xs = np.arange(len(CATS)); w = 0.36
    for i, layer in enumerate(LAYERS):
        vals = [float(next(x for x in B if x["definition"] == "PRIMARY_centered" and x["seed"] == "0"
                            and x["category"] == c and x["layer"] == layer)["gamma_vs_brightness_sim_K8"])
                for c in CATS]
        ax.bar(xs + (i - 0.5) * w, vals, w, label=layer, edgecolor="black", lw=0.6)
    ax.axhline(0.60, ls="--", c="#C44E52", lw=1.2, label="CASE A threshold 0.60")
    ax.set_xticks(xs); ax.set_xticklabels(CATS); ax.set_ylim(0, 1.05)
    ax.set_ylabel("mean cos²(principal angles), K=8")
    ax.set_title("Fig2 — gamma-subspace vs brightness-subspace similarity (seed0)")
    ax.legend(fontsize=8); ax.grid(alpha=0.3, axis="y")
    fig.tight_layout(); fig.savefig(FIG / "fig2_augmentation_similarity.png", dpi=170, bbox_inches="tight")
    plt.close(fig)

    for layer, fname in ((LAYERS[0], "fig3_cross_category_layer2.png"),
                         (LAYERS[1], "fig3_cross_category_layer3.png")):
        M = np.eye(len(CATS))
        for r in C:
            if r["definition"] != "PRIMARY_centered" or r["layer"] != layer:
                continue
            i, j = CATS.index(r["cat_a"]), CATS.index(r["cat_b"])
            M[i, j] = M[j, i] = float(r["sim_K8_seed0"])
        fig, ax = plt.subplots(figsize=(5.2, 4.4))
        im = ax.imshow(M, vmin=0, vmax=1, cmap="viridis")
        for i in range(len(CATS)):
            for j in range(len(CATS)):
                ax.text(j, i, "%.2f" % M[i, j], ha="center", va="center", fontsize=8,
                        color="white" if M[i, j] < 0.6 else "black")
        ax.set_xticks(range(len(CATS))); ax.set_xticklabels(CATS, fontsize=8)
        ax.set_yticks(range(len(CATS))); ax.set_yticklabels(CATS, fontsize=8)
        ax.set_title("Fig3 — cross-category subspace similarity (%s, K=8, seed0)" % layer, fontsize=9.5)
        fig.colorbar(im, ax=ax, fraction=0.046)
        fig.tight_layout(); fig.savefig(FIG / fname, dpi=170, bbox_inches="tight"); plt.close(fig)
    print("figures 1-3 written ->", FIG)


if __name__ == "__main__":
    main()
