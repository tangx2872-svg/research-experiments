#!/usr/bin/env python
"""E22 figures (Fig 1-8), PNG + PDF, from results/e22 analysis outputs.

Content requirements (protocol section 7):
  Fig1 10-category F0 vs R1 AUROC (mean ± cross-seed std)
  Fig2 10-category mean ΔAUROC with per-seed variability
  Fig3 10-category ΔAP
  Fig4 category x seed ΔAUROC heatmap (10x3)
  Fig5 positive-seed ratio / stability
  Fig6 illumination shift vs gain (exploratory association, n=10)
  Fig7 forest plot (category-level seed-mean-score bootstrap CI)
  Fig8 best / median / worst category illumination breakdown  (POST-HOC descriptive)
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(os.environ.get("E22_OUT", str(ROOT / "results" / "e22")))
FIG = OUT / "figures"
CATS = ["Bird", "Car", "Cube", "Dice", "Doll", "Holder", "Motor", "Ring", "Teapot", "Tube"]
SEEDS = [0, 1, 2]

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

plt.rcParams.update({"figure.dpi": 120, "savefig.bbox": "tight", "font.size": 9,
                     "axes.grid": True, "grid.alpha": 0.3, "axes.axisbelow": True})
C_F0, C_R1 = "#4C72B0", "#C44E52"


def save(fig, name: str) -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(FIG / f"{name}.{ext}")
    plt.close(fig)
    print(f"[E22 figures] wrote {name}.png/.pdf", flush=True)


def load_cells() -> dict:
    cells = {}
    for cat in CATS:
        for seed in SEEDS:
            p = OUT / "cells" / f"{cat}_seed{seed}.json"
            if p.exists():
                cells[(cat, seed)] = json.loads(p.read_text())
    return cells


def main() -> int:
    cells = load_cells()
    main_rows = list(__import__("csv").DictReader(open(OUT / "main_table_10cat_3seed.csv")))
    boot = json.loads((OUT / "bootstrap_e22.json").read_text())
    shift = json.loads((OUT / "shift_gain_association.json").read_text())
    spread = list(__import__("csv").DictReader(open(OUT / "illumination_spread.csv")))
    illum = list(__import__("csv").DictReader(open(OUT / "illumination_breakdown.csv")))
    cats = [r["category"] for r in main_rows]
    y = np.arange(len(cats))[::-1]
    d_au = np.array([float(r["d_auroc_mean"]) for r in main_rows])
    d_ap = np.array([float(r["d_ap_mean"]) for r in main_rows])
    au_f0 = np.array([float(r["auroc_f0_mean"]) for r in main_rows])
    au_r1 = np.array([float(r["auroc_r1_mean"]) for r in main_rows])
    au_f0_s = np.array([float(r["auroc_f0_std"]) for r in main_rows])
    au_r1_s = np.array([float(r["auroc_r1_std"]) for r in main_rows])
    d_au_s = np.array([float(r["d_auroc_std"]) for r in main_rows])

    # ---- Fig 1: F0 vs R1 AUROC
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.barh(y + 0.2, au_f0, height=0.38, xerr=au_f0_s, color=C_F0, label="F0 (frozen DINOv2-B)",
            error_kw={"lw": 0.8})
    ax.barh(y - 0.2, au_r1, height=0.38, xerr=au_r1_s, color=C_R1, label="R1-ILA (frozen)",
            error_kw={"lw": 0.8})
    ax.set_yticks(y, cats)
    ax.set_xlabel("AUROC (view120, official labels; mean ± cross-seed std over 3 seeds)")
    ax.set_title("Fig 1 — M²AD 10 categories: F0 vs R1-ILA")
    ax.legend(loc="lower right", fontsize=8)
    ax.set_xlim(0.5, 1.0)
    save(fig, "fig1_f0_vs_r1_auroc_10cat")

    # ---- Fig 2 / Fig 3: ΔAUROC and ΔAP with per-seed points
    for met, d, s, name, lab in (("AUROC", d_au, d_au_s, "fig2_delta_auroc_10cat", "ΔAUROC"),
                                 ("AP", d_ap, np.array([float(r["d_ap_std"]) for r in main_rows]),
                                  "fig3_delta_ap_10cat", "ΔAP")):
        fig, ax = plt.subplots(figsize=(6.4, 4.2))
        ax.barh(y, d, height=0.55, xerr=s, color=np.where(d > 0, C_R1, "#7F7F7F"),
                error_kw={"lw": 0.8})
        for i, cat in enumerate(cats):
            vals = [cells[(cat, sd)]["r1"][met.lower()] - cells[(cat, sd)]["f0"][met.lower()]
                    for sd in SEEDS if (cat, sd) in cells]
            ax.plot(vals, [y[i]] * len(vals), "o", ms=3.5, color="k", alpha=0.65, zorder=3)
        ax.axvline(0, color="k", lw=1)
        ax.set_yticks(y, cats)
        ax.set_xlabel(f"{lab} (R1-ILA − F0); bars = 3-seed mean ± std, dots = individual seeds")
        fignum = "2" if met == "AUROC" else "3"
        ax.set_title(f"Fig {fignum} — per-category {lab} (10 categories)")
        save(fig, name)

    # ---- Fig 4: heatmap category x seed
    M = np.full((len(cats), len(SEEDS)), np.nan)
    for i, cat in enumerate(cats):
        for j, sd in enumerate(SEEDS):
            if (cat, sd) in cells:
                M[i, j] = cells[(cat, sd)]["delta_auroc"]
    fig, ax = plt.subplots(figsize=(4.0, 5.0))
    vmax = max(0.05, float(np.nanmax(np.abs(M))))
    im = ax.imshow(M, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(SEEDS)), [f"seed{s}" for s in SEEDS])
    ax.set_yticks(range(len(cats)), cats)
    for i in range(len(cats)):
        for j in range(len(SEEDS)):
            if math.isfinite(M[i, j]):
                ax.text(j, i, f"{M[i, j]:+.3f}", ha="center", va="center", fontsize=7)
    ax.set_title("Fig 4 — ΔAUROC heatmap (category × seed)", fontsize=9)
    fig.colorbar(im, ax=ax, shrink=0.8, label="ΔAUROC")
    ax.grid(False)
    save(fig, "fig4_heatmap_delta_auroc")

    # ---- Fig 5: positive-seed ratio / stability
    pos = np.array([int(r["pos_seeds_auroc"]) for r in main_rows], float)
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    ax.barh(y, pos, height=0.55, color=np.where(pos == 3, "#2E7D32",
                                                np.where(pos == 0, "#B71C1C", "#F9A825")))
    for i, p in enumerate(pos):
        ax.text(p + 0.05, y[i], f"{int(p)}/3", va="center", fontsize=8)
    ax.set_yticks(y, cats)
    ax.set_xlim(0, 3.8)
    ax.set_xlabel("number of seeds with ΔAUROC > 0")
    ax.set_title("Fig 5 — seed-level stability (green 3/3, amber 1–2/3, red 0/3)")
    save(fig, "fig5_positive_seed_ratio")

    # ---- Fig 6: illumination shift vs gain (exploratory)
    xs = np.array([shift["shift"][c]["D_shift_F0_seed_mean"] for c in cats], float)
    fig, ax = plt.subplots(figsize=(5.4, 4.2))
    ax.scatter(xs, d_au, s=28, color=C_R1)
    for c, x, yy in zip(cats, xs, d_au):
        ax.annotate(c, (x, yy), fontsize=7, xytext=(3, 3), textcoords="offset points")
    if np.all(np.isfinite(xs)):
        b, a = np.polyfit(xs, d_au, 1)
        xr = np.linspace(xs.min(), xs.max(), 50)
        ax.plot(xr, a + b * xr, "--", color="k", lw=0.9)
    ass = shift["association"].get("D_shift_seed_mean_vs_dAUROC", {})
    txt = (f"Pearson r={ass.get('pearson_r', float('nan')):+.3f} "
           f"(p={ass.get('pearson_p', float('nan')):.3f})\n"
           f"Spearman ρ={ass.get('spearman_rho', float('nan')):+.3f} "
           f"(p={ass.get('spearman_p', float('nan')):.3f})\nn=10 categories — exploratory only")
    ax.text(0.03, 0.97, txt, transform=ax.transAxes, va="top", fontsize=7,
            bbox={"facecolor": "white", "alpha": 0.8, "lw": 0.5})
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xlabel("illumination shift strength  D_illum_shift_F0 (seed mean, frozen E16 metric)")
    ax.set_ylabel("category 3-seed mean ΔAUROC")
    ax.set_title("Fig 6 — illumination shift vs gain (association, not causal)")
    save(fig, "fig6_illumination_shift_vs_gain")

    # ---- Fig 7: forest plot with paired-bootstrap CI (seed-mean scores)
    fig, ax = plt.subplots(figsize=(6.0, 4.2))
    for i, cat in enumerate(cats):
        sm = boot.get(cat, {}).get("seed_mean_scores", {})
        if not sm:
            continue
        lo, hi = sm["d_auroc_ci"]
        m = float(main_rows[i]["d_auroc_mean"])
        sig = (lo > 0) or (hi < 0)
        ax.plot([lo, hi], [y[i], y[i]], color="k" if sig else "gray", lw=1.4)
        ax.plot([m], [y[i]], "s", ms=5, color=C_R1 if m > 0 else "#7F7F7F")
    ax.axvline(0, color="k", lw=1)
    ax.set_yticks(y, cats)
    ax.set_xlabel("ΔAUROC (seed-mean scores) with 95% paired image-bootstrap CI "
                  "(n_boot=3000) — test-set sampling uncertainty, NOT cross-seed")
    ax.set_title("Fig 7 — category-level forest plot")
    save(fig, "fig7_forest_plot")

    # ---- Fig 8: illumination breakdown for best / median / worst category (POST-HOC)
    order = np.argsort(d_au)
    sel = {"worst": cats[int(order[0])], "median": cats[int(order[len(cats) // 2])],
           "best": cats[int(order[-1])]}
    fig, axes = plt.subplots(1, 3, figsize=(11.0, 3.6), sharey=False)
    for ax, (label, cat) in zip(axes, sel.items()):
        for meth, col in (("F0", C_F0), ("R1", C_R1)):
            gs, vals = [], []
            for g in sorted({r["illumination"] for r in illum if r["illumination"] != "ALL"}):
                v = [float(r["good_mean"]) for r in illum
                     if r["source"] == f"{meth}|{cat}|seed0" and r["illumination"] == g
                     and r["good_mean"]]
                if v:
                    gs.append(g)
                    vals.append(v[0])
            ax.plot(gs, vals, "o-", ms=3.5, lw=1.0, color=col, label=meth)
        ax.set_title(f"{label}: {cat} (seed0)", fontsize=9)
        ax.set_xlabel("illumination group")
        ax.tick_params(axis="x", rotation=45, labelsize=7)
    axes[0].set_ylabel("mean anomaly score of GOOD images")
    axes[-1].legend(fontsize=8)
    fig.suptitle("Fig 8 — illumination breakdown, best/median/worst ΔAUROC category "
                 "(POST-HOC descriptive visualization, seed0, 3-seed-mean selection)", fontsize=9)
    save(fig, "fig8_illumination_best_median_worst")

    print(f"[E22 figures] done -> {FIG}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
