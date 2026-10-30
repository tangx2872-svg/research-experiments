#!/usr/bin/env python
"""Export the three camera-ready paper figures.

Read-only with respect to existing results: no training, no inference, no re-scoring.
All numbers are read from four frozen CSV files copied into ``inputs/``:

  main_table_10cat_3seed.csv   (E22)  10-category 3-seed mean / sample std of AUROC
  all10_3seed_cells.csv        (E22)  per-cell AUROC (view120), used for the per-seed dots
  doll_per_image_delta.csv     (E23)  per-image F0/R1 scores for Doll (detectable filter)
  analysis_cells.csv           (E21)  category x view AUROC, seed 0

Differences from the original runtime scripts (scripts/e22_figures.py,
scripts/e23_figures.py, scripts/e21_figures.py) are restricted to presentation:

  * in-figure run codes / figure numbers ("E21", "E23", "Fig 1", "Fig 2") and the
    suptitle/title lines that carry them are removed -- captions live in the manuscript;
  * export resolution is raised to 300 dpi (PNG) and PDF text is embedded as Type 42;
  * axis ranges are locked to the values produced by the original scripts
    (captured read-only from those scripts and verified below).

Nothing else is changed: data, detectable filtering, error-bar definition
(cross-seed sample std, ddof=1), colour mapping and axis ranges are identical.

Usage
-----
    python export_paper_figures.py                 # uses ../inputs and ../figures
    python export_paper_figures.py --out DIR        # custom output directory
    python export_paper_figures.py --no-limits-check
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
DEFAULT_INPUTS = HERE.parent / "inputs"
DEFAULT_OUT = HERE.parent / "figures"

# Colours are copied verbatim from the original scripts so that they keep matching
# the rest of the manuscript.
C_F0, C_R1 = "#4C72B0", "#C44E52"          # frozen baseline / adapted model
C_NEG = "#7F7F7F"                          # negative Delta AUROC bars
C_GOOD, C_ANOM = "#55A868", "#C44E52"      # Doll scatter: normal / anomalous
VIEWS = ["090", "120", "240", "330"]
CATS_MULTIVIEW = ["Bird", "Car", "Cube"]

# Axis ranges produced by the original scripts (captured read-only, see README).
# They are re-applied explicitly so that the exported files cannot drift.
ORIGINAL_LIMITS = {
    "delta_auroc_10cat_horizontal": {"xlim": (-0.048014, 0.043554), "ylim": (-0.7525, 9.7525)},
    "doll_f0_vs_r1_per_image_scatter": {"xlim": (0.094993, 0.520450), "ylim": (0.094993, 0.520450)},
    "multiview_delta_auroc_heatmap": {"xlim": (-0.5, 3.5), "ylim": (2.5, -0.5)},
}

STYLE = {
    "figure.dpi": 120,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "font.size": 10,
    "axes.labelsize": 10.5,
    "xtick.labelsize": 9.5,
    "ytick.labelsize": 9.5,
    "legend.fontsize": 9,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "axes.axisbelow": True,
}


def _hash_array(arr: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(np.round(arr, 12)).tobytes()).hexdigest()


def _read_csv(path: Path) -> list[dict]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


# --------------------------------------------------------------------------------------
# Figure 1: 10-category Delta AUROC (horizontal bars, 3-seed mean +- sample std, per-seed dots)
# --------------------------------------------------------------------------------------
def fig_delta_auroc_10cat(main_csv: Path, cells_csv: Path, lock_limits: bool) -> tuple:
    main_rows = _read_csv(main_csv)
    cells = {(r["category"], int(r["seed"])): r
             for r in _read_csv(cells_csv) if r["view"] == "120"}

    cats = [r["category"] for r in main_rows]
    y = np.arange(len(cats))[::-1]
    d_au = np.array([float(r["d_auroc_mean"]) for r in main_rows])
    d_au_s = np.array([float(r["d_auroc_std"]) for r in main_rows])

    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.barh(y, d_au, height=0.55, xerr=d_au_s,
            color=np.where(d_au > 0, C_R1, C_NEG), error_kw={"lw": 0.8})
    dots = {}
    for i, cat in enumerate(cats):
        vals = [float(cells[(cat, sd)]["d_auroc"]) for sd in (0, 1, 2) if (cat, sd) in cells]
        dots[cat] = vals
        ax.plot(vals, [y[i]] * len(vals), "o", ms=3.5, color="k", alpha=0.65, zorder=3)
    ax.axvline(0, color="k", lw=1)
    ax.set_yticks(y, cats)
    ax.set_xlabel("$\\Delta$AUROC (R1-ILA $-$ F0)\n"
                  "bars = 3-seed mean $\\pm$ std, dots = individual seeds")

    lim = ORIGINAL_LIMITS["delta_auroc_10cat_horizontal"]
    if lock_limits:
        ax.set_xlim(*lim["xlim"])
        ax.set_ylim(*lim["ylim"])

    payload = {
        "categories": cats,
        "bar_values": d_au.tolist(),
        "xerr_sample_std_ddof1": d_au_s.tolist(),
        "per_seed_dots": dots,
        "positive_categories": int((d_au > 0).sum()),
        "xlim": list(ax.get_xlim()),
        "ylim": list(ax.get_ylim()),
    }
    return fig, payload


# --------------------------------------------------------------------------------------
# Figure 2: Doll per-image F0 vs R1 scores (one panel per seed, detectable-filtered)
# --------------------------------------------------------------------------------------
def _rank_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    """Rank-based AUROC with mid-rank ties (matches the stored protocol metric)."""
    order = np.argsort(scores, kind="mergesort")
    s, y = scores[order], labels[order]
    n = len(y)
    ranks = np.empty(n, dtype=float)
    i = 0
    while i < n:
        j = i
        while j + 1 < n and s[j + 1] == s[i]:
            j += 1
        ranks[i:j + 1] = (i + j) / 2.0 + 1.0
        i = j + 1
    y_sorted = y
    n_pos = float(y_sorted.sum())
    n_neg = float(n - n_pos)
    r_pos = ranks[y_sorted == 1].sum()
    return float((r_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def fig_doll_scatter(per_img_csv: Path, cells_csv: Path, lock_limits: bool) -> tuple:
    rows = _read_csv(per_img_csv)
    stored = {(r["category"], int(r["seed"])): r
              for r in _read_csv(cells_csv) if r["view"] == "120"}

    fig, axes = plt.subplots(1, 3, figsize=(11.0, 3.6), sharex=True, sharey=True)
    per_seed = {}

    for ax, s in zip(axes, (0, 1, 2)):
        sel = [r for r in rows if int(r["seed"]) == s and r["detectable"] == "True"]
        good = [(float(r["score_f0"]), float(r["score_r1"]))
                for r in sel if int(r["label_official"]) == 0]
        anom = [(float(r["score_f0"]), float(r["score_r1"]))
                for r in sel if int(r["label_official"]) == 1]
        gx, gy = [p[0] for p in good], [p[1] for p in good]
        axx, ayy = [p[0] for p in anom], [p[1] for p in anom]

        lo = min(min(gx), min(gy), min(axx), min(ayy)) * 0.98
        hi = max(max(gx), max(gy), max(axx), max(ayy)) * 1.02
        ax.plot([lo, hi], [lo, hi], color="0.6", lw=0.8, ls="--", zorder=1)
        ax.scatter(gx, gy, s=5, c=C_GOOD, alpha=0.45, lw=0, label="normal", zorder=2)
        ax.scatter(axx, ayy, s=5, c=C_ANOM, alpha=0.35, lw=0, label="anomalous", zorder=3)

        # panel-level statistics: stored AUROC values + independent re-computation
        # from the plotted per-image scores (guards against a silent data swap)
        lab = np.array([int(r["label_official"]) for r in sel], dtype=int)
        rec_f0 = _rank_auc(lab, np.array([float(r["score_f0"]) for r in sel]))
        rec_r1 = _rank_auc(lab, np.array([float(r["score_r1"]) for r in sel]))
        st = stored[("Doll", s)]
        st_f0, st_r1 = float(st["auroc_f0"]), float(st["auroc_r1"])
        ax.set_title(f"seed {s}: AUROC {st_f0:.4f} \u2192 {st_r1:.4f} "
                     f"($\\Delta$ {st_r1 - st_f0:+.4f})", fontsize=9.5)

        per_seed[s] = {
            "n_images": len(sel),
            "n_normal": len(good),
            "n_anomalous": len(anom),
            "auroc_stored_f0": round(st_f0, 6),
            "auroc_stored_r1": round(st_r1, 6),
            "auroc_recomputed_f0": round(rec_f0, 6),
            "auroc_recomputed_r1": round(rec_r1, 6),
            "auroc_max_abs_diff": round(max(abs(rec_f0 - st_f0), abs(rec_r1 - st_r1)), 9),
            "offsets_sha256": _hash_array(np.array(good + anom, dtype=float)),
        }
        ax.set_xlabel("F0 score")
        if ax is axes[0]:
            ax.set_ylabel("R1 score")
            ax.legend(loc="lower right")

    lim = ORIGINAL_LIMITS["doll_f0_vs_r1_per_image_scatter"]
    if lock_limits:
        for ax in axes:
            ax.set_xlim(*lim["xlim"])
            ax.set_ylim(*lim["ylim"])

    payload = {
        "per_seed": per_seed,
        "detectable_filter": "detectable == True (670 images per seed: 270 normal + 400 anomalous)",
        "xlim": list(axes[0].get_xlim()),
        "ylim": list(axes[0].get_ylim()),
    }
    return fig, payload


# --------------------------------------------------------------------------------------
# Figure 3: category x view Delta AUROC heatmap (seed 0)
# --------------------------------------------------------------------------------------
def fig_multiview_heatmap(cells_csv: Path, lock_limits: bool) -> tuple:
    cells = [r for r in _read_csv(cells_csv) if int(r["seed"]) == 0]
    mat = np.full((len(CATS_MULTIVIEW), len(VIEWS)), np.nan)
    for r in cells:
        mat[CATS_MULTIVIEW.index(r["category"]), VIEWS.index(r["view"])] = float(r["d_auroc"])

    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    lim = max(0.03, float(np.nanmax(np.abs(mat))))
    im = ax.imshow(mat, cmap="RdBu_r", vmin=-lim, vmax=lim, aspect="auto")
    for i in range(len(CATS_MULTIVIEW)):
        for j in range(len(VIEWS)):
            ax.text(j, i, f"{mat[i, j]:+.4f}", ha="center", va="center", fontsize=9)
    ax.set_xticks(range(len(VIEWS)), VIEWS)
    ax.set_yticks(range(len(CATS_MULTIVIEW)), CATS_MULTIVIEW)
    ax.set_xlabel("view (seed 0)")
    ax.set_ylabel("category")
    fig.colorbar(im, ax=ax, label="$\\Delta$AUROC")
    ax.grid(False)

    limits = ORIGINAL_LIMITS["multiview_delta_auroc_heatmap"]
    if lock_limits:
        ax.set_xlim(*limits["xlim"])
        ax.set_ylim(*limits["ylim"])

    payload = {
        "categories": CATS_MULTIVIEW,
        "views": VIEWS,
        "matrix": mat.tolist(),
        "colour_limit": lim,
        "n_negative_cells": int((mat < 0).sum()),
        "xlim": list(ax.get_xlim()),
        "ylim": list(ax.get_ylim()),
    }
    return fig, payload


def _save(fig, out: Path, stem: str) -> list[str]:
    out.mkdir(parents=True, exist_ok=True)
    paths = []
    for ext in ("png", "pdf"):
        p = out / f"{stem}.{ext}"
        # PNG at 300 dpi; PDF stays vector, the dpi only raises the resolution of the
        # one raster element (the heatmap colour field drawn by imshow).
        fig.savefig(p, dpi=300 if ext == "png" else 600)
        paths.append(str(p))
        print(f"  wrote {p.name}  ({p.stat().st_size / 1024:.0f} KB)", flush=True)
    plt.close(fig)
    return paths


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", type=Path, default=DEFAULT_INPUTS)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--checks", type=Path, default=HERE.parent / "checks")
    ap.add_argument("--no-limits-check", action="store_true",
                    help="do not force the axis ranges of the original scripts")
    args = ap.parse_args()

    plt.rcParams.update(STYLE)
    lock = not args.no_limits_check
    checks = {"limits_locked_to_original": lock, "figures": {}}

    print("[paper figures] Figure 1/3 -> delta AUROC, 10 categories", flush=True)
    fig, payload = fig_delta_auroc_10cat(args.inputs / "main_table_10cat_3seed.csv",
                                         args.inputs / "all10_3seed_cells.csv", lock)
    _save(fig, args.out, "delta_auroc_10cat_horizontal")
    checks["figures"]["delta_auroc_10cat_horizontal"] = payload

    print("[paper figures] Figure 2/3 -> Doll per-image F0 vs R1", flush=True)
    fig, payload = fig_doll_scatter(args.inputs / "doll_per_image_delta.csv",
                                    args.inputs / "all10_3seed_cells.csv", lock)
    _save(fig, args.out, "doll_f0_vs_r1_per_image_scatter")
    checks["figures"]["doll_f0_vs_r1_per_image_scatter"] = payload

    print("[paper figures] Figure 3/3 -> category x view heatmap, seed 0", flush=True)
    fig, payload = fig_multiview_heatmap(args.inputs / "analysis_cells.csv", lock)
    _save(fig, args.out, "multiview_delta_auroc_heatmap")
    checks["figures"]["multiview_delta_auroc_heatmap"] = payload

    args.checks.mkdir(parents=True, exist_ok=True)
    p = args.checks / "export_payload.json"
    p.write_text(json.dumps(checks, indent=2))
    print(f"[paper figures] payload -> {p}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
