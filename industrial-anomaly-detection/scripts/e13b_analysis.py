#!/usr/bin/env python
"""E13-B R1 cross-seed stability analysis (CPU-only). Combines seed0 (E13-A) + seed1/2 (E13-B)."""
from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "e13b"
FIGS = OUT / "figures"

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

CATS = ["Bird", "Car", "Cube"]
SEEDS = [0, 1, 2]


def load_cells():
    """-> dict[(cat, seed)] = {d_auroc, d_ap, r0_auroc, r0_ap, r1_auroc, r1_ap, ...}"""
    cells = {}
    # seed0 from E13-A screening CSV
    with open(ROOT / "results" / "e13a" / "e13a_screening.csv") as f:
        for r in csv.DictReader(f):
            if r["config"] not in ("R0", "R1"):
                continue
            cat = r["category"]
            if r["config"] == "R0":
                cells[(cat, 0)] = {"r0_auroc": float(r["auroc"]), "r0_ap": float(r["ap"])}
            else:
                cells[(cat, 0)]["r1_auroc"] = float(r["auroc"])
                cells[(cat, 0)]["r1_ap"] = float(r["ap"])
                cells[(cat, 0)]["d_auroc"] = float(r["d_auroc"])
                cells[(cat, 0)]["d_ap"] = float(r["d_ap"])
    # seed1/2 from E13-B results CSV
    with open(OUT / "e13b_results.csv") as f:
        for r in csv.DictReader(f):
            cat = r["category"]
            s = int(r["seed"])
            cells[(cat, s)] = {
                "r0_auroc": float(r["r0_auroc"]), "r0_ap": float(r["r0_ap"]),
                "r1_auroc": float(r["r1_auroc"]), "r1_ap": float(r["r1_ap"]),
                "d_auroc": float(r["d_auroc"]), "d_ap": float(r["d_ap"]),
                "steps_used": int(r["steps_used"]), "std_ratio": float(r["std_ratio"]),
                "pairwise_cos": float(r["pairwise_cos"]),
                "collapsed": r["status"] == "STOP_COLLAPSE",
            }
    return cells


def summarize(cells):
    d_au = [cells[(c, s)]["d_auroc"] for c in CATS for s in SEEDS]
    d_ap = [cells[(c, s)]["d_ap"] for c in CATS for s in SEEDS]
    out = {
        "overall_mean_d_auroc": float(np.mean(d_au)),
        "overall_std_d_auroc": float(np.std(d_au, ddof=1)),
        "overall_min_d_auroc": float(np.min(d_au)),
        "overall_max_d_auroc": float(np.max(d_au)),
        "overall_mean_d_ap": float(np.mean(d_ap)),
        "overall_std_d_ap": float(np.std(d_ap, ddof=1)),
        "positive_auroc_cells": int(sum(1 for x in d_au if x > 0)),
        "positive_ap_cells": int(sum(1 for x in d_ap if x > 0)),
        "n_cells": len(d_au),
        "per_category": {},
        "per_seed": {},
        "cells": {f"{c}|{s}": {"d_auroc": cells[(c, s)]["d_auroc"],
                                "d_ap": cells[(c, s)]["d_ap"]}
                  for c in CATS for s in SEEDS},
    }
    for c in CATS:
        v = [cells[(c, s)]["d_auroc"] for s in SEEDS]
        a = [cells[(c, s)]["d_ap"] for s in SEEDS]
        out["per_category"][c] = {"mean_d_auroc": float(np.mean(v)),
                                  "std_d_auroc": float(np.std(v, ddof=1)),
                                  "mean_d_ap": float(np.mean(a)),
                                  "seeds_d_auroc": {s: cells[(c, s)]["d_auroc"] for s in SEEDS}}
    for s in SEEDS:
        v = [cells[(c, s)]["d_auroc"] for c in CATS]
        out["per_seed"][s] = {"mean_d_auroc": float(np.mean(v)),
                              "cells": {c: cells[(c, s)]["d_auroc"] for c in CATS}}
    return out


def decide(summary, collapsed_any):
    m = summary
    if collapsed_any:
        return "STOP"  # collapse gate
    per_cat_mean = [m["per_category"][c]["mean_d_auroc"] for c in CATS]
    if (m["overall_mean_d_auroc"] >= 0.010 and m["positive_auroc_cells"] >= 7
            and all(x >= 0.005 for x in per_cat_mean) and m["overall_mean_d_ap"] >= 0.005):
        return "PASS"
    if (m["overall_mean_d_auroc"] >= 0.0075 and m["positive_auroc_cells"] >= 6
            and all(x >= 0 for x in per_cat_mean) and m["overall_mean_d_ap"] >= 0):
        return "HOLD"
    return "STOP"


# ------------------------------------------------------------------ figures
def fig_by_cat_seed(summary, metric, fname):
    key = "d_auroc" if metric == "auroc" else "d_ap"
    fig, ax = plt.subplots(figsize=(7, 4.5))
    x = np.arange(len(CATS))
    width = 0.25
    colors = {0: "#bdbdbd", 1: "#2c7fb8", 2: "#7a0177"}
    for j, s in enumerate(SEEDS):
        vals = [summary["cells"][f"{c}|{s}"][key] for c in CATS]
        ax.bar(x + (j - 1) * width, vals, width, label=f"seed{s}", color=colors[s])
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(CATS)
    ax.set_ylabel(f"delta {metric.upper()} vs R0")
    ax.set_title(f"E13-B R1 delta {metric.upper()} by category and seed")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGS / fname, dpi=150)
    plt.close(fig)


def fig_mean_std(summary, fname):
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    x = np.arange(len(CATS))
    means = [summary["per_category"][c]["mean_d_auroc"] for c in CATS]
    stds = [summary["per_category"][c]["std_d_auroc"] for c in CATS]
    ax.bar(x, means, yerr=stds, capsize=5, color="#2c7fb8", alpha=0.85)
    ax.axhline(0, color="black", lw=0.8)
    ax.axhline(0.005, color="grey", ls="--", lw=0.8, label="PASS per-cat bar (+0.005)")
    ax.set_xticks(x)
    ax.set_xticklabels(CATS)
    ax.set_ylabel("mean dAUROC (3 seeds)")
    ax.set_title("E13-B R1 per-category mean +/- std")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGS / fname, dpi=150)
    plt.close(fig)


def fig_heatmap(summary, fname):
    fig, ax = plt.subplots(figsize=(5.5, 4))
    mat = np.array([[summary["cells"][f"{c}|{s}"]["d_auroc"] for s in SEEDS] for c in CATS])
    im = ax.imshow(mat, cmap="RdBu_r", vmin=-0.02, vmax=0.04, aspect="auto")
    ax.set_xticks(range(len(SEEDS)))
    ax.set_xticklabels([f"seed{s}" for s in SEEDS])
    ax.set_yticks(range(len(CATS)))
    ax.set_yticklabels(CATS)
    for i in range(len(CATS)):
        for j in range(len(SEEDS)):
            ax.text(j, i, f"{mat[i, j]:+.3f}", ha="center", va="center", fontsize=9)
    ax.set_title("E13-B R1 dAUROC heatmap (cat x seed)")
    fig.colorbar(im, ax=ax, label="dAUROC")
    fig.tight_layout()
    fig.savefig(FIGS / fname, dpi=150)
    plt.close(fig)


def main() -> int:
    FIGS.mkdir(parents=True, exist_ok=True)
    cells = load_cells()
    summary = summarize(cells)
    collapsed_any = any(cells[(c, s)].get("collapsed", False)
                        for c in CATS for s in SEEDS)
    verdict = decide(summary, collapsed_any)

    result = {"experiment": "E13-B", "candidate": "R1", "verdict": verdict,
              "summary": summary, "collapsed": collapsed_any,
              "pass_criteria": {
                  "overall_mean_d_auroc_ge_010": summary["overall_mean_d_auroc"] >= 0.010,
                  "positive_cells_ge_7": summary["positive_auroc_cells"] >= 7,
                  "per_cat_mean_ge_005": all(summary["per_category"][c]["mean_d_auroc"] >= 0.005
                                             for c in CATS),
                  "mean_d_ap_ge_005": summary["overall_mean_d_ap"] >= 0.005},
              "hold_criteria": {
                  "overall_mean_d_auroc_ge_0075": summary["overall_mean_d_auroc"] >= 0.0075,
                  "positive_cells_ge_6": summary["positive_auroc_cells"] >= 6,
                  "per_cat_mean_ge_0": all(summary["per_category"][c]["mean_d_auroc"] >= 0
                                           for c in CATS),
                  "mean_d_ap_ge_0": summary["overall_mean_d_ap"] >= 0}}
    (OUT / "e13b_summary.json").write_text(json.dumps(result, indent=2, default=float))

    fig_by_cat_seed(summary, "auroc", "fig1_dauroc_by_cat_seed.png")
    fig_by_cat_seed(summary, "ap", "fig2_dap_by_cat_seed.png")
    fig_mean_std(summary, "fig3_mean_std.png")
    fig_heatmap(summary, "fig4_seed_consistency_heatmap.png")

    print(json.dumps({k: v for k, v in result.items() if k != "cells"}, indent=2,
                     default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
