#!/usr/bin/env python
"""E16 analysis: frozen GO/HOLD/STOP verdict + summary tables + figures (CPU-only).

Applies results/e16/protocol.md §4 verbatim. No criteria are re-tuned here.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "e16"
FIGS = OUT / "figures"

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

CATS = ["Bird", "Car", "Cube"]
SEEDS = [0, 1, 2]
BC = ["Bird", "Car"]


def load_cells():
    return json.loads((OUT / "e16_partial.json").read_text())["cells"]


def d(cell, meth, key):
    return cell[meth][key]


def main() -> int:
    FIGS.mkdir(parents=True, exist_ok=True)
    cells = load_cells()
    assert all(f"{c}|{s}" in cells for c in CATS for s in SEEDS), "matrix incomplete"
    assert all(cells[f"{c}|{s}"]["repro_ok"] for c in CATS for s in SEEDS), "repro failed"

    # ---------- deltas
    delta = {}
    for c in CATS:
        for s in SEEDS:
            cl = cells[f"{c}|{s}"]
            delta[(c, s)] = {
                "d_auroc": d(cl, "r1", "auroc") - d(cl, "f0", "auroc"),
                "d_ap": d(cl, "r1", "ap") - d(cl, "f0", "ap"),
                "d_dprime": d(cl, "r1", "d_prime") - d(cl, "f0", "d_prime"),
                "d_rob_feat": d(cl, "r1", "robustness_feat") - d(cl, "f0", "robustness_feat"),
                "d_rob_score": d(cl, "r1", "robustness_score") - d(cl, "f0", "robustness_score"),
                "rel_rob_feat": (d(cl, "r1", "robustness_feat") / d(cl, "f0", "robustness_feat") - 1),
                "rel_rob_score": (d(cl, "r1", "robustness_score") / d(cl, "f0", "robustness_score") - 1),
            }

    bc = [(c, s) for c in BC for s in SEEDS]
    bc_au = np.array([delta[k]["d_auroc"] for k in bc])
    bc_ap = np.array([delta[k]["d_ap"] for k in bc])
    cube_au = np.array([delta[("Cube", s)]["d_auroc"] for s in SEEDS])

    g1 = int((bc_au > 0).sum())
    g2 = float(bc_au.mean())
    g3 = float(bc_ap.mean())
    rob_feat_mean = float(np.mean([delta[k]["d_rob_feat"] for k in bc]))
    rob_score_mean = float(np.mean([delta[k]["d_rob_score"] for k in bc]))
    worst_rel = max([delta[k]["rel_rob_feat"] for k in bc] + [delta[k]["rel_rob_score"] for k in bc])
    g4 = bool(rob_feat_mean < 0 and rob_score_mean < 0 and worst_rel <= 0.10)
    g5 = float(cube_au.mean())

    go = (g1 >= 4) and (g2 >= 0.010) and (g3 >= 0) and g4 and (g5 >= -0.005)
    per_cat_mean = {c: float(np.mean([delta[(c, s)]["d_auroc"] for s in SEEDS])) for c in CATS}
    stable = [c for c in BC if per_cat_mean[c] >= 0.005]
    hold_reasons = []
    if 0.005 <= g2 < 0.010:
        hold_reasons.append("bird_car_mean_in_hold_band")
    if len(stable) == 1:
        hold_reasons.append("only_one_of_bird_car_stable")
    if (not g4) and rob_feat_mean <= 0 and rob_score_mean <= 0:
        hold_reasons.append("weak_robustness_signal")
    hold = bool(hold_reasons)

    # stop checks
    stop_reasons = []
    if g2 < 0.005:
        stop_reasons.append("bird_car_mean_lt_+0.005")
    for c in BC:
        vals = [delta[(c, s)]["d_auroc"] for s in SEEDS]
        if min(vals) < 0 < max(vals) and np.mean(vals) < 0.005:
            stop_reasons.append(f"{c}_sign_flip")
        srt = sorted(vals, reverse=True)
        if float(np.mean(srt[1:])) < 0.005:
            stop_reasons.append(f"{c}_single_seed_dependent")
    if g5 < -0.005:
        stop_reasons.append("cube_harmed")
    if (not g4) and g2 < 0.010 and float(np.mean([delta[k]["d_dprime"] for k in bc])) <= 0:
        stop_reasons.append("robustness_and_detection_both_not_stable")

    verdict = "GO" if go else ("STOP" if stop_reasons else ("HOLD" if hold else "STOP"))
    if go:
        stop_reasons = []

    out = {
        "experiment": "E16", "verdict": verdict,
        "go_conditions": {"1_ge4of6_positive": [g1, g1 >= 4], "2_bc_mean_dauroc": [g2, g2 >= 0.010],
                          "3_bc_mean_dap": [g3, g3 >= 0],
                          "4_robustness": [{"mean_d_rob_feat": rob_feat_mean,
                                            "mean_d_rob_score": rob_score_mean,
                                            "worst_relative": worst_rel}, g4],
                          "5_cube_mean_dauroc": [g5, g5 >= -0.005]},
        "hold_reasons": hold_reasons, "stop_reasons": stop_reasons,
        "per_category_mean_d_auroc": per_cat_mean,
        "per_category_mean_d_ap": {c: float(np.mean([delta[(c, s)]["d_ap"] for s in SEEDS]))
                                   for c in CATS},
        "bird_car": {"mean_d_auroc": g2, "std_d_auroc": float(bc_au.std(ddof=1)),
                     "mean_d_ap": g3, "positive_cells": g1, "n": 6},
        "cube": {"mean_d_auroc": g5, "std_d_auroc": float(cube_au.std(ddof=1))},
        "deltas": {f"{c}|{s}": delta[(c, s)] for c in CATS for s in SEEDS},
        "raw": {f"{c}|{s}": cells[f"{c}|{s}"][m] for c in CATS for s in SEEDS
                for m in ("f0", "r1")},
    }
    (OUT / "verdict.json").write_text(json.dumps(out, indent=2, default=float))

    # ---------- figures
    x = np.arange(3)
    for metric, fname, ylab in (("d_auroc", "fig1_delta_auroc_by_seed.png", "delta AUROC"),
                                ("d_ap", "fig2_delta_ap_by_seed.png", "delta AP")):
        fig, ax = plt.subplots(figsize=(7, 4.5))
        w = 0.25
        for j, s in enumerate(SEEDS):
            vals = [delta[(c, s)][metric] for c in CATS]
            ax.bar(x + (j - 1) * w, vals, w, label=f"seed{s}")
        ax.axhline(0, color="black", lw=0.8)
        ax.set_xticks(x); ax.set_xticklabels(CATS)
        ax.set_ylabel(ylab); ax.set_title(f"E16 R1 {ylab} by category and seed")
        ax.legend()
        fig.tight_layout(); fig.savefig(FIGS / fname, dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    means = [per_cat_mean[c] for c in CATS]
    stds = [float(np.std([delta[(c, s)]["d_auroc"] for s in SEEDS], ddof=1)) for c in CATS]
    ax.bar(CATS, means, yerr=stds, capsize=5, color=["#2c7fb8", "#d95f02", "#999999"])
    ax.axhline(0, color="black", lw=0.8)
    ax.axhline(0.010, color="green", ls="--", lw=0.8, label="GO bar (+0.010)")
    ax.axhline(0.005, color="orange", ls="--", lw=0.8, label="HOLD bar (+0.005)")
    ax.set_ylabel("mean delta AUROC (3 seeds)")
    ax.set_title("E16 R1 per-category mean +/- std")
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(FIGS / "fig3_category_mean_delta.png", dpi=150); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, key, ttl in ((axes[0], "robustness_feat", "feature-level (1-cos)"),
                         (axes[1], "robustness_score", "score-level (norm. score shift)")):
        w = 0.25
        for j, s in enumerate(SEEDS):
            f0 = [d(cells[f"{c}|{s}"], "f0", key) for c in CATS]
            r1 = [d(cells[f"{c}|{s}"], "r1", key) for c in CATS]
            ax.plot(x + (j - 1) * w, f0, "o", color=f"C{j}", alpha=0.45, ms=5)
            ax.plot(x + (j - 1) * w, r1, "o", color=f"C{j}", ms=5)
            for xi, (a, b) in enumerate(zip(f0, r1)):
                ax.annotate("", xy=(xi + (j - 1) * w, b), xytext=(xi + (j - 1) * w, a),
                            arrowprops=dict(arrowstyle="->", color=f"C{j}", lw=1))
        ax.set_xticks(x); ax.set_xticklabels(CATS)
        ax.set_title(f"E16 illumination robustness: F0 -> R1 ({ttl})")
        ax.set_ylabel("lower = more robust")
    fig.tight_layout(); fig.savefig(FIGS / "fig4_illumination_dispersion_before_after.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    w = 0.25
    for j, s in enumerate(SEEDS):
        f0 = [d(cells[f"{c}|{s}"], "f0", "d_prime") for c in CATS]
        r1 = [d(cells[f"{c}|{s}"], "r1", "d_prime") for c in CATS]
        ax.bar(x + (j - 1) * w, np.array(r1) - np.array(f0), w, label=f"seed{s}")
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(x); ax.set_xticklabels(CATS)
    ax.set_ylabel("delta d'")
    ax.set_title("E16 good-vs-defect separation (d' change)")
    ax.legend()
    fig.tight_layout(); fig.savefig(FIGS / "fig5_good_defect_separation.png", dpi=150); plt.close(fig)

    print(json.dumps({k: v for k, v in out.items() if k not in ("deltas", "raw")},
                     indent=2, default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
