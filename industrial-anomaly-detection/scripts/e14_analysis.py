#!/usr/bin/env python
"""E14 family screening analysis (CPU-only)."""
from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "e14"
FIGS = OUT / "figures"

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

CATS = ["Bird", "Car", "Cube"]
FAMS = ["F0", "F1", "F2"]


def load():
    rows = {}
    with open(OUT / "e14_screening.csv") as f:
        for r in csv.DictReader(f):
            if r["auroc"] == "":
                rows[(r["category"], r["family"])] = None
            else:
                rows[(r["category"], r["family"])] = {
                    "auroc": float(r["auroc"]), "ap": float(r["ap"]),
                    "d_auroc": float(r["d_auroc"]), "d_ap": float(r["d_ap"])}
    return rows


def main() -> int:
    FIGS.mkdir(parents=True, exist_ok=True)
    rows = load()

    # summary
    summary = {}
    for fam in ["F1", "F2"]:
        d_au = [rows[(c, fam)]["d_auroc"] for c in CATS if rows.get((c, fam))]
        d_ap = [rows[(c, fam)]["d_ap"] for c in CATS if rows.get((c, fam))]
        summary[fam] = {
            "mean_d_auroc_bird_car": float(np.mean(
                [rows[(c, fam)]["d_auroc"] for c in ("Bird", "Car")])),
            "mean_d_ap_bird_car": float(np.mean(
                [rows[(c, fam)]["d_ap"] for c in ("Bird", "Car")])),
            "bird_d_auroc": rows[("Bird", fam)]["d_auroc"],
            "car_d_auroc": rows[("Car", fam)]["d_auroc"],
            "cube_run": rows.get(("Cube", fam)) is not None,
        }

    verdicts = {"F1": "STOP", "F2": "STOP"}  # both fail Stage A (any-cat dAUROC < -0.015)
    result = {"experiment": "E14", "overall_verdict": "STOP",
              "verdicts": verdicts, "summary": summary,
              "note": "Both F1/F2 fail Stage A (Bird dAUROC -0.045/-0.103, both < -0.015); no family reaches Cube (Stage B)."}
    (OUT / "e14_summary.json").write_text(json.dumps(result, indent=2, default=float))

    # fig1 family auroc
    fig, ax = plt.subplots(figsize=(7, 4.5))
    x = np.arange(len(CATS)); w = 0.25
    for j, fam in enumerate(FAMS):
        vals = [rows[(c, fam)]["auroc"] if rows.get((c, fam)) else np.nan for c in CATS]
        ax.bar(x + (j - 1) * w, vals, w, label=fam)
    ax.set_xticks(x); ax.set_xticklabels(CATS)
    ax.set_ylabel("AUROC"); ax.set_title("E14 family AUROC")
    ax.legend()
    fig.tight_layout(); fig.savefig(FIGS / "fig1_family_auroc.png", dpi=150); plt.close(fig)

    # fig2 family AP
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for j, fam in enumerate(FAMS):
        vals = [rows[(c, fam)]["ap"] if rows.get((c, fam)) else np.nan for c in CATS]
        ax.bar(x + (j - 1) * w, vals, w, label=fam)
    ax.set_xticks(x); ax.set_xticklabels(CATS)
    ax.set_ylabel("AP"); ax.set_title("E14 family AP")
    ax.legend()
    fig.tight_layout(); fig.savefig(FIGS / "fig2_family_ap.png", dpi=150); plt.close(fig)

    # fig3 delta by category
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for fam in ["F1", "F2"]:
        vals = [rows[(c, fam)]["d_auroc"] if rows.get((c, fam)) else np.nan for c in CATS]
        ax.plot(CATS, vals, "-o", label=fam)
    ax.axhline(0, color="black", lw=0.8)
    ax.axhline(-0.015, color="red", ls="--", lw=0.8, label="Stage A lower bound")
    ax.set_ylabel("dAUROC vs F0"); ax.set_title("E14 delta AUROC (F1/F2 below F0)")
    ax.legend()
    fig.tight_layout(); fig.savefig(FIGS / "fig3_delta_by_category.png", dpi=150); plt.close(fig)

    # fig4 cube ceiling comparison (Cube not reached -> only F0 shown)
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.bar(["F0"], [rows[("Cube", "F0")]["auroc"]], color="#2c7fb8")
    ax.set_ylabel("Cube AUROC"); ax.set_title("E14 Cube: F1/F2 did not reach Stage B")
    ax.text(0, rows[("Cube", "F0")]["auroc"] - 0.05, f"{rows[('Cube','F0')]['auroc']:.4f}",
            ha="center")
    fig.tight_layout(); fig.savefig(FIGS / "fig4_cube_ceiling_comparison.png", dpi=150); plt.close(fig)

    # fig5 runtime vs gain
    fig, ax = plt.subplots(figsize=(6, 4.5))
    rt = {"F0": 0, "F1": 1.5, "F2": 4.0}  # approx min/category
    gain = {"F0": 0.0,
            "F1": summary["F1"]["mean_d_auroc_bird_car"],
            "F2": summary["F2"]["mean_d_auroc_bird_car"]}
    for fam in FAMS:
        ax.scatter(rt[fam], gain[fam], s=90, label=fam)
        ax.annotate(fam, (rt[fam], gain[fam]), textcoords="offset points", xytext=(6, 6))
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xlabel("approx runtime (min/category)"); ax.set_ylabel("mean dAUROC (Bird/Car)")
    ax.set_title("E14 runtime vs gain (F0 dominates)")
    fig.tight_layout(); fig.savefig(FIGS / "fig5_runtime_vs_gain.png", dpi=150); plt.close(fig)

    print(json.dumps(result, indent=2, default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
