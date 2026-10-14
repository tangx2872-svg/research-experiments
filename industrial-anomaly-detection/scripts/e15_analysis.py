#!/usr/bin/env python
"""E15 EfficientAD feasibility analysis (CPU-only)."""
from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "e15"
FIGS = OUT / "figures"

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

F0 = {"Bird": [0.74725, 0.78863], "Car": [0.78552, 0.84207], "Cube": [0.85410, 0.91686]}
CATS = ["Bird", "Car", "Cube"]


def main() -> int:
    FIGS.mkdir(parents=True, exist_ok=True)
    cells = {}
    with open(OUT / "e15_metrics.csv") as f:
        for r in csv.DictReader(f):
            if r["family"] == "EfficientAD" and r["auroc"] != "":
                cells[r["category"]] = {"auroc": float(r["auroc"]), "ap": float(r["ap"]),
                                        "d_auroc": float(r["d_auroc"]), "d_ap": float(r["d_ap"])}

    result = {"experiment": "E15", "overall_verdict": "STOP",
              "verdicts": {"EfficientAD": "STOP"},
              "cells": cells,
              "note": "Bird dAUROC -0.135 < -0.020 -> STOP per Stage A rule; Car/Cube not run."}
    (OUT / "e15_summary.json").write_text(json.dumps(result, indent=2, default=float))

    # fig1/fig2 auroc/ap comparison (Bird has EfficientAD; Car/Cube only F0)
    for metric, fname in (("auroc", "fig1_auroc_comparison.png"), ("ap", "fig2_ap_comparison.png")):
        fig, ax = plt.subplots(figsize=(6, 4.5))
        x = np.arange(len(CATS)); w = 0.35
        f0 = [F0[c][0 if metric == "auroc" else 1] for c in CATS]
        ea = [cells[c][metric] if c in cells else np.nan for c in CATS]
        ax.bar(x - w / 2, f0, w, label="F0 (DINOv2-B)")
        ax.bar(x + w / 2, ea, w, label="EfficientAD")
        ax.set_xticks(x); ax.set_xticklabels(CATS)
        ax.set_ylabel(metric.upper())
        ax.set_title(f"E15 {metric.upper()} (EfficientAD only Bird)")
        ax.legend()
        fig.tight_layout(); fig.savefig(FIGS / fname, dpi=150); plt.close(fig)

    # fig3 delta by category
    fig, ax = plt.subplots(figsize=(6, 4.5))
    d_au = [cells[c]["d_auroc"] if c in cells else np.nan for c in CATS]
    d_ap = [cells[c]["d_ap"] if c in cells else np.nan for c in CATS]
    ax.bar(x - 0.18, d_au, 0.36, label="dAUROC", color="#2c7fb8")
    ax.bar(x + 0.18, d_ap, 0.36, label="dAP", color="#d95f02")
    ax.axhline(0, color="black", lw=0.8)
    ax.axhline(-0.020, color="red", ls="--", lw=0.8, label="STOP threshold")
    ax.set_xticks(x); ax.set_xticklabels(CATS)
    ax.set_ylabel("delta vs F0")
    ax.set_title("E15 delta (Bird far below STOP threshold)")
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(FIGS / "fig3_delta_by_category.png", dpi=150); plt.close(fig)

    # fig4 cube ceiling (not reached)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(["F0"], [F0["Cube"][0]], color="#2c7fb8")
    ax.set_ylabel("Cube AUROC"); ax.set_title("E15 Cube: EfficientAD did not reach Stage C")
    fig.tight_layout(); fig.savefig(FIGS / "fig4_cube_ceiling.png", dpi=150); plt.close(fig)

    # fig5 runtime vs performance
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.scatter(0, 0, s=100, label="F0 (0 min)")
    ax.scatter(29.2, cells["Bird"]["d_auroc"], s=100, label="EfficientAD Bird (29 min)")
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xlabel("train time (min)"); ax.set_ylabel("dAUROC vs F0")
    ax.set_title("E15 runtime vs performance (EfficientAD slow + worse)")
    ax.legend()
    fig.tight_layout(); fig.savefig(FIGS / "fig5_runtime_vs_performance.png", dpi=150); plt.close(fig)

    # fig6 example anomaly maps (read 2 tiff from Bird output)
    import tifffile
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    base = OUT / "efficientad_output" / "Bird_s70000" / "anomaly_maps" / "mvtec_ad" / "Bird" / "test"
    for ax, cls in zip(axes, ("good", "anomaly")):
        d = base / cls
        files = sorted(d.glob("*.tiff")) if d.exists() else []
        if files:
            m = tifffile.imread(files[0])
            ax.imshow(m, cmap="jet")
            ax.set_title(f"{cls} map (max={np.max(m):.3f})")
        else:
            ax.set_title(f"{cls} (no map)")
        ax.axis("off")
    fig.suptitle("E15 EfficientAD Bird example anomaly maps")
    fig.tight_layout(); fig.savefig(FIGS / "fig6_example_anomaly_maps.png", dpi=150); plt.close(fig)

    print(json.dumps(result, indent=2, default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
