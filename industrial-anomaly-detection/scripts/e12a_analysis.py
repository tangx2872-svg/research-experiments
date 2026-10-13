#!/usr/bin/env python
"""E12-A analysis: summary + figures (category x config dAUROC, mean dAUROC/dAP)."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "e12a"
CSV = OUT / "e12a_screening.csv"
FIG = OUT / "figures"
FIG.mkdir(parents=True, exist_ok=True)

CONFIGS = ["S1-AGG-1", "S1-AGG-2", "S2-AGG-1", "S2-AGG-2", "S3-AGG-1", "S3-AGG-2"]
CONFIG_LABEL = {"S1-AGG-1": "S1\nmax", "S1-AGG-2": "S1\ntop-1%",
                "S2-AGG-1": "S2\nmax", "S2-AGG-2": "S2\ntop-1%",
                "S3-AGG-1": "S3\nmax", "S3-AGG-2": "S3\ntop-1%"}
CATS = ["Bird", "Car", "Cube"]
COLORS = {"Bird": "#4C72B0", "Car": "#DD8452", "Cube": "#55A868"}


def main() -> int:
    rows = list(csv.DictReader(open(CSV)))
    cells = {}
    for r in rows:
        if r["config"] == "S0":
            continue
        cells[(r["category"], r["config"])] = {"d_auroc": float(r["d_auroc"]),
                                               "d_ap": float(r["d_ap"]),
                                               "auroc": float(r["auroc"]),
                                               "ap": float(r["ap"])}

    # ---- summary per config
    summary = {}
    for cfg in CONFIGS:
        ds = [cells[(c, cfg)]["d_auroc"] for c in CATS]
        aps = [cells[(c, cfg)]["d_ap"] for c in CATS]
        summary[cfg] = {
            "mean_d_auroc": round(float(np.mean(ds)), 5),
            "mean_d_ap": round(float(np.mean(aps)), 5),
            "positive_categories": int(sum(1 for x in ds if x > 0)),
            "worst_category": CATS[int(np.argmin(ds))],
            "worst_d_auroc": round(float(np.min(ds)), 5),
            "per_category_d_auroc": {c: round(d, 5) for c, d in zip(CATS, ds)},
        }

    # ---- figure 1: category x config dAUROC (grouped bars)
    x = np.arange(len(CONFIGS))
    w = 0.28
    fig, ax = plt.subplots(figsize=(10, 5.5))
    for i, cat in enumerate(CATS):
        vals = [cells[(cat, cfg)]["d_auroc"] for cfg in CONFIGS]
        ax.bar(x + (i - 1) * w, vals, w, label=cat, color=COLORS[cat])
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([CONFIG_LABEL[c] for c in CONFIGS], fontsize=8)
    ax.set_ylabel(r"$\Delta$AUROC vs S0 (1-NN-max)")
    ax.set_title("E12-A · PCA/subspace residual scoring vs 1-NN-max\n(frozen DINOv2-B, view120, seed0, 300-bank)")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG / "e12a_dauroc_by_category.png", dpi=150)
    plt.close(fig)

    # ---- figure 2: mean dAUROC / dAP per config
    mean_au = [summary[c]["mean_d_auroc"] for c in CONFIGS]
    mean_ap = [summary[c]["mean_d_ap"] for c in CONFIGS]
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - 0.2, mean_au, 0.4, label="mean ΔAUROC", color="#4C72B0")
    ax.bar(x + 0.2, mean_ap, 0.4, label="mean ΔAP", color="#DD8452")
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([CONFIG_LABEL[c] for c in CONFIGS], fontsize=8)
    ax.set_ylabel("mean Δ (vs S0)")
    ax.set_title("E12-A · mean ΔAUROC / ΔAP per configuration (3 categories)")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG / "e12a_mean_delta.png", dpi=150)
    plt.close(fig)

    (OUT / "e12a_summary.json").write_text(json.dumps(summary, indent=2))
    for cfg in CONFIGS:
        s = summary[cfg]
        print(f"{cfg:10s} mean_dAUROC={s['mean_d_auroc']:+.5f} mean_dAP={s['mean_d_ap']:+.5f} "
              f"pos={s['positive_categories']}/3 worst={s['worst_category']} "
              f"({s['worst_d_auroc']:+.5f})", flush=True)
    print(f"\n[E12-A] figures -> {FIG}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
