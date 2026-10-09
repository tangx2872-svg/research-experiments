#!/usr/bin/env python
"""E7-1 / E7-2 figures (CPU-only)."""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

E71 = ROOT / "results" / "e7_1"
E72 = ROOT / "results" / "e7_2"
COL = {"ADVANCE": "tab:green", "KEEP": "tab:olive", "BACKUP": "tab:orange",
       "STOP": "tab:red", "REFERENCE": "black"}


def fig_e71() -> None:
    rows = list(csv.DictReader(open(E71 / "summary.csv")))
    rows = [r for r in rows if r["image_auroc"] not in ("nan", "")]
    rows.sort(key=lambda r: -float(r["image_auroc"]))
    ids = [r["pipeline"] for r in rows]
    au = np.array([float(r["image_auroc"]) for r in rows])
    da = np.array([float(r["d_auroc"]) for r in rows])
    rt = np.array([float(r["runtime_s"]) for r in rows])
    col = [COL.get(r["decision"], "grey") for r in rows]

    fig, ax = plt.subplots(figsize=(11, 5.0))
    b = ax.bar(ids, au, color=col, edgecolor="black")
    base = float([r for r in rows if r["pipeline"] == "P00"][0]["image_auroc"])
    ax.axhline(base, ls="--", color="black", lw=1.2, label=f"P00 baseline {base:.5f}")
    for bb, v, d in zip(b, au, da):
        ax.text(bb.get_x() + bb.get_width() / 2, v + 0.002, f"{v:.4f}\n{d:+.4f}",
                ha="center", fontsize=7)
    ax.set_ylim(min(au) - 0.02, max(au) + 0.045)
    ax.set_ylabel("image AUROC (primary)")
    ax.set_title("E7-1 Figure 1 - pipeline ranking by image AUROC\n"
                 "green=ADVANCE olive=KEEP orange=BACKUP red=STOP")
    ax.legend(fontsize=8); ax.grid(alpha=0.3, axis="y")
    plt.xticks(rotation=30, ha="right"); fig.tight_layout()
    fig.savefig(E71 / "figures" / "fig1_pipeline_auroc_ranking.png", dpi=150); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    o = np.argsort(da)
    axes[0].bar([ids[i] for i in o], da[o], color=[col[i] for i in o], edgecolor="black")
    axes[0].axhline(0.020, ls="--", color="green", label="ADVANCE +0.020")
    axes[0].axhline(0.010, ls="--", color="olive", label="KEEP +0.010")
    axes[0].axhline(0.0, color="black", lw=1)
    axes[0].set_ylabel("dAUROC vs P00"); axes[0].legend(fontsize=8)
    axes[0].set_title("E7-1 Figure 2a - dAUROC (tiers)")
    axes[0].grid(alpha=0.3, axis="y"); axes[0].tick_params(axis="x", rotation=45, labelsize=7)

    o2 = np.argsort(rt)
    axes[1].bar([ids[i] for i in o2], rt[o2], color=[col[i] for i in o2], edgecolor="black")
    axes[1].set_ylabel("runtime (s)"); axes[1].set_title("E7-1 Figure 2b - runtime per pipeline")
    axes[1].grid(alpha=0.3, axis="y"); axes[1].tick_params(axis="x", rotation=45, labelsize=7)
    fig.tight_layout(); fig.savefig(E71 / "figures" / "fig2_delta_and_runtime.png", dpi=150)
    plt.close(fig)

    syn = list(csv.DictReader(open(E71 / "synergy.csv")))
    if syn:
        fig, ax = plt.subplots(figsize=(10.5, 4.8))
        labels, av, bv, abv = [], [], [], []
        for s in syn:
            labels.append(f"{s['A']}\n+\n{s['B']}\n=\n{s['A_plus_B']}")
            av.append(float(s["d_auroc_A"])); bv.append(float(s["d_auroc_B"]))
            abv.append(float(s["d_auroc_AB"]))
        x = np.arange(len(labels)); w = 0.26
        ax.bar(x - w, av, w, label="A alone", color="tab:blue", edgecolor="black")
        ax.bar(x, bv, w, label="B alone", color="tab:cyan", edgecolor="black")
        ax.bar(x + w, abv, w, label="A+B", color="tab:purple", edgecolor="black")
        for i, s in enumerate(syn):
            ax.text(i, max(av[i], bv[i], abv[i]) + 0.008, f"syn {float(s['synergy_gain']):+.4f}",
                    ha="center", fontsize=7.5,
                    color="green" if s["positive_composition"] == "True" else "red")
        ax.axhline(0, color="black", lw=1)
        ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=7)
        ax.set_ylabel("dAUROC vs P00")
        ax.set_title("E7-1 Figure 3 - composition synergy (A, B, A+B, and the synergy gain)")
        ax.legend(fontsize=8); ax.grid(alpha=0.3, axis="y"); fig.tight_layout()
        fig.savefig(E71 / "figures" / "fig3_synergy.png", dpi=150); plt.close(fig)


def fig_e72() -> None:
    p = E72 / "cross_view.json"
    if not p.exists():
        return
    rows = json.loads(p.read_text())
    views = list(json.loads((E72 / "progress.json").read_text())["views"])
    base = [json.loads((E72 / "raw" / f"P00_view{v}.json").read_text())["metrics"]["image_auroc"]
            for v in views]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    x = np.arange(len(views)); w = 0.35
    axes[0].plot(x, base, marker="s", lw=2, color="black", label="P00 baseline")
    for r in rows:
        axes[0].plot(x, r["view_auroc"], marker="o", lw=1.5, label=f"{r['pipeline']} ({r['verdict'][:8]})")
    axes[0].set_xticks(x); axes[0].set_xticklabels([f"view {v}" for v in views])
    axes[0].set_ylabel("image AUROC"); axes[0].legend(fontsize=8)
    axes[0].set_title("E7-2 Figure 1 - per-view image AUROC")
    axes[0].grid(alpha=0.3)
    for r in rows:
        axes[1].plot(x, r["view_d_auroc"], marker="o", lw=1.6,
                     label=f"{r['pipeline']} mean {r['mean_d_auroc']:+.4f} "
                           f"({r['positive_views']}/{r['n_views']} positive)")
    axes[1].axhline(0, color="black", lw=1)
    axes[1].axhline(0.015, ls="--", color="green", label="Path A +0.015")
    axes[1].axhline(0.010, ls="--", color="olive", label="Path B +0.010")
    axes[1].set_xticks(x); axes[1].set_xticklabels([f"view {v}" for v in views])
    axes[1].set_ylabel("dAUROC vs same-view P00"); axes[1].legend(fontsize=7)
    axes[1].set_title("E7-2 Figure 2 - per-view delta and finalist bars")
    axes[1].grid(alpha=0.3)
    fig.tight_layout(); (E72 / "figures").mkdir(exist_ok=True)
    fig.savefig(E72 / "figures" / "fig1_2_cross_view.png", dpi=150); plt.close(fig)


if __name__ == "__main__":
    (E71 / "figures").mkdir(exist_ok=True)
    fig_e71()
    fig_e72()
    for d in (E71 / "figures", E72 / "figures"):
        for f in sorted(Path(d).glob("*.png")):
            print(" ", f.relative_to(ROOT), f"{f.stat().st_size/1024:.0f} KB")
