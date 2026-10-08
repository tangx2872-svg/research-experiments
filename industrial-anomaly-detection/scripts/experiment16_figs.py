"""Experiment 16 — P9 Figures 1-5（论文可读命名；PNG + PDF）。全 CPU。"""
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
sys.path.insert(0, str(ROOT / "scripts"))
import experiment14_metrics as M  # noqa: E402

E16 = ROOT / "results" / "experiment_16_final_validation"
ANA, FIG = E16 / "analysis", E16 / "figures"
METHODS = ["Original", "Adaptive B2", "C2", "C6", "X6c"]
COLOR = {"Original": "#7F7F7F", "Adaptive B2": "#C44E52", "C2": "#55A868", "C6": "#8172B2", "X6c": "#4C72B0"}


def save(fig, name):
    fig.savefig(FIG / f"{name}.png", dpi=180, bbox_inches="tight")
    fig.savefig(FIG / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    rows = list(csv.DictReader(open(ANA / "per_unit_50.csv")))
    tab = {r["method"]: r for r in csv.DictReader(open(ANA / "final_50unit_table.csv"))}
    blk = list(csv.DictReader(open(ANA / "seed_block_stability.csv")))
    stat = json.loads((ANA / "statistics.json").read_text())
    n = len(rows)

    KM = {"Adaptive B2": "B2"}          # per_unit CSV 中 B2 的字段前缀

    def arr(m, f):
        return np.array([float(r[f"{KM.get(m, m)}_{f}"]) for r in rows]) if m != "Original" else None

    dd = {m: (arr(m, "ddp") if m != "Original" else np.zeros(n)) for m in METHODS}

    # ---- Fig1 PASS + catastrophic ----
    fig, axs = plt.subplots(1, 2, figsize=(12.4, 4.4))
    xs = np.arange(len(METHODS)); w = 0.38
    axs[0].bar(xs, [int(tab[m]["PASS"]) for m in METHODS], color=[COLOR[m] for m in METHODS],
               edgecolor="black", lw=0.6)
    for i, m in enumerate(METHODS):
        axs[0].text(i, int(tab[m]["PASS"]) + 0.4, "%d/%d\n%.0f%%" % (int(tab[m]["PASS"]), n,
                    100 * float(tab[m]["pass_rate"])), ha="center", fontsize=8)
    axs[0].set_xticks(xs); axs[0].set_xticklabels(METHODS); axs[0].set_ylim(0, n * 1.15)
    axs[0].set_ylabel("PASS count (of %d units)" % n)
    axs[0].set_title("(a) PASS rate  (criterion: ΔR ≤ −0.02 and Δd′ ≥ −0.10 vs same-unit Original)", fontsize=9)
    axs[0].grid(alpha=0.3, axis="y")
    axs[1].bar(xs, [int(tab[m]["catastrophic"]) for m in METHODS], color=[COLOR[m] for m in METHODS],
               edgecolor="black", lw=0.6)
    for i, m in enumerate(METHODS):
        v = int(tab[m]["catastrophic"])
        axs[1].text(i, v + 0.06, "%d (%.1f%%)" % (v, 100 * float(tab[m]["cat_rate"])), ha="center", fontsize=8)
    axs[1].set_xticks(xs); axs[1].set_xticklabels(METHODS); axs[1].set_ylim(0, 6)
    axs[1].set_ylabel("catastrophic count (Δd′ ≤ −0.25)")
    axs[1].set_title("(b) catastrophic failures", fontsize=9); axs[1].grid(alpha=0.3, axis="y")
    fig.suptitle("Exp16 Fig1 — Final validation on 5 MVTec categories × 10 seeds (50 units)", fontsize=10.5)
    fig.tight_layout(rect=(0, 0, 1, 0.93)); save(fig, "fig1_pass_catastrophic")

    # ---- Fig2 Δd′ distribution ----
    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    bins = np.linspace(-0.55, 1.0, 32)
    for m in METHODS:
        if m == "Original":
            continue
        ax.hist(dd[m], bins=bins, histtype="step", lw=1.8, label="%s (n=%d)" % (m, n), color=COLOR[m])
    ax.axvline(-0.25, ls="--", c="black", lw=1.2, label="catastrophic threshold (−0.25)")
    ax.axvline(-0.10, ls=":", c="gray", lw=1.2, label="PASS preservation bound (−0.10)")
    ax.set_xlabel("Δd′ per unit (vs same-unit Original)")
    ax.set_ylabel("count of units"); ax.grid(alpha=0.3); ax.legend(fontsize=8)
    ax.set_title("Exp16 Fig2 — defect-preservation change distribution (50 units)", fontsize=10)
    fig.tight_layout(); save(fig, "fig2_ddp_distribution")

    # ---- Fig3 worst-tail ----
    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    kk = np.arange(1, 11)
    for m in METHODS:
        if m == "Original":
            continue
        s = np.sort(dd[m])
        ax.plot(kk, s[:10], "-o", ms=4, color=COLOR[m], label=m)
    ax.axhline(-0.25, ls="--", c="black", lw=1.2, label="catastrophic threshold")
    ax.set_xlabel("rank (1 = worst unit)"); ax.set_ylabel("Δd′ (worst-first)")
    ax.set_title("Exp16 Fig3 — worst-tail comparison (the decision-relevant tail)", fontsize=10)
    ax.grid(alpha=0.3); ax.legend(fontsize=8); fig.tight_layout(); save(fig, "fig3_worst_tail")

    # ---- Fig4 robustness–preservation frontier ----
    fig, axs = plt.subplots(1, 2, figsize=(12.8, 5.0))
    for m in METHODS:
        if m == "Original":
            axs[0].scatter(0, 0, s=170, marker="*", c=COLOR[m], edgecolor="black", zorder=5, label="Original")
            continue
        axs[0].scatter(float(tab[m]["mean_dR"]), float(tab[m]["mean_ddp"]), s=90, c=COLOR[m],
                       edgecolor="black", lw=0.6, zorder=4)
        axs[0].annotate(m, (float(tab[m]["mean_dR"]), float(tab[m]["mean_ddp"])),
                        textcoords="offset points", xytext=(6, 5), fontsize=8.5)
        axs[1].scatter(int(tab[m]["PASS"]), float(tab[m]["worst_ddp"]), s=90, c=COLOR[m],
                       edgecolor="black", lw=0.6, zorder=4)
        axs[1].annotate(m, (int(tab[m]["PASS"]), float(tab[m]["worst_ddp"])),
                        textcoords="offset points", xytext=(6, -10), fontsize=8.5)
    axs[0].axhline(0, ls=":", c="gray", lw=1); axs[0].axvline(0, ls=":", c="gray", lw=1)
    axs[0].set_xlabel("mean ΔR  (negative = more robust than Original)")
    axs[0].set_ylabel("mean Δd′  (positive = better preservation)")
    axs[0].set_title("(a) mean-level trade-off plane", fontsize=10); axs[0].grid(alpha=0.3); axs[0].legend(fontsize=8)
    axs[1].axhline(-0.25, ls="--", c="black", lw=1.2, label="catastrophic threshold")
    axs[1].set_xlabel("PASS count (of %d)" % n); axs[1].set_ylabel("worst Δd′")
    axs[1].set_title("(b) PASS vs worst-case safety", fontsize=10); axs[1].grid(alpha=0.3); axs[1].legend(fontsize=8)
    fig.suptitle("Exp16 Fig4 — robustness–preservation frontier and downside decision view", fontsize=10.5)
    fig.tight_layout(rect=(0, 0, 1, 0.93)); save(fig, "fig4_robustness_preservation_frontier")

    # ---- Fig5 seed-block stability ----
    labels = [b["block"] for b in blk]
    fig, axs = plt.subplots(1, 2, figsize=(13.0, 4.6))
    xs = np.arange(len(labels)); w = 0.16
    for i, m in enumerate(METHODS):
        vals = [float(b[f"{m}_PASS"].split("/")[0]) / float(b[f"{m}_PASS"].split("/")[1]) for b in blk]
        axs[0].bar(xs + (i - 2) * w, vals, w, label=m, color=COLOR[m], edgecolor="black", lw=0.5)
    axs[0].set_xticks(xs); axs[0].set_xticklabels(labels, fontsize=8)
    axs[0].set_ylabel("PASS rate within block"); axs[0].legend(fontsize=8); axs[0].grid(alpha=0.3, axis="y")
    axs[0].set_title("(a) PASS rate by seed block (D = unseen seeds 7–9)", fontsize=10)
    for i, m in enumerate(METHODS):
        vals = [float(b[f"{m}_worst_ddp"]) for b in blk]
        axs[1].plot(xs, vals, "-o", color=COLOR[m], label=m, ms=4)
    axs[1].axhline(-0.25, ls="--", c="black", lw=1.1, label="catastrophic threshold")
    axs[1].set_xticks(xs); axs[1].set_xticklabels(labels, fontsize=8)
    axs[1].set_ylabel("worst Δd′ within block"); axs[1].legend(fontsize=8); axs[1].grid(alpha=0.3)
    axs[1].set_title("(b) worst-case Δd′ by seed block", fontsize=10)
    fig.suptitle("Exp16 Fig5 — seed-block stability (selection-effect check on unseen seeds)", fontsize=10.5)
    fig.tight_layout(rect=(0, 0, 1, 0.93)); save(fig, "fig5_seed_block_stability")
    print("figures ->", FIG)


if __name__ == "__main__":
    main()
