"""Experiment 15 — 图表（Fig1 敏感度 / Fig2 leaderboard / Fig3 Pareto+downside / Fig4 per-seed / Fig5 bootstrap CI）。CPU-only。"""
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
E15 = ROOT / "results" / "experiment_15"
ANA, FIG = E15 / "analysis", E15 / "figures"

TOP = ["X6c", "C6", "C2", "X5", "C1"]


def load(p):
    return list(csv.DictReader(open(p))) if p.exists() else []


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    sens = json.loads((ANA / "sensitivity_curve.json").read_text())
    summ = json.loads((ANA / "tournament.json").read_text())["leaderboard"]
    boot = json.loads((ANA / "bootstrap_ci.json").read_text())
    rows = load(ANA / "scoreside_per_unit.csv")
    for r in rows:
        r["seed"] = int(r["seed"])

    # ---- Fig1 sensitivity curve ----
    fig, ax = plt.subplots(figsize=(7.6, 4.4))
    w = [x["w_orig"] for x in sens]
    ax.plot(w, [x["PASS"] for x in sens], "-o", label="PASS / 25 units")
    ax.plot(w, [x["catastrophic"] for x in sens], "-s", c="#C44E52", label="catastrophic")
    ax.axhline(19, ls=":", c="#C44E52", lw=1.2, label="Adaptive B2 PASS = 19")
    ax.set_xlabel("w_orig  (0 = pure B2,  1 = pure Original)")
    ax.set_ylabel("PASS count (35 units: 5 cats × seeds 0-6)")
    ax.set_title("Exp15 Fig1 — fusion-weight sensitivity (frozen coarse grid, no refinement)", fontsize=10)
    ax.grid(alpha=0.3); ax.legend(fontsize=8)
    ax2 = ax.twinx()
    ax2.plot(w, [x["worst_ddp"] for x in sens], "--^", c="#55A868", label="worst Δd′")
    ax2.axhline(-0.25, ls="--", c="black", lw=1.0)
    ax2.set_ylabel("worst Δd′", color="#55A868")
    ax2.legend(fontsize=8, loc="center right")
    fig.tight_layout(); fig.savefig(FIG / "fig1_sensitivity.png", dpi=170, bbox_inches="tight"); plt.close(fig)

    # ---- Fig2 leaderboard ----
    lb = [r for r in summ if r["n_units"] >= 15]
    lb.sort(key=lambda r: (-r["PASS"], r["catastrophic"], -r["worst_ddp"]))
    lb = lb[:8] + [{"method": "Adaptive B2 (ref)", "PASS": 19, "n_units": 25, "catastrophic": 3,
                    "worst_ddp": -0.41, "evidence": "E3"}]
    fig, axs = plt.subplots(1, 2, figsize=(13.2, 4.5))
    xs = np.arange(len(lb)); wd = 0.38
    axs[0].bar(xs - wd / 2, [r["PASS"] for r in lb], wd, label="PASS", edgecolor="black", lw=0.5)
    axs[0].bar(xs + wd / 2, [r["catastrophic"] for r in lb], wd, color="#C44E52", label="catastrophic",
               edgecolor="black", lw=0.5)
    axs[0].set_xticks(xs)
    axs[0].set_xticklabels([f"{r['method']}\n({r.get('evidence','')})" for r in lb], fontsize=7)
    axs[0].legend(fontsize=8); axs[0].set_title("(a) PASS and catastrophic count", fontsize=10)
    axs[0].grid(alpha=0.3, axis="y")
    axs[1].bar(xs, [r["worst_ddp"] for r in lb], edgecolor="black", lw=0.5, color="#55A868")
    axs[1].axhline(-0.25, ls="--", c="black", lw=1.1, label="catastrophic threshold")
    axs[1].set_xticks(xs); axs[1].set_xticklabels([r["method"] for r in lb], fontsize=7)
    axs[1].set_ylabel("worst Δd′"); axs[1].legend(fontsize=8)
    axs[1].set_title("(b) worst-case downside", fontsize=10); axs[1].grid(alpha=0.3, axis="y")
    fig.suptitle("Exp15 Fig2 — unified leaderboard (score-side E3, feature-side as-run)", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.93)); fig.savefig(FIG / "fig2_leaderboard.png", dpi=170,
                                                         bbox_inches="tight"); plt.close(fig)

    # ---- Fig3 Pareto + downside ----
    fig, axs = plt.subplots(1, 2, figsize=(13.2, 5.0))
    for r in summ:
        if r["n_units"] < 15:
            continue
        c = "#C44E52" if r["method"].startswith("Adaptive B2") else "#4C72B0"
        axs[0].scatter(r["mean_dR"], r["mean_ddp"], s=70, c=c, edgecolor="black", lw=0.5)
        axs[0].annotate(r["method"], (r["mean_dR"], r["mean_ddp"]), textcoords="offset points",
                        xytext=(5, 4), fontsize=7.5)
        axs[1].scatter(r["PASS"], r["worst_ddp"], s=70, c=c, edgecolor="black", lw=0.5)
        axs[1].annotate(r["method"], (r["PASS"], r["worst_ddp"]), textcoords="offset points",
                        xytext=(5, -9), fontsize=7.5)
    axs[0].scatter(0, 0, s=140, marker="*", c="black", label="Original (reference)")
    axs[0].set_xlabel("mean ΔR (lower = more robust gain)"); axs[0].set_ylabel("mean Δd′ (higher = better preservation)")
    axs[0].set_title("(a) trade-off plane", fontsize=10); axs[0].grid(alpha=0.3); axs[0].legend(fontsize=8)
    axs[1].axhline(-0.25, ls="--", c="black", lw=1.1, label="catastrophic threshold")
    axs[1].set_xlabel("PASS count"); axs[1].set_ylabel("worst Δd′")
    axs[1].set_title("(b) downside decision view", fontsize=10); axs[1].grid(alpha=0.3); axs[1].legend(fontsize=8)
    fig.suptitle("Exp15 Fig3 — Pareto / downside (E3 = 25 units unless noted)", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.93)); fig.savefig(FIG / "fig3_pareto_downside.png", dpi=170,
                                                         bbox_inches="tight"); plt.close(fig)

    # ---- Fig4 per-seed stability ----
    seeds = sorted({r["seed"] for r in rows})
    methods = [m for m in TOP if any(r["method"] == m for r in rows)]
    Mx = np.full((len(methods), len(seeds)), np.nan)
    for i, m in enumerate(methods):
        for j, sd in enumerate(seeds):
            v = [r for r in rows if r["method"] == m and r["seed"] == sd]
            if v:
                Mx[i, j] = sum(1 for r in v if str(r["PASS"]) == "True") / len(v)
    fig, ax = plt.subplots(figsize=(1.05 * len(seeds) + 3.4, 0.62 * len(methods) + 2.2))
    im = ax.imshow(Mx, vmin=0, vmax=1, cmap="RdYlGn", aspect="auto")
    ax.set_xticks(range(len(seeds))); ax.set_xticklabels([f"s{s}" for s in seeds], fontsize=8)
    ax.set_yticks(range(len(methods))); ax.set_yticklabels(methods, fontsize=8)
    for i in range(len(methods)):
        for j in range(len(seeds)):
            if not np.isnan(Mx[i, j]):
                ax.text(j, i, "%.0f%%" % (100 * Mx[i, j]), ha="center", va="center", fontsize=7)
    fig.colorbar(im, ax=ax, fraction=0.03)
    ax.set_title("Exp15 Fig4 — per-seed PASS rate (5 categories per seed)", fontsize=10)
    fig.tight_layout(); fig.savefig(FIG / "fig4_per_seed.png", dpi=170, bbox_inches="tight"); plt.close(fig)

    # ---- Fig5 bootstrap CI ----
    names = [n for n in TOP + ["Adaptive B2"] if n in boot]
    fig, axs = plt.subplots(1, 2, figsize=(12.8, 4.3))
    ys = np.arange(len(names))
    axs[0].errorbar([boot[n]["pass_rate"] for n in names], ys, xerr=[
        [boot[n]["pass_rate"] - boot[n]["pass_rate_CI"][0] for n in names],
        [boot[n]["pass_rate_CI"][1] - boot[n]["pass_rate"] for n in names]],
        fmt="o", capsize=4, c="#4C72B0")
    axs[0].set_yticks(ys); axs[0].set_yticklabels([f"{n} ({boot[n]['PASS']}/{boot[n]['n_units']})" for n in names],
                                                  fontsize=8)
    axs[0].set_xlabel("PASS rate (bootstrap 95% CI)"); axs[0].grid(alpha=0.3, axis="x")
    axs[0].set_title("(a) PASS rate CI", fontsize=10)
    axs[1].errorbar([boot[n]["mean_ddp"] for n in names], ys, xerr=[
        [boot[n]["mean_ddp"] - boot[n]["mean_ddp_CI"][0] for n in names],
        [boot[n]["mean_ddp_CI"][1] - boot[n]["mean_ddp"] for n in names]],
        fmt="o", capsize=4, c="#55A868")
    axs[1].set_yticks(ys); axs[1].set_yticklabels(names, fontsize=8)
    axs[1].set_xlabel("mean Δd′ (bootstrap 95% CI)"); axs[1].grid(alpha=0.3, axis="x")
    axs[1].set_title("(b) preservation CI", fontsize=10)
    fig.suptitle("Exp15 Fig5 — bootstrap CIs (paired over common units)", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.92)); fig.savefig(FIG / "fig5_bootstrap.png", dpi=170,
                                                        bbox_inches="tight"); plt.close(fig)
    print("figures ->", FIG)


if __name__ == "__main__":
    main()
