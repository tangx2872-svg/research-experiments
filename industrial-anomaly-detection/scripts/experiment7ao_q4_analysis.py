"""Overnight Queue Q4 — analysis（CPU-only）。

产出：uniform normalization strength（历史已定义 α points）→ preservation / robustness 响应图，
合并 Q4-A 新单元与历史 raw（bottle/grid 复用，零 GPU 重跑）。
"""

from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import experiment7ao_config as c7  # noqa: E402
import experiment7ao_q4 as q4  # noqa: E402
import experiment7ao_analysis as A  # noqa: E402

SUM, FIG = c7.SUM_DIR, c7.FIG_DIR
ALPHAS = [0.0, 0.125, 0.20, 0.25, 0.30, c7.UNIFORM_ALPHA, 0.5, 0.601369125, 0.8018255]
METRICS = ["mean_dprime", "mean_abs_delta_z", "image_auroc", "fpr_clean"]


def akey_pair(a: float) -> tuple:
    return ("alpha", c7.akey(a), c7.akey(a))


def main() -> None:
    lookup, meta = A.build_lookup(extra_alpha=[("Q4", q4.Q4_RAW)])
    print(f"[Q4] lookup keys = {len(lookup)}")
    rows, per_cat, missing = [], [], []
    for a in ALPHAS:
        key = akey_pair(a)
        units = []
        for cat in c7.CATEGORIES:
            for seed in c7.SEEDS:
                k = (cat, seed) + key
                if k in lookup:
                    units.append({"category": cat, "seed": seed, **lookup[k]})
        cov = len(units)
        if cov < len(c7.CATEGORIES) * len(c7.SEEDS):
            missing.append({"alpha": a, "n_units": cov,
                            "missing": [f"{c}:{s}" for c in c7.CATEGORIES for s in c7.SEEDS
                                        if (c, s) + key not in lookup]})
        if not units:
            continue
        r = {"alpha": a, "n_units": cov, "n_categories": len({u["category"] for u in units}),
             "n_seeds": len({u["seed"] for u in units})}
        for m in METRICS:
            r[m] = round(float(np.mean([u[m] for u in units])), 6) if m in units[0] else None
        r["runtime_mean_seconds"] = round(float(np.mean(
            [meta[(u["category"], u["seed"]) + key]["runtime_seconds"] for u in units
             if meta[(u["category"], u["seed"]) + key]["runtime_seconds"]])), 1)
        rows.append(r)
        for cat in c7.CATEGORIES:
            cu = [u for u in units if u["category"] == cat]
            if cu:
                per_cat.append({"alpha": a, "category": cat, "n_units": len(cu),
                                "mean_dprime": round(float(np.mean([u["mean_dprime"] for u in cu])), 6),
                                "mean_abs_delta_z": round(float(np.mean([u["mean_abs_delta_z"] for u in cu])), 6),
                                "image_auroc": round(float(np.mean([u["image_auroc"] for u in cu])), 6)})
    c7.write_csv(SUM / "q4_alpha_response.csv", rows)
    c7.write_csv(SUM / "q4_alpha_response_per_category.csv", per_cat)
    if missing:
        c7.write_csv(SUM / "q4_alpha_missing.csv",
                     [{"alpha": m["alpha"], "n_units": m["n_units"], "missing": ";".join(m["missing"])}
                      for m in missing])
    print("\n=== Q4-A uniform-alpha response (merged historical + new) ===")
    print(f"{'alpha':>12}{'units':>7}{'cats':>6}{'seeds':>7}{'d-prime':>10}{'|dz|':>9}{'AUROC':>9}{'fpr_clean':>11}")
    for r in rows:
        print(f"{r['alpha']:>12.8f}{r['n_units']:>7}{r['n_categories']:>6}{r['n_seeds']:>7}"
              f"{r['mean_dprime']:>10.4f}{r['mean_abs_delta_z']:>9.4f}{r['image_auroc']:>9.4f}"
              f"{(r['fpr_clean'] if r['fpr_clean'] is not None else float('nan')):>11.4f}")
    if missing:
        print("\n[Q4] incomplete alpha points:")
        for m in missing:
            print(f"  alpha={m['alpha']} n_units={m['n_units']}")
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.0))
    cats = c7.CATEGORIES
    cols = plt.cm.tab10(np.linspace(0, 1, 10))
    xs = [r["alpha"] for r in rows]
    for ax, met, ylab in ((axes[0], "mean_dprime", "mean defect d′"),
                          (axes[1], "mean_abs_delta_z", "mean |ΔNormalScore_z|")):
        for i, cat in enumerate(cats):
            cx = [r["alpha"] for r in per_cat if r["category"] == cat]
            cy = [r[met] for r in per_cat if r["category"] == cat]
            order = np.argsort(cx)
            ax.plot(np.array(cx)[order], np.array(cy)[order], marker="o", ms=3.5, lw=1.1, alpha=0.75,
                    color=cols[i], label=cat)
        o = np.argsort(xs)
        ax.plot(np.array(xs)[order], np.array([r[met] for r in rows])[order], marker="s", lw=2.8,
                color="black", label="pooled")
        ax.axvline(c7.UNIFORM_ALPHA, color="#C44E52", ls="--", lw=1.4)
        ax.annotate("Uniform 0.40091275", (c7.UNIFORM_ALPHA, ax.get_ylim()[0]), fontsize=7,
                    color="#C44E52", rotation=90, va="bottom", ha="right")
        ax.set_xlabel("uniform normalization strength α"); ax.set_ylabel(ylab)
        ax.grid(alpha=0.3); ax.legend(fontsize=7, ncol=2)
    fig.suptitle("Q4 — uniform normalization strength response (merged historical raw + Q4-A units)", y=1.02)
    fig.tight_layout(); fig.savefig(FIG / "q4_alpha_response.png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    print("[Q4] wrote summary/q4_alpha_response*.csv + figures/q4_alpha_response.png")


if __name__ == "__main__":
    main()
