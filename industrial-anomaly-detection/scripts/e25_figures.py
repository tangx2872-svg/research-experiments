#!/usr/bin/env python
"""E25 multi-seed figures (PNG+PDF) from results/e25/multiseed_analysis.json.

Fig 1: per-domain multi-seed ΔAUROC / ΔAP (bars = mean over splits, dots = per-seed means)
Fig 2: per-target mean ± std across seeds, grouped by domain
No data is recomputed here beyond simple means of the stored per-split values.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np  # noqa: E402
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

OUTDIR = ROOT / "results" / "e25"
FIGDIR = OUTDIR / "figures"
COL = {"PCB": "#4C72B0", "MetalParts": "#C44E52"}


def save(fig, name: str) -> None:
    FIGDIR.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(FIGDIR / f"{name}.{ext}", dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    a = json.loads((OUTDIR / "multiseed_analysis.json").read_text())
    rows = a["per_split"]
    targets = [r for r in rows if r["valid"] == 1 and r["is_source"] == 0]
    doms = ["PCB", "MetalParts"]
    seeds = [str(s) for s in a["seeds"]]

    # ---------------- Fig 1: per-domain, per-seed means
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.4))
    for ax, key, label in [(axes[0], "d_auroc", "ΔAUROC"), (axes[1], "d_ap", "ΔAP")]:
        means = [a["per_domain"][d][f"{key}_mean"] for d in doms]
        ax.bar(np.arange(2), means, color=[COL[d] for d in doms], width=0.55)
        for i, d in enumerate(doms):
            vals = [a["per_domain"][d]["per_seed_d_auroc_mean"][s] if key == "d_auroc"
                    else None for s in seeds]
            if key == "d_auroc":
                ax.scatter(np.full(len(seeds), i), vals, color="k", s=22, zorder=3,
                           label="per-seed mean" if i == 0 else None)
        ax.axhline(0, color="k", lw=0.8)
        ax.set_xticks(np.arange(2), [f"{d}\n(n={a['per_domain'][d]['n_splits']} splits)"
                                     for d in doms], fontsize=9)
        ax.set_ylabel(label)
        ax.set_title(f"{label}: mean over valid target splits")
        ax.grid(axis="y", alpha=0.25)
    axes[0].legend(fontsize=8, frameon=False)
    fig.suptitle("E25 RobustAD, frozen R1-ILA, seeds 0/1/2", fontsize=10)
    save(fig, "e25_multiseed_domain_delta")

    # ---------------- Fig 2: per-target mean ± std
    order = sorted(targets, key=lambda r: (doms.index(r["domain"]),
                                           int(r["split"].replace("test", ""))))
    labels = [f"{r['domain'][:2]}-{r['split'].replace('test', 't')}" for r in order]
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.6), sharex=True)
    x = np.arange(len(order))
    for ax, key, label in [(axes[0], "d_auroc", "ΔAUROC"), (axes[1], "d_ap", "ΔAP")]:
        m = [r[f"{key}_mean"] for r in order]
        s = [r[f"{key}_std"] for r in order]
        ax.bar(x, m, yerr=s, capsize=2.5,
               color=[COL[r["domain"]] for r in order], width=0.62)
        ax.axhline(0, color="k", lw=0.8)
        ax.set_ylabel(label)
        ax.set_title(f"{label} per target split (mean ± std over 3 seeds)")
        ax.tick_params(axis="x", labelsize=7)
        ax.grid(axis="y", alpha=0.25)
    axes[0].set_xticks(x, labels, rotation=45, ha="right")
    fig.suptitle("E25 RobustAD per-target robustness (blue = PCB, red = MetalParts)",
                 fontsize=10)
    save(fig, "e25_multiseed_per_target")

    print(f"[E25 figures] wrote 2 figures x2 formats -> {FIGDIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
