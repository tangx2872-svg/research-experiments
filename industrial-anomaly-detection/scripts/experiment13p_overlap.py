"""Experiment 13-P — E: defect overlap vs random-subspace control（CPU-only）。

使用**冻结**的 K=8 illumination 子空间（analysis/illumination_subspace_freeze.json）：
  overlap(patch) = ||P_illum d||^2 / ||d||^2,  P = U U^T,  d = F_patch - mu_normal
  mean overlap = trace(U^T S U) / sum||d||^2   （S = Σ d dᵀ）
random control：100 个同维随机正交子空间（ambient C），比较 aggregated mean overlap
               -> percentile / z-score / empirical p-value
"""
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
EXP = ROOT / "results" / "experiment_13p"
ANA, FIG = EXP / "analysis", EXP / "figures"
CATS = ["bottle", "cable", "hazelnut", "screw", "grid"]
LAYERS = ["layer2", "layer3"]
K = 8
R = 100


def wcsv(p, rows):
    if not rows:
        return
    keys = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(p, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader(); w.writerows(rows)


def random_orth(C: int, k: int, rng) -> np.ndarray:
    return np.linalg.qr(rng.standard_normal((C, k)))[0][:, :k]


def main() -> None:
    freeze = json.loads((ANA / "illumination_subspace_freeze.json").read_text())
    rng = np.random.RandomState(0)
    rows, dist = [], {}
    for cat in CATS:
        f = ANA / f"defect_stats_{cat}.npz"
        if not f.exists():
            print("  [skip] missing", f.name); continue
        z = dict(np.load(f))
        tags = sorted({k.split("_", 2)[2] for k in z if k.startswith("S_")})
        for layer in LAYERS:
            U = np.load(ANA / f"illum_subspace_{cat}_{layer}_k8.npz")["U"].astype(np.float64)
            C = U.shape[0]
            rand = np.stack([random_orth(C, K, rng) for _ in range(R)])       # (R,C,K)
            for tag in tags:
                S = z[f"S_{layer}_{tag}"].astype(np.float64)
                nrm = float(z[f"norm_{layer}_{tag}"][0])
                cnt = int(z[f"cnt_{layer}_{tag}"][0])
                obs = float(np.trace(U.T @ S @ U) / max(nrm, 1e-30))
                # trace(U_rᵀ S U_r)（精确，无需存储逐 patch 向量）
                SU = np.einsum("ab,rbk->rak", S, rand)
                tr = np.einsum("rak,rak->r", SU, rand)
                null = tr / max(nrm, 1e-30)
                pct = float((null < obs).mean() * 100.0)
                pval = float((null >= obs).mean())
                rows.append({"category": cat, "layer": layer, "subset": tag, "n_patches": cnt,
                             "mean_overlap": round(obs, 6),
                             "random_mean": round(float(null.mean()), 6),
                             "random_p95": round(float(np.percentile(null, 95)), 6),
                             "percentile": round(pct, 2),
                             "z_score": round(float((obs - null.mean()) / (null.std() + 1e-12)), 3),
                             "p_value": round(pval, 4),
                             "elevated": bool(obs > np.percentile(null, 95))})
                if tag != "good":
                    dist.setdefault((cat, layer), []).append(obs)
    wcsv(ANA / "E_defect_overlap.csv", rows)

    (ANA / "E_overlap_summary.json").write_text(json.dumps(
        {"K": K, "random_draws": R, "definition": "overlap = ||P_illum d||^2/||d||^2, d = F_patch - mu_normal",
         "rows": rows,
         "defect_vs_good_note": "good 行为 normal control；defect 行为 defect（含各 defect type）"}, indent=2, ensure_ascii=False))

    # ---- Figure 4 ----
    fig, axes = plt.subplots(1, 2, figsize=(12.6, 4.8))
    for ax, layer in zip(axes, LAYERS):
        cats = [c for c in CATS if (c, layer) in dist]
        xs = np.arange(len(cats)); w = 0.2
        for i, tag in enumerate(["broken_large", "broken_small", "contamination", "good"]):
            vals = [next((r["mean_overlap"] for r in rows if r["category"] == c and r["layer"] == layer
                          and r["subset"] == tag), np.nan) for c in cats]
            ax.bar(xs + (i - 1.5) * w, vals, w, label=tag, edgecolor="black", lw=0.5)
        rnd = [next((r["random_mean"] for r in rows if r["category"] == c and r["layer"] == layer
                     and r["subset"] == "good"), np.nan) for c in cats]
        ax.plot(xs, rnd, "k_", ms=22, label="random-subspace mean")
        ax.set_xticks(xs); ax.set_xticklabels(cats, fontsize=9)
        ax.set_title("%s (K=%d)" % (layer, K), fontsize=10.5)
        ax.set_ylabel("mean overlap ||P d||²/||d||²"); ax.grid(alpha=0.3, axis="y")
        ax.legend(fontsize=7)
    fig.suptitle("Experiment 13-P — Defect-direction overlap with frozen K=8 illumination subspace "
                 "(vs 100 random subspaces)", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(FIG / "fig4_defect_overlap_vs_random.png", dpi=170, bbox_inches="tight")
    plt.close(fig)

    print("=" * 104)
    print("E. Defect overlap vs random-subspace control (K=8 frozen illumination subspace)")
    print("=" * 104)
    print("%-10s%-8s%-14s%8s%12s%12s%10s%8s%8s" % ("cat", "layer", "subset", "n_patch",
                                                   "overlap", "rand_mean", "pctile", "z", "p"))
    for r in rows:
        print("%-10s%-8s%-14s%8d%12.5f%12.5f%10.1f%8.1f%8.4f%s" % (
            r["category"], r["layer"], r["subset"], r["n_patches"], r["mean_overlap"],
            r["random_mean"], r["percentile"], r["z_score"], r["p_value"],
            "  ELEVATED" if r["elevated"] else ""))
    print("=" * 104)
    print("figure ->", FIG / "fig4_defect_overlap_vs_random.png")


if __name__ == "__main__":
    main()
