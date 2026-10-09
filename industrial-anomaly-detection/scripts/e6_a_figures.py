#!/usr/bin/env python
"""E6-A figures (CPU-only). Reads results/e6_a/*.json and writes results/e6_a/figures/."""
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

OUT = ROOT / "results" / "e6_a"
FIGS = OUT / "figures"
REAL = ["B2", "X6c", "C01", "C06", "C07", "C08"]


def main() -> int:
    FIGS.mkdir(parents=True, exist_ok=True)
    summ = json.loads((OUT / "summary.json").read_text())
    base = summ["metrics"]
    rows = summ["tournament"]
    bmap = summ["blinded_procedure"]["map"]
    path = summ["pathology"]
    inv = {v: k for k, v in bmap.items()}

    methods = ["Original"] + REAL
    mid = [r["metric_id"] for r in rows]
    fields = {r["metric_id"]: r["field"] for r in rows}
    dirs = {r["metric_id"]: r["direction"] for r in rows}
    verdict = {r["metric_id"]: r["verdict"] for r in rows}

    # normalise each metric row to [0,1] where 1 = best (robust) for colouring
    M = np.full((len(mid), len(methods)), np.nan)
    for i, m_ in enumerate(mid):
        f, d = fields[m_], dirs[m_]
        v = np.array([base[k].get(f, np.nan) for k in methods], dtype=float)
        ok = np.isfinite(v)
        if ok.sum() < 2:
            continue
        lo, hi = v[ok].min(), v[ok].max()
        n = (v - lo) / (hi - lo + 1e-12)
        M[i] = (n if d == "higher" else 1 - n)

    # ---------------- F1 heatmap, blind + unblinded
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 6.2))
    for ax, labels, title in ((axes[0], ["Original"] + [bmap[k] for k in REAL], "BLINDED"),
                              (axes[1], methods, "UNBLINDED")):
        im = ax.imshow(M, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
        ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_yticks(range(len(mid)))
        ax.set_yticklabels([f"{m_} [{verdict[m_][:4]}]" for m_ in mid], fontsize=8)
        for i in range(len(mid)):
            for j in range(len(methods)):
                if np.isfinite(M[i, j]):
                    ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=6.5)
        ax.set_title(f"Figure 1 ({title}) — metric x method, green = more robust")
        fig.colorbar(im, ax=ax, fraction=0.03)
    fig.tight_layout(); fig.savefig(FIGS / "fig1_metric_method_heatmap.png", dpi=150); plt.close(fig)

    # ---------------- F2 detection vs illumination sensitivity (Pareto)
    fig, ax = plt.subplots(figsize=(7.2, 5.4))
    gate = {k: base[k]["auroc"] >= 0.5 and base[k]["d_dprime"] >= -0.10 for k in methods}
    for k in methods:
        x, y = base[k]["auroc"], base[k]["M5_A_all"]
        ax.scatter([x], [y], s=150 if k == "Original" else 110,
                   marker="*" if k == "Original" else "o",
                   color="black" if k == "Original" else ("tab:blue" if gate[k] else "tab:red"),
                   edgecolor="black", zorder=3)
        ax.annotate(f"{k}{'' if gate[k] else ' (gate FAIL)'}", (x, y),
                    textcoords="offset points", xytext=(8, -3), fontsize=9)
    ax.axvline(0.5, ls="--", lw=1, color="grey")
    ax.set_xlabel("Axis X — image AUROC (detection)")
    ax.set_ylabel("Axis Y — M5_A standardized illumination sensitivity (lower = more robust)")
    ax.set_title("Figure 2 — detection x illumination-sensitivity\nred = fails the detection gate "
                 "(AUROC>=0.5 and dd' >= -0.10)")
    ax.grid(alpha=0.3); fig.tight_layout()
    fig.savefig(FIGS / "fig2_detection_vs_sensitivity_pareto.png", dpi=150); plt.close(fig)

    # ---------------- F3 per-illumination AUROC stability
    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    ill = [f"{i:02d}" for i in range(1, 11)]
    for k in methods:
        vals = []
        for i in ill:
            vals.append(base[k]["_M8_per_illum"][int(i) - 1] if "_M8_per_illum" in base[k] else np.nan)
        ax.plot(ill, vals, marker="o", lw=1.4, label=f"{k} ({base[k]['auroc']:.3f})")
    ax.axhline(0.5, ls="--", lw=1, color="grey")
    ax.set_xlabel("illumination id"); ax.set_ylabel("image AUROC within that illumination")
    ax.set_title("Figure 3 — per-illumination AUROC (collapse is visible, nothing hidden)")
    ax.legend(fontsize=8); ax.grid(alpha=0.3); fig.tight_layout()
    fig.savefig(FIGS / "fig3_per_illumination_auroc.png", dpi=150); plt.close(fig)

    # ---------------- F4 pathology
    t6 = path["T6_illumination_injection"]["per_c"]
    cs = ["0.0", "0.2", "0.5", "1.0", "2.0"]
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.8))
    for f_ in ["R_all", "M3_all", "M5_A_all", "M7_std_all_mean"]:
        ys = [t6[c]["C01"].get(f_, np.nan) for c in cs]
        axes[0].plot(cs, ys, marker="o", label=f_)
    axes[0].set_xlabel("injected illumination offset c (x sigma_all)")
    axes[0].set_ylabel("metric value (C01)")
    axes[0].set_title("Figure 4a — T6 injection response\n(R_all and M7_std DIP at c=0.2 => fail "
                      "monotone detection)")
    axes[0].legend(fontsize=8); axes[0].grid(alpha=0.3)

    t1m = path["T1_scale_invariance"]["per_method"]
    names = [k for k in t1m]
    worst = [max([v for v in t1m[k].values() if np.isfinite(v)] or [np.nan]) for k in names]
    axes[1].bar(names, worst, color="tab:green", edgecolor="black")
    axes[1].axhline(1e-6, ls="--", color="red", label="invariance tolerance 1e-6")
    axes[1].set_yscale("symlog", linthresh=1e-9)
    axes[1].set_ylabel("worst relative deviation under s -> a*s+b")
    axes[1].set_title("Figure 4b — T1 scale invariance per method\n(all below tolerance => PASS)")
    axes[1].legend(fontsize=8); axes[1].grid(alpha=0.3, axis="y")
    fig.tight_layout(); fig.savefig(FIGS / "fig4_pathology.png", dpi=150); plt.close(fig)

    # ---------------- F5 C01 vs Original specimen-level paired comparison
    l1 = json.loads((OUT / "c01_reevaluation.json").read_text())
    fields5 = ["M3_good", "M3_ng", "M5_A_all", "M5_C_all", "M4_good_mean", "M4_ng_mean"]
    fig, ax = plt.subplots(figsize=(8.6, 5.2))
    x = np.arange(len(fields5)); w = 0.36
    vo = [base["Original"].get(f, np.nan) for f in fields5]
    vc = [base["C01"].get(f, np.nan) for f in fields5]
    ax.bar(x - w / 2, vo, w, label="Original", color="black", edgecolor="black")
    ax.bar(x + w / 2, vc, w, label="C01 PIAD-Retinex", color="tab:orange", edgecolor="black")
    for i, f in enumerate(fields5):
        d = l1["l1oo_deltas_c01_minus_original"].get(f, {}).get("deltas", [])
        dl = [q for q in d if q is not None]
        if dl:
            n = sum(1 for q in dl if q < 0)
            ax.text(i, max(vo[i], vc[i]) * 1.03, f"L1oO {n}/10 lower", ha="center", fontsize=7.5)
    ax.set_xticks(x); ax.set_xticklabels(fields5, rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("metric value")
    ax.set_title("Figure 5 — C01 vs Original (C01 = AMBIGUOUS on M4, better on M3/M5)\n"
                 "paired, same specimens / same illuminations")
    ax.legend(fontsize=8); ax.grid(alpha=0.3, axis="y"); fig.tight_layout()
    fig.savefig(FIGS / "fig5_c01_vs_original.png", dpi=150); plt.close(fig)

    # ---------------- F6 sign-disagreement across metrics (the key audit finding)
    mets = ["M1_R_all", "M3_good", "M3_ng", "M3_all", "M5_A", "M5_C", "M5_B", "M5_D",
            "M7_std", "M7_raw", "M2_raw_var", "M8_sd", "M4_good", "M4_ng"]
    dev = []
    for m_ in mets:
        f, d = fields[m_], dirs[m_]
        o, c = base["Original"].get(f, np.nan), base["C01"].get(f, np.nan)
        if not (np.isfinite(o) and np.isfinite(c)) or abs(o) < 1e-12:
            dev.append(np.nan); continue
        rel = (c - o) / abs(o)
        dev.append(rel if d == "lower" else -rel)     # negative = C01 more robust
    fig, ax = plt.subplots(figsize=(10.5, 5.0))
    cols = ["tab:green" if (np.isfinite(v) and v < 0) else
            ("tab:red" if (np.isfinite(v) and v > 0) else "grey") for v in dev]
    ax.bar(mets, dev, color=cols, edgecolor="black")
    ax.axhline(0, color="black", lw=1)
    ax.set_ylabel("relative change C01 vs Original  (negative = C01 more robust)")
    ax.set_title("Figure 6 — the sign disagreement: 'is C01 more illumination-robust?' depends on the metric\n"
                 "green = C01 better, red = C01 worse")
    for i, m_ in enumerate(mets):
        ax.text(i, 0, verdict.get(m_, "?"), ha="center", va="bottom", fontsize=6, rotation=90)
    plt.xticks(rotation=45, ha="right", fontsize=8)
    ax.grid(alpha=0.3, axis="y"); fig.tight_layout()
    fig.savefig(FIGS / "fig6_c01_metric_sign_disagreement.png", dpi=150); plt.close(fig)

    print(f"[written] {FIGS}")
    for f in sorted(FIGS.glob("*.png")):
        print("  ", f.name, f"{f.stat().st_size/1024:.0f} KB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
