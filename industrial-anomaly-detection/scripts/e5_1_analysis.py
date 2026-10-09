#!/usr/bin/env python
"""E5-1 — final tables, ranks and figures.

Reads `results/e5_1/raw/*.json` (+ the frozen Original reference) and writes:
  results/e5_1/analysis/summary.csv
  results/e5_1/analysis/summary.md
  results/e5_1/figures/fig1_auroc_vs_robustness.png
  results/e5_1/figures/fig2_deltas.png
  results/e5_1/figures/fig3_illumination_stability.png

Ranking is reported on four axes (protocol section 9): Detection / Robustness / Preservation
and Overall Decision. Failed candidates are shown, never hidden.
"""
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

import e4x_common as C  # noqa: E402

RAW = ROOT / "results" / "e5_1" / "raw"
ANALYSIS = ROOT / "results" / "e5_1" / "analysis"
FIGS = ROOT / "results" / "e5_1" / "figures"
ORIG = ROOT / "results" / "e4_x" / "per_image_original.csv"

ORDER = ["C01", "C06", "C07", "C08"]
LABEL = {"C01": "C01 PIAD-Retinex (input)", "C06": "C06 SimpleNet (post-concat)",
         "C07": "C07 ReConPatch (post-concat)", "C08": "C08 CRAD (memory)"}


def load_original() -> list[dict]:
    with open(ORIG) as f:
        return [{"specimen": r["specimen"], "kind": r["kind"], "illumination": r["illumination"],
                 "score": float(r["score"])} for r in csv.DictReader(f)]


def main() -> int:
    ANALYSIS.mkdir(parents=True, exist_ok=True)
    FIGS.mkdir(parents=True, exist_ok=True)

    orig_rows = load_original()
    orig = C.metric_block(orig_rows, "Original")

    rows = []
    for cand in ORDER:
        p = RAW / f"{cand}.json"
        if not p.exists():
            rows.append({"candidate": cand, "status": "NOT_RUN", "decision": "-",
                         "auroc": np.nan, "d_auroc": np.nan, "dprime": np.nan, "d_dprime": np.nan,
                         "R_all": np.nan, "R_ratio": np.nan})
            continue
        d = json.loads(p.read_text())
        m, dec = d["metrics"], d["decision"]
        rows.append({"candidate": cand, "status": "OK", "decision": dec["decision"],
                     "auroc": m["image_auroc"], "d_auroc": dec["d_auroc"],
                     "dprime": m["d_prime"], "d_dprime": dec["d_dprime"],
                     "R_all": m["R_all"], "R_ratio": dec["R_ratio"],
                     "R_Good": m.get("R_Good"), "R_NG": m.get("R_NG"),
                     "single_cond": bool(dec.get("SINGLE-CONDITION-DRIVEN", False))})
    ok = [r for r in rows if r["status"] == "OK"]

    # ---- ranking on four axes (protocol section 9)
    det = sorted(ok, key=lambda r: -r["auroc"])
    rob = sorted(ok, key=lambda r: r["R_ratio"])
    pre = sorted(ok, key=lambda r: -r["dprime"])
    rank = {}
    for name, seq in (("Detection", det), ("Robustness", rob), ("Preservation", pre)):
        for i, r in enumerate(seq, 1):
            rank.setdefault(r["candidate"], {})[name] = i

    # ---- summary.csv
    with open(ANALYSIS / "summary.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Candidate", "AUROC", "dAUROC", "d_prime", "dd_prime", "R_all", "R_ratio",
                    "Decision", "Detection_Rank", "Robustness_Rank", "Preservation_Rank",
                    "SingleConditionDriven"])
        w.writerow(["Original", f"{orig['image_auroc']:.5f}", "0.00000", f"{orig['d_prime']:.5f}",
                    "0.00000", f"{orig['R_all']:.5f}", "1.0000", "REFERENCE", "1", "1", "1", "No"])
        for r in rows:
            if r["status"] != "OK":
                w.writerow([r["candidate"], "n/a", "n/a", "n/a", "n/a", "n/a", "n/a",
                            "NOT_RUN/FAILED", "-", "-", "-", "-"])
                continue
            w.writerow([r["candidate"], f"{r['auroc']:.5f}", f"{r['d_auroc']:+.5f}",
                        f"{r['dprime']:.5f}", f"{r['d_dprime']:+.5f}", f"{r['R_all']:.5f}",
                        f"{r['R_ratio']:.4f}", r["decision"],
                        rank[r["candidate"]]["Detection"], rank[r["candidate"]]["Robustness"],
                        rank[r["candidate"]]["Preservation"],
                        "YES" if r["single_cond"] else "No"])

    # ---- summary.md
    lines = ["# E5-1 — Broad Module Composition Screening (results)", "",
             f"Frozen reference: Original AUROC={orig['image_auroc']:.5f}, "
             f"d'={orig['d_prime']:.5f}, R_all={orig['R_all']:.5f}", "",
             "| Candidate | AUROC | dAUROC | d' | dd' | R_all | R_ratio | Decision |",
             "|---|---|---|---|---|---|---|---|",
             f"| Original | {orig['image_auroc']:.5f} | +0.00000 | {orig['d_prime']:.5f} | +0.00000 | "
             f"{orig['R_all']:.5f} | 1.0000 | REFERENCE |"]
    for r in rows:
        if r["status"] != "OK":
            lines.append(f"| {r['candidate']} | — | — | — | — | — | — | NOT_RUN/FAILED |")
        else:
            lines.append(f"| {LABEL.get(r['candidate'], r['candidate'])} | {r['auroc']:.5f} | "
                         f"{r['d_auroc']:+.5f} | {r['dprime']:.5f} | {r['d_dprime']:+.5f} | "
                         f"{r['R_all']:.5f} | {r['R_ratio']:.4f} | **{r['decision']}** |")
    lines += ["", "## Ranks (not AUROC-only)", "",
              "| Candidate | Detection | Robustness | Preservation | Overall |", "|---|---|---|---|---|"]
    for r in rows:
        if r["status"] != "OK":
            continue
        k = rank[r["candidate"]]
        lines.append(f"| {r['candidate']} | {k['Detection']} | {k['Robustness']} | "
                     f"{k['Preservation']} | **{r['decision']}** |")
    (ANALYSIS / "summary.md").write_text("\n".join(lines) + "\n")

    # ---- Figure 1: AUROC vs robustness trade-off
    plt.figure(figsize=(6.4, 5.0))
    plt.scatter([orig["R_all"]], [orig["image_auroc"]], marker="*", s=260, color="black",
                label="Original", zorder=3)
    plt.annotate("Original", (orig["R_all"], orig["image_auroc"]),
                 textcoords="offset points", xytext=(6, 6))
    cmap = {"GO": "tab:green", "HOLD": "tab:orange", "STOP": "tab:red"}
    for r in ok:
        plt.scatter([r["R_all"]], [r["auroc"]], s=90, color=cmap.get(r["decision"], "grey"),
                    edgecolor="black", zorder=3)
        plt.annotate(r["candidate"], (r["R_all"], r["auroc"]), textcoords="offset points",
                     xytext=(7, -3))
    plt.axvline(orig["R_all"], ls="--", lw=0.8, color="grey")
    plt.axhline(orig["image_auroc"], ls="--", lw=0.8, color="grey")
    plt.xlabel("R_all  (mean cross-illumination dispersion of z-scores; lower = more robust)")
    plt.ylabel("image AUROC (higher = better detection)")
    plt.title("Figure 1 — detection vs illumination robustness\ngreen = GO, orange = HOLD, red = STOP")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIGS / "fig1_auroc_vs_robustness.png", dpi=150)
    plt.close()

    # ---- Figure 2: deltas
    if ok:
        fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
        names = [r["candidate"] for r in ok]
        for ax, key, thr, ttl in ((axes[0], "d_auroc", 0.03, "dAUROC (>= +0.03 = detection win)"),
                                  (axes[1], "d_dprime", -0.10, "dd' (>= -0.10 allowed)"),
                                  (axes[2], "R_ratio", 0.90, "R_ratio (<= 0.90 = robustness win)")):
            vals = [r[key] for r in ok]
            base = 1.0 if key == "R_ratio" else 0.0
            bars = ax.bar(names, [v - base for v in vals], bottom=base,
                          color=[cmap.get(r["decision"], "grey") for r in ok], edgecolor="black")
            ax.axhline((thr if key != "R_ratio" else thr), ls="--", lw=1, color="green")
            if key != "R_ratio":
                ax.axhline(0.0, ls="-", lw=0.8, color="black")
            ax.set_title(ttl, fontsize=9)
            ax.grid(alpha=0.3, axis="y")
            for b, v in zip(bars, vals):
                ax.annotate(f"{v:+.4f}" if key != "R_ratio" else f"{v:.4f}",
                            (b.get_x() + b.get_width() / 2, b.get_height()), ha="center",
                            va="bottom", fontsize=8)
        fig.suptitle("Figure 2 — deltas vs frozen Original", fontsize=11)
        fig.tight_layout()
        fig.savefig(FIGS / "fig2_deltas.png", dpi=150)
        plt.close(fig)

    # ---- Figure 3: illumination-level stability (all candidates, failures included)
    plt.figure(figsize=(9.5, 5.0))
    ill = C.ILLUMINATIONS
    plt.plot(ill, [p["image_auroc"] for p in orig["per_illumination"]], marker="o", lw=2,
             color="black", label="Original")
    for cand in ORDER:
        p = RAW / f"{cand}.json"
        if not p.exists():
            continue
        d = json.loads(p.read_text())
        plt.plot(ill, [q["image_auroc"] for q in d["metrics"]["per_illumination"]], marker="o",
                 lw=1.2, alpha=0.85, label=f"{cand} ({d['decision']['decision']})")
    plt.xlabel("illumination id")
    plt.ylabel("image AUROC")
    plt.title("Figure 3 — per-illumination AUROC (nothing hidden)")
    plt.legend(fontsize=8)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIGS / "fig3_illumination_stability.png", dpi=150)
    plt.close()

    print((ANALYSIS / "summary.md").read_text())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
