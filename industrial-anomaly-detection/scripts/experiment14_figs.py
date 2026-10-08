"""Experiment 14 — 图表（Figure 1-5）。CPU-only。"""
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

E14 = ROOT / "results" / "experiment_14"
ANA, FIG = E14 / "analysis", E14 / "figures"


def load(p):
    return list(csv.DictReader(open(p))) if p.exists() else []


def baselines() -> dict:
    """Original / Fixed C2 / Adaptive B2 绝对指标（5 cats × 3 seeds，复用历史，0 GPU）。"""
    out = {}
    srcs = {"Original": lambda c, s: M.source(c, s, "orig"),
            "Fixed C2": lambda c, s: M.find([M.E10], c, s, "P10_C2_L2resid025_L3a025"),
            "Adaptive B2": lambda c, s: M.source(c, s, "b2")}
    for label, fn in srcs.items():
        units = {(c, s, label): fn(c, s) for c in M.CATS for s in M.SEEDS}
        units = {k: v for k, v in units.items() if v is not None}
        mt = M.metrics_of(units)
        out[label] = {"n": len(mt),
                      "absz": float(np.mean([v["mean_abs_delta_z"] for v in mt.values()])),
                      "dprime": float(np.mean([v["mean_dprime"] for v in mt.values()])),
                      "per": {(k[0], k[1]): (v["mean_abs_delta_z"], v["mean_dprime"]) for k, v in mt.items()}}
    return out


def cand_agg(rows) -> dict:
    agg = {}
    for cid in sorted({r["candidate"] for r in rows}):
        r = [x for x in rows if x["candidate"] == cid]
        agg[cid] = {"family": r[0].get("family", "C_late_fusion"), "n": len(r),
                    "absz": float(np.mean([float(x["absz"]) for x in r])),
                    "dprime": float(np.mean([float(x["dprime"]) for x in r])),
                    "PASS": sum(1 for x in r if str(x["PASS_vs_orig"]) == "True"),
                    "cat": sum(1 for x in r if str(x["catastrophic"]) == "True"),
                    "worst_ddp": min(float(x["ddp_vs_orig"]) for x in r),
                    "per": {(x["category"], int(x["seed"])): (float(x["dR_vs_orig"]), float(x["ddp_vs_orig"]),
                                                              str(x["PASS_vs_orig"]) == "True",
                                                              str(x["catastrophic"]) == "True") for x in r}}
    return agg


def fig1(bl, agg):
    fig, axs = plt.subplots(1, 2, figsize=(14.4, 5.9))
    ax = axs[0]
    for label, d in bl.items():
        ax.scatter(d["absz"], d["dprime"], s=210, marker="*",
                   c={"Original": "#4C72B0", "Fixed C2": "#55A868", "Adaptive B2": "#C44E52"}[label],
                   edgecolor="black", zorder=5, label=f"{label} (5cat x 3seed)")
        ax.annotate(label, (d["absz"], d["dprime"]), textcoords="offset points", xytext=(8, 6), fontsize=9)
    cmap = {"A_dual_path": "#8172B2", "A_layerwise": "#937860", "B_tiny_inss": "#DA8BC3",
            "C_late_fusion": "#64B5CD"}
    best = max(x["PASS"] for x in agg.values())
    for cid, d in agg.items():
        ax.scatter(d["absz"], d["dprime"], s=48, c=cmap.get(d["family"], "gray"),
                   edgecolor="black", lw=0.4, alpha=0.9, zorder=4)
        if d["PASS"] >= best - 1:
            ax.annotate(cid, (d["absz"], d["dprime"]), textcoords="offset points", xytext=(5, -9), fontsize=7.5)
    ax.set_xlabel("robustness: mean |dNormalScore_z|  (lower = better)")
    ax.set_ylabel("preservation: mean defect d'  (higher = better)")
    ax.set_title("Exp14 Fig1 - Robustness vs Preservation (each candidate averaged over its evaluated units)")
    ax.grid(alpha=0.3); ax.legend(fontsize=8, loc="lower left")
    ax.set_title("(a) mean-level Pareto view", fontsize=10)

    # (b) 决策视图：PASS 数 vs worst Δd'（catastrophic 线）
    ax2 = axs[1]
    for label, d in bl.items():
        pc = sum(1 for v in d["per"].values() if v[0] <= -M.EPS_RZ and v[1] >= -M.EPS_DP)
        wd = min(v[1] for v in d["per"].values())
        ax2.scatter(pc, wd, s=210, marker="*",
                    c={"Original": "#4C72B0", "Fixed C2": "#55A868", "Adaptive B2": "#C44E52"}[label],
                    edgecolor="black", zorder=5)
        ax2.annotate(label, (pc, wd), textcoords="offset points", xytext=(8, 6), fontsize=9)
    for cid, d in agg.items():
        pc = d["PASS"]
        ax2.scatter(pc, d["worst_ddp"], s=48, c=cmap.get(d["family"], "gray"),
                    edgecolor="black", lw=0.4, zorder=4)
        if d["PASS"] >= best - 1 or d["worst_ddp"] < -0.2:
            ax2.annotate(cid, (pc, d["worst_ddp"]), textcoords="offset points", xytext=(5, -9), fontsize=7.5)
    ax2.axhline(-M.CATASTROPHIC, ls="--", c="black", lw=1.1, label="catastrophic threshold (-0.25)")
    ax2.axhline(-0.2852, ls=":", c="#C44E52", lw=1.4, label="Adaptive B2 worst dd' (-0.2852)")
    ax2.set_xlabel("PASS count (units tested)")
    ax2.set_ylabel("worst-case Δd′  (higher = safer downside)")
    ax2.set_title("(b) decision view: PASS vs worst-case downside", fontsize=10)
    ax2.grid(alpha=0.3); ax2.legend(fontsize=7.5, loc="lower left")

    fig.suptitle("Exp14 Fig1 — Robustness vs Preservation (a) and downside-risk decision view (b)", fontsize=10.5)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(FIG / "fig1_pareto_scatter.png", dpi=170, bbox_inches="tight"); plt.close(fig)


def fig2(agg):
    fams = sorted({d["family"] for d in agg.values()})
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.6))
    xs = np.arange(len(fams)); w = 0.38
    axes[0].bar(xs - w / 2, [max((agg[c]["PASS"] for c in agg if agg[c]["family"] == f), default=0) for f in fams],
                w, label="best PASS (units tested)", edgecolor="black", lw=0.5)
    axes[0].bar(xs + w / 2, [max((agg[c]["cat"] for c in agg if agg[c]["family"] == f), default=0) for f in fams],
                w, label="worst catastrophic", edgecolor="black", lw=0.5)
    axes[0].axhline(12, ls="--", c="#C44E52", lw=1.2, label="Adaptive B2 = 12/15")
    axes[0].set_xticks(xs); axes[0].set_xticklabels(fams, fontsize=8); axes[0].legend(fontsize=7.5)
    axes[0].set_title("Family leaderboard (best candidate per family)", fontsize=10)
    axes[1].bar(xs - w / 2, [min((agg[c]["worst_ddp"] for c in agg if agg[c]["family"] == f), default=0) for f in fams],
                w, label="worst dd'", edgecolor="black", lw=0.5)
    axes[1].bar(xs + w / 2, [min((agg[c]["absz"] for c in agg if agg[c]["family"] == f), default=0) for f in fams],
                w, label="best mean |dz|", edgecolor="black", lw=0.5)
    axes[1].axhline(-0.2852, ls="--", c="#C44E52", lw=1.2, label="B2 worst dd' = -0.2852")
    axes[1].set_xticks(xs); axes[1].set_xticklabels(fams, fontsize=8); axes[1].legend(fontsize=7.5)
    axes[1].set_title("Family downside risk", fontsize=10)
    fig.tight_layout(); fig.savefig(FIG / "fig2_family_leaderboard.png", dpi=170, bbox_inches="tight"); plt.close(fig)


def fig34(agg, top):
    cats = M.CATS
    fig, axes = plt.subplots(1, len(top), figsize=(3.1 * len(top), 3.7), squeeze=False)
    for ax, cid in zip(axes[0], top):
        d = agg[cid]
        Mx = np.zeros((len(cats), 3))
        for (c, s), v in d["per"].items():
            Mx[cats.index(c), s] = v[1]
        im = ax.imshow(Mx, cmap="viridis", aspect="auto")
        ax.set_xticks(range(3)); ax.set_xticklabels(["s0", "s1", "s2"], fontsize=7)
        ax.set_yticks(range(len(cats))); ax.set_yticklabels(cats, fontsize=7)
        for i in range(len(cats)):
            for j in range(3):
                ax.text(j, i, "%.2f" % Mx[i, j], ha="center", va="center", fontsize=6,
                        color="white" if Mx[i, j] < Mx.max() * 0.55 else "black")
        ax.set_title("%s\nPASS %d/%d, cat %d" % (cid, d["PASS"], d["n"], d["cat"]), fontsize=8)
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle("Fig3 - delta d' per category x seed (top5 candidates)", fontsize=9.5)
    fig.tight_layout(rect=(0, 0, 1, 0.90)); fig.savefig(FIG / "fig3_top_heatmap.png", dpi=170, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, len(top), figsize=(3.1 * len(top), 3.7), squeeze=False)
    for ax, cid in zip(axes[0], top):
        d = agg[cid]
        Mx = np.zeros((len(cats), 3))
        for (c, s), v in d["per"].items():
            Mx[cats.index(c), s] = -1.0 if v[3] else (1.0 if v[2] else 0.0)
        ax.imshow(Mx, cmap="RdYlGn", vmin=-1, vmax=1, aspect="auto")
        ax.set_xticks(range(3)); ax.set_xticklabels(["s0", "s1", "s2"], fontsize=7)
        ax.set_yticks(range(len(cats))); ax.set_yticklabels(cats, fontsize=7)
        for i in range(len(cats)):
            for j in range(3):
                ax.text(j, i, {1.0: "P", 0.0: "H", -1.0: "C"}[Mx[i, j]], ha="center", va="center", fontsize=7)
        ax.set_title("%s (PASS %d/%d)" % (cid, d["PASS"], d["n"]), fontsize=8)
    fig.suptitle("Fig4 - PASS/HOLD/FAIL matrix vs same-(cat,seed) Original (P=PASS, H=FAIL/HOLD, C=catastrophic)",
                 fontsize=8.5)
    fig.tight_layout(rect=(0, 0, 1, 0.90)); fig.savefig(FIG / "fig4_pass_matrix.png", dpi=170, bbox_inches="tight")
    plt.close(fig)


def fig5(bl, agg, top):
    fig, ax = plt.subplots(figsize=(7.6, 4.2))
    for cid in top:
        d = agg[cid]
        ys = [sum(1 for (c, s), v in d["per"].items() if s == sd and v[2]) /
              max(1, sum(1 for (c, s) in d["per"] if s == sd)) for sd in (0, 1, 2)]
        ax.plot([0, 1, 2], ys, "-o", label=f"{cid} (n={d['n']})")
    for label, d in bl.items():
        ys = [sum(1 for (c, s), v in d["per"].items() if s == sd
                  and (v[0] <= -M.EPS_RZ and v[1] >= -M.EPS_DP)) /
              max(1, sum(1 for (c, s) in d["per"] if s == sd)) for sd in (0, 1, 2)]
        ax.plot([0, 1, 2], ys, "--", c="gray", lw=1.0, label=f"{label} PASS-rate")
    ax.set_xticks([0, 1, 2]); ax.set_xlabel("seed"); ax.set_ylabel("PASS rate within seed")
    ax.set_title("Exp14 Fig5 - per-seed stability", fontsize=10); ax.legend(fontsize=7); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(FIG / "fig5_seed_stability.png", dpi=170, bbox_inches="tight"); plt.close(fig)


def fig6():
    """O1: 25 单元（seeds 0-4）合并对比。"""
    rows = load(ANA / "O1_all_per_unit.csv")
    if not rows:
        return
    import csv as _c
    c15 = load(ANA / "C_late_fusion_per_unit.csv")
    agg = {}
    for cid in ("C1", "C2", "C3", "C4", "C5"):
        a = [x for x in c15 if x["candidate"] == cid]
        b = [x for x in rows if x["candidate"] == cid]
        both = a + b
        agg[cid] = {"PASS": sum(1 for x in both if str(x["PASS_vs_orig"]) == "True"),
                    "n": len(both),
                    "cat": sum(1 for x in both if str(x.get("catastrophic", x.get("catastrophic_vs_orig"))) == "True"),
                    "worst": min(float(x["ddp_vs_orig"]) for x in both),
                    "dR": float(np.mean([float(x["dR_vs_orig"]) for x in both]))}
    b2 = json.loads((ANA / "O1_b2_reference.json").read_text())
    agg["B2"] = {"PASS": b2["B2_seeds0_4"]["PASS"], "n": 25,
                 "cat": b2["B2_seeds0_4"]["catastrophic"],
                 "worst": min([v[1] for v in baselines()["Adaptive B2"]["per"].values()] +
                              [r["ddp"] for r in b2["B2_seeds34"]["per_unit"]]),
                 "dR": float(np.mean([baselines()["Adaptive B2"]["per"][k][0] for k in baselines()["Adaptive B2"]["per"]]
                                     + [r["dR"] for r in b2["B2_seeds34"]["per_unit"]]))}
    order = ["B2", "C1", "C2", "C3", "C4", "C5"]
    fig, axs = plt.subplots(1, 2, figsize=(12.6, 4.5))
    xs = np.arange(len(order)); w = 0.38
    axs[0].bar(xs - w / 2, [agg[c]["PASS"] for c in order], w, label="PASS/25", edgecolor="black", lw=0.5,
               color="#4C72B0")
    axs[0].bar(xs + w / 2, [agg[c]["cat"] for c in order], w, label="catastrophic/25", edgecolor="black",
               lw=0.5, color="#C44E52")
    axs[0].set_xticks(xs); axs[0].set_xticklabels(order); axs[0].legend(fontsize=8)
    axs[0].set_title("(a) PASS count and catastrophic count over 25 units (5 cats x seeds 0-4)", fontsize=9.5)
    axs[0].grid(alpha=0.3, axis="y")
    axs[1].bar(xs, [agg[c]["worst"] for c in order], edgecolor="black", lw=0.5, color="#55A868")
    axs[1].axhline(-0.25, ls="--", c="black", lw=1.1, label="catastrophic threshold")
    axs[1].set_xticks(xs); axs[1].set_xticklabels(order); axs[1].legend(fontsize=8)
    axs[1].set_ylabel("worst-case dd' over 25 units")
    axs[1].set_title("(b) worst-case downside (higher = safer)", fontsize=9.5)
    axs[1].grid(alpha=0.3, axis="y")
    fig.suptitle("Exp14 Fig6 - O1 extension: 25-unit comparison (B2 baseline vs late-fusion candidates)", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(FIG / "fig6_o1_25units.png", dpi=170, bbox_inches="tight"); plt.close(fig)
    print("fig6 aggregates:", {k: (v["PASS"], v["cat"], round(v["worst"], 4)) for k, v in agg.items()})


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    bl = baselines()
    rows = load(ANA / "AB_per_unit.csv") + load(ANA / "C_late_fusion_per_unit.csv")
    for r in rows:
        r.setdefault("family", "C_late_fusion")
        r["candidate"] = r["candidate"].split("__")[0]
        r.setdefault("catastrophic", r.get("catastrophic_vs_orig"))
    agg = cand_agg(rows)
    top = sorted(agg, key=lambda c: (-agg[c]["PASS"], agg[c]["cat"], -agg[c]["worst_ddp"]))[:5]
    fig1(bl, agg); fig2(agg); fig34(agg, top); fig5(bl, agg, top); fig6()
    json.dump({"baselines": {k: {kk: vv for kk, vv in v.items() if kk != "per"} for k, v in bl.items()},
               "candidates": {k: {kk: vv for kk, vv in v.items() if kk != "per"} for k, v in agg.items()},
               "top5": top}, open(ANA / "fig_summary.json", "w"), indent=2, ensure_ascii=False)
    print("figures ->", FIG)
    print("baselines:", {k: (round(v["absz"], 4), round(v["dprime"], 4)) for k, v in bl.items()})
    print("top5:", top)


if __name__ == "__main__":
    main()
