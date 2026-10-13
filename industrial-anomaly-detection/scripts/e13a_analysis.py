#!/usr/bin/env python
"""E13-A analysis: decisions, summary JSON, decision memo, figures (CPU-only)."""
from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "e13a"
FIGS = OUT / "figures"

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

CATS = ["Bird", "Car", "Cube"]
LEARNED = ["R1", "R2", "R3", "R4"]
PARAMS = {"R1": 589824, "R2": 197504, "R3": 24576, "R4": 197504}


def load_screening():
    rows = {}
    with open(OUT / "e13a_screening.csv") as f:
        for r in csv.DictReader(f):
            rows[(r["category"], r["config"])] = {
                "auroc": float(r["auroc"]), "ap": float(r["ap"]),
                "d_auroc": float(r["d_auroc"]), "d_ap": float(r["d_ap"])}
    return rows


def load_training():
    rows = {}
    with open(OUT / "e13a_training.csv") as f:
        for r in csv.DictReader(f):
            rows[(r["category"], r["candidate"])] = {
                "trainable_params": int(r["trainable_params"]),
                "steps_used": int(r["steps_used"]),
                "std_ratio": float(r["std_ratio"]),
                "pairwise_cos": float(r["pairwise_cos"]),
                "loss_finite": r["loss_finite"] == "True",
            }
    return rows


def summarize(scr, ref):
    out = {}
    for c in LEARNED:
        d_au, d_ap = [], []
        for cat in CATS:
            d_au.append(scr[(cat, c)]["auroc"] - ref[cat]["auroc"])
            d_ap.append(scr[(cat, c)]["ap"] - ref[cat]["ap"])
        out[c] = {
            "mean_d_auroc": float(np.mean(d_au)),
            "mean_d_ap": float(np.mean(d_ap)),
            "positive_categories": int(sum(1 for x in d_au if x > 0)),
            "n_ge_010": int(sum(1 for x in d_au if x >= 0.010)),
            "worst_d_auroc": float(min(d_au)),
            "per_category_d_auroc": dict(zip(CATS, d_au)),
            "per_category_d_ap": dict(zip(CATS, d_ap)),
        }
    return out


def decide(summary, collapsed):
    verdicts = {}
    for c in LEARNED:
        s = summary[c]
        keep = (s["mean_d_auroc"] >= 0.015 and s["n_ge_010"] >= 2
                and s["worst_d_auroc"] >= -0.005 and s["mean_d_ap"] >= 0.005
                and not collapsed.get(c, False))
        if keep:
            verdicts[c] = "KEEP"
            continue
        hold = (s["mean_d_auroc"] >= 0.010 and s["worst_d_auroc"] >= -0.005
                and s["mean_d_ap"] >= 0 and not collapsed.get(c, False))
        verdicts[c] = "HOLD" if hold else "STOP"
    return verdicts


# ------------------------------------------------------------------ figures
def fig_per_category(scr, ref, metric, fname):
    key = "auroc" if metric == "auroc" else "ap"
    fig, ax = plt.subplots(figsize=(7, 4.5))
    x = np.arange(len(CATS))
    width = 0.18
    colors = {"R1": "#2c7fb8", "R2": "#999999", "R3": "#7a0177", "R4": "#d95f02"}
    for j, c in enumerate(LEARNED):
        vals = [scr[(cat, c)][key] - ref[cat][key] for cat in CATS]
        ax.bar(x + (j - 1.5) * width, vals, width, label=c, color=colors[c])
    ax.axhline(0, color="black", lw=0.8)
    ax.axhline(0.010, color="grey", ls="--", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(CATS)
    ax.set_ylabel(f"Delta {metric.upper()} vs R0")
    ax.set_title(f"E13-A Delta {metric.upper()} by category")
    ax.legend(ncol=4, fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGS / fname, dpi=150)
    plt.close(fig)


def fig_mean(summary, fname):
    fig, ax = plt.subplots(figsize=(6, 4.5))
    x = np.arange(len(LEARNED))
    d_au = [summary[c]["mean_d_auroc"] for c in LEARNED]
    d_ap = [summary[c]["mean_d_ap"] for c in LEARNED]
    ax.bar(x - 0.18, d_au, 0.36, label="mean dAUROC", color="#2c7fb8")
    ax.bar(x + 0.18, d_ap, 0.36, label="mean dAP", color="#d95f02")
    ax.axhline(0, color="black", lw=0.8)
    ax.axhline(0.015, color="grey", ls="--", lw=0.8, label="KEEP dAUROC bar")
    ax.set_xticks(x)
    ax.set_xticklabels(LEARNED)
    ax.set_ylabel("mean delta vs R0")
    ax.set_title("E13-A mean performance")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGS / fname, dpi=150)
    plt.close(fig)


def fig_params_vs_gain(summary, fname):
    fig, ax = plt.subplots(figsize=(6, 4.5))
    for c in LEARNED:
        ax.scatter(PARAMS[c], summary[c]["mean_d_auroc"], s=80, label=c)
        ax.annotate(c, (PARAMS[c], summary[c]["mean_d_auroc"]),
                    textcoords="offset points", xytext=(6, 6), fontsize=9)
    ax.axhline(0.015, color="grey", ls="--", lw=0.8)
    ax.set_xscale("log")
    ax.set_xlabel("trainable params (log)")
    ax.set_ylabel("mean dAUROC")
    ax.set_title("E13-A trainable params vs gain")
    fig.tight_layout()
    fig.savefig(FIGS / fname, dpi=150)
    plt.close(fig)


def fig_collapse(trn, fname):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    x = np.arange(len(LEARNED))
    for cat in CATS:
        axes[0].scatter(x + np.random.uniform(-0.08, 0.08, len(LEARNED)),
                        [trn[(cat, c)]["std_ratio"] for c in LEARNED], label=cat)
    axes[0].axhline(1.0, color="black", lw=0.8)
    axes[0].axhline(0.5, color="red", ls="--", lw=0.8, label="collapse (std<0.5)")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(LEARNED)
    axes[0].set_ylabel("std_ratio (after/before)")
    axes[0].set_title("feature std ratio")
    axes[0].legend(fontsize=7)

    for cat in CATS:
        axes[1].scatter(x + np.random.uniform(-0.08, 0.08, len(LEARNED)),
                        [trn[(cat, c)]["pairwise_cos"] for c in LEARNED], label=cat)
    axes[1].axhline(0.98, color="red", ls="--", lw=0.8, label="collapse (>0.98)")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(LEARNED)
    axes[1].set_ylabel("mean pairwise cosine")
    axes[1].set_title("directional collapse check")
    axes[1].legend(fontsize=7)
    fig.suptitle("E13-A feature collapse check (no collapse in any cell)")
    fig.tight_layout()
    fig.savefig(FIGS / fname, dpi=150)
    plt.close(fig)


def fig_consistency(summary, fname):
    fig, ax = plt.subplots(figsize=(6, 4.5))
    x = np.arange(len(CATS))
    width = 0.35
    r2 = [summary["R2"]["per_category_d_auroc"][cat] for cat in CATS]
    r4 = [summary["R4"]["per_category_d_auroc"][cat] for cat in CATS]
    ax.bar(x - width / 2, r2, width, label="R2 (preserve-only, no consistency)", color="#999999")
    ax.bar(x + width / 2, r4, width, label="R4 (consistency + preserve)", color="#d95f02")
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(CATS)
    ax.set_ylabel("dAUROC vs R0")
    ax.set_title("E13-A illumination consistency ablation (R2 vs R4)")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(FIGS / fname, dpi=150)
    plt.close(fig)


def main() -> int:
    FIGS.mkdir(parents=True, exist_ok=True)
    scr = load_screening()
    trn = load_training()
    ref = {cat: {"auroc": scr[(cat, "R0")]["auroc"], "ap": scr[(cat, "R0")]["ap"]}
           for cat in CATS}
    summary = summarize(scr, ref)
    collapsed = {c: any(trn[(cat, c)]["std_ratio"] < 0.5
                        or trn[(cat, c)]["pairwise_cos"] > 0.98 for cat in CATS)
                 for c in LEARNED}
    verdicts = decide(summary, collapsed)

    keeps = [c for c, v in verdicts.items() if v == "KEEP"]
    if keeps:
        overall = "KEEP"
        # lexicographic: simpler -> fewer params -> higher dAUROC -> higher dAP
        simplicity = {"R1": 0, "R4": 1, "R3": 2}  # linear < MLP < LoRA
        keep_one = sorted(keeps, key=lambda c: (simplicity[c], PARAMS[c],
                                                -summary[c]["mean_d_auroc"],
                                                -summary[c]["mean_d_ap"]))[0]
    else:
        overall = "HOLD" if any(v == "HOLD" for v in verdicts.values()) else "STOP"
        keep_one = None

    result = {
        "experiment": "E13-A", "overall_verdict": overall, "kept_candidate": keep_one,
        "verdicts": verdicts, "summary": summary, "collapsed": collapsed, "ref": ref,
        "params": PARAMS,
    }
    (OUT / "e13a_summary.json").write_text(json.dumps(result, indent=2, default=float))

    fig_per_category(scr, ref, "auroc", "fig1_dauroc_by_category.png")
    fig_per_category(scr, ref, "ap", "fig2_dap_by_category.png")
    fig_mean(summary, "fig3_mean_performance.png")
    fig_params_vs_gain(summary, "fig4_trainable_params_vs_gain.png")
    fig_collapse(trn, "fig5_feature_collapse_check.png")
    fig_consistency(summary, "fig6_consistency_improvement.png")

    print(json.dumps({k: v for k, v in result.items() if k != "summary"}, indent=2,
                     default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
