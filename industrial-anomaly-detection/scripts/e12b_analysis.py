#!/usr/bin/env python
"""E12-B analysis: screening decisions, summary JSON, decision memo, figures (CPU-only)."""
from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "e12b"
FIGS = OUT / "figures"

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

CATS = ["Bird", "Car", "Cube"]
LEARNED = ["S2", "S3", "S4"]


def load_screening():
    rows = {}
    with open(OUT / "e12b_screening.csv") as f:
        for r in csv.DictReader(f):
            rows[(r["category"], r["config"])] = {
                "auroc": float(r["auroc"]), "ap": float(r["ap"]),
                "d_auroc": float(r["d_auroc"]), "d_ap": float(r["d_ap"])}
    return rows


def load_training():
    rows = {}
    with open(OUT / "e12b_training.csv") as f:
        for r in csv.DictReader(f):
            rows[(r["category"], r["candidate"])] = {
                "trainable_params": int(r["trainable_params"]),
                "steps_used": int(r["steps_used"]),
                "best_val_cons": float(r["best_val_cons"]),
                "final_cons": float(r["final_cons"]),
                "std_before": float(r["std_fused_before"]),
                "std_after": float(r["std_fused_after"]),
                "loss_finite": r["loss_finite"] == "True",
                "gate_weights": json.loads(r["gate_weights"]),
            }
    return rows


def load_consistency():
    p = OUT / "e12b_consistency.json"
    if p.exists():
        return json.loads(p.read_text())
    return {}


def summarize(scr, ref):
    """ref = S0 metrics dict per category."""
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
            "worst_d_auroc": float(min(d_au)),
            "per_category_d_auroc": dict(zip(CATS, d_au)),
            "per_category_d_ap": dict(zip(CATS, d_ap)),
        }
    return out


def decide(summary, s1_mean_d_auroc, s1_mean_d_ap):
    """Apply frozen decision rules. Returns {cand: verdict, ...} + reasons."""
    verdicts = {}
    for c in LEARNED:
        s = summary[c]
        keep = (s["mean_d_auroc"] >= 0.010 and s["positive_categories"] >= 2
                and s["worst_d_auroc"] >= -0.010 and s["mean_d_ap"] >= 0)
        if keep:
            # complexity gate vs S1
            gate_a = s["mean_d_auroc"] >= s1_mean_d_auroc + 0.003
            gate_b = (s["mean_d_ap"] >= s1_mean_d_ap + 0.010
                      and s["mean_d_auroc"] >= s1_mean_d_auroc - 0.002)
            if gate_a or gate_b:
                verdicts[c] = "KEEP"
            else:
                verdicts[c] = "HOLD"  # passed S0 KEEP but failed complexity gate
        else:
            hold = (s["mean_d_auroc"] >= 0.005
                    and all(s["per_category_d_auroc"][cat] >= 0 for cat in CATS)
                    and s["mean_d_ap"] >= -0.005)
            verdicts[c] = "HOLD" if hold else "STOP"
    return verdicts


# ------------------------------------------------------------------ figures
def fig_delta(scr, ref, metric, fname):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    x = np.arange(len(CATS))
    width = 0.2
    key = "auroc" if metric == "auroc" else "ap"
    colors = {"S1": "#999999", "S2": "#2c7fb8", "S3": "#41b6c4", "S4": "#7a0177"}
    for j, c in enumerate(["S1"] + LEARNED):
        vals = []
        for cat in CATS:
            if c == "S1":
                vals.append(scr[(cat, "S1")][key] - ref[cat][key])
            else:
                vals.append(scr[(cat, c)][key] - ref[cat][key])
        ax.bar(x + (j - 1.5) * width, vals, width, label=c, color=colors[c])
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(CATS)
    ax.set_ylabel(f"Δ {metric.upper()} vs S0")
    ax.set_title(f"E12-B Δ{metric.upper()} by category")
    ax.legend(ncol=4, fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGS / fname, dpi=150)
    plt.close(fig)


def fig_weights(trn, fname):
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for j, c in enumerate(LEARNED):
        ax = axes[j]
        vals = []
        for cat in CATS:
            w = trn[(cat, c)]["gate_weights"]
            if c == "S2":
                vals.append(w["w_block9"])
            elif c == "S3":
                vals.append(w["lambda_mean"])
            else:
                vals.append(w["w_block9_mean"])
        ax.bar(CATS, vals, color=["#2c7fb8", "#41b6c4", "#7a0177"][j])
        ax.axhline(0.5, color="grey", ls="--", lw=1)
        ax.set_ylim(0.4, 0.8)
        ax.set_title(f"{c} learned block9 weight")
        ax.set_ylabel("w_block9 (init 0.5)")
    fig.suptitle("E12-B learned scale weights (all tilt toward block9)")
    fig.tight_layout()
    fig.savefig(FIGS / fname, dpi=150)
    plt.close(fig)


def fig_consistency(trn, cons, fname):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    x = np.arange(len(CATS))
    width = 0.2
    colors = {"S2": "#2c7fb8", "S3": "#41b6c4", "S4": "#7a0177"}
    for j, c in enumerate(LEARNED):
        after = [trn[(cat, c)]["final_cons"] for cat in CATS]
        ax.bar(x + (j - 1) * width, after, width, label=f"{c} after", color=colors[c])
    before = [cons[cat]["fixed_equal_weight_consistency"] for cat in CATS] if cons else None
    if before:
        ax.bar(x + 1.5 * width, before, width, label="fixed equal-weight (before)",
               color="#999999")
    ax.set_xticks(x)
    ax.set_xticklabels(CATS)
    ax.set_ylabel("illumination consistency (1-cos, lower=better)")
    ax.set_title("E12-B illumination consistency before/after")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(FIGS / fname, dpi=150)
    plt.close(fig)


def main() -> int:
    FIGS.mkdir(parents=True, exist_ok=True)
    scr = load_screening()
    trn = load_training()
    cons = load_consistency()

    ref = {cat: {"auroc": scr[(cat, "S0")]["auroc"], "ap": scr[(cat, "S0")]["ap"]}
           for cat in CATS}
    s1 = {cat: {"auroc": scr[(cat, "S1")]["auroc"], "ap": scr[(cat, "S1")]["ap"]}
          for cat in CATS}
    s1_mean_d_auroc = float(np.mean([s1[c]["auroc"] - ref[c]["auroc"] for c in CATS]))
    s1_mean_d_ap = float(np.mean([s1[c]["ap"] - ref[c]["ap"] for c in CATS]))

    summary = summarize(scr, ref)
    verdicts = decide(summary, s1_mean_d_auroc, s1_mean_d_ap)

    # overall verdict
    keeps = [c for c, v in verdicts.items() if v == "KEEP"]
    holds = [c for c, v in verdicts.items() if v == "HOLD"]
    if keeps:
        overall = "KEEP"
        keep_one = keeps[0] if len(keeps) == 1 else "S2"  # simplest
    elif holds:
        overall = "HOLD"
        keep_one = "S2" if "S2" in holds else holds[0]  # simplest (1 param)
    else:
        overall = "NO_SURVIVOR"
        keep_one = None

    result = {
        "experiment": "E12-B",
        "overall_verdict": overall,
        "kept_candidate": keep_one,
        "verdicts": verdicts,
        "summary": summary,
        "s1_mean_d_auroc": s1_mean_d_auroc,
        "s1_mean_d_ap": s1_mean_d_ap,
        "ref": ref,
        "s1": s1,
    }
    (OUT / "e12b_summary.json").write_text(json.dumps(result, indent=2, default=float))

    fig_delta(scr, ref, "auroc", "fig1_dauroc_by_category.png")
    fig_delta(scr, ref, "ap", "fig2_dap_by_category.png")
    fig_weights(trn, "fig3_learned_scale_weights.png")
    fig_consistency(trn, cons, "fig4_consistency_before_after.png")

    print(json.dumps({k: v for k, v in result.items() if k != "summary"}, indent=2,
                     default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
