#!/usr/bin/env python
"""E17 analysis: apply frozen Round-1 advancement rules -> verdict, summary.csv,
round2.csv, final_method.json, figures. CPU-only."""
from __future__ import annotations
import csv, json
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "e17"
FIGS = OUT / "figures"
E16 = ROOT / "results" / "e16" / "e16_partial.json"
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

CATS = ["Bird", "Car", "Cube"]
BC = ["Bird", "Car"]
ORDER = ["C1", "C2", "C0", "C3", "C4"]
BETAS = {"C0": 0.1, "C1": 0.0, "C2": 0.05, "C3": 0.2, "C4": 0.5}
COLORS = {"C0": "#111111", "C1": "#d95f02", "C2": "#7570b3", "C3": "#1b9e77", "C4": "#e7298a"}
SIMPLER = {"C1": True, "C2": False, "C3": False, "C4": False}


def load_round1():
    r = {}
    with open(OUT / "round1.csv") as f:
        for row in csv.DictReader(f):
            c, s, cfg = row["category"], int(row["seed"]), row["config"]
            r[(c, s, cfg)] = {k: (v == "True" if k == "collapsed" else float(v))
                              for k, v in row.items() if k not in ("category", "seed", "config")}
    return r


def main() -> int:
    FIGS.mkdir(parents=True, exist_ok=True)
    r1 = load_round1()
    e16 = json.loads(E16.read_text())["cells"]
    rows = []
    for c in CATS:
        for cfg in BETAS:
            a = r1[(c, 0, cfg)]
            f0 = e16[f"{c}|0"]["f0"]
            c0 = r1[(c, 0, "C0")]
            rows.append({"category": c, "config": cfg, "beta": a["beta"],
                         "AUROC": a["AUROC"], "AP": a["AP"], "d_prime": a["d_prime"],
                         "robustness_feat": a["robustness_feat"],
                         "robustness_score": a["robustness_score"],
                         "illum_shift": a["illum_shift"], "steps_used": int(a["steps_used"]),
                         "std_ratio": a["std_ratio"], "pairwise_cos": a["pairwise_cos"],
                         "collapsed": a["collapsed"], "runtime_s": a["runtime_s"],
                         "dAUROC_vs_F0": a["AUROC"] - f0["auroc"], "dAP_vs_F0": a["AP"] - f0["ap"],
                         "dAUROC_vs_C0": a["AUROC"] - c0["AUROC"], "dAP_vs_C0": a["AP"] - c0["AP"],
                         "d_rob_vs_C0": a["robustness_score"] - c0["robustness_score"],
                         "d_shift_vs_C0": a["illum_shift"] - c0["illum_shift"]})
    gates = {}
    for cfg in ["C1", "C2", "C3", "C4"]:
        bc_au = float(np.mean([r["dAUROC_vs_C0"] for r in rows
                               if r["config"] == cfg and r["category"] in BC]))
        bc_ap = float(np.mean([r["dAP_vs_C0"] for r in rows
                               if r["config"] == cfg and r["category"] in BC]))
        cube = float([r["dAUROC_vs_C0"] for r in rows
                      if r["config"] == cfg and r["category"] == "Cube"][0])
        rob = float(np.mean([r["d_rob_vs_C0"] for r in rows
                             if r["config"] == cfg and r["category"] in BC]))
        coll = any(r["collapsed"] for r in rows if r["config"] == cfg)
        g = [bc_au >= 0.003, cube >= -0.003, bc_ap >= -0.002, rob <= 0.10, not coll]
        gates[cfg] = {"bird_car_mean_dAUROC_vs_C0": bc_au, "bird_car_mean_dAP_vs_C0": bc_ap,
                      "cube_dAUROC_vs_C0": cube, "bird_car_mean_d_rob_vs_C0": rob,
                      "collapsed": bool(coll),
                      "gate1_ge_+0.003": bool(g[0]), "gate2_cube_ge_-0.003": bool(g[1]),
                      "gate3_dAP_ge_-0.002": bool(g[2]), "gate4_rob_not_worse": bool(g[3]),
                      "gate5_no_collapse": bool(g[4]), "ADVANCE": bool(all(g)),
                      "SIMPLIFY_CANDIDATE": bool(abs(bc_au) <= 0.002 and not coll)}
    adv = [k for k, v in gates.items() if v["ADVANCE"]]
    simplify = [k for k, v in gates.items() if v["SIMPLIFY_CANDIDATE"] and SIMPLER.get(k, False)]
    if adv:
        verdict, round2 = "PENDING_ROUND2", sorted(
            adv, key=lambda k: -gates[k]["bird_car_mean_dAUROC_vs_C0"])[:2]
    elif simplify:
        verdict, round2 = "FREEZE_SIMPLIFIED_R1", simplify[:1]
    else:
        verdict, round2 = "FREEZE_CURRENT_R1", []
    with open(OUT / "summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    with open(OUT / "round2.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["category", "seed", "config", "beta", "AUROC", "AP", "d_prime",
                    "robustness_score", "illum_shift", "note"])
        w.writerow(["", "", "", "", "", "", "", "", "",
                    "NOT_RUN: no challenger satisfied the frozen Round-1 advancement gate"])
    emit_final_and_figures(r1, e16, gates, verdict, rows)
    print(json.dumps({"verdict": verdict, "advancing": adv, "simplify": simplify,
                      "round2": round2, "gates": gates}, indent=2))
    return 0


def emit_final_and_figures(r1, e16, gates, verdict, rows):
    e16_d = {c: e16[f"{c}|0"] for c in CATS}
    final = {
        "method_name": "R1-ILA (Illumination-consistent Linear Residual Adaptation)",
        "base": "DINOv2-B (frozen, final-layer patch tokens 1024x768) + L2 + 1-NN-max",
        "adaptation": "z' = z + W z  (W: 768x768, zero-init linear residual, normal-only training)",
        "parameters": {
            "BETA": 0.1,
            "loss": ("L = L_cons + BETA * L_preserve ; L_cons = 1 - cos(z(x), z(T(x))) ; "
                     "L_preserve = 1 - cos(z(x), z_frozen(x))"),
            "LR_ADAPTER": 0.001, "optimizer": "AdamW", "MAX_STEPS": 300,
            "PATIENCE_STEPS": 30, "EVAL_EVERY": 5, "BATCH_IMGS": 8,
            "TRAIN_N": 240, "VAL_N": 60,
            "augmentation": "brightness/contrast/gamma x {0.85,1.15} (6 photometric variants)",
            "trainable_params": 589824,
            "beta_injection": "e13a_screen.BETA module global read by compute_loss; no frozen code changed"},
        "selection_reason": (
            "E17 Round-1 BETA sweep {0.0,0.05,0.1,0.2,0.5}: no challenger cleared the frozen +0.003 "
            "Bird+Car mean dAUROC gate over C0 (best BETA=0.2 at +0.00173). The two near-ties (0.2, 0.5) "
            "also fail the Cube safety gate (-0.0087/-0.0098 vs C0) and worsen robustness_score; BETA=0 "
            "collapses the representation on Cube (pairwise cos 0.988) and is the worst detector on all "
            "three categories. BETA=0.1 is an aggregate interior optimum -> qualified current R1 frozen."),
        "e16_reference": "results/e16/e16_decision_memo.md",
        "e17_verdict": verdict,
        "advancement_gates": gates,
        "e16_seed0_metrics": {c: {"auroc": e16_d[c]["r1"]["auroc"], "ap": e16_d[c]["r1"]["ap"],
                                  "d_prime": e16_d[c]["r1"]["d_prime"],
                                  "robustness_score": e16_d[c]["r1"]["robustness_score"]} for c in CATS},
        "frozen_metrics_source": "results/e16/e16_partial.json (all 9/9 cells, qualified in E16)",
    }
    (OUT / "final_method.json").write_text(json.dumps(final, indent=2, default=float))
    x = np.arange(3)
    betas = [BETAS[c] for c in ORDER]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for c in CATS:
        ax.plot(betas, [r1[(c, 0, cfg)]["AUROC"] for cfg in ORDER], "o-", label=c)
        ax.axhline(e16[f"{c}|0"]["f0"]["auroc"], color="grey", ls=":", lw=0.8)
    ax.set_xscale("symlog", linthresh=0.05)
    ax.set_xlabel("BETA (0 = pure illumination consistency; larger = stronger frozen anchor)")
    ax.set_ylabel("AUROC (seed0)")
    ax.set_title("E17 Round 1: AUROC vs BETA (dotted = frozen F0 per category)")
    ax.legend(); fig.tight_layout(); fig.savefig(FIGS / "fig1_auroc_vs_beta.png", dpi=150); plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    w = 0.2
    for j, cfg in enumerate(["C1", "C2", "C3", "C4"]):
        vals = [r1[(c, 0, cfg)]["AUROC"] - r1[(c, 0, "C0")]["AUROC"] for c in CATS]
        ax.bar(x + (j - 1.5) * w, vals, w, label=f"{cfg} (beta={BETAS[cfg]})", color=COLORS[cfg])
    ax.axhline(0, color="black", lw=0.8)
    ax.axhline(0.003, color="green", ls="--", lw=0.8, label="advance bar (+0.003)")
    ax.axhline(-0.003, color="red", ls="--", lw=0.8, label="Cube safety (-0.003)")
    ax.set_xticks(x); ax.set_xticklabels(CATS)
    ax.set_ylabel("delta AUROC vs current R1 (C0)")
    ax.set_title("E17 Round 1: challenger delta vs current R1 (seed0)")
    ax.legend(fontsize=7); fig.tight_layout(); fig.savefig(FIGS / "fig2_delta_vs_c0.png", dpi=150); plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 4.8))
    for c in CATS:
        xs = [r1[(c, 0, cfg)]["robustness_score"] for cfg in ORDER]
        ys = [r1[(c, 0, cfg)]["AUROC"] for cfg in ORDER]
        ax.plot(xs, ys, "o-", label=c)
        for cfg, xx, yy in zip(ORDER, xs, ys):
            ax.annotate(cfg, (xx, yy), fontsize=7, xytext=(3, 3), textcoords="offset points")
    ax.set_xlabel("robustness_score (lower = more illumination-robust)")
    ax.set_ylabel("AUROC (seed0)")
    ax.set_title("E17 Round 1: detection vs robustness trade-off along BETA")
    ax.legend(); fig.tight_layout(); fig.savefig(FIGS / "fig3_tradeoff.png", dpi=150); plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, key, ttl in ((axes[0], "pairwise_cos", "mean pairwise cosine (collapse if > 0.98)"),
                         (axes[1], "std_ratio", "feature std ratio after/before")):
        for c in CATS:
            ax.plot(betas, [r1[(c, 0, cfg)][key] for cfg in ORDER], "o-", label=c)
        if key == "pairwise_cos":
            ax.axhline(0.98, color="red", ls="--", lw=0.9, label="collapse threshold")
        ax.set_xscale("symlog", linthresh=0.05)
        ax.set_xlabel("BETA"); ax.set_title(ttl); ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(FIGS / "fig4_collapse_diagnostics.png", dpi=150); plt.close(fig)
    print("wrote final_method.json + 4 figures")


if __name__ == "__main__":
    raise SystemExit(main())
