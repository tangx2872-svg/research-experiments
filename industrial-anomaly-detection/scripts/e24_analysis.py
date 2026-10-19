#!/usr/bin/env python
"""E24 analysis - photometric-augmentation sensitivity of frozen R1-ILA (M2AD).

Inputs : results/e24/cells/{strength,type,stability}_<label>_<cat>_s<seed>.json
Outputs: results/e24/final_strength_analysis.json
         results/e24/tables/*.csv (ranked summaries)
         results/e24/figures/*.png|pdf

Interpretation rules (fixed BEFORE looking at the numbers, see E24 brief):
  * Stability first: rank by (#positive cells desc, worst cell desc, mean ΔAUROC desc).
  * S15 (= the frozen 0.85/1.15) is the reference; a challenger only becomes a
    "candidate better configuration" if it beats S15 by > 0.002 on the 3-seed mean
    AND is consistent across categories. Nothing here modifies E17/E22/E25.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np  # noqa: E402

OUTDIR = ROOT / "results" / "e24"
CELLDIR = OUTDIR / "cells"
TABDIR = OUTDIR / "tables"
FIGDIR = OUTDIR / "figures"
CATS = ["Bird", "Car", "Cube"]
S15_MARGIN = 0.002

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def load(stage: str) -> list[dict]:
    out = []
    for p in sorted(CELLDIR.glob(f"{stage}_*.json")):
        c = json.loads(p.read_text())
        if c.get("status") == "OK":
            out.append(c)
    return out


def agg(cells: list[dict]) -> dict:
    d_au = [c["d_auroc"] for c in cells]
    d_ap = [c["d_ap"] for c in cells]
    return {"n_cells": len(cells), "d_auroc_mean": float(np.mean(d_au)),
            "d_auroc_median": float(np.median(d_au)), "d_auroc_std": float(np.std(d_au, ddof=1))
            if len(d_au) > 1 else 0.0,
            "d_ap_mean": float(np.mean(d_ap)), "d_ap_median": float(np.median(d_ap)),
            "n_pos_auroc": int(sum(x > 0 for x in d_au)),
            "n_pos_ap": int(sum(x > 0 for x in d_ap)),
            "worst_d_auroc": float(np.min(d_au)), "best_d_auroc": float(np.max(d_au)),
            "worst_cell": min(cells, key=lambda c: c["d_auroc"])["category"],
            "mean_auroc_f0": float(np.mean([c["f0"]["auroc"] for c in cells])),
            "mean_steps": float(np.mean([c["steps_used"] for c in cells])),
            "runtime_s_mean": float(np.mean([c["runtime_s"] for c in cells]))}


def by_label(cells: list[dict]) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}
    for c in cells:
        out.setdefault(c["label"], []).append(c)
    return out


def savefig(fig, name: str) -> None:
    FIGDIR.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(FIGDIR / f"{name}.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    strength = load("strength")
    types = load("type")
    stab = load("stability")
    if not strength:
        raise SystemExit("no strength cells yet")
    TABDIR.mkdir(parents=True, exist_ok=True)
    FIGDIR.mkdir(parents=True, exist_ok=True)

    # ---------------- round 1 (seed 0)
    r1_groups = by_label(strength)
    r1 = {}
    for lab, cells in r1_groups.items():
        a = agg(cells)
        per_cat = {c["category"]: {"d_auroc": c["d_auroc"], "d_ap": c["d_ap"],
                                   "steps": c["steps_used"]} for c in cells}
        r1[lab] = {**a, "per_category": per_cat,
                   "levels": cells[0]["levels"], "families": cells[0]["families"]}
    order = sorted(r1, key=lambda l: (-r1[l]["n_pos_auroc"],
                                      -r1[l]["worst_d_auroc"],
                                      -r1[l]["d_auroc_mean"]))
    top2 = order[:2]

    # ---------------- equivalence: S15 (seed 0) must reproduce the frozen E18 table
    repro = {}
    e18_path = ROOT / "results" / "e18" / "main_table.csv"
    if e18_path.exists() and "S15" in r1_groups:
        import csv as _csv
        rows18 = {r["method"]: r for r in _csv.DictReader(open(e18_path))}
        ours = next((v for k, v in rows18.items() if k.startswith("Ours")), None)
        f0row = rows18.get("F0")
        s15 = {c["category"]: c for c in r1_groups["S15"]}
        checks = {}
        for cat in CATS:
            if cat not in s15 or ours is None:
                continue
            checks[f"{cat}_auroc"] = abs(s15[cat]["r1"]["auroc"] - float(ours[f"{cat}_AUROC"]))
            checks[f"{cat}_ap"] = abs(s15[cat]["r1"]["ap"] - float(ours[f"{cat}_AP"]))
            if f0row:
                checks[f"{cat}_f0_auroc"] = abs(s15[cat]["f0"]["auroc"]
                                                - float(f0row[f"{cat}_AUROC"]))
        repro = {"reference": "results/e18/main_table.csv (Ours = frozen R1-ILA, F0)",
                 "max_abs_diff": max(checks.values()) if checks else None,
                 "per_check": checks,
                 "tolerance": 1e-6,
                 "pass": bool(checks and max(checks.values()) <= 1e-6)}

    # ---------------- row-2 stability (merge seed0 from round 1 if absent)
    stab_groups = by_label(stab)
    stab_all = {}
    for lab in r1:
        cells = list(stab_groups.get(lab, []))
        if lab in r1_groups:
            have = {c["seed"] for c in cells}
            for c in r1_groups[lab]:
                if c["seed"] not in have:
                    cells.append(c)
        if cells:
            seeds = sorted({c["seed"] for c in cells})
            per_cat = {}
            for cat in CATS:
                sub = [c for c in cells if c["category"] == cat]
                if sub:
                    per_cat[cat] = {"d_auroc_mean": float(np.mean([c["d_auroc"] for c in sub])),
                                    "d_auroc_std": float(np.std([c["d_auroc"] for c in sub],
                                                                ddof=1)) if len(sub) > 1 else 0.0,
                                    "d_ap_mean": float(np.mean([c["d_ap"] for c in sub])),
                                    "n_seeds": len(sub),
                                    "seeds": sorted(c["seed"] for c in sub)}
            stab_all[lab] = {"n_cells": len(cells), "seeds": seeds,
                             "d_auroc_mean": float(np.mean([c["d_auroc"] for c in cells])),
                             "d_auroc_std": float(np.std([c["d_auroc"] for c in cells], ddof=1))
                             if len(cells) > 1 else 0.0,
                             "d_ap_mean": float(np.mean([c["d_ap"] for c in cells])),
                             "d_ap_std": float(np.std([c["d_ap"] for c in cells], ddof=1))
                             if len(cells) > 1 else 0.0,
                             "n_pos_auroc": int(sum(c["d_auroc"] > 0 for c in cells)),
                             "per_category": per_cat,
                             "category_consistency": int(sum(
                                 v["d_auroc_mean"] > 0 for v in per_cat.values()))}

    # ---------------- S15 comparison
    s15_r1 = r1.get("S15")
    best_lab = max(r1, key=lambda l: r1[l]["d_auroc_mean"]) if r1 else None
    decision = {"reference": "S15", "margin": S15_MARGIN}
    if s15_r1 and best_lab and best_lab != "S15":
        gain = r1[best_lab]["d_auroc_mean"] - s15_r1["d_auroc_mean"]
        consistent = r1[best_lab]["n_pos_auroc"] >= s15_r1["n_pos_auroc"]
        decision.update({"best_round1": best_lab, "gain_vs_s15": float(gain),
                         "consistent_vs_s15": bool(consistent),
                         "keep_s15": bool(gain <= S15_MARGIN or not consistent),
                         "rule": "keep S15 unless a challenger gains > margin AND is at "
                                 "least as consistent"})
    elif s15_r1:
        decision.update({"best_round1": "S15", "gain_vs_s15": 0.0,
                         "consistent_vs_s15": True, "keep_s15": True,
                         "rule": "S15 is itself the round-1 maximum"})

    # ---------------- type ablation
    t = {}
    for lab, cells in by_label(types).items():
        a = agg(cells)
        t[lab] = {**a, "families": cells[0]["families"],
                  "levels": cells[0]["levels"],
                  "per_category": {c["category"]: c["d_auroc"] for c in cells}}
    best_single = max((l for l in t if len(t[l]["families"]) == 1),
                      key=lambda l: t[l]["d_auroc_mean"], default=None)
    best_double = max((l for l in t if len(t[l]["families"]) == 2),
                      key=lambda l: t[l]["d_auroc_mean"], default=None)

    out = {
        "experiment": "E24 photometric sensitivity (M2AD, frozen R1-ILA)",
        "note": ("EXPLANATORY AUDIT ONLY. No frozen definition is changed; any better "
                 "setting is recorded as a candidate for a future pre-registered study. "
                 "RobustAD target/test results were never used for selection."),
        "data": "M2AD Bird/Car/Cube, view120, 300-bank, official image_anomaly labels",
        "fixed_parameters": {"BETA": 0.1, "lr": 0.001, "max_steps": 300, "patience": 30,
                             "batch_size": 8, "backbone": "DINOv2-B frozen",
                             "scoring": "1-NN-max", "bank": 300, "train/val": [240, 60]},
        "round1_seed0": r1,
        "round1_rank_stability_first": order,
        "top2_by_stability": top2,
        "round2_multiseed": stab_all,
        "s15_decision": decision,
        "frozen_config_reproduction": repro,
        "type_ablation_seed0": t,
        "type_ranking": sorted(t, key=lambda l: -t[l]["d_auroc_mean"]),
        "best_single_type": best_single,
        "best_double_type": best_double,
        "full_type": "A6",
        "verification": {"n_strength_cells": len(strength), "n_type_cells": len(types),
                         "n_stability_cells": len(stab)},
    }
    (OUTDIR / "final_strength_analysis.json").write_text(json.dumps(out, indent=2, default=float))

    # ---------------- tables
    import csv
    with open(TABDIR / "strength_ranked.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["rank", "strength", "levels", "n_cells", "d_auroc_mean", "d_auroc_median",
                    "d_ap_mean", "n_pos_auroc", "worst_d_auroc", "best_d_auroc",
                    "mean_steps", "runtime_s_mean"])
        for i, lab in enumerate(order, 1):
            r = r1[lab]
            w.writerow([i, lab, f"{r['levels'][0]}/{r['levels'][1]}", r["n_cells"],
                        f"{r['d_auroc_mean']:+.6f}", f"{r['d_auroc_median']:+.6f}",
                        f"{r['d_ap_mean']:+.6f}", r["n_pos_auroc"],
                        f"{r['worst_d_auroc']:+.6f}", f"{r['best_d_auroc']:+.6f}",
                        f"{r['mean_steps']:.0f}", f"{r['runtime_s_mean']:.0f}"])
    with open(TABDIR / "type_ranked.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["rank", "type", "families", "n_cells", "d_auroc_mean", "d_ap_mean",
                    "n_pos_auroc", "worst_d_auroc"])
        for i, lab in enumerate(out["type_ranking"], 1):
            v = t[lab]
            w.writerow([i, lab, "+".join(v["families"]), v["n_cells"],
                        f"{v['d_auroc_mean']:+.6f}", f"{v['d_ap_mean']:+.6f}",
                        v["n_pos_auroc"], f"{v['worst_d_auroc']:+.6f}"])
    with open(TABDIR / "stability_multiseed.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["strength", "seeds", "n_cells", "d_auroc_mean", "d_auroc_std",
                    "d_ap_mean", "d_ap_std", "n_pos_auroc", "category_consistency"])
        for lab in sorted(stab_all):
            v = stab_all[lab]
            w.writerow([lab, ",".join(map(str, v["seeds"])), v["n_cells"],
                        f"{v['d_auroc_mean']:+.6f}", f"{v['d_auroc_std']:.6f}",
                        f"{v['d_ap_mean']:+.6f}", f"{v['d_ap_std']:.6f}",
                        v["n_pos_auroc"], v["category_consistency"]])

    # ---------------- figures
    labs = sorted(r1)
    x = np.arange(len(labs))
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    means = [r1[l]["d_auroc_mean"] for l in labs]
    ax.bar(x, means, color="#4C72B0")
    for i, l in enumerate(labs):
        vals = [r1[l]["per_category"][c]["d_auroc"] for c in CATS
                if c in r1[l]["per_category"]]
        ax.scatter(np.full(len(vals), x[i]), vals, color="#C44E52", s=18, zorder=3)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(x, [f"{l}\n{r1[l]['levels'][0]}/{r1[l]['levels'][1]}" for l in labs],
                  fontsize=8)
    ax.set_ylabel("ΔAUROC (R1 − F0)")
    ax.set_title("E24 strength sweep, M²AD seed 0 (bar=mean, dots=categories)")
    savefig(fig, "e24_strength_vs_dauroc")

    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    ax.bar(x, [r1[l]["d_ap_mean"] for l in labs], color="#55A868")
    for i, l in enumerate(labs):
        vals = [r1[l]["per_category"][c]["d_ap"] for c in CATS if c in r1[l]["per_category"]]
        ax.scatter(np.full(len(vals), x[i]), vals, color="#8172B2", s=18, zorder=3)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(x, [f"{l}\n{r1[l]['levels'][0]}/{r1[l]['levels'][1]}" for l in labs],
                  fontsize=8)
    ax.set_ylabel("ΔAP (R1 − F0)")
    ax.set_title("E24 strength sweep, M²AD seed 0 (bar=mean, dots=categories)")
    savefig(fig, "e24_strength_vs_dap")

    if t:
        tl = out["type_ranking"]
        xt = np.arange(len(tl))
        fig, ax = plt.subplots(figsize=(7.4, 3.6))
        ax.bar(xt, [t[l]["d_auroc_mean"] for l in tl], color="#4C72B0", label="ΔAUROC")
        ax.plot(xt, [t[l]["d_ap_mean"] for l in tl], "o-", color="#C44E52", label="ΔAP")
        ax.axhline(0, color="k", lw=0.8)
        ax.set_xticks(xt, [f"{l}\n{'+'.join(f[:1].upper() for f in t[l]['families'])}"
                           for l in tl], fontsize=8)
        ax.set_title("E24 perturbation-type ablation, M²AD seed 0")
        ax.legend(fontsize=8)
        savefig(fig, "e24_type_ablation")

    print("[E24 analysis] round1 ranking (stability first):")
    for i, l in enumerate(order, 1):
        r = r1[l]
        print(f"  {i}. {l} {r['levels']}  mean {r['d_auroc_mean']:+.5f} "
              f"median {r['d_auroc_median']:+.5f} pos {r['n_pos_auroc']}/{r['n_cells']} "
              f"worst {r['worst_d_auroc']:+.5f}")
    print(f"[E24 analysis] top2 = {top2} | S15 decision = {decision}")
    if repro:
        print(f"[E24 analysis] frozen-config reproduction vs E18: "
              f"max|d|={repro['max_abs_diff']:.3e} pass={repro['pass']}")
    if stab_all:
        print("[E24 analysis] multiseed:")
        for lab in sorted(stab_all):
            v = stab_all[lab]
            print(f"  {lab}: {v['d_auroc_mean']:+.5f} ± {v['d_auroc_std']:.5f} "
                  f"(seeds {v['seeds']}, pos {v['n_pos_auroc']}/{v['n_cells']})")
    if t:
        print("[E24 analysis] type ranking:")
        for l in out["type_ranking"]:
            print(f"  {l} {t[l]['families']}: {t[l]['d_auroc_mean']:+.5f} "
                  f"(AP {t[l]['d_ap_mean']:+.5f}, pos {t[l]['n_pos_auroc']}/{t[l]['n_cells']})")
    print(f"[E24 analysis] -> {OUTDIR / 'final_strength_analysis.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
