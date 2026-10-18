#!/usr/bin/env python
"""E25 analysis - aggregate RobustAD cells into summary tables + descriptive verdict.

Reads results/e25/cells/*.json (per-split, per-image scores written by e25_robustad_eval.py)
and produces:
  results/e25/e25_summary.csv          per split, with paired-bootstrap CIs
  results/e25/e25_domain_summary.csv   per domain (valid target splits only)
  results/e25/e25_group_summary.csv    per condition group
  results/e25/analysis.json            machine-readable full summary
  results/e25/final_verdict.json       descriptive verdict (NO pre-registered gate exists)

Guards
------
* Recomputes AUROC/AP from the stored per-image scores; requires |delta| <= 1e-12.
* Aggregates use ONLY valid=1 splits and ONLY non-source splits (test0 = source reference).
* Paired bootstrap resamples IMAGES (same index set for B0 and B1), n_boot=3000,
  rng=20261018; replicates that lose a class are skipped and counted.
* No method parameter is read from or written to this script.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np  # noqa: E402
from sklearn.metrics import roc_auc_score, average_precision_score  # noqa: E402

OUTDIR = ROOT / "results" / "e25"
CELLDIR = OUTDIR / "cells"
N_BOOT = 3000
RNG = 20261018
TOL = 1e-12
DOMAIN_ORDER = ["PCB", "MetalParts"]
GROUP_ORDER = ["illumination", "illumination_related_white_balance",
               "illumination_related_shadow", "pose", "scale", "background",
               "source_reference", "source_reference_INVALID"]


def auroc_ap(scores, labels) -> tuple[float, float]:
    return roc_auc_score(labels, scores), average_precision_score(labels, scores)


def paired_bootstrap(s0, s1, labels, n_boot=N_BOOT, seed=RNG) -> dict:
    """Percentile CI for dAUROC/dAP, resampling images (paired, same indices)."""
    s0, s1 = np.asarray(s0, float), np.asarray(s1, float)
    y = np.asarray(labels, int)
    n = len(y)
    rng = np.random.default_rng(seed)
    d_au, d_ap, skipped = [], [], 0
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        yy = y[idx]
        if yy.min() == yy.max():
            skipped += 1
            continue
        d_au.append(roc_auc_score(yy, s1[idx]) - roc_auc_score(yy, s0[idx]))
        d_ap.append(average_precision_score(yy, s1[idx])
                    - average_precision_score(yy, s0[idx]))
    d_au, d_ap = np.asarray(d_au), np.asarray(d_ap)
    return {"n_boot_used": int(len(d_au)), "n_boot_skipped": int(skipped),
            "d_auroc_ci95": [float(np.percentile(d_au, 2.5)), float(np.percentile(d_au, 97.5))],
            "d_ap_ci95": [float(np.percentile(d_ap, 2.5)), float(np.percentile(d_ap, 97.5))],
            "p_d_auroc_le_0": float((d_au <= 0).mean()),
            "p_d_ap_le_0": float((d_ap <= 0).mean())}


def load_cells() -> list[dict]:
    cells = []
    for p in sorted(CELLDIR.glob("*.json")):
        c = json.loads(p.read_text())
        if c.get("status") != "OK":
            continue
        cells.append(c)
    if not cells:
        raise SystemExit(f"no cells found in {CELLDIR}")
    cells.sort(key=lambda c: (DOMAIN_ORDER.index(c["domain"]),
                              int(c["split"].replace("test", ""))))
    return cells


def verify(c: dict) -> dict:
    a0, p0 = auroc_ap(c["scores_f0"], c["labels"])
    a1, p1 = auroc_ap(c["scores_r1"], c["labels"])
    diffs = [abs(a0 - c["f0"]["auroc"]), abs(p0 - c["f0"]["ap"]),
             abs(a1 - c["r1"]["auroc"]), abs(p1 - c["r1"]["ap"])]
    return {"key": f"{c['domain']}|{c['split']}", "n": len(c["labels"]),
            "recomputed_auroc_f0": a0, "recomputed_ap_f0": p0,
            "recomputed_auroc_r1": a1, "recomputed_ap_r1": p1,
            "max_abs_diff": max(diffs), "pass": bool(max(diffs) <= TOL)}


def main() -> int:
    cells = load_cells()
    verif = [verify(c) for c in cells]
    if not all(v["pass"] for v in verif):
        print("!! recomputation gate FAILED",
              json.dumps([v for v in verif if not v["pass"]], indent=1)[:2000])
        return 2

    rows = []
    for c in cells:
        ci = paired_bootstrap(c["scores_f0"], c["scores_r1"], c["labels"])
        rows.append({
            "domain": c["domain"], "split": c["split"],
            "condition_group": c["condition_group"], "valid": c["valid"],
            "invalid_reason": c.get("invalid_reason") or "",
            "is_source": int(c["split"] == "test0"),
            "n_normal": c["n_ok"], "n_anomaly": c["n_ng"],
            "auroc_f0": c["f0"]["auroc"], "ap_f0": c["f0"]["ap"],
            "auroc_r1": c["r1"]["auroc"], "ap_r1": c["r1"]["ap"],
            "d_auroc": c["r1"]["auroc"] - c["f0"]["auroc"],
            "d_ap": c["r1"]["ap"] - c["f0"]["ap"],
            "d_auroc_ci_lo": ci["d_auroc_ci95"][0], "d_auroc_ci_hi": ci["d_auroc_ci95"][1],
            "d_ap_ci_lo": ci["d_ap_ci95"][0], "d_ap_ci_hi": ci["d_ap_ci95"][1],
            "ci_excludes_0_auroc": int(ci["d_auroc_ci95"][0] > 0 or ci["d_auroc_ci95"][1] < 0),
            "ci_excludes_0_ap": int(ci["d_ap_ci95"][0] > 0 or ci["d_ap_ci95"][1] < 0),
            "steps_used": c["steps_used"], "runtime_s": c["runtime_s"],
            "n_boot_skipped": ci["n_boot_skipped"]})

    # ---------------- CSVs
    with open(OUTDIR / "e25_summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.6f}" if isinstance(v, float) and k != "invalid_reason"
                            else v) for k, v in r.items()})

    targets = [r for r in rows if r["valid"] == 1 and r["is_source"] == 0]
    sources = [r for r in rows if r["is_source"] == 1]

    dom_rows = []
    for dom in DOMAIN_ORDER:
        sub = [r for r in targets if r["domain"] == dom]
        src = [r for r in sources if r["domain"] == dom]
        if not sub:
            continue
        dom_rows.append({
            "domain": dom, "n_target_splits": len(sub),
            "auroc_f0_mean": float(np.mean([r["auroc_f0"] for r in sub])),
            "auroc_r1_mean": float(np.mean([r["auroc_r1"] for r in sub])),
            "d_auroc_mean": float(np.mean([r["d_auroc"] for r in sub])),
            "d_auroc_median": float(np.median([r["d_auroc"] for r in sub])),
            "d_ap_mean": float(np.mean([r["d_ap"] for r in sub])),
            "d_ap_median": float(np.median([r["d_ap"] for r in sub])),
            "n_pos_auroc": int(sum(r["d_auroc"] > 0 for r in sub)),
            "n_pos_ap": int(sum(r["d_ap"] > 0 for r in sub)),
            "worst_split_auroc": min(sub, key=lambda r: r["d_auroc"])["split"],
            "worst_d_auroc": min(r["d_auroc"] for r in sub),
            "best_split_auroc": max(sub, key=lambda r: r["d_auroc"])["split"],
            "best_d_auroc": max(r["d_auroc"] for r in sub),
            "source_split": src[0]["split"] if src else "",
            "source_auroc_f0": src[0]["auroc_f0"] if src else float("nan"),
            "source_auroc_r1": src[0]["auroc_r1"] if src else float("nan"),
            "source_valid": src[0]["valid"] if src else "",
            "source_d_auroc": src[0]["d_auroc"] if src else float("nan")})
    with open(OUTDIR / "e25_domain_summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(dom_rows[0]))
        w.writeheader()
        for r in dom_rows:
            w.writerow({k: (f"{v:.6f}" if isinstance(v, float) else v) for k, v in r.items()})

    grp_rows = []
    for g in GROUP_ORDER:
        sub = [r for r in rows if r["condition_group"] == g]
        if not sub:
            continue
        grp_rows.append({
            "condition_group": g, "n_splits": len(sub),
            "splits": ";".join(f"{r['domain']}:{r['split']}" for r in sub),
            "valid": sub[0]["valid"],
            "d_auroc_mean": float(np.mean([r["d_auroc"] for r in sub])),
            "d_ap_mean": float(np.mean([r["d_ap"] for r in sub])),
            "n_pos_auroc": int(sum(r["d_auroc"] > 0 for r in sub)),
            "n_ci_excludes_0_auroc": int(sum(r["ci_excludes_0_auroc"] for r in sub))})
    with open(OUTDIR / "e25_group_summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(grp_rows[0]))
        w.writeheader()
        for r in grp_rows:
            w.writerow({k: (f"{v:.6f}" if isinstance(v, float) else v) for k, v in r.items()})

    # ---------------- descriptive verdict (NO pre-registered gate)
    n_t = len(targets)
    pos_a = sum(r["d_auroc"] > 0 for r in targets)
    pos_p = sum(r["d_ap"] > 0 for r in targets)
    frac_a = pos_a / n_t
    dom_means = [d["d_auroc_mean"] for d in dom_rows]
    if frac_a >= 0.70 and min(dom_means) > 0:
        pattern = "POSITIVE_DOMINANT"
    elif frac_a <= 0.30 or max(dom_means) < 0:
        pattern = "NEGATIVE_DOMINANT"
    else:
        pattern = "MIXED"
    from scipy.stats import spearmanr
    rho, p_rho = spearmanr([r["auroc_f0"] for r in targets],
                           [r["d_auroc"] for r in targets])
    rho_ap, p_rho_ap = spearmanr([r["ap_f0"] for r in targets],
                                 [r["d_ap"] for r in targets])
    verdict = {
        "experiment": "E25",
        "title": "RobustAD external generalization test of frozen R1-ILA",
        "date": "2026-10-18",
        "config_source": "results/e17/final_method.json",
        "method": "R1-ILA",
        "BETA": 0.1, "augmentations": "brightness+contrast+gamma", "strength": [0.85, 1.15],
        "lr": 0.001, "max_steps": 300, "patience": 30, "batch_size": 8, "seed": 0,
        "domains_run": sorted({r["domain"] for r in rows}),
        "domain_not_run": {"PiledBags": "not downloaded; 9.41 GB archive vs 6.1 GB free disk"},
        "n_target_splits": n_t,
        "n_target_splits_positive_auroc": pos_a,
        "n_target_splits_positive_ap": pos_p,
        "macro_d_auroc_over_target_splits": float(np.mean([r["d_auroc"] for r in targets])),
        "macro_d_ap_over_target_splits": float(np.mean([r["d_ap"] for r in targets])),
        "macro_d_auroc_over_domains": float(np.mean(dom_means)),
        "n_ci_excludes_0_auroc": int(sum(r["ci_excludes_0_auroc"] for r in targets)),
        "n_ci_excludes_0_ap": int(sum(r["ci_excludes_0_ap"] for r in targets)),
        "descriptive_pattern": pattern,
        "exploratory_baseline_dependence": {
            "n_target_splits": n_t,
            "spearman_rho_auroc_f0_vs_d_auroc": float(rho),
            "p_auroc": float(p_rho),
            "spearman_rho_ap_f0_vs_d_ap": float(rho_ap),
            "p_ap": float(p_rho_ap),
            "note": "EXPLORATORY, not pre-registered: negative rho means the adapter helps most "
                    "where the frozen baseline is weakest (ceiling/headroom pattern already "
                    "reported in E13-B2). n=11 splits, no causal claim."},
        "pre_registered_gate": None,
        "verdict_basis": ("DESCRIPTIVE ONLY. E25 has no pre-registered success threshold. "
                          "descriptive_pattern is mechanical: POSITIVE_DOMINANT if >=70% of "
                          "valid non-source splits have dAUROC>0 AND every domain mean is >0; "
                          "NEGATIVE_DOMINANT if <=30% positive OR every domain mean is <0; "
                          "else MIXED. It is not a hypothesis test."),
        "per_domain": {d["domain"]: {k: v for k, v in d.items() if k != "domain"}
                       for d in dom_rows},
        "per_split": [{k: r[k] for k in ("domain", "split", "condition_group", "valid",
                                        "n_normal", "n_anomaly", "auroc_f0", "ap_f0",
                                        "auroc_r1", "ap_r1", "d_auroc", "d_ap",
                                        "d_auroc_ci_lo", "d_auroc_ci_hi",
                                        "d_ap_ci_lo", "d_ap_ci_hi")} for r in rows],
        "limitations": [
            "single seed (seed 0); adapter-training randomness is NOT averaged out",
            "single backbone (DINOv2-B); single scorer (1-NN-max)",
            "plain image-level AUROC/AP (RobustAD has no `detectable` field) -> not "
            "numerically comparable to the M2AD `official` convention",
            "MetalParts test0 excluded from every aggregate (145 files overlap source train)",
            "PiledBags not run (disk)",
            "bank = 180 source-train normals (protocol-documented bank-size constraint; "
            "TRAIN_N=240 cannot be met) -> adapter train 120 / val 60",
            "no causal claim: a delta here does not identify which shift factor changes"],
        "verification": {"cells": len(verif), "all_recomputed_within_tol": True,
                         "tolerance": TOL,
                         "max_abs_diff": max(v["max_abs_diff"] for v in verif)},
    }
    (OUTDIR / "final_verdict.json").write_text(json.dumps(verdict, indent=2, default=float))
    (OUTDIR / "analysis.json").write_text(json.dumps(
        {"rows": rows, "domains": dom_rows, "groups": grp_rows,
         "verification": verif}, indent=1, default=float))

    print(f"[E25 analysis] cells={len(rows)}  recomputation PASS "
          f"(max|d|={max(v['max_abs_diff'] for v in verif):.3e})")
    print(f"{'domain':12s} {'split':7s} {'group':34s} {'n':>7s} {'AUROC F0 -> R1':>21s} "
          f"{'dAUROC':>9s} {'dAP':>9s}")
    for r in rows:
        print(f"{r['domain']:12s} {r['split']:7s} {r['condition_group'][:33]:34s} "
              f"{r['n_normal']:3d}/{r['n_anomaly']:<3d} {r['auroc_f0']:.5f} -> "
              f"{r['auroc_r1']:.5f} {r['d_auroc']:+.5f} {r['d_ap']:+.5f}"
              f"{'  (valid=0)' if r['valid'] == 0 else ''}")
    print(f"\n[E25 analysis] target splits={n_t} | positive dAUROC={pos_a} dAP={pos_p} | "
          f"macro dAUROC={verdict['macro_d_auroc_over_target_splits']:+.5f} "
          f"dAP={verdict['macro_d_ap_over_target_splits']:+.5f} | pattern={pattern}")
    print(f"[E25 analysis] -> {OUTDIR / 'final_verdict.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
