#!/usr/bin/env python
"""E21 analysis: pooling, paired image bootstrap, gate decision, figures.

Frozen definitions (results/e21/protocol.md §5/§7):
  - per-category 4-view pooled: concat per-image (score,label) over the 4 views (same score space)
  - per-view 3-category pooled: unweighted mean of the 3 category AUROCs for that view
  - overall pooled (primary): unweighted mean over categories of the 4-view pooled delta
  - bootstrap: paired image bootstrap, n_boot=3000, RNG seed fixed = 20261015 + cell index
  - labels = official image_anomaly; detectable != False filter (raw boolean), as official_metrics

Reads: results/e21/e21_partial.json (all 4 views' per-image scores, incl. view120 which is
bit-identical with E16 history). Writes: results/e21/{analysis_*.csv, bootstrap_ci.csv,
gate_decision.json, all_seeds.csv, figures/*}.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUTDIR = ROOT / "results" / "e21"
PARTIAL = OUTDIR / "e21_partial.json"
FIGDIR = OUTDIR / "figures"

VIEWS = ["090", "120", "240", "330"]
CATS = ["Bird", "Car", "Cube"]
N_BOOT = 3000
BOOT_SEED_BASE = 20261015

from sklearn.metrics import roc_auc_score, average_precision_score  # noqa: E402


def auroc_ap(y, s):
    y = np.asarray(y, int)
    s = np.asarray(s, float)
    if len(set(y.tolist())) < 2:
        return float("nan"), float("nan")
    return float(roc_auc_score(y, s)), float(average_precision_score(y, s))


def paired_bootstrap_delta(y, s0, s1, n_boot=N_BOOT, seed=0):
    """Paired image bootstrap of dAUROC = AUROC(s1)-AUROC(s0); same index resample."""
    y = np.asarray(y, int)
    s0 = np.asarray(s0, float)
    s1 = np.asarray(s1, float)
    n = len(y)
    rng = np.random.default_rng(seed)
    d = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        yy = y[idx]
        if len(set(yy.tolist())) < 2:
            continue
        d.append(roc_auc_score(yy, s1[idx]) - roc_auc_score(yy, s0[idx]))
    d = np.asarray(d)
    return {"delta": float(np.mean(d)), "lo": float(np.percentile(d, 2.5)),
            "hi": float(np.percentile(d, 97.5)), "n_eff": int(len(d))}


def cell_view(partial, cat, seed, view):
    """Filtered per-image arrays for (cat, seed, view): scores f0/r1 + labels."""
    c = partial["cells"][f"{cat}|{seed}"]["views"][view]
    det = np.asarray([x is not False for x in c["detectable_raw"]], bool)
    y = np.asarray(c["labels"], int)[det]
    s0 = np.asarray(c["scores_f0"], float)[det]
    s1 = np.asarray(c["scores_r1"], float)[det]
    return y, s0, s1


def main() -> int:
    partial = json.load(open(PARTIAL))
    seeds = sorted({int(k.split("|")[1]) for k in partial["cells"]})
    print(f"cells present: {sorted(partial['cells'].keys())}")

    # ---------------- A. per-cell (all seeds)
    cell_rows = []
    for cat in CATS:
        for seed in seeds:
            if f"{cat}|{seed}" not in partial["cells"]:
                continue
            for v in VIEWS:
                y, s0, s1 = cell_view(partial, cat, seed, v)
                a0, p0 = auroc_ap(y, s0)
                a1, p1 = auroc_ap(y, s1)
                cell_rows.append({
                    "category": cat, "view": v, "seed": seed,
                    "n_good": int((y == 0).sum()), "n_anomaly": int((y == 1).sum()),
                    "auroc_f0": a0, "auroc_r1": a1, "d_auroc": a1 - a0,
                    "ap_f0": p0, "ap_r1": p1, "d_ap": p1 - p0})

    # ---------------- B. per-category 4-view pooled (per seed)
    cat_pool = []
    for cat in CATS:
        for seed in seeds:
            if f"{cat}|{seed}" not in partial["cells"]:
                continue
            ys, s0s, s1s = [], [], []
            for v in VIEWS:
                y, s0, s1 = cell_view(partial, cat, seed, v)
                ys.append(y); s0s.append(s0); s1s.append(s1)
            y = np.concatenate(ys); s0 = np.concatenate(s0s); s1 = np.concatenate(s1s)
            a0, p0 = auroc_ap(y, s0)
            a1, p1 = auroc_ap(y, s1)
            cat_pool.append({"category": cat, "seed": seed, "n": len(y),
                             "n_good": int((y == 0).sum()), "n_anomaly": int((y == 1).sum()),
                             "pooled_auroc_f0": a0, "pooled_auroc_r1": a1, "d_auroc": a1 - a0,
                             "pooled_ap_f0": p0, "pooled_ap_r1": p1, "d_ap": p1 - p0})

    # ---------------- C. per-view 3-category pooled (per seed) = unweighted mean
    view_pool = []
    for v in VIEWS:
        for seed in seeds:
            rows = [r for r in cell_rows if r["view"] == v and r["seed"] == seed]
            if len(rows) != 3:
                continue
            view_pool.append({"view": v, "seed": seed,
                              "mean_auroc_f0": float(np.mean([r["auroc_f0"] for r in rows])),
                              "mean_auroc_r1": float(np.mean([r["auroc_r1"] for r in rows])),
                              "d_auroc": float(np.mean([r["d_auroc"] for r in rows])),
                              "d_ap": float(np.mean([r["d_ap"] for r in rows]))})

    # ---------------- D. overall pooled (primary = macro over categories of 4-view pooled)
    overall = []
    for seed in seeds:
        rows = [r for r in cat_pool if r["seed"] == seed]
        if len(rows) != 3:
            continue
        sel = [r for r in cell_rows if r["seed"] == seed]
        overall.append({
            "seed": seed,
            "macro_pooled_d_auroc": float(np.mean([r["d_auroc"] for r in rows])),
            "macro_pooled_d_ap": float(np.mean([r["d_ap"] for r in rows])),
            "cell_mean_d_auroc": float(np.mean([r["d_auroc"] for r in sel])),
            "weighted_pooled_d_auroc": float(np.average(
                [r["d_auroc"] for r in rows], weights=[r["n"] for r in rows])),
            "n_cats_pooled_auroc_pos": int(sum(1 for r in rows if r["d_auroc"] > 0)),
            "n_cells_pos": int(sum(1 for r in sel if r["d_auroc"] > 0)),
            "min_cell_d_auroc": float(min(r["d_auroc"] for r in sel)),
        })

    # ---------------- E. bootstrap CI
    boot = []
    idx = 0
    for r in cell_rows:
        idx += 1
        y, s0, s1 = cell_view(partial, r["category"], r["seed"], r["view"])
        b = paired_bootstrap_delta(y, s0, s1, seed=BOOT_SEED_BASE + idx * 3)
        boot.append({"level": "cell", "key": f"{r['category']}|v{r['view']}|s{r['seed']}",
                     "delta_auroc_point": r["d_auroc"], **b})
    for r in cat_pool:
        idx += 1
        ys, s0s, s1s = [], [], []
        for v in VIEWS:
            y, s0, s1 = cell_view(partial, r["category"], r["seed"], v)
            ys.append(y); s0s.append(s0); s1s.append(s1)
        b = paired_bootstrap_delta(np.concatenate(ys), np.concatenate(s0s),
                                   np.concatenate(s1s), seed=BOOT_SEED_BASE + idx * 3)
        boot.append({"level": "category_pooled", "key": f"{r['category']}|pooled|s{r['seed']}",
                     "delta_auroc_point": r["d_auroc"], **b})
    # overall: stratified bootstrap (resample within each category, macro mean)
    for seed in seeds:
        if len([r for r in cat_pool if r["seed"] == seed]) != 3:
            continue
        idx += 1
        rng = np.random.default_rng(BOOT_SEED_BASE + idx * 3)
        per_cat = {}
        for cat in CATS:
            ys, s0s, s1s = [], [], []
            for v in VIEWS:
                y, s0, s1 = cell_view(partial, cat, seed, v)
                ys.append(y); s0s.append(s0); s1s.append(s1)
            per_cat[cat] = (np.concatenate(ys), np.concatenate(s0s), np.concatenate(s1s))
        ds = []
        for _ in range(N_BOOT):
            deltas = []
            ok = True
            for cat in CATS:
                y, s0, s1 = per_cat[cat]
                j = rng.integers(0, len(y), len(y))
                yy = y[j]
                if len(set(yy.tolist())) < 2:
                    ok = False
                    break
                deltas.append(roc_auc_score(yy, s1[j]) - roc_auc_score(yy, s0[j]))
            if ok:
                ds.append(float(np.mean(deltas)))
        ds = np.asarray(ds)
        ov = [r for r in overall if r["seed"] == seed][0]
        boot.append({"level": "overall_stratified", "key": f"overall|s{seed}",
                     "delta_auroc_point": ov["macro_pooled_d_auroc"],
                     "delta": float(np.mean(ds)), "lo": float(np.percentile(ds, 2.5)),
                     "hi": float(np.percentile(ds, 97.5)), "n_eff": int(len(ds))})

    # ---------------- F. gate decision (protocol §7; Round 1 = seed0)
    gate = None
    seed0 = sorted(seeds)[0]
    o = [r for r in overall if r["seed"] == seed0]
    if o:
        o = o[0]
        cond_auroc = o["macro_pooled_d_auroc"] > 0
        cond_ap = o["macro_pooled_d_ap"] > 0
        cond_cats = o["n_cats_pooled_auroc_pos"] >= 2
        cond_catastrophic = o["min_cell_d_auroc"] >= -0.03
        if not cond_catastrophic:
            decision = "STOP"
        elif (not cond_auroc) and (not cond_ap):
            decision = "STOP"
        elif o["n_cats_pooled_auroc_pos"] <= 1:
            decision = "HOLD" if cond_auroc else "STOP"
        elif cond_auroc and cond_ap and cond_cats:
            decision = "GO"
        else:
            decision = "HOLD"
        gate = {"seed0_macro_pooled_d_auroc": o["macro_pooled_d_auroc"],
                "seed0_macro_pooled_d_ap": o["macro_pooled_d_ap"],
                "n_cats_pooled_auroc_pos": o["n_cats_pooled_auroc_pos"],
                "min_cell_d_auroc": o["min_cell_d_auroc"],
                "cond_pooled_auroc_pos": bool(cond_auroc),
                "cond_pooled_ap_pos": bool(cond_ap),
                "cond_2of3_cat_pos": bool(cond_cats),
                "cond_no_catastrophic": bool(cond_catastrophic),
                "decision": decision}

    # ---------------- write outputs
    def wcsv(path, rows):
        if not rows:
            return
        cols = list(rows[0].keys())
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            for r in rows:
                w.writerow({c: (f"{r[c]:.6f}" if isinstance(r[c], float) else r[c]) for c in cols})

    wcsv(OUTDIR / "analysis_cells.csv", cell_rows)
    wcsv(OUTDIR / "analysis_category_pooled.csv", cat_pool)
    wcsv(OUTDIR / "analysis_view_pooled.csv", view_pool)
    wcsv(OUTDIR / "analysis_overall.csv", overall)
    wcsv(OUTDIR / "bootstrap_ci.csv", boot)
    if gate:
        json.dump(gate, open(OUTDIR / "gate_decision.json", "w"), indent=1)

    if len(seeds) >= 2:
        with open(OUTDIR / "all_seeds.csv", "w", newline="") as f:
            cols = ["category", "view", "seed", "n_good", "n_anomaly", "auroc_f0", "auroc_r1",
                    "d_auroc", "ap_f0", "ap_r1", "d_ap"]
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            for r in cell_rows:
                w.writerow({c: (f"{r[c]:.6f}" if isinstance(r[c], float) else r[c]) for c in cols})

    # ---------------- print summary
    print("\n=== per-cell (seed0) ===")
    for r in cell_rows:
        if r["seed"] == seed0:
            print(f"  {r['category']:4s} v{r['view']}: F0={r['auroc_f0']:.5f} R1={r['auroc_r1']:.5f} "
                  f"dAUROC={r['d_auroc']:+.5f} dAP={r['d_ap']:+.5f}")
    print("\n=== per-category 4-view pooled (seed0) ===")
    for r in cat_pool:
        if r["seed"] == seed0:
            print(f"  {r['category']:4s} n={r['n']}: pooled F0={r['pooled_auroc_f0']:.5f} "
                  f"R1={r['pooled_auroc_r1']:.5f} dAUROC={r['d_auroc']:+.5f} dAP={r['d_ap']:+.5f}")
    print("\n=== per-view 3-category mean (seed0) ===")
    for r in view_pool:
        if r["seed"] == seed0:
            print(f"  v{r['view']}: mean dAUROC={r['d_auroc']:+.5f} dAP={r['d_ap']:+.5f}")
    print("\n=== overall (seed0) ===")
    if o:
        for k, v in o.items():
            print(f"  {k}: {v}")
    print("\n=== GATE DECISION ===")
    if gate:
        print(json.dumps(gate, indent=1))
    print(f"\n[E21 analysis] wrote analysis_*.csv / bootstrap_ci.csv / gate_decision.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
