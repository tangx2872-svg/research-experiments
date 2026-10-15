#!/usr/bin/env python
"""E22 analysis phase: 10-category x 3-seed tables, win rates, bootstrap, illumination,
shift-gain association, failure cases, statistical summary + pre-registered verdict.

Reads results/e22/cells/*.json (written by e22_full_category_validation.py).
Protocol: results/e22/protocol.md  (analysis plan section 7; verdict section 6)

Uncertainty is reported in TWO strictly separated forms:
  (1) image-level paired bootstrap CI  (sampling uncertainty of the test set)
  (2) cross-seed variability (std over the 3 banks/adapters)
  They are NOT interchangeable and are never merged into one interval.

Frozen metric definitions reused (no new metrics invented):
  - official AUROC/AP = detectable != False filtered, labels = metadata image_anomaly
    (e11_p2_runner.official_metrics semantics)
  - illumination spread = max(good_mean) - min(good_mean) over the 10 illumination groups
    (E18-P2 / e18p2_main_table.illumination_breakdown semantics)
  - illumination shift strength (category property) = e16_qualify diagnostics
    D_illum_shift_F0 (patch-level relative shift) and B_illum_group_centroid_dist_F0
"""
from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__import__("os").environ.get("E22_OUT", str(ROOT / "results" / "e22")))
CELLD = Path(__import__("os").environ.get("E22_CELLS", str(OUT / "cells")))
HIST_CATS = ["Bird", "Car", "Cube"]
NEW_CATS = ["Dice", "Doll", "Holder", "Motor", "Ring", "Teapot", "Tube"]
CATS = HIST_CATS + NEW_CATS
SEEDS = [0, 1, 2]
N_BOOT = 3000
BOOT_RNG = 20261015
STRONG_GO_POS_CATS = 7
SEVERE_NEG = -0.03

import numpy as np  # noqa: E402
from sklearn.metrics import roc_auc_score, average_precision_score  # noqa: E402
from scipy import stats as sps  # noqa: E402


# ------------------------------------------------------------------ load
def load_cells() -> tuple[dict, list]:
    cells, missing = {}, []
    for cat in CATS:
        for seed in SEEDS:
            p = CELLD / f"{cat}_seed{seed}.json"
            if not p.exists():
                missing.append(f"{cat}|{seed}")
                continue
            cells[(cat, seed)] = json.loads(p.read_text())
    return cells, missing


def fnum(x, nd=6):
    return f"{x:.{nd}f}" if isinstance(x, (int, float)) and math.isfinite(x) else ""


# ------------------------------------------------------------------ 1. cell-level csv
def write_cells_csv(cells: dict) -> None:
    cols = ["category", "seed", "view", "source", "status", "n_test", "n_good", "n_anomaly",
            "n_filtered", "auroc_f0", "auroc_r1", "d_auroc", "ap_f0", "ap_r1", "d_ap",
            "d_prime_f0", "d_prime_r1", "rob_feat_f0", "rob_feat_r1", "rob_score_f0",
            "rob_score_r1", "bank_hash", "adapter_hash", "steps_used", "runtime_s",
            "peak_gpu_mb", "timestamp", "shift_D_f0", "shift_D_r1", "centroid_B_f0",
            "centroid_B_r1"]
    rows = []
    for (cat, seed), c in sorted(cells.items()):
        d = c.get("diagnostics", {})
        rows.append({"category": cat, "seed": seed, "view": c.get("view", "120"),
                     "source": c.get("source", ""), "status": c.get("status", ""),
                     "n_test": c.get("n_test", ""), "n_good": c["n_ok"], "n_anomaly": c["n_ng"],
                     "n_filtered": c["n_filtered"],
                     "auroc_f0": fnum(c["f0"]["auroc"]), "auroc_r1": fnum(c["r1"]["auroc"]),
                     "d_auroc": fnum(c["r1"]["auroc"] - c["f0"]["auroc"]),
                     "ap_f0": fnum(c["f0"]["ap"]), "ap_r1": fnum(c["r1"]["ap"]),
                     "d_ap": fnum(c["r1"]["ap"] - c["f0"]["ap"]),
                     "d_prime_f0": fnum(c["f0"]["d_prime"], 4),
                     "d_prime_r1": fnum(c["r1"]["d_prime"], 4),
                     "rob_feat_f0": fnum(c["f0"]["robustness_feat"]),
                     "rob_feat_r1": fnum(c["r1"]["robustness_feat"]),
                     "rob_score_f0": fnum(c["f0"]["robustness_score"]),
                     "rob_score_r1": fnum(c["r1"]["robustness_score"]),
                     "bank_hash": c.get("bank_hash", ""), "adapter_hash": c.get("adapter_hash") or "",
                     "steps_used": c.get("steps_used", ""), "runtime_s": c.get("runtime_s", ""),
                     "peak_gpu_mb": c.get("peak_gpu_mb", ""), "timestamp": c.get("timestamp", ""),
                     "shift_D_f0": fnum(d.get("D_illum_shift_F0")),
                     "shift_D_r1": fnum(d.get("D_illum_shift_R1")),
                     "centroid_B_f0": fnum(d.get("B_illum_group_centroid_dist_F0")),
                     "centroid_B_r1": fnum(d.get("B_illum_group_centroid_dist_R1"))})
    with open(OUT / "all10_3seed_cells.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    print(f"[E22 analysis] wrote all10_3seed_cells.csv ({len(rows)} cells)", flush=True)


# ------------------------------------------------------------------ 2. main table
def main_table(cells: dict) -> list:
    rows = []
    for cat in CATS:
        seeds_present = [s for s in SEEDS if (cat, s) in cells]
        if not seeds_present:
            continue
        rec = {"category": cat}
        rec["n_seeds"] = len(seeds_present)
        rec["source"] = cells[(cat, seeds_present[0])].get("source", "")
        for meth in ("f0", "r1"):
            for met in ("auroc", "ap"):
                v = np.array([cells[(cat, s)][meth][met] for s in seeds_present], float)
                rec[f"{met}_{meth}_mean"] = float(v.mean()) if v.size else float("nan")
                rec[f"{met}_{meth}_std"] = float(v.std(ddof=1)) if v.size > 1 else float("nan")
        for met in ("auroc", "ap"):
            d = np.array([cells[(cat, s)]["r1"][met] - cells[(cat, s)]["f0"][met]
                          for s in seeds_present], float)
            rec[f"d_{met}_mean"] = float(d.mean()) if d.size else float("nan")
            rec[f"d_{met}_std"] = float(d.std(ddof=1)) if d.size > 1 else float("nan")
            rec[f"d_{met}_min"] = float(d.min()) if d.size else float("nan")
            rec[f"d_{met}_max"] = float(d.max()) if d.size else float("nan")
            rec[f"pos_seeds_{met}"] = int((d > 0).sum())
        rec["d_prime_mean_f0"] = float(np.mean([cells[(cat, s)]["f0"]["d_prime"]
                                                for s in seeds_present]))
        rec["d_prime_mean_r1"] = float(np.mean([cells[(cat, s)]["r1"]["d_prime"]
                                                for s in seeds_present]))
        rec["n_good"] = cells[(cat, seeds_present[0])]["n_ok"] if seeds_present else ""
        rec["n_anomaly"] = cells[(cat, seeds_present[0])]["n_ng"] if seeds_present else ""
        rows.append(rec)
    cols = ["category", "source", "n_seeds", "n_good", "n_anomaly",
            "auroc_f0_mean", "auroc_f0_std", "auroc_r1_mean", "auroc_r1_std",
            "d_auroc_mean", "d_auroc_std", "d_auroc_min", "d_auroc_max", "pos_seeds_auroc",
            "ap_f0_mean", "ap_f0_std", "ap_r1_mean", "ap_r1_std",
            "d_ap_mean", "d_ap_std", "d_ap_min", "d_ap_max", "pos_seeds_ap",
            "d_prime_mean_f0", "d_prime_mean_r1"]
    with open(OUT / "main_table_10cat_3seed.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({k: (fnum(r[k]) if isinstance(r[k], float) else r[k]) for k in cols})
    print(f"[E22 analysis] wrote main_table_10cat_3seed.csv ({len(rows)} categories)", flush=True)
    return rows


# ------------------------------------------------------------------ 3/4. summary + win rates
def category_summary(main_rows: list) -> list:
    rows = []
    for r in main_rows:
        pos = r["pos_seeds_auroc"]
        d = r["d_auroc_mean"]
        std = r["d_auroc_std"]
        if pos == 3 and std <= 0.01:
            stab = "stable_positive"
        elif pos == 0:
            stab = "stable_negative"
        elif std > 0.01:
            stab = "seed_sensitive"
        else:
            stab = "mixed"
        rows.append({"category": r["category"], "source": r["source"],
                     "d_auroc_mean": fnum(d), "d_auroc_std": fnum(std),
                     "d_auroc_min": fnum(r["d_auroc_min"]), "d_auroc_max": fnum(r["d_auroc_max"]),
                     "pos_seeds_auroc": pos, "d_ap_mean": fnum(r["d_ap_mean"]),
                     "d_ap_std": fnum(r["d_ap_std"]), "pos_seeds_ap": r["pos_seeds_ap"],
                     "auroc_pos": int(d > 0), "ap_pos": int(r["d_ap_mean"] > 0),
                     "both_pos": int(d > 0 and r["d_ap_mean"] > 0),
                     "stability_label": stab,
                     "mean_d_auroc_ge_0.005": int(d >= 0.005)})
    with open(OUT / "category_summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"[E22 analysis] wrote category_summary.csv", flush=True)
    return rows


def win_rates(summary_rows: list, main_rows: list) -> dict:
    n = len(summary_rows)
    pos_a = [r["category"] for r in summary_rows if r["auroc_pos"]]
    pos_p = [r["category"] for r in summary_rows if r["ap_pos"]]
    both = [r["category"] for r in summary_rows if r["both_pos"]]
    d_au = np.array([r["d_auroc_mean"] for r in main_rows], float)
    d_ap = np.array([r["d_ap_mean"] for r in main_rows], float)
    counts = {f"{k}/3": sum(1 for r in summary_rows if r["pos_seeds_auroc"] == k) for k in (3, 2, 1, 0)}
    wr = {"n_categories": n,
          "positive_mean_dauroc_categories": pos_a, "positive_mean_dap_categories": pos_p,
          "both_positive_categories": both,
          "n_positive_dauroc": len(pos_a), "n_positive_dap": len(pos_p), "n_both": len(both),
          "win_rate_dauroc": f"{len(pos_a)}/{n}", "win_rate_dap": f"{len(pos_p)}/{n}",
          "pos_seed_count_distribution": counts,
          "macro_mean_d_auroc": float(d_au.mean()), "median_d_auroc": float(np.median(d_au)),
          "macro_mean_d_ap": float(d_ap.mean()), "median_d_ap": float(np.median(d_ap)),
          "min_category_mean_d_auroc": float(d_au.min()),
          "max_category_mean_d_auroc": float(d_au.max()),
          "n_severe_negative": int((d_au < SEVERE_NEG).sum()),
          "severe_negative_categories": [r["category"] for r in main_rows
                                         if r["d_auroc_mean"] < SEVERE_NEG]}
    with open(OUT / "win_rates.json", "w") as f:
        json.dump(wr, f, indent=1)
    print(f"[E22 analysis] win rate ΔAUROC = {wr['win_rate_dauroc']}, ΔAP = {wr['win_rate_dap']} "
          f"| macro mean ΔAUROC {wr['macro_mean_d_auroc']:+.5f} "
          f"| median {wr['median_d_auroc']:+.5f}", flush=True)
    return wr


# ------------------------------------------------------------------ 5. bootstrap
def paired_bootstrap(a, b, y, n_boot=N_BOOT, rng_seed=BOOT_RNG) -> dict:
    """Paired image-level bootstrap of ΔAUROC / ΔAP (same resampled indices for F0 and R1)."""
    rng = np.random.default_rng(rng_seed)
    n = len(y)
    das, dps = [], []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        yy = y[idx]
        if len(np.unique(yy)) < 2:
            continue
        das.append(roc_auc_score(yy, b[idx]) - roc_auc_score(yy, a[idx]))
        dps.append(average_precision_score(yy, b[idx]) - average_precision_score(yy, a[idx]))
    das, dps = np.array(das, float), np.array(dps, float)
    if das.size == 0:
        return {"n_boot_effective": 0}
    return {"d_auroc_mean": float(das.mean()),
            "d_auroc_ci": [float(np.percentile(das, 2.5)), float(np.percentile(das, 97.5))],
            "p_d_auroc_gt0": float((das > 0).mean()),
            "d_ap_mean": float(dps.mean()),
            "d_ap_ci": [float(np.percentile(dps, 2.5)), float(np.percentile(dps, 97.5))],
            "p_d_ap_gt0": float((dps > 0).mean()),
            "n_boot_requested": n_boot, "n_boot_effective": int(das.size),
            "rng_seed": rng_seed}


def bootstrap_all(cells: dict) -> dict:
    out = {}
    for cat in CATS:
        seeds = [s for s in SEEDS if (cat, s) in cells]
        if not seeds:
            continue
        per_seed = {}
        for s in seeds:
            c = cells[(cat, s)]
            keep = np.array(c["detectable_raw"], bool)
            y = np.array(c["labels"], int)[keep]
            a = np.array(c["scores_f0"], float)[keep]
            b = np.array(c["scores_r1"], float)[keep]
            per_seed[str(s)] = paired_bootstrap(a, b, y)
        # seed-mean scoring: per-image mean over seeds, computed BEFORE metric evaluation
        if len(seeds) > 1:
            base = cells[(cat, seeds[0])]
            assert all(cells[(cat, s)]["labels"] == base["labels"] for s in seeds), \
                f"{cat}: per-seed label order mismatch"
            keep = np.array(base["detectable_raw"], bool)
            y = np.array(base["labels"], int)[keep]
            A = np.array([np.array(cells[(cat, s)]["scores_f0"], float) for s in seeds]).mean(axis=0)[keep]
            B = np.array([np.array(cells[(cat, s)]["scores_r1"], float) for s in seeds]).mean(axis=0)[keep]
            out.setdefault(cat, {})["seed_mean_scores"] = paired_bootstrap(A, B, y)
        out.setdefault(cat, {})["per_seed"] = per_seed
    with open(OUT / "bootstrap_e22.json", "w") as f:
        json.dump(out, f, indent=1)
    print(f"[E22 analysis] wrote bootstrap_e22.json (n_boot={N_BOOT}, rng={BOOT_RNG})", flush=True)
    return out


# ------------------------------------------------------------------ 6. illumination
def illumination_breakdown(cells: dict) -> tuple[list, list]:
    rows, spreads = [], []
    for (cat, seed), c in sorted(cells.items()):
        ill = c["illumination"]
        keep = np.array(c["detectable_raw"], bool)
        y_all = np.array(c["labels"], int)
        for meth, key in (("F0", "scores_f0"), ("R1", "scores_r1")):
            s_all = np.array(c[key], float)
            tag = f"{meth}|{cat}|seed{seed}"
            for g in ["ALL"] + sorted(set(ill)):
                idx = np.array([True] * len(ill), bool) if g == "ALL" else \
                    np.array([x == g for x in ill], bool)
                idx = idx & keep
                y, s = y_all[idx], s_all[idx]
                if len(y) == 0:
                    continue
                ok, ng = int((y == 0).sum()), int((y == 1).sum())
                a = float(roc_auc_score(y, s)) if len(np.unique(y)) == 2 else float("nan")
                p = float(average_precision_score(y, s)) if len(np.unique(y)) == 2 else float("nan")
                good = s[y == 0]
                rows.append({"source": tag, "category": cat, "seed": seed, "method": meth,
                             "illumination": g, "n_ok": ok, "n_ng": ng,
                             "auroc": a, "ap": p,
                             "good_mean": float(good.mean()) if len(good) else float("nan"),
                             "good_std": float(good.std(ddof=1)) if len(good) > 1 else float("nan")})
            gs = [r["good_mean"] for r in rows
                  if r["source"] == tag and r["illumination"] != "ALL" and math.isfinite(r["good_mean"])]
            spreads.append({"category": cat, "seed": seed, "method": meth,
                            "good_mean_min": float(min(gs)), "good_mean_max": float(max(gs)),
                            "spread": float(max(gs) - min(gs))})
    with open(OUT / "illumination_breakdown.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    with open(OUT / "illumination_spread.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(spreads[0]))
        w.writeheader()
        w.writerows(spreads)
    print(f"[E22 analysis] wrote illumination_breakdown.csv ({len(rows)} rows) + "
          f"illumination_spread.csv", flush=True)
    return rows, spreads


def illumination_summary(spreads: list) -> list:
    out = []
    for cat in CATS:
        r = {"category": cat}
        for meth in ("F0", "R1"):
            v = np.array([s["spread"] for s in spreads
                          if s["category"] == cat and s["method"] == meth], float)
            r[f"spread_{meth}_mean"] = float(v.mean()) if v.size else float("nan")
            r[f"spread_{meth}_std"] = float(v.std(ddof=1)) if v.size > 1 else float("nan")
        r["spread_ratio_R1_over_F0"] = (r["spread_R1_mean"] / r["spread_F0_mean"]
                                        if r["spread_F0_mean"] else float("nan"))
        r["spread_reduced"] = int(math.isfinite(r["spread_ratio_R1_over_F0"])
                                  and r["spread_ratio_R1_over_F0"] < 1.0)
        out.append(r)
    with open(OUT / "illumination_summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0]))
        w.writeheader()
        for r in out:
            w.writerow({k: (fnum(v) if isinstance(v, float) else v) for k, v in r.items()})
    return out


# ------------------------------------------------------------------ 7. shift vs gain
def shift_gain(cells: dict, main_rows: list, cells_csv_rows: list = None) -> dict:
    cats = [r["category"] for r in main_rows]        # only categories with cells present
    shift = {}
    for cat in cats:
        seeds = [s for s in SEEDS if (cat, s) in cells]
        dvals = [cells[(cat, s)]["diagnostics"].get("D_illum_shift_F0", float("nan"))
                 for s in seeds]
        bvals = [cells[(cat, s)]["diagnostics"].get("B_illum_group_centroid_dist_F0", float("nan"))
                 for s in seeds]
        shift[cat] = {"D_shift_F0_seed_mean": float(np.nanmean(dvals)),
                      "D_shift_F0_seed0": float(cells[(cat, seeds[0])]["diagnostics"]
                                                .get("D_illum_shift_F0", float("nan"))),
                      "B_centroid_F0_seed_mean": float(np.nanmean(bvals)),
                      "B_centroid_F0_seed0": float(cells[(cat, seeds[0])]["diagnostics"]
                                                   .get("B_illum_group_centroid_dist_F0",
                                                        float("nan")))}
    gain = {r["category"]: r["d_auroc_mean"] for r in main_rows}
    gain_ap = {r["category"]: r["d_ap_mean"] for r in main_rows}
    res = {"n_categories": len(cats), "n_categories_expected": len(CATS), "shift": shift,
           "association": {}}
    x_all = np.array([shift[c]["D_shift_F0_seed_mean"] for c in cats], float)
    x_s0 = np.array([shift[c]["D_shift_F0_seed0"] for c in cats], float)
    xb_all = np.array([shift[c]["B_centroid_F0_seed_mean"] for c in cats], float)
    for name, x, y in (("D_shift_seed_mean_vs_dAUROC", x_all, np.array([gain[c] for c in cats])),
                       ("D_shift_seed0_vs_dAUROC", x_s0, np.array([gain[c] for c in cats])),
                       ("B_centroid_seed_mean_vs_dAUROC", xb_all, np.array([gain[c] for c in cats])),
                       ("D_shift_seed_mean_vs_dAP", x_all, np.array([gain_ap[c] for c in cats]))):
        if not (np.all(np.isfinite(x)) and np.all(np.isfinite(y))):
            res["association"][name] = {"error": "non-finite values"}
            continue
        pr, pp = sps.pearsonr(x, y)
        sr, sp = sps.spearmanr(x, y)
        res["association"][name] = {"pearson_r": float(pr), "pearson_p": float(pp),
                                    "spearman_rho": float(sr), "spearman_p": float(sp),
                                    "n": int(len(x))}
    with open(OUT / "shift_gain_association.json", "w") as f:
        json.dump(res, f, indent=1)
    print("[E22 analysis] wrote shift_gain_association.json "
          f"(n={len(CATS)} categories, exploratory only)", flush=True)
    return res


# ------------------------------------------------------------------ 8. failure cases
def failure_cases(cells: dict, main_rows: list, summary_rows: list, bootstrap: dict) -> None:
    L = ["# E22 — Failure-case analysis (all negative results retained)\n",
         f"Generated from {len(cells)}/30 cells. Every negative cell listed, none removed.\n"]
    d = {r["category"]: r for r in main_rows}
    s = {r["category"]: r for r in summary_rows}
    worst_cat = min(main_rows, key=lambda r: r["d_auroc_mean"])
    best_cat = max(main_rows, key=lambda r: r["d_auroc_mean"])
    worst_cell = min(cells.items(), key=lambda kv: kv[1]["delta_auroc"])
    worst_ap_cell = min(cells.items(), key=lambda kv: kv[1]["delta_ap"])
    var_cat = max(main_rows, key=lambda r: (r["d_auroc_std"] if math.isfinite(r["d_auroc_std"]) else -1))
    L += ["## 1. Worst category (3-seed mean ΔAUROC)\n",
          f"- **{worst_cat['category']}**: ΔAUROC {worst_cat['d_auroc_mean']:+.5f} "
          f"(std {worst_cat['d_auroc_std']:.5f}), ΔAP {worst_cat['d_ap_mean']:+.5f}, "
          f"positive seeds {worst_cat['pos_seeds_auroc']}/3.",
          f"- F0 AUROC {worst_cat['auroc_f0_mean']:.5f} → R1 AUROC {worst_cat['auroc_r1_mean']:.5f}\n",
          "## 2. Best category (for contrast)\n",
          f"- **{best_cat['category']}**: ΔAUROC {best_cat['d_auroc_mean']:+.5f}, "
          f"ΔAP {best_cat['d_ap_mean']:+.5f}, positive seeds {best_cat['pos_seeds_auroc']}/3.\n",
          "## 3. Worst single cell (ΔAUROC / ΔAP)\n"]
    for label, item in (("ΔAUROC", worst_cell), ("ΔAP", worst_ap_cell)):
        (cat, seed), c = item
        L.append(f"- **{cat}|seed{seed}** {label}: ΔAUROC {c['delta_auroc']:+.5f}, "
                 f"ΔAP {c['delta_ap']:+.5f} (F0 {c['f0']['auroc']:.5f}/{c['f0']['ap']:.5f} → "
                 f"R1 {c['r1']['auroc']:.5f}/{c['r1']['ap']:.5f}), bank={c['bank_hash'][:12]}, "
                 f"steps={c.get('steps_used')}")
    L += ["", "## 4. Largest cross-seed variability (std of ΔAUROC)\n",
          f"- **{var_cat['category']}**: std {var_cat['d_auroc_std']:.5f} "
          f"(min {var_cat['d_auroc_min']:+.5f}, max {var_cat['d_auroc_max']:+.5f}, "
          f"positive seeds {var_cat['pos_seeds_auroc']}/3)\n",
          "## 5. All negative cells (ΔAUROC < 0)\n",
          "| category | seed | ΔAUROC | ΔAP | F0 AUROC | R1 AUROC | bank |",
          "|---|---|---|---|---|---|---|"]
    neg = [(k, v) for k, v in sorted(cells.items()) if v["delta_auroc"] < 0]
    for (cat, seed), c in neg:
        L.append(f"| {cat} | {seed} | {c['delta_auroc']:+.5f} | {c['delta_ap']:+.5f} | "
                 f"{c['f0']['auroc']:.5f} | {c['r1']['auroc']:.5f} | {c['bank_hash'][:10]} |")
    if not neg:
        L.append("| (none) | | | | | | |")
    L += ["", "## 6. Categories with a negative 3-seed mean\n",
          "| category | ΔAUROC mean | ΔAP mean | positive seeds | CI (seed-mean, ΔAUROC) |",
          "|---|---|---|---|---|"]
    for r in main_rows:
        if r["d_auroc_mean"] < 0:
            ci = bootstrap.get(r["category"], {}).get("seed_mean_scores", {}).get("d_auroc_ci",
                                                                                  ["", ""])
            L.append(f"| {r['category']} | {r['d_auroc_mean']:+.5f} | {r['d_ap_mean']:+.5f} | "
                     f"{r['pos_seeds_auroc']}/3 | [{ci[0]:+.4f}, {ci[1]:+.4f}] |")
    L += ["", "## 7. Seed-stability description (descriptive labels only; no category removed)\n",
          "| category | positive seeds (AUROC) | std ΔAUROC | label |", "|---|---|---|---|"]
    for r in summary_rows:
        L.append(f"| {r['category']} | {r['pos_seeds_auroc']}/3 | {r['d_auroc_std']} | "
                 f"{r['stability_label']} |")
    L += ["", "**Rule for the descriptive labels** (defined in README): `stable_positive` = 3/3 "
          "positive seeds and std ≤ 0.01; `stable_negative` = 0/3 positive; `seed_sensitive` = "
          "std > 0.01; `mixed` otherwise. Labels are descriptive; they never remove a category."
          "\n"]
    (OUT / "failure_cases.md").write_text("\n".join(L))
    print("[E22 analysis] wrote failure_cases.md", flush=True)


# ------------------------------------------------------------------ 9. verdict + stats md
def verdict(win: dict) -> dict:
    if win["n_categories"] < len(CATS):
        out = {"verdict": "INCOMPLETE_MATRIX",
               "n_categories": win["n_categories"], "n_expected": len(CATS),
               "note": "pre-registered verdict requires the full 10 categories x 3 seeds; "
                       "partial win rates are reported but NOT interpreted as a verdict",
               "partial_win_rates": {k: win[k] for k in ("win_rate_dauroc", "win_rate_dap",
                                                         "n_positive_dauroc", "n_positive_dap",
                                                         "macro_mean_d_auroc", "median_d_auroc",
                                                         "macro_mean_d_ap", "median_d_ap")}}
        with open(OUT / "verdict.json", "w") as f:
            json.dump(out, f, indent=1)
        print(f"[E22 analysis] verdict = INCOMPLETE_MATRIX ({win['n_categories']}/10 categories)",
              flush=True)
        return out
    cond = {"c1_pos_categories_ge_7": win["n_positive_dauroc"] >= STRONG_GO_POS_CATS,
            "c2_macro_mean_gt_0": win["macro_mean_d_auroc"] > 0,
            "c3_median_gt_0": win["median_d_auroc"] > 0,
            "c4_no_category_below_-0.03": win["n_severe_negative"] == 0}
    strong = all(cond.values())
    stop = (win["n_positive_dauroc"] <= 4
            or (win["macro_mean_d_auroc"] <= 0 and win["median_d_auroc"] <= 0)
            or win["n_severe_negative"] >= 2)
    v = "STRONG GO" if strong else ("STOP BROAD-GENERALIZATION CLAIM" if stop else "HOLD")
    out = {"verdict": v, "conditions": cond, "thresholds": {
        "strong_go_pos_categories": STRONG_GO_POS_CATS, "severe_negative": SEVERE_NEG},
        "win_rates": {k: win[k] for k in ("win_rate_dauroc", "win_rate_dap", "n_positive_dauroc",
                                          "n_positive_dap", "n_both", "macro_mean_d_auroc",
                                          "median_d_auroc", "macro_mean_d_ap", "median_d_ap",
                                          "n_severe_negative")}}
    with open(OUT / "verdict.json", "w") as f:
        json.dump(out, f, indent=1)
    return out


def statistical_summary(main_rows, summary_rows, win, boot, shift, illum_sum, verdict_obj) -> None:
    d = {r["category"]: r for r in main_rows}
    L = ["# E22 — Statistical summary (M²AD 10 categories × 3 seeds × view120)\n",
         f"Verdict: **{verdict_obj['verdict']}**\n",
         "## 1–2. Categories with positive mean ΔAUROC / ΔAP\n",
         f"- ΔAUROC > 0: **{win['n_positive_dauroc']}/10** — {', '.join(win['positive_mean_dauroc_categories']) or '(none)'}",
         f"- ΔAP > 0: **{win['n_positive_dap']}/10** — {', '.join(win['positive_mean_dap_categories']) or '(none)'}",
         f"- both > 0: **{win['n_both']}/10** — {', '.join(win['both_positive_categories']) or '(none)'}\n",
         "## 3–4. Macro mean / median\n",
         f"- macro mean ΔAUROC = **{win['macro_mean_d_auroc']:+.5f}**; median = "
         f"**{win['median_d_auroc']:+.5f}**",
         f"- macro mean ΔAP = **{win['macro_mean_d_ap']:+.5f}**; median = "
         f"**{win['median_d_ap']:+.5f}**",
         f"- worst category mean ΔAUROC = {win['min_category_mean_d_auroc']:+.5f}; "
         f"best = {win['max_category_mean_d_auroc']:+.5f}\n",
         "## 5. Seed stability\n",
         "| category | ΔAUROC mean ± std | positive seeds | label |", "|---|---|---|---|"]
    for r in summary_rows:
        L.append(f"| {r['category']} | {r['d_auroc_mean']} ± {r['d_auroc_std']} | "
                 f"{r['pos_seeds_auroc']}/3 | {r['stability_label']} |")
    L += ["", "Seed-count distribution: " + ", ".join(f"{k}: {v}"
          for k, v in sorted(win["pos_seed_count_distribution"].items())), "",
          "## 6. Uncertainty (two separate forms — never merged)\n",
          "**(a) Image-level paired bootstrap** (test-set sampling uncertainty; n_boot=3000, "
          f"rng={BOOT_RNG}; paired indices; detectable-filtered):",
          "| category | ΔAUROC CI (seed-mean scores) | p(ΔAUROC>0) | ΔAP CI (seed-mean scores) | p(ΔAP>0) |",
          "|---|---|---|---|---|"]
    for cat in CATS:
        sm = boot.get(cat, {}).get("seed_mean_scores", {})
        if not sm:
            continue
        L.append(f"| {cat} | [{sm['d_auroc_ci'][0]:+.4f}, {sm['d_auroc_ci'][1]:+.4f}] | "
                 f"{sm['p_d_auroc_gt0']:.3f} | [{sm['d_ap_ci'][0]:+.4f}, {sm['d_ap_ci'][1]:+.4f}] | "
                 f"{sm['p_d_ap_gt0']:.3f} |")
    L += ["", "**(b) Cross-seed variability** = std of ΔAUROC over the 3 independent banks "
          "(reported per category in §5 and in `main_table_10cat_3seed.csv`). This is a different "
          "quantity from (a) and is never combined with it.\n",
          "## 7–8. Largest gain / largest regression\n",
          f"- largest category gain: **{max(d, key=lambda c: d[c]['d_auroc_mean'])}** "
          f"({max(r['d_auroc_mean'] for r in main_rows):+.5f} ΔAUROC)",
          f"- largest category regression: **{min(d, key=lambda c: d[c]['d_auroc_mean'])}** "
          f"({min(r['d_auroc_mean'] for r in main_rows):+.5f} ΔAUROC)\n",
          "## 9. Illumination shift ↔ gain association (exploratory, n = 10 categories)\n",
          "| association | Pearson r (p) | Spearman ρ (p) |", "|---|---|---|"]
    for k, v in shift["association"].items():
        if "error" in v:
            L.append(f"| {k} | {v['error']} | {v['error']} |")
        else:
            L.append(f"| {k} | {v['pearson_r']:+.3f} ({v['pearson_p']:.3f}) | "
                     f"{v['spearman_rho']:+.3f} ({v['spearman_p']:.3f}) |")
    L += ["", "**Interpretation boundary**: n = 10 categories → exploratory / supportive "
          "association only. **No causal claim may be written from this table.**\n",
          "## Normal-image score spread across illumination (descriptive)\n",
          "| category | F0 spread (3-seed mean) | R1 spread | ratio R1/F0 |", "|---|---|---|---|"]
    for r in illum_sum:
        L.append(f"| {r['category']} | {r['spread_F0_mean']:.4f} | {r['spread_R1_mean']:.4f} | "
                 f"{r['spread_ratio_R1_over_F0']:.3f} |")
    L += ["", "Only the SPREAD across groups is interpretable (R1-ILA changes the feature scale, so "
          "absolute score levels are not comparable between F0 and R1).\n",
          "## 10. Broad-generalization claim verdict\n",
          f"**{verdict_obj['verdict']}**  ",
          "| pre-registered condition | value | satisfied |", "|---|---|---|",
          f"| ≥ 7/10 categories with 3-seed mean ΔAUROC > 0 | {win['n_positive_dauroc']}/10 | "
          f"{verdict_obj.get('conditions', {}).get('c1_pos_categories_ge_7', 'n/a')} |",
          f"| macro mean ΔAUROC > 0 | {win['macro_mean_d_auroc']:+.5f} | "
          f"{verdict_obj.get('conditions', {}).get('c2_macro_mean_gt_0', 'n/a')} |",
          f"| median ΔAUROC > 0 | {win['median_d_auroc']:+.5f} | "
          f"{verdict_obj.get('conditions', {}).get('c3_median_gt_0', 'n/a')} |",
          f"| no category mean ΔAUROC < −0.03 | {win['n_severe_negative']} severe | "
          f"{verdict_obj.get('conditions', {}).get('c4_no_category_below_-0.03', 'n/a')} |", "",
          "## Claim boundary\n",
          "- **May be claimed**: the unselected result of frozen R1-ILA across 10 M²AD categories × "
          "3 seeds × view120 (per-category means, seed variability, win rates, CIs, all negatives).",
          "- **May not be claimed**: causal attribution (illumination shift ↔ gain is an "
          "association at n=10); generalization beyond M²AD / this backbone / view120; image "
          "bootstrap CIs as if they were cross-seed uncertainty; a STOP verdict as 'paper failed'."]
    (OUT / "statistical_summary.md").write_text("\n".join(L))
    print(f"[E22 analysis] wrote statistical_summary.md | VERDICT = {verdict_obj['verdict']}",
          flush=True)


# ------------------------------------------------------------------ main
def main() -> int:
    cells, missing = load_cells()
    print(f"[E22 analysis] cells found: {len(cells)}/30 | missing: {missing}", flush=True)
    if len(cells) < 30:
        (OUT / "analysis_incomplete.txt").write_text(
            f"analysis run with {len(cells)}/30 cells; missing {missing}\n")
    write_cells_csv(cells)
    main_rows = main_table(cells)
    summary_rows = category_summary(main_rows)
    win = win_rates(summary_rows, main_rows)
    boot = bootstrap_all(cells)
    _, spreads = illumination_breakdown(cells)
    illum_sum = illumination_summary(spreads)
    shift = shift_gain(cells, main_rows, [])
    failure_cases(cells, main_rows, summary_rows, boot)
    v = verdict(win)
    statistical_summary(main_rows, summary_rows, win, boot, shift, illum_sum, v)
    print("\n[E22 analysis] ===== main table =====")
    for r in main_rows:
        print(f"  {r['category']:7s} F0 {r['auroc_f0_mean']:.5f}±{r['auroc_f0_std']:.5f} "
              f"R1 {r['auroc_r1_mean']:.5f}±{r['auroc_r1_std']:.5f} "
              f"ΔAUROC {r['d_auroc_mean']:+.5f}±{r['d_auroc_std']:.5f} "
              f"ΔAP {r['d_ap_mean']:+.5f}±{r['d_ap_std']:.5f} "
              f"({r['pos_seeds_auroc']}/3 pos)", flush=True)
    print(f"[E22 analysis] VERDICT: {v['verdict']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
