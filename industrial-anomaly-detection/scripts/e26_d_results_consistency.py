#!/usr/bin/env python
"""E26-D: automated consistency audit of the already-published final numbers.

No training, no inference, no GPU. Every metric is RE-COMPUTED from the stored
per-image scores in the committed cell JSONs and then compared against every
place the number is published.

Sources covered (each file has exactly ONE owner function, so no published
number is checked twice and none is left unattributed):
    results/e22/cells/*.json            30 cells  -> audit_m2ad()
    results/e22/main_table_10cat_3seed.csv
    results/paper_ready/paper_main_table.csv
    results/e25/cells/*.json            39 cells  -> audit_robustad()
    results/e25/multiseed_analysis.json
    results/e25/multiseed_summary.csv    (3-seed statistics)
    results/e25/e25_summary.csv          (SEED-0 run table, see note in code)
    results/e21/e21_partial.json        12 views  -> audit_e21()
    results/paper_ready/paper_multiview_table.csv
    results/e26_anomalydino/scores/*.json  30 cells -> audit_anomalydino()
    results/e26_anomalydino/baseline_30cells.csv
    results/e26_anomalydino/baseline_category_summary.csv
    results/e26_anomalydino/baseline_macro.csv

Not covered, on purpose: bootstrap CIs (would require re-running the original
resampling RNG), E24 derived columns (d', robustness_feat/score, illum_shift),
and figures. See results/e26/results_consistency/REPORT.md.

Tolerance policy (every check is labelled with its class):
  * full_precision  -- source stores ~16 significant digits (cell JSON). The
    assertion is abs(diff) <= 1e-6, i.e. far below any scientific relevance.
  * published_{n}dp -- source is a printed table. The assertion is
    abs(diff) <= 10^-n, one unit in the last published place (1 ulp). A check
    using the whole ulp is further flagged, and the flag says WHY: the printed
    digit is reproduced by truncation, or by double rounding (round to n+1 dp,
    then print n dp). Both are mechanical properties of the publishing script,
    not disagreements. They are reported separately, never silently absorbed.

Usage
-----
    python scripts/e26_d_results_consistency.py
"""
from __future__ import annotations

import csv
import json
import statistics as st
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "e26" / "results_consistency"
TOL = 1e-6

CATS = ["Bird", "Car", "Cube", "Dice", "Doll", "Holder", "Motor", "Ring", "Teapot", "Tube"]
DEV = {"Bird", "Car", "Cube"}
SEEDS = (0, 1, 2)
CHECKS: list[dict] = []
FIELDS = ["id", "scope", "metric", "check_type", "source_a", "source_b",
          "expected", "actual", "diff", "precision_dp", "tol",
          "rounds_to_published", "truncates_to_published", "double_rounds_to_published", "pass"]


def add(check_id: str, scope: str, metric: str, source_a: str, source_b: str,
        expected: float, actual: float, tol: float = TOL,
        check_type: str = "full_precision", precision_dp: int | None = None,
        rounds_to_published: bool | None = None,
        truncates_to_published: bool | None = None,
        double_rounds_to_published: bool | None = None) -> dict:
    diff = abs(float(expected) - float(actual))
    rec = {"id": check_id, "scope": scope, "metric": metric, "check_type": check_type,
           "source_a": source_a, "source_b": source_b,
           "expected": float(expected), "actual": float(actual),
           "diff": diff, "precision_dp": precision_dp, "tol": tol,
           "rounds_to_published": rounds_to_published,
           "truncates_to_published": truncates_to_published,
           "double_rounds_to_published": double_rounds_to_published,
           "pass": bool(diff <= tol)}
    CHECKS.append(rec)
    return rec


def decimals(raw: str) -> int:
    raw = raw.strip()
    return len(raw.split(".")[1]) if "." in raw else 0


def add_published(check_id: str, scope: str, metric: str, raw: str, recomputed: float,
                  source: str) -> dict:
    """Compare a recomputed value against a value printed with n decimals."""
    dp = decimals(raw)
    tol = 10.0 ** (-dp)
    published = float(raw)
    trunc = int(recomputed * 10 ** dp) / 10 ** dp  # int() truncates toward zero
    dbl = float(f"{round(recomputed, dp + 1):.{dp}f}")  # one extra digit, then printed
    rec = add(check_id, scope, metric, "recomputed from stored per-image scores",
              f"{source} (printed with {dp} dp, tolerance = 1 ulp = {tol:g})",
              published, recomputed, tol=tol + 1e-12, check_type=f"published_{dp}dp",
              precision_dp=dp, rounds_to_published=round(recomputed, dp) == published,
              truncates_to_published=trunc == published,
              double_rounds_to_published=dbl == published)
    return rec


def metrics(scores, labels, detectable):
    keep = [i for i, d in enumerate(detectable) if d is not False]
    y = np.array([labels[i] for i in keep], dtype=int)
    s = np.array([scores[i] for i in keep], dtype=float)
    return float(roc_auc_score(y, s)), float(average_precision_score(y, s))


# --------------------------------------------------------------------------- M2AD
def audit_m2ad() -> dict:
    cell_dir = ROOT / "results" / "e22" / "cells"
    per_cell = {}
    for cat in CATS:
        for seed in SEEDS:
            rec = json.loads((cell_dir / f"{cat}_seed{seed}.json").read_text())
            a0, p0 = metrics(rec["scores_f0"], rec["labels"], rec["detectable_raw"])
            a1, p1 = metrics(rec["scores_r1"], rec["labels"], rec["detectable_raw"])
            per_cell[(cat, seed)] = {"auroc_f0": a0, "ap_f0": p0, "auroc_r1": a1, "ap_r1": p1,
                                     "d_auroc": a1 - a0, "d_ap": p1 - p0}
            # 1) recomputation vs the stored per-cell values
            add(f"E22-cell-{cat}-s{seed}-auroc_f0", "M2AD", "auroc_f0",
                "recomputed from scores_f0", f"results/e22/cells/{cat}_seed{seed}.json:f0.auroc",
                rec["f0"]["auroc"], a0)
            add(f"E22-cell-{cat}-s{seed}-ap_f0", "M2AD", "ap_f0",
                "recomputed from scores_f0", f"results/e22/cells/{cat}_seed{seed}.json:f0.ap",
                rec["f0"]["ap"], p0)
            add(f"E22-cell-{cat}-s{seed}-auroc_r1", "M2AD", "auroc_r1",
                "recomputed from scores_r1", f"results/e22/cells/{cat}_seed{seed}.json:r1.auroc",
                rec["r1"]["auroc"], a1)
            add(f"E22-cell-{cat}-s{seed}-ap_r1", "M2AD", "ap_r1",
                "recomputed from scores_r1", f"results/e22/cells/{cat}_seed{seed}.json:r1.ap",
                rec["r1"]["ap"], p1)

    # 2) category aggregates vs E22 main table
    main = {r["category"]: r for r in csv.DictReader(open(ROOT / "results" / "e22" / "main_table_10cat_3seed.csv"))}
    cat_stats = {}
    for cat in CATS:
        vals = [per_cell[(cat, s)] for s in SEEDS]
        agg = {k: st.mean(v[k] for v in vals) for k in ("auroc_f0", "ap_f0", "auroc_r1", "ap_r1", "d_auroc", "d_ap")}
        for k in ("auroc_f0", "ap_f0", "auroc_r1", "ap_r1", "d_auroc", "d_ap"):
            agg[f"{k}_std"] = st.stdev(v[k] for v in vals)
        cat_stats[cat] = agg
        row = main[cat]
        for ours, col in (("auroc_f0", "auroc_f0_mean"), ("auroc_r1", "auroc_r1_mean"),
                          ("d_auroc", "d_auroc_mean"), ("ap_f0", "ap_f0_mean"),
                          ("ap_r1", "ap_r1_mean"), ("d_ap", "d_ap_mean"),
                          ("auroc_f0_std", "auroc_f0_std"), ("d_auroc_std", "d_auroc_std")):
            add_published(f"E22-main-{cat}-{col}", "M2AD", ours, row[col], agg[ours],
                          f"results/e22/main_table_10cat_3seed.csv:{col}")

    # 3) macro / dev / frozen aggregation (baseline_macro.csv itself is verified in
    #    audit_anomalydino(), which owns every AnomalyDINO comparison table)
    macro = {k: st.mean(cat_stats[c][k] for c in CATS)
             for k in ("auroc_f0", "ap_f0", "auroc_r1", "ap_r1", "d_auroc", "d_ap")}
    dev = {k: st.mean(cat_stats[c][k] for c in CATS if c in DEV) for k in macro}
    ext = {k: st.mean(cat_stats[c][k] for c in CATS if c not in DEV) for k in macro}

    # 4) paper_ready main table: 5 dp printed values -> 1 ulp tolerance, flagged
    pr = {r["Category"]: r for r in csv.DictReader(open(ROOT / "results" / "paper_ready" / "paper_main_table.csv"))}
    for cat in CATS:
        row = pr[cat]
        for ours, col in (("auroc_f0", "F0_AUROC_mean"), ("auroc_r1", "R1_AUROC_mean"),
                          ("d_auroc", "dAUROC_mean"), ("ap_f0", "F0_AP_mean"),
                          ("ap_r1", "R1_AP_mean"), ("d_ap", "dAP_mean"),
                          ("auroc_f0_std", "F0_AUROC_std"), ("d_auroc_std", "dAUROC_std")):
            add_published(f"paper_ready-{cat}-{col}", "M2AD", ours, row[col], cat_stats[cat][ours],
                          f"results/paper_ready/paper_main_table.csv:{col}")

    pos_auroc = sum(1 for c in CATS if cat_stats[c]["d_auroc"] > 0)
    pos_ap = sum(1 for c in CATS if cat_stats[c]["d_ap"] > 0)
    neg = [c for c in CATS if cat_stats[c]["d_auroc"] < 0]
    return {"per_cell": {f"{c}_s{s}": per_cell[(c, s)] for c in CATS for s in SEEDS},
            "category": cat_stats, "macro_10cat": macro, "dev_3cat": dev, "frozen_7cat": ext,
            "positive_d_auroc_classes": pos_auroc, "positive_d_ap_classes": pos_ap,
            "negative_classes": neg, "doll": cat_stats["Doll"]}


# ----------------------------------------------------------------------- RobustAD
def audit_robustad() -> dict:
    cell_dir = ROOT / "results" / "e25" / "cells"
    recs = []
    for f in sorted(cell_dir.glob("*.json")):
        rec = json.loads(f.read_text())
        a0, p0 = metrics(rec["scores_f0"], rec["labels"], [True] * len(rec["labels"]))
        a1, p1 = metrics(rec["scores_r1"], rec["labels"], [True] * len(rec["labels"]))
        # cells carry no is_source field; the published convention is prefix-based:
        # condition_group "source_reference" and "source_reference_INVALID" are both
        # source-reference splits (the INVALID one is additionally valid=0).
        cg = str(rec.get("condition_group", ""))
        # seed-0 cells are named "<Domain>_testN.json", later seeds "<Domain>_seedK_testN.json"
        seed = int(f.stem.split("_seed")[1].split("_")[0]) if "_seed" in f.stem else 0
        recs.append({"file": f.name, "seed": seed, "domain": rec["domain"], "split": rec["split"],
                     "valid": rec.get("valid", 1),
                     "is_source": int(cg.startswith("source_reference")),
                     "condition_group": cg,
                     "auroc_f0": a0, "ap_f0": p0, "auroc_r1": a1, "ap_r1": p1,
                     "d_auroc": a1 - a0, "d_ap": p1 - p0,
                     "stored": rec})
        add(f"E25-cell-{f.stem}-auroc_f0", "RobustAD", "auroc_f0", "recomputed from scores_f0",
            f"results/e25/cells/{f.name}:f0.auroc", rec["f0"]["auroc"], a0)
        add(f"E25-cell-{f.stem}-auroc_r1", "RobustAD", "auroc_r1", "recomputed from scores_r1",
            f"results/e25/cells/{f.name}:r1.auroc", rec["r1"]["auroc"], a1)
        add(f"E25-cell-{f.stem}-d_auroc", "RobustAD", "d_auroc", "recomputed",
            f"results/e25/cells/{f.name}:d_auroc", rec["d_auroc"], a1 - a0)

    per_split = {}
    for r in recs:
        per_split.setdefault((r["domain"], r["split"]), []).append(r)
    split_rows = []
    for (dom, split), group in sorted(per_split.items()):
        row = {"domain": dom, "split": split, "seed": None,
               "valid": group[0]["valid"], "is_source": group[0]["is_source"],
               "condition_group": group[0]["condition_group"], "n_seeds": len(group)}
        for k in ("auroc_f0", "auroc_r1", "ap_f0", "ap_r1", "d_auroc", "d_ap"):
            row[k] = st.mean(g[k] for g in group)
            row[f"{k}_std"] = st.stdev(g[k] for g in group)
        split_rows.append(row)

    split_seed0 = {(r["domain"], r["split"]): r for r in recs if r["seed"] == 0}
    assert len(split_seed0) == len(split_rows), (len(split_seed0), len(split_rows))

    ms = json.loads((ROOT / "results" / "e25" / "multiseed_analysis.json").read_text())
    ms_split = {(r["domain"], r["split"]): r for r in ms["per_split"]}
    for row in split_rows:
        ref = ms_split[(row["domain"], row["split"])]
        for k in ("auroc_f0", "auroc_r1", "d_auroc", "d_ap"):
            add(f"E25-ms-{row['domain']}-{row['split']}-{k}", "RobustAD", k, "recomputed 3-seed mean",
                f"results/e25/multiseed_analysis.json:per_split[{row['domain']}:{row['split']}].{k}_mean",
                float(ref[f"{k}_mean"]), row[k])

    targets = [r for r in split_rows if r["valid"] == 1 and r["is_source"] == 0]
    dom_means = {}
    for dom in ("PCB", "MetalParts"):
        vals = [r["d_auroc"] for r in targets if r["domain"] == dom]
        dom_means[dom] = {"d_auroc": st.mean(vals),
                          "d_ap": st.mean([r["d_ap"] for r in targets if r["domain"] == dom]),
                          "n_splits": len(vals)}
        ref = ms["per_domain"][dom]
        add(f"E25-domain-{dom}-d_auroc", "RobustAD", "d_auroc", "recomputed domain mean",
            f"results/e25/multiseed_analysis.json:per_domain.{dom}.d_auroc_mean",
            float(ref["d_auroc_mean"]), dom_means[dom]["d_auroc"])
        add(f"E25-domain-{dom}-d_ap", "RobustAD", "d_ap", "recomputed domain mean",
            f"results/e25/multiseed_analysis.json:per_domain.{dom}.d_ap_mean",
            float(ref["d_ap_mean"]), dom_means[dom]["d_ap"])

    overall = {"macro_d_auroc": st.mean(r["d_auroc"] for r in targets),
               "macro_d_ap": st.mean(r["d_ap"] for r in targets),
               "n_target_splits": len(targets),
               "subdataset_equal_auroc": (dom_means["PCB"]["d_auroc"] + dom_means["MetalParts"]["d_auroc"]) / 2}
    add("E25-overall-macro_d_auroc", "RobustAD", "macro_d_auroc", "recomputed 11-split mean",
        "results/e25/multiseed_analysis.json:overall.macro_d_auroc_mean_over_splits",
        float(ms["overall"]["macro_d_auroc_mean_over_splits"]), overall["macro_d_auroc"])
    add("E25-overall-macro_d_ap", "RobustAD", "macro_d_ap", "recomputed 11-split mean",
        "results/e25/multiseed_analysis.json:overall.macro_d_ap_mean_over_splits",
        float(ms["overall"]["macro_d_ap_mean_over_splits"]), overall["macro_d_ap"])

    # printed published split tables (6 dp): 1 ulp tolerance, flags re-checked.
    # The two tables do NOT share semantics -- established empirically (all 13 splits,
    # see the README table), not assumed:
    #   multiseed_summary.csv : *_mean / *_std columns are 3-seed statistics
    #   e25_summary.csv       : a SEED-0 run table. auroc_r1/ap_r1/d_auroc/d_ap match
    #                           seed 0 in 13/13 splits and the 3-seed mean in 0/13;
    #                           its bootstrap CIs are seed-0 CIs.
    # E25 f0 metrics are seed-invariant (bank_hash identical across seeds 0/1/2 for
    # every split, so the 300-image reference set is fixed and only the adapter
    # varies). E22 is the opposite: its 300-bank is redrawn per seed, and its f0
    # values differ across seeds by up to 0.0285 AUROC.
    table_specs = (
        ("results/e25/multiseed_summary.csv",
         {"auroc_f0_mean": ("mean", "auroc_f0"), "auroc_r1_mean": ("mean", "auroc_r1"),
          "d_auroc_mean": ("mean", "d_auroc"), "d_ap_mean": ("mean", "d_ap"),
          "d_auroc_std": ("std", "d_auroc"), "d_ap_std": ("std", "d_ap")}),
        ("results/e25/e25_summary.csv",
         {"auroc_f0": ("seed0", "auroc_f0"), "ap_f0": ("seed0", "ap_f0"),
          "auroc_r1": ("seed0", "auroc_r1"), "ap_r1": ("seed0", "ap_r1"),
          "d_auroc": ("seed0", "d_auroc"), "d_ap": ("seed0", "d_ap")}),
    )
    for rel, spec in table_specs:
        tag = Path(rel).stem
        for row in csv.DictReader(open(ROOT / rel)):
            key = (row["domain"], row["split"])
            ours = next((r for r in split_rows if (r["domain"], r["split"]) == key), None)
            if ours is None:
                add(f"E25-{tag}-{key[0]}-{key[1]}-missing", "RobustAD", "split_coverage",
                    "published split must exist in the recomputed set", f"{rel}:{key}",
                    1.0, 0.0, tol=0.0, check_type="categorical_exact", precision_dp=0)
                continue
            seed0 = split_seed0[key]
            for col, (kind, k) in spec.items():
                if col in row and row[col].strip() != "":
                    ref = {"mean": ours[k], "std": ours[f"{k}_std"], "seed0": seed0[k]}[kind]
                    add_published(f"E25-{tag}-{row['domain']}-{row['split']}-{col}", "RobustAD",
                                  f"{k}({kind})", row[col], ref, f"{rel}:{col}")
            for flag in ("is_source", "valid"):
                if flag in row:
                    add(f"E25-{tag}-{row['domain']}-{row['split']}-{flag}", "RobustAD", f"{flag}_flag",
                        "derived from cell metadata", f"{rel}:{flag}",
                        float(row[flag]), float(ours[flag]), tol=0.0,
                        check_type="categorical_exact", precision_dp=0)
            if "condition_group" in row and row["condition_group"].strip() != "":
                add(f"E25-{tag}-{row['domain']}-{row['split']}-condition_group", "RobustAD",
                    "condition_group", "copied from cell metadata", f"{rel}:condition_group",
                    1.0, float(row["condition_group"] == ours["condition_group"]), tol=0.0,
                    check_type="categorical_exact", precision_dp=0)

    return {"n_cells": len(recs), "n_valid_target_splits": len(targets),
            "domain_means": dom_means, "overall": overall,
            "splits": [{k: v for k, v in r.items()} for r in split_rows]}


# -------------------------------------------------------------------------- E21
def audit_e21(m2ad: dict) -> dict:
    """Verify the multi-view table (E21, 3 cats x 4 views, seed 0) from archived scores."""
    doc = json.loads((ROOT / "results" / "e21" / "e21_partial.json").read_text())
    views = {}
    for cell in doc["cells"].values():
        cat, seed = cell["category"], cell["seed"]
        for view, v in cell["views"].items():
            a0, p0 = metrics(v["scores_f0"], v["labels"], v["detectable_raw"])
            a1, p1 = metrics(v["scores_r1"], v["labels"], v["detectable_raw"])
            views[(cat, view)] = {"auroc_f0": a0, "ap_f0": p0, "auroc_r1": a1, "ap_r1": p1,
                                  "d_auroc": a1 - a0, "d_ap": p1 - p0,
                                  "n_ok": v["n_ok"], "n_ng": v["n_ng"], "seed": seed,
                                  "stored": v}
            add(f"E21-view-{cat}-{view}-auroc_f0", "E21", "auroc_f0", "recomputed from scores_f0",
                f"results/e21/e21_partial.json:views.{view}.auroc_f0", v["auroc_f0"], a0)
            add(f"E21-view-{cat}-{view}-ap_f0", "E21", "ap_f0", "recomputed from scores_f0",
                f"results/e21/e21_partial.json:views.{view}.ap_f0", v["ap_f0"], p0)
            add(f"E21-view-{cat}-{view}-auroc_r1", "E21", "auroc_r1", "recomputed from scores_r1",
                f"results/e21/e21_partial.json:views.{view}.auroc_r1", v["auroc_r1"], a1)
            add(f"E21-view-{cat}-{view}-ap_r1", "E21", "ap_r1", "recomputed from scores_r1",
                f"results/e21/e21_partial.json:views.{view}.ap_r1", v["ap_r1"], p1)

    rel = "results/paper_ready/paper_multiview_table.csv"
    for row in csv.DictReader(open(ROOT / rel)):
        key = (row["category"], row["view"])
        ours = views[key]
        for col, k in (("F0_AUROC", "auroc_f0"), ("R1_AUROC", "auroc_r1"), ("dAUROC", "d_auroc"),
                       ("F0_AP", "ap_f0"), ("R1_AP", "ap_r1"), ("dAP", "d_ap")):
            add_published(f"E21-table-{key[0]}-{key[1]}-{col}", "E21", k, row[col], ours[k],
                          f"{rel}:{col}")
        for col, k in (("n_good", "n_ok"), ("n_anomaly", "n_ng")):
            add(f"E21-table-{key[0]}-{key[1]}-{col}", "E21", col, "recomputed from view metadata",
                f"{rel}:{col}", float(row[col]), float(ours[k]), tol=0.0,
                check_type="categorical_exact", precision_dp=0)

    # cross-source coherence: view 120 is shared with E22 (same seed 0). This does not
    # verify a published number; it checks that the two independently stored artifacts
    # describe the same cell (published 5 dp, hence 1e-5).
    for cat in ("Bird", "Car", "Cube"):
        add(f"E21-E22-cross-view120-{cat}-d_auroc", "cross-source", "d_auroc",
            "E21 view120 (recomputed)", "E22 seed0 (recomputed from scores)",
            views[(cat, "120")]["d_auroc"],
            m2ad["per_cell"][f"{cat}_s0"]["d_auroc"], tol=1e-5,
            check_type="cross_source_1e-5", precision_dp=5)
    return {"n_views": len(views),
            "cats": sorted({c for c, _ in views}),
            "views": sorted({v for _, v in views}),
            "d_auroc": {f"{c}-{v}": round(m["d_auroc"], 5) for (c, v), m in views.items()}}


# -------------------------------------------------------------------- AnomalyDINO
def audit_anomalydino(m2ad: dict) -> dict:
    """Verify every published AnomalyDINO comparison number from its per-image scores."""
    per_cell = m2ad["per_cell"]
    cells = {}
    for cat in CATS:
        for seed in SEEDS:
            rec = json.loads((ROOT / "results" / "e26_anomalydino" / "scores"
                              / f"{cat}_seed{seed}.json").read_text())
            keep = [t for t in rec["tests"] if t.get("detectable") is not False]
            y = np.array([t["label"] for t in keep], dtype=int)
            s = np.array([t["score"] for t in keep], dtype=float)
            aad, pad = float(roc_auc_score(y, s)), float(average_precision_score(y, s))
            m = per_cell[f"{cat}_s{seed}"]
            cells[(cat, seed)] = {"auroc_ad": aad, "ap_ad": pad,
                                  "auroc_f0": m["auroc_f0"], "ap_f0": m["ap_f0"],
                                  "auroc_r1": m["auroc_r1"], "ap_r1": m["ap_r1"],
                                  "d_auroc_r1_minus_ad": m["auroc_r1"] - aad,
                                  "d_ap_r1_minus_ad": m["ap_r1"] - pad,
                                  "d_auroc_ad_minus_f0": aad - m["auroc_f0"],
                                  "d_ap_ad_minus_f0": pad - m["ap_f0"]}

    # 1) 30-cell table (6 dp)
    for row in csv.DictReader(open(ROOT / "results" / "e26_anomalydino" / "baseline_30cells.csv")):
        key = (row["category"], int(row["seed"]))
        ours = cells[key]
        for col in ("auroc_ad", "ap_ad", "d_auroc_r1_minus_ad", "d_ap_r1_minus_ad",
                    "d_auroc_ad_minus_f0", "d_ap_ad_minus_f0"):
            add_published(f"AD-30cells-{key[0]}-s{key[1]}-{col}", "AnomalyDINO", col, row[col],
                          ours[col], f"results/e26_anomalydino/baseline_30cells.csv:{col}")

    # 2) per-category summary (mean / sd over 3 seeds)
    cat_val = {}
    for cat in CATS:
        vals = [cells[(cat, s)] for s in SEEDS]
        agg = {}
        for k in vals[0]:
            agg[k] = st.mean(v[k] for v in vals)
            agg[f"{k}_sd"] = st.stdev(v[k] for v in vals)
        cat_val[cat] = agg
    for row in csv.DictReader(open(ROOT / "results" / "e26_anomalydino"
                                   / "baseline_category_summary.csv")):
        cat = row["category"]
        for col in row:
            if col.endswith("_mean"):
                k = col[: -len("_mean")]
                add_published(f"AD-catsum-{cat}-{col}", "AnomalyDINO", col, row[col],
                              cat_val[cat][k],
                              f"results/e26_anomalydino/baseline_category_summary.csv:{col}")
            elif col.endswith("_sd"):
                k = f"{col[: -len('_sd')]}_sd"
                add_published(f"AD-catsum-{cat}-{col}", "AnomalyDINO", col, row[col],
                              cat_val[cat][k],
                              f"results/e26_anomalydino/baseline_category_summary.csv:{col}")

    # 3) macro / dev / frozen levels
    groups = {"macro_10cat": CATS, "dev_3cat": [c for c in CATS if c in DEV],
              "ext_7cat": [c for c in CATS if c not in DEV]}
    macro = {}
    for level, group in groups.items():
        macro[level] = {f"{k}_macro": st.mean(cat_val[c][k] for c in group)
                        for k in cat_val[CATS[0]] if not k.endswith("_sd")}
    for row in csv.DictReader(open(ROOT / "results" / "e26_anomalydino" / "baseline_macro.csv")):
        level = row["level"]
        add(f"AD-macro-{level}-n_categories", "AnomalyDINO", "n_categories", "level definition",
            "results/e26_anomalydino/baseline_macro.csv:n_categories",
            float(row["n_categories"]), float(len(groups[level])), tol=0.0,
            check_type="categorical_exact", precision_dp=0)
        for col in macro[level]:
            add_published(f"AD-macro-{level}-{col}", "AnomalyDINO", col, row[col],
                          macro[level][col],
                          f"results/e26_anomalydino/baseline_macro.csv:{col}")

    keep_macro = ("auroc_ad_macro", "auroc_f0_macro", "auroc_r1_macro",
                  "d_auroc_r1_minus_ad_macro", "d_auroc_ad_minus_f0_macro")
    return {"n_cells": len(cells),
            "macro": {lvl: {k: v for k, v in m.items() if k in keep_macro}
                      for lvl, m in macro.items()}}


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    m2ad = audit_m2ad()
    robust = audit_robustad()
    ad = audit_anomalydino(m2ad)
    e21 = audit_e21(m2ad)

    fails = [c for c in CHECKS if not c["pass"]]
    ulp_only = [c for c in CHECKS if c["pass"] and c["rounds_to_published"] is False]
    by_type: dict[str, dict[str, int]] = {}
    for c in CHECKS:
        d = by_type.setdefault(c["check_type"], {"n": 0, "n_fail": 0})
        d["n"] += 1
        d["n_fail"] += 0 if c["pass"] else 1
    summary = {
        "date": "2026-11-01",
        "experiment": "E26-D",
        "tolerance_full_precision": TOL,
        "n_checks": len(CHECKS),
        "n_pass": len(CHECKS) - len(fails),
        "n_fail": len(fails),
        "by_check_type": by_type,
        "max_abs_diff_numeric": max(c["diff"] for c in CHECKS if c["check_type"] != "categorical_exact"),
        "n_pass_only_within_1ulp": len(ulp_only),
        "all_1ulp_explained": bool(ulp_only) and all(
            c["truncates_to_published"] or c["double_rounds_to_published"] for c in ulp_only),
        "n_1ulp_by_truncation": sum(1 for c in ulp_only if c["truncates_to_published"]),
        "n_1ulp_by_double_rounding": sum(1 for c in ulp_only
                                         if not c["truncates_to_published"]
                                         and c["double_rounds_to_published"]),
        "verdict": "PASS" if not fails else "FAIL",
        "failures": fails,
        "one_ulp_notes": ulp_only,
    }
    (OUT / "consistency_report.json").write_text(json.dumps(
        {"summary": summary, "m2ad": m2ad, "robustad": robust, "e21": e21,
         "anomalydino": ad, "checks": CHECKS}, indent=2))

    with open(OUT / "consistency_checks.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(CHECKS)

    print(json.dumps({k: summary[k] for k in ("n_checks", "n_pass", "n_fail", "max_abs_diff_numeric",
                                              "n_pass_only_within_1ulp", "verdict")}, indent=1))
    print("by_check_type:", json.dumps(by_type, indent=1))
    if fails:
        print("FAILURES:")
        for c in fails:
            print(f"  {c['id']}: expected={c['expected']!r} actual={c['actual']!r} diff={c['diff']:.3e}")
    if ulp_only:
        print("1-ulp notes (pass at published precision, printed digit not equal to round-half-up):")
        for c in ulp_only:
            print(f"  {c['id']}: printed={c['expected']!r} recomputed={c['actual']!r} diff={c['diff']:.3e}")
    print("M2AD macro:", json.dumps(m2ad["macro_10cat"], indent=1))
    print("visible per-category dAUROC:", {c: round(m2ad["category"][c]["d_auroc"], 5) for c in CATS})
    print("RobustAD overall:", json.dumps(robust["overall"], indent=1))
    print(f"[E26-D] -> {OUT}")
    return 0 if not fails else 1


if __name__ == "__main__":
    raise SystemExit(main())
