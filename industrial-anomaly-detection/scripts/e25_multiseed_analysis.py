#!/usr/bin/env python
"""E25 multi-seed summary - seed 0/1/2 for frozen R1-ILA on RobustAD.

Reads results/e25/cells/*.json (seed 0 files keep the original names), and writes:
  results/e25/multiseed_summary.csv     per (domain, split): mean/std over seeds
  results/e25/multiseed_analysis.json   full machine-readable summary
  results/e25/figures/                  E25 multi-seed figures (PNG+PDF)

Guards
------
* Recomputes AUROC/AP per (cell, method) from stored per-image scores; tol 1e-12.
* Aggregates only valid=1 non-source splits, identical to e25_analysis.py.
* Per-seed paired bootstrap CI (images resampled, both methods paired) is reported
  next to the across-seed spread; the two are NOT interchangeable (bootstrap = test-set
  sampling noise, seed spread = adapter-training randomness).
* No parameter is re-selected here.
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
FIGDIR = OUTDIR / "figures"
N_BOOT = 3000
RNG = 20261018
TOL = 1e-12
DOMAINS = ["PCB", "MetalParts"]


def load_cells() -> list[dict]:
    out = []
    for p in sorted(CELLDIR.glob("*.json")):
        c = json.loads(p.read_text())
        if c.get("status") != "OK":
            continue
        out.append(c)
    if not out:
        raise SystemExit(f"no cells in {CELLDIR}")
    return out


def verify(c: dict) -> float:
    d = [abs(roc_auc_score(c["labels"], c["scores_f0"]) - c["f0"]["auroc"]),
         abs(average_precision_score(c["labels"], c["scores_f0"]) - c["f0"]["ap"]),
         abs(roc_auc_score(c["labels"], c["scores_r1"]) - c["r1"]["auroc"]),
         abs(average_precision_score(c["labels"], c["scores_r1"]) - c["r1"]["ap"])]
    return max(d)


def boot(s0, s1, labels, n_boot=N_BOOT, seed=RNG) -> dict:
    s0, s1 = np.asarray(s0, float), np.asarray(s1, float)
    y = np.asarray(labels, int)
    rng = np.random.default_rng(seed)
    dau, dap = [], []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        yy = y[idx]
        if yy.min() == yy.max():
            continue
        dau.append(roc_auc_score(yy, s1[idx]) - roc_auc_score(yy, s0[idx]))
        dap.append(average_precision_score(yy, s1[idx]) - average_precision_score(yy, s0[idx]))
    dau, dap = np.asarray(dau), np.asarray(dap)
    return {"d_auroc_ci95": [float(np.percentile(dau, 2.5)), float(np.percentile(dau, 97.5))],
            "d_ap_ci95": [float(np.percentile(dap, 2.5)), float(np.percentile(dap, 97.5))]}


def t_interval(vals) -> list[float]:
    """Descriptive 95% t-interval over seeds (df = n-1); NOT a bootstrap."""
    v = np.asarray(vals, float)
    if len(v) < 2:
        return [float(v[0]), float(v[0])]
    from scipy.stats import t
    half = t.ppf(0.975, len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v))
    return [float(v.mean() - half), float(v.mean() + half)]


def main() -> int:
    cells = load_cells()
    worst = max(verify(c) for c in cells)
    if worst > TOL:
        print(f"!! recomputation gate FAILED max|d|={worst:.3e}")
        return 2
    seeds = sorted({int(c["seed"]) for c in cells})
    print(f"[E25-ms] cells={len(cells)} seeds={seeds} recomputation PASS (max|d|={worst:.1e})")

    keyed: dict[tuple, dict[int, dict]] = {}
    for c in cells:
        keyed.setdefault((c["domain"], c["split"]), {})[int(c["seed"])] = c

    rows, detail = [], {}
    for (dom, split) in sorted(keyed, key=lambda k: (DOMAINS.index(k[0]),
                                                     int(k[1].replace("test", "")))):
        per = keyed[(dom, split)]
        sids = sorted(per)
        c0 = per[sids[0]]
        d_au = [per[s]["d_auroc"] for s in sids]
        d_ap = [per[s]["d_ap"] for s in sids]
        cis = {s: boot(per[s]["scores_f0"], per[s]["scores_r1"], per[s]["labels"])
               for s in sids}
        row = {
            "domain": dom, "split": split,
            "condition_group": c0["condition_group"], "valid": c0["valid"],
            "is_source": int(split == "test0"), "n_seeds": len(sids),
            "n_normal": c0["n_ok"], "n_anomaly": c0["n_ng"],
            "auroc_f0_mean": float(np.mean([per[s]["f0"]["auroc"] for s in sids])),
            "auroc_r1_mean": float(np.mean([per[s]["r1"]["auroc"] for s in sids])),
            "d_auroc_mean": float(np.mean(d_au)), "d_auroc_std": float(np.std(d_au, ddof=1))
            if len(sids) > 1 else 0.0,
            "d_ap_mean": float(np.mean(d_ap)), "d_ap_std": float(np.std(d_ap, ddof=1))
            if len(sids) > 1 else 0.0,
            "n_pos_seeds_auroc": int(sum(x > 0 for x in d_au)),
            "n_pos_seeds_ap": int(sum(x > 0 for x in d_ap)),
            "seed_values_d_auroc": ";".join(f"{s}:{per[s]['d_auroc']:+.6f}" for s in sids),
            "seed_values_d_ap": ";".join(f"{s}:{per[s]['d_ap']:+.6f}" for s in sids),
            "ci95_seed0": cis[sids[0]]["d_auroc_ci95"],
        }
        rows.append(row)
        detail[f"{dom}|{split}"] = {"seeds": sids,
                                    "d_auroc": d_au, "d_ap": d_ap,
                                    "bootstrap_by_seed": {str(s): cis[s] for s in sids},
                                    "t_interval_d_auroc": t_interval(d_au)}

    # ---------------- CSV
    out_rows = []
    for r in rows:
        d = dict(r)
        d["ci95_seed0"] = f"[{r['ci95_seed0'][0]:+.5f},{r['ci95_seed0'][1]:+.5f}]"
        out_rows.append(d)
    with open(OUTDIR / "multiseed_summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out_rows[0]))
        w.writeheader()
        for r in out_rows:
            w.writerow({k: (f"{v:.6f}" if isinstance(v, float) and k not in
                            ("is_source",) else v) for k, v in r.items()})

    # ---------------- aggregates
    targets = [r for r in rows if r["valid"] == 1 and r["is_source"] == 0]
    dom_stats = {}
    for dom in DOMAINS:
        sub = [r for r in targets if r["domain"] == dom]
        per_seed_means = {}
        for s in seeds:
            vals = [keyed[(r["domain"], r["split"])][s]["d_auroc"] for r in sub
                    if s in keyed[(r["domain"], r["split"])]]
            if vals:
                per_seed_means[s] = float(np.mean(vals))
        dom_stats[dom] = {
            "n_splits": len(sub),
            "d_auroc_mean": float(np.mean([r["d_auroc_mean"] for r in sub])),
            "d_auroc_median": float(np.median([r["d_auroc_mean"] for r in sub])),
            "d_ap_mean": float(np.mean([r["d_ap_mean"] for r in sub])),
            "auroc_f0_mean": float(np.mean([r["auroc_f0_mean"] for r in sub])),
            "auroc_r1_mean": float(np.mean([r["auroc_r1_mean"] for r in sub])),
            "per_seed_d_auroc_mean": {str(k): v for k, v in per_seed_means.items()},
            "per_seed_d_auroc_std": float(np.std(list(per_seed_means.values()), ddof=1))
            if len(per_seed_means) > 1 else 0.0,
            "n_splits_pos_mean": int(sum(r["d_auroc_mean"] > 0 for r in sub)),
            "n_splits_pos_all_seeds": int(sum(r["n_pos_seeds_auroc"] == len(seeds)
                                              for r in sub)),
            "n_splits_neg_all_seeds": int(sum(r["n_pos_seeds_auroc"] == 0 for r in sub)),
            "n_splits_ci_excl0_seed0": int(sum(
                (r["ci95_seed0"][0] > 0 or r["ci95_seed0"][1] < 0) for r in sub)),
        }

    macro = float(np.mean([r["d_auroc_mean"] for r in targets]))
    macro_ap = float(np.mean([r["d_ap_mean"] for r in targets]))
    macro_per_seed = {}
    for s in seeds:
        vals = [keyed[(r["domain"], r["split"])][s]["d_auroc"] for r in targets
                if s in keyed[(r["domain"], r["split"])]]
        macro_per_seed[s] = float(np.mean(vals))
    structure = {
        "pcb_positive": dom_stats["PCB"]["d_auroc_mean"] > 0,
        "mp_negative": dom_stats["MetalParts"]["d_auroc_mean"] < 0,
        "pcb_positive_all_seeds": all(v > 0 for v in
                                      dom_stats["PCB"]["per_seed_d_auroc_mean"].values()),
        "mp_negative_all_seeds": all(v < 0 for v in
                                     dom_stats["MetalParts"]["per_seed_d_auroc_mean"].values()),
    }
    analysis = {
        "experiment": "E25 multi-seed",
        "cells": len(cells), "seeds": seeds,
        "config_source": "results/e17/final_method.json",
        "frozen": {"BETA": 0.1, "augmentations": "brightness+contrast+gamma",
                   "strength": [0.85, 1.15], "lr": 0.001, "max_steps": 300,
                   "patience": 30, "batch_size": 8},
        "verification": {"max_abs_diff": worst, "tolerance": TOL, "pass": True},
        "per_domain": dom_stats,
        "per_split": rows,
        "overall": {
            "n_target_splits": len(targets),
            "macro_d_auroc_mean_over_splits": macro,
            "macro_d_ap_mean_over_splits": macro_ap,
            "macro_d_auroc_per_seed": {str(k): v for k, v in macro_per_seed.items()},
            "macro_d_auroc_seed_std": float(np.std(list(macro_per_seed.values()), ddof=1)),
            "macro_d_auroc_t_interval95": t_interval(list(macro_per_seed.values())),
            "positive_split_ratio_auroc": float(np.mean([r["d_auroc_mean"] > 0
                                                         for r in targets])),
            "positive_split_ratio_ap": float(np.mean([r["d_ap_mean"] > 0
                                                      for r in targets])),
            "splits_positive_all_seeds": int(sum(r["n_pos_seeds_auroc"] == len(seeds)
                                                 for r in targets)),
            "splits_negative_all_seeds": int(sum(r["n_pos_seeds_auroc"] == 0
                                                 for r in targets)),
            "worst_split": min(targets, key=lambda r: r["d_auroc_mean"])["domain"] + ":"
                           + min(targets, key=lambda r: r["d_auroc_mean"])["split"],
            "worst_d_auroc_mean": min(r["d_auroc_mean"] for r in targets),
            "best_split": max(targets, key=lambda r: r["d_auroc_mean"])["domain"] + ":"
                          + max(targets, key=lambda r: r["d_auroc_mean"])["split"],
            "best_d_auroc_mean": max(r["d_auroc_mean"] for r in targets),
        },
        "structure_preserved": structure,
        "limitations": [
            "3 seeds only; adapter-training randomness is sampled, not exhausted",
            "single backbone / single scorer",
            "plain AUROC/AP (no `detectable` field in RobustAD)",
            "MetalParts inputs are anisotropically resized to 448x448 (protocol §10)",
            "MetalParts test0 excluded (overlaps source train)",
            "PiledBags not covered"],
    }
    (OUTDIR / "multiseed_analysis.json").write_text(json.dumps(analysis, indent=2, default=float))

    print(f"\n{'domain':12s} {'split':7s} {'dAUROC mean±std':>22s} {'dAP mean±std':>22s} "
          f"{'pos/3':>6s}")
    for r in rows:
        print(f"{r['domain']:12s} {r['split']:7s} "
              f"{r['d_auroc_mean']:+.5f} ± {r['d_auroc_std']:.5f}   "
              f"{r['d_ap_mean']:+.5f} ± {r['d_ap_std']:.5f}   "
              f"{r['n_pos_seeds_auroc']}/3"
              f"{'  (valid=0)' if r['valid'] == 0 else ''}")
    print(f"\n[E25-ms] PCB mean {dom_stats['PCB']['d_auroc_mean']:+.5f} "
          f"(per-seed {dom_stats['PCB']['per_seed_d_auroc_mean']})")
    print(f"[E25-ms] MP  mean {dom_stats['MetalParts']['d_auroc_mean']:+.5f} "
          f"(per-seed {dom_stats['MetalParts']['per_seed_d_auroc_mean']})")
    print(f"[E25-ms] overall {macro:+.5f} dAUROC / {macro_ap:+.5f} dAP | "
          f"positive {analysis['overall']['positive_split_ratio_auroc']:.3f} | "
          f"structure {structure}")
    print(f"[E25-ms] -> {OUTDIR / 'multiseed_analysis.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
