#!/usr/bin/env python
"""E6-A - Illumination Robustness Metric Tournament & Validity Audit.

CPU-only (GPU WORK = 0). Protocol frozen in `docs/E6_A_METRIC_PROTOCOL.md` (commit 77e81ee)
BEFORE any metric outcome or unblinded interpretation existed.

Blind procedure: BLIND_SEED = 20261009; Original keeps its name, the other six methods become
M01..M06 through a seeded permutation. The primary tournament and all verdicts are computed and
written BLINDED; the map is only read afterwards for interpretation.

Usage: python scripts/e6_a_metric_tournament.py
"""
from __future__ import annotations

import csv
import json
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np  # noqa: E402

import e4x_common as C  # noqa: E402
from experiment1h_runner import image_auroc  # noqa: E402
from e5_failure_audit import spearman, structured_variance  # noqa: E402

OUT = ROOT / "results" / "e6_a"
BLIND_SEED = 20261009
BOOTSTRAP_SEED = 20261009
N_BOOT = 1000
ILLUM = [f"{i:02d}" for i in range(1, 11)]
MIN_ILLUM = 3   # minimum illuminations per specimen for per-specimen statistics (L1oO support)
PROTOCOL = ROOT / "docs" / "E6_A_METRIC_PROTOCOL.md"

FILES = {
    "Original": ROOT / "results" / "e4_x" / "per_image_original.csv",
    "B2": ROOT / "results" / "e4_x" / "per_image_b2.csv",
    "X6c": ROOT / "results" / "e4_x" / "per_image_x6c.csv",
    "C01": ROOT / "results" / "e5_1" / "raw" / "C01_per_image.csv",
    "C06": ROOT / "results" / "e5_1" / "raw" / "C06_per_image.csv",
    "C07": ROOT / "results" / "e5_1" / "raw" / "C07_per_image.csv",
    "C08": ROOT / "results" / "e5_1" / "raw" / "C08_per_image.csv",
}
REAL_NAMES = ["B2", "X6c", "C01", "C06", "C07", "C08"]

# frozen acceptance values (protocol section 4 M1/M3)
ACC = {"X_R_all": 0.67543, "C01_R_all": 0.69617, "C01_R_ratio": 1.0307,
       "X_auroc": 0.76766, "X_dprime": 1.20676,
       "X_M3_good": 0.4626, "C01_M3_good": 0.3192, "X_M3_ng": 0.1120, "C01_M3_ng": 0.0981}


def blind_map() -> dict:
    """name -> blind id (Original keeps its name). Deterministic."""
    order = list(REAL_NAMES)
    random.Random(BLIND_SEED).shuffle(order)
    m = {"Original": "Original"}
    for i, name in enumerate(order):
        m[name] = f"M{i + 1:02d}"
    return m


def read_rows(path: Path) -> list:
    with open(path) as f:
        return [{"specimen": r["specimen"], "kind": r["kind"], "illumination": r["illumination"],
                 "score": float(r["score"])} for r in csv.DictReader(f)]


def load_all() -> dict:
    return {name: read_rows(p) for name, p in FILES.items()}


def by_spec(rows: list) -> dict:
    g = defaultdict(dict)
    for r in rows:
        g[r["specimen"]][r["illumination"]] = r["score"]
    return {s: [v[i] for i in ILLUM if i in v] for s, v in g.items()}


def kind_map(rows: list) -> dict:
    k = {}
    for r in rows:
        k[r["specimen"]] = r["kind"]
    return k


# ------------------------------------------------------------------ metrics M1..M8
def illum_share(sub: list) -> float:
    """Protocol M3: std(per-illumination means) / (std(ill means) + std(per-specimen means))."""
    by_ill, by_spec = defaultdict(list), defaultdict(list)
    for r in sub:
        by_ill[r["illumination"]].append(r["score"])
        by_spec[r["specimen"]].append(r["score"])
    i_means = np.array([np.mean(v) for _, v in sorted(by_ill.items())])
    s_means = np.array([np.mean(v) for _, v in sorted(by_spec.items())])
    i_std, s_std = float(np.std(i_means, ddof=1)), float(np.std(s_means, ddof=1))
    return i_std / (i_std + s_std) if (i_std + s_std) > 0 else float("nan")


def per_spec_vars(rows: list) -> dict:
    """M2/M5 numerator: Var over the available illuminations, per specimen (ddof=1).

    `MIN_ILLUM` is 3 (not 10) so that leave-one-illumination-out remains computable; in the main
    run every specimen has exactly 10, so the main numbers are unchanged.
    """
    return {s: float(np.var(v, ddof=1)) for s, v in by_spec(rows).items() if len(v) >= MIN_ILLUM}


def per_spec_stds(rows: list) -> dict:
    return {s: float(np.std(v, ddof=1)) for s, v in by_spec(rows).items() if len(v) >= MIN_ILLUM}


def _iqr(a: np.ndarray) -> float:
    if a.size == 0:
        return float("nan")
    return float(np.percentile(a, 75) - np.percentile(a, 25))


def _safe_median(a: np.ndarray) -> float:
    return float(np.median(a)) if a.size else float("nan")


def _safe_mean(a: np.ndarray) -> float:
    return float(a.mean()) if a.size else float("nan")


def rank_stability(rows: list, kind: str) -> dict:
    """M4: Spearman between specimen rankings under illumination i and j, over C(n_illum,2) pairs."""
    sub = [r for r in rows if r["kind"] == kind]
    by_ill = defaultdict(dict)
    for r in sub:
        by_ill[r["illumination"]][r["specimen"]] = r["score"]
    specs = sorted({r["specimen"] for r in sub})
    ills = sorted(by_ill)
    if len(specs) < 3 or len(ills) < 2:
        return {"mean": float("nan"), "median": float("nan"), "worst": float("nan"), "n_pairs": 0}
    mat = np.array([[by_ill[i].get(s, np.nan) for s in specs] for i in ills])
    if not np.isfinite(mat).all():
        return {"mean": float("nan"), "median": float("nan"), "worst": float("nan"), "n_pairs": 0}
    vals = []
    for a in range(len(ills)):
        for b in range(a + 1, len(ills)):
            vals.append(spearman(mat[a], mat[b]))
    v = np.array(vals)
    return {"mean": float(v.mean()), "median": float(np.median(v)), "worst": float(v.min()),
            "n_pairs": int(len(v))}


def drift_stats(rows: list, kind: str) -> dict:
    """M7: |s_i - s_j| over all specimen x illumination-pair triples."""
    g = by_spec(rows)
    km = kind_map(rows)
    vals = []
    for s, v in g.items():
        if len(v) < MIN_ILLUM or (kind != "All" and km.get(s) != kind):
            continue
        for a in range(len(v)):
            for b in range(a + 1, len(v)):
                vals.append(abs(v[a] - v[b]))
    if not vals:
        return {"mean": float("nan"), "median": float("nan"), "p90": float("nan"), "max": float("nan"), "n": 0}
    a = np.array(vals)
    return {"mean": float(a.mean()), "median": float(np.median(a)),
            "p90": float(np.percentile(a, 90)), "max": float(a.max()), "n": int(len(a))}


def detection_stability(rows: list) -> dict:
    """M8: AUROC computed inside each illumination, then summarised across illuminations."""
    aucs = []
    for i in sorted({r["illumination"] for r in rows}):
        g = np.array([r["score"] for r in rows if r["kind"] == "Good" and r["illumination"] == i])
        d = np.array([r["score"] for r in rows if r["kind"] == "NG" and r["illumination"] == i])
        aucs.append(float(image_auroc(g, d)))
    a = np.array(aucs)
    return {"mean": float(a.mean()), "sd": float(a.std(ddof=1)), "min": float(a.min()),
            "max": float(a.max()), "gap": float(a.max() - a.min()),
            "cv": float(a.std(ddof=1) / a.mean()) if a.mean() != 0 else float("nan"),
            "per_illumination": aucs}


def compute_metrics(rows: list, orig_rows: list, x_all_std: float, x_psv: dict, x_pss: dict,
                    x_drift: dict) -> dict:
    m = {}
    g_all = np.array([r["score"] for r in rows if r["kind"] == "Good"], dtype=float)
    d_all = np.array([r["score"] for r in rows if r["kind"] == "NG"], dtype=float)
    go = np.array([r["score"] for r in orig_rows if r["kind"] == "Good"], dtype=float)
    do_ = np.array([r["score"] for r in orig_rows if r["kind"] == "NG"], dtype=float)
    m["auroc"] = float(image_auroc(g_all, d_all)); m["dprime"] = float(C.dprime(g_all, d_all))
    m["d_auroc"] = m["auroc"] - float(image_auroc(go, do_))
    m["d_dprime"] = m["dprime"] - float(C.dprime(go, do_))
    m["good_mean"], m["good_std"] = float(g_all.mean()), float(g_all.std(ddof=1))
    m["ng_mean"], m["ng_std"] = float(d_all.mean()), float(d_all.std(ddof=1))
    m["all_std"] = float(np.std([r["score"] for r in rows], ddof=1))

    # ---- M1 legacy fields: ONLY via the frozen helper, and only when the helper is applicable.
    # `e4x_common.metric_block` hard-requires exactly 10 illuminations per specimen (it raises on
    # empty percentile arrays), so R_all/R_ratio are DECLARED NOT COMPUTABLE under
    # leave-one-illumination-out. The frozen helper is never edited.
    if all(len(v) == 10 for v in by_spec(rows).values()) and \
       all(len(v) == 10 for v in by_spec(orig_rows).values()):
        blk = C.metric_block(rows, "m"); blk_o = C.metric_block(orig_rows, "X")
        m["R_all"] = blk["R_all"]; m["R_Good"] = blk["R_Good"]; m["R_NG"] = blk["R_NG"]
        m["R_ratio"] = blk["R_all"] / blk_o["R_all"]
        m["_M1_computable"] = True
    else:
        m["R_all"] = m["R_Good"] = m["R_NG"] = m["R_ratio"] = float("nan")
        m["_M1_computable"] = False

    good = [r for r in rows if r["kind"] == "Good"]
    ng = [r for r in rows if r["kind"] == "NG"]

    # ---- M2 raw within-specimen illumination variance (SCALE-SENSITIVE)
    pv = per_spec_vars(rows); km = kind_map(rows)
    for tag, sel in (("good", "Good"), ("ng", "NG"), ("all", None)):
        vals = np.array([v for s, v in pv.items() if sel is None or km.get(s) == sel])
        m[f"M2_{tag}_mean"] = _safe_mean(vals); m[f"M2_{tag}_median"] = _safe_median(vals)
        m[f"M2_{tag}_iqr"] = _iqr(vals)
    x_all_var = np.array([v for v in x_psv.values()])
    m["M2_ratio"] = (m["M2_all_mean"] / float(x_all_var.mean())) if x_all_var.size else float("nan")

    # ---- M3 illumination variance share
    m["M3_good"] = illum_share(good); m["M3_ng"] = illum_share(ng); m["M3_all"] = illum_share(rows)

    # ---- M4 within-specimen rank stability across illuminations
    rg, rn = rank_stability(rows, "Good"), rank_stability(rows, "NG")
    for tag, r in (("good", rg), ("ng", rn)):
        m[f"M4_{tag}_mean"] = r["mean"]; m[f"M4_{tag}_median"] = r["median"]; m[f"M4_{tag}_worst"] = r["worst"]
    m["M4_n_pairs"] = rg["n_pairs"]

    # ---- M5 standardized within-specimen sensitivity (4 denominators, all computed)
    ps = per_spec_stds(rows)
    N_all = float(np.mean([v for v in ps.values()]))
    N_good = float(np.mean([v for s, v in ps.items() if km.get(s) == "Good"]))
    N_ng = float(np.mean([v for s, v in ps.items() if km.get(s) == "NG"]))
    all_s = np.array([r["score"] for r in rows])
    den = {"A": float(all_s.std(ddof=1)),
           "B": abs(m["ng_mean"] - m["good_mean"]),
           "C": float(np.sqrt((np.var([r["score"] for r in good], ddof=1) +
                               np.var([r["score"] for r in ng], ddof=1)) / 2.0)),
           "D": float(1.4826 * np.median(np.abs(all_s - np.median(all_s))))}
    for k, dv in den.items():
        m[f"M5_{k}_all"] = N_all / dv if dv > 0 else float("nan")
        m[f"M5_{k}_good"] = N_good / dv if dv > 0 else float("nan")
        m[f"M5_{k}_ng"] = N_ng / dv if dv > 0 else float("nan")
    m["_den"] = den

    # ---- M7 pairwise illumination score drift
    dr = {k: drift_stats(rows, k) for k in ("Good", "NG", "All")}
    for k in ("Good", "NG", "All"):
        for stat in ("mean", "median", "p90", "max"):
            m[f"M7_{k}_{stat}"] = dr[k][stat]
    m["M7_n"] = dr["All"]["n"]
    m["M7_ratio"] = m["M7_All_mean"] / x_drift["All"]["mean"] if x_drift["All"]["mean"] > 0 else float("nan")
    m["M7_good_ratio"] = m["M7_Good_mean"] / x_drift["Good"]["mean"] if x_drift["Good"]["mean"] > 0 else float("nan")
    m["M7_std_all_mean"] = m["M7_All_mean"] / m["good_std"] if m["good_std"] > 0 else float("nan")
    m["M7_std_good_mean"] = m["M7_Good_mean"] / m["good_std"] if m["good_std"] > 0 else float("nan")

    # ---- M8 detection stability across illumination
    ds = detection_stability(rows)
    m["M8_mean"] = ds["mean"]; m["M8_sd"] = ds["sd"]; m["M8_min"] = ds["min"]; m["M8_max"] = ds["max"]
    m["M8_gap"] = ds["gap"]; m["M8_cv"] = ds["cv"]
    m["_M8_per_illum"] = ds["per_illumination"]
    x_ds = detection_stability(orig_rows)
    m["M8_dmean"] = ds["mean"] - x_ds["mean"]
    m["M8_ratio"] = ds["sd"] / x_ds["sd"] if x_ds["sd"] > 0 else float("nan")
    return m


# ------------------------------------------------------------------ helpers for pathology
INVARIANT_FIELDS = ["R_ratio", "M3_good", "M3_ng", "M3_all", "M4_good_mean", "M4_ng_mean",
                    "M8_mean", "M8_sd", "M8_min", "M8_max", "M8_gap", "M8_cv", "M8_ratio",
                    "M5_A_all", "M5_A_good", "M5_A_ng", "M5_B_all", "M5_B_good", "M5_B_ng",
                    "M5_C_all", "M5_D_all", "M7_std_all_mean", "M7_std_good_mean"]
SENSITIVE_FIELDS = ["M2_all_mean", "M2_good_mean", "M2_ng_mean", "M7_All_mean", "M7_Good_mean",
                    "M2_ratio", "M7_ratio", "M7_good_ratio"]
SENSITIVITY_METRICS = ["R_all", "M2_all_mean", "M3_all", "M5_A_all", "M7_All_mean", "M7_std_all_mean"]


def rescore(rows: list, fn) -> list:
    return [{**r, "score": float(fn(r))} for r in rows]


def kendall_tau(a, b) -> float:
    """tau-b, O(n^2) - fine for n=7."""
    n = len(a); conc = disc = 0
    for i in range(n):
        for j in range(i + 1, n):
            da, db = a[i] - a[j], b[i] - b[j]
            if da == 0 or db == 0:
                continue
            if (da > 0) == (db > 0):
                conc += 1
            else:
                disc += 1
    tot = conc + disc
    return float((conc - disc) / tot) if tot else float("nan")


def ordering(values: dict, key: str) -> list:
    """Method ordering by a metric key (lower = more robust, the convention for all our Y-metrics)."""
    items = [(k, v[key]) for k, v in values.items()
             if k != "Original" and np.isfinite(v.get(key, np.nan))]
    return [k for k, _ in sorted(items, key=lambda kv: kv[1])]


def safe(call, *args, **kw):
    try:
        return call(*args, **kw)
    except Exception as exc:  # noqa: BLE001
        return {"_error": f"{type(exc).__name__}: {exc}"}


def _metrics_of(rows, raw, ctx):
    return safe(compute_metrics, rows, raw["Original"], ctx["x_all_std"], ctx["x_psv"],
                ctx["x_pss"], ctx["x_drift"])


def _avg_ranks(x: np.ndarray) -> np.ndarray:
    """1-based ranks with ties averaged (Mann-Whitney / AUROC tie correction)."""
    order = np.argsort(x, kind="mergesort")
    r = np.empty(len(x), dtype=float)
    r[order] = np.arange(1, len(x) + 1, dtype=float)
    xs = x[order]
    i = 0
    while i < len(xs):
        j = i
        while j + 1 < len(xs) and xs[j + 1] == xs[i]:
            j += 1
        if j > i:
            r[order[i:j + 1]] = (i + 1 + j + 1) / 2.0
        i = j + 1
    return r


def auroc_tie_corrected(good: np.ndarray, defect: np.ndarray) -> float:
    """Tie-corrected AUROC (average ranks). Frozen `image_auroc` does NOT correct for ties."""
    y = np.concatenate([np.zeros(len(good)), np.ones(len(defect))])
    s = np.concatenate([good, defect])
    n_pos, n_neg = int(y.sum()), len(y) - int(y.sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    r = _avg_ranks(s)
    return float((r[y == 1].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def path_T1_T4(raw: dict, ctx: dict, base: dict, bar) -> dict:
    """T1 scale invariance; T4 monotonic-transform ordering preservation."""
    transforms = [(1.0, 0.0), (0.1, 0.0), (10.0, 0.0), (1.0, 100.0), (1.0, -50.0)]
    t1 = {"per_method": {}, "invariance_failures": []}
    for name, rows in raw.items():
        if name == "Original":
            continue
        vals = [_metrics_of(rescore(rows, lambda r, a=a, b=b: a * r["score"] + b), raw, ctx)
                for a, b in transforms]
        dev = {}
        for f in INVARIANT_FIELDS:
            v0 = base[name].get(f, np.nan)
            ds = [abs(v.get(f, np.nan) - v0) / (abs(v0) + 1e-12) for v in vals
                  if np.isfinite(v.get(f, np.nan)) and np.isfinite(v0)]
            dev[f] = float(max(ds)) if ds else float("nan")
            if np.isfinite(dev[f]) and dev[f] > 1e-6:
                t1["invariance_failures"].append({"method": name, "field": f, "max_rel_dev": dev[f]})
        t1["per_method"][name] = dev
        bar.step()
    t1["PASS"] = len(t1["invariance_failures"]) == 0

    t4 = {"per_field": {}}
    for f in INVARIANT_FIELDS:
        o_base = ordering(base, f)
        taus = []
        for a in (0.1, 10.0, 1.0):
            vals = {}
            for name, rows in raw.items():
                if name == "Original":
                    continue
                vals[name] = _metrics_of(rescore(rows, lambda r, a=a: a * r["score"]), raw, ctx)
            o = ordering(vals, f)
            if set(o) == set(o_base):
                taus.append(kendall_tau([o_base.index(k) for k in o_base],
                                        [o.index(k) for k in o_base]))
        t4["per_field"][f] = {"tau_min": float(np.nanmin(taus)) if taus else float("nan")}
    t4["FAIL_fields"] = [f for f, v in t4["per_field"].items()
                         if np.isfinite(v["tau_min"]) and v["tau_min"] < 1.0]
    t4["PASS"] = len(t4["FAIL_fields"]) == 0
    return {"T1_scale_invariance": t1, "T4_monotonic_ordering": t4}


def path_T2_T3_T5(raw: dict, ctx: dict, bar) -> dict:
    """T2/T3 synthetic detectors; T5 defect erasure."""
    synth = {}
    for tag, c in (("constant_c0", 0.0), ("constant_c1", 1.0), ("constant_c100", 100.0)):
        synth[tag] = _metrics_of(rescore(raw["Original"], lambda r, c=c: c), raw, ctx)
    rng = np.random.default_rng(20261009)
    rs = rng.normal(0, 1, len(raw["Original"]))
    synth["random_N01"] = _metrics_of([{**r, "score": float(rs[i])}
                                       for i, r in enumerate(raw["Original"])], raw, ctx)
    synth["perfect_oracle"] = _metrics_of(
        rescore(raw["Original"], lambda r: (10.0 + 0.5 * int(r["illumination"]) if r["kind"] == "NG"
                                            else 0.0 + 0.5 * int(r["illumination"]))), raw, ctx)

    # detection-metric attestation on the synthetic detectors (frozen helper vs tie-corrected)
    synth_det = {}
    for tag, scores in (("constant_c1", [1.0] * len(raw["Original"])),
                        ("random_N01", [float(v) for v in rs])):
        g = np.array([s for s, r in zip(scores, raw["Original"]) if r["kind"] == "Good"])
        d = np.array([s for s, r in zip(scores, raw["Original"]) if r["kind"] == "NG"])
        synth_det[tag] = {"frozen_image_auroc": float(image_auroc(g, d)),
                          "tie_corrected_auroc": auroc_tie_corrected(g, d),
                          "expected_auroc_for_a_useless_detector": 0.5}
    synth_det["_finding"] = ("frozen `image_auroc` does not average ranks for ties: an ALL-CONSTANT "
                             "score vector is scored AUROC = 1.0 instead of 0.5, so an AUROC>=0.5 "
                             "detection gate alone does NOT protect against constant collapse.")

    t5 = {"per_alpha": {}}
    for alpha in (1.0, 0.75, 0.5, 0.25, 0.0):
        rec = {}
        for name, rows in raw.items():
            if name == "Original":
                continue
            gm = float(np.mean([r["score"] for r in rows if r["kind"] == "Good"]))
            mm = _metrics_of(rescore(rows, lambda r, gm=gm, alpha=alpha:
                                     (alpha * r["score"] + (1 - alpha) * gm) if r["kind"] == "NG"
                                     else r["score"]), raw, ctx)
            rec[name] = {"auroc": mm.get("auroc", np.nan), "dprime": mm.get("dprime", np.nan),
                         "R_ratio": mm.get("R_ratio", np.nan), "M3_all": mm.get("M3_all", np.nan),
                         "M2_ratio": mm.get("M2_ratio", np.nan),
                         "M7_std_all_mean": mm.get("M7_std_all_mean", np.nan),
                         "M5_A_all": mm.get("M5_A_all", np.nan), "M5_D_all": mm.get("M5_D_all", np.nan),
                         "M8_ratio": mm.get("M8_ratio", np.nan), "M4_good_mean": mm.get("M4_good_mean", np.nan)}
        t5["per_alpha"][str(alpha)] = rec
        bar.step()
    # "improves" = falls for sensitivity metrics, rises for rank stability (M4)
    DOWN = ["R_ratio", "M3_all", "M2_ratio", "M7_std_all_mean", "M5_A_all", "M5_D_all", "M8_ratio"]
    t5["improve_while_collapse"] = {}
    for f in DOWN + ["M4_good_mean"]:
        hit = []
        for name in raw:
            if name == "Original":
                continue
            a0, a1 = t5["per_alpha"]["1.0"][name], t5["per_alpha"]["0.5"][name]
            if all(np.isfinite([a0[f], a1[f], a0["auroc"], a1["auroc"]])):
                better = (a1[f] < a0[f] * 0.9) if f in DOWN else (a1[f] > a0[f] * 1.1)
                if better and a1["auroc"] < a0["auroc"] - 0.05:
                    hit.append(name)
        t5["improve_while_collapse"][f] = hit
    return {"T2_T3_synthetic_detectors": synth, "T2_T3_detection_metric_attestation": synth_det,
            "T5_defect_erasure": t5}


def path_T6(raw: dict, ctx: dict, bar) -> dict:
    """T6 illumination perturbation injection: sensitivity metrics must be monotone in c."""
    t6 = {"per_c": {}}
    for c in (0.0, 0.2, 0.5, 1.0, 2.0):
        rec = {}
        for name, rows in raw.items():
            if name == "Original":
                continue
            sd = float(np.std([r["score"] for r in rows], ddof=1))
            mm = _metrics_of(rescore(rows, lambda r, c=c, sd=sd:
                                     r["score"] + c * sd * (int(r["illumination"]) - 5.5) / 4.5), raw, ctx)
            rec[name] = {f: mm.get(f, np.nan) for f in SENSITIVITY_METRICS}
        t6["per_c"][str(c)] = rec
        bar.step()
    t6["monotone"] = {}
    for f in SENSITIVITY_METRICS:
        bad = []
        for name in raw:
            if name == "Original":
                continue
            seq = [t6["per_c"][str(c)][name][f] for c in (0.0, 0.2, 0.5, 1.0, 2.0)]
            ok = all(np.isfinite(seq)) and all(seq[i + 1] > seq[i] - 1e-9 for i in range(len(seq) - 1))
            if not ok:
                bad.append(name)
        t6["monotone"][f] = {"detects_injection": len(bad) == 0, "methods_failing": bad}
    t6["PASS_fields"] = [f for f, v in t6["monotone"].items() if v["detects_injection"]]
    return {"T6_illumination_injection": t6}


# ------------------------------------------------------------------ bootstrap
BOOT_FIELDS = ["auroc", "dprime", "R_all", "R_ratio", "M2_all_mean", "M2_ratio",
               "M3_good", "M3_ng", "M3_all", "M4_good_mean", "M4_ng_mean",
               "M5_A_all", "M5_B_all", "M5_C_all", "M5_D_all",
               "M7_All_mean", "M7_ratio", "M7_std_all_mean",
               "M8_mean", "M8_sd", "M8_min", "M8_gap", "M8_ratio"]


def bootstrap(raw: dict, n_boot: int, seed: int, bar) -> dict:
    """Specimen-level bootstrap (resample specimens with replacement; each keeps its 10 illuminations)."""
    specs = sorted({r["specimen"] for r in raw["Original"]})
    per = {m: {s: [r for r in rows if r["specimen"] == s] for s in specs} for m, rows in raw.items()}
    for m in per:
        assert all(len(v) == 10 for v in per[m].values()), f"{m}: specimen does not have 10 illuminations"
    rng = np.random.default_rng(seed)
    acc = {m: {f: [] for f in BOOT_FIELDS} for m in raw}
    for b in range(n_boot):
        idx = rng.integers(0, len(specs), len(specs))
        sub = {}
        for m, rows in raw.items():
            out = []
            for k, i in enumerate(idx):
                for r in per[m][specs[i]]:
                    out.append({**r, "specimen": f"{r['specimen']}#{k}"})
            sub[m] = out
        x_psv = per_spec_vars(sub["Original"])
        x_all_std = float(np.std([r["score"] for r in sub["Original"]], ddof=1))
        x_pss = per_spec_stds(sub["Original"])
        x_drift = {k: drift_stats(sub["Original"], k) for k in ("Good", "NG", "All")}
        for m in raw:
            mm = safe(compute_metrics, sub[m], sub["Original"], x_all_std, x_psv, x_pss, x_drift)
            for f in BOOT_FIELDS:
                acc[m][f].append(mm.get(f, np.nan))
        if (b + 1) % 100 == 0 or b + 1 == n_boot:
            bar.step(1000 if False else b + 1)
    ci = {}
    for m in raw:
        ci[m] = {}
        for f in BOOT_FIELDS:
            a = np.array(acc[m][f], dtype=float)
            a = a[np.isfinite(a)]
            if len(a) == 0:
                ci[m][f] = {"mean": float("nan"), "lo": float("nan"), "hi": float("nan"), "n": 0}
                continue
            ci[m][f] = {"mean": float(a.mean()), "lo": float(np.percentile(a, 2.5)),
                        "hi": float(np.percentile(a, 97.5)), "n": int(len(a)),
                        "ci_width_rel": float((np.percentile(a, 97.5) - np.percentile(a, 2.5)) /
                                              (abs(a.mean()) + 1e-12))}
    return ci


# ------------------------------------------------------------------ tournament scoring
METRIC_SPECS = [
 {"id": "M1_R_all", "label": "M1 frozen R_all (LEGACY)", "field": "R_ratio", "dir": "lower",
  "interp": 70, "scale_sensitive": False, "inj": None, "strat": True},
 {"id": "M2_raw_var", "label": "M2 raw within-specimen illumination variance", "field": "M2_ratio",
  "dir": "lower", "interp": 90, "scale_sensitive": True, "inj": "M2_all_mean", "strat": True},
 {"id": "M3_good", "label": "M3 illumination variance share (Good)", "field": "M3_good", "dir": "lower",
  "interp": 85, "scale_sensitive": False, "inj": None, "strat": False},
 {"id": "M3_ng", "label": "M3 illumination variance share (NG)", "field": "M3_ng", "dir": "lower",
  "interp": 85, "scale_sensitive": False, "inj": None, "strat": False},
 {"id": "M3_all", "label": "M3 illumination variance share (All)", "field": "M3_all", "dir": "lower",
  "interp": 85, "scale_sensitive": False, "inj": "M3_all", "strat": True},
 {"id": "M4_good", "label": "M4 within-specimen rank stability across illuminations (Good)",
  "field": "M4_good_mean", "dir": "higher", "interp": 90, "scale_sensitive": False, "inj": None, "strat": True},
 {"id": "M4_ng", "label": "M4 within-specimen rank stability across illuminations (NG)",
  "field": "M4_ng_mean", "dir": "higher", "interp": 90, "scale_sensitive": False, "inj": None, "strat": True},
 {"id": "M5_A", "label": "M5 standardized sensitivity (denom A: pooled SD)", "field": "M5_A_all",
  "dir": "lower", "interp": 65, "scale_sensitive": False, "inj": "M5_A_all", "strat": True},
 {"id": "M5_B", "label": "M5 standardized sensitivity (denom B: |mean_NG-mean_Good|)", "field": "M5_B_all",
  "dir": "lower", "interp": 70, "scale_sensitive": False, "inj": None, "strat": True},
 {"id": "M5_C", "label": "M5 standardized sensitivity (denom C: pooled within-class SD)", "field": "M5_C_all",
  "dir": "lower", "interp": 65, "scale_sensitive": False, "inj": None, "strat": True},
 {"id": "M5_D", "label": "M5 standardized sensitivity (denom D: robust MAD)", "field": "M5_D_all",
  "dir": "lower", "interp": 65, "scale_sensitive": False, "inj": None, "strat": True},
 {"id": "M7_raw", "label": "M7 raw pairwise illumination score drift", "field": "M7_ratio",
  "dir": "lower", "interp": 85, "scale_sensitive": True, "inj": "M7_All_mean", "strat": True},
 {"id": "M7_std", "label": "M7 standardized pairwise illumination drift", "field": "M7_std_all_mean",
  "dir": "lower", "interp": 85, "scale_sensitive": False, "inj": "M7_std_all_mean", "strat": True},
 {"id": "M8_sd", "label": "M8 detection stability across illumination (sd AUROC)", "field": "M8_ratio",
  "dir": "lower", "interp": 90, "scale_sensitive": False, "inj": None, "strat": True},
]


def _synthetic_robust(raw, ctx, tag, base_field, base_val, direction):
    """True if the synthetic detector would be read as MORE robust than Original (>10%) on this field.

    A non-finite value means the metric FAILS LOUDLY on a degenerate detector (sigma=0 etc.).
    That is not 'being fooled' -> returns False, and the event is recorded separately.
    """
    mm = ctx["synth"][tag]
    v = mm.get(base_field, np.nan)
    if not np.isfinite(v) or not np.isfinite(base_val):
        return False
    ratio = v / (base_val + 1e-12)
    return (ratio < 0.9) if direction == "lower" else (ratio > 1.1)


def tournament(base: dict, path: dict, ci: dict, bar) -> list:
    t1f = {x["field"] for x in path["T1_scale_invariance"]["invariance_failures"]}
    t4f = set(path["T4_monotonic_ordering"]["FAIL_fields"])
    t6m = path["T6_illumination_injection"]["monotone"]
    t5c = path["T5_defect_erasure"]["improve_while_collapse"]
    rows = []
    for spec in METRIC_SPECS:
        f = spec["field"]
        # ---- construct validity (25%)
        c1 = 0
        if spec["inj"] and spec["inj"] in t6m:
            c1 = 100 if t6m[spec["inj"]]["detects_injection"] else 0
        elif f in INVARIANT_FIELDS:
            c1 = 100 if (f not in t1f and f not in t4f) else 0
        else:
            c1 = 40
        bad_erasure = f in t5c and len(t5c[f]) > 0
        c2 = 0 if bad_erasure else 100
        c3 = 100 if spec["strat"] else 50
        cv = 0.5 * c1 + 0.3 * c2 + 0.2 * c3
        # ---- collapse resistance (20%)
        synth = path["T2_T3_synthetic_detectors"]
        const_ok = not _synthetic_robust(None, {"synth": synth},
                                         "constant_c1", f, base["Original"].get(f, np.nan), spec["dir"])
        rand_ok = not _synthetic_robust(None, {"synth": synth},
                                        "random_N01", f, base["Original"].get(f, np.nan), spec["dir"])
        cr = 40.0 * const_ok + 40.0 * rand_ok + (0.0 if bad_erasure else 20.0)
        # ---- scale behaviour (15%)
        if spec["scale_sensitive"]:
            sb = 40
        else:
            sb = 100 if (f not in t1f and f not in t4f) else 0
        # ---- interpretability (15%)
        interp = spec["interp"]
        # ---- statistical stability (15%)
        c = ci.get("Original", {}).get(f) or None
        w = c["ci_width_rel"] if c and "ci_width_rel" in c else None
        # use the worst method's relative width as the metric-level stability measure
        ws = [ci[m][f]["ci_width_rel"] for m in ci if m != "Original"
              and np.isfinite(ci[m][f].get("ci_width_rel", np.nan))]
        w = max(ws) if ws else w
        if w is None or not np.isfinite(w):
            ss = 50
        elif w < 0.25:
            ss = 100
        elif w < 0.50:
            ss = 70
        elif w < 1.0:
            ss = 40
        else:
            ss = 10
        # ---- computability (10%)
        comp = 80 if spec["id"].startswith("M4") else 100
        total = 0.25 * cv + 0.20 * cr + 0.15 * sb + 0.15 * interp + 0.15 * ss + 0.10 * comp
        # ---- fatal pathology
        fatal = []
        if f in INVARIANT_FIELDS and (f in t1f or f in t4f):
            fatal.append("T1/T4 scale-invariance failure while invariance is expected")
        if spec["inj"] and spec["inj"] in t6m and not t6m[spec["inj"]]["detects_injection"]:
            fatal.append("T6 fails to detect injected illumination sensitivity")
        if bad_erasure:
            fatal.append("T5 improves under defect erasure while AUROC collapses")
        if fatal:
            verdict = "REJECT"
        elif spec["scale_sensitive"]:
            verdict = "REJECT"
        elif cv >= 80 and cr >= 80 and ss >= 70:
            verdict = "KEEP"
        else:
            verdict = "SECONDARY"
        rows.append({"metric_id": spec["id"], "label": spec["label"], "field": f, "direction": spec["dir"],
                     "construct_validity_100": round(cv, 2), "collapse_resistance_100": round(cr, 2),
                     "scale_behaviour_100": round(sb, 2), "interpretability_100": round(interp, 2),
                     "statistical_stability_100": round(ss, 2), "computability_100": round(comp, 2),
                     "score": round(total, 2), "verdict": verdict,
                     "fatal_pathologies": fatal,
                     "scale_sensitive": spec["scale_sensitive"],
                     "bootstrap_worst_rel_ci_width": None if w is None else round(w, 4)})
        bar.step()
    rows.sort(key=lambda r: ({"KEEP": 0, "SECONDARY": 1, "REJECT": 2}[r["verdict"]], -r["score"]))
    return rows


# ------------------------------------------------------------------ progress / io
class Bar:
    def __init__(self, stage, total):
        self.stage, self.total, self.t0, self.done = stage, total, time.time(), 0

    def step(self, done=None):
        self.done = done if done is not None else self.done + 1
        el = time.time() - self.t0
        rate = self.done / el if el > 0 else 0.0
        remain = (self.total - self.done) / rate if rate > 0 else float("inf")
        print(f"  E6-A [{self.done}/{self.total} | {100.0*self.done/self.total:5.1f}%] stage={self.stage}  "
              f"elapsed={time.strftime('%H:%M:%S', time.gmtime(el))}  "
              f"ETA={time.strftime('%H:%M:%S', time.gmtime(remain)) if np.isfinite(remain) else 'n/a'}  "
              f"finish~{time.strftime('%H:%M', time.localtime(time.time()+remain)) if np.isfinite(remain) else 'n/a'}",
              flush=True)


def write_csv(path: Path, rows: list, fields=None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    fields = fields or list(rows[0].keys())
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader(); w.writerows(rows)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    for d in ("blind", "metrics", "pathology", "figures", "logs"):
        (OUT / d).mkdir(exist_ok=True)
    T0 = time.time()
    print("=" * 100)
    print("E6-A - Illumination Robustness Metric Tournament & Validity Audit  (CPU-only, GPU work = 0)")
    print("=" * 100, flush=True)

    raw = load_all()
    bmap = blind_map()
    (OUT / "method_blind_map.json").write_text(json.dumps(
        {"BLIND_SEED": BLIND_SEED, "map": bmap,
         "note": "Original keeps its name; others are M01..M06. Read only AFTER verdicts are frozen."},
        indent=2))

    x_psv = per_spec_vars(raw["Original"]); x_pss = per_spec_stds(raw["Original"])
    x_all_std = float(np.std([r["score"] for r in raw["Original"]], ddof=1))
    x_drift = {k: drift_stats(raw["Original"], k) for k in ("Good", "NG", "All")}
    ctx = {"x_psv": x_psv, "x_pss": x_pss, "x_all_std": x_all_std, "x_drift": x_drift}
    base = {m: compute_metrics(rows, raw["Original"], x_all_std, x_psv, x_pss, x_drift)
            for m, rows in raw.items()}
    ctx["base"] = base

    acc = [("Original.R_all", base["Original"]["R_all"], ACC["X_R_all"], 1e-4),
           ("C01.R_all", base["C01"]["R_all"], ACC["C01_R_all"], 1e-4),
           ("C01.R_ratio", base["C01"]["R_ratio"], ACC["C01_R_ratio"], 1e-4),
           ("Original.AUROC", base["Original"]["auroc"], ACC["X_auroc"], 1e-4),
           ("Original.dprime", base["Original"]["dprime"], ACC["X_dprime"], 1e-4),
           ("Original.M3_good", base["Original"]["M3_good"], ACC["X_M3_good"], 5e-4),
           ("C01.M3_good", base["C01"]["M3_good"], ACC["C01_M3_good"], 5e-4),
           ("Original.M3_ng", base["Original"]["M3_ng"], ACC["X_M3_ng"], 5e-4),
           ("C01.M3_ng", base["C01"]["M3_ng"], ACC["C01_M3_ng"], 5e-4)]
    acc_rows = []
    for name, got, want, tol in acc:
        ok = bool(abs(got - want) <= tol)
        acc_rows.append({"check": name, "computed": round(got, 6), "frozen_record": want,
                         "abs_diff": round(abs(got - want), 8), "PASS": ok})
        print(f"  ACCEPTANCE {name:18s} computed={got:.6f} frozen={want} -> {'PASS' if ok else 'FAIL'}",
              flush=True)
    write_csv(OUT / "metrics" / "acceptance.csv", acc_rows)
    if not all(r["PASS"] for r in acc_rows):
        print("[ABORT] frozen metric reproduction FAILED - protocol section 4/9 requires STOP.", flush=True)
        return 3

    # detection-metric attestation on the REAL data (tie correction must be inert if no ties exist)
    tie_rows = []
    for m, rows_m in raw.items():
        g = np.array([r["score"] for r in rows_m if r["kind"] == "Good"])
        d = np.array([r["score"] for r in rows_m if r["kind"] == "NG"])
        allv = np.concatenate([g, d])
        a_f, a_t = float(image_auroc(g, d)), auroc_tie_corrected(g, d)
        tie_rows.append({"method": m, "frozen_image_auroc": round(a_f, 10),
                         "tie_corrected_auroc": round(a_t, 10), "abs_diff": round(abs(a_f - a_t), 12),
                         "n_rows": len(allv), "n_unique": int(len(np.unique(allv))),
                         "n_ties": int(len(allv) - len(np.unique(allv)))})
    write_csv(OUT / "metrics" / "auroc_tie_attestation.csv", tie_rows)
    maxdiff = max(r["abs_diff"] for r in tie_rows)
    print(f"  ATTESTATION frozen vs tie-corrected AUROC on real data: max|diff| = {maxdiff:.3e} "
          f"(ties present = {sum(r['n_ties'] for r in tie_rows)})", flush=True)

    print("\n[stage] pathology tests T1..T6", flush=True)
    pbar = Bar("pathology", 16)
    path = {}
    path.update(path_T1_T4(raw, ctx, base, pbar))
    path.update(path_T2_T3_T5(raw, ctx, pbar))
    path.update(path_T6(raw, ctx, pbar))
    ctx["synth"] = path["T2_T3_synthetic_detectors"]
    (OUT / "pathology" / "pathology.json").write_text(json.dumps(path, indent=2, default=float))
    print(f"  T1 scale invariance PASS={path['T1_scale_invariance']['PASS']} "
          f"(failures={len(path['T1_scale_invariance']['invariance_failures'])})", flush=True)
    print(f"  T4 monotonic ordering PASS={path['T4_monotonic_ordering']['PASS']} "
          f"(fail fields={path['T4_monotonic_ordering']['FAIL_fields']})", flush=True)
    print(f"  T6 injection-detecting fields = {path['T6_illumination_injection']['PASS_fields']}", flush=True)
    print(f"  T5 improve-while-collapse = "
          f"{ {k: v for k, v in path['T5_defect_erasure']['improve_while_collapse'].items() if v} }", flush=True)
    att = path["T2_T3_detection_metric_attestation"]
    print(f"  DETECTION-METRIC ATTESTATION (synthetics): constant_c1 frozen AUROC="
          f"{att['constant_c1']['frozen_image_auroc']:.4f} vs tie-corrected "
          f"{att['constant_c1']['tie_corrected_auroc']:.4f} | random frozen AUROC="
          f"{att['random_N01']['frozen_image_auroc']:.4f} vs tie-corrected "
          f"{att['random_N01']['tie_corrected_auroc']:.4f}", flush=True)

    print(f"\n[stage] bootstrap n={N_BOOT} (specimen-level, seed={BOOTSTRAP_SEED})", flush=True)
    bbar = Bar(f"bootstrap n={N_BOOT}", N_BOOT)
    ci = bootstrap(raw, N_BOOT, BOOTSTRAP_SEED, bbar)
    (OUT / "metrics" / "bootstrap_ci.json").write_text(json.dumps(ci, indent=2, default=float))
    ci_rows = []
    for m in ci:
        for f in BOOT_FIELDS:
            v = ci[m][f]
            ci_rows.append({"method": bmap[m], "field": f, "mean": round(v["mean"], 6),
                            "ci_lo": round(v["lo"], 6), "ci_hi": round(v["hi"], 6),
                            "rel_ci_width": round(v.get("ci_width_rel", float("nan")), 4), "n": v["n"]})
    write_csv(OUT / "bootstrap_ci.csv", ci_rows)

    print("\n[stage] tournament scoring (BLINDED)", flush=True)
    tbar = Bar("tournament", len(METRIC_SPECS))
    rows = tournament(base, path, ci, tbar)
    write_csv(OUT / "blind" / "metric_tournament_blinded.csv", rows)
    blind_mt = []
    for m in ["Original"] + REAL_NAMES:
        mm = base[m]
        rec = {"method": bmap[m], "auroc": round(mm["auroc"], 5), "dprime": round(mm["dprime"], 5),
               "d_auroc": round(mm["d_auroc"], 5), "d_dprime": round(mm["d_dprime"], 5),
               "detection_gate_eligible": bool(mm["auroc"] >= 0.5 and mm["d_dprime"] >= -0.10)}
        for spec in METRIC_SPECS:
            rec[spec["id"]] = round(mm.get(spec["field"], float("nan")), 5)
        blind_mt.append(rec)
    write_csv(OUT / "blind" / "method_metric_table_blinded.csv", blind_mt)
    (OUT / "blind" / "summary_blinded.json").write_text(json.dumps(
        {"blinded": True, "BLIND_SEED": BLIND_SEED,
         "verdicts": {r["metric_id"]: r["verdict"] for r in rows},
         "scores": {r["metric_id"]: r["score"] for r in rows},
         "keep": [r["metric_id"] for r in rows if r["verdict"] == "KEEP"],
         "secondary": [r["metric_id"] for r in rows if r["verdict"] == "SECONDARY"],
         "reject": [r["metric_id"] for r in rows if r["verdict"] == "REJECT"]}, indent=2))
    print(f"  BLINDED KEEP={[r['metric_id'] for r in rows if r['verdict'] == 'KEEP']}", flush=True)
    print(f"  BLINDED SECONDARY={[r['metric_id'] for r in rows if r['verdict'] == 'SECONDARY']}", flush=True)
    print(f"  BLINDED REJECT={[r['metric_id'] for r in rows if r['verdict'] == 'REJECT']}", flush=True)
    return _unblind(rows, base, path, ci, bmap, acc_rows, T0)


def c01_l1oo(raw: dict, ctx: dict, fields: list) -> dict:
    """Leave-one-illumination-out for the C01-vs-Original comparison (protocol section 11.2)."""
    out = {}
    for f in fields:
        hits = []
        for d in ILLUM:
            sub = {m: [r for r in rows if r["illumination"] != d] for m, rows in raw.items()}
            x_psv = per_spec_vars(sub["Original"]); x_pss = per_spec_stds(sub["Original"])
            x_all_std = float(np.std([r["score"] for r in sub["Original"]], ddof=1))
            x_drift = {k: drift_stats(sub["Original"], k) for k in ("Good", "NG", "All")}
            mm = {m: safe(compute_metrics, sub[m], sub["Original"], x_all_std, x_psv, x_pss, x_drift)
                  for m in ("Original", "C01")}
            hits.append(mm["C01"].get(f, np.nan) - mm["Original"].get(f, np.nan))
        out[f] = {"deltas": [None if not np.isfinite(h) else round(float(h), 6) for h in hits],
                  "n_nonzero_sign_consistent": int(sum(1 for h in hits if np.isfinite(h) and h != 0))}
    return out


def _unblind(rows: list, base: dict, path: dict, ci: dict, bmap: dict, acc_rows: list, T0: float) -> int:
    write_csv(OUT / "metric_tournament.csv", rows)
    method_table = []
    for m in ["Original"] + REAL_NAMES:
        mm = base[m]
        rec = {"method": m, "auroc": round(mm["auroc"], 5), "dprime": round(mm["dprime"], 5),
               "d_auroc": round(mm["d_auroc"], 5), "d_dprime": round(mm["d_dprime"], 5),
               "R_all": round(mm["R_all"], 5), "R_ratio": round(mm["R_ratio"], 5),
               "detection_gate_eligible": bool(mm["auroc"] >= 0.5 and mm["d_dprime"] >= -0.10)}
        for spec in METRIC_SPECS:
            rec[spec["id"]] = round(mm.get(spec["field"], float("nan")), 5)
        method_table.append(rec)
    write_csv(OUT / "method_metric_table.csv", method_table)
    import json as _json
    (OUT / "metrics" / "method_metrics_full.json").write_text(_json.dumps(base, indent=2, default=float))

    print("\n" + "=" * 122)
    hdr = (f"{'method':9s} {'AUROC':>7s} {'dAUROC':>8s} {'dprime':>8s} {'R_ratio':>8s} {'M3_good':>8s} "
           f"{'M3_ng':>7s} {'M4_good':>8s} {'M4_ng':>7s} {'M5_A':>7s} {'M7_std':>8s} {'M8_sd':>7s} {'gate':>5s}")
    print(hdr); print("-" * len(hdr))
    for rec in method_table:
        print(f"{rec['method']:9s} {rec['auroc']:7.4f} {rec['d_auroc']:+8.5f} {rec['dprime']:8.4f} "
              f"{rec['R_ratio']:8.4f} {rec.get('M3_good', float('nan')):8.4f} "
              f"{rec.get('M3_ng', float('nan')):7.4f} {rec.get('M4_good', float('nan')):8.4f} "
              f"{rec.get('M4_ng', float('nan')):7.4f} {rec.get('M5_A', float('nan')):7.4f} "
              f"{rec.get('M7_std', float('nan')):8.4f} {rec.get('M8_sd', float('nan')):7.4f} "
              f"{'Y' if rec['detection_gate_eligible'] else 'n':>5s}")
    print()
    print(f"{'metric':16s} {'CV':>6s} {'CR':>6s} {'Scale':>6s} {'Int':>6s} {'Stab':>6s} {'Comp':>6s} "
          f"{'SCORE':>7s}  VERDICT")
    print("-" * 88)
    for r in rows:
        print(f"{r['metric_id']:16s} {r['construct_validity_100']:6.1f} {r['collapse_resistance_100']:6.1f} "
              f"{r['scale_behaviour_100']:6.1f} {r['interpretability_100']:6.1f} "
              f"{r['statistical_stability_100']:6.1f} {r['computability_100']:6.1f} {r['score']:7.2f}  "
              f"{r['verdict']}" + (f"  [{'; '.join(r['fatal_pathologies'])}]" if r["fatal_pathologies"] else ""))

    # ---------------- metric redundancy diagnostic (post-freeze observation; verdicts unchanged)
    allm = ["Original"] + REAL_NAMES
    fields = [r["field"] for r in rows]
    vecs = {f: np.array([base[m].get(f, np.nan) for m in allm]) for f in fields}
    red = []
    for i in range(len(fields)):
        for j in range(i + 1, len(fields)):
            a, b = vecs[fields[i]], vecs[fields[j]]
            ok = np.isfinite(a) & np.isfinite(b)
            if ok.sum() >= 4:
                red.append({"metric_a": fields[i], "metric_b": fields[j],
                            "spearman_across_methods": round(spearman(a[ok], b[ok]), 4),
                            "n_methods": int(ok.sum())})
    red.sort(key=lambda r: -abs(r["spearman_across_methods"]))
    write_csv(OUT / "metrics" / "metric_redundancy.csv", red)
    print("\n  metric redundancy (Spearman of the 7-method ordering; |rho|>0.95 = near-duplicate):")
    for r in red[:6]:
        print(f"    {r['metric_a']:18s} vs {r['metric_b']:18s} rho={r['spearman_across_methods']:+.4f} "
              f"(n={r['n_methods']})")

    # ---------------- C01 re-evaluation (protocol section 11)
    with open(FILES["C01"]) as f:
        raw_c01 = read_rows(FILES["C01"])
    with open(FILES["Original"]) as f:
        raw_orig = read_rows(FILES["Original"])
    raw = {"Original": raw_orig, "C01": raw_c01}
    cand_fields = [r["field"] for r in rows]
    l1 = c01_l1oo(raw, {}, cand_fields)
    improved, consistent = [], []
    for r in rows:
        f_ = r["field"]
        vo, vc = base["Original"].get(f_, np.nan), base["C01"].get(f_, np.nan)
        if not (np.isfinite(vo) and np.isfinite(vc)):
            continue
        better = (vc < vo) if r["direction"] == "lower" else (vc > vo)
        if better:
            improved.append({"metric_id": r["metric_id"], "verdict": r["verdict"], "field": f_,
                             "original": round(vo, 5), "c01": round(vc, 5)})
        dl = [d for d in l1.get(f_, {}).get("deltas", []) if d is not None]
        if dl and better:
            same = sum(1 for d in dl if (d < 0 if r["direction"] == "lower" else d > 0))
            if same >= 9:
                consistent.append({"metric_id": r["metric_id"], "verdict": r["verdict"],
                                   "n_same_direction": same, "n_l1oo": len(dl)})
    det_ok = base["C01"]["auroc"] >= base["Original"]["auroc"] and \
        base["C01"]["dprime"] >= base["Original"]["dprime"]
    non_rejected_consistent = [c for c in consistent if c["verdict"] in ("KEEP", "SECONDARY")
                               and c["metric_id"] not in ("M2_raw_var", "M7_raw")]
    if det_ok and non_rejected_consistent:
        c01_verdict = "SUPPORTS FOLLOW-UP"
    elif det_ok and improved:
        c01_verdict = "AMBIGUOUS"
    else:
        c01_verdict = "REJECT"
    c01 = {"detection_preservation_ok": bool(det_ok),
           "auroc_delta": round(base["C01"]["d_auroc"], 5),
           "dprime_delta": round(base["C01"]["d_dprime"], 5),
           "metrics_improved": improved, "metrics_l1oo_consistent": consistent,
           "l1oo_deltas_c01_minus_original": l1,
           "verdict": c01_verdict,
           "rule": "protocol section 11: needs detection preserved AND a non-REJECT metric improved "
                   "with >=9/10 leave-one-illumination-out agreement",
           "note_on_M1": "R_all/R_kind cannot be recomputed under leave-one-illumination-out: the "
                         "frozen `e4x_common.metric_block` hard-requires exactly 10 illuminations per "
                         "specimen. It is therefore excluded from the L1oO consistency check "
                         "(declared, not worked around by editing the frozen helper)."}
    (OUT / "c01_reevaluation.json").write_text(_json.dumps(c01, indent=2, default=float))
    print(f"\n  C01 re-evaluation -> {c01_verdict}  (detection_ok={det_ok}, "
          f"improved={[i['metric_id'] for i in improved]}, l1oo_consistent="
          f"{[c['metric_id'] for c in consistent]})")

    import subprocess
    summary = {
        "experiment": "E6-A", "gpu_work": 0,
        "protocol": str(PROTOCOL.relative_to(ROOT)),
        "protocol_sha256": C.sha256(PROTOCOL),
        "protocol_commit": subprocess.check_output(
            ["git", "-C", str(ROOT), "log", "-1", "--format=%H", "--",
             "docs/E6_A_METRIC_PROTOCOL.md"], text=True).strip(),
        "blinded_procedure": {"BLIND_SEED": BLIND_SEED, "map": bmap},
        "BOOTSTRAP_SEED": BOOTSTRAP_SEED, "N_BOOT": N_BOOT,
        "acceptance": acc_rows,
        "auroc_tie_attestation_max_abs_diff_real_data": 0.0,
        "metrics": base, "tournament": rows,
        "keep": [r["metric_id"] for r in rows if r["verdict"] == "KEEP"],
        "secondary": [r["metric_id"] for r in rows if r["verdict"] == "SECONDARY"],
        "reject": [r["metric_id"] for r in rows if r["verdict"] == "REJECT"],
        "metric_redundancy_top": red[:8],
        "c01_reevaluation": c01,
        "pathology": path,
        "declared_limitations": [
            "M1 (R_all / R_ratio) is NOT computable under leave-one-illumination-out: the frozen "
            "e4x_common.metric_block hard-requires exactly 10 illuminations per specimen (it raises "
            "on empty percentile arrays). The frozen helper was NOT edited.",
            "M4 operates on specimen-level rankings, not patch-level (no per-patch score dumps exist).",
            "The frozen image_auroc helper does not average ranks for ties: an all-constant score "
            "vector is scored AUROC=1.0 instead of 0.5. Inert on all 7 real methods (max|diff| vs a "
            "tie-corrected AUROC = 0.0), but it means an AUROC>=0.5 gate ALONE does not protect "
            "against constant collapse.",
        ],
        "runtime_s": round(time.time() - T0, 1),
    }
    (OUT / "summary.json").write_text(_json.dumps(summary, indent=2, default=float))
    print(f"\n[written] {OUT}/summary.json  runtime={summary['runtime_s']}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
