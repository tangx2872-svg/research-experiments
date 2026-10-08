#!/usr/bin/env python
"""E4-X — shared constants, frozen definitions and metrics (KEEP OR KILL).

Protocol: `docs/E4_X_PROTOCOL.md` (frozen BEFORE any B2/X6c result was read).

Frozen constants here are HARD-CODED so that no target-driven threshold change is
possible at run time (protocol sanity S12).
"""
from __future__ import annotations

import hashlib
import random
import resource
from collections import defaultdict
from pathlib import Path

import numpy as np

from experiment1h_runner import image_auroc

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "e4_x"
E4D1 = ROOT / "results" / "e4_d1_smoke"

SEED = 0
BATCH = 16
VIEWS = ["000", "030", "060", "090", "120", "150", "180", "210", "240", "270", "300", "330"]
VIEW_SELECTION_SEED = 20261008
FROZEN_VIEW = random.Random(VIEW_SELECTION_SEED).choice(VIEWS)          # == "120"

BETA0, BMIN, BMAX = 0.25, 0.05, 0.50
Q_REF = 1.5726841688159168          # q_bottle, frozen in experiment_11 normal_statistics.csv
L3_ALPHA = 0.25                     # frozen Adaptive B2 layer-3 alpha
W_C2, W_C6 = 0.35, 0.65             # frozen C2 weights
FUSE_C2, FUSE_C6 = 0.5, 0.5         # frozen X6c = mean(z_C2, z_C6)

# ---- screening thresholds (docs/E4_X_PROTOCOL.md section 7.1) ----
EPS_DET = 0.03                      # AUROC band
DELTA_ROBUST = 0.05                 # GO Pattern A: R_all <= orig * (1 - delta)
EPS_ROBUST = 0.05                   # GO Pattern B: R_all <= orig * (1 + eps)
MEANINGFUL_GAIN = 0.03              # GO Pattern B: AUROC > orig + gain

ILLUMINATIONS = [f"{i:02d}" for i in range(1, 11)]


def ram_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def sha256(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def dprime(good: np.ndarray, defect: np.ndarray) -> float:
    if len(good) < 2 or len(defect) < 2:
        return float("nan")
    pooled = float(np.sqrt((defect.std(ddof=1) ** 2 + good.std(ddof=1) ** 2) / 2.0))
    return float((defect.mean() - good.mean()) / pooled) if pooled > 0 else float("nan")


def metric_block(rows: list[dict], tag: str) -> dict:
    """rows: [{specimen, kind, illumination, score}].

    R (primary)  = mean over (specimen x frozen view) of std_across_illuminations(z),
                   z = (s - mu_g)/sigma_g  with mu_g/sigma_g from THIS method's own
                   Good scores (normal-only calibration; protocol section 6.2).
    Rraw         = same but on raw scores (literal task wording; NOT primary).
    """
    g = np.array([r["score"] for r in rows if r["kind"] == "Good"])
    d = np.array([r["score"] for r in rows if r["kind"] == "NG"])
    mu, sd = float(g.mean()), float(g.std(ddof=1))

    grp: dict[str, list[float]] = defaultdict(list)
    kind_of = {}
    for r in rows:
        grp[r["specimen"]].append(r["score"])
        kind_of[r["specimen"]] = r["kind"]

    def disp(normalize: bool, kinds: set[str] | None = None) -> np.ndarray:
        vals = []
        for spec, v in grp.items():
            if len(v) != 10:
                continue
            if kinds and kind_of[spec] not in kinds:
                continue
            a = np.array(v)
            vals.append(((a - mu) / sd if normalize else a).std(ddof=1))
        return np.array(vals)

    out = {"method": tag, "n_good": len(g), "n_defect": len(d),
           "image_auroc": float(image_auroc(g, d)), "d_prime": dprime(g, d),
           "good_mean": mu, "good_std": sd,
           "ng_mean": float(d.mean()), "ng_std": float(d.std(ddof=1))}
    for kind in ("Good", "NG"):
        a = disp(True, {kind})
        out[f"R_{kind}"] = float(a.mean())
        out[f"R_{kind}_median"] = float(np.median(a))
        out[f"R_{kind}_p25"] = float(np.percentile(a, 25))
        out[f"R_{kind}_p75"] = float(np.percentile(a, 75))
        out[f"Rraw_{kind}"] = float(disp(False, {kind}).mean())
    allr = disp(True)
    out.update({"R_all": float(allr.mean()), "R_all_sd": float(allr.std(ddof=1)),
                "R_all_median": float(np.median(allr)),
                "R_all_p25": float(np.percentile(allr, 25)),
                "R_all_p75": float(np.percentile(allr, 75)),
                "Rraw_all": float(disp(False).mean())})

    per_ill = []
    for il in ILLUMINATIONS:
        gg = np.array([r["score"] for r in rows if r["kind"] == "Good" and r["illumination"] == il])
        dd = np.array([r["score"] for r in rows if r["kind"] == "NG" and r["illumination"] == il])
        per_ill.append({"illumination": il, "n_good": len(gg), "n_defect": len(dd),
                        "image_auroc": float(image_auroc(gg, dd)), "d_prime": dprime(gg, dd),
                        "good_mean": float(gg.mean()), "ng_mean": float(dd.mean())})
    out["per_illumination"] = per_ill
    return out


def decide(cand: dict, orig: dict) -> dict:
    """Mechanical GO / HOLD / STOP per docs/E4_X_PROTOCOL.md section 7."""
    d_auc = cand["image_auroc"] - orig["image_auroc"]
    r_ratio = cand["R_all"] / orig["R_all"]
    go_a = (d_auc >= -EPS_DET) and (r_ratio <= 1.0 - DELTA_ROBUST)
    go_b = (d_auc > MEANINGFUL_GAIN) and (r_ratio <= 1.0 + EPS_ROBUST)

    s1 = (d_auc <= EPS_DET) and (r_ratio > 1.0 - DELTA_ROBUST)          # both axes flat
    s2 = (d_auc > EPS_DET and r_ratio > 1.0 + EPS_ROBUST) or \
         (r_ratio <= 1.0 - DELTA_ROBUST and d_auc < -3 * EPS_DET)        # one up, other clearly worse
    s3 = (abs(d_auc) <= EPS_DET) and (abs(r_ratio - 1.0) <= DELTA_ROBUST)  # nearly identical

    lv = np.array([p["image_auroc"] for p in cand["per_illumination"]])
    lo = np.array([p["image_auroc"] for p in orig["per_illumination"]])
    best = int(np.argmax(lv))
    d_auc_no_best = float(np.mean(np.delete(lv, best)) - np.mean(np.delete(lo, best)))
    s6 = bool(d_auc > EPS_DET) and bool(d_auc_no_best <= EPS_DET)       # single-illumination driven

    decision = "GO" if (go_a or go_b) else ("STOP" if (s1 or s2 or s3 or s6) else "HOLD")
    return {"decision": decision, "d_auroc": d_auc, "R_ratio": r_ratio,
            "GO_pattern_A": bool(go_a), "GO_pattern_B": bool(go_b),
            "stop_s1_both_flat": bool(s1), "stop_s2_one_axis_harm": bool(s2),
            "stop_s3_nearly_identical": bool(s3), "stop_s6_single_illumination": bool(s6),
            "d_auroc_excluding_best_illumination": d_auc_no_best,
            "best_illumination": cand["per_illumination"][best]["illumination"]}
