#!/usr/bin/env python
"""E5-1 — frozen metrics + advance decision (E5-1 specific).

Metric computation is REUSED verbatim from `e4x_common.metric_block` (AUROC, d', R_all, per-illumination).
This module ONLY adds the E5-1 advance rule frozen in `docs/E5_1_FROZEN_PROTOCOL.md` §7.

Thresholds are HARD-CODED here so that no target-driven threshold change is possible at run time.
"""
from __future__ import annotations

import numpy as np

# ---- frozen thresholds (docs/E5_1_FROZEN_PROTOCOL.md section 7) ----
DELTA_AUROC_WIN = 0.03      # GO-A detection win
RATIO_ROBUST_WIN = 0.90     # GO-B robustness win
DD_CLEAR_DEGRADE = -0.10    # "clear defect-preservation degradation" / GO-B guard
DD_SLIGHT_LOSS = -0.02      # "slight preservation loss"
HOLD_BAND_LO = 0.01         # +0.01 <= dA < +0.03  => HOLD
RATIO_NO_GAIN = 0.95        # rho > 0.95 => "no robustness gain"
EPS_INDISTINCT = 0.01       # |dA| < 0.01
EPS_RHO_IDENT = 0.05        # |rho-1| < 0.05
EPS_DD_IDENT = 0.05         # |dd|   < 0.05


def decide_e5_1(cand: dict, orig: dict) -> dict:
    """Mechanical GO / HOLD / STOP per the E5-1 frozen protocol.

    dA  = AUROC_cand - AUROC_orig
    dd  = d'_cand  - d'_orig
    rho = R_all_cand / R_all_orig          (smaller is better)
    """
    d_auc = float(cand["image_auroc"] - orig["image_auroc"])
    d_dp = float(cand["d_prime"] - orig["d_prime"])
    rho = float(cand["R_all"] / orig["R_all"])

    go_a = (d_auc >= DELTA_AUROC_WIN) and (d_dp >= DD_CLEAR_DEGRADE) and (rho <= 1.00)
    go_b = (rho <= RATIO_ROBUST_WIN) and (d_dp >= DD_CLEAR_DEGRADE)

    s1_flat = (d_auc < HOLD_BAND_LO) and (rho > RATIO_NO_GAIN)
    s2_harm = ((d_auc >= DELTA_AUROC_WIN) and (rho > 1.00)) or \
              ((rho <= RATIO_ROBUST_WIN) and (d_dp < DD_CLEAR_DEGRADE)) or \
              ((d_auc >= DELTA_AUROC_WIN) and (d_dp < DD_CLEAR_DEGRADE))
    s3_ident = (abs(d_auc) < EPS_INDISTINCT) and (abs(rho - 1.0) < EPS_RHO_IDENT) and \
               (abs(d_dp) < EPS_DD_IDENT)

    lv = np.array([p["image_auroc"] for p in cand["per_illumination"]], dtype=float)
    lo = np.array([p["image_auroc"] for p in orig["per_illumination"]], dtype=float)
    best = int(np.argmax(lv))
    d_auc_no_best = float(np.mean(np.delete(lv, best)) - np.mean(np.delete(lo, best)))
    s6_single = bool(d_auc >= DELTA_AUROC_WIN) and bool(d_auc_no_best <= HOLD_BAND_LO)

    if go_a or go_b:
        decision = "GO"
    elif s1_flat or s2_harm or s3_ident or s6_single:
        decision = "STOP"
    else:
        decision = "HOLD"

    return {
        "decision": decision,
        "d_auroc": d_auc, "d_dprime": d_dp, "R_ratio": rho,
        "GO_pattern_A_detection": bool(go_a), "GO_pattern_B_robustness": bool(go_b),
        "stop_s1_both_flat": bool(s1_flat),
        "stop_s2_one_axis_harm": bool(s2_harm),
        "stop_s3_indistinguishable": bool(s3_ident),
        "stop_s6_single_illumination": bool(s6_single),
        "d_auroc_excluding_best_illumination": d_auc_no_best,
        "best_illumination": cand["per_illumination"][best]["illumination"],
        "meaningful_detection_gain": bool(d_auc >= HOLD_BAND_LO),   # +0.001/+0.003 are never "gains"
    }


def leave_one_illumination_out(per_image_cand: list[dict], per_image_orig: list[dict],
                               illuminations: list[str], auroc_fn) -> list[dict]:
    """§15 / §8: recompute dAUROC with each illumination removed in turn (score-level only).

    `per_image_*` rows: {specimen, kind, illumination, score}. No re-fitting.
    """
    out = []
    for drop in illuminations:
        def _auc(rows):
            g = np.array([r["score"] for r in rows if r["kind"] == "Good"], dtype=float)
            d = np.array([r["score"] for r in rows if r["kind"] == "NG"], dtype=float)
            return float(auroc_fn(g, d))

        c = _auc([r for r in per_image_cand if r["illumination"] != drop])
        o = _auc([r for r in per_image_orig if r["illumination"] != drop])
        out.append({"dropped_illumination": drop, "auroc_cand": c, "auroc_orig": o,
                    "d_auroc": c - o,
                    "advantage_lost": bool((c - o) < HOLD_BAND_LO)})
    return out
