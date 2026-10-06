"""Experiment 5D — Matched-Control Gate Validation: policy 冻结模块。

职责（全部在读取任何 5D target 分析结果之前完成）：
  1. 机械复现 5C 冻结的 Geometry Conservative gate（primary method）；不一致即 abort
  2. 生成 Matched Sensitivity Conservative（同一 gate size / alpha set / mean alpha / mechanical rule）：
       SC-A（PRIMARY）  ：沿用 5C 冻结的 rank->alpha 方向（predictor 低值 = 最脆弱）
       SC-B（SECONDARY）: 按 5B registry 冻结的 direction_hypothesis = "high->fragile"
     —— 两者都是 normal-only predictor，唯一区别是 direction 解释；5D 对两者对称全量报告。
  3. 枚举全部 C(5,2) = 10 个 matched gates（EXHAUSTIVE CONTROLS ARE EVALUATION-ONLY）
  4. 写 reference/{policy_freeze.json, geometry_assignment.csv, sensitivity_assignment.csv,
                 exhaustive_gate_assignments.csv, asset_hashes.json} 并打印 SHA256

本模块只读 5B-Final（normal-only predictors）与 5C 冻结 policy 模块，
不读任何 target 指标（不做任何 policy 级 d' / robustness 聚合）。
"""

from __future__ import annotations

import csv
import hashlib
import itertools
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import experiment5c_policy as pol5c  # noqa: E402

OUT = ROOT / "results" / "experiment_5d"
REF_DIR = OUT / "reference"
SUM_DIR = OUT / "summary"
SAN_DIR = OUT / "sanity"
FIG_DIR = OUT / "figures"
LOG_DIR = OUT / "logs"

CATEGORIES = ["bottle", "cable", "grid", "hazelnut", "screw"]
SEEDS = [0, 1, 2]

PRED_PRIMARY = "radius_ratio_L3L2"
PRED_CONTROL = "sens_radius_L2"
PRED_SECONDARY = "eff_dim_L3"
ALPHA_SET = [0.0, 0.5]
GATE_SIZE = 2
MEAN_ALPHA = 0.3
EPS = 0.10

GC_5C_FROZEN = {"bottle": 0.0, "grid": 0.0, "cable": 0.5, "hazelnut": 0.5, "screw": 0.5}
BEST_FIXED_ALPHA_5C = 0.5
HIGH_DAMAGE_SUBSET = ["bottle", "grid"]

TRACKED_ASSETS = [
    ROOT / "results" / "experiment_5b_final" / "summary" / "normal_only_predictors.csv",
    ROOT / "results" / "experiment_5b_final" / "reference" / "predictor_registry.csv",
    ROOT / "results" / "experiment_5b_final" / "reference" / "group_c_freeze.json",
    ROOT / "results" / "experiment_5b_final" / "summary" / "correlation_summary.csv",
    ROOT / "results" / "experiment_5b_final" / "summary" / "final_summary.json",
    ROOT / "results" / "experiment_5c" / "reference" / "geometry_policy_freeze.json",
    ROOT / "results" / "experiment_5c" / "summary" / "policy_assignment.csv",
    ROOT / "results" / "experiment_5a_h" / "summary" / "per_unit.csv",
    ROOT / "results" / "experiment_5a_h" / "summary" / "statistical_tests.csv",
]

ORIENTATION_DOC = {
    "low_predictor_is_fragile": (
        "predictor low end = most fragile -> ascending rank 1..2 maps to alpha=0. This is the mechanical "
        "application of the 5C frozen direction (higher_geometry_to_higher_alpha) plus 5C README section 4 "
        "(GF/GC/SG share one rank->alpha function) restricted to the legal set [0, 0.5]. It is also "
        "consistent with the 5B README section 17.4 measured sign for sens_radius_L2 "
        "(rho(C2) = -0.30, same sign as the geometry family)."
    ),
    "high_predictor_is_fragile": (
        "predictor high end = most fragile -> descending rank 1..2 maps to alpha=0. Literal implementation "
        "of the 5B registry frozen direction_hypothesis = 'high->fragile' (predictor_registry.csv / "
        "group_c_freeze.json, frozen_at 2026-10-06T17:07:27) for sens_radius_L2. It contradicts the sign "
        "measured by 5B itself (rho(C2) = -0.30), so in 5D it is only a secondary ambiguity-resolving control."
    ),
}


def md5_file(p: Path) -> str:
    return hashlib.md5(p.read_bytes()).hexdigest()


def sha256_file(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def git_head() -> str:
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                          capture_output=True, text=True).stdout.strip()


def predictor_values(name: str, preds: dict) -> dict:
    """{category: seed-mean value}；断言 3 seeds 排序一致，否则不得机械应用 rank 规则。"""
    vals = {c: float(np.mean([float(preds[(c, s)][name]) for s in SEEDS])) for c in CATEGORIES}
    orders = []
    for s in SEEDS:
        per_seed = {c: float(preds[(c, s)][name]) for c in CATEGORIES}
        orders.append(tuple(sorted(CATEGORIES, key=lambda c: per_seed[c])))
    if len(set(orders)) != 1:
        raise SystemExit(f"[FATAL] {name} per-seed rank inconsistent: {orders}")
    return vals


def fragile_ranks(values: dict, orientation: str) -> dict:
    """rank 1 = 最脆弱。orientation 决定哪一端是脆弱端。"""
    if orientation == "low_predictor_is_fragile":
        order = sorted(CATEGORIES, key=lambda c: values[c])
    elif orientation == "high_predictor_is_fragile":
        order = sorted(CATEGORIES, key=lambda c: -values[c])
    else:
        raise ValueError(orientation)
    return {c: i + 1 for i, c in enumerate(order)}


def alpha_map_from_ranks(ranks: dict, legal: list) -> dict:
    """alpha(r) = legal[min(K-1, ceil(r*K/n)-1)]（与 5C 冻结 mapping 函数逐字一致）。"""
    return {c: pol5c.alpha_for_rank(ranks[c], legal, len(CATEGORIES)) for c in CATEGORIES}


def subset_policy(gated: tuple, gate_alpha: float = 0.0, keep: float = 0.5) -> dict:
    return {c: (gate_alpha if c in gated else keep) for c in CATEGORIES}


def write_csv(path: Path, rows: list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def main() -> None:
    for d in (REF_DIR, SUM_DIR, SAN_DIR, FIG_DIR, LOG_DIR):
        d.mkdir(parents=True, exist_ok=True)

    preds = pol5c.load_predictors()
    if len(preds) != 15:
        raise SystemExit(f"[FATAL] predictor table must have 15 rows, got {len(preds)}")

    geom_vals = predictor_values(PRED_PRIMARY, preds)
    ctrl_vals = predictor_values(PRED_CONTROL, preds)
    sec_vals = predictor_values(PRED_SECONDARY, preds)

    geom_ranks = fragile_ranks(geom_vals, "low_predictor_is_fragile")
    gc = alpha_map_from_ranks(geom_ranks, ALPHA_SET)

    sc_a_ranks = fragile_ranks(ctrl_vals, "low_predictor_is_fragile")
    sc_a = alpha_map_from_ranks(sc_a_ranks, ALPHA_SET)

    sc_b_ranks = fragile_ranks(ctrl_vals, "high_predictor_is_fragile")
    sc_b = alpha_map_from_ranks(sc_b_ranks, ALPHA_SET)

    sec_ranks = fragile_ranks(sec_vals, "low_predictor_is_fragile")
    sec = alpha_map_from_ranks(sec_ranks, ALPHA_SET)

    if gc != GC_5C_FROZEN:
        raise SystemExit(f"[FATAL] GC assignment != 5C frozen: {gc} != {GC_5C_FROZEN}")
    for name, amap in (("GC", gc), ("SC_A", sc_a), ("SC_B", sc_b), ("SEC", sec)):
        gated = [c for c in CATEGORIES if amap[c] == 0.0]
        assert len(gated) == GATE_SIZE, f"{name}: gate size {len(gated)}"
        assert all(a in ALPHA_SET for a in amap.values()), f"{name}: alpha outside {ALPHA_SET}"
        assert abs(float(np.mean(list(amap.values()))) - MEAN_ALPHA) < 1e-12, f"{name}: mean alpha"

    gate_subsets = [tuple(sorted(p)) for p in itertools.combinations(CATEGORIES, 2)]
    assert len(gate_subsets) == 10, len(gate_subsets)
    gate_policies = {f"GATE:{'+'.join(g)}": subset_policy(g) for g in gate_subsets}
    for g in gate_subsets:
        a = subset_policy(g)
        assert abs(float(np.mean(list(a.values()))) - MEAN_ALPHA) < 1e-12
        assert sorted([c for c in CATEGORIES if a[c] == 0.0]) == list(g)

    save_assignments(preds, geom_vals, ctrl_vals, sec_vals, geom_ranks, sc_a_ranks, sc_b_ranks,
                     gc, sc_a, sc_b, sec, gate_subsets, gate_policies)
    assets = track_assets()
    freeze = build_freeze(geom_vals, ctrl_vals, sec_vals, geom_ranks, sc_a_ranks, sc_b_ranks,
                          gc, sc_a, sc_b, sec, gate_subsets)
    fp = REF_DIR / "policy_freeze.json"
    fp.write_text(json.dumps(freeze, indent=2))
    digest = sha256_file(fp)
    (REF_DIR / "policy_freeze.sha256").write_text(digest + "  policy_freeze.json\n")
    print_pre_run_plan(freeze, gc, sc_a, sc_b, sec, gate_subsets, digest, len(assets), fp)


def assign_rows(pname, pred_name, vals, ranks, amap, orientation):
    return [{
        "policy": pname, "predictor": pred_name, "orientation": orientation,
        "category": c,
        "predictor_value_seed_mean": f"{vals[c]:.6f}",
        "fragile_rank": ranks[c],
        "alpha": f"{amap[c]:.1f}",
        "role": "gate(alpha=0)" if amap[c] == 0.0 else "keep(alpha=0.5)",
    } for c in CATEGORIES]


def save_assignments(preds, geom_vals, ctrl_vals, sec_vals, geom_ranks, sc_a_ranks, sc_b_ranks,
                     gc, sc_a, sc_b, sec, gate_subsets, gate_policies):
    write_csv(REF_DIR / "geometry_assignment.csv",
              assign_rows("GC", PRED_PRIMARY, geom_vals, geom_ranks, gc, "low_predictor_is_fragile"))
    write_csv(REF_DIR / "sensitivity_assignment.csv",
              assign_rows("SC_A", PRED_CONTROL, ctrl_vals, sc_a_ranks, sc_a, "low_predictor_is_fragile")
              + assign_rows("SC_B", PRED_CONTROL, ctrl_vals, sc_b_ranks, sc_b, "high_predictor_is_fragile"))
    gc_gate = sorted([c for c in CATEGORIES if gc[c] == 0.0])
    sca_gate = sorted([c for c in CATEGORIES if sc_a[c] == 0.0])
    scb_gate = sorted([c for c in CATEGORIES if sc_b[c] == 0.0])
    rows = []
    for key, amap in gate_policies.items():
        gated = [c for c in CATEGORIES if amap[c] == 0.0]
        rows.append({
            "gate_id": key,
            "gated_categories": "+".join(gated),
            "kept_categories": "+".join([c for c in CATEGORIES if amap[c] == 0.5]),
            **{f"alpha_{c}": f"{amap[c]:.1f}" for c in CATEGORIES},
            "gate_size": GATE_SIZE,
            "mean_alpha": f"{float(np.mean(list(amap.values()))):.4f}",
            "is_geometry_gate": "true" if sorted(gated) == gc_gate else "false",
            "is_sc_a_gate": "true" if sorted(gated) == sca_gate else "false",
            "is_sc_b_gate": "true" if sorted(gated) == scb_gate else "false",
            "usage": "EVALUATION-ONLY (no best-random-gate selection allowed)",
        })
    write_csv(REF_DIR / "exhaustive_gate_assignments.csv", rows)


def track_assets() -> dict:
    assets = {}
    for p in TRACKED_ASSETS:
        if not p.exists():
            raise SystemExit(f"[FATAL] tracked asset missing: {p}")
        assets[str(p.relative_to(ROOT))] = {"md5": md5_file(p), "sha256": sha256_file(p),
                                            "bytes": p.stat().st_size}
    (REF_DIR / "asset_hashes.json").write_text(json.dumps(assets, indent=2))
    return assets


def build_freeze(geom_vals, ctrl_vals, sec_vals, geom_ranks, sc_a_ranks, sc_b_ranks,
                 gc, sc_a, sc_b, sec, gate_subsets) -> dict:
    return {
        "experiment": "5D",
        "title": "Matched-Control Gate Validation of Geometry-Guided Safe Normalization",
        "purpose": ("under identical normalization budget / gate size / alpha set, test whether "
                    "normal-feature geometry selects the categories whose normalization should be suppressed "
                    "more reliably than a matched sensitivity predictor or the exhaustive matched-gate "
                    "distribution"),
        "paper_claim": ("Normal-feature geometry contains actionable information for deciding when "
                        "normalization should be suppressed to preserve defect-sensitive representations."),
        "frozen_predictors": {
            "primary": PRED_PRIMARY,
            "matched_control": PRED_CONTROL,
            "secondary_identity_check_only_no_eval": PRED_SECONDARY,
        },
        "primary_comparison": "GC vs SC_A (primary); GC vs SC_B (secondary, ambiguity-resolving)",
        "predictor_source_file": "results/experiment_5b_final/summary/normal_only_predictors.csv",
        "predictor_source_is_normal_only": True,
        "direction": {
            "geometry_GC": "low_predictor_is_fragile",
            "sensitivity_SC_A_primary": "low_predictor_is_fragile",
            "sensitivity_SC_B_secondary": "high_predictor_is_fragile",
            "SC_A_rationale": ORIENTATION_DOC["low_predictor_is_fragile"],
            "SC_B_rationale": ORIENTATION_DOC["high_predictor_is_fragile"],
            "protocol_ambiguity_declared": True,
            "protocol_ambiguity_note": (
                "the conservative direction of the sensitivity predictor is not unique in the historical "
                "record: the 5B registry pre-registered direction_hypothesis='high->fragile', while the 5B "
                "measured rho(C2)=-0.30 (negative) and the 5C frozen rank->alpha mechanical rule both imply "
                "low->fragile. Human decision 2026-10-06: pre-register BOTH; SC_A is the primary comparison "
                "and SC_B is reported in full as a symmetric ambiguity-resolving control. The direction was "
                "NOT chosen by looking at any target result."),
        },
        "mapping_type": "rank_monotonic_nearest_rank_quantile",
        "mapping_formula": ("alpha(r) = legal[min(K-1, ceil(r*K/n)-1)], r = fragile-rank "
                            "(1 = most fragile), n = 5"),
        "alpha_set": ALPHA_SET,
        "gate_size": f"{GATE_SIZE}/5",
        "mean_alpha_all_matched_policies": MEAN_ALPHA,
        "epsilon": EPS,
        "epsilon_source": "Experiment 5A-H section 2 (frozen d-prime non-inferiority band), unchanged",
        "best_fixed_alpha": BEST_FIXED_ALPHA_5C,
        "best_fixed_source": "5C frozen best fixed (5A-H B2), NOT re-selected",
        "high_damage_subset_frozen": HIGH_DAMAGE_SUBSET,
        "roles": {
            "A_original": "reference (all alpha=0, = 5A-H B0)",
            "B_best_fixed": "frozen baseline (all alpha=0.5, = 5A-H B2)",
            "GC": "PRIMARY METHOD (geometry conservative)",
            "SC_A": "PRIMARY MATCHED CONTROL (sensitivity, 5C mechanical orientation)",
            "SC_B": "SECONDARY MATCHED CONTROL (sensitivity, 5B registry hypothesis orientation)",
        },
        "category_assignments": {"GC": gc, "SC_A": sc_a, "SC_B": sc_b,
                                 "SEC_eff_dim_L3_identity_check_no_eval": sec},
        "category_fragile_ranks": {"GC": geom_ranks, "SC_A": sc_a_ranks, "SC_B": sc_b_ranks},
        "predictor_category_values_seed_mean": {PRED_PRIMARY: geom_vals, PRED_CONTROL: ctrl_vals,
                                                PRED_SECONDARY: sec_vals},
        "exhaustive_gates": [f"GATE:{'+'.join(g)}" for g in gate_subsets],
        "exhaustive_gates_rule": ("EXHAUSTIVE CONTROLS ARE EVALUATION-ONLY. "
                                  "NO BEST RANDOM GATE MAY REPLACE THE FROZEN GEOMETRY POLICY."),
        "categories": CATEGORIES,
        "seeds": SEEDS,
        "n_category": len(CATEGORIES),
        "n_seed": len(SEEDS),
        "illumination_protocol": "4 photometric shifts (brightness 0.7/1.3, gamma 0.7/1.3), 5A-H frozen",
        "target_results_read": False,
        "no_gpu": True,
        "data_source": "CPU-only recombination of frozen 5A-H raw per-image scores (B0/B2)",
        "git_head": git_head(),
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }


def print_pre_run_plan(freeze, gc, sc_a, sc_b, sec, gate_subsets, digest, n_assets, fp):
    L = 100
    print("=" * L)
    print("[5D PRE-RUN PLAN] Matched-Control Gate Validation - Geometry-Guided Safe Normalization")
    print("=" * L)
    print(f"git HEAD                       : {freeze['git_head']}")
    print(f"primary predictor (frozen)     : {PRED_PRIMARY}")
    print(f"matched control predictor      : {PRED_CONTROL}")
    print(f"alpha set / gate size / mean a : {ALPHA_SET} / {GATE_SIZE} of 5 / {MEAN_ALPHA}")
    print(f"frozen best fixed baseline     : alpha = {BEST_FIXED_ALPHA_5C} (5C, reused)")
    print(f"epsilon (preservation band)    : {EPS}")
    print(f"target_results_read            : {freeze['target_results_read']}")
    print("-" * L)
    hdr = f"{'policy':<8}{'orientation':<30}" + "".join(f"{c[:6]:>9}" for c in CATEGORIES) + "   gate"
    print(hdr)
    for pname, ori, amap in [("GC", "low->fragile (5C frozen)", gc),
                             ("SC_A", "low->fragile (5C mech.)", sc_a),
                             ("SC_B", "high->fragile (5B reg.)", sc_b),
                             ("SEC", "low->fragile (id-check)", sec)]:
        gated = "+".join([c for c in CATEGORIES if amap[c] == 0.0])
        print(f"{pname:<8}{ori:<30}" + "".join(f"{amap[c]:9.1f}" for c in CATEGORIES) + f"   {gated}")
    print("-" * L)
    print("exhaustive matched gates (C(5,2)=10, EVALUATION-ONLY):")
    gc_gate = sorted([c for c in CATEGORIES if gc[c] == 0.0])
    sca_gate = sorted([c for c in CATEGORIES if sc_a[c] == 0.0])
    scb_gate = sorted([c for c in CATEGORIES if sc_b[c] == 0.0])
    for g in gate_subsets:
        tag = []
        if sorted(g) == gc_gate:
            tag.append("= GC")
        if sorted(g) == sca_gate:
            tag.append("= SC_A")
        if sorted(g) == scb_gate:
            tag.append("= SC_B")
        print(f"    GATE:{'+'.join(g):<28}" + (("<- " + ", ".join(tag)) if tag else ""))
    print("-" * L)
    print(f"GC == 5C frozen assignment     : {gc == GC_5C_FROZEN}  (hard assertion passed)")
    print(f"freeze JSON                    : {fp.relative_to(ROOT)}")
    print(f"freeze SHA256                  : {digest}")
    print(f"asset hashes                   : {n_assets} files -> reference/asset_hashes.json")
    print("=" * L)
    print("EXHAUSTIVE CONTROLS ARE EVALUATION-ONLY.")
    print("NO BEST RANDOM GATE MAY REPLACE THE FROZEN GEOMETRY POLICY.")
    print("=" * L)


if __name__ == "__main__":
    main()
