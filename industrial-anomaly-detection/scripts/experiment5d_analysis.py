"""Experiment 5D — Matched-Control Gate Validation: analysis (CPU-only recombination).

数据来源（零 GPU、零新 inference）：
  A. Original (alpha=0) 与 B. Best Fixed (alpha=0.5)：results/experiment_5a_h/raw 的 B0 / B2
  （5A-H 已在 5 cat x 3 seed 上完整覆盖 alpha in {0, 0.5}，30/30 condition key）

指标计算完全复用 experiment5a_h_analysis.load_all / unit_metrics（与 5A-H / 5C 同一口径，
不新造主指标）。policy assignment 只来自 results/experiment_5d/reference/policy_freeze.json
（冻结于任何 target 分析之前）。

用法：python -u scripts/experiment5d_analysis.py
"""

from __future__ import annotations

import csv
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import experiment5a_h_analysis as h5  # noqa: E402
import experiment5d_policy as pol5d  # noqa: E402

OUT = ROOT / "results" / "experiment_5d"
REF_DIR, SUM_DIR, SAN_DIR, FIG_DIR, LOG_DIR = (OUT / "reference", OUT / "summary",
                                               OUT / "sanity", OUT / "figures", OUT / "logs")

CATEGORIES = pol5d.CATEGORIES
SEEDS = pol5d.SEEDS
EPS = pol5d.EPS
HD_SUBSET = pol5d.HIGH_DAMAGE_SUBSET

RAW_5AH = ROOT / "results" / "experiment_5a_h" / "raw"
RAW_5C = ROOT / "results" / "experiment_5c" / "raw"

METRIC_KEYS = ["mean_dprime", "mean_abs_delta_z", "image_auroc", "pixel_auroc", "aupro",
               "fpr_clean", "fpr_shift", "clipped_pixel_ratio", "tau_val"]

POLICY_LABEL = {
    "A_original": "A. Original (alpha=0)",
    "B_best_fixed": "B. Best Fixed (alpha=0.5)",
    "GC": "Geometry Conservative (PRIMARY METHOD)",
    "SC_A": "Sensitivity Conservative SC-A (PRIMARY CONTROL)",
    "SC_B": "Sensitivity Conservative SC-B (secondary control)",
}


def akey(a: float) -> str:
    return f"{float(a):.9f}"


def md5_file(p: Path) -> str:
    return hashlib.md5(p.read_bytes()).hexdigest()


def git_head() -> str:
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                          capture_output=True, text=True).stdout.strip()


def git_branch() -> str:
    return subprocess.run(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=ROOT,
                          capture_output=True, text=True).stdout.strip()


def git_is_clean() -> bool:
    r = subprocess.run(["git", "status", "--porcelain"], cwd=ROOT,
                       capture_output=True, text=True).stdout.strip()
    return r == ""


# ---------------------------------------------------------------------------
# 1. 历史 raw score -> (cat, seed, alpha) metrics lookup
# ---------------------------------------------------------------------------
def build_lookup() -> dict:
    h5.RAW_DIR = RAW_5AH
    h5.CONFIGS = ["B0", "B2"]
    m = h5.unit_metrics(h5.load_all())
    lookup = {}
    for (cat, seed, cfg), mm in m.items():
        lookup[(cat, seed, akey(mm["alpha_l2"]))] = {k: float(mm[k]) for k in METRIC_KEYS} | {
            "alpha_l2": float(mm["alpha_l2"]), "source": f"5A-H:{cfg}"}
    need = {(c, s, a) for c in CATEGORIES for s in SEEDS for a in (0.0, 0.5)}
    missing = {k for k in need if (k[0], k[1], akey(k[2])) not in lookup}
    if missing:
        raise SystemExit(f"[FATAL] 5A-H raw 缺少 condition key: {sorted(missing)}")
    if len(lookup) != 30:
        raise SystemExit(f"[FATAL] 5D 需要的 key 应为 30，实得 {len(lookup)}")
    return lookup


def get_metrics(lookup: dict, cat: str, seed: int, alpha: float) -> dict:
    key = (cat, seed, akey(alpha))
    if key not in lookup:
        raise KeyError(f"alpha {alpha} not measured for {cat}:{seed}")
    return lookup[key]


def eval_policy(lookup: dict, amap: dict) -> dict:
    """返回 per-unit / per-category / aggregate。amap: {category: alpha}。"""
    units = []
    for c in CATEGORIES:
        for s in SEEDS:
            mm = get_metrics(lookup, c, s, amap[c])
            units.append({"category": c, "seed": s, "alpha": float(amap[c]),
                          **{k: mm[k] for k in METRIC_KEYS}, "source": mm["source"]})
    dp = np.array([u["mean_dprime"] for u in units])
    rz = np.array([u["mean_abs_delta_z"] for u in units])
    per_cat = {}
    for c in CATEGORIES:
        us = [u for u in units if u["category"] == c]
        per_cat[c] = {"mean_dprime": float(np.mean([u["mean_dprime"] for u in us])),
                      "std_dprime": float(np.std([u["mean_dprime"] for u in us], ddof=1)),
                      "mean_abs_delta_z": float(np.mean([u["mean_abs_delta_z"] for u in us])),
                      "std_abs_delta_z": float(np.std([u["mean_abs_delta_z"] for u in us], ddof=1)),
                      "alpha": float(amap[c])}
    return {"units": units, "per_cat": per_cat,
            "mean_dprime": float(dp.mean()), "mean_abs_delta_z": float(rz.mean()),
            "mean_alpha": float(np.mean([amap[c] for c in CATEGORIES])),
            "alpha": dict(amap)}


# ---------------------------------------------------------------------------
# 2. tables
# ---------------------------------------------------------------------------
def pair_win_units(ev_p: dict, ev_q: dict) -> int:
    """5A-H 冻结 PAIR-WIN 判据逐 unit 计数：dRobust<0 AND dPres>= -eps。"""
    n = 0
    for a, b in zip(ev_p["units"], ev_q["units"]):
        assert (a["category"], a["seed"]) == (b["category"], b["seed"])
        if (a["mean_abs_delta_z"] - b["mean_abs_delta_z"] < 0
                and a["mean_dprime"] - b["mean_dprime"] >= -EPS):
            n += 1
    return n


def win_tie_loss(gains: list) -> str:
    if all(g > EPS for g in gains):
        return "win"
    if all(g < -EPS for g in gains):
        return "loss"
    if all(abs(g) <= EPS for g in gains):
        return "tie"
    return "mixed"


def harm_metrics(ev: dict, ev_b: dict) -> dict:
    gains = {c: ev["per_cat"][c]["mean_dprime"] - ev_b["per_cat"][c]["mean_dprime"] for c in CATEGORIES}
    drob = {c: ev["per_cat"][c]["mean_abs_delta_z"] - ev_b["per_cat"][c]["mean_abs_delta_z"] for c in CATEGORIES}
    worst_cat = min(gains, key=lambda c: gains[c])
    return {
        "mean_gain_vs_best_fixed": float(np.mean(list(gains.values()))),
        "worst_category": worst_cat,
        "worst_category_gain": float(gains[worst_cat]),
        "high_damage_recovery": float(np.mean([gains[c] for c in HD_SUBSET])),
        "negative_transfer_count": int(sum(1 for g in gains.values() if g < -EPS)),
        "pair_win_units_vs_best_fixed": pair_win_units(ev, ev_b),
        "win_tie_loss_vs_best_fixed": win_tie_loss(list(gains.values())),
        "mean_delta_robust_vs_best_fixed": float(np.mean(list(drob.values()))),
        "delta_dprime_vs_best_fixed": ev["mean_dprime"] - ev_b["mean_dprime"],
        "delta_robust_vs_best_fixed": ev["mean_abs_delta_z"] - ev_b["mean_abs_delta_z"],
        "per_category_gain": gains,
        "per_category_delta_robust": drob,
    }


def build_per_seed(evals: dict, ev_b: dict) -> list:
    rows = []
    for pname, ev in evals.items():
        for c in CATEGORIES:
            for u in [x for x in ev["units"] if x["category"] == c]:
                ub = get_metrics(LOOKUP, c, u["seed"], ev_b["alpha"][c])
                u0 = get_metrics(LOOKUP, c, u["seed"], 0.0)
                rows.append({
                    "policy": pname, "category": c, "seed": u["seed"],
                    "alpha": f"{u['alpha']:.1f}",
                    "mean_dprime": round(u["mean_dprime"], 6),
                    "mean_abs_delta_z": round(u["mean_abs_delta_z"], 6),
                    "damage_vs_alpha0": round(u0["mean_dprime"] - u["mean_dprime"], 6),
                    "gain_dprime_vs_best_fixed": round(u["mean_dprime"] - ub["mean_dprime"], 6),
                    "delta_robustness_vs_best_fixed": round(u["mean_abs_delta_z"] - ub["mean_abs_delta_z"], 6),
                    "image_auroc": round(u["image_auroc"], 6),
                    "pixel_auroc": round(u["pixel_auroc"], 6),
                    "aupro": round(u["aupro"], 6),
                    "fpr_clean": round(u["fpr_clean"], 6),
                    "source": u["source"],
                })
    return rows


def build_per_category(evals: dict, ev_b: dict) -> list:
    rows = []
    for pname, ev in evals.items():
        for c in CATEGORIES:
            pc = ev["per_cat"][c]
            rows.append({
                "policy": pname, "category": c, "alpha": f"{pc['alpha']:.1f}",
                "mean_dprime_mean": round(pc["mean_dprime"], 6),
                "mean_dprime_std": round(pc["std_dprime"], 6),
                "mean_abs_delta_z_mean": round(pc["mean_abs_delta_z"], 6),
                "mean_abs_delta_z_std": round(pc["std_abs_delta_z"], 6),
                "gain_vs_best_fixed_mean": round(pc["mean_dprime"] - ev_b["per_cat"][c]["mean_dprime"], 6),
                "delta_robust_vs_best_fixed_mean": round(
                    pc["mean_abs_delta_z"] - ev_b["per_cat"][c]["mean_abs_delta_z"], 6),
            })
    return rows


def build_policy_summary(evals: dict, ev_b: dict) -> tuple:
    harm = {}
    rows = []
    for pname, ev in evals.items():
        hm = harm_metrics(ev, ev_b)
        harm[pname] = hm
        rows.append({
            "policy": pname, "label": POLICY_LABEL[pname],
            "gated_categories": "+".join([c for c in CATEGORIES if ev["alpha"][c] == 0.0]) or "-",
            "mean_alpha": round(ev["mean_alpha"], 6),
            "mean_defect_dprime": round(ev["mean_dprime"], 6),
            "mean_robustness": round(ev["mean_abs_delta_z"], 6),
            "delta_dprime_vs_fixed": round(hm["delta_dprime_vs_best_fixed"], 6),
            "delta_robust_vs_fixed": round(hm["delta_robust_vs_best_fixed"], 6),
            "worst_category": hm["worst_category"],
            "worst_category_gain": round(hm["worst_category_gain"], 6),
            "negative_transfer_count": hm["negative_transfer_count"],
            "high_damage_recovery": round(hm["high_damage_recovery"], 6),
            "pair_win_units": f"{hm['pair_win_units_vs_best_fixed']}/15",
            "win_tie_loss": hm["win_tie_loss_vs_best_fixed"],
        })
    return rows, harm


# ---------------------------------------------------------------------------
# 3. Pareto / ranks / gate identity
# ---------------------------------------------------------------------------
LOOKUP: dict = {}


def pareto_analysis(points: list, tol: float = 1e-9) -> dict:
    """points: [{key, rob, dp}]；rob 越小越好、dp 越大越好。返回 {key: dominated_by_list}。"""
    dom = {}
    for p in points:
        dom_by = []
        for q in points:
            if q["key"] == p["key"]:
                continue
            if (q["dp"] >= p["dp"] - tol and q["rob"] <= p["rob"] + tol
                    and (q["dp"] > p["dp"] + tol or q["rob"] < p["rob"] - tol)):
                dom_by.append(q["key"])
        dom[p["key"]] = sorted(dom_by)
    return dom


def ranks_within(gate_metrics: dict, key: str, field: str, lower_is_better: bool) -> int:
    v = gate_metrics[key][field]
    if lower_is_better:
        return 1 + sum(1 for k, m in gate_metrics.items() if k != key and m[field] < v - 1e-12)
    return 1 + sum(1 for k, m in gate_metrics.items() if k != key and m[field] > v + 1e-12)


def build_gate_identity(evals: dict, harm: dict, gate_ids: list, dom: dict) -> list:
    """10 exhaustive gates + A/B/GC/SC-A/SC-B，含 gate 内 rank 与 pareto 状态。"""
    gate_metrics = {g: {"mean_defect_dprime": evals[g]["mean_dprime"],
                        "mean_robustness": evals[g]["mean_abs_delta_z"],
                        "worst_category_gain": harm[g]["worst_category_gain"],
                        "negative_transfer_count": harm[g]["negative_transfer_count"],
                        "high_damage_recovery": harm[g]["high_damage_recovery"]}
                    for g in gate_ids}
    rows = []
    for key in gate_ids + ["A_original", "B_best_fixed", "GC", "SC_A", "SC_B"]:
        ev, hm = evals[key], harm[key]
        gated = [c for c in CATEGORIES if ev["alpha"][c] == 0.0]
        row = {
            "policy": key,
            "role": ("exhaustive_control" if key in gate_ids else
                     "primary_method" if key == "GC" else
                     "matched_control_primary" if key == "SC_A" else
                     "matched_control_alt" if key == "SC_B" else "reference"),
            "gated_categories": "+".join(gated) if gated else "-",
            "mean_alpha": round(ev["mean_alpha"], 6),
            "mean_defect_dprime": round(ev["mean_dprime"], 6),
            "mean_robustness": round(ev["mean_abs_delta_z"], 6),
            "delta_dprime_vs_fixed": round(hm["delta_dprime_vs_best_fixed"], 6),
            "delta_robust_vs_fixed": round(hm["delta_robust_vs_best_fixed"], 6),
            "worst_category": hm["worst_category"],
            "worst_category_gain": round(hm["worst_category_gain"], 6),
            "negative_transfer_count": hm["negative_transfer_count"],
            "high_damage_recovery": round(hm["high_damage_recovery"], 6),
            "pair_win_count": hm["pair_win_units_vs_best_fixed"],
            "win_tie_loss": hm["win_tie_loss_vs_best_fixed"],
            "pareto_dominated": "true" if dom[key] else "false",
            "dominated_by": ";".join(dom[key]),
        }
        if key in gate_metrics:
            row["rank_dprime_within_gates"] = ranks_within(gate_metrics, key, "mean_defect_dprime", False)
            row["rank_robust_within_gates"] = ranks_within(gate_metrics, key, "mean_robustness", True)
            row["rank_worst_gain_within_gates"] = ranks_within(gate_metrics, key, "worst_category_gain", False)
            row["rank_neg_transfer_within_gates"] = ranks_within(gate_metrics, key, "negative_transfer_count", True)
        else:
            for f in ("rank_dprime_within_gates", "rank_robust_within_gates",
                      "rank_worst_gain_within_gates", "rank_neg_transfer_within_gates"):
                row[f] = ""
        rows.append(row)
    return rows


def build_loco(evals: dict, ev_b: dict) -> list:
    """drop-one-category 聚合（不重新拟合 predictor、不重算 ranking）。"""
    rows = []
    for held in CATEGORIES:
        rest = [c for c in CATEGORIES if c != held]
        base = {p: {} for p in ("GC", "SC_A", "SC_B")}
        for p in base:
            g = [evals[p]["per_cat"][c]["mean_dprime"] - ev_b["per_cat"][c]["mean_dprime"] for c in rest]
            r = [evals[p]["per_cat"][c]["mean_abs_delta_z"] - ev_b["per_cat"][c]["mean_abs_delta_z"] for c in rest]
            base[p] = {"delta_dprime": float(np.mean(g)), "delta_robust": float(np.mean(r)),
                       "mean_dprime": float(np.mean([evals[p]["per_cat"][c]["mean_dprime"] for c in rest])),
                       "mean_robustness": float(np.mean([evals[p]["per_cat"][c]["mean_abs_delta_z"] for c in rest]))}
        rows.append({
            "held_out": held, "n_categories": len(rest), "remaining": "+".join(rest),
            "GC_delta_dprime_vs_fixed": round(base["GC"]["delta_dprime"], 6),
            "GC_delta_robust_vs_fixed": round(base["GC"]["delta_robust"], 6),
            "SC_A_delta_dprime_vs_fixed": round(base["SC_A"]["delta_dprime"], 6),
            "SC_A_delta_robust_vs_fixed": round(base["SC_A"]["delta_robust"], 6),
            "SC_B_delta_dprime_vs_fixed": round(base["SC_B"]["delta_dprime"], 6),
            "GC_minus_SC_A_delta_dprime": round(base["GC"]["delta_dprime"] - base["SC_A"]["delta_dprime"], 6),
            "GC_minus_SC_A_delta_robust": round(base["GC"]["delta_robust"] - base["SC_A"]["delta_robust"], 6),
            "GC_minus_SC_B_delta_dprime": round(base["GC"]["delta_dprime"] - base["SC_B"]["delta_dprime"], 6),
        })
    return rows


def build_seed_stability(evals: dict, ev_b: dict) -> list:
    rows = []
    for pname in ("A_original", "B_best_fixed", "GC", "SC_A", "SC_B"):
        ev = evals[pname]
        for s in SEEDS:
            us = [u for u in ev["units"] if u["seed"] == s]
            rows.append({"policy": pname, "seed": s,
                         "mean_dprime": round(float(np.mean([u["mean_dprime"] for u in us])), 6),
                         "mean_abs_delta_z": round(float(np.mean([u["mean_abs_delta_z"] for u in us])), 6),
                         "n_categories": len(us)})
    # GC vs control 的逐 seed 差
    for ctrl in ("SC_A", "SC_B"):
        for s in SEEDS:
            g = [u for u in evals["GC"]["units"] if u["seed"] == s]
            q = [u for u in evals[ctrl]["units"] if u["seed"] == s]
            d_dp = float(np.mean([a["mean_dprime"] for a in g])) - float(np.mean([b["mean_dprime"] for b in q]))
            d_rb = float(np.mean([a["mean_abs_delta_z"] for a in g])) - float(np.mean([b["mean_abs_delta_z"] for b in q]))
            rows.append({"policy": f"GC_minus_{ctrl}", "seed": s,
                         "mean_dprime": round(d_dp, 6), "mean_abs_delta_z": round(d_rb, 6),
                         "n_categories": len(g)})
    return rows


# ---------------------------------------------------------------------------
# 4. 等价性 / sanity / verdict
# ---------------------------------------------------------------------------
def raw_score_equivalence() -> dict:
    """同一 (cat, seed, alpha) 在 5A-H 与 5C 的 raw per-image 分数应逐位一致。"""
    def collect(root: Path, cfgs):
        out = {}
        for info_p in sorted(root.glob("*/seed_*/config_*/info.json")):
            d = json.loads(info_p.read_text())
            if d.get("status") != "OK" or d["config"] not in cfgs:
                continue
            out[(d["category"], int(d["seed"]), akey(d["alpha_l2"]))] = info_p.parent
        return out

    a = collect(RAW_5AH, {"B0", "B2", "C2", "C3"})
    b = collect(RAW_5C, {"GF", "GC", "SG"})
    common = sorted(set(a) & set(b))
    max_ds, max_dt, n_rows = 0.0, 0.0, 0
    for k in common:
        ia = json.loads((a[k] / "info.json").read_text())
        ib = json.loads((b[k] / "info.json").read_text())
        max_dt = max(max_dt, abs(float(ia["tau_val"]) - float(ib["tau_val"])))
        sa = {(r["image_path"], r["subset"], r["shift"]): float(r["score"])
              for r in csv.DictReader(open(a[k] / "per_image.csv", newline=""))}
        sb = {(r["image_path"], r["subset"], r["shift"]): float(r["score"])
              for r in csv.DictReader(open(b[k] / "per_image.csv", newline=""))}
        keys = set(sa) & set(sb)
        if keys:
            max_ds = max(max_ds, max(abs(sa[x] - sb[x]) for x in keys))
            n_rows += len(keys)
    return {"n_common": len(common), "n_score_rows": n_rows,
            "max_abs_score_diff": max_ds, "max_abs_tau_diff": max_dt}



def sanity_checks(freeze: dict, policies: dict, gate_ids: list, equiv: dict,
                  evals: dict, harm: dict, dom: dict, pre_hashes: dict) -> list:
    checks = []
    f5c = json.loads((ROOT / "results" / "experiment_5c" / "reference" / "geometry_policy_freeze.json").read_text())
    gc5c = f5c["policies"]["D_geometry_conservative"]["alpha_by_category"]

    mod = subprocess.run(["git", "status", "--porcelain"], cwd=ROOT,
                         capture_output=True, text=True).stdout.splitlines()
    dirty = [x for x in mod if not x.startswith("??")]
    checks.append(("S1_P0_repo_state",
                   bool(git_branch() == "main" and not dirty and git_head() == freeze["git_head"]),
                   f"branch={git_branch()} head={git_head()[:7]}==freeze_head, "
                   f"modified_tracked_files={len(dirty)} (untracked 5D outputs excluded)"))
    checks.append(("S2_primary_predictor_frozen",
                   freeze["frozen_predictors"]["primary"] == pol5d.PRED_PRIMARY == f5c["primary_predictor"],
                   f"primary={freeze['frozen_predictors']['primary']} (identical to 5C freeze)"))
    checks.append(("S3_control_predictor_frozen",
                   freeze["frozen_predictors"]["matched_control"] == pol5d.PRED_CONTROL == f5c["negative_control_predictor"],
                   f"control={freeze['frozen_predictors']['matched_control']} (identical to 5C freeze)"))
    src = freeze["predictor_source_file"]
    checks.append(("S4_normal_only_predictors",
                   bool("normal_only_predictors" in src and "test" not in src
                        and freeze["predictor_source_is_normal_only"]),
                   f"source={src} (5B normal-only asset; no test-image path)"))
    checks.append(("S5_target_results_read_false",
                   freeze["target_results_read"] is False,
                   "policy_freeze.json target_results_read=false (freeze precedes any target analysis)"))
    checks.append(("S6_geometry_assignment_matches_5C",
                   bool(policies["GC"]["alpha"] == gc5c == pol5d.GC_5C_FROZEN),
                   f"GC={policies['GC']['alpha']} == 5C frozen D_geometry_conservative"))
    matched = ["GC", "SC_A", "SC_B"] + gate_ids
    checks.append(("S7_gate_size_2of5",
                   all(sum(1 for c in CATEGORIES if policies[p]["alpha"][c] == 0.0) == 2 for p in matched),
                   "all adaptive/matched policies close exactly 2/5 categories"))
    checks.append(("S8_mean_alpha_0p3",
                   all(abs(policies[p]["mean_alpha"] - 0.3) < 1e-12 for p in matched),
                   "all adaptive/matched policies have mean alpha = 0.3 (budget matched)"))
    checks.append(("S9_alpha_set",
                   all(policies[p]["alpha"][c] in (0.0, 0.5) for p in matched for c in CATEGORIES),
                   "all assigned alpha in {0, 0.5}"))
    checks.append(("S10_fixed_baseline_frozen",
                   bool(abs(freeze["best_fixed_alpha"] - 0.5) < 1e-12
                        and abs(f5c["best_fixed_alpha"] - 0.5) < 1e-12
                        and all(policies["B_best_fixed"]["alpha"][c] == 0.5 for c in CATEGORIES)),
                   "best fixed = alpha 0.5 (5C frozen, reused as 5A-H B2, not re-selected)"))
    checks.append(("S11_epsilon_unchanged",
                   bool(abs(EPS - 0.10) < 1e-12 and abs(h5.EPS - 0.10) < 1e-12
                        and abs(freeze["epsilon"] - 0.10) < 1e-12),
                   f"eps={EPS} == 5A-H frozen band"))
    checks.append(("S12_categories_unchanged",
                   bool(CATEGORIES == h5.CATEGORIES and len(CATEGORIES) == 5
                        and freeze["categories"] == CATEGORIES),
                   f"categories={CATEGORIES}"))
    checks.append(("S13_seeds_unchanged",
                   bool(SEEDS == h5.SEEDS == [0, 1, 2] and freeze["seeds"] == SEEDS),
                   f"seeds={SEEDS}"))
    checks.append(("S14_illumination_protocol_unchanged",
                   bool(list(h5.SHIFTS) == ["brightness_0.7", "brightness_1.3", "gamma_0.7", "gamma_1.3"]),
                   f"shifts={list(h5.SHIFTS)} (5A-H frozen photometric protocol)"))
    checks.append(("S15_historical_raw_score_equivalence",
                   bool(equiv["n_common"] > 0 and equiv["max_abs_score_diff"] <= 1e-9
                        and equiv["max_abs_tau_diff"] <= 1e-9),
                   f"5C vs 5A-H at identical alpha: n_keys={equiv['n_common']}, "
                   f"rows={equiv['n_score_rows']}, max|dscore|={equiv['max_abs_score_diff']:.3e}, "
                   f"max|dtau|={equiv['max_abs_tau_diff']:.3e}"))
    checks.append(("S16_exhaustive_gates_complete",
                   bool(len(gate_ids) == 10
                        and all(sum(1 for c in CATEGORIES if policies[g]["alpha"][c] == 0.0) == 2 for g in gate_ids)),
                   f"n_gates={len(gate_ids)} (C(5,2)); each closes 2/5 with mean alpha 0.3"))
    vals = []
    for p in matched + ["A_original", "B_best_fixed"]:
        vals += [evals[p]["mean_dprime"], evals[p]["mean_abs_delta_z"]]
        vals += [harm[p]["worst_category_gain"], harm[p]["high_damage_recovery"]]
    checks.append(("S17_all_outputs_finite", bool(np.all(np.isfinite(vals))),
                   f"{len(vals)} aggregate metric values all finite"))
    post = {}
    for rel, rec in pre_hashes.items():
        p = ROOT / rel
        post[rel] = md5_file(p)
    unchanged = all(post[r] == pre_hashes[r]["md5"] for r in pre_hashes)
    checks.append(("S18_frozen_asset_hashes_unchanged", bool(unchanged),
                   f"{len(pre_hashes)} tracked frozen assets md5 identical before/after analysis"))
    return checks


def compute_verdict(evals: dict, harm: dict, gate_ids: list, dom: dict, loco: list,
                    seed_stab: list, ctrl: str) -> dict:
    """按 PRE-RUN README §10 预注册的判定链（首个命中即判定）。"""
    dpg, robg = evals["GC"]["mean_dprime"], evals["GC"]["mean_abs_delta_z"]
    dpc, robc = evals[ctrl]["mean_dprime"], evals[ctrl]["mean_abs_delta_z"]
    d_dp, d_rob = dpg - dpc, robg - robc
    rank_dp = 1 + sum(1 for g in gate_ids if evals[g]["mean_dprime"] > dpg + 1e-12)
    rank_rb = 1 + sum(1 for g in gate_ids if evals[g]["mean_abs_delta_z"] < robg - 1e-12)
    n_ge = sum(1 for g in gate_ids if evals[g]["mean_dprime"] >= dpg - 1e-9)
    dominated_by_ctrl = ctrl in dom.get("GC", [])
    lk = f"GC_minus_{ctrl}_delta_dprime"
    rk = f"GC_minus_{ctrl}_delta_robust"
    loco_positive = [r[lk] > 0 for r in loco]
    sw = [r for r in seed_stab if r["policy"] == f"GC_minus_{ctrl}"]
    seed_pair_wins = sum(1 for r in sw if r["mean_abs_delta_z"] < 0 and r["mean_dprime"] >= -EPS)

    cond_D = {
        "GC_dprime_clearly_worse_than_control": d_dp <= -EPS,
        "control_pareto_dominates_GC": dominated_by_ctrl,
        "GC_rank_dprime_among_gates_ge_8": rank_dp >= 8,
        "single_category_driver_of_identity_gain": any(not p for p in loco_positive),
    }
    cond_A = {
        "GC_dprime_gain_over_control_ge_eps": d_dp >= EPS,
        "GC_robustness_not_clearly_worse": d_rob <= 0.02,
        "GC_worst_category_gain_ge_control": harm["GC"]["worst_category_gain"] >= harm[ctrl]["worst_category_gain"],
        "GC_negative_transfer_le_control": harm["GC"]["negative_transfer_count"] <= harm[ctrl]["negative_transfer_count"],
        "seed_level_pair_win_ge_2of3": seed_pair_wins >= 2,
        "rank_dprime_le_3_and_pareto_efficient": bool(rank_dp <= 3 and len(dom.get("GC", [])) == 0),
        "loco_gain_positive_all_holdouts": all(loco_positive),
    }
    cond_C = {
        "GC_and_control_indistinguishable": abs(d_dp) < EPS,
        "at_least_4_gates_not_worse_than_GC": n_ge >= 4,
    }
    if any(cond_D.values()):
        case, means = "CASE_D", "Geometry predictor does not provide sufficient actionable policy information."
    elif all(cond_A.values()):
        case, means = "CASE_A", "Geometry identity receives direct matched-control support (supported, not proved)."
    elif all(cond_C.values()):
        case, means = "CASE_C", "5C benefit comes mainly from conservative gating itself, not from geometry."
    else:
        case, means = "CASE_B", "Geometry-guided gating remains useful, but predictor identity is not established."
    return {
        "primary_control": ctrl,
        "case": case, "meaning": means,
        "GC_mean_dprime": round(dpg, 6), "control_mean_dprime": round(dpc, 6),
        "GC_mean_robustness": round(robg, 6), "control_mean_robustness": round(robc, 6),
        "delta_dprime_GC_minus_control": round(d_dp, 6),
        "delta_robust_GC_minus_control": round(d_rob, 6),
        "rank_dprime_within_gates": rank_dp, "rank_robust_within_gates": rank_rb,
        "n_gates_not_worse_than_GC": n_ge,
        "dominated_by": dom.get("GC", []),
        "seed_level_pair_wins": f"{seed_pair_wins}/3",
        "loco_gain_positive": loco_positive,
        "GC_worst_category_gain": harm["GC"]["worst_category_gain"],
        "control_worst_category_gain": harm[ctrl]["worst_category_gain"],
        "GC_negative_transfer": harm["GC"]["negative_transfer_count"],
        "control_negative_transfer": harm[ctrl]["negative_transfer_count"],
        "conditions_CASE_D": cond_D, "conditions_CASE_A": cond_A, "conditions_CASE_C": cond_C,
    }


# ---------------------------------------------------------------------------
# 5. figures
# ---------------------------------------------------------------------------
STYLE = {"A_original": ("#999999", "o"), "B_best_fixed": ("#333333", "s"),
         "GC": ("#2E8B57", "*"), "SC_A": ("#C44E52", "D"), "SC_B": ("#DD8452", "v")}


def fig_gate_pareto(evals: dict, gate_ids: list) -> None:
    fig, ax = plt.subplots(figsize=(8.2, 5.6))
    for g in gate_ids:
        if g in ("GATE:bottle+grid", "GATE:bottle+cable", "GATE:grid+screw"):
            continue
        ax.scatter(evals[g]["mean_abs_delta_z"], evals[g]["mean_dprime"], s=55,
                   facecolors="none", edgecolors="#7F7F7F", marker="o", zorder=3)
    pts = [{"key": k, "rob": evals[k]["mean_abs_delta_z"], "dp": evals[k]["mean_dprime"]}
           for k in gate_ids + ["A_original", "B_best_fixed", "GC", "SC_A", "SC_B"]]
    dom = pareto_analysis(pts)
    front = sorted([p for p in pts if not dom[p["key"]]], key=lambda p: p["rob"])
    ax.plot([p["rob"] for p in front], [p["dp"] for p in front], ls="--", lw=1.2,
            color="#4C72B0", alpha=0.8, zorder=2, label="Pareto frontier (10 gates + Original + Best Fixed)")
    for name in ("GC", "SC_A", "SC_B"):
        c, m = STYLE[name]
        ax.scatter(evals[name]["mean_abs_delta_z"], evals[name]["mean_dprime"], s=330 if name == "GC" else 190,
                   color=c, marker=m, edgecolors="black", linewidths=1.4 if name == "GC" else 0.8,
                   zorder=6, label=POLICY_LABEL[name])
    for name in ("A_original", "B_best_fixed"):
        c, m = STYLE[name]
        ax.scatter(evals[name]["mean_abs_delta_z"], evals[name]["mean_dprime"], s=130, color=c,
                   marker=m, zorder=5, label=POLICY_LABEL[name])
    ax.set_xlabel("mean |ΔNormalScore_z|  (lower = better robustness)")
    ax.set_ylabel("mean defect d′  (higher = better preservation)")
    ax.set_title("Experiment 5D — preservation/robustness Pareto plane\n"
                 "(10 matched gates of size 2/5, mean α=0.3; GC highlighted)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "gate_pareto.png", dpi=160)
    plt.close(fig)


def fig_gate_exhaustive_rank(evals: dict, gate_ids: list) -> None:
    labels = [g.replace("GATE:", "") for g in gate_ids]
    r_dp = [1 + sum(1 for h in gate_ids if evals[h]["mean_dprime"] > evals[g]["mean_dprime"] + 1e-12)
            for g in gate_ids]
    r_rb = [1 + sum(1 for h in gate_ids if evals[h]["mean_abs_delta_z"] < evals[g]["mean_abs_delta_z"] - 1e-12)
            for g in gate_ids]
    colors = ["#2E8B57" if g == "GATE:bottle+grid" else "#C44E52" if g == "GATE:bottle+cable"
              else "#DD8452" if g == "GATE:grid+screw" else "#BBBBBB" for g in gate_ids]
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.8), sharey=True)
    y = np.arange(len(gate_ids))[::-1]
    for ax, vals, ttl in ((axes[0], r_dp, "rank by defect preservation (1 = best d′)"),
                          (axes[1], r_rb, "rank by robustness (1 = lowest |Δz|)")):
        ax.barh(y, vals, color=colors, edgecolor="black", linewidth=0.5)
        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=9)
        ax.set_xlabel(ttl)
        ax.set_xlim(0, len(gate_ids) + 0.6)
        ax.invert_xaxis()
        ax.grid(alpha=0.3, axis="x")
    axes[0].set_title("Experiment 5D — Geometry Conservative's position among the 10 matched gates\n"
                      "(green = GC/bottle+grid, red = SC-A/bottle+cable, orange = SC-B/grid+screw)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "gate_exhaustive_rank.png", dpi=160)
    plt.close(fig)


def fig_geometry_vs_sensitivity(evals: dict, harm: dict, seed_stab: list) -> None:
    order = ["GC", "SC_A", "SC_B"]
    cols = [STYLE[p][0] for p in order]
    fig, axes = plt.subplots(1, 3, figsize=(14.6, 4.6))
    axes[0].bar(order, [harm[p]["delta_dprime_vs_best_fixed"] for p in order], color=cols,
                edgecolor="black", linewidth=0.6)
    axes[0].axhline(0, color="k", lw=0.9)
    axes[0].set_title("Δ defect d′ vs Best Fixed α=0.5\n(higher = better preservation)")
    axes[0].set_ylabel("Δ mean defect d′")
    axes[1].bar(order, [harm[p]["mean_delta_robust_vs_best_fixed"] for p in order], color=cols,
                edgecolor="black", linewidth=0.6)
    axes[1].axhline(0, color="k", lw=0.9)
    axes[1].set_title("Δ mean |ΔNormalScore_z| vs Best Fixed\n(lower = better robustness)")
    axes[1].set_ylabel("Δ mean |Δz|")
    x = np.arange(len(SEEDS)); w = 0.36
    for i, ctrl in enumerate(("SC_A", "SC_B")):
        vals = [r["mean_dprime"] for r in seed_stab if r["policy"] == f"GC_minus_{ctrl}"]
        axes[2].bar(x + (i - 0.5) * w, vals, w, label=f"GC − {ctrl}",
                    color=STYLE[ctrl][0], edgecolor="black", linewidth=0.6)
    axes[2].axhline(0, color="k", lw=0.9)
    axes[2].axhline(EPS, color="gray", ls=":", lw=1.0)
    axes[2].axhline(-EPS, color="gray", ls=":", lw=1.0)
    axes[2].set_xticks(x); axes[2].set_xticklabels([f"seed {s}" for s in SEEDS])
    axes[2].set_title("per-seed Δ d′: GC − control\n(dotted = ±ε preservation band)")
    axes[2].legend(fontsize=8)
    for ax in axes:
        ax.grid(alpha=0.3, axis="y")
    fig.suptitle("Experiment 5D — Geometry Conservative vs matched Sensitivity Conservative "
                 "(identical budget: gate 2/5, mean α=0.3, α∈{0,0.5})", y=1.02)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "geometry_vs_sensitivity.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def fig_per_category_harm(evals: dict, harm: dict) -> None:
    fig, ax = plt.subplots(figsize=(10.4, 4.8))
    x = np.arange(len(CATEGORIES)); w = 0.26
    for i, p in enumerate(("GC", "SC_A", "SC_B")):
        vals = [harm[p]["per_category_gain"][c] for c in CATEGORIES]
        ax.bar(x + (i - 1) * w, vals, w, label=POLICY_LABEL[p], color=STYLE[p][0],
               edgecolor="black", linewidth=0.6)
    ax.axhline(0, color="k", lw=0.9)
    ax.axhline(EPS, color="gray", ls=":", lw=1.0, label="±ε (0.10) preservation band")
    ax.axhline(-EPS, color="gray", ls=":", lw=1.0)
    ax.set_xticks(x); ax.set_xticklabels(CATEGORIES)
    ax.set_ylabel("per-category Δ defect d′ vs Best Fixed α=0.5")
    ax.set_title("Experiment 5D — per-category harm relative to Best Fixed α=0.5\n"
                 "(gated categories α=0, kept categories unchanged at α=0.5)")
    ax.grid(alpha=0.3, axis="y")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "per_category_harm.png", dpi=160)
    plt.close(fig)


def fig_seed_stability(evals: dict, seed_stab: list) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.6))
    for ax, field, ttl in ((axes[0], "mean_dprime", "mean defect d′"),
                           (axes[1], "mean_abs_delta_z", "mean |ΔNormalScore_z|")):
        for p in ("A_original", "B_best_fixed", "GC", "SC_A", "SC_B"):
            vals = [r[field] for r in seed_stab if r["policy"] == p]
            ax.plot(SEEDS, vals, marker="o", label=POLICY_LABEL[p], color=STYLE[p][0],
                    lw=2.4 if p == "GC" else 1.4, zorder=5 if p == "GC" else 3)
        ax.set_xticks(SEEDS); ax.set_xlabel("seed")
        ax.set_title(f"Experiment 5D — per-seed {ttl}")
        ax.grid(alpha=0.3); ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "seed_stability.png", dpi=160)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 6. main
# ---------------------------------------------------------------------------
def write_csv(path: Path, rows: list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def main() -> None:
    global LOOKUP
    for d in (SUM_DIR, SAN_DIR, FIG_DIR, LOG_DIR):
        d.mkdir(parents=True, exist_ok=True)
    print(f"[5D] git HEAD = {git_head()}  branch = {git_branch()}", flush=True)

    freeze = json.loads((REF_DIR / "policy_freeze.json").read_text())
    pre_hashes = json.loads((REF_DIR / "asset_hashes.json").read_text())
    ca = freeze["category_assignments"]
    gate_ids = sorted(freeze["exhaustive_gates"])

    LOOKUP = build_lookup()
    print(f"[5D] (cat, seed, alpha) lookup keys = {len(LOOKUP)} (expected 30, CPU-only, no GPU)", flush=True)

    alpha_maps = {
        "A_original": {c: 0.0 for c in CATEGORIES},
        "B_best_fixed": {c: 0.5 for c in CATEGORIES},
        "GC": {c: float(ca["GC"][c]) for c in CATEGORIES},
        "SC_A": {c: float(ca["SC_A"][c]) for c in CATEGORIES},
        "SC_B": {c: float(ca["SC_B"][c]) for c in CATEGORIES},
    }
    for g in gate_ids:
        gated = tuple(g.split(":", 1)[1].split("+"))
        alpha_maps[g] = {c: (0.0 if c in gated else 0.5) for c in CATEGORIES}

    evals = {k: eval_policy(LOOKUP, amap) for k, amap in alpha_maps.items()}
    ev_b = evals["B_best_fixed"]
    named = ["A_original", "B_best_fixed", "GC", "SC_A", "SC_B"]
    eval_cmp = {k: evals[k] for k in named + gate_ids}

    ps_rows, harm = build_policy_summary({k: evals[k] for k in named}, ev_b)
    for g in gate_ids:
        hm = harm_metrics(evals[g], ev_b)
        harm[g] = hm
    per_seed = build_per_seed(eval_cmp, ev_b)
    per_cat = build_per_category(eval_cmp, ev_b)
    seed_stab = build_seed_stability(evals, ev_b)
    loco = build_loco(evals, ev_b)

    pts = [{"key": k, "rob": evals[k]["mean_abs_delta_z"], "dp": evals[k]["mean_dprime"]}
           for k in gate_ids + named]
    dom = pareto_analysis(pts)
    gate_rows = build_gate_identity(evals, harm, gate_ids, dom)
    exhaust_rows = [r for r in gate_rows if r["policy"] in gate_ids]
    pareto_rows = [{"policy": p["key"], "mean_defect_dprime": round(p["dp"], 6),
                    "mean_robustness": round(p["rob"], 6),
                    "is_pareto_efficient": "true" if not dom[p["key"]] else "false",
                    "dominated_by": ";".join(dom[p["key"]]),
                    "is_geometry_policy": "true" if p["key"] in ("GATE:bottle+grid", "GC") else "false",
                    "is_sc_a_policy": "true" if p["key"] in ("GATE:bottle+cable", "SC_A") else "false",
                    "is_sc_b_policy": "true" if p["key"] in ("GATE:grid+screw", "SC_B") else "false"}
                   for p in pts]

    write_csv(SUM_DIR / "policy_summary.csv", ps_rows)
    write_csv(SUM_DIR / "per_category.csv", per_cat)
    write_csv(SUM_DIR / "per_seed.csv", per_seed)
    write_csv(SUM_DIR / "gate_identity.csv", gate_rows)
    write_csv(SUM_DIR / "exhaustive_gates.csv", exhaust_rows)
    write_csv(SUM_DIR / "pareto_summary.csv", pareto_rows)
    write_csv(SUM_DIR / "loco_summary.csv", loco)
    write_csv(SUM_DIR / "seed_stability.csv", seed_stab)

    v_primary = compute_verdict(evals, harm, gate_ids, dom, loco, seed_stab, "SC_A")
    v_secondary = compute_verdict(evals, harm, gate_ids, dom, loco, seed_stab, "SC_B")
    equiv = raw_score_equivalence()
    checks = sanity_checks(freeze, evals, gate_ids, equiv, evals, harm, dom, pre_hashes)
    fails = [c for c, ok, _ in checks if not ok]
    write_csv(SAN_DIR / "sanity_checks.csv",
              [{"check": c, "status": "PASS" if ok else "FAIL", "detail": d} for c, ok, d in checks])

    verdict = {
        "experiment": "5D",
        "title": "Matched-Control Gate Validation of Geometry-Guided Safe Normalization",
        "freeze_sha256": (REF_DIR / "policy_freeze.sha256").read_text().split()[0],
        "git_head": git_head(),
        "primary_control": "SC_A",
        "primary_verdict": v_primary,
        "secondary_control_SC_B": v_secondary,
        "sanity_failed": fails,
        "historical_equivalence": equiv,
        "no_gpu": True,
        "note": ("EXHAUSTIVE CONTROLS ARE EVALUATION-ONLY. NO BEST RANDOM GATE MAY REPLACE "
                 "THE FROZEN GEOMETRY POLICY. Verdict ladder was pre-registered in "
                 "experiments/experiment5d/README.md section 10 before any target analysis."),
    }
    (SUM_DIR / "verdict.json").write_text(json.dumps(verdict, indent=2))

    fig_gate_pareto(evals, gate_ids)
    fig_gate_exhaustive_rank(evals, gate_ids)
    fig_geometry_vs_sensitivity(evals, harm, seed_stab)
    fig_per_category_harm(evals, harm)
    fig_seed_stability(evals, seed_stab)
    print("[5D] figures written: gate_pareto / gate_exhaustive_rank / geometry_vs_sensitivity / "
          "per_category_harm / seed_stability", flush=True)

    report(evals, harm, gate_rows, exhaust_rows, loco, seed_stab, v_primary, v_secondary, checks, equiv)



def report(evals, harm, gate_rows, exhaust_rows, loco, seed_stab, v1, v2, checks, equiv):
    L = 118
    print("=" * L)
    print("[5D RESULTS] Matched-Control Gate Validation (CPU-only recombination of frozen 5A-H raw scores)")
    print("=" * L)
    print(f"historical raw-score equivalence : n_keys={equiv['n_common']}, rows={equiv['n_score_rows']}, "
          f"max|dscore|={equiv['max_abs_score_diff']:.3e}, max|dtau|={equiv['max_abs_tau_diff']:.3e}")
    npass = sum(1 for _, ok, _ in checks if ok)
    print(f"sanity                           : {npass}/{len(checks)} PASS")
    for c, ok, d in checks:
        if not ok:
            print(f"    FAIL {c}: {d}")
    print("-" * L)
    hdr = (f"{'policy':<12}{'gate':<22}{'mean a':>7}{'d-prime':>10}{'|dz|':>9}"
           f"{'dd-prime':>10}{'d|dz|':>9}{'worstG':>9}{'negTr':>7}{'HD':>9}{'pairWin':>9}")
    print(hdr)
    for r in gate_rows:
        print(f"{r['policy']:<12}{r['gated_categories']:<22}{r['mean_alpha']:>7.4f}"
              f"{r['mean_defect_dprime']:>10.4f}{r['mean_robustness']:>9.4f}"
              f"{r['delta_dprime_vs_fixed']:>+10.4f}{r['delta_robust_vs_fixed']:>+9.4f}"
              f"{r['worst_category_gain']:>+9.4f}{r['negative_transfer_count']:>7d}"
              f"{r['high_damage_recovery']:>+9.4f}{r['pair_win_count']:>6d}/15")
    print("-" * L)
    print("GC position among the 10 matched gates")
    gc = [r for r in exhaust_rows if r["policy"] == "GATE:bottle+grid"][0]
    print(f"    rank by defect preservation : {gc['rank_dprime_within_gates']}/10")
    print(f"    rank by robustness          : {gc['rank_robust_within_gates']}/10")
    print(f"    rank by worst-category gain : {gc['rank_worst_gain_within_gates']}/10")
    print(f"    rank by negative transfer   : {gc['rank_neg_transfer_within_gates']}/10")
    print(f"    pareto dominated?           : {gc['pareto_dominated']}  ({gc['dominated_by'] or '-'})")
    print("-" * L)
    print("Q1  GC vs matched controls (identical budget: gate 2/5, mean alpha 0.3)")
    for v in (v1, v2):
        print(f"    GC vs {v['primary_control']:<5}: dd-prime={v['delta_dprime_GC_minus_control']:+.4f} "
              f"d|dz|={v['delta_robust_GC_minus_control']:+.4f} "
              f"worst_gain GC/ctrl={v['GC_worst_category_gain']:+.3f}/{v['control_worst_category_gain']:+.3f} "
              f"negTr GC/ctrl={v['GC_negative_transfer']}/{v['control_negative_transfer']} "
              f"seed_pair_wins={v['seed_level_pair_wins']} -> {v['case']}")
    print("-" * L)
    print("LOCO (drop-one-category aggregation, no refit)")
    for r in loco:
        print(f"    hold out {r['held_out']:<9} GC-SC_A dd-prime={r['GC_minus_SC_A_delta_dprime']:+.4f} "
              f"GC-SC_B dd-prime={r['GC_minus_SC_B_delta_dprime']:+.4f} "
              f"(GC {r['GC_delta_dprime_vs_fixed']:+.4f}, SC_A {r['SC_A_delta_dprime_vs_fixed']:+.4f})")
    print("-" * L)
    print("per-seed aggregate delta (GC - control)")
    for ctrl in ("SC_A", "SC_B"):
        vals = [r for r in seed_stab if r["policy"] == f"GC_minus_{ctrl}"]
        print(f"    GC - {ctrl}: dd-prime={[round(x['mean_dprime'], 4) for x in vals]} "
              f"d|dz|={[round(x['mean_abs_delta_z'], 4) for x in vals]}")
    print("=" * L)
    print(f"FINAL VERDICT (primary control SC_A) : {v1['case']} - {v1['meaning']}")
    print(f"with SC_B as control (informational) : {v2['case']} - {v2['meaning']}")
    print("EXHAUSTIVE CONTROLS ARE EVALUATION-ONLY. NO BEST RANDOM GATE MAY REPLACE THE FROZEN GEOMETRY POLICY.")
    print("=" * L)


if __name__ == "__main__":
    main()
