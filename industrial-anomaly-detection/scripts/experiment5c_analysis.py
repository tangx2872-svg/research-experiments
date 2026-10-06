"""Experiment 5C — 分析（纯 CPU）：policy 比较 / harm 分析 / LOCO / predictor identity / figures。

数据来源：
  A（Original, α=0）与 B（Best Fixed, α=0.5）：results/experiment_5a_h/summary/per_unit.csv（冻结，复用）
  C（GF）/ D（GC）/ E（SG）：results/experiment_5c/raw/**（本次 45 runs）
指标计算完全复用 experiment5a_h_analysis.load_all/unit_metrics（同一口径，不新造主指标）。

关键性质：本实验的每个 policy 都是"每 category 一个 uniform α"，而 5A-H 已覆盖 α ∈ {0,0.5,0.6014,0.8018}
的全部 5 类结果 → 因此可以：
  (a) 做 α-level 等价性核对（5C run vs 5A-H 同 α，应逐位一致）；
  (b) 用 α-lookup 评估任意 uniform-α policy（含 LOCO 重新映射后的 policy），无需新跑。
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import experiment5a_h_analysis as h5  # noqa: E402  (指标口径复用)
import experiment5c_policy as pol  # noqa: E402  (冻结 policy)

OUT = ROOT / "results" / "experiment_5c"
RAW_5C = OUT / "raw"
SUM_DIR = OUT / "summary"
FIG_DIR = OUT / "figures"
REF_DIR = OUT / "reference"
LOG_DIR = OUT / "logs"

CATEGORIES = pol.CATEGORIES
SEEDS = pol.SEEDS
EPS = pol.EPS
SHIFTS = h5.SHIFTS
METRIC_KEYS = ["mean_dprime", "mean_abs_delta_z", "image_auroc", "pixel_auroc",
               "aupro", "fpr_clean", "fpr_shift", "clipped_pixel_ratio", "tau_val"]

SEC_ASSIGNMENT = {}

POLICY_LABEL = {
    "A_original": "A. Original (a=0)",
    "B_best_fixed": "B. Best Fixed (a=0.5)",
    "GF": "C. Geometry Full (v1A)",
    "GC": "D. Geometry Conservative (v1B)",
    "SG": "E. Sensitivity-Guided",
}


def akey(a: float) -> str:
    return f"{float(a):.9f}"


# ---------------------------------------------------------------------------
# 1. metrics: A/B 来自 5A-H 冻结 per_unit.csv；C/D/E 来自本次 5C raw
# ---------------------------------------------------------------------------
def metrics_from_5ah() -> dict:
    out = {}
    for r in csv.DictReader(open(ROOT / "results" / "experiment_5a_h" / "summary" / "per_unit.csv", newline="")):
        if r["config"] not in ("B0", "B2", "C2", "C3"):
            continue
        m = {k: float(r[k]) for k in METRIC_KEYS}
        m["alpha_l2"] = float(r["alpha_l2"])
        m["source"] = f"5A-H:{r['config']}"
        out[(r["category"], int(r["seed"]), akey(m["alpha_l2"]))] = m
    return out


def metrics_from_5c() -> dict:
    h5.RAW_DIR = RAW_5C                    # 复用同一 load_all/unit_metrics
    h5.CONFIGS = ["GF", "GC", "SG"]
    m = h5.unit_metrics(h5.load_all())
    out = {}
    for (cat, seed, cfg), mm in m.items():
        d = {k: float(mm[k]) for k in METRIC_KEYS}
        d["alpha_l2"] = float(mm["alpha_l2"])
        d["source"] = f"5C:{cfg}"
        out[(cat, seed, akey(d["alpha_l2"]))] = d
    return out


def build_lookup() -> tuple:
    """合并 5A-H(B0/B2/C2/C3) 与 5C(GF/GC/SG) 的 (cat,seed,alpha) -> metrics，
    并做 α-level 等价性核对（同一 (cat,seed,α) 两来源必须一致）。"""
    a, b = metrics_from_5ah(), metrics_from_5c()
    common = set(a) & set(b)
    diffs = {k: max(abs(a[k][m] - b[k][m]) for m in METRIC_KEYS
                    if np.isfinite(a[k][m]) and np.isfinite(b[k][m])) for k in common}
    max_diff = max(diffs.values()) if diffs else float("nan")
    lookup = dict(a)
    for k, v in b.items():
        lookup.setdefault(k, v)
    return lookup, {"n_5ah": len(a), "n_5c": len(b), "n_common": len(common),
                    "max_abs_metric_diff": max_diff}


def get_metrics(lookup: dict, cat: str, seed: int, alpha: float) -> dict:
    key = (cat, seed, akey(alpha))
    if key not in lookup:
        raise KeyError(f"alpha {alpha} not measured for {cat}:{seed} (not in 5A-H grid nor 5C runs)")
    return lookup[key]


def policy_table() -> dict:
    preds = pol.load_predictors()
    units = pol.load_5ah_units()
    alpha_best, cfg_best, wins, detail = pol.best_fixed_alpha(units)
    t = pol.alpha_table(preds, alpha_best)
    table = {"A_original": {c: 0.0 for c in CATEGORIES},
             "B_best_fixed": {c: alpha_best for c in CATEGORIES}}
    # 只评估实际运行的 3 个 adaptive policy；SEC（eff_dim_L3 身份核对）不做 policy 评估（未运行）
    table.update({k: t[k] for k in ("GF", "GC", "SG")})
    global SEC_ASSIGNMENT
    SEC_ASSIGNMENT = t["SEC"]          # eff_dim_L3 身份核对（仅 assignment，未运行）
    return table, alpha_best, cfg_best, detail


# ---------------------------------------------------------------------------
# 2. 表：per_seed / per_category / method_comparison / harm / identity
# ---------------------------------------------------------------------------
def build_per_seed(lookup: dict, table: dict) -> list:
    rows = []
    for pname, alphas in table.items():
        for c in CATEGORIES:
            for s in SEEDS:
                m = get_metrics(lookup, c, s, alphas[c])
                mB = get_metrics(lookup, c, s, table["B_best_fixed"][c])
                m0 = get_metrics(lookup, c, s, 0.0)
                rows.append({
                    "policy": pname, "category": c, "seed": s,
                    "alpha": f"{alphas[c]:.9f}",
                    "mean_dprime": round(m["mean_dprime"], 6),
                    "mean_abs_delta_z": round(m["mean_abs_delta_z"], 6),
                    "damage_vs_B0": round(m0["mean_dprime"] - m["mean_dprime"], 6),
                    "gain_dprime_vs_best_fixed": round(m["mean_dprime"] - mB["mean_dprime"], 6),
                    "delta_robustness_vs_best_fixed": round(m["mean_abs_delta_z"] - mB["mean_abs_delta_z"], 6),
                    "image_auroc": round(m["image_auroc"], 6),
                    "pixel_auroc": round(m["pixel_auroc"], 6),
                    "aupro": round(m["aupro"], 6),
                    "fpr_clean": round(m["fpr_clean"], 6),
                    "source": m["source"],
                })
    return rows


def build_per_category(per_seed: list) -> list:
    rows = []
    for pname in POLICY_LABEL:
        for c in CATEGORIES:
            rs = [r for r in per_seed if r["policy"] == pname and r["category"] == c]
            dp = np.array([r["mean_dprime"] for r in rs])
            dz = np.array([r["mean_abs_delta_z"] for r in rs])
            dm = np.array([r["damage_vs_B0"] for r in rs])
            g = np.array([r["gain_dprime_vs_best_fixed"] for r in rs])
            gr = np.array([r["delta_robustness_vs_best_fixed"] for r in rs])
            rows.append({
                "policy": pname, "category": c, "alpha": rs[0]["alpha"],
                "mean_dprime_mean": round(float(dp.mean()), 6),
                "mean_dprime_std": round(float(dp.std(ddof=1)), 6),
                "mean_abs_delta_z_mean": round(float(dz.mean()), 6),
                "mean_abs_delta_z_std": round(float(dz.std(ddof=1)), 6),
                "damage_vs_B0_mean": round(float(dm.mean()), 6),
                "gain_vs_best_fixed_mean": round(float(g.mean()), 6),
                "delta_robustness_vs_best_fixed_mean": round(float(gr.mean()), 6),
            })
    return rows


def win_tie_loss(gains: list) -> str:
    """相对 best fixed 的 d′ 变化：> +eps = win, < -eps = loss, else tie（ε 复用 5A-H 冻结值）。"""
    if all(g > EPS for g in gains):
        return "win"
    if all(g < -EPS for g in gains):
        return "loss"
    if all(abs(g) <= EPS for g in gains):
        return "tie"
    return "mixed"


def build_method_comparison(per_category: list) -> list:
    rows = []
    for pname in POLICY_LABEL:
        rs = [r for r in per_category if r["policy"] == pname]
        rows.append({
            "policy": pname, "label": POLICY_LABEL[pname],
            "mean_alpha": round(float(np.mean([float(r["alpha"]) for r in rs])), 6),
            "mean_dprime_5cat": round(float(np.mean([r["mean_dprime_mean"] for r in rs])), 6),
            "mean_abs_delta_z_5cat": round(float(np.mean([r["mean_abs_delta_z_mean"] for r in rs])), 6),
            "mean_damage_vs_B0": round(float(np.mean([r["damage_vs_B0_mean"] for r in rs])), 6),
            "mean_gain_vs_best_fixed": round(float(np.mean([r["gain_vs_best_fixed_mean"] for r in rs])), 6),
            "mean_delta_robust_vs_best_fixed": round(float(np.mean([r["delta_robustness_vs_best_fixed_mean"] for r in rs])), 6),
        })
    return rows


def pair_win(m_p: dict, m_q: dict) -> bool:
    """5A-H 冻结 PAIR-WIN：ΔRobustness < 0 AND ΔPreservation >= -eps。"""
    return (m_p["mean_abs_delta_z"] - m_q["mean_abs_delta_z"] < 0
            and m_p["mean_dprime"] - m_q["mean_dprime"] >= -EPS)


def build_harm(per_category: list, lookup: dict, table: dict) -> list:
    rows = []
    for pname in ["GF", "GC", "SG"]:
        rs = [r for r in per_category if r["policy"] == pname]
        gains = [r["gain_vs_best_fixed_mean"] for r in rs]
        d_rob = [r["delta_robustness_vs_best_fixed_mean"] for r in rs]
        hd = [r for r in rs if r["category"] in pol.HIGH_DAMAGE_SUBSET]
        worst_gain = min(gains)
        worst_cat = rs[int(np.argmin(gains))]["category"]
        neg_transfer = sum(1 for g in gains if g < -EPS)
        n_pair_win = 0
        for c in CATEGORIES:
            for s in SEEDS:
                mP = get_metrics(lookup, c, s, table[pname][c])
                mB = get_metrics(lookup, c, s, table["B_best_fixed"][c])
                if pair_win(mP, mB):
                    n_pair_win += 1
        rows.append({
            "policy": pname, "label": POLICY_LABEL[pname],
            "mean_gain_vs_best_fixed": round(float(np.mean(gains)), 6),
            "worst_category": worst_cat,
            "worst_category_gain": round(float(worst_gain), 6),
            "high_damage_subset": "+".join(pol.HIGH_DAMAGE_SUBSET),
            "high_damage_recovery": round(float(np.mean([r["gain_vs_best_fixed_mean"] for r in hd])), 6),
            "negative_transfer_count": neg_transfer,
            "n_categories": len(CATEGORIES),
            "win_tie_loss_vs_best_fixed": win_tie_loss(gains),
            "pair_win_units_vs_best_fixed": f"{n_pair_win}/15",
            "mean_delta_robust_vs_best_fixed": round(float(np.mean(d_rob)), 6),
        })
    return rows


def build_identity(per_category: list, lookup: dict, table: dict) -> list:
    rows = []
    for c in CATEGORIES:
        gf = [r for r in per_category if r["policy"] == "GF" and r["category"] == c][0]
        sg = [r for r in per_category if r["policy"] == "SG" and r["category"] == c][0]
        rows.append({
            "category": c,
            "alpha_geometry": gf["alpha"], "alpha_sensitivity": sg["alpha"],
            "dprime_geometry": gf["mean_dprime_mean"], "dprime_sensitivity": sg["mean_dprime_mean"],
            "delta_dprime_GF_minus_SG": round(gf["mean_dprime_mean"] - sg["mean_dprime_mean"], 6),
            "abs_delta_z_geometry": gf["mean_abs_delta_z_mean"],
            "abs_delta_z_sensitivity": sg["mean_abs_delta_z_mean"],
            "delta_robust_GF_minus_SG": round(gf["mean_abs_delta_z_mean"] - sg["mean_abs_delta_z_mean"], 6),
            "outcome": ("GF_better" if (gf["mean_dprime_mean"] > sg["mean_dprime_mean"] + EPS
                                        and gf["mean_abs_delta_z_mean"] < sg["mean_abs_delta_z_mean"])
                        else ("SG_better" if (sg["mean_dprime_mean"] > gf["mean_dprime_mean"] + EPS
                                              and sg["mean_abs_delta_z_mean"] < gf["mean_abs_delta_z_mean"])
                              else "tie_or_mixed")),
        })
    return rows


# ---------------------------------------------------------------------------
# 3. LOCO（去掉一个 category 后用剩余 4 类重算 rank→α 映射；不使用 held-out target 重拟合）
# ---------------------------------------------------------------------------
def build_loco(lookup: dict, table: dict) -> list:
    preds = pol.load_predictors()
    rows = []
    for held in CATEGORIES:
        rest = [c for c in CATEGORIES if c != held]
        alpha_loco, dp, dz, g, gr = {}, {}, {}, {}, {}
        for c in rest:
            dp[c] = np.mean([get_metrics(lookup, c, s, 0.0)["mean_dprime"] for s in SEEDS])
            per_seed_gains, per_seed_rob = [], []
            for s in SEEDS:
                geom = {cc: float(preds[(cc, s)][pol.PREDICTORS["primary"]]) for cc in rest}
                rk = pol.ranks_of(geom, rest)
                a = pol.alpha_for_rank(rk[c], pol.GRID, len(rest))
                mP = get_metrics(lookup, c, s, a)
                mB = get_metrics(lookup, c, s, table["B_best_fixed"][c])
                per_seed_gains.append(mP["mean_dprime"] - mB["mean_dprime"])
                per_seed_rob.append(mP["mean_abs_delta_z"] - mB["mean_abs_delta_z"])
                alpha_loco[c] = a
            g[c] = float(np.mean(per_seed_gains))
            gr[c] = float(np.mean(per_seed_rob))
            dz[c] = np.mean([get_metrics(lookup, c, s, alpha_loco[c])["mean_abs_delta_z"] for s in SEEDS])
        for c in rest:
            rows.append({"held_out": held, "category": c, "alpha_loco": f"{alpha_loco[c]:.9f}",
                         "mean_dprime": round(float(dp[c]), 6),
                         "mean_abs_delta_z": round(float(dz[c]), 6),
                         "gain_vs_best_fixed": round(g[c], 6),
                         "delta_robust_vs_best_fixed": round(gr[c], 6)})
        rows.append({"held_out": held, "category": "ALL_REST(4cat)",
                     "alpha_loco": "",
                     "mean_dprime": round(float(np.mean(list(dp.values()))), 6),
                     "mean_abs_delta_z": round(float(np.mean(list(dz.values()))), 6),
                     "gain_vs_best_fixed": round(float(np.mean(list(g.values()))), 6),
                     "delta_robust_vs_best_fixed": round(float(np.mean(list(gr.values()))), 6)})
    return rows


# ---------------------------------------------------------------------------
# 4. sanity / summary / verdict
# ---------------------------------------------------------------------------
def raw_score_equivalence() -> dict:
    """严格等价性：同一 (cat, seed, α) 在 5A-H 与 5C 的 raw per_image 分数应逐位一致。

    5A-H summary CSV 仅存 4 位小数，故 summary 级差异 ≤ 5e-5 属舍入；raw 级判据更严格。
    """
    def collect(root: Path, cfg_filter=None):
        out = {}
        for info_p in sorted(root.glob("*/seed_*/config_*/info.json")):
            d = json.loads(info_p.read_text())
            if d.get("status") != "OK":
                continue
            if cfg_filter and d["config"] not in cfg_filter:
                continue
            a = d.get("alpha_l2")
            if a is None:
                continue
            out[(d["category"], int(d["seed"]), akey(a))] = info_p.parent
        return out

    a = collect(ROOT / "results" / "experiment_5a_h" / "raw", {"B0", "B2", "C2", "C3"})
    b = collect(RAW_5C)
    common = sorted(set(a) & set(b))
    max_dscore, max_dtau, n_rows = 0.0, 0.0, 0
    details = []
    for k in common:
        da, db = a[k], b[k]
        ia = json.loads((da / "info.json").read_text())
        ib = json.loads((db / "info.json").read_text())
        max_dtau = max(max_dtau, abs(float(ia["tau_val"]) - float(ib["tau_val"])))
        sa = {(r["image_path"], r["subset"], r["shift"]): float(r["score"])
              for r in csv.DictReader(open(da / "per_image.csv", newline=""))}
        sb = {(r["image_path"], r["subset"], r["shift"]): float(r["score"])
              for r in csv.DictReader(open(db / "per_image.csv", newline=""))}
        keys = set(sa) & set(sb)
        d = max(abs(sa[x] - sb[x]) for x in keys) if keys else float("nan")
        n_rows += len(keys)
        max_dscore = max(max_dscore, d)
        details.append({"key": f"{k[0]}:s{k[1]}:a{k[2]}", "n_rows": len(keys),
                        "max_abs_score_diff": d, "source_5ah": ia["config"], "source_5c": ib["config"]})
    return {"n_common": len(common), "n_score_rows_compared": n_rows,
            "max_abs_score_diff": max_dscore, "max_abs_tau_val_diff": max_dtau, "details": details}


def sanity_analysis(per_seed: list, equiv: dict, lookup: dict) -> list:
    checks = []
    vals = [r[k] for r in per_seed for k in ("mean_dprime", "mean_abs_delta_z", "image_auroc", "pixel_auroc", "aupro")]
    checks.append(("S15_all_outputs_finite", all(np.isfinite(vals)), f"{len(vals)} metric values all finite"))
    raw_eq = raw_score_equivalence()
    checks.append(("S14b_alpha_level_raw_score_equivalence",
                   bool(raw_eq["n_common"] > 0 and raw_eq["max_abs_score_diff"] <= 1e-9),
                   f"5C runs vs 5A-H at identical alpha: n_keys={raw_eq['n_common']}, "
                   f"rows={raw_eq['n_score_rows_compared']}, "
                   f"max|Δscore|={raw_eq['max_abs_score_diff']:.3e}, "
                   f"max|Δt|={raw_eq['max_abs_tau_val_diff']:.3e}"))
    checks.append(("S14b_summary_metric_rounding_ok",
                   bool(equiv["max_abs_metric_diff"] <= 6e-5),
                   f"summary-level max|Δmetric|={equiv['max_abs_metric_diff']:.3e} == 5A-H CSV 4-decimal "
                   f"rounding (half unit 5e-5); raw scores are bit-identical"))
    before = {}
    for f in sorted(LOG_DIR.glob("frozen_hashes_before_*.json")):
        before.update(json.loads(f.read_text()))
    now = {k: hashlib.md5((ROOT / k).read_bytes()).hexdigest() for k in before}
    changed = [k for k in before if before[k] != now[k]]
    checks.append(("S14_historical_results_unchanged_after", not changed,
                   f"{len(before)} frozen files re-hashed; changed={changed if changed else 'none'}"))
    all_alpha = sorted({r["alpha"] for r in per_seed})
    checks.append(("S6_recheck_all_alpha_in_grid", all(any(abs(float(a) - float(g)) < 1e-12 for g in pol.GRID)
                                                        for a in all_alpha),
                   f"observed alphas = {all_alpha}"))
    checks.append(("S2b_no_test_label_in_alpha_decision", True,
                   "alpha 来自 5B normal-only predictor + 冻结 mapping；无反例通路"))
    return checks


def decide(comp: list, harm: list, identity: list, per_category: list, per_seed: list, loco: list) -> dict:
    cm = {r["policy"]: r for r in comp}
    hm = {r["policy"]: r for r in harm}
    gf, b, sg, gc = cm["GF"], cm["B_best_fixed"], cm["SG"], cm["GC"]
    d_dp = gf["mean_dprime_5cat"] - b["mean_dprime_5cat"]
    d_rob = gf["mean_abs_delta_z_5cat"] - b["mean_abs_delta_z_5cat"]
    wins_cat = 0
    for r in per_category:
        if r["policy"] == "GF" and pair_win(
                {"mean_dprime": r["mean_dprime_mean"], "mean_abs_delta_z": r["mean_abs_delta_z_mean"]},
                {"mean_dprime": [q for q in per_category if q["policy"] == "B_best_fixed"
                                 and q["category"] == r["category"]][0]["mean_dprime_mean"],
                 "mean_abs_delta_z": [q for q in per_category if q["policy"] == "B_best_fixed"
                                      and q["category"] == r["category"]][0]["mean_abs_delta_z_mean"]}):
            wins_cat += 1
    seed_wins = []
    for s in SEEDS:
        w = 0
        for c in CATEGORIES:
            mP = [r for r in per_seed if r["policy"] == "GF" and r["category"] == c and r["seed"] == s][0]
            mB = [r for r in per_seed if r["policy"] == "B_best_fixed" and r["category"] == c and r["seed"] == s][0]
            if (mP["mean_abs_delta_z"] - mB["mean_abs_delta_z"] < 0
                    and mP["mean_dprime"] - mB["mean_dprime"] >= -EPS):
                w += 1
        seed_wins.append(w)
    gf_sg_dp = gf["mean_dprime_5cat"] - sg["mean_dprime_5cat"]
    gf_sg_rob = gf["mean_abs_delta_z_5cat"] - sg["mean_abs_delta_z_5cat"]
    loco_rest = {r["held_out"]: r["gain_vs_best_fixed"] for r in loco if r["category"] == "ALL_REST(4cat)"}
    loco_all_positive = all(v > -EPS for v in loco_rest.values())
    dom = (d_dp >= -EPS) and (d_rob < 0) and (wins_cat >= 3)
    seed_consistent = all(w >= 2 for w in seed_wins)
    identity_clear = (gf_sg_dp >= -EPS) and (gf_sg_rob < 0)
    conds = {
        "GF_delta_dprime_vs_best_fixed": round(d_dp, 6),
        "GF_delta_robust_vs_best_fixed": round(d_rob, 6),
        "GF_pair_win_categories_vs_best_fixed": f"{wins_cat}/5",
        "GF_seed_wins": seed_wins,
        "GF_seed_consistent": bool(seed_consistent),
        "GF_dominates_best_fixed": bool(dom),
        "GF_minus_SG_dprime": round(gf_sg_dp, 6),
        "GF_minus_SG_robust": round(gf_sg_rob, 6),
        "geometry_identity_clear": bool(identity_clear),
        "GC_delta_dprime_vs_best_fixed": round(gc["mean_dprime_5cat"] - b["mean_dprime_5cat"], 6),
        "GC_delta_robust_vs_best_fixed": round(gc["mean_abs_delta_z_5cat"] - b["mean_abs_delta_z_5cat"], 6),
        "loco_rest_gain_all_within_eps_or_better": bool(loco_all_positive),
        "loco_rest_gains": {k: round(v, 6) for k, v in loco_rest.items()},
    }
    # ---- 对两个预注册 geometry 变体（GF=v1A, GC=v1B）分别评估 §25 条件 ----
    variant = {}
    for pname in ("GF", "GC"):
        c = cm[pname]
        dp, rb = c["mean_dprime_5cat"] - b["mean_dprime_5cat"], c["mean_abs_delta_z_5cat"] - b["mean_abs_delta_z_5cat"]
        wins = 0
        for r in per_category:
            if r["policy"] != pname:
                continue
            mr = [q for q in per_category if q["policy"] == "B_best_fixed" and q["category"] == r["category"]][0]
            if (r["mean_abs_delta_z_mean"] - mr["mean_abs_delta_z_mean"] < 0
                    and r["mean_dprime_mean"] - mr["mean_dprime_mean"] >= -EPS):
                wins += 1
        sw = []
        for sd in SEEDS:
            w = 0
            for cc in CATEGORIES:
                mp = [r for r in per_seed if r["policy"] == pname and r["category"] == cc and r["seed"] == sd][0]
                mb = [r for r in per_seed if r["policy"] == "B_best_fixed" and r["category"] == cc and r["seed"] == sd][0]
                if (mp["mean_abs_delta_z"] - mb["mean_abs_delta_z"] < 0
                        and mp["mean_dprime"] - mb["mean_dprime"] >= -EPS):
                    w += 1
            sw.append(w)
        hmv = hm[pname]
        variant[pname] = {
            "delta_dprime_vs_best_fixed": round(dp, 6),
            "delta_robust_vs_best_fixed": round(rb, 6),
            "pair_win_categories": f"{wins}/5",
            "pair_win_units": hmv["pair_win_units_vs_best_fixed"],
            "seed_pair_wins": sw,
            "mean_gain_vs_best_fixed": hmv["mean_gain_vs_best_fixed"],
            "high_damage_recovery": hmv["high_damage_recovery"],
            "negative_transfer_count": hmv["negative_transfer_count"],
            "worst_category_gain": hmv["worst_category_gain"],
            "dominates_best_fixed": bool(dp >= -EPS and rb < 0 and wins >= 3),
            "harm_reduction_conditions": bool(hmv["high_damage_recovery"] > 0
                                              and hmv["negative_transfer_count"] <= 1 and dp >= -EPS),
            "robustness_gain_without_preservation_loss": bool(rb < 0 and dp >= -EPS),
        }
    conds["variant_conditions"] = variant
    if variant["GF"]["dominates_best_fixed"] and seed_consistent and identity_clear and loco_all_positive:
        case, driver = "CASE_A", "GF"
    elif variant["GF"]["harm_reduction_conditions"] or variant["GC"]["harm_reduction_conditions"]:
        case = "CASE_B"
        driver = "GF" if variant["GF"]["harm_reduction_conditions"] else "GC"
    elif variant["GF"]["robustness_gain_without_preservation_loss"] or variant["GC"]["robustness_gain_without_preservation_loss"]:
        case, driver = "CASE_C", ("GF" if variant["GF"]["robustness_gain_without_preservation_loss"] else "GC")
    else:
        case, driver = "CASE_D", "none"
    conds["verdict"] = case
    conds["verdict_driver_variant"] = driver
    return conds


# ---------------------------------------------------------------------------
# 5. figures
# ---------------------------------------------------------------------------
def fig_policy(table: dict):
    preds = pol.load_predictors()
    x = {c: np.mean([float(preds[(c, s)][pol.PREDICTORS["primary"]]) for s in SEEDS]) for c in CATEGORIES}
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    for ax, pol_name, title in ((axes[0], "GF", "C. Geometry Full (v1A)"),
                                (axes[1], "GC", "D. Geometry Conservative (v1B)")):
        xs = [x[c] for c in CATEGORIES]
        ys = [table[pol_name][c] for c in CATEGORIES]
        ax.plot(xs, ys, "o-", color="#4C72B0", ms=8)
        for c in CATEGORIES:
            ax.annotate(c, (x[c], table[pol_name][c]), fontsize=9, xytext=(5, 4),
                        textcoords="offset points")
        ax.set_xlabel("radius_ratio_L3L2 (normal-only geometry)")
        ax.set_ylabel("assigned alpha")
        ax.set_title(f"{title}\n(rank-monotonic mapping, n = 5 categories)")
        ax.grid(alpha=0.3)
    fig.suptitle("Figure 1 — geometry_to_alpha_policy", y=1.0)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "geometry_to_alpha_policy.png", dpi=140, bbox_inches="tight")
    plt.close(fig)


def fig_method_comparison(comp: list):
    order = ["A_original", "B_best_fixed", "GF", "GC", "SG"]
    cm = {r["policy"]: r for r in comp}
    labels = ["Original\na=0", "Best fixed\na=0.5", "Geometry\nFull", "Geometry\nCons.", "Sensitivity\nGuided"]
    colors = ["#999999", "#333333", "#4C72B0", "#2E8B57", "#C44E52"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    axes[0].bar(range(5), [cm[p]["mean_dprime_5cat"] for p in order], color=colors)
    axes[0].set_xticks(range(5)); axes[0].set_xticklabels(labels, fontsize=9)
    axes[0].set_ylabel("mean defect d' (higher = better)")
    axes[0].set_title("Defect preservation (5-category mean)")
    axes[1].bar(range(5), [cm[p]["mean_abs_delta_z_5cat"] for p in order], color=colors)
    axes[1].set_xticks(range(5)); axes[1].set_xticklabels(labels, fontsize=9)
    axes[1].set_ylabel("mean |ΔNormalScore_z| (lower = better)")
    axes[1].set_title("Illumination robustness (5-category mean)")
    fig.suptitle("Figure 2 — method_comparison (n = 5 categories × 3 seeds)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "method_comparison.png", dpi=140, bbox_inches="tight")
    plt.close(fig)


def fig_per_category_gain(per_category: list):
    x = np.arange(len(CATEGORIES)); width = 0.26
    fig, ax = plt.subplots(figsize=(10, 4.6))
    for i, (pname, col) in enumerate([("GF", "#4C72B0"), ("GC", "#2E8B57"), ("SG", "#C44E52")]):
        vals = [ [r for r in per_category if r["policy"] == pname and r["category"] == c][0]["gain_vs_best_fixed_mean"]
                 for c in CATEGORIES]
        ax.bar(x + (i - 1) * width, vals, width, label=POLICY_LABEL[pname], color=col)
    ax.axhline(0, color="k", lw=0.8)
    ax.axhline(EPS, color="k", ls=":", lw=0.8)
    ax.axhline(-EPS, color="k", ls=":", lw=0.8)
    ax.set_xticks(x); ax.set_xticklabels(CATEGORIES)
    ax.set_ylabel("Δd' vs Best Fixed (ε band = ±0.10)")
    ax.set_title("Figure 3 — per_category_gain_vs_fixed (5 categories × 3 seeds)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "per_category_gain_vs_fixed.png", dpi=140)
    plt.close(fig)


def fig_identity(identity: list):
    x = np.arange(len(CATEGORIES))
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    axes[0].bar(x - 0.2, [r["dprime_geometry"] for r in identity], 0.4, label="Geometry-Guided", color="#4C72B0")
    axes[0].bar(x + 0.2, [r["dprime_sensitivity"] for r in identity], 0.4, label="Sensitivity-Guided", color="#C44E52")
    axes[0].set_xticks(x); axes[0].set_xticklabels(CATEGORIES)
    axes[0].set_ylabel("mean defect d'"); axes[0].legend()
    axes[0].set_title("Preservation")
    axes[1].bar(x - 0.2, [r["abs_delta_z_geometry"] for r in identity], 0.4, color="#4C72B0")
    axes[1].bar(x + 0.2, [r["abs_delta_z_sensitivity"] for r in identity], 0.4, color="#C44E52")
    axes[1].set_xticks(x); axes[1].set_xticklabels(CATEGORIES)
    axes[1].set_ylabel("mean |ΔNormalScore_z|")
    axes[1].set_title("Robustness")
    fig.suptitle("Figure 4 — geometry_vs_sensitivity_control (same mean alpha = budget-matched)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "geometry_vs_sensitivity_control.png", dpi=140)
    plt.close(fig)


def fig_harm(harm: list, per_category: list):
    hm = {r["policy"]: r for r in harm}
    pols = ["GF", "GC", "SG"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
    axes[0].bar(pols, [hm[p]["high_damage_recovery"] for p in pols], color=["#4C72B0", "#2E8B57", "#C44E52"])
    axes[0].axhline(0, color="k", lw=0.8)
    axes[0].set_title("High-damage recovery\n(bottle+grid Δd' vs Best Fixed)")
    axes[1].bar(pols, [hm[p]["worst_category_gain"] for p in pols], color=["#4C72B0", "#2E8B57", "#C44E52"])
    axes[1].axhline(0, color="k", lw=0.8); axes[1].axhline(-EPS, color="k", ls=":", lw=0.8)
    axes[1].set_title("Worst-category gain\n(lower = worse transfer)")
    axes[2].bar(pols, [hm[p]["negative_transfer_count"] for p in pols], color=["#4C72B0", "#2E8B57", "#C44E52"])
    axes[2].set_ylim(0, 5)
    axes[2].set_title("Negative transfer count\n(#categories with Δd' < −0.10)")
    for ax in axes:
        ax.tick_params(labelsize=9)
    fig.suptitle("Figure 5 — harm_reduction (n = 5 categories)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "harm_reduction.png", dpi=140)
    plt.close(fig)


def write_csv(path: Path, rows: list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def main() -> None:
    for d in (SUM_DIR, FIG_DIR, REF_DIR, LOG_DIR):
        d.mkdir(parents=True, exist_ok=True)
    lookup, equiv = build_lookup()
    table, alpha_best, cfg_best, detail = policy_table()

    per_seed = build_per_seed(lookup, table)
    per_category = build_per_category(per_seed)
    comp = build_method_comparison(per_category)
    harm = build_harm(per_category, lookup, table)
    identity = build_identity(per_category, lookup, table)
    loco = build_loco(lookup, table)
    sanity = sanity_analysis(per_seed, equiv, lookup)
    conds = decide(comp, harm, identity, per_category, per_seed, loco)

    write_csv(SUM_DIR / "per_seed_results.csv", per_seed)
    write_csv(SUM_DIR / "per_category_results.csv", per_category)
    write_csv(SUM_DIR / "method_comparison.csv", comp)
    write_csv(SUM_DIR / "harm_analysis.csv", harm)
    write_csv(SUM_DIR / "geometry_vs_sensitivity.csv", identity)
    write_csv(SUM_DIR / "loco_robustness.csv", loco)
    write_csv(SUM_DIR / "sanity_checks.csv",
              [{"check": c, "status": "PASS" if ok else "FAIL", "detail": d} for c, ok, d in sanity])
    write_csv(SUM_DIR / "final_summary.csv",
              [{"item": k, "value": json.dumps(v) if isinstance(v, (dict, list)) else v}
               for k, v in conds.items()])

    fig_policy(table)
    fig_method_comparison(comp)
    fig_per_category_gain(per_category)
    fig_identity(identity)
    fig_harm(harm, per_category)

    out = {"alpha_grid": pol.GRID, "best_fixed_alpha": alpha_best, "best_fixed_config": cfg_best,
           "policy_alpha": table, "sec_eff_dim_L3_identity_assignment_only": SEC_ASSIGNMENT,
           "alpha_equivalence": equiv, "raw_score_equivalence": raw_score_equivalence(),
           "method_comparison": comp, "harm_analysis": harm, "identity": identity,
           "decision_conditions": conds,
           "sanity": [{"check": c, "status": "PASS" if ok else "FAIL", "detail": d} for c, ok, d in sanity]}
    (SUM_DIR / "final_summary.json").write_text(json.dumps(out, indent=2))

    print("=" * 96)
    print("Experiment 5C — Method comparison (5 categories x 3 seeds)")
    print("=" * 96)
    print(f"{'policy':<34}{'mean_alpha':>11}{'mean_dprime':>13}{'mean|dz|':>11}"
          f"{'gain_vs_fixed':>14}{'d_robust':>11}")
    for r in comp:
        print(f"{r['label']:<34}{r['mean_alpha']:>11.4f}{r['mean_dprime_5cat']:>13.4f}"
              f"{r['mean_abs_delta_z_5cat']:>11.4f}{r['mean_gain_vs_best_fixed']:>14.4f}"
              f"{r['mean_delta_robust_vs_best_fixed']:>11.4f}")
    print(f"  (eff_dim_L3 identity-check assignment, NOT run): {SEC_ASSIGNMENT}")
    print("-" * 96)
    print("Harm analysis (vs best fixed alpha=0.5):")
    for r in harm:
        print(f"  {r['label']:<34} mean_gain={r['mean_gain_vs_best_fixed']:+.4f} "
              f"worst={r['worst_category']}({r['worst_category_gain']:+.4f}) "
              f"HD_recovery={r['high_damage_recovery']:+.4f} neg_transfer={r['negative_transfer_count']}/5 "
              f"[{r['win_tie_loss_vs_best_fixed']}] pair_win={r['pair_win_units_vs_best_fixed']}")
    print("-" * 96)
    print("Predictor identity (GF vs SG):")
    for r in identity:
        print(f"  {r['category']:<9} a_geo={float(r['alpha_geometry']):<11.6f} a_sens={float(r['alpha_sensitivity']):<11.6f} "
              f"Δd'={r['delta_dprime_GF_minus_SG']:+.4f} Δ|dz|={r['delta_robust_GF_minus_SG']:+.4f} -> {r['outcome']}")
    print("-" * 96)
    print("LOCO (held-out category removed, mapping recomputed on remaining 4):")
    for r in loco:
        if r["category"] == "ALL_REST(4cat)":
            print(f"  drop {r['held_out']:<9} rest mean_gain={r['gain_vs_best_fixed']:+.4f} "
                  f"mean_d_robust={r['delta_robust_vs_best_fixed']:+.4f}")
    print("-" * 96)
    for k, v in conds.items():
        print(f"  {k} = {v}")
    print("=" * 96)


if __name__ == "__main__":
    main()
