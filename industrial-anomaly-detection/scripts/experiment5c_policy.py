"""Experiment 5C — Geometry-Guided Category-Adaptive α v1：policy 冻结模块。

职责（在读取任何 5C target 结果之前完成）：
  1. 机械恢复 historical α grid（5A-H 全覆盖的 uniform α）
  2. 机械确定 best fixed α（复用 5A-H 冻结 PAIR-WIN 规则，ε=0.10，零新参数）
  3. 生成 rank→α mapping 与三个 policy 的 category α 分配
  4. 写 reference CSV + policy_assignment.csv + geometry_policy_freeze.json（并计算 sha256）

本模块只读 5B-Final（predictors）与 5A-H（historical fixed α）的冻结结果，不读任何 5C 运行结果。
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "experiment_5c"
REF_DIR = OUT / "reference"
SUM_DIR = OUT / "summary"
LOG_DIR = OUT / "logs"

PER_UNIT_5AH = ROOT / "results" / "experiment_5a_h" / "summary" / "per_unit.csv"
PREDICTORS_5B = ROOT / "results" / "experiment_5b_final" / "summary" / "normal_only_predictors.csv"
CORR_5B = ROOT / "results" / "experiment_5b_final" / "summary" / "correlation_summary.csv"

CATEGORIES = ["bottle", "cable", "grid", "hazelnut", "screw"]
SEEDS = [0, 1, 2]

# 5A-H exact α 值（与 5A-H runner 常量一致）
RATIO = 0.603651
A2_G2 = 0.75 * RATIO
A_C2 = (A2_G2 + 0.75) / 2.0
A_C3 = (RATIO * 1.0 + 1.0) / 2.0

# historical α grid：5A-H 中 5 categories × 3 seeds 全覆盖的 uniform α
GRID = [0.0, 0.5, A_C2, A_C3]
GRID_SOURCE = {"0.0": "B0", "0.5": "B2", "0.601369125": "C2", "0.8018255": "C3"}
FIXED_CANDIDATES = ["B2", "C2", "C3"]          # B0 是 reference；G2 非 uniform 故排除
EPS = 0.10                                     # 5A-H §2 冻结的 d′ 非劣带
PREDICTORS = {"primary": "radius_ratio_L3L2", "secondary": "eff_dim_L3", "control": "sens_radius_L2"}
CONSERVATIVE_MAX = 0.5                         # 占位；由 best_fixed_alpha() 结果覆盖

HIGH_DAMAGE_SUBSET = ["bottle", "grid"]        # 预指定（依据 5A-H/5B 历史 damage，不看 5C 结果）
FROZEN_FILES = [
    PER_UNIT_5AH, PREDICTORS_5B, CORR_5B,
    ROOT / "results" / "experiment_5b_final" / "summary" / "final_summary.json",
    ROOT / "results" / "experiment_5b_final" / "reference" / "group_c_freeze.json",
    ROOT / "results" / "experiment_5a_h" / "summary" / "statistical_tests.csv",
]


def md5_file(p: Path) -> str:
    return hashlib.md5(p.read_bytes()).hexdigest()


def sha256_file(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def load_5ah_units() -> dict:
    units = {}
    for r in csv.DictReader(open(PER_UNIT_5AH, newline="")):
        units[(r["category"], int(r["seed"]), r["config"])] = r
    return units


def load_predictors() -> dict:
    out = {}
    for r in csv.DictReader(open(PREDICTORS_5B, newline="")):
        out[(r["category"], int(r["seed"]))] = r
    return out


def best_fixed_alpha(units: dict) -> tuple:
    """冻结规则：统计每个 uniform α 用 5A-H PAIR-WIN 规则击败 α=0 的 unit 数；
    argmax（tie → 更小 α）。返回 (alpha, config, wins, detail)。"""
    detail = {}
    for cfg in FIXED_CANDIDATES:
        wins = 0
        per_cat = {}
        for c in CATEGORIES:
            w = 0
            for s in SEEDS:
                b, a = units[(c, s, "B0")], units[(c, s, cfg)]
                d_rob = float(a["mean_abs_delta_z"]) - float(b["mean_abs_delta_z"])
                d_pre = float(a["mean_dprime"]) - float(b["mean_dprime"])
                if d_rob < 0 and d_pre >= -EPS:
                    w += 1
            per_cat[c] = w
            wins += w
        alpha = float(units[(CATEGORIES[0], SEEDS[0], cfg)]["alpha_l2"])
        detail[cfg] = {"alpha": alpha, "wins_units": wins, "wins_per_category": per_cat}
    best_cfg = sorted(detail, key=lambda k: (-detail[k]["wins_units"], detail[k]["alpha"]))[0]
    return detail[best_cfg]["alpha"], best_cfg, detail[best_cfg]["wins_units"], detail


def write_fixed_alpha_reference(units: dict, best: dict) -> None:
    REF_DIR.mkdir(parents=True, exist_ok=True)
    fields = ["config", "alpha", "is_best_fixed", "wins_vs_B0_units",
              "mean_dprime_5cat", "mean_abs_delta_z_5cat", "damage_5cat_mean",
              "category", "seed", "mean_dprime", "mean_abs_delta_z", "damage_vs_B0", "source_file"]
    rows = []
    for cfg in ["B0"] + FIXED_CANDIDATES + ["G2"]:
        rs = [r for r in units.values() if r["config"] == cfg]
        alpha = float(rs[0]["alpha_l2"])
        uniform = abs(float(rs[0]["alpha_l2"]) - float(rs[0]["alpha_l3"])) < 1e-12
        for r in rs:
            c, s = r["category"], int(r["seed"])
            b = units[(c, s, "B0")]
            dp = float(r["mean_dprime"])
            rows.append({
                "config": cfg, "alpha": "" if not uniform else f"{alpha:.9f}",
                "is_best_fixed": "true" if cfg == best["config"] else "false",
                "wins_vs_B0_units": best["wins_units"] if cfg == best["config"] else "",
                "mean_dprime_5cat": f"{np.mean([float(x['mean_dprime']) for x in rs]):.6f}",
                "mean_abs_delta_z_5cat": f"{np.mean([float(x['mean_abs_delta_z']) for x in rs]):.6f}",
                "damage_5cat_mean": f"{np.mean([float(units[(x['category'], int(x['seed']), 'B0')]['mean_dprime']) - float(x['mean_dprime']) for x in rs]):.6f}",
                "category": c, "seed": s, "mean_dprime": r["mean_dprime"],
                "mean_abs_delta_z": r["mean_abs_delta_z"],
                "damage_vs_B0": f"{float(b['mean_dprime']) - dp:.6f}",
                "source_file": str(PER_UNIT_5AH.relative_to(ROOT)),
            })
    # 按 config 汇总行（category/seed 留空）便于直接读 “哪个 α 是 best fixed”
    summ = []
    for cfg in ["B0"] + FIXED_CANDIDATES + ["G2"]:
        rs = [r for r in units.values() if r["config"] == cfg]
        alpha = float(rs[0]["alpha_l2"])
        uniform = abs(float(rs[0]["alpha_l2"]) - float(rs[0]["alpha_l3"])) < 1e-12
        summ.append({
            "config": cfg, "alpha": "" if not uniform else f"{alpha:.9f}",
            "is_best_fixed": "true" if cfg == best["config"] else "false",
            "wins_vs_B0_units": best["detail"][cfg]["wins_units"] if cfg in best["detail"] else "",
            "mean_dprime_5cat": f"{np.mean([float(x['mean_dprime']) for x in rs]):.6f}",
            "mean_abs_delta_z_5cat": f"{np.mean([float(x['mean_abs_delta_z']) for x in rs]):.6f}",
            "damage_5cat_mean": f"{np.mean([float(units[(x['category'], int(x['seed']), 'B0')]['mean_dprime']) - float(x['mean_dprime']) for x in rs]):.6f}",
            "category": "", "seed": "", "mean_dprime": "", "mean_abs_delta_z": "",
            "damage_vs_B0": "", "source_file": str(PER_UNIT_5AH.relative_to(ROOT)),
        })
    with open(REF_DIR / "fixed_alpha_reference.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(summ + rows)


# ---------------------------------------------------------------------------
# rank → α mapping（冻结：rank-based monotonic, nearest-rank quantile）
# ---------------------------------------------------------------------------
def ranks_of(values: dict, cats: list) -> dict:
    """升序 rank（r=1 最小 predictor 值 = 最脆弱/最不耐受）。"""
    order = sorted(cats, key=lambda c: values[c])
    return {c: i + 1 for i, c in enumerate(order)}


def alpha_for_rank(rank: int, legal: list, n: int) -> float:
    """α(r) = legal[min(K-1, ceil(r*K/n)-1)]；单调不减、deterministic。"""
    k = len(legal)
    idx = min(k - 1, math.ceil(rank * k / n) - 1)
    return float(legal[idx])


def policy_assignments(preds: dict, best_alpha: float) -> list:
    """返回 policy_assignment 行（每 category×seed）。GF=primary full; GC=primary ceiling;
    SG=control full; 单一 mapping 函数用于三者。"""
    legal_full = list(GRID)
    legal_cons = [a for a in GRID if a <= best_alpha + 1e-12]
    rows = []
    for s in SEEDS:
        geom = {c: float(preds[(c, s)][PREDICTORS["primary"]]) for c in CATEGORIES}
        sec = {c: float(preds[(c, s)][PREDICTORS["secondary"]]) for c in CATEGORIES}
        ctrl = {c: float(preds[(c, s)][PREDICTORS["control"]]) for c in CATEGORIES}
        r_g, r_s, r_c = ranks_of(geom, CATEGORIES), ranks_of(sec, CATEGORIES), ranks_of(ctrl, CATEGORIES)
        for c in CATEGORIES:
            rows.append({
                "category": c, "seed": s,
                "radius_ratio_L3L2": f"{geom[c]:.6f}", "geometry_rank": r_g[c],
                "geometry_alpha_full": f"{alpha_for_rank(r_g[c], legal_full, len(CATEGORIES)):.9f}",
                "geometry_alpha_conservative": f"{alpha_for_rank(r_g[c], legal_cons, len(CATEGORIES)):.9f}",
                "sens_radius_L2": f"{ctrl[c]:.6f}", "sensitivity_rank": r_c[c],
                "sensitivity_alpha": f"{alpha_for_rank(r_c[c], legal_full, len(CATEGORIES)):.9f}",
                "eff_dim_L3": f"{sec[c]:.6f}", "eff_dim_rank": r_s[c],
                "eff_dim_alpha_identity_check": f"{alpha_for_rank(r_s[c], legal_full, len(CATEGORIES)):.9f}",
            })
    return rows


def write_policy_assignment(rows: list) -> None:
    SUM_DIR.mkdir(parents=True, exist_ok=True)
    with open(SUM_DIR / "policy_assignment.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def alpha_table(preds: dict, best_alpha: float) -> dict:
    """{policy: {category: alpha}}（逐 seed 一致时取 seed0 并校验三 seed 相同）。"""
    rows = policy_assignments(preds, best_alpha)
    legal_full = list(GRID)
    legal_cons = [a for a in GRID if a <= best_alpha + 1e-12]
    out = {"GF": {}, "GC": {}, "SG": {}, "SEC": {}}
    rank_key = {"GF": "geometry_rank", "GC": "geometry_rank",
                "SG": "sensitivity_rank", "SEC": "eff_dim_rank"}
    legal = {"GF": legal_full, "GC": legal_cons, "SG": legal_full, "SEC": legal_full}
    for pol in out:
        for c in CATEGORIES:
            rk = [int(r[rank_key[pol]]) for r in rows if r["category"] == c]
            assert len(set(rk)) == 1, f"{pol}:{c} rank 逐 seed 不一致 {rk}"
            out[pol][c] = alpha_for_rank(rk[0], legal[pol], len(CATEGORIES))
    return out


def write_freeze(best: dict, table: dict, preds: dict) -> dict:
    REF_DIR.mkdir(parents=True, exist_ok=True)
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                          capture_output=True, text=True).stdout.strip()
    freeze = {
        "primary_predictor": PREDICTORS["primary"],
        "secondary_predictor": PREDICTORS["secondary"],
        "negative_control_predictor": PREDICTORS["control"],
        "mapping_type": "rank_monotonic_nearest_rank_quantile",
        "mapping_formula": "alpha(r) = legal[min(K-1, ceil(r*K/n)-1)], r = ascending predictor rank, n = 5",
        "direction": "higher_geometry_to_higher_alpha",
        "alpha_grid": GRID,
        "alpha_grid_source": GRID_SOURCE,
        "alpha_grid_excluded": {
            "values": [0.25, 0.75, 1.0],
            "reason": "5A-only (bottle/seed0)，无 5-category × 3-seed 覆盖；纳入会需要新跑 baseline 且不公平",
        },
        "best_fixed_alpha": best["alpha"],
        "best_fixed_config": best["config"],
        "best_fixed_rule": ("argmax over uniform alpha in {B2,C2,C3} of #units satisfying the frozen 5A-H "
                           "PAIR-WIN criterion vs alpha=0 (dRobust<0 AND dPres>=-eps, eps=0.10); tie -> smaller alpha"),
        "best_fixed_wins_units": best["wins_units"],
        "conservative_ceiling": best["alpha"],
        "conservative_ceiling_rule": "alpha_max = best fixed alpha (no new parameter)",
        "epsilon_preservation_band": EPS,
        "eps_source": "Experiment 5A-H section 2 (frozen d' non-inferiority band)",
        "policies": {
            "A_original": {"alpha": 0.0, "source": "5A-H B0 (reused, not re-run)"},
            "B_best_fixed": {"alpha": best["alpha"], "source": f"5A-H {best['config']} (reused, not re-run)"},
            "C_geometry_full": {"predictor": PREDICTORS["primary"], "alpha_by_category": table["GF"]},
            "D_geometry_conservative": {"predictor": PREDICTORS["primary"], "alpha_by_category": table["GC"]},
            "E_sensitivity_guided": {"predictor": PREDICTORS["control"], "alpha_by_category": table["SG"]},
            "secondary_identity_only_no_run": {"predictor": PREDICTORS["secondary"],
                                               "alpha_by_category": table["SEC"]},
        },
        "high_damage_subset_frozen": HIGH_DAMAGE_SUBSET,
        "categories": CATEGORIES,
        "seeds": SEEDS,
        "target_results_read": False,
        "git_head": head,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "input_md5": {str(p.relative_to(ROOT)): md5_file(p) for p in FROZEN_FILES},
    }
    fp = REF_DIR / "geometry_policy_freeze.json"
    fp.write_text(json.dumps(freeze, indent=2))
    digest = sha256_file(fp)
    (REF_DIR / "geometry_policy_freeze.sha256").write_text(digest + "  geometry_policy_freeze.json\n")
    freeze["sha256_of_this_file"] = digest
    return freeze


def main() -> None:
    for d in (REF_DIR, SUM_DIR, LOG_DIR):
        d.mkdir(parents=True, exist_ok=True)
    units = load_5ah_units()
    preds = load_predictors()
    assert len(units) == 75, len(units)
    assert len(preds) == 15, len(preds)

    alpha_best, cfg_best, wins_best, detail = best_fixed_alpha(units)
    best = {"alpha": alpha_best, "config": cfg_best, "wins_units": wins_best, "detail": detail}
    write_fixed_alpha_reference(units, best)

    rows = policy_assignments(preds, alpha_best)
    write_policy_assignment(rows)
    table = alpha_table(preds, alpha_best)
    freeze = write_freeze(best, table, preds)

    # ---- mapping 性质自检（S3/S4/S5/S6）----
    assert all(alpha_for_rank(r, GRID, 5) <= alpha_for_rank(r + 1, GRID, 5) + 1e-12 for r in range(1, 5))
    assert [alpha_for_rank(r, GRID, 5) for r in range(1, 6)] == [0.0, 0.5, A_C2, A_C3, A_C3]
    assert all(abs(float(r["geometry_alpha_full"]) - alpha_for_rank(int(r["geometry_rank"]), GRID, 5)) < 1e-12
               for r in rows)
    assert all(float(v) in [float(g) for g in GRID] for v in table["GF"].values())
    assert all(float(v) in [0.0, 0.5] for v in table["GC"].values())

    print("=" * 78)
    print("[5C PRE-RUN PLAN] Geometry-Guided Category-Adaptive alpha v1")
    print("=" * 78)
    print(f"historical alpha grid (5A-H, 5catx3seed coverage) : {GRID}")
    print(f"excluded (5A-only bottle/seed0)                    : [0.25, 0.75, 1.0]")
    print(f"best fixed alpha (frozen PAIR-WIN rule, eps=0.10)  : {alpha_best}  [{cfg_best}]")
    for cfg, d in sorted(detail.items(), key=lambda kv: -kv[1]["wins_units"]):
        print(f"    {cfg:<3} alpha={d['alpha']:<10.6f} wins_vs_B0={d['wins_units']:>2}/15  per-cat={d['wins_per_category']}")
    print(f"conservative ceiling = best fixed alpha            : {alpha_best}")
    print(f"mapping: alpha(r) = grid[min(K-1, ceil(r*K/n)-1)], r=ascending rank, n=5")
    print(f"  full-range (K=4) ranks 1..5 -> {[alpha_for_rank(r, GRID, 5) for r in range(1, 6)]}")
    print(f"  conservative (K=2) ranks 1..5 -> {[alpha_for_rank(r, [0.0, 0.5], 5) for r in range(1, 6)]}")
    print("-" * 78)
    print(f"{'policy':<26} " + " ".join(f"{c[:5]:>9}" for c in CATEGORIES) + "   mean_alpha")
    for pol, name in [("GF", "C. Geometry Full"), ("GC", "D. Geometry Cons."),
                      ("SG", "E. Sensitivity"), ("SEC", "  (eff_dim_L3 id-check, not run)")]:
        vals = [table[pol][c] for c in CATEGORIES]
        print(f"{name:<26} " + " ".join(f"{v:9.6f}" for v in vals) + f"   {np.mean(vals):.4f}")
    print(f"{'A. Original (reuse B0)':<26} " + " ".join(f"{0.0:9.6f}" for _ in CATEGORIES) + "   0.0000")
    print(f"{'B. Best Fixed (reuse B2)':<26} " + " ".join(f"{alpha_best:9.6f}" for _ in CATEGORIES) + f"   {alpha_best:.4f}")
    print("-" * 78)
    print(f"runs: C/D/E = 3 policies x 5 categories x 3 seeds = 45 config runs (A/B reused from 5A-H)")
    print(f"freeze sha256: {freeze['sha256_of_this_file']}")
    print("=" * 78)


if __name__ == "__main__":
    main()
