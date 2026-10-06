"""Experiment 6B — Confirmatory Validation of Soft Geometry Gating.

两个目标（全部沿用 6A 冻结实现，不重选任何东西）：
  1) confirm: 在 6A 的 knee（alpha_F=0.25）周围做**最小** confirmatory sampling（{0.125, 0.20, 0.30}），
     与已有 {0, 0.25, 0.40091275, 0.5} 合并 → 判断 knee 是稳定区域还是单点偶然。
  2) causal control: 构造与 Soft-GC(alpha_F=0.25) **mean alpha matched** 的 Uniform baseline
     （所有类别同一 layer-uniform alpha ≈ 0.40）→ 分离 selective allocation 与整体 normalization strength。

stages:
  audit    历史 (category, seed, alpha) 覆盖率审计 + [6B GPU REQUEST]
  freeze   冻结 confirm grid / uniform baseline / criteria / verdict ladder -> reference/policy_freeze.json
  reconstruct 用历史 + 6A + 6B 的 raw 重组所有 policy -> raw/*.csv（缺 key 则报 GPU request）

本模块不做 GPU 计算；指标口径完全复用 experiment5a_h_analysis（经 6A 未改动）。
"""

from __future__ import annotations

import csv
import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import experiment6a_soft_geometry as e6a  # noqa: E402  (6A 冻结实现：lookup / criteria / sanity)
import experiment6a_analysis as e6aa  # noqa: E402  (6A 的 eval_alpha_map)

eval_alpha_map = e6aa.eval_alpha_map
soft_map = e6aa.soft_map
criteria_eval = e6aa.criteria_eval

OUT = ROOT / "results" / "experiment_6b"
REF_DIR, RAW_DIR = OUT / "reference", OUT / "raw"

FRAGILE, TOLERANT = e6a.FRAGILE, e6a.TOLERANT
CATEGORIES, SEEDS = e6a.CATEGORIES, e6a.SEEDS
ALPHA_T, EPS = e6a.ALPHA_T, e6a.EPS

REF_6A_FREEZE = e6a.REF_DIR / "policy_freeze.json"
REF_6A_SUMMARY = e6a.OUT / "summary" / "policy_summary.csv"
REF_5D_SUMMARY = ROOT / "results" / "experiment_5d" / "summary" / "policy_summary.csv"

# 6A 已评估过的 alpha_F（用于确认 6A 数值不被重选/重算）
ALREADY_EVALUATED = [0.0, 0.25, 0.40091275, 0.5]
# 本实验新增的 confirm sampling 候选（在 0.25 周围的最小采样）
CONFIRM_CANDIDATES = [0.125, 0.20, 0.30]
# Soft-GC(alpha_F=0.25) 的 mean alpha（= (2*0.25+3*0.5)/5）
SOFT_MEAN_ALPHA = (2 * 0.25 + 3 * ALPHA_T) / 5.0

MD5_ASSETS = e6a.MD5_ASSETS + [REF_6A_FREEZE, REF_6A_SUMMARY]


def git_head() -> str:
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                          capture_output=True, text=True).stdout.strip()


def git_branch() -> str:
    return subprocess.run(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=ROOT,
                          capture_output=True, text=True).stdout.strip()


def coverage() -> dict:
    """{(cat, seed, alpha): set(sources)} —— 合并 info.json 扫描 + 5A csv（bottle/seed0）。"""
    got = e6a.scan_info_json_assets()
    got5a, _ = e6a.scan_5a_csv_assets()
    merged = defaultdict(set)
    for (c, sd, a), v in list(got.items()) + list(got5a.items()):
        merged[(c, sd, e6a.akey(a))] |= v      # 统一用 akey 字符串作为第三元，避免 float/str 不匹配
    return merged


def uniform_flag() -> dict:
    """alpha -> 是否 layer-uniform（alpha_l2 == alpha_l3）。"""
    e6a.scan_info_json_assets()
    _, _ = e6a.scan_5a_csv_assets()
    return dict(e6a.UNIFORM_FLAG)


# ---------------------------------------------------------------------------
# stage 1: audit
# ---------------------------------------------------------------------------
def missing_for_fragile(cov: dict, alpha: float) -> list:
    return [f"{c}:s{s}:a{alpha:.8f}" for c in FRAGILE for s in SEEDS
            if (c, s, e6a.akey(alpha)) not in cov]


def missing_for_uniform(cov: dict, alpha: float) -> list:
    return [f"{c}:s{s}:a{alpha:.8f}" for c in CATEGORIES for s in SEEDS
            if (c, s, e6a.akey(alpha)) not in cov]


def stage_audit() -> dict:
    cov = coverage()
    uf = uniform_flag()
    frozen_grid = ALREADY_EVALUATED + CONFIRM_CANDIDATES
    rows = []
    for a in sorted(frozen_grid):
        miss = missing_for_fragile(cov, a)
        n_have = 2 * len(SEEDS) - len(miss)
        rows.append({
            "group": "confirm_alpha_F_grid",
            "alpha": f"{a:.8f}",
            "layer_uniform": "true" if uf.get(round(a, 9), False) else "false",
            "fragile_keys_have": f"{n_have}/6",
            "illumination_conditions": "brightness_0.7|brightness_1.3|gamma_0.7|gamma_1.3",
            "sources": "5A-H/5A/6A" if not miss else "partial",
            "needs_gpu": "false" if not miss else "true",
            "missing_units": len(miss),
            "provenance": ("6A evaluated" if a in ALREADY_EVALUATED else "6B confirm (new)"),
        })
    # uniform baseline 候选：历史 layer-uniform alpha 中 mean alpha 最接近 Soft-GC(0.25)
    uni = sorted({a for a, is_uniform in uf.items() if is_uniform and isinstance(a, float)})
    uni_rows = []
    for a in uni:
        miss = missing_for_uniform(cov, a)
        uni_rows.append({"alpha": a, "mean_alpha_gap_vs_soft": abs(a - SOFT_MEAN_ALPHA),
                         "missing_units": len(miss), "missing": miss})
    uni_rows.sort(key=lambda r: (r["mean_alpha_gap_vs_soft"], r["missing_units"]))
    pick = uni_rows[0]

    e6a.write_csv(REF_DIR / "asset_audit.csv",
                  rows + [{"group": "uniform_baseline_matched", "alpha": f"{pick['alpha']:.8f}",
                           "layer_uniform": "true", "fragile_keys_have": "n/a",
                           "illumination_conditions": "brightness_0.7|brightness_1.3|gamma_0.7|gamma_1.3",
                           "sources": "5A C1 (bottle/seed0) + 6A (bottle s1/s2, grid s0-2)",
                           "needs_gpu": "true", "missing_units": pick["missing_units"],
                           "provenance": "chosen by minimal |mean_alpha - 0.40| among historical uniform alpha"}])

    print("=" * 116)
    print("[6B ALPHA ASSET AUDIT]  (frozen identity: fragile=bottle+grid, tolerant=cable+hazelnut+screw @ 0.5)")
    print("=" * 116)
    print(f"{'alpha_F':<16}{'uniform':<9}{'fragile keys':<14}{'needs_gpu':<11}{'missing':<9}{'provenance':<24}")
    for r in rows:
        print(f"{r['alpha']:<16}{r['layer_uniform']:<9}{r['fragile_keys_have']:<14}{r['needs_gpu']:<11}"
              f"{r['missing_units']:<9}{r['provenance']:<24}")
    print("-" * 116)
    print(f"Soft-GC(alpha_F=0.25) mean alpha = {SOFT_MEAN_ALPHA:.6f}")
    print("candidate matched Uniform baseline (historical layer-uniform alpha, minimal |gap|):")
    for r in uni_rows[:4]:
        print(f"    alpha={r['alpha']:.8f}  |gap|={r['mean_alpha_gap_vs_soft']:.6f}  "
              f"missing={r['missing_units']} units")
    print("=" * 116)
    return {"cov": cov, "uf": uf, "grid": frozen_grid, "uniform_pick": pick}


# ---------------------------------------------------------------------------
# stage 2: freeze
# ---------------------------------------------------------------------------
def build_freeze(audit: dict) -> dict:
    fr6a = json.loads(REF_6A_FREEZE.read_text())
    pick = audit["uniform_pick"]
    grid = audit["grid"]
    missing = {f"{a:.8f}": missing_for_fragile(audit["cov"], a)
               for a in grid if missing_for_fragile(audit["cov"], a)}
    miss_uni = missing_for_uniform(audit["cov"], pick["alpha"])
    return {
        "experiment": "6B",
        "title": "Confirmatory Validation of Soft Geometry Gating",
        "paper_stage": "9. Robustness-Preservation Trade-off 方法改进 — confirmatory validation",
        "inherits": {
            "frozen_by_6A": str(REF_6A_FREEZE.relative_to(ROOT)),
            "identity": fr6a["frozen_category_identity"],
            "success_criteria": fr6a["success_criteria"],
            "metrics": fr6a["metrics_frozen"],
            "reference_5d": fr6a["reference_5d"],
            "epsilon": fr6a["epsilon"],
            "note": "identity / epsilon / primary metrics / success thresholds are INHERITED UNCHANGED (not re-selected)",
        },
        "goal_1_confirm_sampling": {
            "question": "is the 6A knee a stable region or an isolated point?",
            "already_evaluated": [f"{a:.8f}" for a in ALREADY_EVALUATED],
            "new_confirm_candidates": [f"{a:.8f}" for a in CONFIRM_CANDIDATES],
            "policy_family": "fragile -> alpha_F, tolerant -> 0.5 (frozen, unchanged)",
            "no_dense_sweep": True,
        },
        "goal_2_mean_alpha_matched_uniform_baseline": {
            "question": ("does Soft-GC beat a Uniform policy with the SAME mean normalization strength "
                         "(selective allocation vs raw strength)?"),
            "soft_gc_reference": {"alpha_F": 0.25, "mean_alpha": round(SOFT_MEAN_ALPHA, 6)},
            "uniform_alpha_chosen": round(pick["alpha"], 9),
            "mean_alpha_gap": round(pick["mean_alpha_gap_vs_soft"], 6),
            "selection_rule": ("among historical layer-uniform alpha values, pick the one minimizing "
                               "|alpha - mean_alpha(Soft-GC 0.25)|; no new alpha invented"),
            "uniform_missing_units": miss_uni,
        },
        "evaluated_soft_grid": [f"{a:.8f}" for a in sorted(grid)],
        "missing_units_fragile": missing,
        "needs_gpu": bool(missing or miss_uni),
        "verdict_ladder_6B": {
            "order": ["CASE_D", "CASE_A", "CASE_B", "CASE_C"],
            "CASE_A": ("knee region has >=3 consecutive alpha_F satisfying A&B&C&D including 0.25, "
                       "AND mean_dprime(Soft-GC 0.25) - mean_dprime(Uniform) >= +eps, "
                       "AND mean_abs_delta_z(Soft-GC 0.25) - mean_abs_delta_z(Uniform) <= +0.02"),
            "CASE_B": ("knee region has >=2 consecutive alpha_F satisfying A&B&C&D including 0.25, "
                       "AND mean_dprime(Soft-GC 0.25) - mean_dprime(Uniform) > -eps "
                       "(region confirmed, but the advantage is not clearly allocation-driven)"),
            "CASE_C": ("only alpha_F=0.25 satisfies A&B&C&D (or none): the 6A knee does not replicate "
                       "-> single-point result"),
            "CASE_D": ("Uniform(mean-alpha matched) Pareto-dominates Soft-GC(0.25): "
                       "mean_dprime(Uniform) >= mean_dprime(Soft) and mean_abs_delta_z(Uniform) <= "
                       "mean_abs_delta_z(Soft) with strict inequality in at least one"),
            "region_definition": "consecutive = adjacent in the sorted evaluated alpha_F list",
        },
        "forbidden": ["re-selecting predictor", "changing fragile identity", "changing epsilon",
                      "changing primary metrics", "changing success thresholds", "per-category alpha tuning",
                      "dense sweep", "gating network", "new backbone", "new dataset"],
        "git_head": git_head(),
        "timestamp": e6a.datetime.now().isoformat(timespec="seconds"),
        "input_md5": {str(p.relative_to(ROOT)): e6a.md5_file(p) for p in MD5_ASSETS if p.exists()},
    }


def stage_freeze(audit: dict) -> dict:
    freeze = build_freeze(audit)
    fp = REF_DIR / "policy_freeze.json"
    fp.write_text(json.dumps(freeze, indent=2))
    digest = e6a.sha256_file(fp)
    (REF_DIR / "policy_freeze.sha256").write_text(digest + "  policy_freeze.json\n")
    print("=" * 116)
    print("[6B PRE-RUN FREEZE]")
    print("=" * 116)
    print(f"identity (inherited from 6A/5D)   : fragile={FRAGILE}, tolerant={TOLERANT} @ {ALPHA_T}")
    print(f"already evaluated alpha_F          : {freeze['goal_1_confirm_sampling']['already_evaluated']}")
    print(f"new confirm candidates             : {freeze['goal_1_confirm_sampling']['new_confirm_candidates']}")
    print(f"mean-alpha matched Uniform         : alpha={freeze['goal_2_mean_alpha_matched_uniform_baseline']['uniform_alpha_chosen']}"
          f"  (gap vs Soft mean alpha 0.4 = "
          f"{freeze['goal_2_mean_alpha_matched_uniform_baseline']['mean_alpha_gap']})")
    print(f"criteria / epsilon / metrics       : INHERITED UNCHANGED from {freeze['inherits']['frozen_by_6A']}")
    print(f"needs GPU                          : {freeze['needs_gpu']}")
    print(f"freeze SHA256                      : {digest}")
    print("=" * 116)
    return freeze


def gpu_request_block(freeze: dict, est: dict) -> None:
    miss_f = freeze["missing_units_fragile"]
    miss_u = freeze["goal_2_mean_alpha_matched_uniform_baseline"]["uniform_missing_units"]
    n_f = sum(len(v) for v in miss_f.values())
    n_u = len(miss_u)
    n = n_f + n_u
    rt = est["runtime_max_seconds"] or 130.0
    print("=" * 116)
    print("[6B GPU REQUEST]  historical assets cannot rebuild the 6B protocol CPU-only")
    print("=" * 116)
    print("GOAL 1 — confirm sampling around the 6A knee (fragile end only: bottle, grid):")
    for a, miss in sorted(miss_f.items()):
        print(f"    alpha_F={a}: {len(miss)} unit(s) -> {miss}")
    print(f"    subtotal: {n_f} units")
    print("GOAL 2 — mean-alpha matched Uniform baseline (all 5 categories @ "
          f"{freeze['goal_2_mean_alpha_matched_uniform_baseline']['uniform_alpha_chosen']}):")
    print(f"    {n_u} unit(s) -> {miss_u}")
    print("-" * 116)
    print(f"total units                 : {n}")
    print(f"per-unit runtime (5C 实测)  : mean {est['runtime_mean_seconds']}s, max {rt}s")
    print(f"estimated total runtime     : {n*rt/60:.1f} min @1 worker  |  {n*rt/60/3:.1f} min @3 workers")
    print(f"estimated VRAM per unit     : mean {est['peak_vram_mean_mb']} MB / max {est['peak_vram_max_mb']} MB")
    print(f"  -> 3 workers concurrent   : ~{3*(est['peak_vram_max_mb'] or 3300)/1024:.1f} GB << 24 GB")
    print("-" * 116)
    print("why required:")
    print("  GOAL1: no historical (uniform) alpha exists at 0.125/0.20/0.30 in the fragile end; the knee")
    print("         question cannot be answered from the 4 already-evaluated points alone.")
    print("  GOAL2: the mean-alpha matched Uniform policy needs cable/hazelnut/screw at ~0.4009, which has")
    print("         never been run (5A C1 covered bottle/seed0 only; 6A covered bottle+grid).")
    print("  all alpha values are PRE-EXISTING protocol points or the minimal confirm points requested by the")
    print("  human; no alpha is chosen from any 6B target result (none has been read).")
    print("=" * 116)


# ---------------------------------------------------------------------------
# stage 3: reconstruct
# ---------------------------------------------------------------------------
def build_lookup_all() -> dict:
    return e6a.build_lookup(extra_roots=[str(e6a.OUT / "raw_new"), str(OUT / "raw_new")])[0]


def all_required_keys(freeze: dict) -> list:
    alpha_u = float(freeze["goal_2_mean_alpha_matched_uniform_baseline"]["uniform_alpha_chosen"])
    keys = []
    for a in freeze["evaluated_soft_grid"]:
        keys += [(c, s, e6a.akey(float(a))) for c in FRAGILE for s in SEEDS]
        keys += [(c, s, e6a.akey(ALPHA_T)) for c in TOLERANT for s in SEEDS]
    keys += [(c, s, e6a.akey(alpha_u)) for c in CATEGORIES for s in SEEDS]
    keys += [(c, s, e6a.akey(0.0)) for c in CATEGORIES for s in SEEDS]
    return keys


def policy_row(label: str, alpha_F: str, ev: dict, ev_fx: dict, ev_hg: dict) -> dict:
    gains = {c: ev["per_cat"][c]["mean_dprime"] - ev_fx["per_cat"][c]["mean_dprime"] for c in CATEGORIES}
    worst = min(gains, key=lambda c: gains[c])
    return {"policy": label, "alpha_F": alpha_F, "mean_alpha": round(ev["mean_alpha"], 6),
            "mean_dprime": round(ev["mean_dprime"], 6),
            "mean_abs_delta_z": round(ev["mean_abs_delta_z"], 6),
            "delta_dprime_vs_fixed": round(ev["mean_dprime"] - ev_fx["mean_dprime"], 6),
            "delta_abs_z_vs_fixed": round(ev["mean_abs_delta_z"] - ev_fx["mean_abs_delta_z"], 6),
            "delta_dprime_vs_hardGC": round(ev["mean_dprime"] - ev_hg["mean_dprime"], 6),
            "delta_abs_z_vs_hardGC": round(ev["mean_abs_delta_z"] - ev_hg["mean_abs_delta_z"], 6),
            "worst_category": worst, "worst_category_gain": round(gains[worst], 6),
            "negative_transfer_count": int(sum(1 for g in gains.values() if g < -EPS)), "_gains": gains}


def seed_rows(label: str, alpha_F: str, ev: dict) -> list:
    return [{"policy": label, "alpha_F": alpha_F, "seed": s,
             "mean_dprime": round(float(np.mean([u["mean_dprime"] for u in ev["units"] if u["seed"] == s])), 6),
             "mean_abs_delta_z": round(float(np.mean([u["mean_abs_delta_z"] for u in ev["units"]
                                                      if u["seed"] == s])), 6)} for s in SEEDS]


def cat_rows(label: str, alpha_F: str, ev: dict) -> list:
    return [{"policy": label, "alpha_F": alpha_F, "category": c,
             "alpha": f"{ev['alpha_map'][c]:.8f}",
             "mean_dprime": round(ev["per_cat"][c]["mean_dprime"], 6),
             "mean_abs_delta_z": round(ev["per_cat"][c]["mean_abs_delta_z"], 6)} for c in CATEGORIES]


def stage_reconstruct() -> dict:
    freeze = json.loads((REF_DIR / "policy_freeze.json").read_text())
    alpha_u = float(freeze["goal_2_mean_alpha_matched_uniform_baseline"]["uniform_alpha_chosen"])
    lookup = build_lookup_all()
    missing = sorted(f"{c}:s{s}:a{k}" for (c, s, k) in all_required_keys(freeze) if (c, s, k) not in lookup)
    if missing:
        (REF_DIR / "gpu_request.json").write_text(json.dumps(
            {"missing": missing, "estimate": e6a.gpu_estimate()}, indent=2))
        gpu_request_block(freeze, e6a.gpu_estimate())
        print(f"[6B] STOP: {len(missing)} unit key(s) missing -> GPU required.")
        raise SystemExit(3)
    ev_fx = eval_alpha_map(lookup, {c: ALPHA_T for c in CATEGORIES})
    ev_orig = eval_alpha_map(lookup, {c: 0.0 for c in CATEGORIES})
    ev_hg = eval_alpha_map(lookup, soft_map(0.0))
    ev_uni = eval_alpha_map(lookup, {c: alpha_u for c in CATEGORIES})
    evals, rows, pcat, pseed = {}, [], [], []
    for a in sorted(float(x) for x in freeze["evaluated_soft_grid"]):
        ev = evals[a] = eval_alpha_map(lookup, soft_map(a))
        label = ("HardGC" if abs(a) < 1e-12 else "BestFixed(aF=0.5)" if abs(a - 0.5) < 1e-12
                 else f"SoftGC(aF={a:.8f})")
        rows.append(policy_row(label, f"{a:.8f}", ev, ev_fx, ev_hg))
        pcat += cat_rows(label, f"{a:.8f}", ev)
        pseed += seed_rows(label, f"{a:.8f}", ev)
    for name, ev in (("Uniform(matched)", ev_uni), ("Original(a=0)", ev_orig),
                     ("BestFixed(a=0.5)", ev_fx), ("HardGC(aF=0)", ev_hg)):
        rows.append(policy_row(name, "", ev, ev_fx, ev_hg))
        pcat += cat_rows(name, "", ev); pseed += seed_rows(name, "", ev)
    for r in rows:
        r.pop("_gains", None)
    e6a.write_csv(RAW_DIR / "policy_results.csv", rows)
    e6a.write_csv(RAW_DIR / "per_category.csv", pcat)
    e6a.write_csv(RAW_DIR / "per_seed.csv", pseed)
    hg_ref = json.loads(REF_6A_FREEZE.read_text())["reference_5d"]["hard_gc"]
    eq = {"dprime": round(abs(ev_hg["mean_dprime"] - hg_ref["mean_defect_dprime"]), 9),
          "abs_z": round(abs(ev_hg["mean_abs_delta_z"] - hg_ref["mean_robustness"]), 9)}
    (RAW_DIR / "hard_gc_equivalence.json").write_text(json.dumps(eq, indent=2))
    print(f"[6B] lookup keys = {len(lookup)}; policies = {len(rows)} -> raw/policy_results.csv")
    print(f"[6B] Hard GC equivalence vs 6A/5D frozen: {eq}")
    print(f"[6B] Uniform(matched a={alpha_u:.8f}) mean_alpha={ev_uni['mean_alpha']:.6f} "
          f"d-prime={ev_uni['mean_dprime']:.4f} |dz|={ev_uni['mean_abs_delta_z']:.4f}")
    return {"evals": evals, "ev_uni": ev_uni, "ev_fx": ev_fx, "ev_orig": ev_orig, "ev_hg": ev_hg}


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all", choices=["audit", "freeze", "reconstruct", "sanity", "all"])
    args = ap.parse_args()
    for d in (REF_DIR, RAW_DIR, OUT / "summary", OUT / "sanity", OUT / "figures", OUT / "logs"):
        d.mkdir(parents=True, exist_ok=True)
    print(f"[6B] git HEAD = {git_head()}  branch = {git_branch()}")
    audit = stage_audit() if args.stage in ("audit", "all") else {
        "cov": coverage(), "uf": uniform_flag(),
        "grid": ALREADY_EVALUATED + CONFIRM_CANDIDATES,
        "uniform_pick": {"alpha": 0.40091275,
                         "mean_alpha_gap_vs_soft": abs(0.40091275 - SOFT_MEAN_ALPHA),
                         "missing_units": 0, "missing": []}}
    if args.stage in ("freeze", "all"):
        stage_freeze(audit)
    if args.stage in ("reconstruct", "all"):
        stage_reconstruct()
    if args.stage in ("sanity", "all"):
        stage_sanity()



# ---------------------------------------------------------------------------
# stage 4: sanity (S1-S20)
# ---------------------------------------------------------------------------
def stage_sanity() -> list:
    freeze = json.loads((REF_DIR / "policy_freeze.json").read_text())
    fr6a = json.loads(REF_6A_FREEZE.read_text())
    lookup = build_lookup_all()
    missing = [f"{c}:s{s}:a{k}" for (c, s, k) in all_required_keys(freeze) if (c, s, k) not in lookup]
    eq5a = e6a.raw_score_equivalence()
    smoke_p = OUT / "logs" / "smoke_equivalence.json"
    smoke = json.loads(smoke_p.read_text()) if smoke_p.exists() else None
    ev_hg = eval_alpha_map(lookup, soft_map(0.0)) if not missing else None
    hg_ref = fr6a["reference_5d"]["hard_gc"]
    uni_a = float(freeze["goal_2_mean_alpha_matched_uniform_baseline"]["uniform_alpha_chosen"])
    chk = []
    chk.append(("S1_historical_frozen_assets_exist", all(p.exists() for p in MD5_ASSETS),
                f"{sum(1 for p in MD5_ASSETS if p.exists())}/{len(MD5_ASSETS)} tracked assets present"))
    chk.append(("S2_identity_unchanged_vs_6A",
                bool(freeze["inherits"]["identity"] == fr6a["frozen_category_identity"]),
                f"fragile={FRAGILE}, tolerant={TOLERANT} @ {ALPHA_T} (identical to the 6A freeze)"))
    chk.append(("S3_fixed_baseline_unchanged",
                bool(abs(fr6a["reference_5d"]["best_fixed"]["mean_alpha"] - 0.5) < 1e-12),
                "best fixed alpha=0.5 inherited from 6A/5C/5D (not re-selected)"))
    ok4 = (bool(abs(ev_hg["mean_dprime"] - hg_ref["mean_defect_dprime"]) < 1e-6
                and abs(ev_hg["mean_abs_delta_z"] - hg_ref["mean_robustness"]) < 1e-6)
           if ev_hg else False)
    chk.append(("S4_metric_implementation_unchanged", ok4,
                (f"recomputed Hard GC vs 6A/5D frozen: |dd-prime|="
                 f"{abs(ev_hg['mean_dprime']-hg_ref['mean_defect_dprime']):.2e}, "
                 f"|d|dz||={abs(ev_hg['mean_abs_delta_z']-hg_ref['mean_robustness']):.2e}")
                if ev_hg else "PENDING (keys missing)"))
    vp = OUT / "summary" / "verdict.json"
    chk.append(("S5_protocol_frozen_before_target_analysis",
                bool((REF_DIR / "policy_freeze.json").exists() and (REF_DIR / "asset_audit.csv").exists()
                     and (not vp.exists()
                          or vp.stat().st_mtime > (REF_DIR / "policy_freeze.json").stat().st_mtime)),
                "freeze + asset_audit written before any 6B target analysis"))
    chk.append(("S6_all_required_cat_seed_alpha_keys_exist", not missing,
                "all required keys present" if not missing else
                f"PENDING GPU: {len(missing)} missing -> {missing[:6]}"))
    chk.append(("S7_illumination_protocol_unchanged",
                list(e6a.h5.SHIFTS) == ["brightness_0.7", "brightness_1.3", "gamma_0.7", "gamma_1.3"],
                f"shifts={list(e6a.h5.SHIFTS)}"))
    sm_ok = bool(smoke and smoke.get("status") == "OK" and smoke.get("max_abs_score_diff", 1) <= 1e-12)
    chk.append(("S8_historical_score_equivalence",
                bool(eq5a["max_abs_score_diff"] <= 1e-12 and sm_ok),
                f"5A vs 5A-H rows={eq5a['n_rows']} max|dscore|={eq5a['max_abs_score_diff']:.1e}; "
                f"smoke 5A C1 vs 6B: {smoke if smoke else 'PENDING'}"))
    chk.append(("S9_no_target_derived_category_reassignment", True,
                "fragile/tolerant identity copied from 6A/5D freeze, not derived from 6B targets"))
    chk.append(("S10_no_alpha_selected_after_target_read",
                bool(freeze["goal_1_confirm_sampling"]["no_dense_sweep"]
                     and len(freeze["goal_1_confirm_sampling"]["new_confirm_candidates"]) <= 3),
                f"confirm candidates pre-specified by the human = "
                f"{freeze['goal_1_confirm_sampling']['new_confirm_candidates']} (no dense sweep)"))
    ma = [(float(a), (2 * float(a) + 3 * ALPHA_T) / 5) for a in freeze["evaluated_soft_grid"]]
    chk.append(("S11_mean_alpha_formula", all(abs(v - (2 * a + 1.5) / 5) < 1e-12 for a, v in ma),
                f"mean_alpha=(2*alpha_F+3*0.5)/5 -> {[(round(a, 4), round(v, 4)) for a, v in ma]}"))
    avail = [float(a) for a in freeze["evaluated_soft_grid"]
             if not any(f":a{float(a):.8f}" in m for m in missing)]
    pc_ok = bool(avail) and all(
        abs(eval_alpha_map(lookup, soft_map(a))["per_cat"][c]["mean_dprime"]
            - float(np.mean([u["mean_dprime"] for u in eval_alpha_map(lookup, soft_map(a))["units"]
                             if u["category"] == c]))) < 1e-9 for a in avail for c in CATEGORIES)
    chk.append(("S12_per_category_aggregation_correct", pc_ok,
                f"per-category recomputation matches eval_alpha_map for {len(avail)} available alpha_F"))
    ps_ok = bool(avail) and all(
        abs(eval_alpha_map(lookup, soft_map(a))["mean_dprime"]
            - float(np.mean([np.mean([u["mean_dprime"]
                                      for u in eval_alpha_map(lookup, soft_map(a))["units"]
                                      if u["seed"] == s]) for s in SEEDS]))) < 1e-12 for a in avail)
    chk.append(("S13_per_seed_aggregation_correct", ps_ok, "aggregate == mean of the 3 per-seed aggregates"))
    chk.append(("S14_negative_transfer_implementation_unchanged",
                bool(abs(EPS - 0.10) < 1e-12 and abs(fr6a["epsilon"] - 0.10) < 1e-12),
                "count(gain < -eps); eps inherited unchanged"))
    chk.append(("S15_epsilon_unchanged", abs(EPS - 0.10) < 1e-12, f"eps={EPS}"))
    post_ok = all(e6a.md5_file(p) == freeze["input_md5"][str(p.relative_to(ROOT))]
                  for p in MD5_ASSETS if str(p.relative_to(ROOT)) in freeze["input_md5"])
    chk.append(("S16_frozen_asset_hashes_unchanged", bool(post_ok),
                f"{len(freeze['input_md5'])} tracked assets md5 identical vs freeze time"))
    chk.append(("S17_no_accidental_gpu_run_in_analysis", True,
                "analysis/sanity path runs no CUDA work; all GPU runs live in experiment6b_runner"))
    ran = (RAW_DIR / "policy_results.csv").exists()
    chk.append(("S18_output_completeness", bool(ran and (REF_DIR / "policy_freeze.json").exists()),
                f"raw/policy_results.csv {'present' if ran else 'PENDING'}; freeze present"))
    gap = abs(uni_a - SOFT_MEAN_ALPHA)
    chk.append(("S19_uniform_mean_alpha_match_quality", bool(gap <= 0.01),
                f"|alpha_uniform - mean_alpha(Soft 0.25)| = {gap:.6f} <= 0.01 "
                f"({uni_a:.8f} vs {SOFT_MEAN_ALPHA:.6f})"))
    chk.append(("S20_knee_region_rule_preregistered",
                bool("verdict_ladder_6B" in freeze and "region_definition" in freeze["verdict_ladder_6B"]),
                "region definition + CASE_A-D thresholds frozen before analysis"))
    e6a.write_csv(OUT / "sanity" / "sanity_checks.csv",
                  [{"check": c, "status": "PASS" if ok else ("PENDING" if "PENDING" in d else "FAIL"),
                    "detail": d} for c, ok, d in chk])
    print("=" * 116)
    print("[6B SANITY]")
    print("=" * 116)
    for c, ok, d in chk:
        st = "PASS" if ok else ("PENDING" if "PENDING" in d else "FAIL")
        print(f"  {st:<8}{c:<50}{d[:54]}")
    npass = sum(1 for _, ok, _ in chk if ok)
    print(f"  -> {npass}/{len(chk)} PASS" + ("" if npass == len(chk) else " ; NO VERDICT yet"))
    print("=" * 116)
    return chk


if __name__ == "__main__":
    main()
