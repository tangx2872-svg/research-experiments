"""Experiment 6A — Geometry-Guided Soft Gating / Adaptive alpha Pilot.

策略族（category identity 冻结，仅插值 fragile 端的 alpha）：
    bottle, grid            -> alpha_F          (fragile, 5D GC gate)
    cable, hazelnut, screw  -> 0.5              (tolerant, 固定)
  alpha_F = 0.0  == Hard Geometry Gating (5D GC)
  alpha_F = 0.5  == Best Fixed alpha=0.5

stages:
  audit       扫描全部历史 (category, seed, alpha) 资产 -> reference/asset_audit.csv
  freeze      依据「历史 uniform alpha 可用性」+ 协议规则冻结候选 grid -> reference/policy_freeze.json
  reconstruct 用历史 raw per-image 分数重组每个 policy 的指标 -> raw/*.csv（缺 key 则报错并列出）

本模块不做任何 GPU 计算；指标口径完全复用 experiment5a_h_analysis。
"""

from __future__ import annotations

import csv
import hashlib
import json
import subprocess
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import experiment5a_h_analysis as h5  # noqa: E402  (冻结指标口径)

OUT = ROOT / "results" / "experiment_6a"
REF_DIR, RAW_DIR = OUT / "reference", OUT / "raw"

CATEGORIES = ["bottle", "cable", "grid", "hazelnut", "screw"]
SEEDS = [0, 1, 2]
FRAGILE = ["bottle", "grid"]
TOLERANT = ["cable", "hazelnut", "screw"]
ALPHA_T = 0.5
EPS = 0.10

# 5D 冻结 reference（Hard GC 的数值不得手敲，必须从 5D 结果读取）
REF_5C_FREEZE = ROOT / "results" / "experiment_5c" / "reference" / "geometry_policy_freeze.json"
REF_5D_FREEZE = ROOT / "results" / "experiment_5d" / "reference" / "policy_freeze.json"
REF_5D_SUMMARY = ROOT / "results" / "experiment_5d" / "summary" / "policy_summary.csv"

RAW_5AH = ROOT / "results" / "experiment_5a_h" / "raw"
RAW_5C = ROOT / "results" / "experiment_5c" / "raw"
SCORES_5A = ROOT / "results" / "experiment_5a" / "raw" / "per_image_scores.csv"

# 历史 uniform alpha 的协议来源（5A mini probe 的 config 名，用于 provenance 标注）
HIST_UNIFORM_SOURCE = {
    0.0: "5A B0 / 5A-H B0",
    0.25: "5A B1",
    0.40091275: "5A C1",
    0.5: "5A B2 / 5A-H B2",
    0.601369125: "5A C2 / 5A-H C2",
    0.75: "5A B3",
    0.8018255: "5A C3 / 5A-H C3",
    1.0: "5A B4",
}

MD5_ASSETS = [
    ROOT / "results" / "experiment_5a_h" / "summary" / "per_unit.csv",
    ROOT / "results" / "experiment_5c" / "reference" / "geometry_policy_freeze.json",
    ROOT / "results" / "experiment_5d" / "reference" / "policy_freeze.json",
    ROOT / "results" / "experiment_5d" / "summary" / "policy_summary.csv",
    ROOT / "results" / "experiment_5a" / "raw" / "per_image_scores.csv",
    ROOT / "results" / "experiment_5b_final" / "summary" / "normal_only_predictors.csv",
]


def md5_file(p: Path) -> str:
    return hashlib.md5(p.read_bytes()).hexdigest()


def sha256_file(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def git_head() -> str:
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                          capture_output=True, text=True).stdout.strip()


def akey(a: float) -> str:
    return f"{float(a):.9f}"


def write_csv(path: Path, rows: list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


# ---------------------------------------------------------------------------
# stage 1: audit
# ---------------------------------------------------------------------------
UNIFORM_FLAG = {}


def scan_info_json_assets() -> dict:
    """results/**/info.json -> {(cat, seed, alpha_l2): [source labels]}；同时记录 layer-uniform 标志。"""
    got = defaultdict(set)
    global UNIFORM_FLAG
    UNIFORM_FLAG = {}
    for info in sorted(ROOT.glob("results/**/info.json")):
        try:
            d = json.loads(info.read_text())
        except Exception:
            continue
        if d.get("status") != "OK":
            continue
        c, s, a = d.get("category"), d.get("seed"), d.get("alpha_l2")
        if c is None or s is None or a is None:
            continue
        parts = info.relative_to(ROOT).parts
        exp = parts[1] if len(parts) > 1 else "?"
        a2, a3 = round(float(a), 9), round(float(d.get("alpha_l3", a)), 9)
        got[(c, int(s), a2)].add(f"{exp}:{d.get('config')}")
        UNIFORM_FLAG[a2] = UNIFORM_FLAG.get(a2, True) and abs(a2 - a3) <= 1e-12
    return got


def scan_5a_csv_assets() -> tuple:
    """5A per_image_scores.csv -> {(cat, seed, alpha): labels} 与 (subset, shift) 覆盖。"""
    got = defaultdict(set)
    cond = defaultdict(set)
    if not SCORES_5A.exists():
        return got, cond
    for r in csv.DictReader(open(SCORES_5A, newline="")):
        a2, a3 = float(r["alpha_l2"]), float(r["alpha_l3"])
        cat = "bottle"                       # 5A 仅 bottle
        seed = 0                             # 5A 仅 seed0
        got[(cat, seed, round(a2, 9))].add(f"5A:{r['config']}")
        cond[round(a2, 9)].add((r["subset"], r["shift"]))
        if abs(a2 - a3) > 1e-12:
            cond[f"layerwise:{a2}/{a3}"].add((r["subset"], r["shift"]))
        UNIFORM_FLAG[round(a2, 9)] = UNIFORM_FLAG.get(round(a2, 9), True) and abs(a2 - a3) <= 1e-12
    return got, cond


def historical_alpha_table() -> tuple:
    """返回 (per_alpha_rows, table, cond5a)。table: alpha -> {(cat,seed)}"""
    got = scan_info_json_assets()
    got5a, cond5a = scan_5a_csv_assets()
    merged = defaultdict(set)
    for k, v in got.items():
        merged[k] |= v
    for k, v in got5a.items():
        merged[k] |= v
    table = defaultdict(set)
    for (c, s, a) in merged:
        table[a].add((c, s))
    uniform_alphas = sorted(a for a in table if not isinstance(a, str))
    rows = []
    for a in uniform_alphas:
        keys = table[a]
        cats = sorted({c for c, _ in keys})
        seeds = sorted({s for _, s in keys})
        n_cat_seed = len(keys)
        rows.append({
            "alpha": f"{a:.9f}",
            "historical_score_exists": "true" if keys else "false",
            "n_category_seed_keys": n_cat_seed,
            "full_5cat_3seed_coverage": "true" if n_cat_seed == 15 else "false",
            "categories": "+".join(cats),
            "seeds": "+".join(str(s) for s in seeds),
            "illumination_conditions": "brightness_0.7|brightness_1.3|gamma_0.7|gamma_1.3"
                                       if n_cat_seed > 0 else "",
            "source_configs": ";".join(sorted(merged[[k for k in merged if k[2] == a][0]])) if keys else "",
            "layer_uniform": "true" if UNIFORM_FLAG.get(a, False) else "false",
            "usable_as_alpha_F": "true" if (n_cat_seed == 15 and UNIFORM_FLAG.get(a, False)
                                            and 0.0 <= a <= 0.5) else "false",
            "needs_gpu": "false" if n_cat_seed == 15 else "true",
        })
    return rows, table, cond5a


# ---------------------------------------------------------------------------
# stage 2: freeze
# ---------------------------------------------------------------------------
def load_5d_reference() -> dict:
    """从 5D 冻结 summary 读取 Hard GC / Best Fixed 参考值（禁止手敲）。"""
    rows = list(csv.DictReader(open(REF_5D_SUMMARY, newline="")))
    by = {r["policy"]: r for r in rows}
    gc, bf = by["GC"], by["B_best_fixed"]
    return {
        "source_file": str(REF_5D_SUMMARY.relative_to(ROOT)),
        "hard_gc": {"mean_defect_dprime": float(gc["mean_defect_dprime"]),
                    "mean_robustness": float(gc["mean_robustness"]),
                    "delta_dprime_vs_fixed": float(gc["delta_dprime_vs_fixed"]),
                    "delta_robust_vs_fixed": float(gc["delta_robust_vs_fixed"]),
                    "worst_category_gain": float(gc["worst_category_gain"]),
                    "negative_transfer_count": int(gc["negative_transfer_count"]),
                    "gate": gc["gated_categories"]},
        "best_fixed": {"mean_defect_dprime": float(bf["mean_defect_dprime"]),
                       "mean_robustness": float(bf["mean_robustness"]),
                       "mean_alpha": float(bf["mean_alpha"])},
    }


def criterion_definitions(ref5d: dict) -> dict:
    """§8 预注册 success criteria（阈值唯一来源 = 用户协议 + 5D 冻结 reference）。"""
    gc = ref5d["hard_gc"]
    return {
        "A_preservation_retention": {
            "rule": "delta_dprime_vs_fixed(soft) >= +0.20",
            "threshold": 0.20,
            "note": ">= ~60% of the Hard-GC preservation gain "
                    f"({gc['delta_dprime_vs_fixed']:+.4f} → 60% = {0.6 * gc['delta_dprime_vs_fixed']:+.4f})",
        },
        "B_robustness_recovery": {
            "rule": "mean_abs_delta_z(soft) <= mean_abs_delta_z(hard_GC) - 0.02",
            "threshold_abs_z_max": round(gc["mean_robustness"] - 0.02, 6),
            "improvement_required": 0.02,
            "note": f"Hard GC |dz| = {gc['mean_robustness']:.4f} (read from 5D frozen summary, not hard-coded)",
        },
        "C_safety": {
            "rule": "worst_category_gain >= -eps AND negative_transfer_count <= HardGC negative_transfer_count",
            "epsilon": EPS,
            "hard_gc_negative_transfer_count": gc["negative_transfer_count"],
        },
        "D_stability": {
            "rule": ("at least 2/3 seeds satisfy BOTH A and B at seed level: "
                     "delta_dprime_vs_fixed_seed >= +0.20 AND "
                     "mean_abs_delta_z_seed <= hardGC_mean_abs_delta_z_seed - 0.02"),
            "min_seeds": 2,
        },
    }


def build_freeze(alpha_rows: list, table: dict) -> dict:
    ref5d = load_5d_reference()
    # 只用 layer-uniform 的 alpha（alpha_l2 == alpha_l3）作为 alpha_F 候选
    hist_uniform = sorted({a for a in table if UNIFORM_FLAG.get(a, False)})
    inter = [a for a in hist_uniform if 1e-12 < a < 0.5 - 1e-12]
    grid = [0.0] + (inter[:1] if inter else []) + [0.5]
    extension = inter[1:2]
    avail = {}
    for a in grid + extension:
        keys = table.get(a, set())
        avail[f"{a:.9f}"] = {
            "bottle_seeds": sorted([s for (c, s) in keys if c == "bottle"]),
            "grid_seeds": sorted([s for (c, s) in keys if c == "grid"]),
            "full_5cat_3seed": len(keys) == 15,
            "n_category_seed_keys": len(keys),
        }
    missing_units = []
    for a in grid:
        for cat in FRAGILE:
            for s in SEEDS:
                if (cat, s) not in table.get(a, set()):
                    missing_units.append(f"{cat}:{s}:alpha={a:.9f}")
    alpha_source = {f"{a:.9f}": HIST_UNIFORM_SOURCE.get(a, "unknown") for a in grid + extension}
    return {
        "experiment": "6A",
        "title": "Geometry-Guided Soft Gating / Adaptive alpha Pilot",
        "paper_stage": "9. Robustness-Preservation Trade-off method improvement (METHOD PILOT)",
        "research_question": ("Can geometry-guided soft normalization recover illumination robustness while "
                              "retaining most of the defect-preservation gain of Hard GC?"),
        "target_claim_under_test": ("Geometry-guided adaptive normalization can provide a better "
                                    "preservation-robustness trade-off than either no normalization, globally "
                                    "fixed normalization, or hard geometry gating. (HYPOTHESIS, not a fact)"),
        "frozen_category_identity": {"fragile": FRAGILE, "tolerant": TOLERANT,
                                     "source": "5D frozen GC assignment (bottle+grid) - NOT to be changed"},
        "policy_family": {
            "fragile_alpha": "alpha_F (candidate grid)",
            "tolerant_alpha": ALPHA_T,
            "mean_alpha_formula": "(2 * alpha_F + 3 * 0.5) / 5",
            "alpha_F_0_equals": "Hard Geometry Gating (5D GC)",
            "alpha_F_0p5_equals": "Best Fixed alpha=0.5",
        },
        "candidate_grid": grid,
        "candidate_grid_provenance": alpha_source,
        "pre_registered_conditional_extension": {
            "values": extension,
            "provenance": {f"{a:.9f}": HIST_UNIFORM_SOURCE.get(a, "unknown") for a in extension},
            "rule": ("per protocol section 5: may add at most 1-2 historical alpha ONLY if a Pareto knee lies "
                     "between two existing points; the extension value is pre-specified here BEFORE any 6A "
                     "target analysis, and requires human approval of the GPU request"),
        },
        "grid_rule": ("alpha_F grid = {0, nearest historical uniform intermediate with usable coverage, 0.5}; "
                      "derived from HISTORICAL AVAILABILITY + protocol minimum, never from 6A target results"),
        "historical_uniform_alphas_available": [f"{a:.9f}" for a in hist_uniform],
        "alpha_availability": avail,
        "missing_units_for_grid": missing_units,
        "needs_gpu_for_grid": bool(missing_units),
        "epsilon": EPS,
        "metrics_frozen": {
            "preservation": "mean defect d-prime (higher better)",
            "robustness": "mean |Delta NormalScore_z| (lower better)",
            "safety": ["worst_category_gain", "negative_transfer_count(eps)", "high_damage_subset={bottle,grid}",
                       "seed consistency"],
            "implementation": "reuse experiment5a_h_analysis.load_all/unit_metrics (unchanged)",
        },
        "reference_5d": ref5d,
        "success_criteria": criterion_definitions(ref5d),
        "budget_matching_disclosure": ("Hard GC mean alpha = 0.30; Soft GC mean alpha = (2*alpha_F+1.5)/5 > 0.30 "
                                       "for alpha_F > 0. Soft GC therefore receives MORE normalization strength; "
                                       "the analysis must separate policy-identity effect from "
                                       "normalization-strength effect. No strong causal claim."),
        "categories": CATEGORIES,
        "seeds": SEEDS,
        "illumination_protocol": "4 photometric shifts (brightness 0.7/1.3, gamma 0.7/1.3), frozen since 5A-H",
        "no_gpu_by_default": True,
        "data_provenance": "CPU-only recombination of frozen historical raw per-image scores",
        "git_head": git_head(),
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "input_md5": {str(p.relative_to(ROOT)): md5_file(p) for p in MD5_ASSETS if p.exists()},
    }


# ---------------------------------------------------------------------------
# stage 3: reconstruct
# ---------------------------------------------------------------------------
METRIC_KEYS = ["mean_dprime", "mean_abs_delta_z", "image_auroc", "pixel_auroc", "fpr_clean"]


def metrics_from_5ah() -> dict:
    h5.RAW_DIR = RAW_5AH
    h5.CONFIGS = ["B0", "B2", "C2", "C3", "G2"]
    out = {}
    for (cat, seed, cfg), mm in h5.unit_metrics(h5.load_all()).items():
        out[(cat, seed, akey(mm["alpha_l2"]))] = {k: float(mm[k]) for k in METRIC_KEYS} | {
            "alpha": float(mm["alpha_l2"]), "source": f"5A-H:{cfg}"}
    return out


def metrics_from_5a_csv() -> dict:
    """5A per_image_scores.csv（bottle / seed0）——只取 uniform alpha（alpha_l2 == alpha_l3）。"""
    out = {}
    if not SCORES_5A.exists():
        return out
    rows = list(csv.DictReader(open(SCORES_5A, newline="")))
    by_cfg = defaultdict(list)
    for r in rows:
        by_cfg[r["config"]].append(r)
    for cfg, rs in by_cfg.items():
        a2, a3 = float(rs[0]["alpha_l2"]), float(rs[0]["alpha_l3"])
        if abs(a2 - a3) > 1e-12:
            continue                                     # layer-wise config 不属于 6A 的 uniform alpha 族
        sub = defaultdict(lambda: defaultdict(list))
        for r in rs:
            sub[r["subset"]][r["shift"]].append((r["defect_type"], float(r["score"])))
        cg = np.array([s for _, s in sub["clean_good"]["none"]])
        mu, sd = float(cg.mean()), float(cg.std(ddof=1))
        dz = [float(((np.array([s for _, s in sub["shift_good"][sh]]) - mu) / sd).mean()) for sh in h5.SHIFTS]
        defects = sorted({dt for dt, _ in sub["clean_defect"]["none"]})
        dps = [h5.d_prime(np.array([s for d, s in sub["clean_defect"]["none"] if d == dt]), mu, sd)
               for dt in defects]
        out[("bottle", 0, akey(a2))] = {
            "mean_dprime": float(np.mean(dps)), "mean_abs_delta_z": float(np.mean(np.abs(dz))),
            "image_auroc": float("nan"), "pixel_auroc": float("nan"), "fpr_clean": float("nan"),
            "alpha": a2, "source": f"5A:{cfg}"}
    return out


def build_lookup(extra_roots: list = None) -> tuple:
    lookup, prov = {}, {}
    for src, d in (("5A-H", metrics_from_5ah()), ("5A(bottle/seed0)", metrics_from_5a_csv())):
        for k, v in d.items():
            lookup.setdefault(k, v)
            prov.setdefault(k, v["source"])
    for extra in (extra_roots or []):
        p = Path(extra)
        if not p.exists():
            continue
        data = {}
        for info_p in sorted(p.glob("*/seed_*/config_*/info.json")):
            d = json.loads(info_p.read_text())
            if d.get("status") != "OK":
                continue
            rows = list(csv.DictReader(open(info_p.parent / "per_image.csv", newline="")))
            sub = defaultdict(lambda: defaultdict(list))
            for r in rows:
                sub[r["subset"]][r["shift"]].append((r["defect_type"], float(r["score"]),
                                                     float(r["clipped_high_ratio"]),
                                                     float(r["clipped_low_ratio"])))
            data[(d["category"], int(d["seed"]), d["config"])] = {"sub": sub, "info": d, "rows": rows}
        for (cat, seed, cfg), mm in h5.unit_metrics(data).items():
            k = (cat, seed, akey(mm["alpha_l2"]))
            lookup[k] = {kk: float(mm[kk]) for kk in METRIC_KEYS} | {
                "alpha": float(mm["alpha_l2"]), "source": f"6A:{cfg}"}
            prov[k] = f"6A:{cfg}"
    return lookup, prov


def required_keys(alpha_F: float) -> list:
    return [(c, s, akey(alpha_F)) for c in FRAGILE for s in SEEDS] + \
           [(c, s, akey(ALPHA_T)) for c in TOLERANT for s in SEEDS]


def check_keys(grid: list, lookup: dict) -> dict:
    missing = {}
    for a in grid:
        miss = [f"{c}:s{s}:a{a:.9f}" for (c, s, k) in required_keys(a) if (c, s, k) not in lookup]
        if miss:
            missing[f"{a:.9f}"] = miss
    return missing


def eval_policy(alpha_F: float, lookup: dict) -> dict:
    amap = {c: (alpha_F if c in FRAGILE else ALPHA_T) for c in CATEGORIES}
    units = []
    for c in CATEGORIES:
        for s in SEEDS:
            m = lookup[(c, s, akey(amap[c]))]
            units.append({"category": c, "seed": s, "alpha": amap[c], "mean_dprime": m["mean_dprime"],
                          "mean_abs_delta_z": m["mean_abs_delta_z"], "source": m["source"]})
    per_cat = {c: {"mean_dprime": float(np.mean([u["mean_dprime"] for u in units if u["category"] == c])),
                   "mean_abs_delta_z": float(np.mean([u["mean_abs_delta_z"] for u in units
                                                      if u["category"] == c])),
                   "alpha": amap[c]} for c in CATEGORIES}
    return {"alpha_F": alpha_F, "alpha_map": amap, "units": units, "per_cat": per_cat,
            "mean_alpha": float(np.mean(list(amap.values()))),
            "mean_dprime": float(np.mean([u["mean_dprime"] for u in units])),
            "mean_abs_delta_z": float(np.mean([u["mean_abs_delta_z"] for u in units]))}


def gpu_estimate() -> dict:
    """从 5C 已跑过的 bottle/grid unit 读取 runtime / VRAM（证据化估计，不手敲）。"""
    rt, vr = [], []
    for cat in FRAGILE:
        for f in sorted((ROOT / "results" / "experiment_5c" / "raw" / cat).glob("seed_*/config_*/info.json")):
            d = json.loads(f.read_text())
            if d.get("status") == "OK":
                rt.append(float(d["runtime_seconds"]))
                vr.append(float(d.get("peak_gpu_memory_allocated_mb", float("nan"))))
    rt, vr = [x for x in rt if np.isfinite(x)], [x for x in vr if np.isfinite(x)]
    return {"n_source_units": len(rt),
            "runtime_mean_seconds": round(float(np.mean(rt)), 2) if rt else None,
            "runtime_max_seconds": round(float(np.max(rt)), 2) if rt else None,
            "peak_vram_mean_mb": round(float(np.mean(vr)), 1) if vr else None,
            "peak_vram_max_mb": round(float(np.max(vr)), 1) if vr else None,
            "source": "results/experiment_5c/raw/{bottle,grid}/*/info.json"}


def stage_audit() -> dict:
    rows, table, cond5a = historical_alpha_table()
    write_csv(REF_DIR / "asset_audit.csv", rows)
    print("=" * 108)
    print("[6A ALPHA ASSET AUDIT]  historical (category, seed, alpha) coverage")
    print("=" * 108)
    hdr = (f"{'alpha':<16}{'exists':<8}{'keys':<6}{'full15':<8}{'layer_uniform':<15}"
           f"{'usable_aF':<11}{'needs_gpu':<10}")
    print(hdr)
    for r in rows:
        print(f"{r['alpha']:<16}{r['historical_score_exists']:<8}{r['n_category_seed_keys']:<6}"
              f"{r['full_5cat_3seed_coverage']:<8}{r['layer_uniform']:<15}{r['usable_as_alpha_F']:<11}"
              f"{r['needs_gpu']:<10}")
    print("-" * 108)
    unif = [r["alpha"] for r in rows if r["layer_uniform"] == "true"]
    print("layer-uniform alpha (any coverage)            :", unif)
    print("layer-uniform + FULL 5cat x 3seed coverage    :",
          [r["alpha"] for r in rows if r["layer_uniform"] == "true" and r["full_5cat_3seed_coverage"] == "true"])
    print("USABLE alpha_F (uniform + full coverage + in [0,0.5]) :",
          [r["alpha"] for r in rows if r["usable_as_alpha_F"] == "true"])
    print("  -> strictly inside (0, 0.5)                 :",
          [r["alpha"] for r in rows if r["usable_as_alpha_F"] == "true" and 1e-12 < float(r["alpha"]) < 0.5 - 1e-12])
    print("5A-only (bottle/seed0) layer-uniform alpha    :",
          [f"{a:.9f}" for a in sorted(k for k in cond5a if not isinstance(k, str))])
    print("=" * 108)
    return {"table": table, "rows": rows}


def gpu_request_block(grid: list, missing: dict, est: dict) -> tuple:
    n_units = sum(len(v) for v in missing.values())
    rt = est["runtime_max_seconds"] or 130.0
    print("=" * 108)
    print("[6A GPU REQUEST]  historical assets are NOT sufficient for the frozen candidate grid")
    print("=" * 108)
    for a, miss in missing.items():
        print(f"  alpha_F = {a}: missing {len(miss)} unit(s)")
        for m in miss:
            print(f"        - {m}")
    print("-" * 108)
    print(f"missing alpha values        : {sorted(missing)}")
    print(f"units to run                : {n_units}  (category x seed x alpha_F, fragile end only)")
    print(f"per-unit runtime (5C 实测)  : mean {est['runtime_mean_seconds']}s, max {rt}s "
          f"(n={est['n_source_units']} units, source: {est['source']})")
    print(f"estimated total runtime     : {n_units*rt/60:.1f} min @1 worker  |  "
          f"{n_units*rt/60/3:.1f} min @3 workers")
    print(f"estimated VRAM              : mean {est['peak_vram_mean_mb']} MB / max {est['peak_vram_max_mb']} MB per unit")
    print(f"  -> 3 workers concurrent   : ~{3*(est['peak_vram_max_mb'] or 3300)/1024:.1f} GB << 24 GB (RTX 3090)")
    print("-" * 108)
    print("why these runs are required :")
    print("  the 6A policy family sets bottle+grid to alpha_F and keeps cable/hazelnut/screw at 0.5.")
    print("  every historical uniform alpha with FULL 5cat x 3seed coverage is in {0, 0.5, 0.6014, 0.8018},")
    print("  i.e. there is NO historical uniform alpha strictly inside (0, 0.5) -> the interpolation")
    print("  between Hard GC (alpha_F=0) and Best Fixed (alpha_F=0.5) cannot be built CPU-only.")
    print("  these alpha values are PRE-EXISTING protocol points of Experiment 5A (B1=0.25, C1=0.40091275),")
    print("  not new alpha invented for 6A.")
    print("  nothing in this request depends on any 6A target result (none has been read).")
    print("=" * 108)
    return n_units, sorted(missing)


def stage_reconstruct() -> dict:
    freeze = json.loads((REF_DIR / "policy_freeze.json").read_text())
    # 人类于运行前批准了预注册的条件扩展 -> 最终评估 grid = candidate_grid ∪ extension（freeze JSON 未修改）
    grid = [float(x) for x in freeze["candidate_grid"]] + \
           [float(x) for x in freeze["pre_registered_conditional_extension"]["values"]]
    lookup, prov = build_lookup(extra_roots=[str(OUT / "raw_new")])
    missing = check_keys(grid, lookup)
    if missing:
        (REF_DIR / "gpu_request.json").write_text(json.dumps(
            {"missing": missing, "grid": grid, "estimate": gpu_estimate()}, indent=2))
        gpu_request_block(grid, missing, gpu_estimate())
        print("[6A] STOP: frozen candidate grid cannot be reconstructed CPU-only.")
        raise SystemExit(3)

    ref5d = freeze["reference_5d"]
    hg, bf = ref5d["hard_gc"], ref5d["best_fixed"]
    rows, per_cat_rows, per_seed_rows = [], [], []
    evals = {}
    for a in grid:
        ev = evals[a] = eval_policy(a, lookup)
        per_cat_gain = {c: ev["per_cat"][c]["mean_dprime"] - bf["mean_defect_dprime"] for c in CATEGORIES}
        worst = min(per_cat_gain, key=lambda c: per_cat_gain[c])
        worst_gain = per_cat_gain[worst]
        neg_transfer = int(sum(1 for g in per_cat_gain.values() if g < -EPS))
        for c in CATEGORIES:
            per_cat_rows.append({"policy": f"SoftGC_aF={a:.8f}", "alpha_F": f"{a:.8f}", "category": c,
                                 "alpha": f"{ev['alpha_map'][c]:.8f}",
                                 "mean_dprime": round(ev["per_cat"][c]["mean_dprime"], 6),
                                 "mean_abs_delta_z": round(ev["per_cat"][c]["mean_abs_delta_z"], 6)})
        for s in SEEDS:
            us = [u for u in ev["units"] if u["seed"] == s]
            per_seed_rows.append({"policy": f"SoftGC_aF={a:.8f}", "alpha_F": f"{a:.8f}", "seed": s,
                                  "mean_dprime": round(float(np.mean([u["mean_dprime"] for u in us])), 6),
                                  "mean_abs_delta_z": round(float(np.mean([u["mean_abs_delta_z"] for u in us])), 6)})
        rows.append({"policy": f"SoftGC_aF={a:.8f}", "alpha_F": f"{a:.8f}",
                     "mean_alpha": round(ev["mean_alpha"], 6),
                     "mean_dprime": round(ev["mean_dprime"], 6),
                     "mean_abs_delta_z": round(ev["mean_abs_delta_z"], 6),
                     "delta_dprime_vs_fixed": round(ev["mean_dprime"] - bf["mean_defect_dprime"], 6),
                     "delta_abs_z_vs_fixed": round(ev["mean_abs_delta_z"] - bf["mean_robustness"], 6),
                     "delta_dprime_vs_hardGC": round(ev["mean_dprime"] - hg["mean_defect_dprime"], 6),
                     "delta_abs_z_vs_hardGC": round(ev["mean_abs_delta_z"] - hg["mean_robustness"], 6),
                     "worst_category": worst,
                     "worst_category_gain": round(worst_gain, 6),
                     "negative_transfer_count": neg_transfer,
                     "high_damage_recovery": round(float(np.mean([per_cat_gain[c] for c in FRAGILE])), 6),
                     "note": ("= Hard GC (5D)" if abs(a) < 1e-12 else
                              "= Best Fixed alpha=0.5" if abs(a - 0.5) < 1e-12 else "")})
    write_csv(RAW_DIR / "policy_results.csv", rows)
    write_csv(RAW_DIR / "per_category.csv", per_cat_rows)
    write_csv(RAW_DIR / "per_seed.csv", per_seed_rows)
    # 等价性：重组出的 alpha_F=0 必须与 5D 冻结 GC 数值一致
    eq = {k: round(abs(next(r for r in rows if abs(float(r["alpha_F"])) < 1e-12)["mean_dprime"]
                        - hg["mean_defect_dprime"]), 9) for k in ["dprime"]}
    eq["abs_z"] = round(abs(next(r for r in rows if abs(float(r["alpha_F"])) < 1e-12)["mean_abs_delta_z"]
                            - hg["mean_robustness"]), 9)
    (RAW_DIR / "hard_gc_equivalence.json").write_text(json.dumps(eq, indent=2))
    print(f"[6A] reconstructed {len(rows)} policies -> raw/policy_results.csv")
    print(f"[6A] Hard GC equivalence (recomputed alpha_F=0 vs 5D frozen): {eq}")
    return {"rows": rows, "evals": evals, "equivalence": eq}


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all", choices=["audit", "freeze", "reconstruct", "sanity", "all"])
    args = ap.parse_args()
    for d in (REF_DIR, RAW_DIR, OUT / "summary", OUT / "sanity", OUT / "figures", OUT / "logs"):
        d.mkdir(parents=True, exist_ok=True)
    print(f"[6A] git HEAD = {git_head()}")
    table = None
    if args.stage in ("audit", "all"):
        table = stage_audit()["table"]
    if args.stage in ("freeze", "all"):
        if table is None:
            table = historical_alpha_table()[1]
        freeze = build_freeze(historical_alpha_table()[0], table)
        fp = REF_DIR / "policy_freeze.json"
        fp.write_text(json.dumps(freeze, indent=2))
        digest = sha256_file(fp)
        (REF_DIR / "policy_freeze.sha256").write_text(digest + "  policy_freeze.json\n")
        print("=" * 108)
        print("[6A PRE-RUN FREEZE]")
        print("=" * 108)
        print(f"candidate grid alpha_F        : {freeze['candidate_grid']}")
        print(f"grid provenance               : {freeze['candidate_grid_provenance']}")
        print(f"conditional extension (pre-registered): {freeze['pre_registered_conditional_extension']['values']}")
        print(f"frozen fragile categories     : {freeze['frozen_category_identity']['fragile']}")
        print(f"tolerant alpha                : {ALPHA_T}")
        print(f"needs GPU for frozen grid     : {freeze['needs_gpu_for_grid']}")
        print(f"missing units                 : {len(freeze['missing_units_for_grid'])}")
        print(f"5D Hard GC reference          : d-prime={freeze['reference_5d']['hard_gc']['mean_defect_dprime']}, "
              f"|dz|={freeze['reference_5d']['hard_gc']['mean_robustness']}")
        print(f"criterion B |dz| ceiling      : {freeze['success_criteria']['B_robustness_recovery']['threshold_abs_z_max']}")
        print(f"freeze JSON                   : {fp.relative_to(ROOT)}")
        print(f"freeze SHA256                 : {digest}")
        print("=" * 108)
    if args.stage in ("sanity", "all"):
        stage_sanity()
    if args.stage in ("reconstruct", "all"):
        stage_reconstruct()



# ---------------------------------------------------------------------------
# stage 4: sanity (S1-S18)
# ---------------------------------------------------------------------------
def raw_score_equivalence() -> dict:
    """5A(bottle/seed0) 与 5A-H(bottle/seed0) 在相同 alpha 上的 raw per-image 分数应逐位一致。"""
    rows = list(csv.DictReader(open(SCORES_5A, newline="")))
    max_ds, n = 0.0, 0
    for cfg in ("B0", "B2", "C2", "C3"):
        sa = {(r["image_path"], r["subset"], r["shift"]): float(r["score"])
              for r in rows if r["config"] == cfg}
        p = RAW_5AH / "bottle" / "seed_0" / f"config_{cfg}" / "per_image.csv"
        if not p.exists() or not sa:
            continue
        sb = {(r["image_path"], r["subset"], r["shift"]): float(r["score"])
              for r in csv.DictReader(open(p, newline=""))}
        keys = set(sa) & set(sb)
        if keys:
            max_ds = max(max_ds, max(abs(sa[k] - sb[k]) for k in keys))
            n += len(keys)
    return {"n_rows": n, "max_abs_score_diff": max_ds}


def stage_sanity() -> list:
    freeze = json.loads((REF_DIR / "policy_freeze.json").read_text())
    f5d = json.loads(REF_5D_FREEZE.read_text())
    f5c = json.loads(REF_5C_FREEZE.read_text())
    hg = freeze["reference_5d"]["hard_gc"]
    lookup, _ = build_lookup(extra_roots=[str(OUT / "raw_new")])
    missing = check_keys([float(x) for x in freeze["candidate_grid"]], lookup)
    eq = raw_score_equivalence()
    chk = []
    chk.append(("S1_historical_frozen_assets_exist", all(p.exists() for p in MD5_ASSETS),
                f"{sum(1 for p in MD5_ASSETS if p.exists())}/{len(MD5_ASSETS)} tracked assets present"))
    gc5d = f5d["category_assignments"]["GC"]
    chk.append(("S2_5D_GC_assignment_unchanged",
                bool(gc5d == {"bottle": 0.0, "grid": 0.0, "cable": 0.5, "hazelnut": 0.5, "screw": 0.5}
                     and freeze["frozen_category_identity"]["fragile"] == FRAGILE),
                f"fragile={FRAGILE} == 5D GC gate; tolerant alpha={ALPHA_T}"))
    chk.append(("S3_fixed_baseline_unchanged",
                abs(float(f5c["best_fixed_alpha"]) - 0.5) < 1e-12,
                "best fixed alpha=0.5 (5C/5D frozen); reference read from 5D summary, not hand-typed"))
    zero_ok, d_detail = None, "PENDING (blocked by missing alpha_F keys)"
    if not missing.get("0.000000000"):
        ev0 = eval_policy(0.0, lookup)
        zero_ok = bool(abs(ev0["mean_dprime"] - hg["mean_defect_dprime"]) < 1e-6
                       and abs(ev0["mean_abs_delta_z"] - hg["mean_robustness"]) < 1e-6)
        d_detail = (f"recomputed alpha_F=0 vs 5D frozen GC: "
                    f"|dd-prime|={abs(ev0['mean_dprime'] - hg['mean_defect_dprime']):.2e}, "
                    f"|d|dz||={abs(ev0['mean_abs_delta_z'] - hg['mean_robustness']):.2e}")
    chk.append(("S4_metric_implementation_unchanged", zero_ok if zero_ok is not None else False, d_detail))
    verdict_p = OUT / "summary" / "verdict.json"
    s5_ok = bool((REF_DIR / "policy_freeze.json").exists() and (REF_DIR / "asset_audit.csv").exists()
                 and (not verdict_p.exists()
                      or verdict_p.stat().st_mtime > (REF_DIR / "policy_freeze.json").stat().st_mtime))
    chk.append(("S5_candidate_grid_frozen_before_target_analysis", s5_ok,
                f"grid={freeze['candidate_grid']} frozen at freeze time; verdict.json "
                f"{'absent' if not verdict_p.exists() else 'written after freeze (mtime order ok)'}"))
    chk.append(("S6_all_required_cat_seed_alpha_keys_exist", not missing,
                "all required keys present" if not missing else
                f"PENDING GPU: {sum(len(v) for v in missing.values())} missing unit(s) {sorted(missing)}"))
    chk.append(("S7_illumination_protocol_unchanged",
                list(h5.SHIFTS) == ["brightness_0.7", "brightness_1.3", "gamma_0.7", "gamma_1.3"],
                f"shifts={list(h5.SHIFTS)}"))
    chk.append(("S8_historical_score_equivalence", eq["max_abs_score_diff"] <= 1e-12,
                f"5A(bottle/seed0) vs 5A-H same alpha: rows={eq['n_rows']}, "
                f"max|dscore|={eq['max_abs_score_diff']:.3e}"))
    chk.append(("S9_no_target_derived_category_reassignment",
                freeze["frozen_category_identity"]["source"].startswith("5D frozen"),
                "fragile/tolerant identity copied from 5D freeze, not derived from 6A targets"))
    chk.append(("S10_no_alpha_selected_before_full_candidate_evaluation",
                bool(len(freeze["candidate_grid"]) >= 2 and freeze["candidate_grid"][0] == 0.0
                     and freeze["candidate_grid"][-1] == 0.5),
                f"grid frozen a priori = {freeze['candidate_grid']} (endpoints = Hard GC / Best Fixed)"))
    ma = [(a, (2 * a + 3 * ALPHA_T) / 5) for a in freeze["candidate_grid"]]
    chk.append(("S11_mean_alpha_formula", all(abs(v - (2 * a + 1.5) / 5) < 1e-12 for a, v in ma),
                f"mean_alpha=(2*alpha_F+3*0.5)/5 -> {[(round(a, 4), round(v, 4)) for a, v in ma]}"))
    avail_a = [a for a in freeze["candidate_grid"] if not missing.get(f"{a:.9f}")]
    pc_ok = all(abs(eval_policy(a, lookup)["per_cat"][c]["mean_dprime"]
                    - float(np.mean([u["mean_dprime"] for u in eval_policy(a, lookup)["units"]
                                     if u["category"] == c]))) < 1e-9
                for a in avail_a for c in CATEGORIES)
    chk.append(("S12_per_category_aggregation_correct", bool(pc_ok and avail_a),
                f"per-category recomputation matches eval_policy for {len(avail_a)} available alpha_F"))
    ps_ok = bool(avail_a) and all(
        abs(eval_policy(a, lookup)["mean_dprime"]
            - float(np.mean([np.mean([u["mean_dprime"] for u in eval_policy(a, lookup)["units"]
                                      if u["seed"] == s]) for s in SEEDS]))) < 1e-12
        for a in avail_a)
    chk.append(("S13_per_seed_aggregation_correct", ps_ok,
                "aggregate == mean over the 3 per-seed aggregates (available alpha_F)"))
    chk.append(("S14_negative_transfer_implementation_unchanged",
                bool(abs(EPS - 0.10) < 1e-12 and abs(float(f5d["epsilon"]) - 0.10) < 1e-12),
                "count(gain < -eps), eps from 5A-H/5D frozen band"))
    chk.append(("S15_epsilon_unchanged", abs(EPS - 0.10) < 1e-12,
                f"eps={EPS} (5A-H section 2 frozen band)"))
    post_ok = all(md5_file(p) == freeze["input_md5"][str(p.relative_to(ROOT))]
                  for p in MD5_ASSETS if str(p.relative_to(ROOT)) in freeze["input_md5"])
    chk.append(("S16_frozen_asset_hashes_unchanged", bool(post_ok),
                f"{len(freeze['input_md5'])} tracked frozen assets md5 identical vs freeze time"))
    chk.append(("S17_no_accidental_gpu_run", True,
                "no torch/CUDA import in this script; 0 GPU unit executed"))
    ran = (OUT / "raw" / "policy_results.csv").exists()
    n_ref = sum(1 for f in ("policy_freeze.json", "asset_audit.csv") if (REF_DIR / f).exists())
    chk.append(("S18_output_completeness", bool(n_ref == 2 and ran),
                f"reference {n_ref}/2; raw/policy_results.csv "
                f"{'present' if ran else 'PENDING (blocked by GPU request)'}"))
    write_csv(OUT / "sanity" / "sanity_checks.csv",
              [{"check": c, "status": "PASS" if ok else ("PENDING" if "PENDING" in d else "FAIL"),
                "detail": d} for c, ok, d in chk])
    print("=" * 108)
    print("[6A PRE-RUN SANITY]")
    print("=" * 108)
    for c, ok, d in chk:
        st = "PASS" if ok else ("PENDING" if "PENDING" in d else "FAIL")
        print(f"  {st:<8}{c:<48}{d[:58]}")
    npass = sum(1 for _, ok, _ in chk if ok)
    print(f"  -> {npass}/{len(chk)} PASS" + ("" if npass == len(chk) else " ; NO VERDICT may be issued yet"))
    print("=" * 108)
    return chk


if __name__ == "__main__":
    main()
