"""Experiment 7A-O — Overnight Module Screening: 冻结配置与调度定义。

本模块**不含任何 target 指标读取**；只定义：
  - 被 screening 的 module families / 参数 grid（PRE-RUN frozen）
  - baselines（历史 raw 可复用者明确标注）
  - successive-halving 的 Round 0/1/2/3 unit 定义
  - mechanical promotion 规则（Tier S/A/B）与 verdict 规则
  - freeze JSON（SHA256）

设计原则：
  * module 只改 generate_embedding 内的 feature 变换，**不改** PatchCore 的
    bank / coreset / illumination / 评估代码路径（由 experiment7ao_runner 复用
    experiment5a_h_runner.run_config 保证）。
  * 所有超参在 PRE-RUN 冻结；禁止 dense sweep。
"""

from __future__ import annotations

import csv
import hashlib
import json
import subprocess
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "experiment_7a_o"
REF_DIR, CFG_DIR, RAW_DIR = OUT / "reference", OUT / "configs", OUT / "raw"
SUM_DIR, FIG_DIR, LOG_DIR, SAN_DIR = OUT / "summary", OUT / "figures", OUT / "logs", OUT / "sanity"

CATEGORIES = ["bottle", "cable", "grid", "hazelnut", "screw"]
SEEDS = [0, 1, 2]
SHIFTS = [("brightness_0.7", "brightness", 0.7), ("brightness_1.3", "brightness", 1.3),
          ("gamma_0.7", "gamma", 0.7), ("gamma_1.3", "gamma", 1.3)]

UNIFORM_ALPHA = 0.40091275          # 6B 确立的强 baseline
FIXED_ALPHA = 0.5                   # 历史 best fixed
SOFT_GC_ALPHA_F = 0.25              # 6A knee（reference only）
EPS = 0.10                          # preservation band（5A-H 冻结）

# success metric 目标：相对 Uniform baseline 的 Δd′ / Δ|Δz|
TIER = {
    "S": {"delta_dprime_min": 0.0, "delta_abs_z_max": 0.0, "neg_transfer_may_increase": False},
    "A": {"delta_dprime_min": 0.10, "delta_abs_z_max": 0.02, "neg_transfer_may_increase": False},
    "B": {"delta_dprime_min": -0.05, "delta_abs_z_max": -0.02, "neg_transfer_may_increase": True},
}
PROMOTE_TIERS = ("S", "A", "B")

ROUND1_CATEGORIES = ["bottle", "grid", "hazelnut"]
ROUND1_SEEDS = [0]

# ---------------------------------------------------------------------------
# Module families（参数 grid 冻结于运行前）
# ---------------------------------------------------------------------------
def alpha_step(a: float) -> dict:
    return {"kind": "alpha_in", "alpha": float(a)}


def residual_step(lam: float) -> dict:
    return {"kind": "residual", "lambda": float(lam), "norm": "instance",
            "scale_match": "per_sample_per_channel_rms(f)", "eps": 1e-5}


def energy_step(lam: float) -> dict:
    return {"kind": "energy", "lambda": float(lam), "norm": "instance",
            "scale_match": "per_sample_per_channel_rms(f)", "eps": 1e-5}


def dual_step(gamma: float) -> dict:
    return {"kind": "concat_dual", "gamma": float(gamma), "norm": "instance",
            "scale_match": "per_sample_per_channel_rms(f)", "eps": 1e-5}


def layernorm_step(s: float) -> dict:
    return {"kind": "altnorm_strength", "altnorm": "layernorm_all_dims", "strength": float(s),
            "eps": 1e-5, "affine": False}


def groupnorm_step(s: float, groups: int) -> dict:
    return {"kind": "altnorm_strength", "altnorm": "groupnorm_channels", "groups": int(groups),
            "strength": float(s), "eps": 1e-5, "affine": False}


def spec(l2: dict, l3: dict) -> dict:
    return {"l2": l2, "l3": l3}


MODULE_SPECS = {
    # Family A — residual / original-preserving fusion
    "A1_lam010": spec(residual_step(0.10), residual_step(0.10)),
    "A1_lam025": spec(residual_step(0.25), residual_step(0.25)),
    "A1_lam050": spec(residual_step(0.50), residual_step(0.50)),
    "A3_lam025": spec(energy_step(0.25), energy_step(0.25)),
    "A3_lam050": spec(energy_step(0.50), energy_step(0.50)),
    # Family B — dual representation
    "B1_g025": spec(dual_step(0.25), dual_step(0.25)),
    "B1_g050": spec(dual_step(0.50), dual_step(0.50)),
    "B1_g100": spec(dual_step(1.00), dual_step(1.00)),
    # Family C — layer-selective normalization（mean alpha = 0.40091275 与 Uniform 同 budget）
    "C1_l2_030_l3_050": spec(alpha_step(0.30), alpha_step(0.50)),
    "C2_l2_035_l3_045": spec(alpha_step(0.35), alpha_step(0.45)),
    "C3_l2_045_l3_035": spec(alpha_step(0.45), alpha_step(0.35)),
    "C4_l2_050_l3_030": spec(alpha_step(0.50), alpha_step(0.30)),
    # Family D — alternative normalization（strength ≈ Uniform 0.40091275）
    "D1_ln_s040": spec(layernorm_step(UNIFORM_ALPHA), layernorm_step(UNIFORM_ALPHA)),
    "D2_gn8_s040": spec(groupnorm_step(UNIFORM_ALPHA, 8), groupnorm_step(UNIFORM_ALPHA, 8)),
}

# equivalence smoke 用的特殊 spec（不属于 screening candidate）
SMOKE_SPECS = {
    "SMOKE_ORIG_a000": spec(alpha_step(0.0), alpha_step(0.0)),
    "SMOKE_UNIFORM_a040091275": spec(alpha_step(UNIFORM_ALPHA), alpha_step(UNIFORM_ALPHA)),
}

NOT_IMPLEMENTED = {
    "A2_concat_fixed_projection": (
        "NOT SAFE / NOT IMPLEMENTED. A training-free fixed linear projection from the 2C-dim concat "
        "back to C-dim is either (i) P=[I|I]/sqrt(2), which is mathematically an alpha-interpolation in "
        "disguise (explicitly forbidden by the protocol), or (ii) an arbitrary hand-designed operator "
        "without principled justification (also forbidden). The concat representation itself is covered "
        "by Family B1 with a measured dimensionality change instead."
    ),
    "B2_grouped_projection": (
        "NOT RUN. B1 concat was affordable within the runtime/VRAM budget (measured in Round 0), so the "
        "fallback dimension-balancing variant was unnecessary; running it would have added an untested "
        "arbitrary operator."
    ),
}

FAMILY_OF = {k: k[0] for k in MODULE_SPECS}

ASSUMPTIONS_FROZEN = [
    "instance_norm == F.instance_norm (affine=False), identical to 5A/5A-H/5C/6A/6B",
    "residual: N=IN(F); N_hat = N * rms_spatial(F) per (sample, channel); F_out = F + lambda*N_hat",
    "energy: F_out = (F + lambda*N_hat) * rms_spatial(F)/rms_spatial(F + lambda*N_hat) per (sample, channel)",
    "concat_dual: F_out = concat([F, gamma*N_hat], dim=1) -> channel dim doubles (512->1024, 1024->2048)",
    "altnorm layernorm_all_dims: mean/var over (C,H,W) per sample",
    "altnorm groupnorm_channels: G=8 groups of channels, mean/var over (C/G,H,W) per sample, affine=False",
    "all norms use the input's own statistics only; no trainable parameters; no target/test-side statistics",
    "PatchCore bank / coreset(0.1) / kNN(9) / illumination / scoring / metrics are byte-identical to 5A-H",
]


def md5_file(p: Path) -> str:
    return hashlib.md5(p.read_bytes()).hexdigest()


def sha256_file(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def git_head() -> str:
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                          capture_output=True, text=True).stdout.strip()


def write_csv(path: Path, rows: list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


# ---------------------------------------------------------------------------
# 历史 raw 覆盖（用于 baseline 复用判定，不读任何指标）
# ---------------------------------------------------------------------------
HIST_SOURCES = [
    ("5A-H", ROOT / "results" / "experiment_5a_h" / "raw"),
    ("6A", ROOT / "results" / "experiment_6a" / "raw_new"),
    ("6B", ROOT / "results" / "experiment_6b" / "raw_new"),
]
SCORES_5A = ROOT / "results" / "experiment_5a" / "raw" / "per_image_scores.csv"


def akey(a: float) -> str:
    return f"{float(a):.9f}"


def historical_keys() -> dict:
    """{(cat, seed, a_l2_key, a_l3_key): source}"""
    got = {}
    for tag, root in HIST_SOURCES:
        if not root.exists():
            continue
        for info_p in sorted(root.glob("*/seed_*/config_*/info.json")):
            d = json.loads(info_p.read_text())
            if d.get("status") != "OK":
                continue
            got[(d["category"], int(d["seed"]), akey(d["alpha_l2"]), akey(d["alpha_l3"]))] = f"{tag}:{d['config']}"
    if SCORES_5A.exists():
        for r in csv.DictReader(open(SCORES_5A, newline="")):
            a2, a3 = akey(r["alpha_l2"]), akey(r["alpha_l3"])
            got.setdefault(("bottle", 0, a2, a3), f"5A:{r['config']}")
    return got


def baseline_defs() -> dict:
    """历史 baseline（全部可由已冻结 raw 重组，零 GPU）。"""
    return {
        "B0_original": {"type": "alpha", "a_l2": 0.0, "a_l3": 0.0,
                        "note": "= 5A-H B0 (alpha=0), reused"},
        "B1_fixed_050": {"type": "alpha", "a_l2": FIXED_ALPHA, "a_l3": FIXED_ALPHA,
                         "note": "= 5A-H B2 (alpha=0.5), reused"},
        "B2_uniform_040091275": {"type": "alpha", "a_l2": UNIFORM_ALPHA, "a_l3": UNIFORM_ALPHA,
                                 "note": "= 6B/6A Uniform baseline, reused (no rerun)"},
        "B3_soft_gc_af025": {"type": "cat_gate", "fragile": ["bottle", "grid"], "a_fragile": SOFT_GC_ALPHA_F,
                             "a_other": FIXED_ALPHA, "note": "= 6A Soft-GC, reconstructed from 6A raw + 5A-H B2"},
    }


def unit_id(cat: str, seed: int, name: str) -> str:
    return f"{cat}:{seed}:{name}"


def round0_units() -> list:
    return [("bottle", 0, "SMOKE_ORIG_a000"), ("bottle", 0, "SMOKE_UNIFORM_a040091275")]


def round1_units() -> list:
    return [(c, s, n) for n in MODULE_SPECS for c in ROUND1_CATEGORIES for s in ROUND1_SEEDS]


def round2_units(winners: list, done: set) -> list:
    need = [(c, s, n) for n in winners for c in CATEGORIES for s in SEEDS]
    return [u for u in need if unit_id(*u) not in done]


def round3_units(wa: str, wb: str, done: set) -> list:
    name = f"COMB_{wa}__{wb}"
    need = [(c, 0, name) for c in ROUND1_CATEGORIES]
    return [u for u in need if unit_id(*u) not in done]


def compose_spec(wa: str, wb: str) -> dict:
    """Round 3 组合：对同一 layer tensor 依次施加 A 的 step、再施加 B 的 step（顺序固定）。"""
    sa, sb = MODULE_SPECS[wa], MODULE_SPECS[wb]
    return {"l2": {"kind": "compose", "steps": [sa["l2"], sb["l2"]]},
            "l3": {"kind": "compose", "steps": [sa["l3"], sb["l3"]]}}


def gpu_estimate() -> dict:
    """从历史 5C/6B 单元读取 runtime / VRAM（证据化估计）。"""
    import numpy as np
    rt, vr = [], []
    for tag, root in (("5C", ROOT / "results" / "experiment_5c" / "raw"),
                      ("6B", ROOT / "results" / "experiment_6b" / "raw_new")):
        for f in sorted(root.glob("*/seed_*/config_*/info.json")):
            d = json.loads(f.read_text())
            if d.get("status") == "OK":
                rt.append(float(d["runtime_seconds"]))
                vr.append(float(d.get("peak_gpu_memory_allocated_mb", float("nan"))))
    rt = [x for x in rt if np.isfinite(x)]
    vr = [x for x in vr if np.isfinite(x)]
    return {"n_source_units": len(rt),
            "runtime_mean_seconds": round(float(np.mean(rt)), 2) if rt else None,
            "runtime_max_seconds": round(float(np.max(rt)), 2) if rt else None,
            "peak_vram_mean_mb": round(float(np.mean(vr)), 1) if vr else None,
            "peak_vram_max_mb": round(float(np.max(vr)), 1) if vr else None,
            "source": "results/experiment_{5c/raw,6b/raw_new}/*/seed_*/config_*/info.json"}


def stage_audit() -> dict:
    hist = historical_keys()
    rows, new_units = [], []
    for phase, units in (("R0_smoke", round0_units()), ("R1_broad", round1_units())):
        for cat, seed, name in units:
            sp = (MODULE_SPECS | SMOKE_SPECS)[name]
            k = (cat, seed, akey(sp["l2"].get("alpha", -1)), akey(sp["l3"].get("alpha", -1)))
            src = hist.get(k, "")
            rows.append({"phase": phase, "family": FAMILY_OF.get(name, "smoke"), "config": name,
                         "category": cat, "seed": seed,
                         "historical_reuse": "true" if src else "false", "source": src,
                         "needs_gpu": "false" if src else "true"})
            if not src:
                new_units.append((cat, seed, name))
    est = gpu_estimate()
    r2_max = 4 * len(CATEGORIES) * len(SEEDS) - 4 * len(ROUND1_CATEGORIES) * len(ROUND1_SEEDS)
    r3_max = len(ROUND1_CATEGORIES)
    total = len(new_units) + r2_max + r3_max
    write_csv(REF_DIR / "asset_audit.csv", rows)
    L = 112
    print("=" * L)
    print("[7A-O ALPHA/MODULE ASSET AUDIT]")
    print("=" * L)
    print(f"historical (cat,seed,a_l2,a_l3) keys available : {len(hist)}")
    for bname, bd in baseline_defs().items():
        if bd["type"] == "alpha":
            n = sum(1 for c in CATEGORIES for s in SEEDS if (c, s, akey(bd["a_l2"]), akey(bd["a_l3"])) in hist)
            print(f"  baseline {bname:<26} coverage {n}/15 units  ({bd['note']})")
        else:
            n1 = sum(1 for c in bd["fragile"] for s in SEEDS
                     if (c, s, akey(bd["a_fragile"]), akey(bd["a_fragile"])) in hist)
            n2 = sum(1 for c in CATEGORIES if c not in bd["fragile"] for s in SEEDS
                     if (c, s, akey(bd["a_other"]), akey(bd["a_other"])) in hist)
            print(f"  baseline {bname:<26} coverage {n1 + n2}/15 units  ({bd['note']})")
    print("-" * L)
    fam_count = {}
    for _, _, n in round1_units():
        fam_count[FAMILY_OF[n]] = fam_count.get(FAMILY_OF[n], 0) + 1
    print(f"Round0 smoke units        : {len(round0_units())}")
    print(f"Round1 configs            : {len(MODULE_SPECS)}  ({ {k: sum(1 for n in MODULE_SPECS if FAMILY_OF[n] == k) for k in sorted(set(FAMILY_OF.values()))} })")
    print(f"Round1 units              : {len(round1_units())}  ({len(ROUND1_CATEGORIES)} categories x {len(ROUND1_SEEDS)} seed)")
    print(f"Round2 max units (4 cfgs) : {r2_max}")
    print(f"Round3 max units (1 combo): {r3_max}")
    print(f"TOTAL new GPU units (max) : {total}")
    print("-" * L)
    rt = est["runtime_max_seconds"] or 160.0
    print(f"per-unit runtime (历史实测): mean {est['runtime_mean_seconds']}s / max {rt}s (n={est['n_source_units']})")
    print(f"estimated total runtime    : {total*rt/60:.0f} min @1 worker | {total*rt/60/3:.0f} min @3 workers")
    print(f"  (+ Family B concat 单元预计 1.5-2x 更慢：embedding dim 翻倍 -> coreset kNN 更贵)")
    print(f"estimated VRAM per unit    : mean {est['peak_vram_mean_mb']} MB / max {est['peak_vram_max_mb']} MB")
    print(f"  -> 3 workers ~{3*(est['peak_vram_max_mb'] or 3300)/1024:.1f} GB << 24 GB (RTX 3090)")
    print("-" * L)
    print("[7A-O GPU REQUEST]  GPU units required: %d (Round0 %d + Round1 %d + Round2<=%d + Round3<=%d)"
          % (total, len(round0_units()), len(round1_units()), r2_max, r3_max))
    print("  no baseline rerun: B0/B1/B2/B3 all reconstructable from frozen 5A-H/6A/6B raw (0 GPU)")
    print("  A2_concat_fixed_projection: NOT IMPLEMENTED (see reference/policy_freeze.json)")
    print("  B2_grouped_projection: NOT RUN (B1 affordable)")
    print("=" * L)
    return {"rows": rows, "n_new": len(new_units), "estimate": est, "total_max": total}


def build_freeze() -> dict:
    return {
        "experiment": "7A-O",
        "title": "Overnight Module Screening — pluggable defect-preserving representation modules",
        "paper_stage": "Stage 7 — Method Candidate Screening (EXPLORATORY, not confirmatory)",
        "status": "EXPLORATORY SCREENING. The selected winner is NOT a final confirmatory result; "
                  "it must be frozen and re-validated under an independent confirmation protocol.",
        "research_question": ("Is there a simple feature-intervention / fusion module that yields a better "
                              "defect-preservation vs illumination-robustness trade-off than the strong "
                              "Uniform baseline alpha=0.40091275?"),
        "strong_baseline": {"name": "B2_uniform_040091275", "alpha_l2": UNIFORM_ALPHA, "alpha_l3": UNIFORM_ALPHA,
                            "origin": "Experiment 6B (commit ad26ec5), reused without rerun"},
        "baselines": baseline_defs(),
        "module_families": {
            "A_residual_original_preserving": {
                "A1_additive_normalized_residual": "F_out = F + lambda*N_hat, N=IN(F), "
                                                   "N_hat = N * rms_spatial(F) per (sample,channel)",
                "A3_energy_preserving": "F_out = (F + lambda*N_hat) rescaled to rms_spatial(F)",
                "A2_concat_fixed_projection": NOT_IMPLEMENTED["A2_concat_fixed_projection"],
                "note": "F + lambda*(IN(F)-F) is mathematically identical to alpha-interpolation and is "
                        "therefore NOT used; lambda scales are frozen below."},
            "B_dual_representation": {
                "B1_concat": "F_out = concat([F, gamma*N_hat], dim=1); embedding dim doubles",
                "B2_grouped_projection": NOT_IMPLEMENTED["B2_grouped_projection"],
                "dimensionality_change": "layer2 512->1024, layer3 1024->2048, concatenated 1536->3072",
                "fairness": "coreset ratio 0.1, num_neighbors 9, split, illumination and scoring unchanged; "
                            "embedding dim / bank size / runtime / VRAM recorded per unit"},
            "C_layer_selective": {"grid": {k: v for k, v in MODULE_SPECS.items() if FAMILY_OF[k] == "C"},
                                  "C0_equivalence": "C0 = Uniform alpha both layers -> must reproduce the "
                                                    "historical Uniform raw scores exactly"},
            "D_alternative_normalization": {"grid": {k: v for k, v in MODULE_SPECS.items() if FAMILY_OF[k] == "D"},
                                            "strength": UNIFORM_ALPHA, "affine": False, "groups_for_D2": 8,
                                            "stats_source": "input tensor itself only (no target/test statistics)"},
        },
        "frozen_assumptions": ASSUMPTIONS_FROZEN,
        "parameter_grid": MODULE_SPECS,
        "no_dense_sweep": True,
        "no_per_category_tuning": True,
        "categories": CATEGORIES,
        "seeds": SEEDS,
        "primary_metrics": {"preservation": "mean defect d-prime (higher better)",
                            "robustness": "mean |Delta NormalScore_z| (lower better)",
                            "secondary": ["per-category", "per-seed", "worst-category change",
                                          "negative transfer count (eps=0.10)", "image AUROC",
                                          "pixel AUROC / AUPRO", "runtime", "peak VRAM",
                                          "memory-bank size", "embedding dimension"]},
        "epsilon": EPS,
        "ranking_rule": {"tiers": TIER, "promote_tiers": list(PROMOTE_TIERS),
                         "primary_axis": "Pareto over (mean |Delta z| lower, mean d-prime higher) vs Uniform",
                         "tie_break": "higher d-prime, then lower |Delta z|"},
        "schedule": {
            "round0_smoke": {"categories": ["bottle"], "seeds": [0],
                             "purpose": "pipeline runnability + alpha=0 identity equivalence + Uniform "
                                        "equivalence + NaN/Inf/determinism/keys",
                             "units": round0_units()},
            "round1_broad": {"categories": ROUND1_CATEGORIES, "seeds": ROUND1_SEEDS, "units": len(round1_units())},
            "round2_confirm_like": {"categories": CATEGORIES, "seeds": SEEDS, "max_configs": 4,
                                    "rule": "at most Top-1 per family among promoted configs"},
            "round3_combination": {"condition": ">=2 different families promoted in Round 2",
                                   "rule": "exactly ONE combination of the top-2 family winners; composition "
                                           "order = (Winner-A step) then (Winner-B step)",
                                   "gate": "bottle/grid/hazelnut x seed0 first; extend only if clearly better "
                                           "than the best single module"},
        },
        "stop_rules": ["data corruption", "baseline equivalence FAIL", "mass NaN/Inf",
                       "frozen protocol mutation", "unresolvable OOM (3->2->1 workers, then STOP)",
                       "unknown manual repo changes", "definition ambiguity unresolvable without reading targets"],
        "leakage_guard": {"selection_set": "Round1/Round2 (exploration)",
                          "confirmation_set": "MUST be designed separately after freezing structure/params/metrics",
                          "no_threshold_change_after_results": True},
        "environment": {"gpu": "NVIDIA GeForce RTX 3090 24GB", "workers": 3,
                        "backend": "anomalib PatchCore wide_resnet50_2 [layer2,layer3], coreset 0.1, k=9"},
        "target_results_read": False,
        "git_head": git_head(),
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all", choices=["audit", "freeze", "all"])
    args = ap.parse_args()
    for d in (REF_DIR, CFG_DIR, RAW_DIR, SUM_DIR, FIG_DIR, LOG_DIR, SAN_DIR):
        d.mkdir(parents=True, exist_ok=True)
    print(f"[7A-O] git HEAD = {git_head()}")
    if args.stage in ("audit", "all"):
        stage_audit()
    if args.stage in ("freeze", "all"):
        fr = build_freeze()
        fp = REF_DIR / "policy_freeze.json"
        fp.write_text(json.dumps(fr, indent=2))
        digest = sha256_file(fp)
        (REF_DIR / "policy_freeze.sha256").write_text(digest + "  policy_freeze.json\n")
        write_csv(CFG_DIR / "module_specs.csv",
                  [{"config": k, "family": FAMILY_OF[k], "spec_l2": json.dumps(v["l2"]),
                    "spec_l3": json.dumps(v["l3"])} for k, v in MODULE_SPECS.items()]
                  + [{"config": k, "family": "smoke", "spec_l2": json.dumps(v["l2"]),
                      "spec_l3": json.dumps(v["l3"])} for k, v in SMOKE_SPECS.items()])
        print("=" * 112)
        print("[7A-O PRE-RUN FREEZE]")
        print("=" * 112)
        print(f"freeze JSON         : {fp.relative_to(ROOT)}")
        print(f"freeze SHA256       : {digest}")
        print(f"configs frozen      : {len(MODULE_SPECS)} screening + {len(SMOKE_SPECS)} smoke")
        print(f"strong baseline     : Uniform alpha={UNIFORM_ALPHA} (reused, no rerun)")
        print(f"target_results_read : {fr['target_results_read']}")
        print("=" * 112)


if __name__ == "__main__":
    main()
