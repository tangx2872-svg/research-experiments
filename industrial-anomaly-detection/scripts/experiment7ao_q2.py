"""Overnight Queue Q2 — Layer × Representation Composition（冻结规格）。

科研问题（与 Q0 不同）：
  1H/1H-S 已显示 normalization 对 layer2/layer3 的 defect response 不同；Q0 测的是
  representation-level intervention。Q2 问的是：
  **Original / invariant representation 的组合是否应该发生在特定 layer，而不是所有 layer 统一处理？**

4 个机械配置（禁止大 grid）：
  L0: layer2 = Original(α=0)            , layer3 = Uniform(α=0.40091275)
  L1: layer2 = Uniform(α=0.40091275)    , layer3 = Original(α=0)
  L2: layer2 = canonical residual (A3 energy-preserving, λ=0.5), layer3 = Uniform
  L3: layer2 = Uniform                  , layer3 = canonical residual

canonical residual = Q0 PRE-RUN grid 中最简单、定义明确、无训练的 A-family 形式
（A3 energy-preserving λ=0.5），**不根据 Q0/Q2 target performance 重新设计公式**。

输出隔离：raw 单元写入 results/experiment_7a_o_q2/raw/（不污染 Q0 的 sanity 作用域）；
summary/figures 按 addendum 要求写入 results/experiment_7a_o/{summary,figures}/q2_*。
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime
from pathlib import Path

import experiment7ao_config as c7

ROOT = c7.ROOT
Q2_OUT = ROOT / "results" / "experiment_7a_o_q2"
Q2_RAW, Q2_REF, Q2_LOG = Q2_OUT / "raw", Q2_OUT / "reference", Q2_OUT / "logs"
Q2_CFG = Q2_OUT / "configs"

UNIFORM = c7.UNIFORM_ALPHA
CANONICAL_RESIDUAL = c7.residual_step(0.50)      # A3: F + 0.5*IN(F)*rms(F)
ORIG = c7.alpha_step(0.0)
UNI = c7.alpha_step(UNIFORM)

Q2_SPECS = {
    "Q2L0_l2org_l3uni": c7.spec(ORIG, UNI),
    "Q2L1_l2uni_l3org": c7.spec(UNI, ORIG),
    "Q2L2_l2res50_l3uni": c7.spec(dict(CANONICAL_RESIDUAL), UNI),
    "Q2L3_l2uni_l3res50": c7.spec(UNI, dict(CANONICAL_RESIDUAL)),
}
Q2_ORDER = list(Q2_SPECS)

SCREEN_CATS, SCREEN_SEEDS = c7.ROUND1_CATEGORIES, [0]

ASSUMPTIONS = [
    "Original == alpha_in with alpha=0 (严格 short-circuit, 已验证与 5A-H B0 逐位等价)",
    "Uniform-Norm == alpha_in alpha=0.40091275 (已验证与 6B 逐位等价)",
    "canonical residual == A3 energy-preserving: F_out = (F + 0.5*N_hat) * rms(F)/rms(F_out_raw), N_hat = IN(F)*rms(F)",
    "step 施加位置与 Q0 完全一致：layer2 于 concat 前、layer3 于 bilinear upsample 前",
    "bank / coreset 0.1 / kNN 9 / split / illumination / scoring 与 Q0 完全相同（同一 run_config 代码路径）",
    "promotion rule 与 Q0 相同（Tier S/A/B vs Uniform α=0.40091275，同一子集）",
]


def git_head() -> str:
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()


def build_freeze() -> dict:
    hk = c7.historical_keys()
    cov = {n: sum(1 for c in SCREEN_CATS for s in SCREEN_SEEDS
                  if (c, s, c7.akey(sp["l2"].get("alpha", -1)), c7.akey(sp["l3"].get("alpha", -1))) in hk)
           for n, sp in Q2_SPECS.items()}
    return {
        "queue": "Q2",
        "title": "Layer x Representation Composition",
        "parent_experiment": "7A-O (Q0: CASE_C)",
        "status": "EXPLORATORY screening",
        "research_question": "Should the composition of Original vs invariant representation happen at a "
                             "specific layer instead of uniformly at all layers?",
        "relation_to_1H": "1H/1H-S showed layer-specific normalization response; Q2 tests whether that "
                          "translates into a layer-selective representation composition advantage",
        "configs": Q2_SPECS,
        "canonical_residual": {"name": "A3_lam050", "formula": CANONICAL_RESIDUAL,
                               "selection_rule": "Q0 PRE-RUN grid 中最简单、定义明确、无训练的 A-family "
                                                 "canonical residual；不依据 Q0/Q2 target performance 重新设计"},
        "uniform_alpha": UNIFORM,
        "screen": {"categories": SCREEN_CATS, "seeds": SCREEN_SEEDS, "units": len(Q2_SPECS) * len(SCREEN_CATS) * len(SCREEN_SEEDS)},
        "promotion": {"rule": "identical to Q0 section 14 tiers vs Uniform on the same subset",
                      "on_promotion": "extend promoted configs to 5 categories x 3 seeds"},
        "historical_coverage_of_screen_units": cov,
        "frozen_assumptions": ASSUMPTIONS,
        "baseline": "Uniform alpha=0.40091275 (restricted to the same subset for screening)",
        "no_dense_sweep": True,
        "target_results_read": False,
        "git_head": git_head(),
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }


def main() -> None:
    for d in (Q2_RAW, Q2_REF, Q2_LOG, Q2_CFG, c7.SUM_DIR, c7.FIG_DIR):
        d.mkdir(parents=True, exist_ok=True)
    fr = build_freeze()
    fp = Q2_REF / "q2_protocol_freeze.json"
    fp.write_text(json.dumps(fr, indent=2))
    sha = hashlib.sha256(fp.read_bytes()).hexdigest()
    (Q2_REF / "q2_protocol_freeze.sha256").write_text(sha + "  q2_protocol_freeze.json\n")
    c7.write_csv(Q2_CFG / "q2_specs.csv",
                 [{"config": k, "spec_l2": json.dumps(v["l2"]), "spec_l3": json.dumps(v["l3"])}
                  for k, v in Q2_SPECS.items()])
    L = 100
    print("=" * L)
    print("[Q2 PRE-RUN FREEZE] Layer x Representation Composition")
    print("=" * L)
    print(f"freeze      : {fp.relative_to(ROOT)}")
    print(f"sha256      : {sha}")
    print(f"configs     : {len(Q2_SPECS)}  {Q2_ORDER}")
    print(f"screen units: {fr['screen']['units']} (3 cat x seed0); 5x3 extension = "
          f"{len(Q2_SPECS) * len(c7.CATEGORIES) * len(c7.SEEDS)} units if all promoted")
    print(f"canonical residual : A3_lam050 (energy-preserving lambda=0.5), frozen in Q0 PRE-RUN grid")
    print(f"historical coverage of screen units : {fr['historical_coverage_of_screen_units']} -> all must be run")
    print(f"target_results_read : {fr['target_results_read']}")
    print("=" * L)


if __name__ == "__main__":
    main()
