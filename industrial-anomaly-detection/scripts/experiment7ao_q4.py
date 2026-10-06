"""Overnight Queue Q4 — Strong-Baseline Extended Validation（冻结规格）。

目的（addendum）：6B 发现 `Uniform α≈0.40091275` 本身就是极强 baseline。Q4 **不找新方法**，
只把 "uniform normalization strength → preservation / robustness" 的响应图补完整，
使 baseline comparison 更完整。

允许内容（严格）：只用**历史已定义的 α points** + **历史已有 illumination protocol** +
已有 5 categories / seeds / metrics。禁止 new dataset / backbone / per-category tuning / dense α sweep。

资产审计（2026-10-07，seen by historical (cat,seed,alpha_l2,alpha_l3) 键）：
  uniform-α 覆盖（5 cat x 3 seed = 15）：
    0.0 15/15 · 0.125 6/15 · 0.20 6/15 · 0.25 6/15 · 0.30 6/15 ·
    0.40091275 15/15 · 0.5 15/15 · 0.601369 15/15 · 0.801825 15/15 · 0.75 1/15 · 1.0 1/15
  → 只补 {0.125, 0.20, 0.25, 0.30} 缺失的 cable/hazelnut/screw x 3 seeds（各 9 units）。

Q4-B：把 Q3 的 illumination stress-test 从 seed0 扩展到 seed1/2（同一协议、同一 12 conditions、
同一 2 个方法），使 `method x category x seed x condition` 完整。
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime
from pathlib import Path

import experiment7ao_config as c7

ROOT = c7.ROOT
Q4_OUT = ROOT / "results" / "experiment_7a_o_q4"
Q4_RAW, Q4_REF, Q4_LOG, Q4_CFG = Q4_OUT / "raw", Q4_OUT / "reference", Q4_OUT / "logs", Q4_OUT / "configs"

# Q4-A：历史已定义的 uniform α points（补齐到 5x3）
ALPHA_POINTS = [0.125, 0.20, 0.25, 0.30]
Q4_ALPHA_SPECS = {f"Q4a{str(a).replace('.', '')}": c7.spec(c7.alpha_step(a), c7.alpha_step(a))
                  for a in ALPHA_POINTS}
Q4_ALPHA_OF = {k: a for k, a in zip(Q4_ALPHA_SPECS, ALPHA_POINTS)}

# 历史已有的 (cat, seed)（这些不重跑，直接复用历史 raw）
HIST_HAVE = {(c, s) for c in ("bottle", "grid") for s in c7.SEEDS}
Q4_A_UNITS = [(f"Q4a{str(a).replace('.', '')}", c, s)
              for a in ALPHA_POINTS for c in c7.CATEGORIES for s in c7.SEEDS
              if (c, s) not in HIST_HAVE]

# Q4-B：Q3 stress-test 的 seeds 1/2 扩展
Q4_B_METHODS = ["M0_original", "M1_uniform"]
Q4_B_SEEDS = [1, 2]
Q4_B_UNITS = [(m, c, s) for m in Q4_B_METHODS for c in c7.CATEGORIES for s in Q4_B_SEEDS]

ASSUMPTIONS = [
    "Q4-A 只运行历史缺失单元；bottle/grid x 3 seeds 的 Q4a* 直接复用 6A/6B raw（0 GPU）",
    "Q4-B 复用 Q3 完全相同的 12 conditions 与 2 个方法，只补 seeds 1/2",
    "不使用任何新的 α point、新的 illumination family、新的 category/seed/metric",
    "所有模型 fit 走 Q0 的同一代码路径（experiment7ao_runner.patched_fit_dual_model）",
]


def git_head() -> str:
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()


def build_freeze() -> dict:
    return {
        "queue": "Q4",
        "title": "Strong-Baseline Extended Validation",
        "status": "EXPLORATORY / asset completion (no new method, no new benchmark)",
        "purpose": "complete the uniform-normalization-strength response map and the illumination "
                   "condition x seed coverage of the strongest baselines",
        "asset_audit": {
            "uniform_alpha_coverage_before": {"0.0": 15, "0.125": 6, "0.20": 6, "0.25": 6, "0.30": 6,
                                              "0.40091275": 15, "0.5": 15, "0.601369": 15, "0.801825": 15},
            "q4a_new_units": len(Q4_A_UNITS),
            "q4b_new_units": len(Q4_B_UNITS),
            "total_new_units": len(Q4_A_UNITS) + len(Q4_B_UNITS),
        },
        "q4a": {"alpha_points": ALPHA_POINTS, "specs": Q4_ALPHA_SPECS,
                "historical_reuse": sorted(HIST_HAVE),
                "units": [f"{n}:{c}:{s}" for n, c, s in Q4_A_UNITS]},
        "q4b": {"methods": Q4_B_METHODS, "seeds": Q4_B_SEEDS,
                "conditions": "identical to Q3 (12 conditions, brightness/gamma x mild/medium/strong)",
                "units": [f"{m}:{c}:{s}" for m, c, s in Q4_B_UNITS]},
        "forbidden": ["new dataset", "new backbone", "per-category alpha tuning", "dense alpha sweep",
                      "new illumination family"],
        "frozen_assumptions": ASSUMPTIONS,
        "target_results_read": False,
        "git_head": git_head(),
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }


def main() -> None:
    for d in (Q4_RAW, Q4_REF, Q4_LOG, Q4_CFG, c7.SUM_DIR, c7.FIG_DIR):
        d.mkdir(parents=True, exist_ok=True)
    fr = build_freeze()
    fp = Q4_REF / "q4_protocol_freeze.json"
    fp.write_text(json.dumps(fr, indent=2))
    sha = hashlib.sha256(fp.read_bytes()).hexdigest()
    (Q4_REF / "q4_protocol_freeze.sha256").write_text(sha + "  q4_protocol_freeze.json\n")
    print("=" * 100)
    print("[Q4 PRE-RUN FREEZE] Strong-Baseline Extended Validation")
    print("=" * 100)
    print(f"freeze      : {fp.relative_to(ROOT)}")
    print(f"sha256      : {sha}")
    print(f"Q4-A units  : {len(Q4_A_UNITS)}  (alpha {ALPHA_POINTS} x cable/hazelnut/screw x 3 seeds)")
    print(f"Q4-B units  : {len(Q4_B_UNITS)}  (2 methods x 5 categories x seeds {Q4_B_SEEDS})")
    print(f"total new   : {len(Q4_A_UNITS) + len(Q4_B_UNITS)}")
    print(f"historical reuse (0 GPU): {sorted(HIST_HAVE)} for Q4-A")
    print(f"target_results_read : {fr['target_results_read']}")
    print("=" * 100)
    ser = [{"config": k, "alpha": a, "family": "Q4A"} for k, a in Q4_ALPHA_OF.items()]
    c7.write_csv(Q4_CFG / "q4a_specs.csv", ser)
    print("\n".join(f"{n}:{c}:{s}" for n, c, s in Q4_A_UNITS[:6]) + " ...")
    print("\n".join(f"{m}:{c}:{s}" for m, c, s in Q4_B_UNITS[:6]) + " ...")


if __name__ == "__main__":
    main()
