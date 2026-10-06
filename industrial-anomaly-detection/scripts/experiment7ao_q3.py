"""Overnight Queue Q3 — Illumination Stress-Test（冻结规格）。

审计结论（Q3 启动前完成）：项目历史**唯一**的、实现明确且已被反复使用的 illumination 协议是
`experiment1_illumination_tradeoff.apply_photometric(img, type, level)`，family = {brightness, gamma}，
历史 severity = {0.7, 1.3}（±30%）。项目**没有** color-temperature / exposure / blur 等实现。
→ 因此 Q3 **不发明新的 augmentation family**，只在既有实现上把 severity 系统化为 3 档（±10% / ±30% / ±50%）。

方法集合（严格按 addendum 的枚举；Q0 = CASE_C 故"Q0 winner"槽位为空，**不**用非 winner 顶替）：
  M0 = Original      (α_l2 = α_l3 = 0)
  M1 = Uniform        (α_l2 = α_l3 = 0.40091275)   <- 当前最强方法
  (+ Q2 winner，若 Q2 产生 Tier S/A/B 晋级者；由 --methods 注入)

关键性质：medium 档（0.7 / 1.3）与历史 5A-H/6A/6B 完全同条件，因此 **M0/M1 的 medium 分数必须与
历史逐位一致** → 同时充当 baseline equivalence 检查。
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime
from pathlib import Path

import experiment7ao_config as c7

ROOT = c7.ROOT
Q3_OUT = ROOT / "results" / "experiment_7a_o_q3"
Q3_RAW, Q3_REF, Q3_LOG, Q3_CFG = Q3_OUT / "raw", Q3_OUT / "reference", Q3_OUT / "logs", Q3_OUT / "configs"

UNIFORM = c7.UNIFORM_ALPHA
METHOD_SPECS = {
    "M0_original": c7.spec(c7.alpha_step(0.0), c7.alpha_step(0.0)),
    "M1_uniform": c7.spec(c7.alpha_step(UNIFORM), c7.alpha_step(UNIFORM)),
}
BASELINE_METHODS = ["M0_original", "M1_uniform"]

# severity 在**读取任何 Q3 target 之前**冻结：既有 0.7/1.3（medium）+ 对称扩展 ±10% / ±50%
SEVERITIES = [
    ("mild", 0.9), ("mild", 1.1),
    ("medium", 0.7), ("medium", 1.3),
    ("strong", 0.5), ("strong", 1.5),
]
FAMILIES = ["brightness", "gamma"]
CONDITIONS = [(f"{fam}_{lv}", fam, lv, sev) for fam in FAMILIES for sev, lv in SEVERITIES]
MEDIUM_CONDS = [c for c in CONDITIONS if c[3] == "medium"]
SEVERITY_X = {"mild": 0.1, "medium": 0.3, "strong": 0.5}   # |1 - level|，用于 slope 的 x 轴

STAGE1_CATS, STAGE1_SEEDS = c7.CATEGORIES, [0]

ASSUMPTIONS = [
    "photometric perturbation 原样复用 experiment1_illumination_tradeoff.apply_photometric（不新增实现）",
    "brightness: img*level ; gamma: img**(1/level) ; clamp 到 [0,1]",
    "severity 三档 = ±10% (mild) / ±30% (medium, 历史值) / ±50% (strong)，读取 target 前冻结",
    "robustness 指标沿用 5A-H 冻结口径：mu_g/sigma_g 取自 clean_good，Delta_z = mean((s_shift - mu_g)/sigma_g)",
    "preservation 指标 d' 与 illumnation condition 无关（defect 图不施加扰动），沿用 5A-H 公式",
    "medium 档必须与历史 raw 逐位一致（作为 baseline equivalence check）",
    "bank / coreset 0.1 / kNN 9 / split / seed 与 Q0 完全相同（同一 fit 代码路径）",
]


def git_head() -> str:
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()


def build_freeze(methods: list) -> dict:
    return {
        "queue": "Q3",
        "title": "Illumination Stress-Test",
        "parent_experiments": ["7A-O (Q0: CASE_C)", "Q2 (layer x representation)"],
        "status": "EXPLORATORY",
        "purpose": "把当前最强方法放到更系统的 illumination severity 下压力测试，得到 degradation slope",
        "protocol_audit": {
            "existing_implementation": "experiment1_illumination_tradeoff.apply_photometric",
            "existing_families": ["brightness", "gamma"],
            "existing_severities": [0.7, 1.3],
            "absent_families": ["color_temperature", "exposure", "blur", "noise"],
            "decision": "reuse the existing implementation only; extend severity symmetrically; "
                        "do NOT invent a new augmentation family or a new benchmark",
        },
        "methods": {m: METHOD_SPECS.get(m, "injected (Q2 winner)") for m in methods},
        "method_selection_rule": "addendum enumeration: Original | Uniform | Q0 winner (none: Q0 = CASE_C) "
                                 "| Q2 winner (if any)",
        "conditions": [{"name": n, "family": f, "level": lv, "severity": sev} for n, f, lv, sev in CONDITIONS],
        "n_conditions": len(CONDITIONS),
        "severity_x_axis": SEVERITY_X,
        "stage1": {"categories": STAGE1_CATS, "seeds": STAGE1_SEEDS,
                   "units": len(methods) * len(STAGE1_CATS) * len(STAGE1_SEEDS)},
        "stage2_rule": "extend to seeds 1/2 only if a winner's advantage over Uniform holds across most "
                       "illumination conditions (no winner in Q0 -> stage 2 not triggered)",
        "metrics": ["defect d-prime", "mean |Delta_z| per condition", "AUROC", "worst-condition performance",
                    "negative transfer count", "degradation slope vs severity"],
        "frozen_assumptions": ASSUMPTIONS,
        "target_results_read": False,
        "git_head": git_head(),
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--methods", default=",".join(BASELINE_METHODS))
    args = ap.parse_args()
    for d in (Q3_RAW, Q3_REF, Q3_LOG, Q3_CFG, c7.SUM_DIR, c7.FIG_DIR):
        d.mkdir(parents=True, exist_ok=True)
    methods = [m for m in args.methods.split(",") if m]
    fr = build_freeze(methods)
    fp = Q3_REF / "q3_protocol_freeze.json"
    fp.write_text(json.dumps(fr, indent=2))
    sha = hashlib.sha256(fp.read_bytes()).hexdigest()
    (Q3_REF / "q3_protocol_freeze.sha256").write_text(sha + "  q3_protocol_freeze.json\n")
    c7.write_csv(Q3_CFG / "q3_conditions.csv",
                 [{"condition": n, "family": f, "level": lv, "severity": sev} for n, f, lv, sev in CONDITIONS])
    print("=" * 100)
    print("[Q3 PRE-RUN FREEZE] Illumination Stress-Test")
    print("=" * 100)
    print(f"freeze       : {fp.relative_to(ROOT)}")
    print(f"sha256       : {sha}")
    print(f"methods      : {methods}")
    print(f"conditions   : {len(CONDITIONS)} (2 families x 3 severities x 2 directions)")
    print(f"stage1 units : {fr['stage1']['units']} (5 categories x seed0)")
    print(f"medium conds : {[c[0] for c in MEDIUM_CONDS]} -> 必须与历史逐位一致")
    print(f"target_results_read : {fr['target_results_read']}")
    print("=" * 100)


if __name__ == "__main__":
    main()
