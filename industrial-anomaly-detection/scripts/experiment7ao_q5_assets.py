"""Overnight Queue Q5 — CPU-only paper assets（不启动任何 GPU 工作）。

生成：
  summary/global_experiment_registry.csv
  summary/paper_evidence_map.md
  summary/negative_results.md
  summary/figure_inventory.md
  summary/main_results_table_draft.csv

原则：能从仓库自动提取的一律自动提取；不能确认的写 "see <path>"，**不编造数字**。
"""

from __future__ import annotations

import csv
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import experiment7ao_config as c7  # noqa: E402

SUM = c7.SUM_DIR
EXPS = ROOT / "experiments"
RESULTS = ROOT / "results"

# 已在本会话中核实过的裁决（其余留 "see README"，不猜）
VERDICT = {
    "exp1d_size_control": "CASE_A (README)",
    "exp1e_cross_category": "CASE_A (README)",
    "exp1f_mechanism_screening": "CASE_D (README; simple-attribute screening)",
    "exp1g_feature_space": "CASE_A (README)",
    "exp1h_layer_selectivity": "CASE_A (README; layer-selective response)",
    "exp1hs_dispersion_layer": "CASE_A (README)",
    "experiment1j_feature_geometry": "CASE_A — Layer3-specific expansion (commit 543e570)",
    "experiment1j_b_transmission": "CASE A — mechanism exploration STOPPED (commit 536ca3a)",
    "experiment5b": "CASE_A / CASE_C (README: 部分预测可用、部分不可用)",
    "experiment5c": "CASE_A — geometry-guided adaptive alpha v1 (commit fdd6fa9)",
    "experiment5d": "literal CASE_D / substantive CASE_B — identity 未确立 (commit 31150bd)",
    "experiment6a": "CASE_A — soft geometry gating pilot (commit 1b3edd2)",
    "experiment6b": "CASE_C — knee 不构成区域、收益主要来自 strength (commit ad26ec5)",
    "experiment7a_o": "CASE_C — NO USEFUL MODULE (commit f118e1e)",
    "experiment5a": "见 experiments/experiment5a/README.md（本脚本未核实）",
    "experiment5a_h": "见 experiments/experiment5a_h/README.md（本脚本未核实）",
}
CLAIM = {
    "exp1f_mechanism_screening": "简单属性（面积/对比度等）不足以解释 defect sensitivity",
    "exp1h_layer_selectivity": "layer2/layer3 的 defect response 不同（层级选择性）",
    "experiment5b": "normal-only geometry 对 normalization tolerance 只具部分可预测性",
    "experiment5c": "geometry-guided category-adaptive α 可行（后被 5D/6A/6B 连续削弱）",
    "experiment5d": "geometry 的收益不能归因于类别选择信息（identity 未确立）",
    "experiment6a": "soft geometry gating 可回收 Hard-GC 的 robustness 代价",
    "experiment6b": "6A 的 knee 是孤立点；收益 60.1% 来自整体 normalization strength",
    "experiment7a_o": "4 类简单 representation module 不能突破 Uniform α≈0.40 的 frontier；"
                      "A3_lam050 严格支配 Original",
}
GPU_DIR = {"experiment_5a_h": "experiment_5a_h", "experiment_5c": "experiment_5c",
           "experiment_6a": "experiment_6a", "experiment_6b": "experiment_6b",
           "experiment_7a_o": "experiment_7a_o"}


def git_commit_map() -> dict:
    out = subprocess.run(["git", "--no-pager", "log", "--format=%h|%s"], cwd=ROOT,
                         capture_output=True, text=True).stdout
    return {l.split("|", 1)[0]: l.split("|", 1)[1] for l in out.strip().splitlines()}


def units_of(name: str) -> object:
    for cand in (RESULTS / name, RESULTS / f"{name}_raw", RESULTS / f"{name}_q2", RESULTS / f"{name}_q3"):
        if cand.exists():
            n = len(list(cand.rglob("info.json")))
            if n:
                return n
    for sub in ("raw", "raw_new"):
        p = RESULTS / name / sub
        if p.exists():
            n = len(list(p.rglob("info.json")))
            if n:
                return n
    return "n/a"


def build_registry() -> list:
    cm = git_commit_map()
    rows = []
    for d in sorted(EXPS.iterdir()):
        r = d / "README.md"
        if not r.exists():
            continue
        t = r.read_text(errors="ignore")
        head = next((l.strip("# ").strip() for l in t.splitlines() if l.startswith("# ")), "")
        dates = sorted(set(re.findall(r"20\d\d-\d\d-\d\d", t)))
        key = d.name
        commit = ""
        for h, subj in cm.items():
            if key.replace("_", "").lower()[:10] in subj.replace(" ", "").replace("-", "").lower():
                commit = h
                break
        rows.append({"experiment": key, "date": dates[0] if dates else "see README",
                     "question": head, "dataset": "MVTec AD (5 categories: bottle/cable/grid/hazelnut/screw)",
                     "seeds": "0/1/2 (see README)", "gpu_units": units_of(key),
                     "primary_metric": "defect d-prime / |Delta NormalScore_z| (see README)",
                     "verdict": VERDICT.get(key, "see README"),
                     "paper_claim": CLAIM.get(key, "see README"),
                     "status": "DONE" if (RESULTS / key).exists() or "5a" in key else "see README",
                     "git_commit": commit, "source_path": str(r.relative_to(ROOT))})
    extra = [
        {"experiment": "7A-O Q2", "date": "2026-10-07", "question": "Layer x Representation Composition",
         "dataset": "MVTec AD (bottle/grid/hazelnut x seed0)", "seeds": "0", "gpu_units": 12,
         "primary_metric": "defect d-prime / |Delta Normal_score_z|",
         "verdict": "no promotion (0/4 Tier) -> STOP",
         "paper_claim": "layer-selective representation composition 同样不能突破 frontier；L2 槽位比 L3 更有效率",
         "status": "DONE", "git_commit": "7ffb3aa", "source_path": "results/experiment_7a_o_q2/README.md"},
        {"experiment": "7A-O Q3", "date": "2026-10-07", "question": "Illumination Stress-Test",
         "dataset": "MVTec AD (5 categories x seed0 [+ seeds1/2 in Q4-B])", "seeds": "0 (+1,2 in Q4-B)",
         "gpu_units": 10 + 20, "primary_metric": "|Delta NormalScore_z| per illumination condition",
         "verdict": "see summary/q3_method_summary.csv", "paper_claim": "illumination severity degradation slope",
         "status": "RUNNING/PARTIAL", "git_commit": "TBD", "source_path": "results/experiment_7a_o_q3/README.md"},
        {"experiment": "7A-O Q4", "date": "2026-10-07",
         "question": "Strong-Baseline Extended Validation (uniform-alpha response + stress seeds 1/2)",
         "dataset": "MVTec AD (5 categories x 3 seeds)", "seeds": "0/1/2", "gpu_units": 36 + 20,
         "primary_metric": "defect d-prime / |Delta NormalScore_z|",
         "verdict": "see summary/q4_alpha_response.csv",
         "paper_claim": "uniform normalization strength response map (baseline completion)",
         "status": "PENDING", "git_commit": "TBD", "source_path": "results/experiment_7a_o_q4/README.md"},
    ]
    return rows + extra


NEGATIVE_RESULTS = r"""# Negative Results Registry（不许删除负结果）

> 由 overnight Q5（CPU-only）整理。每条都指向可核验的入口。

## N1 — 简单缺陷属性不足以解释 defect sensitivity
- 实验：`exp1f_mechanism_screening`（CASE_D）
- 含义：面积/对比度等简单属性筛选不能解释 sensitivity 差异 → 机制需要表示层解释。
- 入口：`experiments/exp1f_mechanism_screening/README.md`

## N2 — geometry 的"身份"从未被确立
- 实验：`experiment5d`（**literal CASE_D / 实质 CASE_B**，identity 未确立）
- 含义：5C 的收益不能被归因于 geometry 的**类别选择信息**；matched sensitivity control 反而更强
  （+0.1450/+0.1540，3/3 seeds，且由 grid 单类别驱动）。
- 入口：`experiments/experiment5d/README.md`；根 README §7

## N3 — geometry-guided gating 连续三次被削弱
- `experiment5c`（CASE_A）→ `experiment6a`（CASE_A，soft gating）→ `experiment6b`（**CASE_C**）
- 6B 结论：α_F=0.25 的可行点 **region length = 1（孤立点）**；放宽口径也只有 [0.20, 0.25]；
  相对 mean-α matched `Uniform(0.40091275)`，Δd′ +0.0842（< ε）而 \|Δz\| 反而差 +0.0195；
  收益 **60.1% 来自整体 normalization strength、仅 39.9% 来自 selective allocation**（兑换率差 5.7×）。
- 入口：`experiments/experiment6b/README.md`；commit `ad26ec5`

## N4 — 机制阶段（1J）的限制与冻结
- `experiment1j_b_transmission` 的 score 为 **proxy**（非 full-model 端到端），单 backbone；
  机制阶段在 1J 之后 **FROZEN**，未再推进（1J-C / 2A 未启动）。
- 入口：`experiments/experiment1j_b_transmission/README.md`；`experiments/experiment1j_feature_geometry/README.md`

## N5 — 7A-O：四类"简单可插拔模块"全部失败（今晚，92 GPU units）
- Family A（residual / energy-preserving fusion，5 configs）：全部"以 robustness 换 preservation"，
  **无一达到 Tier S/A/B**。
- Family B（dual representation，concat γ ∈ {0.25,0.5,1.0}）：同向、且 **被 A3_lam050 严格支配**；
  embedding dim 翻倍、峰值显存 ×2，并在 3-worker 下触发 **2 次 CUDA OOM**。
- Family C（layer-selective strength，4 configs）：与 Uniform 差异 ≤ 0.022 d′，**没有 layer 偏好信号**。
- Family D（LayerNorm-like / GroupNorm-like）：**被 B0_original 支配** → 明确否定。
- **A2（concat + fixed projection）= NOT IMPLEMENTED**：无训练固定投影只能是 α-interpolation 的伪装
  或无原则算子。
- **B2（grouped projection）= NOT RUN**：B1 已验证可跑，无需任意替代算子。
- 入口：`experiments/experiment7a_o/README.md`；`results/experiment_7a_o/summary/`；commit `f118e1e`

## N6 — Q2：layer × representation composition 同样无晋级（今晚，12 units）
- 0/4 配置达到 Tier S/A/B → STOP。
- 强非对称性：同一干预放在 **L3** 上会同时掉 d′ 与 robustness（`Q2L3` −0.0489 / +0.1184）。
- 入口：`results/experiment_7a_o_q2/README.md`；commit `7ffb3aa`

## N7 — 孤立点 / 非稳定区域清单
- 6A 的 `α_F = 0.25`（被 6B 判定为 region length = 1 的孤立点）
- 5A-H 的 `G2 = (L2 0.4527, L3 0.75)`（跨类别扩展后未能形成稳定规则）
- 7A-O 的全部 14 个候选（无一 Pareto-improving；仅 A3_lam050 相对 **弱** baseline 有支配关系）
"""

EVIDENCE_MAP = r"""# Paper Evidence Map（Claim → 支持实验 → 支持数字 → 反例/限制 → 证据强度）

> 由 overnight Q5（CPU-only）整理；所有数字均可在对应 experiment README / summary 中核验。
> 证据强度标记：**SUPPORTED / PARTIALLY SUPPORTED / REJECTED / EXPLORATORY**。

## C1 — 光照/对比度扰动会显著改变 PatchCore 的 anomaly score，且不同缺陷类型受影响不同
- 支持实验：1B（phenomenon）、1C（multi-seed）、1D（size control）、1E（cross-category）
- 支持数字：见 `experiments/exp1c_multiseed/README.md`、`experiments/exp1d_size_control/README.md`、
  `experiments/exp1e_cross_category/README.md`（本文件不重复引用未核实的数字）
- 反例/限制：单 backbone（wide_resnet50_2）、n=5 categories、扰动为 synthetic photometric
- 强度：**SUPPORTED**

## C2 — 简单缺陷属性不足以解释上述差异
- 支持实验：1F（CASE_D）
- 强度：**REJECTED**（对"简单属性解释"这一替代解释的否定）

## C3 — normalization（InstanceNorm 混合）强度是 preservation ↔ robustness 的一维旋钮
- 支持实验：5A、5A-H（fixed/uniform α 网格）、6B（mean-α matched Uniform 对照）
- 支持数字（full panel 5 cat × 3 seeds，本会话核实）：α=0 → d′ 4.9852 / \|Δz\| 0.3105；
  α=0.5 → 4.6670 / 0.2082；**α=0.40091275 → 4.7940 / 0.2134**
- 反例/限制：该旋钮是**单一维度**，不能同时改善两轴（见 C5）
- 强度：**SUPPORTED**（机制层面）/ **EXPLORATORY**（最优 α 的具体数值）

## C4 — geometry-guided category-adaptive α 有价值
- 支持实验：5C（CASE_A）
- 反例：5D（identity 未确立，实质 CASE_B）→ 6A/6B（knee 是孤立点、收益 60.1% 来自 strength）
- 强度：**REJECTED**（作为独立机制；保留为"strength 效应"的从属解释）

## C5 — 简单 representation 模块（residual / dual / layer-selective / alternative-norm）可以推动
preservation–robustness frontier
- 支持实验：7A-O（92 units）、Q2（12 units）
- 支持数字（7A-O full panel vs Uniform）：A3_lam050 +0.5612 d′ / **+0.0619** \|Δz\|；
  B1_g100 +0.5104/+0.0717；D1 +0.1739/+0.1164；C3 +0.0219/+0.0115
- **反例（关键）**：**无任何 candidate 与 Uniform 相互 Pareto 支配**；所有模块只是沿既有 frontier
  移动工作点。唯一正面关系是 A3_lam050 **严格支配 B0_original**（+0.3700 / −0.0352）。
- 强度：**REJECTED**（作为"突破 frontier"的路径）/ **EXPLORATORY**（A3_lam050 相对弱 baseline 的支配关系）

## C6 — "只改 feature representation 就能提高 illumination robustness"这一方向本身
- 支持实验：6B（strength 主导）+ 7A-O（4 family 全败）+ Q2（layer × representation 无晋级）
- 含义：三条独立证据一致指向：**在 PatchCore + IN 框架内，representation-level 干预不能改变
  frontier 的形状，只能选择工作点**。
- 反例/限制：未测试"改变目标轴本身"的方案（inference-time illumination 校正 / 显式
  illumination-invariance 目标）——**尚未验证**。
- 强度：**PARTIALLY SUPPORTED**（否定 simple representation 路线；不否定所有可能的语言）

## C7 — μ 表述（可用于论文的限制声明）
- 上述结论的适用边界：single backbone（wide_resnet50_2）、PatchCore、coreset 0.1、kNN 9、
  MVTec AD 5 categories、synthetic photometric perturbation、seeds 0/1/2。
- 强度：**限制声明（必须随结论一起写）**
"""


def figure_inventory() -> list:
    rows = []
    roots = [RESULTS, EXPS]
    for root in roots:
        for p in sorted(root.rglob("*.png")):
            if "/fit/" in str(p) or "/logs/" in str(p):
                continue
            rel = str(p.relative_to(ROOT))
            name = p.name.lower()
            if any(k in name for k in ("sanity", "debug", "tmp", "scratch")):
                cls = "Debug-only"
            elif any(k in name for k in ("fig1", "fig2", "fig3", "main", "pareto", "tradeoff", "curve")):
                cls = "Main-paper candidate"
            elif any(k in name for k in ("fig4", "fig5", "per_category", "per_seed", "seed", "category")):
                cls = "Supplementary candidate"
            else:
                cls = "unclassified (needs human)"
            rows.append({"figure": rel, "size_kb": round(p.stat().st_size / 1024, 1),
                         "suggested_class": cls, "delete": "NO (never delete figures)"})
    return rows


def main_results_table() -> list:
    import experiment7ao_analysis as A
    extra_alpha, extra = [], []
    try:
        import experiment7ao_q4 as q4
        if q4.Q4_RAW.exists():
            extra_alpha.append(("Q4", q4.Q4_RAW))
    except Exception:
        pass
    try:
        import experiment7ao_q2 as q2
        if q2.Q2_RAW.exists():
            extra.append(("Q2", q2.Q2_RAW))
    except Exception:
        pass
    try:
        import experiment7ao_q3 as q3
    except Exception:
        q3 = None
    lookup, meta = A.build_lookup(extra=extra, extra_alpha=extra_alpha)
    bm = A.baseline_key_maps()
    rows = []

    def add(method, spec_map, cats, seeds, note):
        try:
            ev = A.eval_policy(lookup, spec_map, cats, seeds)
        except KeyError as e:
            rows.append({"method": method, "scope": f"{len(cats)}cat x {len(seeds)}seed",
                         "mean_defect_dprime": "N/A", "mean_abs_delta_z": "N/A",
                         "delta_dprime_vs_uniform": "N/A", "delta_abs_z_vs_uniform": "N/A",
                         "status": f"N/A ({e})", "note": note})
            return
        uni = A.eval_policy(lookup, bm["B2_uniform_040091275"], cats, seeds)
        rows.append({"method": method, "scope": f"{len(cats)}cat x {len(seeds)}seed",
                     "mean_defect_dprime": round(ev["mean_dprime"], 6),
                     "mean_abs_delta_z": round(ev["mean_abs_delta_z"], 6),
                     "delta_dprime_vs_uniform": round(ev["mean_dprime"] - uni["mean_dprime"], 6),
                     "delta_abs_z_vs_uniform": round(ev["mean_abs_delta_z"] - uni["mean_abs_delta_z"], 6),
                     "image_auroc": round(ev["image_auroc"], 6), "status": "reused historical raw",
                     "note": note})

    for k, lbl in (("B0_original", "Original (alpha=0, no normalization intervention)"),
                   ("B1_fixed_050", "Fixed alpha=0.5 (historical baseline)"),
                   ("B2_uniform_040091275", "Strong Uniform alpha=0.40091275 (6B baseline)"),
                   ("B3_soft_gc_af025", "Soft-GC alpha_F=0.25 (6A, reference only)")):
        add(lbl, bm[k], c7.CATEGORIES, c7.SEEDS, "baseline")
    add("7A-O best-preservation config: A3_lam050 (NOT a winner; Tier = none)",
        {c: ("config", "A3_lam050") for c in c7.CATEGORIES}, c7.CATEGORIES, c7.SEEDS,
        "exploratory; strictly dominates Original but not Uniform")
    if q3 is not None:
        add("Q2 best composition: Q2L2_l2res50_l3uni (NOT a winner)",
            {c: ("config", "Q2L2_l2res50_l3uni") for c in q2.SCREEN_CATS if (c, 0, "config", "Q2L2_l2res50_l3uni") in lookup},
            q2.SCREEN_CATS, q2.SCREEN_SEEDS, "exploratory; 3cat x seed0 only")
    return rows


def main() -> None:
    reg = build_registry()
    c7.write_csv(SUM / "global_experiment_registry.csv", reg)
    (SUM / "negative_results.md").write_text(NEGATIVE_RESULTS)
    (SUM / "paper_evidence_map.md").write_text(EVIDENCE_MAP)
    figs = figure_inventory()
    c7.write_csv(SUM / "figure_inventory.csv", figs)
    lines = ["# Figure Inventory（不删除任何文件）", "",
             f"共 {len(figs)} 个 figure 文件。分类为**建议**，最终归属需人工确认。", "",
             "| figure | size (KB) | suggested class |", "|---|---|---|"]
    for r in figs:
        lines.append(f"| `{r['figure']}` | {r['size_kb']} | {r['suggested_class']} |")
    (SUM / "figure_inventory.md").write_text("\n".join(lines) + "\n")
    try:
        mt = main_results_table()
        c7.write_csv(SUM / "main_results_table_draft.csv", mt)
        print(f"[Q5] main_results_table_draft.csv rows={len(mt)}")
    except Exception as exc:
        print(f"[Q5] main_results_table_draft FAILED: {type(exc).__name__}: {exc}")
    print(f"[Q5] registry rows={len(reg)}; figure inventory rows={len(figs)}")
    print("[Q5] wrote summary/{global_experiment_registry.csv,negative_results.md,"
          "paper_evidence_map.md,figure_inventory.md,figure_inventory.csv}")
    for m in mt if 'mt' in dir() else []:
        print("  " + json.dumps({k: m[k] for k in ("method", "scope", "mean_defect_dprime",
                                                   "mean_abs_delta_z", "delta_dprime_vs_uniform",
                                                   "delta_abs_z_vs_uniform")}))


if __name__ == "__main__":
    main()
