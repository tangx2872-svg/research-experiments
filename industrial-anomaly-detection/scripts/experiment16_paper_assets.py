"""Experiment 16 — O1 Paper Asset Pack / O3 Main Table Draft / O4 Ablation Inventory（全 CPU，不跑新实验）。

只整理**轻量** CSV/JSON/PNG/PDF/MD；不复制模型、checkpoint、特征 bank。
缺失数据一律标 `TODO — requires experiment`，**不得编造**。
"""
from __future__ import annotations

import csv
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

PA = ROOT / "results" / "paper_assets"
E16 = ROOT / "results" / "experiment_16_final_validation"

TABLES = [
    (E16 / "analysis" / "final_50unit_table.csv", "final_50unit_table.csv"),
    (E16 / "analysis" / "final_leaderboard.csv", "final_leaderboard.csv"),
    (E16 / "analysis" / "seed_block_stability.csv", "seed_block_stability.csv"),
    (E16 / "analysis" / "robustness_cost.csv", "robustness_cost.csv"),
    (E16 / "analysis" / "statistics.json", "statistics.json"),
    (E16 / "analysis" / "sanity_checks.csv", "exp16_sanity_checks.csv"),
    (ROOT / "results" / "experiment_15" / "analysis" / "scoreside_per_unit.csv", "exp15_scoreside_per_unit.csv"),
    (E16 / "EXPERIMENT_DEBT_AUDIT.md", "experiment_debt_audit.md"),
    (ROOT / "results" / "experiment_14" / "analysis" / "candidate_leaderboard.csv", "exp14_candidate_leaderboard.csv"),
    (ROOT / "results" / "experiment_14" / "analysis" / "final_leaderboard.csv", "exp14_tier_leaderboard.csv"),
    (ROOT / "results" / "experiment_15" / "analysis" / "scoreside_summary.csv", "exp15_scoreside_summary.csv"),
    (ROOT / "results" / "experiment_15" / "analysis" / "bootstrap_ci.json", "exp15_bootstrap_ci.json"),
    (ROOT / "results" / "experiment_15" / "analysis" / "sensitivity_curve.json", "exp15_sensitivity_curve.json"),
    (ROOT / "results" / "experiment_11_c2_adaptive" / "README.md", "exp11_readme.md"),
    (ROOT / "results" / "experiment_12" / "README.md", "exp12_readme.md"),
    (ROOT / "PAPER_EVIDENCE_MAP.md", "paper_evidence_map.md"),
    (ROOT / "HELD_OUT_VALIDATION_PLAN.md", "held_out_validation_plan.md"),
]
FIGS = [(E16 / "figures", n) for n in ("fig1_pass_catastrophic", "fig2_ddp_distribution", "fig3_worst_tail",
                                       "fig4_robustness_preservation_frontier", "fig5_seed_block_stability")]


def main() -> None:
    (PA / "tables").mkdir(parents=True, exist_ok=True)
    (PA / "figures").mkdir(parents=True, exist_ok=True)
    (PA / "notes").mkdir(parents=True, exist_ok=True)
    copied, missing = [], []
    for src, name in TABLES:
        if src.exists():
            shutil.copy2(src, PA / "tables" / name); copied.append(name)
        else:
            missing.append(str(src.relative_to(ROOT)))
    for src_dir, name in FIGS:
        for ext in ("png", "pdf"):
            p = src_dir / f"{name}.{ext}"
            if p.exists():
                shutil.copy2(p, PA / "figures" / p.name); copied.append(p.name)
            else:
                missing.append(f"{name}.{ext}")

    tab = {r["method"]: r for r in csv.DictReader(open(E16 / "analysis" / "final_50unit_table.csv"))}
    cost = {r["method"]: r for r in csv.DictReader(open(E16 / "analysis" / "robustness_cost.csv"))}
    stat = json.loads((E16 / "analysis" / "statistics.json").read_text())
    blk = list(csv.DictReader(open(E16 / "analysis" / "seed_block_stability.csv")))

    # ---- O3 main table draft ----
    L = ["# Main Table Draft（论文主表草稿）", "",
         "**来源**：Exp16 final 50-unit table（5 MVTec 类 × seeds 0–9）。判据与 5A-H/Exp10–16 完全一致：",
         "`PASS = ΔR ≤ −0.02 且 Δd′ ≥ −0.10`（相对同 (cat,seed) Original）；`catastrophic = Δd′ ≤ −0.25`。", "",
         "| Method | PASS/50 | PASS rate | catastrophic | worst Δd′ | mean Δd′ | median Δd′ | mean ΔR | Complexity |",
         "|---|---:|---:|---:|---:|---:|---:|---:|---|"]
    for m in ("Original", "Adaptive B2", "C2", "C6", "X6c"):
        if m not in tab:
            continue
        r = tab[m]
        L.append("| %s | %s/%s | %.1f%% | %s (%.1f%%) | %.4f | %.4f | %.4f | %s | %s |" % (
            m, r["PASS"], r["n_units"], 100 * float(r["pass_rate"]), r["catastrophic"],
            100 * float(r["cat_rate"]), float(r["worst_ddp"]), float(r["mean_ddp"]),
            float(r["median_ddp"]), r["mean_dR"], r["complexity"]))
    L += ["", "## Historical baselines（方法筛选阶段的冻结结果，供对照）", "",
          "| Method | PASS | Evidence | Source |", "|---|---:|---|---|",
          "| Fixed C2 | 9/15 | E2 | Exp10（`results/experiment_10_preservation_recovery/README.md`） |",
          "| Adaptive B2 | 12/15 | E2 | Exp11（本表已含 50 单元版本） |",
          "| C1（Exp14 规则 winner） | 25/35 | E3+ | Exp15（**在 unseen seeds 上退化，已归档**） |",
          "| Category × Channel (M1/M2/M3) | STOP | E1 | Exp12 |",
          "| Dual-Path（Family A） | 结构性否定 | E1 | Exp14 |",
          "| Tiny INSS（Family B） | STOP | E1 | Exp14 |",
          "", "**TODO — requires experiment**：E4 held-out dataset（MPDD）行，见 `held_out_validation_plan.md`。", ""]
    (PA / "notes" / "main_table_draft.md").write_text("\n".join(L))

    # ---- O4 ablation inventory ----
    A = ["# Ablation Inventory（只盘点已有结果，不运行新实验）", "",
         "| Method | Ablation | Status | Evidence / Source |", "|---|---|---|---|",
         "| Original | α=0 baseline | **already available** | Exp10 `P10_ORIG_a000` + Exp14/15/16 A0；50 单元 |",
         "| Adaptive B2 | category-adaptive β_c（去除 = Fixed C2） | **already available** | Exp11（Fixed C2 9/15 vs B2 12/15）；β_c 表在 `experiment11_candidates.beta_table()` |",
         "| Adaptive B2 | L2-only vs Full | **already available** | Exp11 `abl/`（`E11_B2_L2only`，已核验与 Exp14 A6 bit-exact） |",
         "| C2 | w=0.35 固定权重 | **already available** | Exp15/16（完整 w 曲线：w∈{0.20,0.35,0.50,0.65,0.80}） |",
         "| C6 | max 规则 | **already available** | Exp15/16 |",
         "| X6c | 组成分解：X6c = mean(C2, C6) | **already available** | Exp15/16 三者的独立单元结果（可做组成分解） |",
         "| X6c | 端点 ablation：w=0（=B2）与 w=1（=Original） | **already available** | Exp15 §O3（免费端点） |",
         "| 全部 | 层特异性 ablation（L2/L3 强度） | **partially available** | Exp14 A3/A5/A6 + Exp15 X4a/X4b（均被淘汰，但构成 negative ablation 证据） |",
         "| 全部 | held-out 数据集 ablation | **missing** | 需 E4（`held_out_validation_plan.md`） |",
         "| 全部 | 更多 seeds（>10）/ 多数据集统计功效 | **missing** | 若审稿人要求更强统计，扩到 seeds 10–19 或第二数据集 |",
         "", "**结论**：主表所需 ablation **已基本齐备（0 GPU 可得）**；缺失项仅为 E4 外部验证与更多 seeds 的统计功效，",
         "均属 deferred（未来验证），不影响当前冻结决策。", ""]
    (PA / "notes" / "ablation_inventory.md").write_text("\n".join(A))

    # ---- seed-block summary note ----
    S = ["# Seed-block stability（Exp16）", "",
         "| Block | " + " | ".join(("Original", "Adaptive B2", "C2", "C6", "X6c")) + " |",
         "|---|" + "---|" * 5]
    for b in blk:
        S.append("| %s | " % b["block"] + " | ".join(
            "%s (cat %s, worst %.3f)" % (b[f"{m}_PASS"], b[f"{m}_cat"], float(b[f"{m}_worst_ddp"]))
            for m in ("Original", "Adaptive B2", "C2", "C6", "X6c")) + " |")
    S += ["", "**X6c vs B2（paired bootstrap）**：ΔPASS %+.3f %s | Δmean Δd′ %+.4f %s | Δmean ΔR %+.4f %s" % (
        stat["bootstrap"]["X6c_vs_Adaptive B2"]["dPASS_rate"],
        stat["bootstrap"]["X6c_vs_Adaptive B2"]["dPASS_rate_CI"],
        stat["bootstrap"]["X6c_vs_Adaptive B2"]["dmean_ddp"],
        stat["bootstrap"]["X6c_vs_Adaptive B2"]["dmean_ddp_CI"],
        stat["bootstrap"]["X6c_vs_Adaptive B2"]["dmean_dR"],
        stat["bootstrap"]["X6c_vs_Adaptive B2"]["dmean_dR_CI"]), ""]
    (PA / "notes" / "seed_block_stability.md").write_text("\n".join(S))

    idx = ["# paper_assets — 论文资产索引（轻量：CSV/JSON/PNG/PDF/MD）", "",
           "由 `scripts/experiment16_paper_assets.py` 生成；**不含模型/checkpoint/特征 bank**。", "",
           "## tables/", ""] + [f"- `{n}`" for n in sorted(copied) if n.endswith((".csv", ".json"))] + [
           "", "## figures/", ""] + [f"- `{n}`" for n in sorted(copied) if n.endswith((".png", ".pdf"))] + [
           "", "## notes/", "", "- `main_table_draft.md`（O3）", "- `ablation_inventory.md`（O4）",
           "- `seed_block_stability.md`", "",
           "## 未复制的源（缺失）", ""] + ([f"- `{m}`" for m in missing] if missing else ["- 无"])
    (PA / "README.md").write_text("\n".join(idx) + "\n")
    print("paper_assets ->", PA)
    print("copied %d files | missing %d" % (len(copied), len(missing)))
    if missing:
        print("missing:", missing[:5])


if __name__ == "__main__":
    main()
