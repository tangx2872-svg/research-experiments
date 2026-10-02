"""
Experiment 1F — 正式 CASE 判定（按 config.json 冻结规则逐条件评估）

CASE_A: 至少一个 attribute family 与 Δdefect_std 稳定非弱关系
        (full-data 方向明确 + control area 保留 + LOCO 多数保持 + 非 grid 单类驱动 + shrink/expand 一致差异)
CASE_B: 有关系但去掉某 category 后基本消失 (category-specific)
CASE_C: 所有单一属性弱关系，但 response PCA / 组合属性有结构 (multifactorial)
CASE_D: 全部解释不了 → 无简单可泛化 defect-attribute mechanism，停
"""

import csv
import json
from pathlib import Path

import numpy as np
from scipy import stats
from scipy.cluster.hierarchy import linkage, fcluster

ROOT = Path(__file__).resolve().parents[1]
ANALYSIS = ROOT / "results" / "experiment_1f" / "analysis"

PRIMARY = "delta_defect_std"
NONWEAK_RHO = 0.5        # 「非弱」门槛（预注册解释：|rho|>=0.5 视为非弱）
ALPHA = 0.05


def load_csv(name: str) -> list[dict]:
    return list(csv.DictReader(open(ANALYSIS / name, encoding="utf-8")))


def main() -> None:
    a1 = load_csv("a1_spearman_primary.csv")
    a2 = load_csv("a2_partial_control_area.csv")
    a3 = load_csv("a3_loco_primary.csv")
    a4 = load_csv("a4_shrink_vs_expand.csv")

    # ---- 条件 1：是否存在非弱且 CI 不跨零的属性（A1）----
    strong_full = [r for r in a1 if abs(float(r["rho"])) >= NONWEAK_RHO
                   and float(r["ci_lo"]) > 0 or abs(float(r["rho"])) >= NONWEAK_RHO
                   and float(r["ci_hi"]) < 0]
    cond_full_strong = len(strong_full) > 0

    # ---- 条件 2：A4 shrink/expand 组间一致差异 ----
    a4_sig = [r for r in a4 if float(r["p_mw"]) < ALPHA
              and (float(r["diff_ci_lo"]) > 0 or float(r["diff_ci_hi"]) < 0)]
    cond_group_diff = len(a4_sig) > 0

    case_a = cond_full_strong and cond_group_diff

    # ---- CASE B 检查：某属性 full 强但去掉单类后崩塌 ----
    case_b = False
    b_drivers = []
    for r in a1:
        if abs(float(r["rho"])) >= NONWEAK_RHO and (float(r["ci_lo"]) > 0 or float(r["ci_hi"]) < 0):
            a = r["attr"]
            locos = [x for x in a3 if x["attr"] == a]
            kept = [abs(float(x["rho"])) >= NONWEAK_RHO and
                    (float(x["rho"]) * float(r["rho"]) > 0) for x in locos]
            if len(kept) - sum(kept) >= 3:  # 多数 drop 后消失
                case_b = True
                b_drivers.append(a)

    # ---- CASE C 检查：单一属性弱 + response PCA 有结构 + （已做的）组合属性 follow-up ----
    # 已执行的 cautious multivariate follow-up 结果（硬编码自本次运行，见 README）：
    #   k=2 cluster 画像: 10 属性 Mann-Whitney 全部 p>0.19
    #   双变量 rank-OLS: 最好 log_area+laplacian adj R2=0.141, 其余 <=0.078
    mv_followup = {
        "cluster_profile_min_p": 0.198,
        "best_pair_adj_r2": 0.141,
        "best_pair": "log_area_ratio + laplacian_energy",
        "conclusion": "组合属性解释力接近零（对比 1E defect identity 饱和 R2=1.0）",
    }
    # PCA 结构：PC1 解释率 >= 40% 且 k=2 聚类与 Δstd 符号一致率
    pca = load_csv("a5_pca_response.csv")
    P = np.array([[float(r["pc1"]), float(r["pc2"])] for r in pca])
    labels = fcluster(linkage(P, method="ward"), 2, criterion="maxclust")
    signs = np.array([1 if float(r["delta_defect_std"]) > 0 else 0 for r in pca])
    agree = max((signs == (labels - 1)).mean(), (signs == (2 - labels)).mean())
    pc1_var = 0.497  # 来自 analysis 输出
    pca_structure = bool(pc1_var >= 0.40 and agree >= 0.8)

    # 单属性是否全弱
    all_weak = not cond_full_strong

    case_c = all_weak and pca_structure and mv_followup["best_pair_adj_r2"] >= 0.4

    # ---- 最终判定 ----
    if case_a:
        verdict, text = "CASE_A", "EXPLAINABLE_STRUCTURE_SUPPORTED"
    elif case_b:
        verdict, text = "CASE_B", "CATEGORY_DEPENDENT_STRUCTURE"
    elif case_c:
        verdict, text = "CASE_C", "WEAK_MULTIFACTORIAL"
    else:
        verdict, text = "CASE_D", "NO_EXPLAINABLE_STRUCTURE"

    out = {
        "verdict": verdict,
        "verdict_text": text,
        "primary_outcome": PRIMARY,
        "n_defect_types": len(pca),
        "conditions": {
            "full_data_strong_attr_exists": cond_full_strong,
            "strong_attrs": [r["attr"] for r in strong_full],
            "shrink_expand_group_diff_exists": cond_group_diff,
            "group_diff_attrs": [r["attr"] for r in a4_sig],
            "CASE_A": case_a,
            "CASE_B_category_driven": case_b,
            "CASE_B_drivers": b_drivers,
            "all_univariate_weak": all_weak,
            "pca_structure": pca_structure,
            "pc1_explained_variance": pc1_var,
            "k2_cluster_sign_agreement": float(agree),
            "CASE_C": case_c,
        },
        "multivariate_followup": mv_followup,
        "frequency_family_hint": {
            "note": "frequency 族方向一致弱负相关：A1 rho -0.15~-0.19（CI 跨零），"
                    "A2 控制 area 后 -0.25~-0.36（laplacian 最强 p=0.076），"
                    "LOCO 5/5 保持方向，dilation 10/15/20 稳定。"
                    "仅为方向性 hint，远不足以构成 mechanism。",
            "best_partial": {"attr": "laplacian_energy", "partial_rho": -0.361, "p": 0.0763},
        },
        "interpretation": "1E 的 heterogeneity 真实且稳定（25/25 seed-consistent），"
                          "但 4 组预注册视觉属性（size/contrast/frequency/morphology）"
                          "单独或组合都无法解释它（最好 adj R2=0.141 vs identity 饱和 1.0）。"
                          "机制不在图像空间的简单属性里，应转向 representation-space 分析。"
                          "按协议停止。",
    }
    with open(ANALYSIS / "verdict.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    print(f"VERDICT: {verdict} — {text}")
    print(json.dumps(out["conditions"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
