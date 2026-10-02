"""Experiment 1E Phase 17：正式 verdict 判定。

综合证据：
  1. seed stability（sign consistency across 3 seeds）
  2. cross-category reproducibility（contamination 方差收缩模式是否在其它 category 复现）
  3. area control（area-only 是否充分解释异质性）

CASE A: CROSS_CATEGORY_HETEROGENEITY_SUPPORTED
    多个新增 categories 存在稳定 defect-dependent response heterogeneity，
    且不能由 area-only explanation 充分解释。
CASE B: PARTIAL_SUPPORT（只在部分 category / 部分 dimension / seed stability 有限）
CASE C: BOTTLE_SPECIFIC_OR_NO_GENERALIZATION
"""
from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ANALYSIS_DIR = PROJECT_ROOT / "results" / "experiment_1e" / "analysis"


def main() -> None:
    stab = list(csv.DictReader(open(ANALYSIS_DIR / "seed_stability.csv", encoding="utf-8")))
    summary = list(csv.DictReader(open(ANALYSIS_DIR / "defect_response_signatures_seed_summary.csv", encoding="utf-8")))
    area_control = json.load(open(ANALYSIS_DIR / "area_control_results.json", encoding="utf-8"))

    dims = ["delta_mean_gap", "delta_defect_std", "delta_z", "delta_dprime"]

    # ---- 证据 1：seed stability ----
    n_total = len(stab)
    stable_by_dim = {}
    for d in dims:
        c = Counter(r[f"{d}_sign"] for r in stab)
        stable_by_dim[d] = c
    n_any_stable = sum(
        1 for r in stab
        if any(r[f"{d}_sign"].endswith("3_OF_3") for d in dims)
    )

    # ---- 证据 2：heterogeneity 的方向分化（同一 category 内不同 defect type 方向不同）----
    hetero_by_category: dict[str, dict[str, bool]] = {}
    for cat in sorted(set(r["category"] for r in stab)):
        rows = [r for r in stab if r["category"] == cat]
        het = {}
        for d in dims:
            signs = set()
            for r in rows:
                s = r[f"{d}_sign"]
                if s == "POSITIVE_3_OF_3":
                    signs.add("+")
                elif s == "NEGATIVE_3_OF_3":
                    signs.add("-")
            het[d] = len(signs) >= 2  # 同 category 内出现两个方向的稳定响应
        hetero_by_category[cat] = het

    # bottle 的 Δdefect_std 方向分化（1D 已知：large +1.7 / cont -2.4）
    bottle_stable_dims = sum(1 for d in dims if hetero_by_category["bottle"][d])

    # 新增 categories 中有多少个存在 Δdefect_std 或 Δd' 的方向分化
    new_cats = ["grid", "cable", "screw", "hazelnut"]
    cats_with_hetero = [
        cat for cat in new_cats
        if hetero_by_category[cat]["delta_defect_std"] or hetero_by_category[cat]["delta_dprime"]
    ]

    # ---- 证据 3：方差收缩模式的跨类别复现（Δz<0 stable + Δstd<0 stable + Δd'>0 stable）----
    # sign 信息在 seed_stability.csv，数值在 summary，按 (category, defect_type) join
    sign_lookup = {(r["category"], r["defect_type"]): r for r in stab}
    variance_shrink_pattern = []
    for r in summary:
        s = sign_lookup.get((r["category"], r["defect_type"]))
        if s is None:
            continue
        if (s["delta_z_sign"] == "NEGATIVE_3_OF_3"
                and s["delta_defect_std_sign"] == "NEGATIVE_3_OF_3"
                and s["delta_dprime_sign"] == "POSITIVE_3_OF_3"):
            variance_shrink_pattern.append({
                "category": r["category"], "defect_type": r["defect_type"],
                "delta_z": r["delta_z_mean"], "delta_std": r["delta_defect_std_mean"],
                "delta_dprime": r["delta_dprime_mean"],
            })

    # ---- 证据 4：area control ----
    area_r2 = {k: area_control[k]["Model_A_r2"] for k in area_control}
    area_sufficient = all(v > 0.5 for v in area_r2.values())

    # ---- verdict ----
    print("=" * 70)
    print("Experiment 1E Formal Verdict — 证据汇总")
    print("=" * 70)
    print(f"\n[1] Seed stability: {n_any_stable}/{n_total} defect types 有 >=1 个 3/3 稳定非零维度")
    for d in dims:
        c = stable_by_dim[d]
        print(f"    {d}: POS={c.get('POSITIVE_3_OF_3',0)} NEG={c.get('NEGATIVE_3_OF_3',0)} "
              f"MIXED={c.get('MIXED',0)} NEAR_ZERO={c.get('NEAR_ZERO',0)}")

    print(f"\n[2] Category 内方向分化（heterogeneity within category）:")
    for cat, het in hetero_by_category.items():
        het_dims = [d for d in dims if het[d]]
        print(f"    {cat:10s}: {het_dims if het_dims else '无'}")

    print(f"\n[3] 方差收缩模式复现（Δz<0 & Δstd<0 & Δd'>0, all 3/3 stable）:")
    for v in variance_shrink_pattern:
        print(f"    {v['category']}/{v['defect_type']}: Δz={float(v['delta_z']):.2f} "
              f"Δstd={float(v['delta_std']):.2f} Δd'={float(v['delta_dprime']):.2f}")

    print(f"\n[4] Area control (Model A, area-only R²):")
    for k, v in area_r2.items():
        print(f"    {k}: R²={v:.4f}")
    print(f"    area-only 充分解释? {'是' if area_sufficient else '否'}")

    # ---- 判定逻辑 ----
    cond_a1 = len(cats_with_hetero) >= 2          # 至少 2 个新增 category 有方向分化
    cond_a2 = n_any_stable >= n_total * 0.8        # 绝大多数 defect 有稳定响应
    cond_a3 = not area_sufficient                  # area 不足以解释
    cond_a4 = len(variance_shrink_pattern) >= 2    # 方差收缩模式跨类别复现

    print("\n" + "=" * 70)
    print("CASE A 条件检查:")
    print(f"  A1 至少2个新增category有方向分化: {len(cats_with_hetero)} >= 2 → {cond_a1} ({cats_with_hetero})")
    print(f"  A2 稳定响应覆盖: {n_any_stable}/{n_total} >= 80% → {cond_a2}")
    print(f"  A3 area 不充分: → {cond_a3}")
    print(f"  A4 方差收缩模式跨类别复现: {len(variance_shrink_pattern)} >= 2 → {cond_a4} "
          f"({[v['category']+'/'+v['defect_type'] for v in variance_shrink_pattern]})")

    if cond_a1 and cond_a2 and cond_a3:
        verdict = "CASE_A"
        verdict_text = "CROSS_CATEGORY_HETEROGENEITY_SUPPORTED"
    elif n_any_stable > 0 and (len(cats_with_hetero) >= 1 or not area_sufficient):
        verdict = "CASE_B"
        verdict_text = "PARTIAL_SUPPORT"
    else:
        verdict = "CASE_C"
        verdict_text = "BOTTLE_SPECIFIC_OR_NO_GENERALIZATION"

    print("\n" + "=" * 70)
    print(f"VERDICT: {verdict} — {verdict_text}")
    print("=" * 70)

    out = {
        "verdict": verdict,
        "verdict_text": verdict_text,
        "n_defect_types": n_total,
        "n_any_stable": n_any_stable,
        "stable_by_dim": {d: dict(stable_by_dim[d]) for d in dims},
        "hetero_by_category": {c: {d: bool(v) for d, v in het.items()} for c, het in hetero_by_category.items()},
        "cats_with_hetero": cats_with_hetero,
        "variance_shrink_pattern": variance_shrink_pattern,
        "area_r2_model_a": area_r2,
        "area_sufficient": area_sufficient,
        "conditions": {"A1_cats_hetero": cond_a1, "A2_stable_coverage": cond_a2,
                       "A3_area_insufficient": cond_a3, "A4_shrink_repro": cond_a4},
    }
    with open(ANALYSIS_DIR / "verdict.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"[done] verdict 写入 {ANALYSIS_DIR / 'verdict.json'}")


if __name__ == "__main__":
    main()
