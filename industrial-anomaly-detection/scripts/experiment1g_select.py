"""
Experiment 1G — Step 0: 按 1E Δdefect_std 自动选 9 个代表 defect type

严格按 1E seed_summary.csv 的三 seed mean Δdefect_std 排序：
  Bottom 3 = strongest shrink
  Top 3    = strongest expansion
  |Δstd| 最小 3 = near-neutral

两套 selection：
  primary  : 严格 Top/Bottom/Neutral
  balanced : 每 category 最多 2 个（sensitivity，处理 grid 垄断 Top 的问题）

selection 预注册后不可因结果更换。
"""

import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "experiment_1g"


def load_sorted() -> list[dict]:
    rows = list(csv.DictReader(open(
        ROOT / "results/experiment_1e/analysis/defect_response_signatures_seed_summary.csv",
        encoding="utf-8")))
    rows.sort(key=lambda r: float(r["delta_defect_std_mean"]))
    return rows


def main() -> None:
    rows = load_sorted()
    n = len(rows)

    # ---- primary: 严格 top/bottom/neutral ----
    bottom = rows[:3]           # 最强 shrink
    top = rows[-3:]             # 最强 expansion
    # near-neutral: |Δstd| 最小（在去掉 bottom/top 后选）
    middle = sorted(rows[3:-3], key=lambda r: abs(float(r["delta_defect_std_mean"])))
    neutral = middle[:3]

    primary = []
    for r in bottom:
        primary.append({"selection": "primary", "family": "shrink",
                        "category": r["category"], "defect_type": r["defect_type"],
                        "delta_defect_std_mean": float(r["delta_defect_std_mean"])})
    for r in neutral:
        primary.append({"selection": "primary", "family": "neutral",
                        "category": r["category"], "defect_type": r["defect_type"],
                        "delta_defect_std_mean": float(r["delta_defect_std_mean"])})
    for r in top:
        primary.append({"selection": "primary", "family": "expand",
                        "category": r["category"], "defect_type": r["defect_type"],
                        "delta_defect_std_mean": float(r["delta_defect_std_mean"])})

    # ---- balanced: 每 category 最多 2 个 ----
    def pick_balanced(candidates, family, used):
        picked = []
        for r in candidates:
            cat = r["category"]
            if used.get(cat, 0) >= 2:
                continue
            picked.append({"family": family, "category": cat,
                           "defect_type": r["defect_type"],
                           "delta_defect_std_mean": float(r["delta_defect_std_mean"])})
            used[cat] = used.get(cat, 0) + 1
            if len(picked) >= 3:
                break
        return picked

    used = {}
    balanced = []
    b_shrink = pick_balanced(rows[:6], "shrink", used)      # 从 top6 shrink 里挑，避免单 category
    b_neutral = pick_balanced(sorted(rows[3:-3], key=lambda r: abs(float(r["delta_defect_std_mean"]))),
                              "neutral", used)
    # expand: 从最强 expansion 端往内扫，但每 category ≤2，且补齐到 3 个
    b_expand = pick_balanced(rows[-8:], "expand", used)     # top8 expand 候选，含 broken_large/missing_cable
    # 若仍未满 3 个（quota 限制），放宽为允许继续从剩余 expand 候选补
    if len(b_expand) < 3:
        used_relaxed = dict(used)
        extra = pick_balanced(rows[-8:], "expand", used_relaxed)
        # 简单兜底：直接取剩余最强且未选的
        seen = {x["defect_type"] for x in b_expand}
        for r in rows[-8:]:
            if len(b_expand) >= 3:
                break
            if r["defect_type"] in seen:
                continue
            b_expand.append({"family": "expand", "category": r["category"],
                             "defect_type": r["defect_type"],
                             "delta_defect_std_mean": float(r["delta_defect_std_mean"])})
            seen.add(r["defect_type"])
    for r in b_shrink + b_neutral + b_expand:
        balanced.append({"selection": "balanced", "family": r["family"],
                         "category": r["category"], "defect_type": r["defect_type"],
                         "delta_defect_std_mean": r["delta_defect_std_mean"]})

    # 写 selection.csv（含 family 标注）
    all_rows = primary + balanced
    fields = ["selection", "family", "category", "defect_type", "delta_defect_std_mean"]
    with open(OUT / "selection.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(all_rows)

    print("=== PRIMARY selection (严格 top/bottom/neutral) ===")
    for r in primary:
        print(f"  [{r['family']:7s}] {r['category']:9s} {r['defect_type']:22s} Δstd={r['delta_defect_std_mean']:+7.2f}")

    print("\n=== BALANCED selection (每 category ≤2) ===")
    for r in balanced:
        print(f"  [{r['family']:7s}] {r['category']:9s} {r['defect_type']:22s} Δstd={r['delta_defect_std_mean']:+7.2f}")

    print(f"\nselection.csv 已写入（{len(all_rows)} 行）")


if __name__ == "__main__":
    main()
