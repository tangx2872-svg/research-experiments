"""Experiment 1E Phase 10-13：统一分析（metrics / signatures / stability / trajectory）。

读取全部 5 categories 的 sample_level / group_level 数据，生成：
  - group_level_metrics.csv（合并全部 category）
  - defect_response_signatures.csv（每 category x defect_type x seed）
  - defect_response_signatures_seed_summary.csv（跨 3 seeds）
  - seed_stability.csv（sign_consistency）
  - alpha_trajectory.csv + trajectory_slopes.csv

只读 results/experiment_1e/<category>/seed_*/，不训练、不修改底层结果。
"""
from __future__ import annotations

import csv
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT_ROOT = PROJECT_ROOT / "results" / "experiment_1e"
ANALYSIS_DIR = OUT_ROOT / "analysis"

CATEGORIES = ["bottle", "grid", "cable", "screw", "hazelnut"]
SEEDS = [0, 1, 2]
ALPHAS = [0.0, 0.25, 0.5, 0.75, 1.0]


def load_group(category: str, seed: int) -> list[dict]:
    """统一读取 group_level.csv（bottle 重建后也输出 seed_{seed}/group_level.csv）。"""
    d = OUT_ROOT / category / f"seed_{seed}"
    p = d / "group_level.csv"
    if not p.exists():
        return []
    rows = list(csv.DictReader(open(p, encoding="utf-8")))
    for r in rows:
        r["seed"] = int(r["seed"])
        r["alpha"] = float(r["alpha"])
        for k in ["n_good", "n_defect"]:
            r[k] = int(r[k])
        for k in ["mean_good", "std_good", "mean_defect", "std_defect", "mean_gap", "mean_z", "d_prime"]:
            r[k] = float(r[k])
    return rows


def load_area_map(category: str) -> dict[str, float]:
    """从 sample_level 提取每 defect 样本的 area_ratio（用 seed 0 即可，area 与 seed 无关）。"""
    d = OUT_ROOT / category / "seed_0"
    p = d / "sample_level.csv"
    if not p.exists():
        return {}
    area = {}
    for r in csv.DictReader(open(p, encoding="utf-8")):
        if r["is_good"] == "0" and r["area_ratio"] not in ("", None):
            area.setdefault(r["image_path"], float(r["area_ratio"]))
    return area


def sign_label(vals: list[float], tol: float = 1e-6) -> str:
    """根据 3 seeds 的符号一致性打标签。"""
    pos = sum(1 for v in vals if v > tol)
    neg = sum(1 for v in vals if v < -tol)
    n = len(vals)
    if pos == n:
        return "POSITIVE_3_OF_3"
    if neg == n:
        return "NEGATIVE_3_OF_3"
    if pos == 0 and neg == 0:
        return "NEAR_ZERO"
    return "MIXED"


def main() -> None:
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

    # ---- 收集全部 group 数据 ----
    all_group: list[dict] = []
    for cat in CATEGORIES:
        for seed in SEEDS:
            all_group.extend(load_group(cat, seed))

    # group 索引：(category, defect_type, seed, alpha) -> dict
    gidx: dict[tuple, dict] = {}
    for g in all_group:
        gidx[(g["category"], g["defect_type"], g["seed"], g["alpha"])] = g

    defect_types_by_cat: dict[str, list[str]] = defaultdict(list)
    for g in all_group:
        if g["defect_type"] not in defect_types_by_cat[g["category"]]:
            defect_types_by_cat[g["category"]].append(g["defect_type"])
    for cat in defect_types_by_cat:
        defect_types_by_cat[cat].sort()

    # ---- Phase 10: group_level_metrics.csv（合并）----
    group_fields = ["category", "defect_type", "seed", "alpha", "n_good", "n_defect",
                    "mean_good", "std_good", "mean_defect", "std_defect",
                    "mean_gap", "mean_z", "d_prime"]
    all_group_sorted = sorted(all_group, key=lambda g: (g["category"], g["defect_type"], g["seed"], g["alpha"]))
    with open(ANALYSIS_DIR / "group_level_metrics.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=group_fields)
        w.writeheader()
        w.writerows(all_group_sorted)

    # ---- Phase 11: response signatures（每 seed）----
    sig_rows: list[dict] = []
    for cat in CATEGORIES:
        for dt in defect_types_by_cat[cat]:
            for seed in SEEDS:
                if (cat, dt, seed, 0.0) not in gidx or (cat, dt, seed, 1.0) not in gidx:
                    continue
                a0 = gidx[(cat, dt, seed, 0.0)]
                a1 = gidx[(cat, dt, seed, 1.0)]
                sig_rows.append({
                    "category": cat, "defect_type": dt, "seed": seed,
                    "delta_mean_gap": a1["mean_gap"] - a0["mean_gap"],
                    "delta_defect_std": a1["std_defect"] - a0["std_defect"],
                    "delta_z": a1["mean_z"] - a0["mean_z"],
                    "delta_dprime": a1["d_prime"] - a0["d_prime"],
                })
    sig_fields = ["category", "defect_type", "seed",
                  "delta_mean_gap", "delta_defect_std", "delta_z", "delta_dprime"]
    with open(ANALYSIS_DIR / "defect_response_signatures.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=sig_fields)
        w.writeheader()
        w.writerows(sig_rows)

    # ---- Phase 11: seed summary ----
    summary_rows: list[dict] = []
    for cat in CATEGORIES:
        area_map = load_area_map(cat)
        for dt in defect_types_by_cat[cat]:
            sigs = [s for s in sig_rows if s["category"] == cat and s["defect_type"] == dt]
            if not sigs:
                continue
            n = len(sigs)
            d_mg = np.array([s["delta_mean_gap"] for s in sigs])
            d_std = np.array([s["delta_defect_std"] for s in sigs])
            d_z = np.array([s["delta_z"] for s in sigs])
            d_dp = np.array([s["delta_dprime"] for s in sigs])
            # area：该 defect type 的所有样本（静态属性，只用 seed_0 去重，避免跨 seed/alpha 重复计数）
            areas = []
            d = OUT_ROOT / cat / "seed_0"
            p = d / "sample_level.csv"
            if p.exists():
                seen = set()
                for r in csv.DictReader(open(p, encoding="utf-8")):
                    if r["defect_type"] == dt and r["area_ratio"] not in ("", None):
                        if r["image_path"] not in seen:
                            seen.add(r["image_path"])
                            areas.append(float(r["area_ratio"]))
            summary_rows.append({
                "category": cat, "defect_type": dt,
                "delta_mean_gap_mean": float(d_mg.mean()),
                "delta_mean_gap_std": float(d_mg.std(ddof=1)) if n > 1 else 0.0,
                "delta_defect_std_mean": float(d_std.mean()),
                "delta_defect_std_std": float(d_std.std(ddof=1)) if n > 1 else 0.0,
                "delta_z_mean": float(d_z.mean()),
                "delta_z_std": float(d_z.std(ddof=1)) if n > 1 else 0.0,
                "delta_dprime_mean": float(d_dp.mean()),
                "delta_dprime_std": float(d_dp.std(ddof=1)) if n > 1 else 0.0,
                "median_area_ratio": float(np.median(areas)) if areas else float("nan"),
                "mean_area_ratio": float(np.mean(areas)) if areas else float("nan"),
                "n_defect": int(len(areas)),
            })
    summary_fields = ["category", "defect_type",
                      "delta_mean_gap_mean", "delta_mean_gap_std",
                      "delta_defect_std_mean", "delta_defect_std_std",
                      "delta_z_mean", "delta_z_std",
                      "delta_dprime_mean", "delta_dprime_std",
                      "median_area_ratio", "mean_area_ratio", "n_defect"]
    with open(ANALYSIS_DIR / "defect_response_signatures_seed_summary.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=summary_fields)
        w.writeheader()
        w.writerows(summary_rows)

    # ---- Phase 12: seed stability（sign_consistency）----
    stab_rows: list[dict] = []
    dims = ["delta_mean_gap", "delta_defect_std", "delta_z", "delta_dprime"]
    for cat in CATEGORIES:
        for dt in defect_types_by_cat[cat]:
            sigs = [s for s in sig_rows if s["category"] == cat and s["defect_type"] == dt]
            if not sigs:
                continue
            row = {"category": cat, "defect_type": dt}
            for dim in dims:
                vals = [s[dim] for s in sigs]
                row[f"{dim}_sign"] = sign_label(vals)
                row[f"{dim}_mean"] = float(np.mean(vals))
            stab_rows.append(row)
    stab_fields = ["category", "defect_type"]
    for dim in dims:
        stab_fields += [f"{dim}_sign", f"{dim}_mean"]
    with open(ANALYSIS_DIR / "seed_stability.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=stab_fields)
        w.writeheader()
        w.writerows(stab_rows)

    # ---- Phase 13: full α trajectory + descriptive slope ----
    traj_rows: list[dict] = []
    slope_rows: list[dict] = []
    for cat in CATEGORIES:
        for dt in defect_types_by_cat[cat]:
            for seed in SEEDS:
                for a in ALPHAS:
                    if (cat, dt, seed, a) not in gidx:
                        continue
                    g = gidx[(cat, dt, seed, a)]
                    traj_rows.append({
                        "category": cat, "defect_type": dt, "seed": seed, "alpha": a,
                        "mean_gap": g["mean_gap"], "defect_std": g["std_defect"],
                        "mean_z": g["mean_z"], "d_prime": g["d_prime"],
                    })
            # slope（跨 3 seeds 的均值轨迹做线性拟合）
            for dim_src, dim_out in [("mean_gap", "slope_mean_gap"),
                                     ("std_defect", "slope_defect_std"),
                                     ("mean_z", "slope_z"),
                                     ("d_prime", "slope_dprime")]:
                ys = []
                for a in ALPHAS:
                    vals = [gidx[(cat, dt, s, a)][dim_src] for s in SEEDS if (cat, dt, s, a) in gidx]
                    if vals:
                        ys.append(np.mean(vals))
                if len(ys) == len(ALPHAS):
                    x = np.array(ALPHAS)
                    y = np.array(ys)
                    slope = np.polyfit(x, y, 1)[0]
                else:
                    slope = float("nan")
                slope_rows.append({"category": cat, "defect_type": dt, dim_out: float(slope)})

    traj_fields = ["category", "defect_type", "seed", "alpha",
                   "mean_gap", "defect_std", "mean_z", "d_prime"]
    with open(ANALYSIS_DIR / "alpha_trajectory.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=traj_fields)
        w.writeheader()
        w.writerows(traj_rows)

    # slope 按 defect 合并
    slope_by_key: dict[tuple, dict] = defaultdict(dict)
    for s in slope_rows:
        for k, v in s.items():
            if k not in ("category", "defect_type"):
                slope_by_key[(s["category"], s["defect_type"])][k] = v
    slope_fields = ["category", "defect_type", "slope_mean_gap", "slope_defect_std", "slope_z", "slope_dprime"]
    with open(ANALYSIS_DIR / "trajectory_slopes.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=slope_fields)
        w.writeheader()
        for (cat, dt), d in sorted(slope_by_key.items()):
            w.writerow({"category": cat, "defect_type": dt, **d})

    print(f"[done] group={len(all_group)} sig={len(sig_rows)} summary={len(summary_rows)} "
          f"stability={len(stab_rows)} trajectory={len(traj_rows)} slopes={len(slope_by_key)}")
    print(f"[done] 输出目录 = {ANALYSIS_DIR}")


if __name__ == "__main__":
    main()
