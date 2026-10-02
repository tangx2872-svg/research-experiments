"""Experiment 1E Phase 2：Bottle 重建 1E 统一 schema（不重训）。

数据源：
  - 1C raw all_sample_scores.csv（3 seeds x 5 alphas x validation+test anomaly_score）
  - 1B area_analysis.csv（defect area_ratio）
  - 1D metric_decomposition.csv / verdict.json（equivalence check 基准）

重建内容（与 experiment1e_runner.py 完全相同的字段与定义）：
  - sample_level_results.csv  每图 x 每 alpha 一行，含 good_mean/good_std/z_score/area 字段
  - group_level_metrics.csv   按 category x defect_type x seed x alpha
  - defect_response_signatures.csv      按 category x defect_type x seed（Δ = X_{α=1} - X_{α=0}）
  - defect_response_signatures_seed_summary.csv  跨 3 seeds 汇总

z 标准化基准：contemporaneous category x seed x alpha 的 test/good 分布（ddof=1）。
d' = (mu_D - mu_G) / sqrt((sigma_D^2 + sigma_G^2) / 2)。

Equivalence check（Phase 2 硬性要求）：核对 broken_large / broken_small / contamination
的 Δz / Δdefect_std / Δd' 与 1D 已确认结果一致；不一致则停止正式 Pilot。
"""
from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_1C = PROJECT_ROOT / "results" / "experiment_1c"
RESULTS_1B_AREA = PROJECT_ROOT / "results" / "experiment_1b" / "summary" / "area_analysis.csv"
RESULTS_1D_DECOMP = PROJECT_ROOT / "results" / "experiment_1d" / "summary" / "metric_decomposition.csv"
OUT_ROOT = PROJECT_ROOT / "results" / "experiment_1e" / "bottle"

ALPHAS = [0.0, 0.25, 0.5, 0.75, 1.0]
SEEDS = [0, 1, 2]
EPS = 1e-9
CATEGORY = "bottle"


def load_1c_raw(seed: int) -> list[dict]:
    p = RESULTS_1C / f"seed_{seed}" / "raw" / "all_sample_scores.csv"
    rows = list(csv.DictReader(open(p, encoding="utf-8")))
    for r in rows:
        r["anomaly_score"] = float(r["anomaly_score"])
        r["alpha"] = float(r["alpha"])
    return rows


def load_area() -> dict[str, float]:
    d: dict[str, float] = {}
    with open(RESULTS_1B_AREA, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            d[r["image_path"]] = float(r["defect_area_ratio"])
    return d


def load_1d_decomp() -> dict[tuple[int, str, float], dict]:
    """返回 {(seed, defect_type, alpha): {mu_good, std_good, mu_defect, std_defect, mean_gap, mean_sample_z, pooled_std, dprime}}"""
    d: dict[tuple[int, str, float], dict] = {}
    with open(RESULTS_1D_DECOMP, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            key = (int(r["seed"]), r["defect_type"], float(r["alpha"]))
            d[key] = {
                "mu_good": float(r["mu_good"]),
                "std_good": float(r["std_good"]),
                "mu_defect": float(r["mu_defect"]),
                "std_defect": float(r["std_defect"]),
                "mean_gap": float(r["mean_gap"]),
                "mean_sample_z": float(r["mean_sample_z"]),
                "pooled_std": float(r["pooled_std"]),
                "dprime": float(r["dprime"]),
            }
    return d


def rebuild() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    area = load_area()
    decomp_1d = load_1d_decomp()

    sample_rows: list[dict] = []
    group_rows: list[dict] = []
    signature_rows: list[dict] = []  # 每 seed 每 defect_type

    # 用于 group 汇总：{seed: {alpha: {defect_type: (n, scores, ...)}}}
    # 直接边遍历边聚合
    good_by_seed_alpha: dict[tuple[int, float], dict] = {}

    for seed in SEEDS:
        rows = load_1c_raw(seed)
        # 按 alpha 分组
        by_alpha: dict[float, list[dict]] = defaultdict(list)
        for r in rows:
            by_alpha[r["alpha"]].append(r)

        for alpha in ALPHAS:
            alpha_rows = by_alpha[alpha]
            # test/good 是 z 基准
            test_good = [r for r in alpha_rows if r["split"] == "test" and r["defect_type"] == "good"]
            good_scores = [r["anomaly_score"] for r in test_good]
            mu_good = float(np.mean(good_scores))
            sigma_good = float(np.std(good_scores, ddof=1))
            good_by_seed_alpha[(seed, alpha)] = (mu_good, sigma_good, len(good_scores))

            # 写 good 样本行
            for r in test_good:
                z = (r["anomaly_score"] - mu_good) / sigma_good if sigma_good > 0 else float("nan")
                sample_rows.append({
                    "category": CATEGORY,
                    "defect_type": "good",
                    "image_path": r["image_path"],
                    "seed": seed,
                    "alpha": alpha,
                    "is_good": 1,
                    "gt_label": 0,
                    "anomaly_score": r["anomaly_score"],
                    "good_mean": mu_good,
                    "good_std": sigma_good,
                    "z_score": z,
                    "mask_path": "",
                    "area_pixels": "",
                    "area_ratio": "",
                    "log_area_ratio": "",
                })

            # defect 样本 + group 统计
            defect_rows_by_type: dict[str, list[dict]] = defaultdict(list)
            for r in alpha_rows:
                if r["split"] == "test" and r["defect_type"] != "good":
                    defect_rows_by_type[r["defect_type"]].append(r)

            for dt, d_rows in defect_rows_by_type.items():
                scores = [r["anomaly_score"] for r in d_rows]
                s_arr = np.array(scores)
                mu_d = float(s_arr.mean())
                sigma_d = float(s_arr.std(ddof=1))
                mean_gap = mu_d - mu_good
                pooled_std = math.sqrt((sigma_d ** 2 + sigma_good ** 2) / 2.0)
                d_prime = mean_gap / pooled_std if pooled_std > 0 else float("nan")
                mean_z = float(np.mean([(x - mu_good) / sigma_good for x in scores])) if sigma_good > 0 else float("nan")

                for r in d_rows:
                    ap = area.get(r["image_path"], float("nan"))
                    ar = ap
                    lar = math.log(ar + EPS) if not math.isnan(ar) else float("nan")
                    z = (r["anomaly_score"] - mu_good) / sigma_good if sigma_good > 0 else float("nan")
                    sample_rows.append({
                        "category": CATEGORY,
                        "defect_type": dt,
                        "image_path": r["image_path"],
                        "seed": seed,
                        "alpha": alpha,
                        "is_good": 0,
                        "gt_label": 1,
                        "anomaly_score": r["anomaly_score"],
                        "good_mean": mu_good,
                        "good_std": sigma_good,
                        "z_score": z,
                        "mask_path": "",
                        "area_pixels": "",
                        "area_ratio": ar,
                        "log_area_ratio": lar,
                    })

                group_rows.append({
                    "category": CATEGORY,
                    "defect_type": dt,
                    "seed": seed,
                    "alpha": alpha,
                    "n_good": len(good_scores),
                    "n_defect": len(scores),
                    "mean_good": mu_good,
                    "std_good": sigma_good,
                    "mean_defect": mu_d,
                    "std_defect": sigma_d,
                    "mean_gap": mean_gap,
                    "mean_z": mean_z,
                    "d_prime": d_prime,
                })

    # ---- 写 sample_level（根目录汇总 + 按 seed 子目录拆分，与其余 category 对齐）----
    sample_fields = ["category", "defect_type", "image_path", "seed", "alpha",
                     "is_good", "gt_label", "anomaly_score", "good_mean", "good_std",
                     "z_score", "mask_path", "area_pixels", "area_ratio", "log_area_ratio"]
    with open(OUT_ROOT / "sample_level_results.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=sample_fields)
        w.writeheader()
        w.writerows(sample_rows)
    # 按 seed 拆分（analysis 脚本用 seed_{seed}/sample_level.csv）
    for seed in SEEDS:
        sub = OUT_ROOT / f"seed_{seed}"
        sub.mkdir(parents=True, exist_ok=True)
        seed_rows = [r for r in sample_rows if r["seed"] == seed]
        with open(sub / "sample_level.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=sample_fields)
            w.writeheader()
            w.writerows(seed_rows)

    # ---- 写 group_level（根目录汇总 + 按 seed 子目录拆分）----
    group_fields = ["category", "defect_type", "seed", "alpha", "n_good", "n_defect",
                    "mean_good", "std_good", "mean_defect", "std_defect",
                    "mean_gap", "mean_z", "d_prime"]
    with open(OUT_ROOT / "group_level_metrics.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=group_fields)
        w.writeheader()
        w.writerows(group_rows)
    for seed in SEEDS:
        sub = OUT_ROOT / f"seed_{seed}"
        seed_rows = [r for r in group_rows if r["seed"] == seed]
        with open(sub / "group_level.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=group_fields)
            w.writeheader()
            w.writerows(seed_rows)

    # ---- response signature（每 seed）----
    # group_rows 索引：(seed, alpha, defect_type) -> group dict
    gidx: dict[tuple[int, float, str], dict] = {}
    for g in group_rows:
        gidx[(g["seed"], g["alpha"], g["defect_type"])] = g

    defect_types = sorted(set(g["defect_type"] for g in group_rows))
    for seed in SEEDS:
        for dt in defect_types:
            a0 = gidx[(seed, 0.0, dt)]
            a1 = gidx[(seed, 1.0, dt)]
            signature_rows.append({
                "category": CATEGORY,
                "defect_type": dt,
                "seed": seed,
                "delta_mean_gap": a1["mean_gap"] - a0["mean_gap"],
                "delta_defect_std": a1["std_defect"] - a0["std_defect"],
                "delta_z": a1["mean_z"] - a0["mean_z"],
                "delta_dprime": a1["d_prime"] - a0["d_prime"],
            })

    sig_fields = ["category", "defect_type", "seed",
                  "delta_mean_gap", "delta_defect_std", "delta_z", "delta_dprime"]
    with open(OUT_ROOT / "defect_response_signatures.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=sig_fields)
        w.writeheader()
        w.writerows(signature_rows)

    # ---- seed summary（跨 3 seeds）----
    # 同时需要 area 信息（median/mean area_ratio, n_defect）
    summary_rows: list[dict] = []
    for dt in defect_types:
        sigs = [s for s in signature_rows if s["defect_type"] == dt]
        n_seed = len(sigs)
        d_mean_gap = np.array([s["delta_mean_gap"] for s in sigs])
        d_std = np.array([s["delta_defect_std"] for s in sigs])
        d_z = np.array([s["delta_z"] for s in sigs])
        d_dp = np.array([s["delta_dprime"] for s in sigs])
        # area（用该 defect_type 所有样本的 area_ratio）
        areas = [ar for r in sample_rows if r["defect_type"] == dt and not isinstance(r["area_ratio"], str)]
        areas = [a for a in areas if not math.isnan(a)]
        n_defect = len(set(r["image_path"] for r in sample_rows if r["defect_type"] == dt))
        summary_rows.append({
            "category": CATEGORY,
            "defect_type": dt,
            "delta_mean_gap_mean": float(d_mean_gap.mean()),
            "delta_mean_gap_std": float(d_mean_gap.std(ddof=1)) if n_seed > 1 else 0.0,
            "delta_defect_std_mean": float(d_std.mean()),
            "delta_defect_std_std": float(d_std.std(ddof=1)) if n_seed > 1 else 0.0,
            "delta_z_mean": float(d_z.mean()),
            "delta_z_std": float(d_z.std(ddof=1)) if n_seed > 1 else 0.0,
            "delta_dprime_mean": float(d_dp.mean()),
            "delta_dprime_std": float(d_dp.std(ddof=1)) if n_seed > 1 else 0.0,
            "median_area_ratio": float(np.median(areas)) if areas else float("nan"),
            "mean_area_ratio": float(np.mean(areas)) if areas else float("nan"),
            "n_defect": n_defect,
        })

    summary_fields = ["category", "defect_type",
                      "delta_mean_gap_mean", "delta_mean_gap_std",
                      "delta_defect_std_mean", "delta_defect_std_std",
                      "delta_z_mean", "delta_z_std",
                      "delta_dprime_mean", "delta_dprime_std",
                      "median_area_ratio", "mean_area_ratio", "n_defect"]
    with open(OUT_ROOT / "defect_response_signatures_seed_summary.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=summary_fields)
        w.writeheader()
        w.writerows(summary_rows)

    print(f"[done] sample_rows={len(sample_rows)} group_rows={len(group_rows)} "
          f"signature_rows={len(signature_rows)} summary_rows={len(summary_rows)}")
    return gidx, summary_rows


def equivalence_check() -> bool:
    """核对重建 Bottle 的 Δz/Δdefect_std/Δd' 与 1D 确认结果一致。"""
    print("\n" + "=" * 70)
    print("Phase 2 Equivalence Check：重建 Bottle vs Experiment 1D 确认结果")
    print("=" * 70)

    decomp = load_1d_decomp()
    # 重建 group 数据
    group_rows = list(csv.DictReader(open(OUT_ROOT / "group_level_metrics.csv", encoding="utf-8")))
    gidx: dict[tuple[int, float, str], dict] = {}
    for g in group_rows:
        g["alpha"] = float(g["alpha"]); g["seed"] = int(g["seed"])
        for k in ["mean_good", "std_good", "mean_defect", "std_defect", "mean_gap", "mean_z", "d_prime"]:
            g[k] = float(g[k])
        gidx[(g["seed"], g["alpha"], g["defect_type"])] = g

    defect_types = ["broken_large", "broken_small", "contamination"]
    all_pass = True
    report: list[dict] = []

    for dt in defect_types:
        for seed in SEEDS:
            a0 = gidx[(seed, 0.0, dt)]
            a1 = gidx[(seed, 1.0, dt)]
            rebuilt_delta_z = a1["mean_z"] - a0["mean_z"]
            rebuilt_delta_std = a1["std_defect"] - a0["std_defect"]
            rebuilt_delta_dp = a1["d_prime"] - a0["d_prime"]

            # 1D 基准：从 decomp 计算 delta（注意 1D 的 mean_sample_z 即本脚本的 mean_z）
            d0 = decomp[(seed, dt, 0.0)]
            d1 = decomp[(seed, dt, 1.0)]
            ref_delta_z = d1["mean_sample_z"] - d0["mean_sample_z"]
            ref_delta_std = d1["std_defect"] - d0["std_defect"]
            ref_delta_dp = d1["dprime"] - d0["dprime"]

            report.append({
                "defect_type": dt, "seed": seed,
                "rebuilt_delta_z": rebuilt_delta_z, "ref_delta_z": ref_delta_z,
                "rebuilt_delta_std": rebuilt_delta_std, "ref_delta_std": ref_delta_std,
                "rebuilt_delta_dp": rebuilt_delta_dp, "ref_delta_dp": ref_delta_dp,
            })

            ok_z = abs(rebuilt_delta_z - ref_delta_z) < 1e-6
            ok_std = abs(rebuilt_delta_std - ref_delta_std) < 1e-6
            ok_dp = abs(rebuilt_delta_dp - ref_delta_dp) < 1e-6
            if not (ok_z and ok_std and ok_dp):
                all_pass = False
            print(f"  {dt:16s} seed={seed}  Δz={rebuilt_delta_z:+.4f}(ref {ref_delta_z:+.4f}) "
                  f"Δstd={rebuilt_delta_std:+.4f}(ref {ref_delta_std:+.4f}) "
                  f"Δd'={rebuilt_delta_dp:+.4f}(ref {ref_delta_dp:+.4f}) "
                  f"{'OK' if (ok_z and ok_std and ok_dp) else 'MISMATCH'}")

    # 写 equivalence 报告
    with open(OUT_ROOT / "equivalence_check.json", "w", encoding="utf-8") as f:
        json.dump({"all_pass": all_pass, "checks": report}, f, indent=2, ensure_ascii=False)

    print("-" * 70)
    print(f"EQUIVALENCE CHECK: {'PASS' if all_pass else 'FAIL'}")
    return all_pass


if __name__ == "__main__":
    gidx, summary = rebuild()
    ok = equivalence_check()
    print(f"\n最终判定：{'✅ 重建与 1D 一致，可继续正式 Pilot' if ok else '❌ 不一致，停止正式 Pilot，先排查'}")
    sys_exit = 0 if ok else 1
    import sys
    sys.exit(sys_exit)
