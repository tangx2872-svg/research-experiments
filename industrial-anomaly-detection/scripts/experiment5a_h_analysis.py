"""Experiment 5A-H — 分析脚本（纯 CPU，读 runner 持久化的分数与 info.json）。

输出（results/experiment_5a_h/）：
  summary/per_unit.csv            15 units × 5 configs 全指标
  summary/g2_vs_c2.csv            核心 pairwise（ΔRobustness / ΔPreservation / PAIR WIN / Pareto）
  summary/per_defect.csv          defect-level Δd′ 与 improved/neutral/degraded
  summary/cross_seed_summary.csv  每 category wins x/3
  summary/cross_category_summary.csv  15 / 12(excl bottle) 总计
  summary/pareto_summary.csv      每 unit G2 的 Pareto 状态
  summary/statistical_tests.csv   Wilcoxon signed-rank（15 与 12 pairs）
  summary/sanity_checks.csv       S1–S14 汇总
  figures/g2_vs_c2_robustness.png / g2_vs_c2_preservation.png /
          category_consistency.png / seed_consistency.png /
          pareto_by_category.png / defect_level_effects.png

冻结口径（README §2/§4）：robustness = mean |ΔNormalScore_z|（低好）；
preservation = mean defect d′（高好）；PAIR WIN: ΔRob<0 AND ΔPres >= −ε, ε=0.10。
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parents[1]
OUT_ROOT = ROOT / "results" / "experiment_5a_h"
RAW_DIR = OUT_ROOT / "raw"
LOGS_DIR = OUT_ROOT / "logs"
SUMMARY_DIR = OUT_ROOT / "summary"
FIG_DIR = OUT_ROOT / "figures"

CATEGORIES = ["bottle", "cable", "grid", "hazelnut", "screw"]
SEEDS = [0, 1, 2]
CONFIGS = ["B0", "B2", "C2", "G2", "C3"]
FIXED = ["B0", "B2", "C2", "C3"]
SHIFTS = ["brightness_0.7", "brightness_1.3", "gamma_0.7", "gamma_1.3"]
EPS = 0.10

A2_G2 = 0.75 * 0.603651
A_C2 = (A2_G2 + 0.75) / 2.0
A_C3 = (0.603651 + 1.0) / 2.0
ALPHAS = {"B0": (0.0, 0.0), "B2": (0.5, 0.5), "C2": (A_C2, A_C2),
          "G2": (A2_G2, 0.75), "C3": (A_C3, A_C3)}

STAGE4_FILES = [
    ROOT / "results" / "experiment_1j" / "analysis" / "experiment1j_cross_chain_summary.csv",
    ROOT / "results" / "experiment_1j" / "analysis" / "experiment1j_family_summary.csv",
    ROOT / "results" / "experiment_1j_b" / "analysis" / "transmission_summary.csv",
    ROOT / "results" / "experiment_1j_b" / "analysis" / "per_defect.csv",
    ROOT / "results" / "experiment_5a" / "geometry_rule.json",
]

KIND_COLOR = {"B0": "#4C72B0", "B2": "#8DA0CB", "C2": "#55A868", "G2": "#C44E52", "C3": "#937860"}


def image_auroc(scores_good: np.ndarray, scores_defect: np.ndarray) -> float:
    y = np.concatenate([np.zeros(len(scores_good)), np.ones(len(scores_defect))])
    s = np.concatenate([scores_good, scores_defect])
    if len(np.unique(y)) < 2:
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    y_sorted = y[order]
    n_pos = int(y.sum()); n_neg = len(y) - n_pos
    s_sorted = s[order]
    ranks = np.arange(1, len(y) + 1, dtype=np.float64)
    i = 0
    while i < len(s_sorted):
        j = i
        while j + 1 < len(s_sorted) and s_sorted[j + 1] == s_sorted[i]:
            j += 1
        ranks[i:j + 1] = (i + 1 + j + 1) / 2.0
        i = j + 1
    rank_pos = ranks[y_sorted == 1]
    return float((rank_pos.sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def d_prime(sd_: np.ndarray, mu_g: float, sigma_g: float) -> float:
    mu_d = float(sd_.mean()); sd_d = float(sd_.std(ddof=1))
    pooled = math.sqrt((sd_d**2 + sigma_g**2) / 2.0)
    return (mu_d - mu_g) / pooled if pooled > 0 else float("nan")


def load_all() -> dict:
    """{ (cat, seed, config) -> {subset/shift 结构, info} }"""
    data = {}
    for cat in CATEGORIES:
        for seed in SEEDS:
            for cfg in CONFIGS:
                d = RAW_DIR / cat / f"seed_{seed}" / f"config_{cfg}"
                info = json.loads((d / "info.json").read_text())
                assert info["status"] == "OK", f"{d} not OK: {info.get('status')}"
                with open(d / "per_image.csv", newline="") as f:
                    rows = list(csv.DictReader(f))
                sub = defaultdict(lambda: defaultdict(list))
                for r in rows:
                    sub[r["subset"]][r["shift"]].append(
                        (r["defect_type"], float(r["score"]),
                         float(r["clipped_high_ratio"]), float(r["clipped_low_ratio"])))
                data[(cat, seed, cfg)] = {"sub": sub, "info": info, "rows": rows}
    return data


def unit_metrics(data: dict) -> dict:
    """(cat, seed, config) -> 指标 dict"""
    out = {}
    for key, dd in data.items():
        cat, seed, cfg = key
        sub, info = dd["sub"], dd["info"]
        cg = np.array([s for _, s, _, _ in sub["clean_good"]["none"]])
        mu_g, sd_g = float(cg.mean()), float(cg.std(ddof=1))
        tau = info["tau_val"]

        dz, fpr_shift, clip_h = [], [], []
        for sh in SHIFTS:
            ss = np.array([s for _, s, _, _ in sub["shift_good"][sh]])
            z = (ss - mu_g) / sd_g if sd_g > 0 else np.full_like(ss, np.nan)
            dz.append(float(z.mean()))
            fpr_shift.append(float((ss > tau).mean()))
            clip_h.append(float(np.mean([c for _, _, c, _ in sub["shift_good"][sh]])))

        defects = sorted({dt for dt, _, _, _ in sub["clean_defect"]["none"]})
        per_def = {}
        for dt in defects:
            sd_ = np.array([s for d_, s, _, _ in sub["clean_defect"]["none"] if d_ == dt])
            per_def[dt] = {
                "n": len(sd_),
                "mean_score": float(sd_.mean()),
                "d_prime": d_prime(sd_, mu_g, sd_g),
                "mean_z": float(((sd_ - mu_g) / sd_g).mean()) if sd_g > 0 else float("nan"),
            }
        all_def = np.array([s for _, s, _, _ in sub["clean_defect"]["none"]])
        out[key] = {
            "alpha_l2": info["alpha_l2"], "alpha_l3": info["alpha_l3"],
            "tau_val": tau,
            "normal_clean_mean": mu_g, "normal_clean_std": sd_g,
            "fpr_clean": float((cg > tau).mean()),
            "normal_shift_mean": float(np.mean([np.mean([s for _, s, _, _ in sub["shift_good"][sh]]) for sh in SHIFTS])),
            "mean_delta_z": float(np.mean(dz)),
            "mean_abs_delta_z": float(np.mean(np.abs(dz))),
            "per_shift_dz": dict(zip(SHIFTS, dz)),
            "fpr_shift": float(np.mean(fpr_shift)),
            "clipped_pixel_ratio": float(np.mean(clip_h)),
            "image_auroc": image_auroc(cg, all_def),
            "pixel_auroc": info["pixel_auroc"], "aupro": info["aupro"],
            "per_defect": per_def,
            "mean_dprime": float(np.mean([per_def[dt]["d_prime"] for dt in defects])),
            "defects": defects,
            "runtime": info["runtime_seconds"],
            "peak_vram": info["peak_gpu_memory_allocated_mb"],
        }
    return out


def pareto_status(m: dict, cat: str, seed: int) -> dict:
    """G2 vs fixed set {B0,B2,C2,C3}：dominated / non-dominated / beyond（插值判据）。"""
    fixed_pts = {c: (m[(cat, seed, c)]["mean_abs_delta_z"], m[(cat, seed, c)]["mean_dprime"])
                 for c in FIXED}
    gx, gy = m[(cat, seed, "G2")]["mean_abs_delta_z"], m[(cat, seed, "G2")]["mean_dprime"]
    dominators = [c for c, (x, y) in fixed_pts.items() if x <= gx and y >= gy and (x < gx or y > gy)]
    dominates_c2 = gx < fixed_pts["C2"][0] and gy > fixed_pts["C2"][1]

    # fixed frontier（x 升序扫描，y 严格递增者为 frontier）
    fr = sorted(fixed_pts.items(), key=lambda kv: kv[1][0])
    frontier = []
    for c, (x, y) in fr:
        if not frontier or y > frontier[-1][1]:
            frontier.append((x, y))
    frontier.sort()  # x 升序，y 降序
    if gx <= frontier[0][0]:
        interp_y = frontier[0][1]
    elif gx >= frontier[-1][0]:
        interp_y = frontier[-1][1]
    else:
        for i in range(len(frontier) - 1):
            x0, y0 = frontier[i]; x1, y1 = frontier[i + 1]
            if x0 <= gx <= x1:
                interp_y = y0 + (y1 - y0) * (gx - x0) / (x1 - x0 + 1e-12)
                break
    beyond = (not dominators) and gy > interp_y
    status = ("dominated" if dominators else
              ("beyond_frontier" if beyond else "non_dominated_inside"))
    return {"pareto_status": status, "dominated_by": ";".join(dominators),
            "dominates_C2": dominates_c2, "interp_dprime_at_g2_x": interp_y}


def main() -> None:
    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    data = load_all()
    m = unit_metrics(data)
    units = [(c, s) for c in CATEGORIES for s in SEEDS]
    assert len(m) == 75, f"expected 75 config results, got {len(m)}"

    # ---------- per_unit.csv ----------
    rows = []
    for cat, seed in units:
        for cfg in CONFIGS:
            r = m[(cat, seed, cfg)]
            row = {
                "category": cat, "seed": seed, "config": cfg,
                "alpha_l2": round(r["alpha_l2"], 6), "alpha_l3": round(r["alpha_l3"], 6),
                "normal_clean_mean": round(r["normal_clean_mean"], 4),
                "normal_shift_mean": round(r["normal_shift_mean"], 4),
                "mean_abs_delta_z": round(r["mean_abs_delta_z"], 4),
                "mean_delta_z": round(r["mean_delta_z"], 4),
                "image_auroc": round(r["image_auroc"], 4),
                "pixel_auroc": round(r["pixel_auroc"], 4) if np.isfinite(r["pixel_auroc"]) else "na",
                "aupro": round(r["aupro"], 4) if np.isfinite(r["aupro"]) else "na",
                "mean_dprime": round(r["mean_dprime"], 4),
                "fpr_clean": round(r["fpr_clean"], 4),
                "fpr_shift": round(r["fpr_shift"], 4),
                "clipped_pixel_ratio": round(r["clipped_pixel_ratio"], 5),
                "runtime_seconds": r["runtime"], "peak_vram_mb": r["peak_vram"],
                "tau_val": round(r["tau_val"], 4),
            }
            for dt, v in r["per_defect"].items():
                row[f"dprime_{dt}"] = round(v["d_prime"], 4)
            for sh in SHIFTS:
                row[f"dz_{sh}"] = round(r["per_shift_dz"][sh], 4)
            rows.append(row)
    # fieldnames：取全部行的键并集（不同 category defect 数不同），保持首行顺序 + 追加新增键
    base_fields = list(rows[0].keys())
    extra = sorted({k for r in rows for k in r if k not in base_fields})
    all_fields = base_fields + extra
    with open(SUMMARY_DIR / "per_unit.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=all_fields, extrasaction="ignore")
        w.writeheader(); w.writerows(rows)

    # ---------- g2_vs_c2.csv + pareto_summary ----------
    pair_rows, pareto_rows = [], []
    for cat, seed in units:
        g, c = m[(cat, seed, "G2")], m[(cat, seed, "C2")]
        d_rob = g["mean_abs_delta_z"] - c["mean_abs_delta_z"]
        d_pres = g["mean_dprime"] - c["mean_dprime"]
        ps = pareto_status(m, cat, seed)
        pair_rows.append({
            "category": cat, "seed": seed,
            "robustness_G2": round(g["mean_abs_delta_z"], 4),
            "robustness_C2": round(c["mean_abs_delta_z"], 4),
            "delta_robustness": round(d_rob, 4),
            "robustness_win": int(d_rob < 0),
            "dprime_G2": round(g["mean_dprime"], 4),
            "dprime_C2": round(c["mean_dprime"], 4),
            "delta_preservation": round(d_pres, 4),
            "preservation_noninferior": int(d_pres >= -EPS),
            "pair_win": int(d_rob < 0 and d_pres >= -EPS),
            "is_bottle": int(cat == "bottle"),
        })
        pareto_rows.append({
            "category": cat, "seed": seed,
            "g2_abs_dz": round(g["mean_abs_delta_z"], 4),
            "g2_dprime": round(g["mean_dprime"], 4),
            **{f"{k}_abs_dz": round(m[(cat, seed, k)]["mean_abs_delta_z"], 4) for k in FIXED},
            **{f"{k}_dprime": round(m[(cat, seed, k)]["mean_dprime"], 4) for k in FIXED},
            **ps,
        })
    for name, rr in [("g2_vs_c2.csv", pair_rows), ("pareto_summary.csv", pareto_rows)]:
        with open(SUMMARY_DIR / name, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rr[0].keys()))
            w.writeheader(); w.writerows(rr)

    # ---------- per_defect.csv ----------
    pd_rows = []
    for cat, seed in units:
        g, c = m[(cat, seed, "G2")], m[(cat, seed, "C2")]
        for dt in g["defects"]:
            dd = g["per_defect"][dt]["d_prime"] - c["per_defect"][dt]["d_prime"]
            pd_rows.append({
                "category": cat, "seed": seed, "defect_type": dt,
                "dprime_G2": round(g["per_defect"][dt]["d_prime"], 4),
                "dprime_C2": round(c["per_defect"][dt]["d_prime"], 4),
                "delta_dprime": round(dd, 4),
                "class": "improved" if dd > EPS else ("degraded" if dd < -EPS else "neutral"),
                "is_bottle": int(cat == "bottle"),
            })
    with open(SUMMARY_DIR / "per_defect.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(pd_rows[0].keys()))
        w.writeheader(); w.writerows(pd_rows)

    # ---------- cross_seed_summary / cross_category_summary ----------
    cs_rows = []
    for cat in CATEGORIES:
        prs = [r for r in pair_rows if r["category"] == cat]
        dr = [r["delta_robustness"] for r in prs]
        dp = [r["delta_preservation"] for r in prs]
        ps_stat = [r["pareto_status"] for r in pareto_rows if r["category"] == cat]
        cs_rows.append({
            "category": cat,
            "robustness_wins": sum(r["robustness_win"] for r in prs),
            "preservation_noninferior": sum(r["preservation_noninferior"] for r in prs),
            "pair_wins": sum(r["pair_win"] for r in prs),
            "delta_robustness_mean": round(float(np.mean(dr)), 4),
            "delta_robustness_std": round(float(np.std(dr, ddof=1)), 4),
            "delta_preservation_mean": round(float(np.mean(dp)), 4),
            "delta_preservation_std": round(float(np.std(dp, ddof=1)), 4),
            "pareto_beyond": ps_stat.count("beyond_frontier"),
            "pareto_nondominated_or_beyond":
                ps_stat.count("beyond_frontier") + ps_stat.count("non_dominated_inside"),
            "dominates_C2": sum(r["dominates_C2"] for r in pareto_rows if r["category"] == cat),
        })
    with open(SUMMARY_DIR / "cross_seed_summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(cs_rows[0].keys()))
        w.writeheader(); w.writerows(cs_rows)

    def tally(sel) -> dict:
        prs = [r for r in pair_rows if sel(r)]
        pss = [r for r in pareto_rows if sel(r)]
        return {
            "n_units": len(prs),
            "robustness_wins": sum(r["robustness_win"] for r in prs),
            "preservation_noninferior": sum(r["preservation_noninferior"] for r in prs),
            "pair_wins": sum(r["pair_win"] for r in prs),
            "pareto_beyond": sum(1 for r in pss if r["pareto_status"] == "beyond_frontier"),
            "pareto_nondominated": sum(1 for r in pss
                                       if r["pareto_status"] in ("beyond_frontier", "non_dominated_inside")),
            "dominates_C2": sum(r["dominates_C2"] for r in pss),
        }
    cc_rows = [
        {"scope": "all_15", **tally(lambda r: True)},
        {"scope": "excluding_bottle_12", **tally(lambda r: r["category"] != "bottle")},
    ]
    with open(SUMMARY_DIR / "cross_category_summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(cc_rows[0].keys()))
        w.writeheader(); w.writerows(cc_rows)

    # ---------- statistical_tests.csv ----------
    stat_rows = []
    for scope, sel in [("all_15", lambda r: True),
                       ("excluding_bottle_12", lambda r: r["category"] != "bottle")]:
        prs = [r for r in pair_rows if sel(r)]
        for metric, col in [("robustness", "delta_robustness"), ("preservation", "delta_preservation")]:
            d = np.array([r[col] for r in prs], dtype=float)
            try:
                stat, p = wilcoxon(d) if np.any(d != 0) else (float("nan"), float("nan"))
                note = ""
            except ValueError as exc:
                stat, p, note = float("nan"), float("nan"), f"wilcoxon error: {exc}"
            stat_rows.append({
                "scope": scope, "metric": metric, "n_pairs": len(d),
                "median_paired_delta": round(float(np.median(d)), 4),
                "mean_paired_delta": round(float(d.mean()), 4),
                "wilcoxon_stat": round(float(stat), 4) if np.isfinite(stat) else "na",
                "wilcoxon_p_two_sided": round(float(p), 6) if np.isfinite(p) else "na",
                "note": note or "zero-diffs dropped (scipy default)",
            })
    stat_rows.append({
        "scope": "per_category", "metric": "not_tested", "n_pairs": 3,
        "median_paired_delta": "na", "mean_paired_delta": "na",
        "wilcoxon_stat": "na", "wilcoxon_p_two_sided": "na",
        "note": "n=3 时 Wilcoxon 最小 p≈0.25 无意义，只报 mean±std 与 wins/3",
    })
    with open(SUMMARY_DIR / "statistical_tests.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(stat_rows[0].keys()))
        w.writeheader(); w.writerows(stat_rows)

    # ---------- sanity 汇总（S1–S10/S12/S13 + S11/S14） ----------
    checks = []
    # runner pre sanity（各 worker）
    for f in sorted(LOGS_DIR.glob("sanity_pre_*.csv")):
        with open(f, newline="") as fh:
            for r in csv.DictReader(fh):
                checks.append((r["check"] + f" [{f.stem.split('_')[-1]}]",
                               r["status"] == "PASS", r["detail"]))
    # S9：全部分数有限；S10：行数
    all_scores = [float(r["score"]) for dd in data.values() for r in dd["rows"]]
    checks.append(("S9_nan_inf_scan", all(np.isfinite(all_scores)), f"{len(all_scores)} scores finite"))
    row_ok = all(dd["info"]["n_rows"] == 20 + sum(1 for r in dd["rows"] if r["subset"] == "clean_good")
                 + sum(1 for r in dd["rows"] if r["subset"] == "clean_defect")
                 + 4 * sum(1 for r in dd["rows"] if r["subset"] == "clean_good")
                 for dd in data.values())
    checks.append(("S10_row_counts", row_ok, "per config: 20+n_good+n_defect+4*n_good"))
    # S11：seeds 确实不同（同 config 跨 seed 的 clean good 分数应存在差异）
    s11_ok, s11_detail = True, []
    for cat in CATEGORIES:
        for cfg in CONFIGS:
            by_seed = {seed: [float(r["score"]) for r in data[(cat, seed, cfg)]["rows"]
                              if r["subset"] == "clean_good"] for seed in SEEDS}
            diffs = [max(abs(a - b) for a, b in zip(by_seed[0], by_seed[s])) for s in (1, 2)]
            if max(diffs) <= 0:
                s11_ok = False; s11_detail.append(f"{cat}:{cfg} identical across seeds")
    checks.append(("S11_seeds_differ", s11_ok,
                   "; ".join(s11_detail) if s11_detail else "all configs differ across seeds"))
    # S13：Stage ④ 文件 hash 前后不变
    before = {}
    for f in sorted(LOGS_DIR.glob("stage4_hashes_*.json")):
        before.update(json.loads(f.read_text())["before"])
    s13_ok, s13_detail = True, []
    for p in STAGE4_FILES:
        cur = hashlib.md5(p.read_bytes()).hexdigest()
        if before.get(str(p)) not in (None, cur):
            s13_ok = False; s13_detail.append(f"{p.name} changed")
    checks.append(("S13_stage4_unchanged", s13_ok,
                   "; ".join(s13_detail) if s13_detail
                   else f"{len(STAGE4_FILES)} frozen files md5 unchanged"))
    # S14（补充）：bottle:0 与 5A raw 复现一致性（同协议确定性重跑）
    five_a = {}
    with open(ROOT / "results" / "experiment_5a" / "raw" / "per_image_scores.csv", newline="") as fh:
        for r in csv.DictReader(fh):
            five_a[(r["config"], r["subset"], r["shift"], r["defect_type"], r["image_path"])] = float(r["score"])
    maxdiff, n_cmp = 0.0, 0
    for cfg in CONFIGS:
        for r in data[("bottle", 0, cfg)]["rows"]:
            k = (cfg, r["subset"], r["shift"], r["defect_type"], r["image_path"])
            if k in five_a:
                maxdiff = max(maxdiff, abs(float(r["score"]) - five_a[k]))
                n_cmp += 1
    checks.append(("S14_5A_reproduction_bottle_seed0", maxdiff < 1.0,
                   f"n={n_cmp} rows compared, max|Δscore|={maxdiff:.6f} (threshold 1.0, supplementary)"))

    with open(SUMMARY_DIR / "sanity_checks.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["check", "status", "detail"])
        for name, ok, detail in checks:
            w.writerow([name, "PASS" if ok else "FAIL", detail])
    n_fail = sum(1 for _, ok, _ in checks if not ok)
    print(f"[sanity] {len(checks) - n_fail}/{len(checks)} PASS")

    # ---------- figures ----------
    # 1) g2 vs c2 robustness
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(units))
    for i, (cat, seed) in enumerate(units):
        g = m[(cat, seed, "G2")]["mean_abs_delta_z"]; c = m[(cat, seed, "C2")]["mean_abs_delta_z"]
        ax.plot([i - 0.18, i + 0.18], [c, g], color="#999", lw=0.8, zorder=1)
        ax.scatter(i - 0.18, c, color=KIND_COLOR["C2"], s=45, zorder=3, label="C2 (uniform)" if i == 0 else "")
        ax.scatter(i + 0.18, g, color=KIND_COLOR["G2"], marker="D", s=45, zorder=3, label="G2 (guided)" if i == 0 else "")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{c[:4]}\n{s}" for c, s in units], fontsize=7)
    ax.set_ylabel("mean |ΔNormalScore_z| (lower = more robust)")
    ax.set_title("G2 vs C2 — illumination robustness, per unit (15 = 5 categories × 3 seeds)")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(FIG_DIR / "g2_vs_c2_robustness.png", dpi=150); plt.close(fig)

    # 2) g2 vs c2 preservation
    fig, ax = plt.subplots(figsize=(10, 5))
    for i, (cat, seed) in enumerate(units):
        g = m[(cat, seed, "G2")]["mean_dprime"]; c = m[(cat, seed, "C2")]["mean_dprime"]
        ax.plot([i - 0.18, i + 0.18], [c, g], color="#999", lw=0.8, zorder=1)
        ax.scatter(i - 0.18, c, color=KIND_COLOR["C2"], s=45, zorder=3, label="C2" if i == 0 else "")
        ax.scatter(i + 0.18, g, color=KIND_COLOR["G2"], marker="D", s=45, zorder=3, label="G2" if i == 0 else "")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{c[:4]}\n{s}" for c, s in units], fontsize=7)
    ax.set_ylabel("mean defect d′ (higher = better preservation)")
    ax.set_title("G2 vs C2 — defect preservation, per unit")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(FIG_DIR / "g2_vs_c2_preservation.png", dpi=150); plt.close(fig)

    # 3) category consistency（mean±std Δrob / Δpres）
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax_, col, favorable_sign, ttl in [
            (axes[0], "delta_robustness_mean", -1, "ΔRobustness (|Δz| G2−C2, <0 = G2 wins)"),
            (axes[1], "delta_preservation_mean", +1, "ΔPreservation (d′ G2−C2, ≥−ε acceptable)")]:
        vals = [r[col] for r in cs_rows]
        errs = [r[col.replace("mean", "std")] for r in cs_rows]
        colors = ["#55A868" if v * favorable_sign > 0 else "#C44E52" for v in vals]
        ax_.bar(CATEGORIES, vals, yerr=errs, capsize=4, color=colors)
        ax_.axhline(0, color="k", lw=0.8)
        if favorable_sign == 1:
            ax_.axhline(-EPS, color="#C44E52", lw=0.8, ls="--", label=f"−ε={-EPS}")
            ax_.legend()
        ax_.set_title(ttl, fontsize=10)
        ax_.set_xticklabels(CATEGORIES, rotation=20)
    fig.suptitle("Per-category consistency (mean ± std over 3 seeds)", y=1.02)
    fig.tight_layout(); fig.savefig(FIG_DIR / "category_consistency.png", dpi=150, bbox_inches="tight"); plt.close(fig)

    # 4) seed consistency：每 unit pair_win 标记
    fig, ax = plt.subplots(figsize=(10, 4))
    pw = [r["pair_win"] for r in pair_rows]
    colors = ["#55A868" if v else "#C44E52" for v in pw]
    ax.bar(range(len(units)), [1 if v else 0 for v in pw], color=colors)
    ax.set_xticks(range(len(units)))
    ax.set_xticklabels([f"{c[:4]}:{s}" for c, s in units], rotation=60, fontsize=7)
    ax.set_yticks([0, 1]); ax.set_yticklabels(["LOSE", "WIN"])
    ax.set_title("PAIR WIN (ΔRob<0 AND ΔPres≥−ε) per unit — green = win")
    fig.tight_layout(); fig.savefig(FIG_DIR / "seed_consistency.png", dpi=150); plt.close(fig)

    # 5) pareto by category（seed-mean ± std）
    fig, axes = plt.subplots(1, 5, figsize=(20, 4))
    for ax_, cat in zip(axes, CATEGORIES):
        for cfg in CONFIGS:
            xs = [m[(cat, s, cfg)]["mean_abs_delta_z"] for s in SEEDS]
            ys = [m[(cat, s, cfg)]["mean_dprime"] for s in SEEDS]
            ax_.errorbar(np.mean(xs), np.mean(ys), xerr=np.std(xs), yerr=np.std(ys),
                         fmt="o" if cfg != "G2" else "D", color=KIND_COLOR[cfg],
                         markersize=7, capsize=3, label=cfg)
        ax_.set_title(cat)
        ax_.set_xlabel("mean |Δz|"); ax_.set_ylabel("mean d′")
        ax_.grid(alpha=0.3)
    axes[0].legend(fontsize=8)
    fig.suptitle("Pareto per category (mean ± std over 3 seeds; diamond = G2)", y=1.03)
    fig.tight_layout(); fig.savefig(FIG_DIR / "pareto_by_category.png", dpi=150, bbox_inches="tight"); plt.close(fig)

    # 6) defect-level effects
    fig, ax = plt.subplots(figsize=(13, 5))
    labels, vals, cats = [], [], []
    for cat in CATEGORIES:
        dts = sorted({r["defect_type"] for r in pd_rows if r["category"] == cat})
        for dt in dts:
            vv = [r["delta_dprime"] for r in pd_rows if r["category"] == cat and r["defect_type"] == dt]
            labels.append(f"{cat[:4]}/{dt[:12]}"); vals.append(np.mean(vv)); cats.append(cat)
    colors = ["#4C72B0" if v > EPS else ("#C44E52" if v < -EPS else "#999999") for v in vals]
    ax.bar(range(len(vals)), vals, color=colors)
    ax.axhline(0, color="k", lw=0.8); ax.axhline(EPS, color="#999", ls="--", lw=0.8)
    ax.axhline(-EPS, color="#999", ls="--", lw=0.8, label=f"±ε={EPS}")
    ax.set_xticks(range(len(vals)))
    ax.set_xticklabels(labels, rotation=70, fontsize=7)
    ax.set_ylabel("Δd′ (G2 − C2), seed-mean")
    ax.set_title("Defect-level effect (blue = improved, red = degraded, gray = neutral, |Δ|≤ε)")
    ax.legend()
    fig.tight_layout(); fig.savefig(FIG_DIR / "defect_level_effects.png", dpi=150); plt.close(fig)

    # ---------- 控制台摘要 ----------
    print("\n=== G2 vs C2 per unit ===")
    for r in pair_rows:
        print(f"{r['category']:>9}:{r['seed']} Δrob={r['delta_robustness']:+.4f} "
              f"Δpres={r['delta_preservation']:+.4f} pairwin={r['pair_win']}")
    print("\n=== per category ===")
    for r in cs_rows:
        print(f"{r['category']:>9}: rob_wins={r['robustness_wins']}/3 "
              f"pres_ni={r['preservation_noninferior']}/3 pair_wins={r['pair_wins']}/3 "
              f"beyond={r['pareto_beyond']}/3 domC2={r['dominates_C2']}/3 "
              f"Δrob={r['delta_robustness_mean']:+.4f}±{r['delta_robustness_std']:.4f} "
              f"Δpres={r['delta_preservation_mean']:+.4f}±{r['delta_preservation_std']:.4f}")
    print("\n=== overall ===")
    for r in cc_rows:
        print(r)
    print("\n=== stats ===")
    for r in stat_rows:
        print(r)
    print("\nfigures ->", FIG_DIR)


if __name__ == "__main__":
    main()
