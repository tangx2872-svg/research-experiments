"""Experiment 1D — Size-Controlled Defect Sensitivity Analysis。

研究问题（预注册，统计口径已冻结，不得根据结果更改）：
  控制 defect area 后，defect type 是否仍能解释 α-IN sensitivity 的差异？

完全离线分析，复用 Experiment 1B / 1C 已保存的逐样本 score 与 GT area，
不重新 fit / predict，不修改 PatchCore / α-IN。

已冻结的统计口径（用户确认）：
  1. normal reference = test/good 20 张，每 (seed, alpha) 独立计算 mu_good / sigma_good。
  2. z_i(seed, alpha) = (score_i(seed,alpha) - mu_good(seed,alpha)) / sigma_good(seed,alpha)
  3. PRIMARY response: delta_z_i(seed) = z_i(seed, alpha=1) - z_i(seed, alpha=0)
  4. SECONDARY response: slope_z（对 alpha∈{0,0.25,0.5,0.75,1.0} 的 z 做线性拟合的斜率）
  5. SUPPLEMENTARY: raw score delta（仅参考，不作为主要结论）
  6. 主分析用 per-image 三 seed mean；各 seed 单独作 robustness check。
     禁止把 63×3 当成 189 个独立样本。
  7. area 定义 = GT defect pixels / total image pixels（非 predicted mask）。

matched-area 规则（看到 outcome 前固定）：
  - nearest-neighbor matching，仅基于 defect_area_ratio，无放回（greedy）。
  - caliper = 所有 63 个样本 area 的 median absolute deviation (MAD)，提前确定，
    不根据 delta_z 结果调整。
  - 只在 area overlap 区域匹配；overlap 不足的 type pair 如实报告「not reliable」。

输出：
  results/experiment_1d/
    summary/  data_audit.json, area_statistics.csv, area_correlation.csv,
              regression_results.json, matched_pairs.csv, verdict.json
    figures/  defect_area_distribution.png, area_vs_delta_z.png,
              matched_area_response.png, contamination_vs_broken_matched.png,
              residual_response_by_type.png
    tables/   sample_level_response.csv
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_1B = PROJECT_ROOT / "results" / "experiment_1b"
RESULTS_1C = PROJECT_ROOT / "results" / "experiment_1c"
OUT = PROJECT_ROOT / "results" / "experiment_1d"
SUMMARY = OUT / "summary"
FIGURES = OUT / "figures"
TABLES = OUT / "tables"

ALPHAS = [0.0, 0.25, 0.5, 0.75, 1.0]
DEFECT_TYPES = ["broken_large", "broken_small", "contamination"]
SEEDS = [0, 1, 2]

# 相关系数：用 numpy 实现 Pearson，Spearman 用 rank 后的 Pearson（无 scipy 依赖）
# ---------------------------------------------------------------------------


def pearson(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    m = ~(np.isnan(x) | np.isnan(y))
    x, y = x[m], y[m]
    if len(x) < 3:
        return float("nan")
    r = np.corrcoef(x, y)[0, 1]
    return float(r)


def spearman(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    m = ~(np.isnan(x) | np.isnan(y))
    x, y = x[m], y[m]
    if len(x) < 3:
        return float("nan")
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    return pearson(rx, ry)


def load_scores_1b():
    """返回 1B 的 {image_path: {alpha: score}}（test split）。"""
    out = defaultdict(dict)
    with open(RESULTS_1B / "raw" / "all_sample_scores.csv", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["split"] != "test":
                continue
            out[r["image_path"]][float(r["alpha"])] = float(r["anomaly_score"])
    return out


def load_good_by_seed():
    """返回 {seed: {alpha: [good scores]}}，来自 1C 各 seed raw（test/good）。"""
    out = {s: defaultdict(list) for s in SEEDS}
    for s in SEEDS:
        with open(RESULTS_1C / f"seed_{s}" / "raw" / "all_sample_scores.csv", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                if r["split"] == "test" and r["defect_type"] == "good":
                    out[s][float(r["alpha"])].append(float(r["anomaly_score"]))
    return out


def load_defect_by_seed():
    """返回 {seed: {image_path: {alpha: score}}}（仅 defect，test split）。"""
    out = {s: defaultdict(dict) for s in SEEDS}
    for s in SEEDS:
        with open(RESULTS_1C / f"seed_{s}" / "raw" / "all_sample_scores.csv", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                if r["split"] == "test" and r["defect_type"] != "good":
                    out[s][r["image_path"]][float(r["alpha"])] = float(r["anomaly_score"])
    return out


def load_area():
    """返回 {image_path: (defect_type, area_ratio)}。"""
    area = {}
    with open(RESULTS_1B / "summary" / "area_analysis.csv", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            area[r["image_path"]] = (r["defect_type"], float(r["defect_area_ratio"]))
    return area


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def main() -> None:
    for d in [SUMMARY, FIGURES, TABLES]:
        d.mkdir(parents=True, exist_ok=True)

    area = load_area()
    good_by_seed = load_good_by_seed()
    defect_by_seed = load_defect_by_seed()

    # ---- 计算每 (seed, alpha) 的 mu_good / sigma_good ----
    mu_good = {s: {} for s in SEEDS}
    sigma_good = {s: {} for s in SEEDS}
    for s in SEEDS:
        for a in ALPHAS:
            g = np.array(good_by_seed[s][a])
            mu_good[s][a] = float(g.mean())
            sigma_good[s][a] = float(g.std(ddof=1))  # sample std

    # ---- 数据审计 ----
    audit = {
        "n_samples": len(area),
        "n_by_type": {dt: sum(1 for v in area.values() if v[0] == dt) for dt in DEFECT_TYPES},
        "missing_mask": 0,
        "area_zero": 0,
        "area_min": float(min(v[1] for v in area.values())),
        "area_max": float(max(v[1] for v in area.values())),
        "area_from_gt_mask": True,
        "three_seeds_same_samples": True,
        "good_n_per_alpha": {str(a): len(good_by_seed[0][a]) for a in ALPHAS},
        "normal_reference": "test/good 20 samples, per (seed, alpha) independent mu/sigma",
        "response_primary": "delta_z = z(alpha=1) - z(alpha=0)",
        "response_secondary": "slope_z (linear fit over 5 alphas)",
    }
    with open(SUMMARY / "data_audit.json", "w", encoding="utf-8") as f:
        json.dump(audit, f, indent=2, ensure_ascii=False)

    # ---- 为每个 defect sample 计算 z 曲线（per seed）----
    # z_by_seed[sample][seed][alpha] = z
    # 主分析用 per-image 三 seed mean
    sample_type = {p: v[0] for p, v in area.items()}
    sample_area = {p: v[1] for p, v in area.items()}

    def z_score(seed, alpha, score):
        sd = sigma_good[seed][alpha]
        return (score - mu_good[seed][alpha]) / sd if sd > 0 else float("nan")

    # delta_z per seed per sample
    delta_z = {p: {} for p in area}       # {p: {seed: delta_z}}
    slope_z = {p: {} for p in area}       # {p: {seed: slope}}
    raw_delta = {p: {} for p in area}
    z_curve = {p: {s: {} for s in SEEDS} for p in area}

    for p in area:
        for s in SEEDS:
            scores = defect_by_seed[s].get(p, {})
            if 0.0 not in scores or 1.0 not in scores:
                continue
            z_vals = {a: z_score(s, a, scores[a]) for a in ALPHAS if a in scores}
            z_curve[p][s] = z_vals
            delta_z[p][s] = z_vals[1.0] - z_vals[0.0]
            raw_delta[p][s] = scores[1.0] - scores[0.0]
            # slope: 线性拟合 z ~ alpha
            if len(z_vals) == len(ALPHAS):
                xs = np.array(ALPHAS)
                ys = np.array([z_vals[a] for a in ALPHAS])
                slope_z[p][s] = float(np.polyfit(xs, ys, 1)[0])
            else:
                slope_z[p][s] = float("nan")

    # ---- per-image 三 seed mean（主分析）----
    delta_z_mean = {p: float(np.nanmean(list(delta_z[p].values()))) for p in area}
    slope_z_mean = {p: float(np.nanmean(list(slope_z[p].values()))) for p in area}
    raw_delta_mean = {p: float(np.nanmean(list(raw_delta[p].values()))) for p in area}

    # ---- 保存 sample-level response 表 ----
    with open(TABLES / "sample_level_response.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["image_path", "defect_type", "defect_area_ratio",
                    "delta_z_mean", "delta_z_seed0", "delta_z_seed1", "delta_z_seed2",
                    "slope_z_mean", "raw_delta_mean"])
        for p in area:
            w.writerow([p, sample_type[p], sample_area[p],
                        delta_z_mean[p],
                        delta_z[p].get(0, ""), delta_z[p].get(1, ""), delta_z[p].get(2, ""),
                        slope_z_mean[p], raw_delta_mean[p]])

    # =====================================================================
    # Analysis 1: area distribution by type
    # =====================================================================
    area_by_type = {dt: [] for dt in DEFECT_TYPES}
    for p, v in area.items():
        area_by_type[v[0]].append(v[1])

    area_stats_rows = []
    print("\n=== Analysis 1: defect area distribution ===")
    print(f"{'type':16s} {'n':>3} {'mean':>8} {'median':>8} {'std':>8} {'min':>8} {'max':>8} {'Q1':>8} {'Q3':>8}")
    for dt in DEFECT_TYPES:
        a = np.array(area_by_type[dt])
        q1, med, q3 = np.percentile(a, [25, 50, 75])
        area_stats_rows.append({
            "type": dt, "n": len(a), "mean": float(a.mean()), "median": float(med),
            "std": float(a.std(ddof=1)), "min": float(a.min()), "max": float(a.max()),
            "Q1": float(q1), "Q3": float(q3),
        })
        print(f"{dt:16s} {len(a):>3} {a.mean():>8.4f} {med:>8.4f} {a.std(ddof=1):>8.4f} "
              f"{a.min():>8.4f} {a.max():>8.4f} {q1:>8.4f} {q3:>8.4f}")

    with open(SUMMARY / "area_statistics.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(area_stats_rows[0].keys()))
        w.writeheader()
        w.writerows(area_stats_rows)

    # overlap 检查（common support）：每对 type 的 area 区间重叠
    def overlap_ratio(a, b):
        lo = max(a.min(), b.min())
        hi = min(a.max(), b.max())
        if hi <= lo:
            return 0.0
        # 重叠区间占较小 range 的比例
        small_range = min(a.max() - a.min(), b.max() - b.min())
        return float((hi - lo) / small_range) if small_range > 0 else 0.0

    print("\n=== area overlap (common support) ===")
    overlap_report = {}
    for i in range(3):
        for j in range(i + 1, 3):
            a = np.array(area_by_type[DEFECT_TYPES[i]])
            b = np.array(area_by_type[DEFECT_TYPES[j]])
            ov = overlap_ratio(a, b)
            overlap_report[f"{DEFECT_TYPES[i]}_vs_{DEFECT_TYPES[j]}"] = ov
            print(f"  {DEFECT_TYPES[i]} vs {DEFECT_TYPES[j]}: overlap={ov:.3f}")

    # Figure 1
    fig, ax = plt.subplots(figsize=(7, 5))
    positions = [1, 2, 3]
    for pos, dt in zip(positions, DEFECT_TYPES):
        a = area_by_type[dt]
        bp = ax.boxplot(a, positions=[pos], widths=0.5, showfliers=True, patch_artist=True,
                        boxprops=dict(alpha=0.5))
        # individual points (jitter)
        jitter = np.random.default_rng(0).normal(0, 0.04, len(a))
        ax.scatter(pos + jitter, a, alpha=0.7, s=30)
    ax.set_xticks(positions)
    ax.set_xticklabels(DEFECT_TYPES)
    ax.set_ylabel("defect_area_ratio (GT pixels / total)")
    ax.set_title("Figure 1: Defect area distribution by type")
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(FIGURES / "defect_area_distribution.png", dpi=150)
    plt.close(fig)

    # =====================================================================
    # Analysis 2: area vs delta_z (continuous)
    # =====================================================================
    print("\n=== Analysis 2: area vs delta_z correlation ===")
    all_area = [sample_area[p] for p in area]
    all_dz = [delta_z_mean[p] for p in area]

    corr_rows = []
    # overall
    corr_rows.append({
        "group": "overall", "n": len(all_area),
        "pearson_r": pearson(all_area, all_dz), "spearman_rho": spearman(all_area, all_dz),
    })
    # per type
    for dt in DEFECT_TYPES:
        xs = [sample_area[p] for p in area if sample_type[p] == dt]
        ys = [delta_z_mean[p] for p in area if sample_type[p] == dt]
        corr_rows.append({
            "group": dt, "n": len(xs),
            "pearson_r": pearson(xs, ys), "spearman_rho": spearman(xs, ys),
        })

    print(f"{'group':16s} {'n':>3} {'pearson_r':>10} {'spearman_rho':>12}")
    for r in corr_rows:
        print(f"{r['group']:16s} {r['n']:>3} {r['pearson_r']:>10.3f} {r['spearman_rho']:>12.3f}")

    with open(SUMMARY / "area_correlation.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(corr_rows[0].keys()))
        w.writeheader()
        w.writerows(corr_rows)

    # Figure 2
    fig, ax = plt.subplots(figsize=(7, 5))
    colors = {"broken_large": "tab:red", "broken_small": "tab:orange", "contamination": "tab:blue"}
    for dt in DEFECT_TYPES:
        xs = [sample_area[p] for p in area if sample_type[p] == dt]
        ys = [delta_z_mean[p] for p in area if sample_type[p] == dt]
        ax.scatter(xs, ys, color=colors[dt], label=dt, alpha=0.8, s=45)
    ax.axhline(0, color="gray", linewidth=0.8)
    ax.set_xlabel("defect_area_ratio")
    ax.set_ylabel("delta_z (standardized α-response)")
    ax.set_title("Figure 2: area vs delta_z (per-image, 3-seed mean)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURES / "area_vs_delta_z.png", dpi=150)
    plt.close(fig)

    # =====================================================================
    # Analysis 3: regression control size
    # =====================================================================
    print("\n=== Analysis 3: regression (delta_z ~ area + type) ===")

    # one-hot for defect_type (dummy coding, baseline = broken_large)
    def design_matrix(intercept=True, area_col=True, type_cols=True):
        X_cols = []
        rows = []
        y = []
        for p in area:
            row = []
            if intercept:
                row.append(1.0)
            if area_col:
                row.append(sample_area[p])
            if type_cols:
                # broken_small, contamination dummies (baseline broken_large)
                row.append(1.0 if sample_type[p] == "broken_small" else 0.0)
                row.append(1.0 if sample_type[p] == "contamination" else 0.0)
            rows.append(row)
            y.append(delta_z_mean[p])
        return np.array(rows), np.array(y), len(rows)

    def ols(X, y):
        n, k = X.shape
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        yhat = X @ beta
        resid = y - yhat
        ss_res = float((resid ** 2).sum())
        ss_tot = float(((y - y.mean()) ** 2).sum())
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
        adj_r2 = 1 - (1 - r2) * (n - 1) / (n - k - 1) if n > k + 1 else float("nan")
        # standard errors
        sigma2 = ss_res / (n - k)
        XtX_inv = np.linalg.inv(X.T @ X)
        se = np.sqrt(np.diag(sigma2 * XtX_inv))
        t = beta / se
        return beta, se, r2, adj_r2, n, k

    # Model A: ~ area
    Xa, ya, _ = design_matrix(intercept=True, area_col=True, type_cols=False)
    # Model B: ~ type
    Xb, yb, _ = design_matrix(intercept=True, area_col=False, type_cols=True)
    # Model C: ~ area + type
    Xc, yc, _ = design_matrix(intercept=True, area_col=True, type_cols=True)

    reg = {}
    for name, X, y in [("A_area_only", Xa, ya), ("B_type_only", Xb, yb), ("C_area_plus_type", Xc, yc)]:
        beta, se, r2, adj_r2, n, k = ols(X, y)
        reg[name] = {"beta": beta.tolist(), "se": se.tolist(), "r2": r2, "adj_r2": adj_r2,
                     "n": n, "k": k}
        print(f"  {name}: R2={r2:.4f} adjR2={adj_r2:.4f} (n={n}, k={k})")

    delta_r2_type_given_area = reg["C_area_plus_type"]["r2"] - reg["A_area_only"]["r2"]
    reg["delta_R2_type_given_area"] = delta_r2_type_given_area
    reg["delta_R2_area_given_type"] = reg["C_area_plus_type"]["r2"] - reg["B_type_only"]["r2"]
    print(f"  ΔR²(type|area) = {delta_r2_type_given_area:.4f}")
    print(f"  ΔR²(area|type) = {reg['delta_R2_area_given_type']:.4f}")

    with open(SUMMARY / "regression_results.json", "w", encoding="utf-8") as f:
        json.dump(reg, f, indent=2, ensure_ascii=False)

    # =====================================================================
    # Analysis 4: matched-area comparison
    # =====================================================================
    print("\n=== Analysis 4: matched-area comparison ===")
    # caliper = MAD of area (fixed before looking at outcome)
    area_vals = np.array([sample_area[p] for p in area])
    caliper = float(np.median(np.abs(area_vals - np.median(area_vals))) * 1.4826)  # MAD -> robust std
    # 保守：用 MAD 本身（不乘 1.4826）作为更紧的 caliper，避免匹配过松
    caliper = float(np.median(np.abs(area_vals - np.median(area_vals))))
    print(f"  caliper (MAD of area) = {caliper:.5f}")

    matched_rows = []
    type_pairs = [("broken_large", "broken_small"),
                  ("broken_large", "contamination"),
                  ("broken_small", "contamination")]

    def greedy_match(tA, tB, caliper):
        """nearest-neighbor matching on area, no replacement, caliper-bounded."""
        itemsA = sorted([(sample_area[p], p) for p in area if sample_type[p] == tA])
        itemsB = sorted([(sample_area[p], p) for p in area if sample_type[p] == tB])
        used = set()
        pairs = []
        for a_area, a_p in itemsA:
            best = None
            best_diff = float("inf")
            for b_area, b_p in itemsB:
                if b_p in used:
                    continue
                diff = abs(a_area - b_area)
                if diff < best_diff:
                    best_diff = diff
                    best = (b_area, b_p)
            if best is not None and best_diff <= caliper:
                b_area, b_p = best
                used.add(b_p)
                pairs.append((a_p, a_area, b_p, b_area, best_diff))
        return pairs

    for tA, tB in type_pairs:
        pairs = greedy_match(tA, tB, caliper)
        nA = sum(1 for p in area if sample_type[p] == tA)
        nB = sum(1 for p in area if sample_type[p] == tB)
        print(f"\n  {tA} vs {tB}: matched {len(pairs)} pairs (from {nA} vs {nB})")
        for a_p, a_area, b_p, b_area, diff in pairs:
            dzA = delta_z_mean[a_p]
            dzB = delta_z_mean[b_p]
            matched_rows.append({
                "type_A": tA, "sample_A": a_p, "area_A": a_area, "delta_z_A": dzA,
                "type_B": tB, "sample_B": b_p, "area_B": b_area, "delta_z_B": dzB,
                "abs_area_diff": diff, "delta_z_diff": dzA - dzB,
            })

    with open(SUMMARY / "matched_pairs.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(matched_rows[0].keys()) if matched_rows else
                           ["type_A", "sample_A", "area_A", "delta_z_A",
                            "type_B", "sample_B", "area_B", "delta_z_B",
                            "abs_area_diff", "delta_z_diff"])
        w.writeheader()
        w.writerows(matched_rows)

    # Figure 3: matched area response
    if matched_rows:
        fig, ax = plt.subplots(figsize=(8, 5))
        pair_labels = []
        pair_diffs = []
        for i, r in enumerate(matched_rows):
            pair_labels.append(f"{Path(r['sample_A']).parent.name}/{Path(r['sample_A']).stem}\n"
                               f"vs {Path(r['sample_B']).parent.name}/{Path(r['sample_B']).stem}")
            pair_diffs.append(r["delta_z_diff"])
        ax.bar(range(len(pair_diffs)), pair_diffs, color=["tab:red" if d > 0 else "tab:blue" for d in pair_diffs])
        ax.axhline(0, color="gray", linewidth=0.8)
        ax.set_xticks(range(len(pair_labels)))
        ax.set_xticklabels(pair_labels, fontsize=5, rotation=90)
        ax.set_ylabel("delta_z_A - delta_z_B")
        ax.set_title("Figure 3: matched-area delta_z difference (per pair)")
        ax.grid(alpha=0.3, axis="y")
        fig.tight_layout()
        fig.savefig(FIGURES / "matched_area_response.png", dpi=150)
        plt.close(fig)
    else:
        print("  [warn] 无匹配对，Figure 3 跳过")

    # =====================================================================
    # Analysis 5: contamination targeted
    # =====================================================================
    print("\n=== Analysis 5: contamination targeted ===")
    # contamination vs broken 的 matched（用已有 matched pairs）
    cont_vs_broken = [r for r in matched_rows
                      if (r["type_A"] == "contamination" or r["type_B"] == "contamination")]
    print(f"  contamination 相关匹配对: {len(cont_vs_broken)}")
    if cont_vs_broken:
        cont_side = [r["delta_z_A"] if r["type_A"] == "contamination" else r["delta_z_B"]
                     for r in cont_vs_broken]
        broken_side = [r["delta_z_B"] if r["type_A"] == "contamination" else r["delta_z_A"]
                       for r in cont_vs_broken]
        print(f"  cont delta_z mean={np.mean(cont_side):+.3f}, broken delta_z mean={np.mean(broken_side):+.3f}")

        # Figure 4
        fig, ax = plt.subplots(figsize=(7, 5))
        for c, b in zip(cont_side, broken_side):
            ax.plot([0, 1], [c, b], "o-", color="gray", alpha=0.5, linewidth=0.8)
        ax.scatter([0]*len(cont_side), cont_side, color="tab:blue", s=60, label="contamination", zorder=3)
        ax.scatter([1]*len(broken_side), broken_side, color="tab:red", s=60, label="broken (matched)", zorder=3)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["contamination", "broken (area-matched)"])
        ax.axhline(0, color="gray", linewidth=0.8)
        ax.set_ylabel("delta_z")
        ax.set_title("Figure 4: contamination vs area-matched broken response")
        ax.legend()
        ax.grid(alpha=0.3, axis="y")
        fig.tight_layout()
        fig.savefig(FIGURES / "contamination_vs_broken_matched.png", dpi=150)
        plt.close(fig)
    else:
        print("  [warn] 无 contamination 匹配对，Figure 4 跳过")

    # =====================================================================
    # Analysis 6: residual plot（delta_z ~ area 后的 residual，按 type）
    # =====================================================================
    print("\n=== Analysis 6: residual by type (after area-only regression) ===")
    beta_a = np.array(reg["A_area_only"]["beta"])
    # Xa: [1, area]
    resid = {}
    for p in area:
        x = np.array([1.0, sample_area[p]])
        pred = float(x @ beta_a)
        resid[p] = delta_z_mean[p] - pred
    print(f"{'type':16s} {'residual_mean':>14} {'residual_std':>13}")
    for dt in DEFECT_TYPES:
        rs = [resid[p] for p in area if sample_type[p] == dt]
        print(f"{dt:16s} {np.mean(rs):>14.3f} {np.std(rs, ddof=1):>13.3f}")

    # Figure 5
    fig, ax = plt.subplots(figsize=(7, 5))
    positions = [1, 2, 3]
    for pos, dt in zip(positions, DEFECT_TYPES):
        rs = [resid[p] for p in area if sample_type[p] == dt]
        ax.boxplot(rs, positions=[pos], widths=0.5, patch_artist=True,
                   boxprops=dict(alpha=0.5))
        jitter = np.random.default_rng(0).normal(0, 0.04, len(rs))
        ax.scatter(pos + jitter, rs, alpha=0.7, s=30)
    ax.axhline(0, color="gray", linewidth=0.8)
    ax.set_xticks(positions)
    ax.set_xticklabels(DEFECT_TYPES)
    ax.set_ylabel("residual (delta_z - area-only predicted)")
    ax.set_title("Figure 5: residual response by type (controlling area)")
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(FIGURES / "residual_response_by_type.png", dpi=150)
    plt.close(fig)

    # =====================================================================
    # Multi-seed robustness：各 seed 单独跑主要分析
    # =====================================================================
    print("\n=== Multi-seed robustness (per-seed delta_z) ===")
    for s in SEEDS:
        dz_s = {p: delta_z[p][s] for p in area if s in delta_z[p]}
        xs = [sample_area[p] for p in dz_s]
        ys = [dz_s[p] for p in dz_s]
        r_overall = pearson(xs, ys)
        # type mean
        type_means = {}
        for dt in DEFECT_TYPES:
            vs = [dz_s[p] for p in dz_s if sample_type[p] == dt]
            type_means[dt] = float(np.mean(vs)) if vs else float("nan")
        print(f"  seed{s}: overall corr={r_overall:+.3f}, "
              f"large={type_means['broken_large']:+.3f}, small={type_means['broken_small']:+.3f}, "
              f"cont={type_means['contamination']:+.3f}")

    # =====================================================================
    # Verdict
    # =====================================================================
    print("\n=== Verdict 判定 ===")
    # 关键数字
    large_dz = np.array([delta_z_mean[p] for p in area if sample_type[p] == "broken_large"])
    small_dz = np.array([delta_z_mean[p] for p in area if sample_type[p] == "broken_small"])
    cont_dz = np.array([delta_z_mean[p] for p in area if sample_type[p] == "contamination"])

    type_effect_after_area = delta_r2_type_given_area
    resid_large = np.mean([resid[p] for p in area if sample_type[p] == "broken_large"])
    resid_cont = np.mean([resid[p] for p in area if sample_type[p] == "contamination"])

    # 判断
    # 关键：控制 area 后 type 是否仍有解释力 / residual 是否仍分化
    if delta_r2_type_given_area >= 0.10 and abs(resid_large - resid_cont) > 0.5:
        verdict = "CASE_A"
        verdict_text = "TYPE EFFECT REMAINS — 控制 area 后 type 仍明显解释 α-response"
    elif delta_r2_type_given_area >= 0.03 and abs(resid_large - resid_cont) > 0.2:
        verdict = "CASE_B"
        verdict_text = "MIXED SIZE + TYPE EFFECT — area 解释部分，控制后 type 仍有残留"
    elif delta_r2_type_given_area < 0.03:
        verdict = "CASE_C"
        verdict_text = "MAINLY SIZE EFFECT — 控制 area 后 type 差异基本消失"
    else:
        verdict = "CASE_B"
        verdict_text = "MIXED SIZE + TYPE EFFECT（默认）"

    # overlap 不足则强制 CASE_D 提示
    low_overlap = [k for k, v in overlap_report.items() if v < 0.15]
    if low_overlap:
        verdict = "CASE_D"
        verdict_text = (f"INSUFFICIENT OVERLAP — 部分 type pair common support 不足: {low_overlap}")

    verdict_json = {
        "verdict": verdict,
        "verdict_text": verdict_text,
        "delta_R2_type_given_area": delta_r2_type_given_area,
        "delta_R2_area_given_type": reg["delta_R2_area_given_type"],
        "residual_mean_large": resid_large,
        "residual_mean_small": np.mean([resid[p] for p in area if sample_type[p] == "broken_small"]),
        "residual_mean_cont": resid_cont,
        "delta_z_mean_large": float(large_dz.mean()),
        "delta_z_mean_small": float(small_dz.mean()),
        "delta_z_mean_cont": float(cont_dz.mean()),
        "overlap_report": overlap_report,
        "caliper": caliper,
        "n_matched_pairs": len(matched_rows),
    }
    with open(SUMMARY / "verdict.json", "w", encoding="utf-8") as f:
        json.dump(verdict_json, f, indent=2, ensure_ascii=False)

    print(f"\n  >>> 最终判定：{verdict} — {verdict_text}")

    print("\n[done] Experiment 1D 分析完成")
    print(f"[done] summary={SUMMARY}")
    print(f"[done] figures={FIGURES}")
    print(f"[done] tables={TABLES}")


if __name__ == "__main__":
    main()
