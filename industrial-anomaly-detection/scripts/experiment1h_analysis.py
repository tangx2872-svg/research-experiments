"""Experiment 1H — 分析：defect × location × α response + layer selectivity。

读取 results/experiment_1h/<category>/seed_<seed>/<location>/ 的 group_level.csv，
生成：
  raw_results.csv            —— 合并全部 category×seed×location×alpha 的 group 指标
  defect_layer_summary.csv   —— 每 defect × location 的 Δ(α=1−α=0) mean/std/CI
  layer_selectivity.csv      —— LSI = |Δ_layer2 − Δ_layer3| + preferred_sensitive_layer
  spearman.csv               —— Δ_layer2/Δ_layer3/Δ_concat 两两 Spearman
  figures/ figure1~4

Δ 定义与 1E 一致：delta = alpha_1 − alpha_0，对 group 级指标（mean_defect / std_defect / mean_gap / d_prime / image_auroc）。
primary Δ 用 mean_defect（即 1E 的 mean_gap 由 mean_defect − mean_good 决定，这里直接用 score 分布）。
"""

from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT_ROOT = PROJECT_ROOT / "results" / "experiment_1h"
ANALYSIS_DIR = OUT_ROOT / "analysis"
FIG_DIR = OUT_ROOT / "figures"

CATEGORIES = ["bottle", "cable", "hazelnut", "screw", "grid"]
SEEDS = [0, 1, 2]
LOCATIONS = ["layer2", "layer3", "post_concat"]
ALPHAS = [0.0, 0.25, 0.5, 0.75, 1.0]
# Δ 用的指标（group-level，与 1E 的响应维度对齐）
METRICS = ["mean_defect", "std_defect", "mean_gap", "d_prime", "image_auroc"]


def load_group(category: str, seed: int, location: str) -> list[dict]:
    p = OUT_ROOT / category / f"seed_{seed}" / location / "group_level.csv"
    if not p.exists():
        return []
    rows = list(csv.DictReader(open(p, encoding="utf-8")))
    for r in rows:
        r["seed"] = int(r["seed"])
        r["alpha"] = float(r["alpha"])
        for k in ["n_good", "n_defect"]:
            r[k] = int(r[k])
        for k in ["mean_good", "std_good", "mean_defect", "std_defect", "median_defect",
                  "mean_gap", "mean_z", "d_prime", "image_auroc"]:
            r[k] = float(r[k]) if r[k] not in ("", "nan") else float("nan")
    return rows


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    """真 Spearman：rank 变换后算 Pearson（与 scipy.stats.spearmanr 一致）。"""
    if len(a) < 2:
        return float("nan")
    ra = np.argsort(np.argsort(a)).astype(float)  # rank（平均秩近似足够，无并列时严格一致）
    rb = np.argsort(np.argsort(b)).astype(float)
    ra = ra - ra.mean()
    rb = rb - rb.mean()
    denom = np.sqrt((ra ** 2).sum() * (rb ** 2).sum())
    return float((ra * rb).sum() / denom) if denom > 0 else float("nan")


def main() -> None:
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    # ---- 收集全部 group ----
    all_group = []
    for cat in CATEGORIES:
        for seed in SEEDS:
            for loc in LOCATIONS:
                all_group.extend(load_group(cat, seed, loc))

    gidx = {(g["category"], g["defect_type"], g["seed"], g["location"], g["alpha"]): g
            for g in all_group}

    defect_types_by_cat = defaultdict(list)
    for g in all_group:
        if g["defect_type"] not in defect_types_by_cat[g["category"]]:
            defect_types_by_cat[g["category"]].append(g["defect_type"])
    for c in defect_types_by_cat:
        defect_types_by_cat[c].sort()

    # ---- raw_results.csv ----
    raw_fields = ["category", "defect_type", "seed", "location", "alpha",
                  "mean_good", "std_good", "mean_defect", "std_defect", "median_defect",
                  "mean_gap", "mean_z", "d_prime", "image_auroc"]
    with open(ANALYSIS_DIR / "raw_results.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=raw_fields)
        w.writeheader()
        for g in sorted(all_group, key=lambda x: (x["category"], x["defect_type"], x["seed"], x["location"], x["alpha"])):
            w.writerow({k: g[k] for k in raw_fields})

    # ---- defect_layer_summary.csv：每 defect × location 的 Δ（跨 3 seeds）----
    summary_rows = []
    for cat in CATEGORIES:
        for dt in defect_types_by_cat[cat]:
            for loc in LOCATIONS:
                row = {"category": cat, "defect_type": dt, "location": loc}
                for m in METRICS:
                    deltas = []
                    for seed in SEEDS:
                        if (cat, dt, seed, loc, 0.0) in gidx and (cat, dt, seed, loc, 1.0) in gidx:
                            d = gidx[(cat, dt, seed, loc, 1.0)][m] - gidx[(cat, dt, seed, loc, 0.0)][m]
                            if not math.isnan(d):
                                deltas.append(d)
                    if deltas:
                        mean = float(np.mean(deltas))
                        std = float(np.std(deltas, ddof=1)) if len(deltas) > 1 else 0.0
                        n = len(deltas)
                        # 95% CI (t 分布近似，n=3 用 t=4.303)
                        tcrit = {1: float("nan"), 2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776}.get(n, 1.96)
                        se = std / math.sqrt(n) if n > 1 else float("nan")
                        row[f"delta_{m}_mean"] = mean
                        row[f"delta_{m}_std"] = std
                        row[f"delta_{m}_ci_low"] = mean - tcrit * se if n > 1 else mean
                        row[f"delta_{m}_ci_high"] = mean + tcrit * se if n > 1 else mean
                    else:
                        row[f"delta_{m}_mean"] = float("nan")
                        row[f"delta_{m}_std"] = float("nan")
                        row[f"delta_{m}_ci_low"] = float("nan")
                        row[f"delta_{m}_ci_high"] = float("nan")
                summary_rows.append(row)

    summary_fields = ["category", "defect_type", "location"]
    for m in METRICS:
        summary_fields += [f"delta_{m}_mean", f"delta_{m}_std", f"delta_{m}_ci_low", f"delta_{m}_ci_high"]
    with open(ANALYSIS_DIR / "defect_layer_summary.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=summary_fields)
        w.writeheader()
        w.writerows(summary_rows)

    # ---- layer_selectivity.csv ----
    # primary metric = mean_defect（score 分布），额外给 mean_gap
    primary = "mean_defect"
    sel_rows = []
    for cat in CATEGORIES:
        for dt in defect_types_by_cat[cat]:
            d = {loc: None for loc in LOCATIONS}
            for loc in LOCATIONS:
                for r in summary_rows:
                    if r["category"] == cat and r["defect_type"] == dt and r["location"] == loc:
                        d[loc] = r[f"delta_{primary}_mean"]
            dl2, dl3, dconc = d["layer2"], d["layer3"], d["post_concat"]
            if dl2 is None or dl3 is None:
                continue
            lsi = abs(dl2 - dl3)
            # preferred_sensitive_layer：Δ 绝对值更大的层（descriptive，不强行分类）
            if abs(dl2) > abs(dl3):
                pref = "layer2"
            elif abs(dl3) > abs(dl2):
                pref = "layer3"
            else:
                pref = "tie"
            sel_rows.append({
                "category": cat, "defect_type": dt,
                "delta_layer2": dl2, "delta_layer3": dl3, "delta_concat": dconc,
                "LSI": lsi, "preferred_sensitive_layer": pref,
            })
    sel_fields = ["category", "defect_type", "delta_layer2", "delta_layer3", "delta_concat",
                  "LSI", "preferred_sensitive_layer"]
    with open(ANALYSIS_DIR / "layer_selectivity.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=sel_fields)
        w.writeheader()
        w.writerows(sel_rows)

    # ---- Spearman（跨 defect type，用 3-seed mean Δ）----
    if sel_rows:
        d2 = np.array([r["delta_layer2"] for r in sel_rows])
        d3 = np.array([r["delta_layer3"] for r in sel_rows])
        dc = np.array([r["delta_concat"] for r in sel_rows])
        spearman_rows = [
            {"pair": "layer2_vs_layer3", "rho": spearman(d2, d3), "n": len(sel_rows)},
            {"pair": "layer2_vs_concat", "rho": spearman(d2, dc), "n": len(sel_rows)},
            {"pair": "layer3_vs_concat", "rho": spearman(d3, dc), "n": len(sel_rows)},
        ]
        with open(ANALYSIS_DIR / "spearman.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["pair", "rho", "n"])
            w.writeheader()
            w.writerows(spearman_rows)

    # ---- 图（用 matplotlib，无则跳过）----
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        _make_figures(summary_rows, sel_rows, defect_types_by_cat, primary)
    except Exception as e:
        print(f"[warn] 图生成失败: {e}")

    print(f"[done] group={len(all_group)} summary={len(summary_rows)} selectivity={len(sel_rows)}")
    print(f"[done] 输出目录 = {ANALYSIS_DIR}")


def _make_figures(summary_rows, sel_rows, defect_types_by_cat, primary):
    import matplotlib.pyplot as plt

    # Figure 1: defect × location heatmap（Δ primary）
    cats = CATEGORIES
    all_dt = []
    for c in cats:
        for dt in defect_types_by_cat[c]:
            all_dt.append((c, dt))
    if not all_dt:
        return
    mat = np.full((len(all_dt), 3), np.nan)
    for i, (c, dt) in enumerate(all_dt):
        for j, loc in enumerate(LOCATIONS):
            for r in summary_rows:
                if r["category"] == c and r["defect_type"] == dt and r["location"] == loc:
                    mat[i, j] = r[f"delta_{primary}_mean"]
    labels = [f"{c}/{dt}" for c, dt in all_dt]
    fig, ax = plt.subplots(figsize=(6, max(4, len(all_dt) * 0.35)))
    vmax = np.nanmax(np.abs(mat)) if np.isfinite(mat).any() else 1.0
    im = ax.imshow(mat, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    ax.set_xticks(range(3))
    ax.set_xticklabels(LOCATIONS)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_title(f"Figure 1: Defect × Intervention Layer (Δ {primary})")
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "figure1_defect_layer_heatmap.png", dpi=150)
    plt.close()

    # Figure 2: 代表 defect 的 α trajectory（前 9 个）
    from collections import defaultdict
    traj = defaultdict(lambda: defaultdict(list))
    # 需要原始 raw（重新读）
    for cat in cats:
        for seed in SEEDS:
            for loc in LOCATIONS:
                for g in load_group(cat, seed, loc):
                    key = (g["category"], g["defect_type"], loc)
                    traj[key][g["alpha"]].append(g[primary])
    rep = sorted(set(k[:2] for k in traj.keys()))[:9]
    fig, axes = plt.subplots(3, 3, figsize=(12, 10), sharex=True)
    for idx, (c, dt) in enumerate(rep):
        ax = axes[idx // 3][idx % 3]
        for loc in LOCATIONS:
            if (c, dt, loc) not in traj:
                continue
            xs = sorted(traj[(c, dt, loc)].keys())
            ys = [np.nanmean(traj[(c, dt, loc)][x]) for x in xs]
            ax.plot(xs, ys, marker="o", label=loc)
        ax.set_title(f"{c}/{dt}")
        ax.set_xlabel("alpha")
        ax.set_ylabel(primary)
        ax.legend(fontsize=6)
    plt.suptitle("Figure 2: representative defect α trajectories by layer")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "figure2_trajectories.png", dpi=150)
    plt.close()

    if sel_rows:
        # Figure 3: Δ_layer2 vs Δ_layer3 scatter
        d2 = [r["delta_layer2"] for r in sel_rows]
        d3 = [r["delta_layer3"] for r in sel_rows]
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.scatter(d2, d3)
        lim = max(max(abs(x) for x in d2), max(abs(x) for x in d3)) * 1.1
        ax.plot([-lim, lim], [-lim, lim], "k--", alpha=0.4)
        ax.plot([-lim, lim], [0, 0], "k-", alpha=0.2)
        ax.plot([0, 0], [-lim, lim], "k-", alpha=0.2)
        ax.set_xlabel("Δ layer2-IN")
        ax.set_ylabel("Δ layer3-IN")
        ax.set_title("Figure 3: Δ_layer2 vs Δ_layer3")
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        plt.tight_layout()
        plt.savefig(FIG_DIR / "figure3_layer2_vs_layer3.png", dpi=150)
        plt.close()

        # Figure 4: Δ_concat vs Δ_layer2 & Δ_layer3
        dc = [r["delta_concat"] for r in sel_rows]
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        axes[0].scatter(d2, dc)
        axes[0].set_xlabel("Δ layer2")
        axes[0].set_ylabel("Δ concat")
        axes[0].set_title("Δ_concat vs Δ_layer2")
        axes[1].scatter(d3, dc)
        axes[1].set_xlabel("Δ layer3")
        axes[1].set_ylabel("Δ concat")
        axes[1].set_title("Δ_concat vs Δ_layer3")
        plt.tight_layout()
        plt.savefig(FIG_DIR / "figure4_concat_contribution.png", dpi=150)
        plt.close()


if __name__ == "__main__":
    main()
