"""Experiment 1H-S — Dispersion-Aligned Layer Analysis.

把 1H 的层偏好从 Δmean_defect 口径切换到 Δstd_defect（dispersion），与 1E/1G 主线对齐。
**不训练、不跑 GPU**，纯读 1H 已产出的 raw_results.csv（逐 seed 的 std_defect）。

输入：results/experiment_1h/analysis/raw_results.csv（1125 行 = 25 defect × 3 loc × 3 seed × 5 α）
       results/experiment_1g/selection.csv（9 个 primary defect 的冻结 family 标签）

产出（results/experiment_1hs/analysis/）：
  dispersion_layer_summary.csv  —— 每 defect×location：Δstd 的 3-seed mean/std/sign_consistency/CI
  lsi_std.csv                   —— 每 defect：Δstd_L2 / Δstd_L3 / Δstd_concat + LSI_std（带符号）
  additivity.csv                —— 25 defect 全算的 E_add
  family_alignment.csv          —— 9 primary defect 的 family 级对齐统计
  figures/ figure1 Δstd 热力图、figure2 四象限图、figure3 加和性散点

判据（冻结，见 README）：
  CASE A strong alignment / B partial / C mean-only / D no robust structure
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
H_OUT = PROJECT_ROOT / "results" / "experiment_1h"
RAW = H_OUT / "analysis" / "raw_results.csv"
SEL_CSV = PROJECT_ROOT / "results" / "experiment_1g" / "selection.csv"
OUT_ROOT = PROJECT_ROOT / "results" / "experiment_1hs"
ANALYSIS_DIR = OUT_ROOT / "analysis"
FIG_DIR = OUT_ROOT / "figures"

LOCATIONS = ["layer2", "layer3", "post_concat"]
EPS = 1e-9


def load_raw() -> list[dict]:
    rows = list(csv.DictReader(open(RAW, encoding="utf-8")))
    for r in rows:
        r["seed"] = int(r["seed"])
        r["alpha"] = float(r["alpha"])
        for k in ["std_defect", "mean_defect", "std_good", "d_prime", "image_auroc"]:
            r[k] = float(r[k]) if r[k] not in ("", "nan") else float("nan")
    return rows


def load_family() -> dict[tuple[str, str], str]:
    """返回 {(category, defect_type): family}，仅 primary 组（冻结标签）。"""
    fam = {}
    for r in csv.DictReader(open(SEL_CSV, encoding="utf-8")):
        if r.get("selection") == "primary":
            fam[(r["category"], r["defect_type"])] = r["family"]
    return fam


def delta_std_per_seed(rows: list[dict], cat: str, dt: str, loc: str) -> list[float]:
    """返回该 defect×location 的逐 seed Δstd = std(α=1) − std(α=0)。"""
    by_seed = {}
    for r in rows:
        if r["category"] == cat and r["defect_type"] == dt and r["location"] == loc:
            by_seed.setdefault(r["seed"], {})[r["alpha"]] = r["std_defect"]
    out = []
    for seed in sorted(by_seed):
        if 0.0 in by_seed[seed] and 1.0 in by_seed[seed]:
            a0, a1 = by_seed[seed][0.0], by_seed[seed][1.0]
            if not (np.isnan(a0) or np.isnan(a1)):
                out.append(a1 - a0)
    return out


def bootstrap_ci(x: list[float], n_boot: int = 5000, rng: np.random.Generator | None = None) -> tuple[float, float]:
    """bootstrap 95% CI（percentile），样本数 < 3 时返回 (mean, mean)。"""
    x = np.asarray(x, dtype=float)
    if len(x) < 2:
        return float(x.mean()) if len(x) else float("nan"), float(x.mean()) if len(x) else float("nan")
    rng = rng or np.random.default_rng(42)
    boots = [rng.choice(x, size=len(x), replace=True).mean() for _ in range(n_boot)]
    return float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def sign_consistency(x: list[float]) -> float:
    """3-seed 方向一致率：同号的比例（0 记为不一致）。"""
    x = np.asarray(x, dtype=float)
    if len(x) == 0:
        return float("nan")
    pos = (x > 0).sum()
    neg = (x < 0).sum()
    return float(max(pos, neg) / len(x))


def lsi_std(dl2: float, dl3: float) -> float:
    """LSI_std = (|ΔL3|−|ΔL2|) / (|ΔL3|+|ΔL2|+ε)，−1=layer2 主导，+1=layer3 主导。"""
    a2, a3 = abs(dl2), abs(dl3)
    return (a3 - a2) / (a3 + a2 + EPS)


def main() -> None:
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    rows = load_raw()
    family = load_family()

    # 枚举全部 defect（按 category + defect_type 字典序）
    defects = sorted({(r["category"], r["defect_type"]) for r in rows})

    # ---- 1) 每 defect×location 的 Δstd 逐 seed 统计 ----
    summary_rows = []
    for cat, dt in defects:
        for loc in LOCATIONS:
            d = delta_std_per_seed(rows, cat, dt, loc)
            if not d:
                continue
            mean = float(np.mean(d))
            std = float(np.std(d, ddof=1)) if len(d) > 1 else 0.0
            ci_lo, ci_hi = bootstrap_ci(d)
            summary_rows.append({
                "category": cat, "defect_type": dt, "location": loc,
                "n_seed": len(d), "delta_std_mean": mean, "delta_std_sd": std,
                "sign_consistency": sign_consistency(d),
                "ci_low": ci_lo, "ci_high": ci_hi,
                "seeds": ";".join(f"{v:.6f}" for v in d),
            })
    with open(ANALYSIS_DIR / "dispersion_layer_summary.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        w.writeheader()
        w.writerows(summary_rows)

    sidx = {(r["category"], r["defect_type"], r["location"]): r for r in summary_rows}

    # ---- 2) LSI_std（带符号）+ Δstd_L2/L3/concat ----
    lsi_rows = []
    for cat, dt in defects:
        dl2 = sidx.get((cat, dt, "layer2"), {}).get("delta_std_mean")
        dl3 = sidx.get((cat, dt, "layer3"), {}).get("delta_std_mean")
        dconc = sidx.get((cat, dt, "post_concat"), {}).get("delta_std_mean")
        if dl2 is None or dl3 is None:
            continue
        lsi = lsi_std(dl2, dl3)
        # 描述性分类：谁幅度大
        if abs(dl3) > abs(dl2):
            pref = "layer3"
        elif abs(dl2) > abs(dl3):
            pref = "layer2"
        else:
            pref = "mixed"
        lsi_rows.append({
            "category": cat, "defect_type": dt, "family": family.get((cat, dt), ""),
            "delta_std_L2": dl2, "delta_std_L3": dl3, "delta_std_concat": dconc,
            "LSI_std": lsi, "preferred_layer": pref,
            "sign_L2": "shrink" if dl2 < 0 else ("expand" if dl2 > 0 else "zero"),
            "sign_L3": "shrink" if dl3 < 0 else ("expand" if dl3 > 0 else "zero"),
        })
    with open(ANALYSIS_DIR / "lsi_std.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(lsi_rows[0].keys()))
        w.writeheader()
        w.writerows(lsi_rows)

    # ---- 3) 加和性 E_add（25 defect 全算，不只看 grid）----
    add_rows = []
    for r in lsi_rows:
        s = r["delta_std_L2"] + r["delta_std_L3"]
        c = r["delta_std_concat"]
        e_add = abs(c - s) / (abs(c) + EPS)
        add_rows.append({
            "category": r["category"], "defect_type": r["defect_type"], "family": r["family"],
            "delta_std_L2": r["delta_std_L2"], "delta_std_L3": r["delta_std_L3"],
            "sum_L2_L3": s, "delta_std_concat": c, "E_add": e_add,
            "approx_additive": 1 if e_add < 0.15 else 0,
        })
    with open(ANALYSIS_DIR / "additivity.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(add_rows[0].keys()))
        w.writeheader()
        w.writerows(add_rows)

    # ---- 4) family alignment（冻结 9 个 primary，不做花哨显著性）----
    fam_rows = []
    for fam_name in ["shrink", "neutral", "expand"]:
        members = [r for r in lsi_rows if r["family"] == fam_name]
        if not members:
            continue
        lsis = [r["LSI_std"] for r in members]
        s2 = [r["delta_std_L2"] for r in members]
        s3 = [r["delta_std_L3"] for r in members]
        ci_lo, ci_hi = bootstrap_ci(lsis)
        fam_rows.append({
            "family": fam_name, "n": len(members),
            "lsi_mean": float(np.mean(lsis)), "lsi_sd": float(np.std(lsis, ddof=1)),
            "lsi_ci_low": ci_lo, "lsi_ci_high": ci_hi,
            "sign_L2_consistency": sign_consistency(s2),
            "sign_L3_consistency": sign_consistency(s3),
            "delta_std_L2_mean": float(np.mean(s2)),
            "delta_std_L3_mean": float(np.mean(s3)),
            # effect size：与 0 的距离 / 组内 sd（简单 Cohen-like）
            "lsi_effect_size": float(abs(np.mean(lsis)) / (np.std(lsis, ddof=1) + EPS)),
        })
    if fam_rows:
        with open(ANALYSIS_DIR / "family_alignment.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(fam_rows[0].keys()))
            w.writeheader()
            w.writerows(fam_rows)

    # ---- 图 ----
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        _make_figures(defects, summary_rows, lsi_rows, add_rows, family)
    except Exception as e:
        print(f"[warn] 图生成失败: {e}")

    # ---- 汇总打印 ----
    print(f"[done] defects={len(defects)} summary={len(summary_rows)} lsi={len(lsi_rows)} add={len(add_rows)}")
    n_add = sum(1 for r in add_rows if r["approx_additive"])
    print(f"[additivity] 近似加和(误差<15%) defect: {n_add}/{len(add_rows)}")
    for fam_name in ["shrink", "neutral", "expand"]:
        fr = [r for r in fam_rows if r["family"] == fam_name]
        if fr:
            r = fr[0]
            print(f"[family] {fam_name:8s} n={r['n']} LSI_mean={r['lsi_mean']:+.3f} "
                  f"(CI {r['lsi_ci_low']:+.3f},{r['lsi_ci_high']:+.3f}) "
                  f"ΔL2={r['delta_std_L2_mean']:+.3f} ΔL3={r['delta_std_L3_mean']:+.3f} "
                  f"sign_L2_consist={r['sign_L2_consistency']:.2f} sign_L3_consist={r['sign_L3_consistency']:.2f}")
    print(f"[done] 输出目录 = {ANALYSIS_DIR}")


def _make_figures(defects, summary_rows, lsi_rows, add_rows, family):
    import matplotlib.pyplot as plt

    sidx = {(r["category"], r["defect_type"], r["location"]): r["delta_std_mean"] for r in summary_rows}

    # Figure 1: Δstd 热力图（defect × location），RdBu_r，对称 vmax
    labels = [f"{c}/{d}" for c, d in defects]
    mat = np.full((len(defects), 3), np.nan)
    for i, (c, d) in enumerate(defects):
        for j, loc in enumerate(LOCATIONS):
            mat[i, j] = sidx.get((c, d, loc), np.nan)
    fig, ax = plt.subplots(figsize=(6, max(4, len(defects) * 0.35)))
    vmax = np.nanmax(np.abs(mat)) if np.isfinite(mat).any() else 1.0
    im = ax.imshow(mat, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    ax.set_xticks(range(3)); ax.set_xticklabels(LOCATIONS)
    ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels, fontsize=7)
    ax.set_title("1H-S Figure 1: Defect × Layer — Δstd_defect")
    plt.colorbar(im, ax=ax)
    plt.tight_layout(); plt.savefig(FIG_DIR / "figure1_dstd_heatmap.png", dpi=150); plt.close()

    # Figure 2: 四象限图（横轴 Δstd_L2，纵轴 Δstd_L3）
    d2 = np.array([r["delta_std_L2"] for r in lsi_rows])
    d3 = np.array([r["delta_std_L3"] for r in lsi_rows])
    fig, ax = plt.subplots(figsize=(7, 7))
    fam_colors = {"shrink": "tab:blue", "neutral": "gray", "expand": "tab:red"}
    for r in lsi_rows:
        c = fam_colors.get(r["family"], "lightgray")
        ax.scatter(r["delta_std_L2"], r["delta_std_L3"], c=c, s=45,
                   edgecolors="k" if r["family"] else "none", linewidths=0.6,
                   label=None)
    lim = max(np.nanmax(np.abs(d2)), np.nanmax(np.abs(d3))) * 1.15
    ax.axhline(0, color="k", lw=0.8); ax.axvline(0, color="k", lw=0.8)
    ax.plot([-lim, lim], [-lim, lim], "k--", alpha=0.3, lw=0.8)
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
    ax.set_xlabel("Δstd Layer2 (shrink ← → expand)")
    ax.set_ylabel("Δstd Layer3")
    ax.set_title("1H-S Figure 2: Four-quadrant — Δstd_L2 vs Δstd_L3")
    # 标注 primary 家族点
    for r in lsi_rows:
        if r["family"]:
            ax.annotate(r["defect_type"], (r["delta_std_L2"], r["delta_std_L3"]),
                        fontsize=6, alpha=0.85,
                        xytext=(3, 3), textcoords="offset points")
    from matplotlib.lines import Line2D
    handles = [Line2D([0], [0], marker="o", color="w", markerfacecolor=c, markersize=8, label=f)
               for f, c in fam_colors.items()]
    ax.legend(handles=handles, loc="best")
    plt.tight_layout(); plt.savefig(FIG_DIR / "figure2_four_quadrant.png", dpi=150); plt.close()

    # Figure 3: 加和性散点（横轴 Δstd_L2+Δstd_L3，纵轴 Δstd_concat）
    fig, ax = plt.subplots(figsize=(7, 7))
    for r in add_rows:
        c = "tab:red" if r["category"] == "grid" else "tab:blue"
        ax.scatter(r["sum_L2_L3"], r["delta_std_concat"], c=c, s=45, alpha=0.8)
    allv = [r["sum_L2_L3"] for r in add_rows] + [r["delta_std_concat"] for r in add_rows]
    lim = max(abs(v) for v in allv) * 1.15
    ax.plot([-lim, lim], [-lim, lim], "k--", lw=0.8, label="perfect additive")
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
    ax.set_xlabel("Δstd_L2 + Δstd_L3")
    ax.set_ylabel("Δstd_concat")
    ax.set_title("1H-S Figure 3: Additivity — Δstd_concat vs Δstd_L2+Δstd_L3\n(grid = red)")
    ax.legend()
    plt.tight_layout(); plt.savefig(FIG_DIR / "figure3_additivity.png", dpi=150); plt.close()


if __name__ == "__main__":
    main()
