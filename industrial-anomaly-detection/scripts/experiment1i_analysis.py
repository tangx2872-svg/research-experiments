"""Experiment 1I — Spatial-Statistics Control：分析。

核心：检验「L2 用 256 个 spatial sample 估计 IN 统计量后，是否向 L3 的 response 靠近」。

数据来源：
  - 1H raw_results.csv：α=0 baseline（三 location 共享同一 std_defect）+ L2-standard α=1 + L3-standard α=1
  - 1I <category>/seed_<seed>/phase_<phase>/group_level.csv：matched256 的 α=1 std_defect

Δstd = std(α=1) − std(α=0)，逐 seed 计算（与 1H-S 的 delta_std_per_seed 一致）。
matched256 的 4 phase 共享同一 α=0 baseline（α=0 时 phase 不参与）。

产出（results/experiment_1i/analysis/）：
  experiment1i_primary_results.csv  —— 9 primary defects（逐 seed + 4 phase + gap + LSI）
  experiment1i_all_defects.csv      —— 25 defects
  figures/ figure1~4

判据（冻结，见 README decision tree）：CASE A / B / C。
"""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
H_OUT = PROJECT_ROOT / "results" / "experiment_1h"
I_OUT = PROJECT_ROOT / "results" / "experiment_1i"
RAW = H_OUT / "analysis" / "raw_results.csv"
SEL_CSV = PROJECT_ROOT / "results" / "experiment_1g" / "selection.csv"
ANALYSIS_DIR = I_OUT / "analysis"
FIG_DIR = I_OUT / "figures"

PHASES = ["00", "01", "10", "11"]
EPS = 1e-9


def load_1h() -> dict:
    """返回 {(cat, dt, seed, location, alpha): std_defect}。"""
    idx = {}
    for r in csv.DictReader(open(RAW, encoding="utf-8")):
        key = (r["category"], r["defect_type"], int(r["seed"]), r["location"], float(r["alpha"]))
        v = r["std_defect"]
        idx[key] = float(v) if v not in ("", "nan") else float("nan")
    return idx


def load_1i() -> dict:
    """返回 {(cat, dt, seed, phase): std_defect(α=1)}。"""
    idx = {}
    for cat in ["bottle", "cable", "hazelnut", "screw", "grid"]:
        for seed in [0, 1, 2]:
            for phase in PHASES:
                p = I_OUT / cat / f"seed_{seed}" / f"phase_{phase}" / "group_level.csv"
                if not p.exists():
                    continue
                for r in csv.DictReader(open(p, encoding="utf-8")):
                    if float(r["alpha"]) != 1.0:
                        continue
                    v = r["std_defect"]
                    idx[(cat, r["defect_type"], seed, phase)] = float(v) if v not in ("", "nan") else float("nan")
    return idx


def load_family() -> dict[tuple[str, str], str]:
    fam = {}
    for r in csv.DictReader(open(SEL_CSV, encoding="utf-8")):
        if r.get("selection") == "primary":
            fam[(r["category"], r["defect_type"])] = r["family"]
    return fam


def lsi_std(a2: float, a3: float) -> float:
    return (abs(a3) - abs(a2)) / (abs(a3) + abs(a2) + EPS)


def bootstrap_ci(x: list[float], n_boot: int = 5000, rng=None) -> tuple[float, float]:
    x = np.asarray(x, dtype=float)
    if len(x) < 2:
        m = float(x.mean()) if len(x) else float("nan")
        return m, m
    rng = rng or np.random.default_rng(42)
    boots = [rng.choice(x, size=len(x), replace=True).mean() for _ in range(n_boot)]
    return float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def main() -> None:
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    h1 = load_1h()
    i1 = load_1i()
    family = load_family()

    # 枚举所有 defect
    defects = sorted({(k[0], k[1]) for k in h1})
    primary = [(c, d) for (c, d) in defects if (c, d) in family]

    rows = []
    for cat, dt in defects:
        fam = family.get((cat, dt), "")
        for seed in [0, 1, 2]:
            # baseline α=0（三 location 共享，取 layer2 的值）
            base = h1.get((cat, dt, seed, "layer2", 0.0))
            l2s = h1.get((cat, dt, seed, "layer2", 1.0))
            l3s = h1.get((cat, dt, seed, "layer3", 1.0))
            if base is None or l2s is None or l3s is None:
                continue
            if any(np.isnan(x) for x in (base, l2s, l3s)):
                continue
            d_l2_std = l2s - base
            d_l3_std = l3s - base

            # matched256 各 phase
            phase_d = {}
            for phase in PHASES:
                m = i1.get((cat, dt, seed, phase))
                if m is None or np.isnan(m):
                    phase_d[phase] = float("nan")
                else:
                    phase_d[phase] = m - base
            phase_vals = [phase_d[p] for p in PHASES if not np.isnan(phase_d[p])]
            m_mean = float(np.mean(phase_vals)) if phase_vals else float("nan")
            m_sd = float(np.std(phase_vals, ddof=1)) if len(phase_vals) > 1 else 0.0

            gap_std = abs(d_l3_std) - abs(d_l2_std)
            gap_matched = abs(d_l3_std) - abs(m_mean) if not np.isnan(m_mean) else float("nan")
            gap_red = gap_std - gap_matched
            lsi_std_v = lsi_std(d_l2_std, d_l3_std)
            lsi_matched = lsi_std(m_mean, d_l3_std) if not np.isnan(m_mean) else float("nan")
            delta_lsi = lsi_matched - lsi_std_v

            rows.append({
                "category": cat, "defect_type": dt, "family": fam, "seed": seed,
                "delta_std_l2_standard": d_l2_std,
                "delta_std_l2_matched256_phase00": phase_d["00"],
                "delta_std_l2_matched256_phase01": phase_d["01"],
                "delta_std_l2_matched256_phase10": phase_d["10"],
                "delta_std_l2_matched256_phase11": phase_d["11"],
                "delta_std_l2_matched256_mean": m_mean,
                "delta_std_l2_matched256_sd": m_sd,
                "delta_std_l3_standard": d_l3_std,
                "gap_standard": gap_std,
                "gap_matched": gap_matched,
                "gap_reduction": gap_red,
                "lsi_standard": lsi_std_v,
                "lsi_matched256": lsi_matched,
                "delta_lsi": delta_lsi,
            })

    fields = ["category", "defect_type", "family", "seed",
              "delta_std_l2_standard",
              "delta_std_l2_matched256_phase00", "delta_std_l2_matched256_phase01",
              "delta_std_l2_matched256_phase10", "delta_std_l2_matched256_phase11",
              "delta_std_l2_matched256_mean", "delta_std_l2_matched256_sd",
              "delta_std_l3_standard",
              "gap_standard", "gap_matched", "gap_reduction",
              "lsi_standard", "lsi_matched256", "delta_lsi"]

    with open(ANALYSIS_DIR / "experiment1i_all_defects.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    primary_rows = [r for r in rows if r["family"]]
    with open(ANALYSIS_DIR / "experiment1i_primary_results.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(primary_rows)

    # ---- family-level summary（within-defect 平均跨 seed/phase，再 family 汇总）----
    fam_summary = []
    for fam_name in ["shrink", "neutral", "expand"]:
        members = [r for r in primary_rows if r["family"] == fam_name]
        if not members:
            continue
        # within-defect：对每个 defect 先平均跨 seed 的 lsi_standard / lsi_matched256
        by_defect = defaultdict(list)
        for r in members:
            by_defect[(r["category"], r["defect_type"])].append(r)
        lsi_std_def = [float(np.mean([r["lsi_standard"] for r in v])) for v in by_defect.values()]
        lsi_match_def = [float(np.mean([r["lsi_matched256"] for r in v])) for v in by_defect.values()]
        d2_std_def = [float(np.mean([r["delta_std_l2_standard"] for r in v])) for v in by_defect.values()]
        d2_match_def = [float(np.mean([r["delta_std_l2_matched256_mean"] for r in v])) for v in by_defect.values()]
        d3_std_def = [float(np.mean([r["delta_std_l3_standard"] for r in v])) for v in by_defect.values()]

        ci_std = bootstrap_ci(lsi_std_def)
        ci_match = bootstrap_ci(lsi_match_def)
        fam_summary.append({
            "family": fam_name, "n_defect": len(by_defect),
            "lsi_standard_mean": float(np.mean(lsi_std_def)),
            "lsi_matched256_mean": float(np.mean(lsi_match_def)),
            "delta_lsi_mean": float(np.mean(lsi_match_def)) - float(np.mean(lsi_std_def)),
            "lsi_standard_ci": f"[{ci_std[0]:+.3f},{ci_std[1]:+.3f}]",
            "lsi_matched256_ci": f"[{ci_match[0]:+.3f},{ci_match[1]:+.3f}]",
            "delta_std_l2_standard_mean": float(np.mean(d2_std_def)),
            "delta_std_l2_matched256_mean": float(np.mean(d2_match_def)),
            "delta_std_l3_standard_mean": float(np.mean(d3_std_def)),
        })
    if fam_summary:
        with open(ANALYSIS_DIR / "experiment1i_family_summary.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(fam_summary[0].keys()))
            w.writeheader()
            w.writerows(fam_summary)

    # ---- 图 ----
    try:
        import matplotlib
        matplotlib.use("Agg")
        _make_figures(primary_rows, family, fam_summary)
    except Exception as e:
        print(f"[warn] 图生成失败: {e}")

    # ---- 汇总打印 ----
    print(f"[done] defects={len(defects)} rows={len(rows)} primary={len(primary_rows)}")
    for fs in fam_summary:
        print(f"[family] {fs['family']:8s} n={fs['n_defect']} "
              f"LSI_std={fs['lsi_standard_mean']:+.3f}{fs['lsi_standard_ci']} "
              f"LSI_match={fs['lsi_matched256_mean']:+.3f}{fs['lsi_matched256_ci']} "
              f"ΔLSI={fs['delta_lsi_mean']:+.3f}")
        print(f"         ΔL2_std={fs['delta_std_l2_standard_mean']:+.3f} "
              f"ΔL2_match={fs['delta_std_l2_matched256_mean']:+.3f} "
              f"ΔL3_std={fs['delta_std_l3_standard_mean']:+.3f}")
    print(f"[done] 输出目录 = {ANALYSIS_DIR}")


def _make_figures(primary_rows, family, fam_summary):
    import matplotlib.pyplot as plt

    # 每个 primary defect 的 within-defect 平均（跨 seed/phase）
    by_defect = defaultdict(list)
    for r in primary_rows:
        by_defect[(r["category"], r["defect_type"])].append(r)

    def within_defect_mean(field):
        out = []
        for k, v in sorted(by_defect.items()):
            vals = [r[field] for r in v if not np.isnan(r[field])]
            out.append((k, float(np.mean(vals))))
        return out

    # Figure 1: expand 家族对比（L2-std / L2-match / L3-std）
    expand_defects = [k for k, _ in within_defect_mean("delta_std_l2_standard") if family.get(k) == "expand"]
    fig, ax = plt.subplots(figsize=(8, 6))
    xpos = [0, 1, 2]
    for c, d in expand_defects:
        v2s = float(np.mean([r["delta_std_l2_standard"] for r in by_defect[(c, d)]]))
        v2m = float(np.mean([r["delta_std_l2_matched256_mean"] for r in by_defect[(c, d)]]))
        v3s = float(np.mean([r["delta_std_l3_standard"] for r in by_defect[(c, d)]]))
        ax.plot(xpos, [v2s, v2m, v3s], marker="o", label=f"{c}/{d}")
    ax.set_xticks(xpos)
    ax.set_xticklabels(["L2-standard", "L2-matched256", "L3-standard"])
    ax.axhline(0, color="k", lw=0.8)
    ax.set_ylabel("Δstd_defect")
    ax.set_title("1I Figure 1: Expand family — L2-standard vs L2-matched256 vs L3")
    ax.legend(fontsize=7)
    plt.tight_layout(); plt.savefig(FIG_DIR / "figure1_expand_comparison.png", dpi=150); plt.close()

    # Figure 2: 三个 family panel
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
    for fi, fam in enumerate(["shrink", "neutral", "expand"]):
        ax = axes[fi]
        fam_defects = [k for k in by_defect if family.get(k) == fam]
        for c, d in fam_defects:
            v2s = float(np.mean([r["delta_std_l2_standard"] for r in by_defect[(c, d)]]))
            v2m = float(np.mean([r["delta_std_l2_matched256_mean"] for r in by_defect[(c, d)]]))
            v3s = float(np.mean([r["delta_std_l3_standard"] for r in by_defect[(c, d)]]))
            ax.plot([0, 1, 2], [v2s, v2m, v3s], marker="o", label=d)
        ax.set_xticks([0, 1, 2])
        ax.set_xticklabels(["L2-std", "L2-m256", "L3-std"], fontsize=7)
        ax.axhline(0, color="k", lw=0.8)
        ax.set_title(f"{fam}")
        ax.legend(fontsize=6)
    fig.suptitle("1I Figure 2: All primary families")
    plt.tight_layout(); plt.savefig(FIG_DIR / "figure2_all_families.png", dpi=150); plt.close()

    # Figure 3: LSI before vs after
    fig, ax = plt.subplots(figsize=(7, 7))
    fam_colors = {"shrink": "tab:blue", "neutral": "gray", "expand": "tab:red"}
    for c, d in sorted(by_defect):
        ls = float(np.mean([r["lsi_standard"] for r in by_defect[(c, d)]]))
        lm = float(np.mean([r["lsi_matched256"] for r in by_defect[(c, d)]]))
        ax.scatter(ls, lm, c=fam_colors.get(family.get((c, d)), "lightgray"), s=60, edgecolors="k", linewidths=0.6)
        ax.annotate(d, (ls, lm), fontsize=7, xytext=(3, 3), textcoords="offset points")
    lim = max(abs(ax.get_xlim()[0]), abs(ax.get_xlim()[1]), abs(ax.get_ylim()[0]), abs(ax.get_ylim()[1])) * 1.1
    lim = max(lim, 1.0)
    ax.plot([-lim, lim], [-lim, lim], "k--", lw=0.8, label="y=x")
    ax.axhline(0, color="k", lw=0.6); ax.axvline(0, color="k", lw=0.6)
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
    ax.set_xlabel("LSI_standard (L2-standard)")
    ax.set_ylabel("LSI_matched256 (L2-matched256)")
    ax.set_title("1I Figure 3: LSI before vs after control")
    ax.legend()
    plt.tight_layout(); plt.savefig(FIG_DIR / "figure3_lsi_before_after.png", dpi=150); plt.close()

    # Figure 4: phase robustness（matched256 的 4 phase）
    fig, ax = plt.subplots(figsize=(10, 6))
    for c, d in sorted(by_defect):
        if family.get((c, d)) != "expand":
            continue
        vals = []
        for ph in PHASES:
            ph_vals = [r[f"delta_std_l2_matched256_phase{ph}"] for r in by_defect[(c, d)]
                       if not np.isnan(r[f"delta_std_l2_matched256_phase{ph}"])]
            vals.append(float(np.mean(ph_vals)))
        ax.plot(range(4), vals, marker="o", label=f"{c}/{d}")
    ax.set_xticks(range(4)); ax.set_xticklabels(PHASES)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xlabel("phase"); ax.set_ylabel("Δstd_defect (L2-matched256)")
    ax.set_title("1I Figure 4: Phase robustness (expand)")
    ax.legend(fontsize=7)
    plt.tight_layout(); plt.savefig(FIG_DIR / "figure4_phase_robustness.png", dpi=150); plt.close()


if __name__ == "__main__":
    main()
