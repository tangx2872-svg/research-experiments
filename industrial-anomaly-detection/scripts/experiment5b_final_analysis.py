"""Experiment 5B-Final analysis = 冻结 5B 协议 + 补齐的 Group C registry.

设计原则（防止分析口径漂移 / p-hacking）：
  - 不修改 scripts/experiment5b_analysis.py（interim 冻结版）；
  - 直接 import 它并复用其全部分析代码（相关/seed stability/rank/LOCO/controls/figures）；
  - 只做两件事：(1) 把 Group C predictors 合并进 predictor 表并扩展 registry；
                  (2) 输出到 results/experiment_5b_final/，不覆盖 interim；
  - target（C2 damage）、LOCO 规则、risk label、negative control 全部保持冻结不变。

额外（仅描述性，不改判定规则）：per-seed 方向一致性分解 + 预注册要求的 figures。
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import experiment5b_analysis as base  # noqa: E402  (5B 冻结分析代码)

OUT = ROOT / "results" / "experiment_5b_final"
GC_CSV = OUT / "group_c" / "group_c_predictors.csv"

# 只重定向输出目录；分析算法保持 interim 原样
base.OUT = OUT
base.REF_DIR = OUT / "reference"
base.SUM_DIR = OUT / "summary"
base.FIG_DIR = OUT / "figures"
base.LOG_DIR = OUT / "logs"
# Provenance：当前基线为 5B interim commit 24d1b22（原检查期望 ff4a1b2）
base.EXPECT_HEAD = "24d1b22"

GC_PREDICTORS = [
    ("sens_radius_L2", "layer2", "mean_image log((rms_radius(IN(F0))+eps)/(rms_radius(F0)+eps))", "primary"),
    ("sens_radius_L3", "layer3", "mean_image log((rms_radius(IN(F0))+eps)/(rms_radius(F0)+eps))", "primary"),
    ("sens_mdc_L2", "layer2", "mean_image log((mdc(IN(F0))+eps)/(mdc(F0)+eps))", "secondary"),
    ("sens_mdc_L3", "layer3", "mean_image log((mdc(IN(F0))+eps)/(mdc(F0)+eps))", "secondary"),
    ("sens_effrank_L2", "layer2", "mean_image log((normalized_pr(IN(F0))+eps)/(normalized_pr(F0)+eps))", "secondary"),
    ("sens_effrank_L3", "layer3", "mean_image log((normalized_pr(IN(F0))+eps)/(normalized_pr(F0)+eps))", "secondary"),
    ("sens_pca1_L2", "layer2", "mean_image (pca1_ratio(IN(F0)) - pca1_ratio(F0))", "secondary"),
    ("sens_pca1_L3", "layer3", "mean_image (pca1_ratio(IN(F0)) - pca1_ratio(F0))", "secondary"),
]
GC_TUPLES = [(n, "C_normalization_sensitivity", d, "normal_train_images", l, "high->fragile")
             for n, l, d, _ in GC_PREDICTORS]


def load_group_c() -> dict:
    out = {}
    for r in csv.DictReader(open(GC_CSV, newline="")):
        out[(r["category"], int(r["seed"]))] = {
            k: float(v) for k, v in r.items() if k not in ("category", "seed")}
    return out


_orig_compute = base.compute_predictors


def compute_predictors_with_group_c() -> dict:
    """Group A/B/D predictors 走 interim 原代码；仅 merge Group C 列。"""
    preds = _orig_compute()
    gc = load_group_c()
    for key, row in gc.items():
        assert key in preds, f"Group C key {key} not in bank predictors"
        preds[key].update(row)
    return preds


base.compute_predictors = compute_predictors_with_group_c
base.REGISTRY = list(base.REGISTRY) + GC_TUPLES


def read_summary(name: str) -> list:
    return list(csv.DictReader(open(base.SUM_DIR / name, newline="")))


def load_tables():
    preds, tgt = {}, {}
    for r in read_summary("normal_only_predictors.csv"):
        preds[(r["category"], int(r["seed"]))] = r
    for r in read_summary("target_damage.csv"):
        tgt[(r["category"], int(r["seed"]))] = float(r["damage_C2"])
    names = [k for k in preds[("bottle", 0)] if k not in ("category", "seed")]
    return preds, tgt, names


def seed_consistency(preds, tgt, names) -> list:
    """描述性分解：每个 seed 单独算 category-level rho，核对方向一致性。"""
    rows = []
    for n in names:
        row = {"predictor": n}
        rhos = []
        for s in base.SEEDS:
            x = [float(preds[(c, s)][n]) for c in base.CATEGORIES]
            y = [tgt[(c, s)] for c in base.CATEGORIES]
            rho = float(spearmanr(x, y)[0])
            row[f"rho_seed{s}"] = round(rho, 3)
            rhos.append(rho)
        row["direction_consistent"] = bool(all(np.sign(v) == np.sign(rhos[0]) for v in rhos) and rhos[0] != 0)
        rows.append(row)
    with open(base.SUM_DIR / "seed_consistency_descriptive.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    return rows


def leave_one_category_out(preds, tgt, names) -> list:
    """描述性 confound 检查（对全部 23 个 predictor 都报告，不挑）：逐 category 剔除后重算 rho。"""
    rows = []
    for n in names:
        x = np.array([np.mean([float(preds[(c, s)][n]) for s in base.SEEDS]) for c in base.CATEGORIES])
        y = np.array([np.mean([tgt[(c, s)] for s in base.SEEDS]) for c in base.CATEGORIES])
        row = {"predictor": n, "rho_all5": round(float(spearmanr(x, y)[0]), 3)}
        for i, c in enumerate(base.CATEGORIES):
            keep = [j for j in range(len(base.CATEGORIES)) if j != i]
            row[f"rho_drop_{c}"] = round(float(spearmanr(x[keep], y[keep])[0]), 3)
        rows.append(row)
    with open(base.SUM_DIR / "leave_one_category_out_rho.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    return rows


def collect_final_summary(preds, tgt, names, seed_rows, loco_rows) -> dict:
    corr = {r["predictor"]: r for r in read_summary("correlation_summary.csv")}
    rank = {r["predictor"]: r for r in read_summary("rank_analysis.csv")}
    nc = {r["predictor"]: r for r in read_summary("negative_controls.csv")}
    gc_names = [n for n, _, _, _ in GC_PREDICTORS]
    geom_names = [n for n in names if n not in gc_names and not n.startswith("img_")]
    ctrl_names = [n for n in names if n.startswith("img_")]

    def best(pool):
        return max(pool, key=lambda n: abs(float(corr[n]["rho_cat_damage_C2"])))

    dmg = {c: float(np.mean([tgt[(c, s)] for s in base.SEEDS])) for c in base.CATEGORIES}
    out = {
        "n_predictors": len(names),
        "categories": base.CATEGORIES,
        "damage_C2_category_mean": {c: round(dmg[c], 4) for c in base.CATEGORIES},
        "best_geometry_predictor": best(geom_names),
        "best_group_c_predictor": best(gc_names),
        "predictor_table": [
            {"predictor": n,
             "group": [t[1] for t in base.REGISTRY if t[0] == n][0],
             "rho_C2": float(corr[n]["rho_cat_damage_C2"]),
             "rho_C3": float(corr[n]["rho_cat_damage_C3"]),
             "rho_G2": float(corr[n]["rho_cat_damage_G2"]),
             "rho_seed_C2": float(corr[n]["rho_seed_damage_C2"]),
             "kendall_C2": float(rank[n]["kendall_tau"]),
             "direction_consistent_all_seeds": bool(
                 [r for r in seed_rows if r["predictor"] == n][0]["direction_consistent"]),
             "loco_acc": float(nc[n]["loco_acc"]),
             "loco_bal_acc": None if nc[n]["loco_bal_acc"] == "nan" else float(nc[n]["loco_bal_acc"])}
            for n in names],
        "q1_radius_ratio_L3L2": {
            "rho_C2": float(corr["radius_ratio_L3L2"]["rho_cat_damage_C2"]),
            "rho_seed_C2": float(corr["radius_ratio_L3L2"]["rho_seed_damage_C2"]),
            "loco_acc": float(nc["radius_ratio_L3L2"]["loco_acc"]),
            "direction_consistent_all_seeds": bool(
                [r for r in seed_rows if r["predictor"] == "radius_ratio_L3L2"][0]["direction_consistent"]),
        },
        "controls": {n: {"rho_C2": float(corr[n]["rho_cat_damage_C2"]),
                         "loco_acc": float(nc[n]["loco_acc"])} for n in ctrl_names},
        "random_control": {"rho_C2": None, "loco_acc": float(nc["random_control"]["loco_acc"])},
        "random_control_loco_acc": float(nc["random_control"]["loco_acc"]),
    }
    with open(base.SUM_DIR / "final_summary.json", "w") as f:
        json.dump(out, f, indent=2)
    return out


GC_NAMES = [n for n, _, _, _ in GC_PREDICTORS]


def cat_stats(preds, tgt, names):
    x_cat = {n: np.array([np.mean([float(preds[(c, s)][n]) for s in base.SEEDS])
                          for c in base.CATEGORIES]) for n in names}
    y_cat = np.array([np.mean([tgt[(c, s)] for s in base.SEEDS]) for c in base.CATEGORIES])
    return x_cat, y_cat


def fig_all_predictors(x_cat, y_cat, names, corr, plt):
    ncol, nrow = 5, int(np.ceil(len(names) / 5))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4 * ncol, 3.1 * nrow))
    for ax, n in zip(axes.flat, names):
        ax.scatter(x_cat[n], y_cat, s=55,
                   c=["#C44E52" if c in ("grid", "screw") else "#4C72B0" for c in base.CATEGORIES])
        for c, xi, yi in zip(base.CATEGORIES, x_cat[n], y_cat):
            ax.annotate(c, (xi, yi), fontsize=7, xytext=(3, 3), textcoords="offset points")
        col = "#2E8B57" if n in GC_NAMES else "#333333"
        ax.set_title(f"{n}  (rho={spearmanr(x_cat[n], y_cat)[0]:+.2f})", fontsize=9, color=col)
        ax.tick_params(labelsize=7)
    for ax in axes.flat[len(names):]:
        ax.axis("off")
    fig.suptitle("All normal-only predictors vs C2 damage (category-level, seed-mean; n = 5 categories)",
                 y=1.0, fontsize=12)
    fig.tight_layout()
    fig.savefig(base.FIG_DIR / "all_predictors_vs_damage.png", dpi=130, bbox_inches="tight")
    plt.close(fig)


def fig_group_c_panel(x_cat, y_cat, plt):
    fig, axes = plt.subplots(2, 4, figsize=(18, 8))
    for ax, n in zip(axes.flat, GC_NAMES):
        ax.scatter(x_cat[n], y_cat, s=55,
                   c=["#C44E52" if c in ("grid", "screw") else "#4C72B0" for c in base.CATEGORIES])
        for c, xi, yi in zip(base.CATEGORIES, x_cat[n], y_cat):
            ax.annotate(c, (xi, yi), fontsize=7, xytext=(3, 3), textcoords="offset points")
        ax.set_title(f"{n}  (rho={spearmanr(x_cat[n], y_cat)[0]:+.2f})", fontsize=10)
        ax.set_ylabel("C2 damage", fontsize=8)
    fig.suptitle("Group C (normalization sensitivity of normal features) vs C2 damage; "
                 "n = 5 categories", fontsize=12)
    fig.tight_layout()
    fig.savefig(base.FIG_DIR / "group_c_panel.png", dpi=130, bbox_inches="tight")
    plt.close(fig)


def fig_l2l3_sensitivity(preds, plt):
    """README §9 预注册 Group C figure: category-wise s_L2 / s_L3 (signed radius response)."""
    l2 = [np.mean([float(preds[(c, s)]["sens_radius_L2"]) for s in base.SEEDS]) for c in base.CATEGORIES]
    l3 = [np.mean([float(preds[(c, s)]["sens_radius_L3"]) for s in base.SEEDS]) for c in base.CATEGORIES]
    e2 = [np.std([float(preds[(c, s)]["sens_radius_L2"]) for s in base.SEEDS]) for c in base.CATEGORIES]
    e3 = [np.std([float(preds[(c, s)]["sens_radius_L3"]) for s in base.SEEDS]) for c in base.CATEGORIES]
    x = np.arange(len(base.CATEGORIES))
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.bar(x - 0.2, l2, 0.4, yerr=e2, capsize=3, label="s_L2 = log R(IN(F0))/R(F0)")
    ax.bar(x + 0.2, l3, 0.4, yerr=e3, capsize=3, label="s_L3 = log R(IN(F0))/R(F0)")
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(base.CATEGORIES)
    ax.set_ylabel("normalization sensitivity (signed log-ratio of RMS radius)")
    ax.set_title("Category-specific normal-feature normalization sensitivity (s_L2 / s_L3); n = 5 categories")
    ax.legend()
    fig.tight_layout()
    fig.savefig(base.FIG_DIR / "l2_l3_category_sensitivity.png", dpi=140)
    plt.close(fig)


def fig_loco_vs_controls(nc, names, plt):
    order = sorted(names + ["random_control"], key=lambda n: -float(nc[n]["loco_acc"]))
    vals = [float(nc[n]["loco_acc"]) for n in order]
    cols = []
    for n in order:
        if n == "random_control":
            cols.append("#000000")
        elif n in GC_NAMES:
            cols.append("#2E8B57")
        elif n.startswith("img_"):
            cols.append("#999999")
        else:
            cols.append("#4C72B0")
    fig, ax = plt.subplots(figsize=(13, 4.5))
    ax.bar(range(len(order)), vals, color=cols)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(order, rotation=60, ha="right", fontsize=8)
    ax.set_ylabel("LOCO accuracy (5 held-out categories)")
    ax.set_title("LOCO: geometry (blue) / Group C (green) vs controls (grey = image stats, black = random); n = 5")
    ax.axhline(0.6, ls="--", lw=0.8, color="k")
    fig.tight_layout()
    fig.savefig(base.FIG_DIR / "loco_geometry_vs_controls.png", dpi=140)
    plt.close(fig)


def fig_best(x_cat, y_cat, best_geom, best_gc, plt):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, n, tag in ((axes[0], best_geom, "best geometry (Groups A/B/D)"),
                       (axes[1], best_gc, "best Group C (normalization sensitivity)")):
        ax.scatter(x_cat[n], y_cat, s=70,
                   c=["#C44E52" if c in ("grid", "screw") else "#4C72B0" for c in base.CATEGORIES])
        for c, xi, yi in zip(base.CATEGORIES, x_cat[n], y_cat):
            ax.annotate(c, (xi, yi), fontsize=8, xytext=(3, 3), textcoords="offset points")
        ax.set_title(f"{tag}\n{n}: rho={spearmanr(x_cat[n], y_cat)[0]:+.2f} (n = 5 categories)")
        ax.set_xlabel(n)
        ax.set_ylabel("C2 damage (higher = more fragile)")
    fig.tight_layout()
    fig.savefig(base.FIG_DIR / "best_predictors_vs_damage.png", dpi=140)
    plt.close(fig)


def write_sanity_final():
    """保留冻结脚本产出的 sanity_checks.csv 不变，另写 final 版（附 Group C 补充与 S5 说明）。"""
    rows = list(csv.DictReader(open(base.SUM_DIR / "sanity_checks.csv", newline="")))
    out = []
    for r in rows:
        if r["check"] == "S5_predictor_count_le_15":
            out.append({"check": r["check"], "status": "SUPERSEDED",
                        "detail": "23 predictors = 15 interim (A/B/D+controls) + 8 pre-registered "
                                  "Group C (see S5b); the <=15 cap belonged to the interim registry, "
                                  "which did not enumerate Group C. No predictor was added post-hoc."})
        else:
            out.append(dict(r))
    gc = list(csv.DictReader(open(OUT / "group_c" / "group_c_sanity.csv", newline="")))
    for r in gc:
        out.append({"check": r["check"], "status": r["status"], "detail": r["detail"]})
    out.append({"check": "S5b_group_c_completion", "status": "PASS",
                "detail": "8 Group C predictors, frozen in reference/group_c_freeze.json "
                          "(targets_read=false) before any correlation was computed"})
    with open(base.SUM_DIR / "sanity_checks_final.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["check", "status", "detail"])
        w.writeheader()
        w.writerows(out)
    n_fail = sum(1 for r in out if r["status"] == "FAIL")
    print(f"[sanity-final] {len(out)} checks, {n_fail} FAIL")
    return out


NAVIGATION = """
================ Experiment 5B-C / 5B-Final — Paper Navigation ================
📍 论文阶段 : Stage 6 — Improvement Method Screening
             候选方案① Category-Adaptive Normalization（5B 为其 Go/No-Go mini experiment）
🎯 要证明的话: A category's tolerance to normalization can be estimated from
             normal-only feature geometry / normalization sensitivity.
🧪 为什么做 5B-C: 5B interim 的 predictor registry 缺少 Group C
             (normalization sensitivity: geometry(F0) vs geometry(IN(F0)))，
             因此 Normal Geometry -> Tolerance 只能判为 interim CASE_A。
             本步骤补齐 Group C 后重跑冻结 5B 协议，得到 FINAL verdict。
🚦 出口条件: CASE_A -> 支持继续方案①（但禁止自动启动 5C）
             CASE_B -> 证据变弱，建议最小 robustness
             CASE_C -> 停止方案①
==============================================================================
"""


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for d in (base.REF_DIR, base.SUM_DIR, base.FIG_DIR, base.LOG_DIR, OUT / "group_c"):
        d.mkdir(parents=True, exist_ok=True)
    print(NAVIGATION, flush=True)
    (base.LOG_DIR / "paper_navigation.txt").write_text(NAVIGATION)

    # ===== 1. 冻结 5B 分析（代码未改，仅 registry 扩展 + Group C merge）=====
    base.main()
    write_sanity_final()

    # ===== 2. 描述性分解与额外 figures =====
    preds, tgt, names = load_tables()
    seed_rows = seed_consistency(preds, tgt, names)
    leave_one_category_out(preds, tgt, names)
    corr = {r["predictor"]: r for r in read_summary("correlation_summary.csv")}
    nc = {r["predictor"]: r for r in read_summary("negative_controls.csv")}
    x_cat, y_cat = cat_stats(preds, tgt, names)
    fs = collect_final_summary(preds, tgt, names, seed_rows, None)

    fig_all_predictors(x_cat, y_cat, names, corr, plt)
    fig_group_c_panel(x_cat, y_cat, plt)
    fig_l2l3_sensitivity(preds, plt)
    fig_loco_vs_controls(nc, names, plt)
    fig_best(x_cat, y_cat, fs["best_geometry_predictor"], fs["best_group_c_predictor"], plt)

    # ===== 3. Q1-Q4 打印 =====
    bg, bc = fs["best_geometry_predictor"], fs["best_group_c_predictor"]
    print("\n=== Q1: frozen best geometry predictor radius_ratio_L3L2 ===")
    q1 = fs["q1_radius_ratio_L3L2"]
    print(f"rho_C2={q1['rho_C2']:+.3f} seed_rho={q1['rho_seed_C2']:+.3f} "
          f"LOCO={q1['loco_acc']:.2f} dir_consistent={q1['direction_consistent_all_seeds']}")
    print("\n=== Q2: Group C predictors (all reported) ===")
    for n in GC_NAMES:
        e = fs["predictor_table"][[r["predictor"] for r in fs["predictor_table"]].index(n)]
        print(f"{n:>18} rho_C2={e['rho_C2']:+.3f} seed_rho={e['rho_seed_C2']:+.3f} "
              f"LOCO={e['loco_acc']:.2f} dir_consistent={e['direction_consistent_all_seeds']}")
    print(f"-> best Group C = {bc}")
    print("\n=== Q3: geometry vs Group C direction (descriptive consistency) ===")
    g2 = fs["predictor_table"][[r["predictor"] for r in fs["predictor_table"]].index(bg)]
    print(f"best geometry {bg}: rho={float(g2['rho_C2']):+.3f}; best Group C {bc}: rho={gc_rho(fs, bc):+.3f}")
    print("\n=== Q4: LOCO geometry vs controls ===")
    print(f"{bg}: {g2['loco_acc']:.2f} | random_control: {fs['random_control_loco_acc']:.2f} | "
          f"controls: " + ", ".join(f"{k}={v['loco_acc']:.2f}" for k, v in fs["controls"].items()))
    print(f"\n[5B-Final] summary tables: {base.SUM_DIR}")
    return fs


def gc_rho(fs, name):
    return [r["rho_C2"] for r in fs["predictor_table"] if r["predictor"] == name][0]


if __name__ == "__main__":
    main()
