"""Experiment 5A — 分析脚本（纯 CPU）。

从 results/experiment_5a/raw/per_image_scores.csv + logs/status.json 读取，
计算两轴指标并生成：

  summary/fixed_alpha_summary.csv     （B0–B4 uniform grid + C1–C3 mean-α control）
  summary/geometry_guided_summary.csv （G1–G3）
  summary/pareto_summary.csv          （全部 11 个 config）
  figures/illumination_normal_score.png
  figures/defect_preservation.png
  figures/robustness_preservation_pareto.png      （primary：Y=pooled AUROC）
  figures/robustness_preservation_pareto_dprime.png（secondary：Y=mean d′）

指标口径（协议 §7–§8）：
  Axis A：per shift mean/std score、ΔNormalScore(raw)、ΔNormalScore_z（primary）、
          FPR@τ_val；aggregate = 4 shifts 平均
  Axis B：pooled image AUROC（primary）、per-defect d′（1H 公式）、mean d′、mean_z

不做任何模型/GPU 调用；只读已持久化的分数。
"""

from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "results" / "experiment_5a"
RAW_CSV = OUT_DIR / "raw" / "per_image_scores.csv"
STATUS_JSON = OUT_DIR / "logs" / "status.json"
SUMMARY_DIR = OUT_DIR / "summary"
FIG_DIR = OUT_DIR / "figures"

SHIFTS = ["brightness_0.7", "brightness_1.3", "gamma_0.7", "gamma_1.3"]
DEFECTS = ["broken_large", "broken_small", "contamination"]
EPS = 1e-12

CONFIG_ORDER = ["B0", "B1", "B2", "B3", "B4", "G1", "C1", "G2", "C2", "G3", "C3"]
KIND_LABEL = {"uniform": "Uniform", "guided": "Guided", "mean_control": "Mean-α Control"}
KIND_COLOR = {"uniform": "#4C72B0", "guided": "#C44E52", "mean_control": "#55A868"}


def image_auroc(scores_good: np.ndarray, scores_defect: np.ndarray) -> float:
    y = np.concatenate([np.zeros(len(scores_good)), np.ones(len(scores_defect))])
    s = np.concatenate([scores_good, scores_defect])
    if len(np.unique(y)) < 2:
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    y_sorted = y[order]
    n_pos = int(y.sum())
    n_neg = len(y) - n_pos
    ranks = np.arange(1, len(y) + 1)
    # 处理并列：平均秩
    s_sorted = s[order]
    i = 0
    ranks_avg = np.empty_like(ranks, dtype=np.float64)
    while i < len(s_sorted):
        j = i
        while j + 1 < len(s_sorted) and s_sorted[j + 1] == s_sorted[i]:
            j += 1
        ranks_avg[i:j + 1] = (i + 1 + j + 1) / 2.0
        i = j + 1
    rank_pos = ranks_avg[y_sorted == 1]
    return float((rank_pos.sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def d_prime(scores_defect: np.ndarray, mu_good: float, sigma_good: float) -> float:
    mu_d = float(scores_defect.mean())
    sd_d = float(scores_defect.std(ddof=1))
    pooled = math.sqrt((sd_d**2 + sigma_good**2) / 2.0)
    return (mu_d - mu_good) / pooled if pooled > 0 else float("nan")


def main() -> None:
    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    with open(RAW_CSV, newline="") as f:
        rows = list(csv.DictReader(f))
    status = json.loads(STATUS_JSON.read_text())

    # 组织数据：config -> subset -> shift -> list[(defect_type, score)]
    data: dict[str, dict] = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for r in rows:
        data[r["config"]][r["subset"]][r["shift"]].append((r["defect_type"], float(r["score"])))

    results: dict[str, dict] = {}
    for cfg, dd in data.items():
        clean_good = np.array([s for _, s in dd["clean_good"]["none"]])
        mu_g, sd_g = float(clean_good.mean()), float(clean_good.std(ddof=1))
        tau = status[cfg]["tau_val"]

        # ---- Axis A ----
        axis_a = {}
        for sh in SHIFTS:
            ss = np.array([s for _, s in dd["shift_good"][sh]])
            z = (ss - mu_g) / sd_g if sd_g > 0 else np.full_like(ss, np.nan)
            axis_a[sh] = {
                "mean_score": float(ss.mean()), "std_score": float(ss.std(ddof=1)),
                "delta_normal_score": float(ss.mean() - mu_g),
                "delta_normal_score_z": float(z.mean()),
                "fpr_at_tau_val": float((ss > tau).mean()),
            }
        agg_dz = float(np.mean([axis_a[sh]["delta_normal_score_z"] for sh in SHIFTS]))
        agg_dz_abs = float(np.mean([abs(axis_a[sh]["delta_normal_score_z"]) for sh in SHIFTS]))
        agg_fpr = float(np.mean([axis_a[sh]["fpr_at_tau_val"] for sh in SHIFTS]))

        # ---- Axis B ----
        good_for_auroc = clean_good
        all_defect = np.array([s for sh_l in dd["clean_defect"].values() for _, s in sh_l])
        pooled_auroc = image_auroc(good_for_auroc, all_defect)
        per_defect = {}
        for dt in DEFECTS:
            sd_ = np.array([s for d_, s in dd["clean_defect"]["none"] if d_ == dt])
            per_defect[dt] = {
                "n": len(sd_),
                "mean_score": float(sd_.mean()),
                "d_prime": d_prime(sd_, mu_g, sd_g),
                "mean_z": float(((sd_ - mu_g) / sd_g).mean()) if sd_g > 0 else float("nan"),
                "image_auroc": image_auroc(good_for_auroc, sd_),
            }
        mean_dprime = float(np.mean([per_defect[dt]["d_prime"] for dt in DEFECTS]))
        mean_meanz = float(np.mean([per_defect[dt]["mean_z"] for dt in DEFECTS]))

        results[cfg] = {
            "kind": dd_kind(rows, cfg),
            "alpha_l2": status[cfg]["alpha_l2"], "alpha_l3": status[cfg]["alpha_l3"],
            "tau_val": tau, "mu_good": mu_g, "sd_good": sd_g,
            "axis_a": axis_a,
            "agg_delta_z": agg_dz, "agg_abs_delta_z": agg_dz_abs, "agg_fpr": agg_fpr,
            "pooled_auroc": pooled_auroc,
            "per_defect": per_defect, "mean_dprime": mean_dprime, "mean_meanz": mean_meanz,
            "effective_mean_alpha_simple": (status[cfg]["alpha_l2"] + status[cfg]["alpha_l3"]) / 2,
        }

    ordered = [c for c in CONFIG_ORDER if c in results]
    assert len(ordered) == 11, f"expected 11 configs, got {len(ordered)}"

    # ---- 写 summary CSVs ----
    def metric_row(c: str) -> dict:
        r = results[c]
        row = {
            "config": c, "kind": r["kind"],
            "alpha_l2": r["alpha_l2"], "alpha_l3": r["alpha_l3"],
            "effective_mean_alpha_simple": round(r["effective_mean_alpha_simple"], 4),
            "pooled_image_auroc": round(r["pooled_auroc"], 4),
            "mean_dprime": round(r["mean_dprime"], 4),
            "mean_z_separation": round(r["mean_meanz"], 4),
            "mean_delta_normal_score_z": round(r["agg_delta_z"], 4),
            "mean_abs_delta_normal_score_z": round(r["agg_abs_delta_z"], 4),
            "mean_fpr_at_tau_val": round(r["agg_fpr"], 4),
            "tau_val": round(r["tau_val"], 4),
        }
        for dt in DEFECTS:
            row[f"dprime_{dt}"] = round(r["per_defect"][dt]["d_prime"], 4)
        for sh in SHIFTS:
            row[f"dz_{sh}"] = round(r["axis_a"][sh]["delta_normal_score_z"], 4)
            row[f"fpr_{sh}"] = round(r["axis_a"][sh]["fpr_at_tau_val"], 4)
        return row

    all_rows = [metric_row(c) for c in ordered]
    fields = list(all_rows[0].keys())

    def write_csv(path: Path, cfgs: list[str]) -> None:
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            for c in cfgs:
                w.writerow(next(r for r in all_rows if r["config"] == c))

    write_csv(SUMMARY_DIR / "fixed_alpha_summary.csv", [c for c in ordered if c.startswith(("B", "C"))])
    write_csv(SUMMARY_DIR / "geometry_guided_summary.csv", [c for c in ordered if c.startswith("G")])
    write_csv(SUMMARY_DIR / "pareto_summary.csv", ordered)
    print(f"[ok] wrote 3 summary CSVs ({len(ordered)} configs)")

    # ---- Guided vs Control 对比 ----
    pair_rows = []
    for i in (1, 2, 3):
        g, c = results[f"G{i}"], results[f"C{i}"]
        pair_rows.append({
            "pair": f"G{i} vs C{i}",
            "g_mean_alpha": round(g["effective_mean_alpha_simple"], 4),
            "c_mean_alpha": round(c["effective_mean_alpha_simple"], 4),
            "delta_robustness_mean_dz": round(g["agg_delta_z"] - c["agg_delta_z"], 4),
            "delta_robustness_mean_abs_dz": round(g["agg_abs_delta_z"] - c["agg_abs_delta_z"], 4),
            "delta_robustness_fpr": round(g["agg_fpr"] - c["agg_fpr"], 4),
            "delta_pooled_auroc": round(g["pooled_auroc"] - c["pooled_auroc"], 4),
            "delta_mean_dprime": round(g["mean_dprime"] - c["mean_dprime"], 4),
        })
    with open(SUMMARY_DIR / "guided_vs_control_pairs.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(pair_rows[0].keys()))
        w.writeheader()
        w.writerows(pair_rows)
    print("[ok] wrote guided_vs_control_pairs.csv")

    # ---- Figures ----
    # 1) illumination_normal_score.png：mean Δz per config（按 shift 堆叠分组条形）+ FPR
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    x = np.arange(len(ordered))
    for j, sh in enumerate(SHIFTS):
        vals = [results[c]["axis_a"][sh]["delta_normal_score_z"] for c in ordered]
        axes[0].bar(x + (j - 1.5) * 0.2, vals, width=0.19,
                    label=sh.replace("_", " "))
    axes[0].set_xticks(x); axes[0].set_xticklabels(ordered)
    axes[0].axhline(0, color="k", lw=0.8)
    axes[0].set_ylabel("ΔNormalScore_z (shift − clean)")
    axes[0].set_title("Axis A: illumination-induced normal score shift (lower |Δz| = more robust)")
    axes[0].legend(fontsize=8)
    for c, xi in zip(ordered, x):
        axes[1].bar(xi, results[c]["agg_fpr"], color=KIND_COLOR[results[c]["kind"]])
    axes[1].set_xticks(x); axes[1].set_xticklabels(ordered)
    axes[1].set_ylabel("mean FPR@τ_val")
    axes[1].set_title("Axis A: false positive rate under shift (lower = more robust)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "illumination_normal_score.png", dpi=150)
    plt.close(fig)

    # 2) defect_preservation.png：pooled AUROC + per-defect d′
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for c, xi in zip(ordered, x):
        r = results[c]
        axes[0].bar(xi, r["pooled_auroc"], color=KIND_COLOR[r["kind"]])
    axes[0].set_xticks(x); axes[0].set_xticklabels(ordered)
    axes[0].set_ylim(min(0.5, min(results[c]["pooled_auroc"] for c in ordered) - 0.05), 1.0)
    axes[0].set_ylabel("pooled image AUROC")
    axes[0].set_title("Axis B: clean defect detection (pooled AUROC)")
    wdt = 0.26
    for j, dt in enumerate(DEFECTS):
        axes[1].bar(x + (j - 1) * wdt, [results[c]["per_defect"][dt]["d_prime"] for c in ordered],
                    width=wdt * 0.92, label=dt)
    axes[1].set_xticks(x); axes[1].set_xticklabels(ordered)
    axes[1].set_ylabel("d′ (defect vs good)")
    axes[1].set_title("Axis B: per-defect separability (d′)")
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "defect_preservation.png", dpi=150)
    plt.close(fig)

    # 3) Pareto（primary: Y=AUROC；secondary: Y=mean d′）
    def pareto_fig(ykey: str, ylabel: str, title: str, fname: str) -> None:
        fig, ax = plt.subplots(figsize=(8, 6.5))
        for kind in ("uniform", "mean_control", "guided"):
            pts = [(results[c]["agg_abs_delta_z"], results[c][ykey])
                   for c in ordered if results[c]["kind"] == kind]
            ax.scatter([p[0] for p in pts], [p[1] for p in pts],
                       c=KIND_COLOR[kind], label=KIND_LABEL[kind], s=70, zorder=3)
        for c in ordered:
            r = results[c]
            ax.annotate(c, (r["agg_abs_delta_z"], r[ykey]),
                        textcoords="offset points", xytext=(6, 5), fontsize=9)
        # G-C pair 连线
        for i in (1, 2, 3):
            g, c = results[f"G{i}"], results[f"C{i}"]
            ax.plot([g["agg_abs_delta_z"], c["agg_abs_delta_z"]],
                    [g[ykey], c[ykey]], "k--", lw=0.8, alpha=0.5, zorder=2)
        ax.set_xlabel("mean |ΔNormalScore_z| over 4 shifts  (illumination sensitivity, lower → robust)")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(alpha=0.3)
        ax.legend()
        # uniform frontier 参考（阶梯线）
        uni = sorted([(results[c]["agg_abs_delta_z"], results[c][ykey])
                      for c in ordered if results[c]["kind"] == "uniform"])
        ax.plot([p[0] for p in uni], [p[1] for p in uni], color="#4C72B0",
                lw=1.0, alpha=0.6, zorder=1, label="uniform frontier (ref)")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(FIG_DIR / fname, dpi=150)
        plt.close(fig)

    pareto_fig("pooled_auroc", "pooled clean-defect image AUROC",
               "5A Pareto: illumination robustness vs defect preservation",
               "robustness_preservation_pareto.png")
    pareto_fig("mean_dprime", "mean per-defect d′",
               "5A Pareto (secondary): illumination robustness vs mean d′",
               "robustness_preservation_pareto_dprime.png")
    print("[ok] wrote 4 figures")

    # ---- 打印关键对比（供 CASE 判定）----
    print("\n=== Axis A / B 总表（mean |Δz| 越低越 robust；AUROC/d′ 越高越好）===")
    hdr = f"{'cfg':>4} {'kind':>12} {'a_l2':>6} {'a_l3':>6} {'|Δz|':>7} {'FPR':>6} {'AUROC':>6} {'d′bar':>7}"
    print(hdr)
    for c in ordered:
        r = results[c]
        print(f"{c:>4} {r['kind']:>12} {r['alpha_l2']:>6.3f} {r['alpha_l3']:>6.3f} "
              f"{r['agg_abs_delta_z']:>7.3f} {r['agg_fpr']:>6.3f} "
              f"{r['pooled_auroc']:>6.3f} {r['mean_dprime']:>7.3f}")
    print("\n=== Guided vs Mean-α Control ===")
    for p in pair_rows:
        print(p)


def dd_kind(rows: list[dict], cfg: str) -> str:
    for r in rows:
        if r["config"] == cfg:
            return r["kind"]
    raise KeyError(cfg)


if __name__ == "__main__":
    main()
