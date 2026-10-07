"""Experiment 8B — Phase B 分析：score-level probe + S4 + 最终 verdict。

输入：results/experiment8b/probe/raw/**/per_image.csv（probe runner 产出）
      results/experiment8b/verdict_metriclevel.json（metric-level booleans）
      results/experiment_5a_h/raw/{cat}/seed_{s}/config_B0/per_image.csv（历史 α=0，S4 对照）
输出：results/experiment8b/analysis/probe_metrics.csv、probe_vs_reference.csv、
      sanity/S4.csv、verdict.json、figures/fig4b_probe.png、final_report.md

用法：python -u scripts/experiment8b_probe_analyze.py
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CATEGORIES = ["bottle", "cable", "grid", "hazelnut", "screw"]
SEEDS = [0, 1, 2]
SHIFTS = ["brightness_0.7", "brightness_1.3", "gamma_0.7", "gamma_1.3"]
RANDOM_DRAWS = 10
PROBE = ROOT / "results" / "experiment8b" / "probe"
ANA = ROOT / "results" / "experiment8b" / "analysis"
HIST_B0 = ROOT / "results" / "experiment_5a_h" / "raw"


def d_prime(sd_: np.ndarray, mu_g: float, sigma_g: float) -> float:
    mu_d = float(sd_.mean())
    sd_d = float(sd_.std(ddof=1))
    pooled = math.sqrt((sd_d ** 2 + sigma_g ** 2) / 2.0)
    return (mu_d - mu_g) / pooled if pooled > 0 else float("nan")


def unit_metrics(rows: list[dict]) -> dict:
    cg = np.array([float(r["score"]) for r in rows
                   if r["subset"] == "clean_good" and r["shift"] == "none"])
    mu_g, sd_g = float(cg.mean()), float(cg.std(ddof=1))
    dz = []
    for sh in SHIFTS:
        ss = np.array([float(r["score"]) for r in rows
                       if r["subset"] == "shift_good" and r["shift"] == sh])
        dz.append(float(((ss - mu_g) / sd_g).mean()))
    types = sorted({r["defect_type"] for r in rows if r["subset"] == "clean_defect"})
    per_type = {}
    for dt in types:
        sd_ = np.array([float(r["score"]) for r in rows
                        if r["subset"] == "clean_defect" and r["defect_type"] == dt])
        per_type[dt] = {"n": int(sd_.size), "d_prime": d_prime(sd_, mu_g, sd_g)}
    return {"mu_g": mu_g, "sd_g": sd_g, "mean_dprime": float(np.mean([v["d_prime"] for v in per_type.values()])),
            "mean_abs_delta_z": float(np.mean(np.abs(dz))), "per_shift_dz": dz,
            "per_defect": per_type, "n_clean_good": int(cg.size)}


def load_unit(cat: str, seed: int, mask: str) -> list[dict] | None:
    p = PROBE / "raw" / cat / f"seed_{seed}" / mask / "per_image.csv"
    if not p.exists():
        return None
    with open(p, newline="") as f:
        return list(csv.DictReader(f))


def _secondary(cmp_rows: list[dict]) -> dict:
    """预注册的 secondary ratio 观察（不参与 CASE 判定，仅记录）。"""
    out = {"note": ("secondary ratio 50% 的观察；CASE 判定只用 primary ratio 25% 的 C6b。"
                    "不得据此事后升级 verdict。")}
    for variant in ("PROPOSED50", "D-ONLY50", "I-ONLY50"):
        v = [r for r in cmp_rows if r["variant"] == variant]
        if not v:
            continue
        per_cat = {}
        for c in sorted({r["category"] for r in v}):
            rs = [r for r in v if r["category"] == c]
            per_cat[c] = {"delta_dprime_vs_full": float(np.mean([float(r["delta_dprime_vs_full"]) for r in rs])),
                          "delta_abs_dz_vs_full": float(np.mean([float(r["delta_abs_dz_vs_full"]) for r in rs]))}
        out[variant] = {
            "dprime": float(np.mean([float(r["dprime"]) for r in v])),
            "abs_dz": float(np.mean([float(r["abs_dz"]) for r in v])),
            "delta_dprime_vs_full": float(np.mean([float(r["delta_dprime_vs_full"]) for r in v])),
            "delta_abs_dz_vs_full": float(np.mean([float(r["delta_abs_dz_vs_full"]) for r in v])),
            "n_win_vs_full": sum(r["probe_win_vs_full"] == "True" for r in v),
            "n_win_vs_rand": sum(r["probe_win_vs_rand"] == "True" for r in v),
            "n_cells": len(v),
            "n_cells_dprime_loss": sum(float(r["delta_dprime_vs_full"]) < 0 for r in v),
            "per_category": per_cat,
        }
    return out


def main() -> None:
    ANA.mkdir(parents=True, exist_ok=True)
    masks = json.loads((PROBE / "masks.json").read_text())
    metric_verdict = json.loads((ROOT / "results/experiment8b/verdict_metriclevel.json").read_text())

    rows_out, cmp_rows, s4_rows = [], [], []
    cells = [(c, s) for c in CATEGORIES for s in SEEDS]
    for cat, seed in cells:
        names = list(masks[f"{cat}:{seed}"].keys())
        m = {}
        for name in names:
            raw = load_unit(cat, seed, name)
            if raw is None:
                continue
            m[name] = unit_metrics(raw)
            rows_out.append({
                "category": cat, "seed": seed, "mask": name,
                "n_channels_l2": len(masks[f"{cat}:{seed}"][name]["layer2"]),
                "n_channels_l3": len(masks[f"{cat}:{seed}"][name]["layer3"]),
                "mu_g": m[name]["mu_g"], "sd_g": m[name]["sd_g"],
                "mean_dprime": m[name]["mean_dprime"],
                "mean_abs_delta_z": m[name]["mean_abs_delta_z"],
                "dz_brightness_0.7": m[name]["per_shift_dz"][0],
                "dz_brightness_1.3": m[name]["per_shift_dz"][1],
                "dz_gamma_0.7": m[name]["per_shift_dz"][2],
                "dz_gamma_1.3": m[name]["per_shift_dz"][3],
            })
        if "PROPOSED25" not in m or "FULL25" not in m:
            print(f"[probe] {cat}:{seed} incomplete -> skip comparisons", flush=True)
            continue
        full = m["FULL25"]
        rand = [m[k] for k in m if k.startswith("RANDOM25_")]
        r_d = np.array([r["mean_dprime"] for r in rand])
        r_z = np.array([r["mean_abs_delta_z"] for r in rand])
        for variant in [k for k in m if k in ("PROPOSED25", "PROPOSED50", "D-ONLY25", "D-ONLY50",
                                              "I-ONLY25", "I-ONLY50", "PARETO25")]:
            v = m[variant]
            win_full = bool(v["mean_dprime"] >= full["mean_dprime"]
                            and v["mean_abs_delta_z"] <= full["mean_abs_delta_z"]
                            and (v["mean_dprime"] > full["mean_dprime"]
                                 or v["mean_abs_delta_z"] < full["mean_abs_delta_z"]))
            win_rand = bool(v["mean_dprime"] >= r_d.mean() and v["mean_abs_delta_z"] <= r_z.mean()
                            and (v["mean_dprime"] > r_d.mean() or v["mean_abs_delta_z"] < r_z.mean()))
            cmp_rows.append({
                "category": cat, "seed": seed, "variant": variant,
                "dprime": v["mean_dprime"], "abs_dz": v["mean_abs_delta_z"],
                "dprime_full": full["mean_dprime"], "abs_dz_full": full["mean_abs_delta_z"],
                "dprime_rand_mean": float(r_d.mean()), "abs_dz_rand_mean": float(r_z.mean()),
                "dprime_rand_std": float(r_d.std(ddof=1)) if r_d.size > 1 else 0.0,
                "abs_dz_rand_std": float(r_z.std(ddof=1)) if r_z.size > 1 else 0.0,
                "n_random_draws": int(r_d.size),
                "delta_dprime_vs_full": v["mean_dprime"] - full["mean_dprime"],
                "delta_abs_dz_vs_full": v["mean_abs_delta_z"] - full["mean_abs_delta_z"],
                "delta_dprime_vs_rand": v["mean_dprime"] - float(r_d.mean()),
                "delta_abs_dz_vs_rand": v["mean_abs_delta_z"] - float(r_z.mean()),
                "probe_win_vs_full": win_full, "probe_win_vs_rand": win_rand,
                "oracle_probe_only": True,
            })
        # S4：FULL25 vs 历史 α=0（5A-H config_B0）逐图逐条件对比
        hist = HIST_B0 / cat / f"seed_{seed}" / "config_B0" / "per_image.csv"
        raw_now = load_unit(cat, seed, "FULL25")
        if hist.exists() and raw_now is not None:
            with open(hist, newline="") as f:
                old = list(csv.DictReader(f))
            key = lambda r: (r["subset"], r["shift"], r["defect_type"], Path(r["image_path"]).name)
            o = {key(r): float(r["score"]) for r in old}
            diffs, rels, n_cmp = [], [], 0
            for r in raw_now:
                k = key(r)
                if k in o:
                    a, b = float(r["score"]), o[k]
                    diffs.append(abs(a - b))
                    rels.append(abs(a - b) / max(1e-12, abs(b)))
                    n_cmp += 1
            s4_rows.append({"category": cat, "seed": seed, "n_compared": n_cmp,
                            "max_abs_diff": max(diffs) if diffs else None,
                            "max_rel_diff": max(rels) if rels else None,
                            "pass": bool(diffs) and max(rels) <= 1e-4})
    # 写表
    def w(path: Path, rows: list[dict]) -> None:
        if not rows:
            path.write_text("")
            return
        with open(path, "w", newline="") as f:
            wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            wr.writeheader(); wr.writerows(rows)
    w(ANA / "probe_metrics.csv", rows_out)
    w(ANA / "probe_vs_reference.csv", cmp_rows)
    w(ANA / "S4_score_equivalence.csv", s4_rows)

    # ---- Fig 4b / 4c：score-level（oracle diagnostic）----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    FIG = ROOT / "results" / "experiment8b" / "figures"
    FIG.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2), sharex=True, sharey=True)
    for ax, variant, ttl in ((axes[0], "PROPOSED25", "PROPOSED25 (primary ratio)"),
                             (axes[1], "PROPOSED50", "PROPOSED50 (secondary ratio)")):
        prop = [r for r in cmp_rows if r["variant"] == variant]
        for r in prop:
            rd_, rz = float(r["dprime_rand_mean"]), float(r["abs_dz_rand_mean"])
            f_d, f_z = float(r["dprime_full"]), float(r["abs_dz_full"])
            p_d, p_z = float(r["dprime"]), float(r["abs_dz"])
            ax.plot([f_z, p_z], [f_d, p_d], "-", c="#4C72B0", lw=0.8, alpha=0.5)
            ax.scatter(f_z, f_d, marker="*", s=90, c="#000000", zorder=3)
            ax.scatter(rz, rd_, marker="s", s=26, c="#999999", zorder=2)
            ax.scatter(p_z, p_d, marker="o", s=26, c="#C44E52",
                       edgecolors="k" if r["probe_win_vs_full"] == "True" else "none", linewidths=1.2,
                       zorder=4)
        ax.set_title(ttl); ax.set_xlabel("|Δz| (lower = more robust)"); ax.grid(alpha=0.3)
    axes[0].set_ylabel("defect d′ (higher = better preservation)")
    fig.suptitle("Fig 4b — score-level probe per cell: ★ FULL, ■ random-mean, ● PROPOSED "
                 "(black edge = cell win vs FULL)\nORACLE DIAGNOSTIC — ranking uses test defect masks")
    fig.tight_layout(); fig.savefig(FIG / "fig4b_probe_score_level.png", dpi=150); plt.close(fig)

    cats = ["bottle", "cable", "grid", "hazelnut", "screw"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    for ax, key, ttl in ((axes[0], "delta_dprime_vs_full", "Δd′ vs FULL"),
                         (axes[1], "delta_abs_dz_vs_full", "Δ|Δz| vs FULL")):
        W = 0.38
        for i, variant in enumerate(("PROPOSED25", "PROPOSED50")):
            vals = []
            for c in cats:
                v = [float(r[key]) for r in cmp_rows if r["variant"] == variant and r["category"] == c]
                vals.append(np.mean(v) if v else np.nan)
            ax.bar(np.arange(len(cats)) + (i - 0.5) * W, vals, W, label=variant,
                   color=["#C44E52", "#4C72B0"][i])
        ax.axhline(0.0, c="k", lw=0.8)
        ax.set_xticks(range(len(cats)), cats); ax.set_title(ttl); ax.grid(alpha=0.3, axis="y")
    axes[0].legend()
    fig.suptitle("Fig 4c — per-category score-level change vs FULL (negative transfer = cable)")
    fig.tight_layout(); fig.savefig(FIG / "fig4c_probe_per_category.png", dpi=150); plt.close(fig)

    # ---- C6b ----
    p25 = [r for r in cmp_rows if r["variant"] == "PROPOSED25"]
    n_win_full = sum(1 for r in p25 if r["probe_win_vs_full"])
    n_win_rand = sum(1 for r in p25 if r["probe_win_vs_rand"])
    c6b = {
        "n_cells_evaluated": len(p25),
        "n_win_vs_FULL": n_win_full, "n_win_vs_RANDOM_mean": n_win_rand,
        "mean_delta_dprime_vs_full": float(np.mean([r["delta_dprime_vs_full"] for r in p25])) if p25 else None,
        "mean_delta_abs_dz_vs_full": float(np.mean([r["delta_abs_dz_vs_full"] for r in p25])) if p25 else None,
        "mean_delta_dprime_vs_rand": float(np.mean([r["delta_dprime_vs_rand"] for r in p25])) if p25 else None,
        "mean_delta_abs_dz_vs_rand": float(np.mean([r["delta_abs_dz_vs_rand"] for r in p25])) if p25 else None,
        "oracle_probe_only": True,
    }
    c6b["i_pass"] = bool(c6b["n_win_vs_FULL"] >= 10)
    c6b["ii_pass"] = bool(c6b["n_win_vs_RANDOM_mean"] >= 12)
    c6b["iii_pass"] = bool(c6b["mean_delta_dprime_vs_full"] is not None
                           and c6b["mean_delta_dprime_vs_full"] >= 0
                           and c6b["mean_delta_abs_dz_vs_full"] <= 0)
    c6b["iv_pass"] = bool(c6b["mean_delta_dprime_vs_rand"] is not None
                          and c6b["mean_delta_dprime_vs_rand"] > 0
                          and c6b["mean_delta_abs_dz_vs_rand"] < 0)
    c6b["C6b"] = bool(c6b["i_pass"] and c6b["ii_pass"] and c6b["iii_pass"] and c6b["iv_pass"])

    s4_pass = bool(s4_rows) and all(r["pass"] for r in s4_rows)
    # ---- 最终 verdict ----
    b = dict(metric_verdict["booleans"])
    b["C6b"] = c6b["C6b"]
    san_fail = []
    ex = json.loads((ROOT / "results/experiment8b/extract/sanity_extract.json").read_text())
    if not s4_pass:
        san_fail.append("S4")
    if any(not r.get("s13_pass") for c in ex for r in c.get("s13", []) if r.get("status") == "OK"):
        san_fail.append("S13")
    if any(max(v.values()) != 0.0 for c in ex for v in [c["illum"]["s6_identity_max_abs_delta"]]):
        san_fail.append("S6")
    if san_fail:
        case, plan = "CASE_D", "INVALID"
    elif all(b.get(k) for k in ("C1", "C2", "C3", "C4", "C5", "C6a", "C6b", "C7")):
        case, plan = "CASE_A", "GO"
    elif b.get("C1") and b.get("C2"):
        case, plan = "CASE_B", "HOLD"
    elif b.get("n_directional", 0) >= 8:
        case, plan = "CASE_B", "HOLD"
    else:
        case, plan = "CASE_C", "STOP"
    rho = [r["spearman"] for r in metric_verdict["correlation_pooled"]
           if r["layer"] == "both" and r["variant"] == "all"]
    verdict = {
        "experiment": "8B",
        "question": "Does a defect-sensitive but illumination-stable feature subspace exist?",
        "case": case, "plan_c": plan,
        "primary_evidence": {
            "metric_level_booleans": b,
            "booleans_nodegenerate_variant": metric_verdict["booleans_nodegenerate_variant"],
            "score_level_probe_C6b": c6b,
            "S4_score_equivalence": {"pass": s4_pass, "per_cell": s4_rows},
            "sanity_fail_primary": san_fail,
        },
        "category_stability": metric_verdict["category_stability"],
        "seed_stability": {str(k): v for k, v in metric_verdict["seed_stability"].items()},
        "random_baseline": {"metric_level_draws": 1000, "score_level_draws_per_ratio": RANDOM_DRAWS},
        "layer_control": {"n_layer2_win": b.get("n_layer2_win"), "n_layer3_win": b.get("n_layer3_win"),
                          "C7": b.get("C7")},
        "secondary_ratio_observation": _secondary(cmp_rows),
        "correlation_pooled_spearman_both_layers": rho[0] if rho else None,
        "oracle_probe_only": True,
        "leakage_note": ("ranking 使用 test defect mask（oracle）；score-level probe 结果一律为 "
                         "diagnostic，不作为方法性能主张；8B-B normal-only proxy 本轮未开发"),
        "notes": [
            f"primary ratio 25%（50% 仅 secondary）",
            f"directional cells = {b.get('n_directional')}/15",
            f"mean z_D={b.get('mean_z_D'):.3f}, mean z_I={b.get('mean_z_I'):.3f}",
            (f"pooled Spearman(|D|, I) = {rho[0]:.4f}" if rho else "pooled Spearman unavailable"),
        ],
    }
    (ROOT / "results/experiment8b/verdict.json").write_text(json.dumps(verdict, indent=2, ensure_ascii=False))
    print(json.dumps({"case": case, "plan_c": plan, "C6b": c6b["C6b"],
                      "n_win_vs_full": n_win_full, "n_win_vs_rand": n_win_rand,
                      "S4_pass": s4_pass}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
