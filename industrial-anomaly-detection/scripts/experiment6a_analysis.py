"""Experiment 6A — Geometry-Guided Soft Gating: analysis（CPU-only，GPU 数据到位后运行）。

输入：results/experiment_6a/reference/policy_freeze.json（冻结协议）
      历史 raw（5A-H B0/B2/... + 5A bottle/seed0）+ results/experiment_6a/raw_new（本次 GPU 的 fragile 端）
输出：results/experiment_6a/{raw,summary,sanity,figures}/**

指标口径完全复用 experiment5a_h_analysis（未改动）；判定链按 §9 预注册规则。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import experiment6a_soft_geometry as e6  # noqa: E402

OUT = e6.OUT
REF_DIR, RAW_DIR, SUM_DIR, SAN_DIR, FIG_DIR = (e6.REF_DIR, e6.RAW_DIR, OUT / "summary",
                                               OUT / "sanity", OUT / "figures")
CATEGORIES, SEEDS, FRAGILE, TOLERANT, EPS = e6.CATEGORIES, e6.SEEDS, e6.FRAGILE, e6.TOLERANT, e6.EPS


def eval_alpha_map(lookup: dict, amap: dict) -> dict:
    units = []
    for c in CATEGORIES:
        for s in SEEDS:
            m = lookup[(c, s, e6.akey(amap[c]))]
            units.append({"category": c, "seed": s, "alpha": amap[c], "mean_dprime": m["mean_dprime"],
                          "mean_abs_delta_z": m["mean_abs_delta_z"], "source": m["source"]})
    per_cat = {c: {"mean_dprime": float(np.mean([u["mean_dprime"] for u in units if u["category"] == c])),
                   "mean_abs_delta_z": float(np.mean([u["mean_abs_delta_z"] for u in units
                                                      if u["category"] == c])),
                   "alpha": amap[c]} for c in CATEGORIES}
    return {"alpha_map": amap, "units": units, "per_cat": per_cat,
            "mean_alpha": float(np.mean(list(amap.values()))),
            "mean_dprime": float(np.mean([u["mean_dprime"] for u in units])),
            "mean_abs_delta_z": float(np.mean([u["mean_abs_delta_z"] for u in units]))}


def soft_map(a: float) -> dict:
    return {c: (a if c in FRAGILE else e6.ALPHA_T) for c in CATEGORIES}


def pareto_flags(points: list, tol: float = 1e-9) -> dict:
    dom = {}
    for p in points:
        dom[p["key"]] = sorted(q["key"] for q in points
                               if q["key"] != p["key"] and q["dp"] >= p["dp"] - tol
                               and q["rob"] <= p["rob"] + tol
                               and (q["dp"] > p["dp"] + tol or q["rob"] < p["rob"] - tol))
    return dom


def seed_aggregate(ev: dict, s: int, field: str) -> float:
    return float(np.mean([u[field] for u in ev["units"] if u["seed"] == s]))


def criteria_eval(ev: dict, ev_hg: dict, ev_fx: dict, crit: dict, neg_hg: int) -> dict:
    th_a = crit["A_preservation_retention"]["threshold"]
    b_ceiling = crit["B_robustness_recovery"]["threshold_abs_z_max"]
    b_imp = crit["B_robustness_recovery"]["improvement_required"]
    gains = {c: ev["per_cat"][c]["mean_dprime"] - ev_fx["per_cat"][c]["mean_dprime"] for c in CATEGORIES}
    worst = min(gains, key=lambda c: gains[c])
    worst_gain = gains[worst]
    neg = int(sum(1 for g in gains.values() if g < -EPS))
    A = bool((ev["mean_dprime"] - ev_fx["mean_dprime"]) >= th_a)
    B = bool(ev["mean_abs_delta_z"] <= b_ceiling)
    C = bool(worst_gain >= -EPS and neg <= neg_hg)
    seed_ok, seed_detail = 0, []
    for s in SEEDS:
        d_dp = seed_aggregate(ev, s, "mean_dprime") - seed_aggregate(ev_fx, s, "mean_dprime")
        d_rz = seed_aggregate(ev, s, "mean_abs_delta_z") - seed_aggregate(ev_hg, s, "mean_abs_delta_z")
        ok = bool(d_dp >= th_a and d_rz <= -b_imp)
        seed_ok += int(ok)
        seed_detail.append({"seed": s, "delta_dprime_vs_fixed": round(d_dp, 4),
                            "delta_abs_z_vs_hardGC": round(d_rz, 4), "satisfies_A_and_B": ok})
    D = bool(seed_ok >= crit["D_stability"]["min_seeds"])
    return {"A": A, "B": B, "C": C, "D": D, "A_and_B_and_C_and_D": bool(A and B and C and D),
            "worst_category": worst, "worst_category_gain": round(worst_gain, 6),
            "negative_transfer_count": neg,
            "delta_dprime_vs_fixed": round(ev["mean_dprime"] - ev_fx["mean_dprime"], 6),
            "delta_abs_z_vs_fixed": round(ev["mean_abs_delta_z"] - ev_fx["mean_abs_delta_z"], 6),
            "delta_dprime_vs_hardGC": round(ev["mean_dprime"] - ev_hg["mean_dprime"], 6),
            "delta_abs_z_vs_hardGC": round(ev["mean_abs_delta_z"] - ev_hg["mean_abs_delta_z"], 6),
            "robustness_improvement_vs_hardGC": round(ev_hg["mean_abs_delta_z"] - ev["mean_abs_delta_z"], 6),
            "seeds_satisfying_A_and_B": f"{seed_ok}/3", "seed_detail": seed_detail}


def verdict_ladder(results: list, crit: dict, neg_hg: int) -> dict:
    """§9 预注册判定链（首个命中即判定）。results: [{alpha_F, ev, crit}]"""
    soft = [r for r in results if abs(r["alpha_F"]) > 1e-12 and abs(r["alpha_F"] - 0.5) > 1e-12]
    winners = [r for r in results if r["crit"]["A_and_B_and_C_and_D"]]
    cond_C = {
        "no_soft_alpha_reaches_criterion_A": not any(r["crit"]["A"] for r in soft),
        "some_soft_alpha_has_negative_transfer_worse_than_hardGC":
            any(r["crit"]["negative_transfer_count"] > neg_hg for r in soft),
    }
    cond_D = {
        "no_soft_alpha_recovers_robustness_by_0.02":
            not any(r["crit"]["robustness_improvement_vs_hardGC"] >= 0.02 for r in soft),
    }
    if winners:
        case = "CASE_A"
        means = "Soft Geometry Gating Works: at least one alpha_F satisfies A+B+C+D."
    elif any(cond_C.values()):
        case = "CASE_C"
        means = "Fixed alpha dominates: no soft policy forms a meaningful preservation advantage, or it induces clear negative transfer."
    elif all(cond_D.values()):
        case = "CASE_D"
        means = "Hard GC remains the best Pareto solution: soft policies lose preservation without recovering robustness."
    else:
        case = "CASE_B"
        means = "Trade-off exists but no clear knee: robustness recovers and preservation falls monotonically, but no point satisfies A+B+C+D."
    return {"case": case, "meaning": means,
            "winning_alpha_F": [r["alpha_F"] for r in winners],
            "conditions_CASE_C": cond_C, "conditions_CASE_D": cond_D,
            "criteria_summary": {f"aF={r['alpha_F']:.8f}": {k: r["crit"][k] for k in ("A", "B", "C", "D")}
                                 for r in results}}


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------
COL = {"Original": "#999999", "BestFixed": "#333333", "HardGC": "#2E8B57", "SoftGC": "#4C72B0"}


def fig_soft_gc_pareto(points: list) -> None:
    fig, ax = plt.subplots(figsize=(8.4, 5.8))
    soft = [p for p in points if p["key"].startswith("SoftGC")]
    dom = pareto_flags(points)
    front = sorted([p for p in points if not dom[p["key"]]], key=lambda p: p["rob"])
    ax.plot([p["rob"] for p in front], [p["dp"] for p in front], ls="--", lw=1.2,
            color="#888888", alpha=0.85, zorder=1, label="Pareto frontier")
    ax.plot([p["rob"] for p in soft], [p["dp"] for p in soft], "-", lw=1.6,
            color=COL["SoftGC"], alpha=0.75, zorder=2, label="Soft-GC path (alpha_F 0 -> 0.5)")
    for p in soft:
        ax.scatter(p["rob"], p["dp"], s=110, color=COL["SoftGC"], zorder=4,
                   edgecolors="black", linewidths=0.7)
        ax.annotate(f"aF={p['alpha_F']:.3f}", (p["rob"], p["dp"]),
                    textcoords="offset points", xytext=(7, -11), fontsize=8)
    for key, col, mk, size in (("Original", COL["Original"], "o", 130),
                               ("BestFixed", COL["BestFixed"], "s", 130),
                               ("HardGC", COL["HardGC"], "*", 340)):
        p = next(q for q in points if q["key"] == key)
        ax.scatter(p["rob"], p["dp"], s=size, color=col, marker=mk, zorder=6,
                   edgecolors="black", linewidths=1.2 if key == "HardGC" else 0.8, label=key)
    ax.set_xlabel("mean |ΔNormalScore_z|  (lower = better robustness)")
    ax.set_ylabel("mean defect d′  (higher = better preservation)")
    ax.set_title("Experiment 6A — Soft Geometry Gating: preservation/robustness plane\n"
                 "(fragile = bottle+grid @ alpha_F, tolerant = cable/hazelnut/screw @ 0.5)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "soft_gc_pareto.png", dpi=160)
    plt.close(fig)


def fig_vs_alpha(points: list, crit: dict) -> None:
    """两张独立图：preservation vs alpha_F、robustness vs alpha_F。"""
    soft = sorted([p for p in points if p["key"].startswith("SoftGC")], key=lambda p: p["alpha_F"])
    xs = [p["alpha_F"] for p in soft]
    fx = next(p for p in points if p["key"] == "BestFixed")
    hg = next(p for p in points if p["key"] == "HardGC")
    fig, ax = plt.subplots(figsize=(7.0, 4.8))
    ax.plot(xs, [p["dp"] for p in soft], marker="o", color=COL["SoftGC"], lw=2.0)
    for x, p in zip(xs, soft):
        ax.annotate(f"{p['dp']-fx['dp']:+.3f}", (x, p["dp"]), textcoords="offset points",
                    xytext=(0, 8), fontsize=8)
    ax.axhline(fx["dp"] + crit["A_preservation_retention"]["threshold"], color="gray", ls=":",
               lw=1.0, label="Criterion A threshold (+0.20 over Best Fixed)")
    ax.axhline(hg["dp"], color=COL["HardGC"], ls="--", lw=1.1, label="Hard GC d-prime")
    ax.set_xlabel("alpha_F (fragile end: bottle+grid)")
    ax.set_ylabel("mean defect d′ (5 categories x 3 seeds)")
    ax.set_title("Experiment 6A — preservation vs alpha_F")
    ax.grid(alpha=0.3); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(FIG_DIR / "preservation_vs_alpha.png", dpi=160); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.0, 4.8))
    ax.plot(xs, [p["rob"] for p in soft], marker="o", color=COL["SoftGC"], lw=2.0)
    for x, p in zip(xs, soft):
        ax.annotate(f"{p['rob']-hg['rob']:+.4f}", (x, p["rob"]), textcoords="offset points",
                    xytext=(0, 8), fontsize=8)
    ax.axhline(hg["rob"], color=COL["HardGC"], ls="--", lw=1.1, label="Hard GC |Δz|")
    ax.axhline(crit["B_robustness_recovery"]["threshold_abs_z_max"], color="gray", ls=":",
               lw=1.0, label="Criterion B ceiling (Hard GC − 0.02)")
    ax.axhline(fx["rob"], color=COL["BestFixed"], ls="-.", lw=1.1, label="Best Fixed |Δz|")
    ax.set_xlabel("alpha_F (fragile end: bottle+grid)")
    ax.set_ylabel("mean |ΔNormalScore_z| (lower = better)")
    ax.set_title("Experiment 6A — illumination robustness vs alpha_F")
    ax.grid(alpha=0.3); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(FIG_DIR / "robustness_vs_alpha.png", dpi=160); plt.close(fig)


def fig_category_response(points: list, cat: str, fname: str, crit: dict) -> None:
    soft = sorted([p for p in points if p["key"].startswith("SoftGC")], key=lambda p: p["alpha_F"])
    xs = [p["alpha_F"] for p in soft]
    dp = [p["per_cat"][cat]["mean_dprime"] for p in soft]
    rz = [p["per_cat"][cat]["mean_abs_delta_z"] for p in soft]
    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    ax.plot(xs, dp, marker="o", color="#4C72B0", lw=2.0, label=f"{cat}: defect d′ (left)")
    ax.set_xlabel("alpha_F"); ax.set_ylabel(f"{cat}: mean defect d′")
    ax.grid(alpha=0.3)
    ax2 = ax.twinx()
    ax2.plot(xs, rz, marker="s", color="#C44E52", lw=2.0, ls="--",
             label=f"{cat}: mean |Δz| (right)")
    ax2.set_ylabel(f"{cat}: mean |ΔNormalScore_z|")
    h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=8, loc="center right")
    extra = ("" if cat in FRAGILE else " (kept at alpha=0.5 for every policy -> flat by construction)")
    ax.set_title(f"Experiment 6A — per-category response: {cat}{extra}")
    fig.tight_layout()
    fig.savefig(FIG_DIR / fname, dpi=160)
    plt.close(fig)


def main() -> None:
    for d in (RAW_DIR, SUM_DIR, SAN_DIR, FIG_DIR):
        d.mkdir(parents=True, exist_ok=True)
    freeze = json.loads((REF_DIR / "policy_freeze.json").read_text())
    crit = freeze["success_criteria"]
    ref5d = freeze["reference_5d"]
    neg_hg = int(ref5d["hard_gc"]["negative_transfer_count"])
    grid = [float(x) for x in freeze["candidate_grid"]] + \
           [float(x) for x in freeze["pre_registered_conditional_extension"]["values"]]
    lookup, prov = e6.build_lookup(extra_roots=[str(OUT / "raw_new")])
    missing = e6.check_keys([float(x) for x in freeze["candidate_grid"]], lookup)
    if missing:
        print(f"[6A] STOP: still missing keys {missing}")
        raise SystemExit(3)
    print(f"[6A] lookup keys = {len(lookup)}; alpha_F grid = {grid}")

    ev_fx = eval_alpha_map(lookup, {c: e6.ALPHA_T for c in CATEGORIES})
    ev_orig = eval_alpha_map(lookup, {c: 0.0 for c in CATEGORIES})
    ev_hg = eval_alpha_map(lookup, soft_map(0.0))
    results = []
    for a in grid:
        ev = eval_alpha_map(lookup, soft_map(a))
        results.append({"alpha_F": a, "ev": ev,
                        "crit": criteria_eval(ev, ev_hg, ev_fx, crit, neg_hg)})
    vd = verdict_ladder(results, crit, neg_hg)

    # ---- raw / summary tables ----
    raw_rows, psum_rows, per_cat_rows, per_seed_rows = [], [], [], []
    for r in results:
        a, ev, cr = r["alpha_F"], r["ev"], r["crit"]
        label = ("HardGC" if abs(a) < 1e-12 else "BestFixed(alpha_F=0.5)" if abs(a - 0.5) < 1e-12
                 else f"SoftGC(aF={a:.8f})")
        raw_rows.append({"policy": label, "alpha_F": f"{a:.8f}", "mean_alpha": round(ev["mean_alpha"], 6),
                         "mean_dprime": round(ev["mean_dprime"], 6),
                         "mean_abs_delta_z": round(ev["mean_abs_delta_z"], 6),
                         **{k: cr[k] for k in ("delta_dprime_vs_fixed", "delta_abs_z_vs_fixed",
                                                "delta_dprime_vs_hardGC", "delta_abs_z_vs_hardGC",
                                                "worst_category", "worst_category_gain",
                                                "negative_transfer_count")},
                         "criterion_A": cr["A"], "criterion_B": cr["B"], "criterion_C": cr["C"],
                         "criterion_D": cr["D"], "A_and_B_and_C_and_D": cr["A_and_B_and_C_and_D"]})
        psum_rows.append({"policy": label, "alpha_F": f"{a:.8f}", "mean_alpha": round(ev["mean_alpha"], 6),
                          "mean_defect_dprime": round(ev["mean_dprime"], 6),
                          "mean_robustness": round(ev["mean_abs_delta_z"], 6),
                          **{k: cr[k] for k in ("delta_dprime_vs_fixed", "delta_abs_z_vs_fixed",
                                                 "delta_dprime_vs_hardGC", "delta_abs_z_vs_hardGC")},
                          "worst_category": cr["worst_category"],
                          "worst_category_gain": cr["worst_category_gain"],
                          "negative_transfer_count": cr["negative_transfer_count"],
                          "robustness_improvement_vs_hardGC": cr["robustness_improvement_vs_hardGC"],
                          "seeds_satisfying_A_and_B": cr["seeds_satisfying_A_and_B"],
                          "criterion_A": cr["A"], "criterion_B": cr["B"], "criterion_C": cr["C"],
                          "criterion_D": cr["D"], "A_and_B_and_C_and_D": cr["A_and_B_and_C_and_D"]})
        for c in CATEGORIES:
            per_cat_rows.append({"policy": label, "alpha_F": f"{a:.8f}", "category": c,
                                 "alpha": f"{ev['alpha_map'][c]:.8f}",
                                 "mean_dprime": round(ev["per_cat"][c]["mean_dprime"], 6),
                                 "mean_abs_delta_z": round(ev["per_cat"][c]["mean_abs_delta_z"], 6),
                                 "gain_vs_fixed": round(ev["per_cat"][c]["mean_dprime"]
                                                        - ev_fx["per_cat"][c]["mean_dprime"], 6)})
        for s in SEEDS:
            per_seed_rows.append({"policy": label, "alpha_F": f"{a:.8f}", "seed": s,
                                  "mean_dprime": round(seed_aggregate(ev, s, "mean_dprime"), 6),
                                  "mean_abs_delta_z": round(seed_aggregate(ev, s, "mean_abs_delta_z"), 6),
                                  "delta_dprime_vs_fixed": round(
                                      seed_aggregate(ev, s, "mean_dprime")
                                      - seed_aggregate(ev_fx, s, "mean_dprime"), 6),
                                  "delta_abs_z_vs_hardGC": round(
                                      seed_aggregate(ev, s, "mean_abs_delta_z")
                                      - seed_aggregate(ev_hg, s, "mean_abs_delta_z"), 6)})
    e6.write_csv(RAW_DIR / "policy_results.csv", raw_rows)
    e6.write_csv(RAW_DIR / "per_category.csv", per_cat_rows)
    e6.write_csv(RAW_DIR / "per_seed.csv", per_seed_rows)
    e6.write_csv(SUM_DIR / "policy_summary.csv", psum_rows)
    write_summary_extra(results, ev_orig, ev_fx, ev_hg, crit, vd, neg_hg, ref5d)
    run_figs(results, ev_orig, ev_fx, ev_hg, crit)
    report_final(freeze)
    print("[6A] figures: soft_gc_pareto / preservation_vs_alpha / robustness_vs_alpha / "
          "bottle_response / grid_response")


STATE: dict = {}


def write_summary_extra(results, ev_orig, ev_fx, ev_hg, crit, vd, neg_hg, ref5d) -> None:
    points = [{"key": "Original", "alpha_F": None, "rob": ev_orig["mean_abs_delta_z"],
               "dp": ev_orig["mean_dprime"], "mean_alpha": 0.0},
              {"key": "BestFixed", "alpha_F": 0.5, "rob": ev_fx["mean_abs_delta_z"],
               "dp": ev_fx["mean_dprime"], "mean_alpha": e6.ALPHA_T},
              {"key": "HardGC", "alpha_F": 0.0, "rob": ev_hg["mean_abs_delta_z"],
               "dp": ev_hg["mean_dprime"], "mean_alpha": 0.3}]
    for r in results:
        a = r["alpha_F"]
        if abs(a) < 1e-12 or abs(a - 0.5) < 1e-12:
            continue
        points.append({"key": f"SoftGC(aF={a:.8f})", "alpha_F": a,
                       "rob": r["ev"]["mean_abs_delta_z"], "dp": r["ev"]["mean_dprime"],
                       "mean_alpha": r["ev"]["mean_alpha"], "per_cat": r["ev"]["per_cat"]})
    dom = pareto_flags(points)
    e6.write_csv(SUM_DIR / "pareto_summary.csv",
                 [{"policy": p["key"], "alpha_F": "" if p["alpha_F"] is None else f"{p['alpha_F']:.8f}",
                   "mean_alpha": round(p["mean_alpha"], 6),
                   "mean_defect_dprime": round(p["dp"], 6),
                   "mean_robustness": round(p["rob"], 6),
                   "is_pareto_efficient": "true" if not dom[p["key"]] else "false",
                   "dominated_by": ";".join(dom[p["key"]])} for p in points])
    checks = e6.stage_sanity()
    npass = sum(1 for _, ok, _ in checks if ok)
    verdict = {
        "experiment": "6A",
        "title": "Geometry-Guided Soft Gating / Adaptive alpha Pilot",
        "freeze_sha256": (REF_DIR / "policy_freeze.sha256").read_text().split()[0],
        "git_head": e6.git_head(),
        "alpha_F_grid_evaluated": [f"{r['alpha_F']:.8f}" for r in results],
        "frozen_fragile": FRAGILE, "frozen_tolerant": TOLERANT, "tolerant_alpha": e6.ALPHA_T,
        "primary_control_baselines": {"hard_GC": ref5d["hard_gc"], "best_fixed": ref5d["best_fixed"]},
        "success_criteria_used": crit,
        "criteria_per_policy": vd["criteria_summary"],
        "verdict": vd,
        "sanity_passed": f"{npass}/{len(checks)}",
        "budget_disclosure": ("Soft GC mean alpha = (2*alpha_F+1.5)/5 > 0.30 for alpha_F>0, so soft policies "
                              "receive MORE normalization strength than Hard GC; policy-identity effect and "
                              "normalization-strength effect are NOT separable here. No strong causal claim."),
        "no_gpu_during_analysis": True,
    }
    (SUM_DIR / "verdict.json").write_text(json.dumps(verdict, indent=2))
    STATE.update({"points": points, "dom": dom, "checks": checks, "verdict": verdict,
                  "results": results, "ev_orig": ev_orig, "ev_fx": ev_fx, "ev_hg": ev_hg})


def run_figs(results, ev_orig, ev_fx, ev_hg, crit) -> None:
    points = STATE["points"]
    fig_soft_gc_pareto(points)
    fig_vs_alpha(points, crit)
    fig_category_response(points, "bottle", "bottle_response.png", crit)
    fig_category_response(points, "grid", "grid_response.png", crit)


def report_final(freeze: dict) -> None:
    L = 132
    ref5d = freeze["reference_5d"]
    hg_ref, fx_ref = ref5d["hard_gc"], ref5d["best_fixed"]
    ev_hg, ev_fx = STATE["ev_hg"], STATE["ev_fx"]
    print("=" * L)
    print("[6A RESULTS] Geometry-Guided Soft Gating / Adaptive alpha Pilot")
    print("=" * L)
    print(f"Hard GC equivalence vs 5D frozen : |dd-prime|={abs(ev_hg['mean_dprime']-hg_ref['mean_defect_dprime']):.2e}, "
          f"|d|dz||={abs(ev_hg['mean_abs_delta_z']-hg_ref['mean_robustness']):.2e}  "
          f"(<= 6-decimal rounding of the 5D summary CSV)")
    npass = sum(1 for _, ok, _ in STATE["checks"] if ok)
    print(f"sanity                           : {npass}/{len(STATE['checks'])} PASS")
    for c, ok, d in STATE["checks"]:
        if not ok:
            print(f"    NOT-PASS {c}: {d}")
    print("-" * L)
    hdr = (f"{'policy':<26}{'aF':>9}{'mean a':>8}{'d-prime':>10}{'|dz|':>9}"
           f"{'dd-fix':>9}{'dd-GC':>9}{'d|z|-fix':>10}{'d|z|-GC':>9}{'worst':>9}{'negTr':>7}  A B C D")
    print(hdr)
    for r in STATE["results"]:
        ev, cr = r["ev"], r["crit"]
        lbl = ("HardGC" if abs(r["alpha_F"]) < 1e-12 else
               "BestFixed(aF=0.5)" if abs(r["alpha_F"] - 0.5) < 1e-12 else
               f"SoftGC(aF={r['alpha_F']:.8f})")
        print(f"{lbl:<26}{r['alpha_F']:>9.5f}{ev['mean_alpha']:>8.4f}{ev['mean_dprime']:>10.4f}"
              f"{ev['mean_abs_delta_z']:>9.4f}{cr['delta_dprime_vs_fixed']:>+9.4f}"
              f"{cr['delta_dprime_vs_hardGC']:>+9.4f}{cr['delta_abs_z_vs_fixed']:>+10.4f}"
              f"{cr['delta_abs_z_vs_hardGC']:>+9.4f}{cr['worst_category_gain']:>+9.4f}"
              f"{cr['negative_transfer_count']:>7d}  "
              f"{int(cr['A'])} {int(cr['B'])} {int(cr['C'])} {int(cr['D'])}")
    print(f"{'Original(a=0 all)':<26}{'-':>9}{STATE['ev_orig']['mean_alpha']:>8.4f}"
          f"{STATE['ev_orig']['mean_dprime']:>10.4f}{STATE['ev_orig']['mean_abs_delta_z']:>9.4f}"
          f"{STATE['ev_orig']['mean_dprime']-ev_fx['mean_dprime']:>+9.4f}"
          f"{STATE['ev_orig']['mean_dprime']-ev_hg['mean_dprime']:>+9.4f}"
          f"{STATE['ev_orig']['mean_abs_delta_z']-ev_fx['mean_abs_delta_z']:>+10.4f}"
          f"{STATE['ev_orig']['mean_abs_delta_z']-ev_hg['mean_abs_delta_z']:>+9.4f}")
    print("-" * L)
    for p in STATE["points"]:
        print(f"  pareto {p['key']:<24} d-prime={p['dp']:.4f}  |dz|={p['rob']:.4f}  "
              f"mean_a={p['mean_alpha']:.4f}  dominated_by={STATE['dom'][p['key']] or '-'}")
    print("-" * L)
    print("per-category response along alpha_F (bottle / grid are the fragile end; others flat by construction)")
    for cat in CATEGORIES:
        vals = [(r["alpha_F"], round(r["ev"]["per_cat"][cat]["mean_dprime"], 4),
                 round(r["ev"]["per_cat"][cat]["mean_abs_delta_z"], 4)) for r in STATE["results"]]
        print(f"  {cat:<9} (aF, d-prime, |dz|) = {vals}")
    print("-" * L)
    print("seed stability (delta vs Hard GC / vs Fixed)")
    for r in STATE["results"]:
        if abs(r["alpha_F"]) < 1e-12 or abs(r["alpha_F"] - 0.5) < 1e-12:
            continue
        print(f"  aF={r['alpha_F']:.8f} seeds_AB={r['crit']['seeds_satisfying_A_and_B']}  " +
              "  ".join(f"s{d['seed']}:dd'={d['delta_dprime_vs_fixed']:+.3f},d|z|_GC={d['delta_abs_z_vs_hardGC']:+.4f}"
                        for d in r["crit"]["seed_detail"]))
    print("=" * L)
    vd = STATE["verdict"]["verdict"]
    print(f"FINAL VERDICT : {vd['case']} — {vd['meaning']}")
    print(f"  winning alpha_F = {vd['winning_alpha_F'] or 'none'}")
    print("  budget: mean alpha grows with alpha_F (Hard GC 0.30 -> Soft 0.40 -> Best Fixed 0.50);")
    print("          policy-identity effect and normalization-strength effect are NOT separable.")
    print("=" * L)


if __name__ == "__main__":
    main()
