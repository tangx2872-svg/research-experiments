"""Experiment 6B — Confirmatory Validation: analysis（CPU-only）。

回答两问：
  G1  α_F=0.25 是否为稳定 knee region（用预注册 A–D 与 consecutive 规则）
  G2  Soft-GC(0.25) 的优势是 selective allocation 还是仅 mean normalization strength
      （与 mean-α matched Uniform baseline 对照）

指标口径与判定阈值全部继承 6A 冻结实现（experiment6a_soft_geometry），本脚本不新增任何阈值。
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

import experiment6a_soft_geometry as e6a  # noqa: E402
import experiment6b_confirm as c6  # noqa: E402

OUT = c6.OUT
REF_DIR, RAW_DIR, SUM_DIR, SAN_DIR, FIG_DIR = c6.REF_DIR, c6.RAW_DIR, OUT / "summary", OUT / "sanity", OUT / "figures"
CATEGORIES, SEEDS, FRAGILE, ALPHA_T, EPS = e6a.CATEGORIES, e6a.SEEDS, e6a.FRAGILE, e6a.ALPHA_T, e6a.EPS

COL = {"Original": "#999999", "BestFixed": "#333333", "HardGC": "#2E8B57",
       "SoftGC": "#4C72B0", "Uniform": "#C44E52", "Knee": "#E8A33D"}
STATE: dict = {}


def knee_regions(alpha_sorted: list, ok_map: dict) -> list:
    """返回所有「连续满足 A&B&C&D」的区间 [(a_start, a_end, length)]，按列表相邻定义 consecutive。"""
    regions, cur = [], []
    for a in alpha_sorted:
        if ok_map.get(a, False):
            cur.append(a)
        else:
            if cur:
                regions.append((cur[0], cur[-1], len(cur)))
            cur = []
    if cur:
        regions.append((cur[0], cur[-1], len(cur)))
    return regions


def main() -> None:
    for d in (RAW_DIR, SUM_DIR, SAN_DIR, FIG_DIR):
        d.mkdir(parents=True, exist_ok=True)
    freeze = json.loads((REF_DIR / "policy_freeze.json").read_text())
    fr6a = json.loads(c6.REF_6A_FREEZE.read_text())
    crit = fr6a["success_criteria"]                       # 继承，不新增阈值
    ref5d = fr6a["reference_5d"]
    neg_hg = int(ref5d["hard_gc"]["negative_transfer_count"])
    grid = sorted(float(x) for x in freeze["evaluated_soft_grid"])
    uni_a = float(freeze["goal_2_mean_alpha_matched_uniform_baseline"]["uniform_alpha_chosen"])
    lookup = c6.build_lookup_all()
    missing = [f"{c}:s{s}:a{k}" for (c, s, k) in c6.all_required_keys(freeze) if (c, s, k) not in lookup]
    if missing:
        print(f"[6B] STOP: missing keys {missing}")
        raise SystemExit(3)

    ev_fx = c6.eval_alpha_map(lookup, {c: ALPHA_T for c in CATEGORIES})
    ev_orig = c6.eval_alpha_map(lookup, {c: 0.0 for c in CATEGORIES})
    ev_hg = c6.eval_alpha_map(lookup, c6.soft_map(0.0))
    ev_uni = c6.eval_alpha_map(lookup, {c: uni_a for c in CATEGORIES})
    results = []
    for a in grid:
        ev = c6.eval_alpha_map(lookup, c6.soft_map(a))
        results.append({"alpha_F": a, "ev": ev,
                        "crit": c6.criteria_eval(ev, ev_hg, ev_fx, crit, neg_hg)})
    ok_map = {r["alpha_F"]: bool(r["crit"]["A_and_B_and_C_and_D"]) for r in results}
    regions = knee_regions(grid, ok_map)
    in_region = [r for r in regions if r[0] - 1e-12 <= 0.25 <= r[1] + 1e-12]
    region_len = max([r[2] for r in in_region], default=0)

    soft25 = next(r for r in results if abs(r["alpha_F"] - 0.25) < 1e-12)
    d_dp_alloc = soft25["ev"]["mean_dprime"] - ev_uni["mean_dprime"]
    d_rz_alloc = soft25["ev"]["mean_abs_delta_z"] - ev_uni["mean_abs_delta_z"]
    d_dp_strength = ev_uni["mean_dprime"] - ev_fx["mean_dprime"]
    d_rz_strength = ev_uni["mean_abs_delta_z"] - ev_fx["mean_abs_delta_z"]
    uni_dom = (ev_uni["mean_dprime"] >= soft25["ev"]["mean_dprime"] - 1e-9
               and ev_uni["mean_abs_delta_z"] <= soft25["ev"]["mean_abs_delta_z"] + 1e-9
               and (ev_uni["mean_dprime"] > soft25["ev"]["mean_dprime"] + 1e-9
                    or ev_uni["mean_abs_delta_z"] < soft25["ev"]["mean_abs_delta_z"] - 1e-9))
    cond = {
        "CASE_A": {"region_len_ge_3": region_len >= 3,
                   "allocation_dprime_ge_eps": d_dp_alloc >= EPS,
                   "allocation_robustness_not_worse_than_0.02": d_rz_alloc <= 0.02},
        "CASE_B": {"region_len_ge_2": region_len >= 2,
                   "allocation_dprime_gt_minus_eps": d_dp_alloc > -EPS},
        "CASE_C": {"region_len_le_1": region_len <= 1},
        "CASE_D": {"uniform_pareto_dominates_soft25": bool(uni_dom)},
    }
    if cond["CASE_D"]["uniform_pareto_dominates_soft25"]:
        case, meaning = "CASE_D", "Uniform(mean-alpha matched) Pareto-dominates Soft-GC(0.25): the selective allocation is inferior to plain strength matching."
    elif all(cond["CASE_A"].values()):
        case, meaning = "CASE_A", "Soft gating confirmed: a >=3-point knee region exists AND the advantage over the mean-alpha matched Uniform baseline is attributable to selective allocation."
    elif all(cond["CASE_B"].values()):
        case, meaning = "CASE_B", "The knee region replicates (>=2 consecutive points) but the advantage over the matched Uniform baseline is not clearly allocation-driven (strength effect)."
    else:
        case, meaning = "CASE_C", "The 6A knee does not replicate: alpha_F=0.25 is an isolated point."
    pts, dom = write_tables(grid, results, ok_map, regions, ev_uni, ev_fx, ev_hg, ev_orig, soft25,
                            d_dp_alloc, d_rz_alloc, d_dp_strength, d_rz_strength, uni_a, crit, neg_hg)
    checks = c6.stage_sanity()
    v = verdict_json(case, meaning, cond, grid, ok_map, regions, region_len, soft25, ev_uni,
                     d_dp_alloc, d_rz_alloc, d_dp_strength, d_rz_strength, uni_a, crit, checks)
    make_figures(grid, results, ok_map, regions, ev_uni, ev_fx, ev_hg, ev_orig, soft25, crit, pts, dom)
    report(grid, results, ok_map, regions, region_len, ev_uni, ev_fx, ev_hg, ev_orig, soft25,
           d_dp_alloc, d_rz_alloc, d_dp_strength, d_rz_strength, cond, v, checks, uni_a, crit)


def pareto_flags(points: list, tol: float = 1e-9) -> dict:
    dom = {}
    for p in points:
        dom[p["key"]] = sorted(q["key"] for q in points
                               if q["key"] != p["key"] and q["dp"] >= p["dp"] - tol
                               and q["rob"] <= p["rob"] + tol
                               and (q["dp"] > p["dp"] + tol or q["rob"] < p["rob"] - tol))
    return dom


def row_of(label, aF, ev, cr, ev_fx, ev_hg):
    return {"policy": label, "alpha_F": "" if aF is None else f"{aF:.8f}",
            "mean_alpha": round(ev["mean_alpha"], 6),
            "mean_defect_dprime": round(ev["mean_dprime"], 6),
            "mean_robustness": round(ev["mean_abs_delta_z"], 6),
            "delta_dprime_vs_fixed": round(ev["mean_dprime"] - ev_fx["mean_dprime"], 6),
            "delta_abs_z_vs_fixed": round(ev["mean_abs_delta_z"] - ev_fx["mean_abs_delta_z"], 6),
            "delta_dprime_vs_hardGC": round(ev["mean_dprime"] - ev_hg["mean_dprime"], 6),
            "delta_abs_z_vs_hardGC": round(ev["mean_abs_delta_z"] - ev_hg["mean_abs_delta_z"], 6),
            "worst_category_gain": cr["worst_category_gain"],
            "negative_transfer_count": cr["negative_transfer_count"],
            "criterion_A": cr["A"], "criterion_B": cr["B"], "criterion_C": cr["C"], "criterion_D": cr["D"],
            "A_and_B_and_C_and_D": cr["A_and_B_and_C_and_D"],
            "seeds_satisfying_A_and_B": cr["seeds_satisfying_A_and_B"]}


def write_tables(grid, results, ok_map, regions, ev_uni, ev_fx, ev_hg, ev_orig, soft25,
                 d_dp_alloc, d_rz_alloc, d_dp_strength, d_rz_strength, uni_a, crit, neg_hg):
    rows = []
    for r in sorted(results, key=lambda x: x["alpha_F"]):
        a = r["alpha_F"]
        lbl = ("HardGC(aF=0)" if abs(a) < 1e-12 else "BestFixed(aF=0.5)" if abs(a - 0.5) < 1e-12
               else f"SoftGC(aF={a:.8f})")
        rows.append(row_of(lbl, a, r["ev"], r["crit"], ev_fx, ev_hg))
    for name, ev in (("Uniform(matched)", ev_uni), ("Original(a=0)", ev_orig)):
        rows.append(row_of(name, None, ev, c6.criteria_eval(ev, ev_hg, ev_fx, crit, neg_hg), ev_fx, ev_hg))
    e6a.write_csv(SUM_DIR / "policy_summary.csv", rows)
    e6a.write_csv(SUM_DIR / "knee_region.csv",
                  [{"alpha_F": f"{a:.8f}", "satisfies_A_and_B_and_C_and_D": ok_map[a],
                    "in_knee_region": any(x[0] - 1e-12 <= a <= x[1] + 1e-12 for x in regions)} for a in grid]
                  + [{"alpha_F": "SUMMARY", "satisfies_A_and_B_and_C_and_D": "",
                      "in_knee_region": f"regions={[(round(a, 4), round(b, 4), n) for a, b, n in regions]}"}])
    e6a.write_csv(SUM_DIR / "uniform_match.csv", [
        {"item": "uniform_alpha", "value": round(uni_a, 9), "note": "historical 5A C1 layer-uniform point"},
        {"item": "uniform_mean_alpha", "value": round(ev_uni["mean_alpha"], 6), "note": "matched to Soft 0.25 (0.40) within 0.001"},
        {"item": "uniform_mean_defect_dprime", "value": round(ev_uni["mean_dprime"], 6), "note": ""},
        {"item": "uniform_mean_robustness", "value": round(ev_uni["mean_abs_delta_z"], 6), "note": ""},
        {"item": "soft25_mean_defect_dprime", "value": round(soft25["ev"]["mean_dprime"], 6), "note": ""},
        {"item": "soft25_mean_robustness", "value": round(soft25["ev"]["mean_abs_delta_z"], 6), "note": ""},
        {"item": "allocation_effect_delta_dprime", "value": round(d_dp_alloc, 6),
         "note": "d-prime(Soft 0.25) - d-prime(Uniform): >0 = selective allocation adds preservation"},
        {"item": "allocation_effect_delta_abs_z", "value": round(d_rz_alloc, 6),
         "note": "|dz|(Soft 0.25) - |dz|(Uniform): >0 = Uniform is more robust"},
        {"item": "strength_effect_delta_dprime", "value": round(d_dp_strength, 6), "note": "Uniform - Best Fixed"},
        {"item": "strength_effect_delta_abs_z", "value": round(d_rz_strength, 6), "note": "Uniform - Best Fixed"},
    ])
    pts = ([{"key": f"SoftGC(aF={r['alpha_F']:.8f})", "alpha_F": r["alpha_F"],
             "rob": r["ev"]["mean_abs_delta_z"], "dp": r["ev"]["mean_dprime"],
             "mean_alpha": r["ev"]["mean_alpha"]} for r in results]
           + [{"key": "Uniform(matched)", "alpha_F": None, "rob": ev_uni["mean_abs_delta_z"],
               "dp": ev_uni["mean_dprime"], "mean_alpha": ev_uni["mean_alpha"]},
              {"key": "Original(a=0)", "alpha_F": None, "rob": ev_orig["mean_abs_delta_z"],
               "dp": ev_orig["mean_dprime"], "mean_alpha": 0.0}])
    dom = pareto_flags(pts)
    e6a.write_csv(SUM_DIR / "pareto_summary.csv",
                  [{"policy": p["key"], "alpha_F": "" if p["alpha_F"] is None else f"{p['alpha_F']:.8f}",
                    "mean_alpha": round(p["mean_alpha"], 6), "mean_defect_dprime": round(p["dp"], 6),
                    "mean_robustness": round(p["rob"], 6),
                    "is_pareto_efficient": "true" if not dom[p["key"]] else "false",
                    "dominated_by": ";".join(dom[p["key"]])} for p in pts])
    return pts, dom


def verdict_json(case, meaning, cond, grid, ok_map, regions, region_len, soft25, ev_uni,
                 d_dp_alloc, d_rz_alloc, d_dp_strength, d_rz_strength, uni_a, crit, checks):
    v = {
        "experiment": "6B",
        "title": "Confirmatory Validation of Soft Geometry Gating",
        "freeze_sha256": (REF_DIR / "policy_freeze.sha256").read_text().split()[0],
        "git_head": c6.git_head(),
        "inherited_criteria_unchanged": crit,
        "evaluated_soft_grid": [f"{a:.8f}" for a in grid],
        "knee_region_definition": "consecutive alpha_F satisfying A&B&C&D (adjacent in the sorted grid)",
        "knee_regions": [{"start": f"{s:.8f}", "end": f"{e:.8f}", "length": n} for s, e, n in regions],
        "region_containing_0.25_length": region_len,
        "per_alpha_ABCD": {f"{a:.8f}": bool(ok_map[a]) for a in grid},
        "soft25": {"alpha_F": 0.25, "mean_defect_dprime": round(soft25["ev"]["mean_dprime"], 6),
                   "mean_abs_delta_z": round(soft25["ev"]["mean_abs_delta_z"], 6),
                   "criteria": {k: soft25["crit"][k] for k in ("A", "B", "C", "D")}},
        "uniform_baseline": {"alpha": round(uni_a, 9), "mean_alpha": round(ev_uni["mean_alpha"], 6),
                             "mean_defect_dprime": round(ev_uni["mean_dprime"], 6),
                             "mean_abs_delta_z": round(ev_uni["mean_abs_delta_z"], 6)},
        "allocation_vs_strength": {
            "allocation_delta_dprime": round(d_dp_alloc, 6),
            "allocation_delta_abs_z": round(d_rz_alloc, 6),
            "strength_delta_dprime": round(d_dp_strength, 6),
            "strength_delta_abs_z": round(d_rz_strength, 6),
            "reading": "allocation_delta_dprime > 0 means selective allocation adds preservation beyond mean-strength matching"},
        "verdict": {"case": case, "meaning": meaning, "conditions": cond},
        "sanity_passed": f"{sum(1 for _, ok, _ in checks if ok)}/{len(checks)}",
        "limitations": [
            "n_category=5, gate_size=2, 7 soft points only (no dense sweep by protocol)",
            "Uniform baseline = closest historical layer-uniform alpha (0.40091275); mean-alpha gap 0.000913",
            "single backbone (wide_resnet50_2); synthetic photometric illumination protocol",
            "descriptive only: no significance test, no causal claim beyond the matched-strength control",
        ],
        "no_gpu_during_analysis": True,
    }
    (SUM_DIR / "verdict.json").write_text(json.dumps(v, indent=2))
    return v


def make_figures(grid, results, ok_map, regions, ev_uni, ev_fx, ev_hg, ev_orig, soft25, crit, pts, dom):
    fig, ax = plt.subplots(figsize=(8.6, 5.8))
    for r in results:
        a, ev = r["alpha_F"], r["ev"]
        in_r = any(x[0] - 1e-12 <= a <= x[1] + 1e-12 for x in regions)
        col = COL["Knee"] if in_r else COL["SoftGC"]
        ax.scatter(ev["mean_abs_delta_z"], ev["mean_dprime"], s=170 if in_r else 110, color=col,
                   marker="o", edgecolors="black", linewidths=1.1 if in_r else 0.7, zorder=5)
        ax.annotate(f"{a:.3f}", (ev["mean_abs_delta_z"], ev["mean_dprime"]),
                    textcoords="offset points", xytext=(7, -10), fontsize=8)
    srt = sorted(results, key=lambda x: x["alpha_F"])
    ax.plot([r["ev"]["mean_abs_delta_z"] for r in srt], [r["ev"]["mean_dprime"] for r in srt],
            ls="-", lw=1.4, color=COL["SoftGC"], alpha=0.7, zorder=2, label="Soft-GC path")
    for key, ev, col, mk, sz in (("Original a=0", ev_orig, COL["Original"], "o", 140),
                                 ("Best Fixed 0.5", ev_fx, COL["BestFixed"], "s", 140),
                                 ("Hard GC (aF=0)", ev_hg, COL["HardGC"], "*", 340),
                                 ("Uniform(matched)", ev_uni, COL["Uniform"], "D", 190)):
        ax.scatter(ev["mean_abs_delta_z"], ev["mean_dprime"], s=sz, color=col, marker=mk, zorder=6,
                   edgecolors="black", linewidths=1.2, label=key)
    ax.set_xlabel("mean |ΔNormalScore_z|  (lower = better robustness)")
    ax.set_ylabel("mean defect d′  (higher = better preservation)")
    ax.set_title("Experiment 6B — confirmatory Pareto plane\n(orange = alpha_F inside the pre-registered knee region)")
    ax.grid(alpha=0.3); ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout(); fig.savefig(FIG_DIR / "confirm_pareto.png", dpi=160); plt.close(fig)

    xs = [r["alpha_F"] for r in srt]
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 4.8))
    for x0, x1, _ in regions:
        axes[0].axvspan(x0 - 0.012, x1 + 0.012, color=COL["Knee"], alpha=0.22)
        axes[1].axvspan(x0 - 0.012, x1 + 0.012, color=COL["Knee"], alpha=0.22)
    axes[0].plot(xs, [r["ev"]["mean_dprime"] - ev_fx["mean_dprime"] for r in srt], marker="o",
                 color=COL["SoftGC"], lw=2.0)
    axes[0].axhline(crit["A_preservation_retention"]["threshold"], color="gray", ls=":", lw=1.1,
                    label="Criterion A threshold +0.20")
    axes[0].set_xlabel("alpha_F"); axes[0].set_ylabel("Δ defect d′ vs Best Fixed")
    axes[0].set_title("Criterion A across alpha_F"); axes[0].legend(fontsize=8); axes[0].grid(alpha=0.3)
    axes[1].plot(xs, [r["ev"]["mean_abs_delta_z"] for r in srt], marker="s", color=COL["Uniform"], lw=2.0)
    axes[1].axhline(crit["B_robustness_recovery"]["threshold_abs_z_max"], color="gray", ls=":", lw=1.1,
                    label="Criterion B ceiling (HardGC - 0.02)")
    axes[1].axhline(ev_hg["mean_abs_delta_z"], color=COL["HardGC"], ls="--", lw=1.1, label="Hard GC |Δz|")
    axes[1].set_xlabel("alpha_F"); axes[1].set_ylabel("mean |ΔNormalScore_z|")
    axes[1].set_title("Criterion B across alpha_F"); axes[1].legend(fontsize=8); axes[1].grid(alpha=0.3)
    fig.suptitle("Experiment 6B — knee region confirmation (shaded = consecutive A&B&C&D region)", y=1.02)
    fig.tight_layout(); fig.savefig(FIG_DIR / "knee_region.png", dpi=160, bbox_inches="tight"); plt.close(fig)

    labels = ["Hard GC\naF=0", "Soft-GC\naF=0.25", "Uniform\n0.4009", "Best Fixed\n0.5"]
    dpv = [ev_hg["mean_dprime"], soft25["ev"]["mean_dprime"], ev_uni["mean_dprime"], ev_fx["mean_dprime"]]
    rzv = [ev_hg["mean_abs_delta_z"], soft25["ev"]["mean_abs_delta_z"], ev_uni["mean_abs_delta_z"],
           ev_fx["mean_abs_delta_z"]]
    cols = [COL["HardGC"], COL["SoftGC"], COL["Uniform"], COL["BestFixed"]]
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.6))
    axes[0].bar(labels, dpv, color=cols, edgecolor="black", linewidth=0.6)
    axes[0].set_ylabel("mean defect d′"); axes[0].set_title("preservation: Soft 0.25 vs mean-α matched Uniform")
    axes[1].bar(labels, rzv, color=cols, edgecolor="black", linewidth=0.6)
    axes[1].set_ylabel("mean |ΔNormalScore_z|"); axes[1].set_title("robustness (lower = better)")
    for ax in axes:
        ax.grid(alpha=0.3, axis="y")
    fig.suptitle("Experiment 6B — selective allocation vs mean-strength matching (same mean α ~ 0.40)", y=1.02)
    fig.tight_layout(); fig.savefig(FIG_DIR / "soft_vs_uniform.png", dpi=160, bbox_inches="tight"); plt.close(fig)

    dp_s = ev_uni["mean_dprime"] - ev_fx["mean_dprime"]
    dp_a = soft25["ev"]["mean_dprime"] - ev_uni["mean_dprime"]
    dp_t = soft25["ev"]["mean_dprime"] - ev_fx["mean_dprime"]
    rz_s = ev_uni["mean_abs_delta_z"] - ev_fx["mean_abs_delta_z"]
    rz_a = soft25["ev"]["mean_abs_delta_z"] - ev_uni["mean_abs_delta_z"]
    rz_t = soft25["ev"]["mean_abs_delta_z"] - ev_fx["mean_abs_delta_z"]
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.6))
    for ax, (s_, a_, t_, ttl, unit) in ((axes[0], (dp_s, dp_a, dp_t, "Δ defect d′ vs Best Fixed", "Δd′")),
                                        (axes[1], (rz_s, rz_a, rz_t, "Δ |ΔNormalScore_z| vs Best Fixed", "Δ|Δz|"))):
        ax.bar(["strength\n(0.5→0.4009)", "allocation\n(Soft-Uniform)", "total\n(Soft 0.25)"],
               [s_, a_, t_], color=["#8172B3", COL["SoftGC"], "#555555"], edgecolor="black", linewidth=0.6)
        for i, v in enumerate([s_, a_, t_]):
            ax.annotate(f"{v:+.4f}", (i, v), textcoords="offset points", xytext=(0, 6 if v >= 0 else -14),
                        ha="center", fontsize=9)
        ax.axhline(0, color="k", lw=0.9)
        ax.set_title(ttl); ax.set_ylabel(unit); ax.grid(alpha=0.3, axis="y")
    fig.suptitle("Experiment 6B — decomposition: mean-strength effect vs selective-allocation effect", y=1.02)
    fig.tight_layout(); fig.savefig(FIG_DIR / "allocation_vs_strength.png", dpi=160, bbox_inches="tight"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(10.4, 4.8))
    x = np.arange(len(CATEGORIES)); w = 0.26
    for i, (nm, ev, col) in enumerate((("Soft-GC aF=0.25", soft25["ev"], COL["SoftGC"]),
                                       ("Uniform 0.4009", ev_uni, COL["Uniform"]),
                                       ("Hard GC", ev_hg, COL["HardGC"]))):
        vals = [ev["per_cat"][c]["mean_dprime"] - ev_fx["per_cat"][c]["mean_dprime"] for c in CATEGORIES]
        ax.bar(x + (i - 1) * w, vals, w, label=nm, color=col, edgecolor="black", linewidth=0.6)
    ax.axhline(0, color="k", lw=0.9)
    ax.axhline(EPS, color="gray", ls=":", lw=1.0); ax.axhline(-EPS, color="gray", ls=":", lw=1.0)
    ax.set_xticks(x); ax.set_xticklabels(CATEGORIES)
    ax.set_ylabel("per-category Δ defect d′ vs Best Fixed")
    ax.set_title("Experiment 6B — per-category confirmation (same identity; only fragile-end α differs)")
    ax.grid(alpha=0.3, axis="y"); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(FIG_DIR / "per_category_confirm.png", dpi=160); plt.close(fig)
    print("[6B] figures: confirm_pareto / knee_region / soft_vs_uniform / allocation_vs_strength / "
          "per_category_confirm")


def report(grid, results, ok_map, regions, region_len, ev_uni, ev_fx, ev_hg, ev_orig, soft25,
           d_dp_alloc, d_rz_alloc, d_dp_strength, d_rz_strength, cond, v, checks, uni_a, crit):
    L = 126
    print("=" * L)
    print("[6B RESULTS] Confirmatory Validation of Soft Geometry Gating")
    print("=" * L)
    print(f"sanity : {sum(1 for _, ok, _ in checks if ok)}/{len(checks)} PASS")
    for c, ok, d in checks:
        if not ok:
            print(f"    NOT-PASS {c}: {d}")
    print("-" * L)
    print(f"{'policy':<24}{'aF':>9}{'mean a':>8}{'d-prime':>10}{'|dz|':>9}{'dd-fix':>9}"
          f"{'d|z|-fix':>10}{'dd-GC':>9}{'worst':>9}{'negTr':>7}   A B C D  region")
    for r in sorted(results, key=lambda x: x["alpha_F"]):
        a, ev, cr = r["alpha_F"], r["ev"], r["crit"]
        lbl = ("HardGC" if abs(a) < 1e-12 else "BestFixed(aF=0.5)" if abs(a - 0.5) < 1e-12
               else f"SoftGC(aF={a:.6f})")
        inr = any(x[0] - 1e-12 <= a <= x[1] + 1e-12 for x in regions)
        print(f"{lbl:<24}{a:>9.5f}{ev['mean_alpha']:>8.4f}{ev['mean_dprime']:>10.4f}"
              f"{ev['mean_abs_delta_z']:>9.4f}{cr['delta_dprime_vs_fixed']:>+9.4f}"
              f"{cr['delta_abs_z_vs_fixed']:>+10.4f}{cr['delta_dprime_vs_hardGC']:>+9.4f}"
              f"{cr['worst_category_gain']:>+9.4f}{cr['negative_transfer_count']:>7d}   "
              f"{int(cr['A'])} {int(cr['B'])} {int(cr['C'])} {int(cr['D'])}   {'YES' if inr else '-'}")
    print(f"{'Uniform(matched)':<24}{uni_a:>9.5f}{ev_uni['mean_alpha']:>8.4f}{ev_uni['mean_dprime']:>10.4f}"
          f"{ev_uni['mean_abs_delta_z']:>9.4f}{ev_uni['mean_dprime']-ev_fx['mean_dprime']:>+9.4f}"
          f"{ev_uni['mean_abs_delta_z']-ev_fx['mean_abs_delta_z']:>+10.4f}"
          f"{ev_uni['mean_dprime']-ev_hg['mean_dprime']:>+9.4f}")
    print(f"{'Original(a=0 all)':<24}{'-':>9}{0.0:>8.4f}{ev_orig['mean_dprime']:>10.4f}"
          f"{ev_orig['mean_abs_delta_z']:>9.4f}{ev_orig['mean_dprime']-ev_fx['mean_dprime']:>+9.4f}"
          f"{ev_orig['mean_abs_delta_z']-ev_fx['mean_abs_delta_z']:>+10.4f}"
          f"{ev_orig['mean_dprime']-ev_hg['mean_dprime']:>+9.4f}")
    print("-" * L)
    print(f"knee regions (consecutive A&B&C&D) : "
          f"{[(round(a, 4), round(b, 4), n) for a, b, n in regions]}")
    print(f"region containing alpha_F=0.25      : length = {region_len}")
    print("-" * L)
    print("G2 — mean-alpha matched control (Uniform alpha=%.8f, mean alpha=%.6f):" % (uni_a, ev_uni["mean_alpha"]))
    print(f"    allocation effect : dd-prime(Soft-Uniform) = {d_dp_alloc:+.4f}   "
          f"d|dz|(Soft-Uniform) = {d_rz_alloc:+.4f}")
    print(f"    strength effect   : dd-prime(Uniform-BestFixed) = {d_dp_strength:+.4f}   "
          f"d|dz| = {d_rz_strength:+.4f}")
    print(f"    Soft-GC(0.25) total vs Best Fixed : dd-prime = "
          f"{soft25['crit']['delta_dprime_vs_fixed']:+.4f}, d|dz| = {soft25['crit']['delta_abs_z_vs_fixed']:+.4f}")
    print("-" * L)
    print("Q1  0.20-0.30 是否形成稳定 knee region : "
          f"{'YES' if region_len >= 3 else 'PARTIAL (2 points)' if region_len == 2 else 'NO'} "
          f"(region length = {region_len})")
    print("Q2  alpha_F=0.25 是否为孤立点        : "
          f"{'NO - it lies in a region' if region_len >= 2 else 'YES - isolated'}")
    print("Q3  Soft-GC 是否优于 matched Uniform : "
          f"dd-prime {d_dp_alloc:+.4f} ({'Soft better' if d_dp_alloc >= EPS else 'not beyond eps' if abs(d_dp_alloc) < EPS else 'Uniform better'}), "
          f"d|dz| {d_rz_alloc:+.4f} ({'Soft worse' if d_rz_alloc > 0 else 'Soft better'})")
    print("Q4  优势来自 allocation 还是 strength : "
          f"allocation {d_dp_alloc:+.4f} vs strength {d_dp_strength:+.4f} "
          f"(of total {soft25['crit']['delta_dprime_vs_fixed']:+.4f})")
    print("=" * L)
    print(f"FINAL VERDICT : {v['verdict']['case']} — {v['verdict']['meaning']}")
    print(f"  conditions: {json.dumps(cond, ensure_ascii=False)}")
    print("=" * L)


if __name__ == "__main__":
    main()
