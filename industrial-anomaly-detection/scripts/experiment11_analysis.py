"""Experiment 11 — 分析（CPU-only；复用 Exp10 的 metrics 实现与 Exp10 的参考点 raw）。

判定：PASS = (Δ|Δz| <= -EPS_RZ) and (Δd' >= -EPS_DP)（相对同 (category,seed) 的 corrected Original-189）
       catastrophic = Δd' <= -0.25
参考点（Exp10 已有，0 GPU）：P10_ORIG_a000 / P10_C2_L2resid025_L3a025 / P10_T1_L2a000_L3a025
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import experiment10_analysis as A10  # noqa: E402
import experiment11_candidates as c11  # noqa: E402

E11 = ROOT / "results" / "experiment_11_c2_adaptive"
E10 = ROOT / "results" / "experiment_10_preservation_recovery"
ROOTS = [E11 / "raw", E10 / "raw"]
ANA, FIG, STATS = E11 / "analysis", E11 / "figures", E11 / "stats"
CATS, SEEDS = c11.CATS, [0, 1, 2]
EPS_RZ, EPS_DP, CATA = 0.02, 0.10, -0.25
ORIG, C2F, T1 = "P10_ORIG_a000", "P10_C2_L2resid025_L3a025", "P10_T1_L2a000_L3a025"


def wcsv(p, rows):
    if not rows:
        return
    keys = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader(); w.writerows(rows)


def M(cat, seed, cfg):
    m, d = A10.metrics(cat, seed, cfg, ROOTS)
    return m, d


def refs(cat, seed):
    o, _ = M(cat, seed, ORIG)
    c, _ = M(cat, seed, C2F)
    t, _ = M(cat, seed, T1)
    return {"orig": o, "c2": c, "t1": t}


def eval_unit(cat, seed, cfg):
    m, d = M(cat, seed, cfg)
    r = refs(cat, seed)
    if m is None or r["orig"] is None:
        return None
    dR = m["mean_abs_delta_z"] - r["orig"]["mean_abs_delta_z"]
    dP = m["mean_dprime"] - r["orig"]["mean_dprime"]
    out = {"category": cat, "seed": seed, "cfg": cfg,
           "absdz": m["mean_abs_delta_z"], "dprime": m["mean_dprime"],
           "orig_absdz": r["orig"]["mean_abs_delta_z"], "orig_dprime": r["orig"]["mean_dprime"],
           "dR": dR, "dP": dP,
           "pass": bool(dR <= -EPS_RZ and dP >= -EPS_DP),
           "catastrophic": bool(dP <= CATA),
           "pixel_auroc": m["pixel_auroc"], "aupro": m["aupro"], "tau": m["tau_val"]}
    if r["c2"] is not None:
        out["dR_vs_c2fixed"] = m["mean_abs_delta_z"] - r["c2"]["mean_abs_delta_z"]
        out["dP_vs_c2fixed"] = m["mean_dprime"] - r["c2"]["mean_dprime"]
    return out


def candidate_units(cid, cats, seeds, pattern: str = "{cid}__{cat}"):
    return [eval_unit(cat, s, pattern.format(cid=cid, cat=cat)) for cat in cats for s in seeds]


def summarize(cid, cats, seeds, pattern: str = "{cid}__{cat}"):
    rows = [r for r in candidate_units(cid, cats, seeds, pattern) if r]
    if not rows:
        return None
    per_cat = {}
    for c in cats:
        rs = [r for r in rows if r["category"] == c]
        if rs:
            per_cat[c] = {"n": len(rs), "pass": sum(r["pass"] for r in rs),
                          "cata": sum(r["catastrophic"] for r in rs),
                          "mean_dR": float(np.mean([r["dR"] for r in rs])),
                          "mean_dP": float(np.mean([r["dP"] for r in rs])),
                          "worst_dP": float(min(r["dP"] for r in rs))}
    return {"candidate": cid, "n_units": len(rows), "n_pass": sum(r["pass"] for r in rows),
            "n_cata": sum(r["catastrophic"] for r in rows),
            "worst_dP": float(min(r["dP"] for r in rows)),
            "mean_dR": float(np.mean([r["dR"] for r in rows])),
            "mean_dP": float(np.mean([r["dP"] for r in rows])),
            "per_cat": per_cat, "units": rows}


SIMPLE = {c: i for i, c in enumerate(c11._CAND_IDS)}
HARD_CATS = ["bottle", "cable", "hazelnut"]
# --- B1/B2 晋级阈值（在 11B-1 仅完成 2/18 units、未查看任何候选数值时写定）---
B1_WEAK_GAIN = 0.10      # cable/hazelnut 相对 C2-fixed 的 preservation 明显改善门槛
B2_BOTTLE_SCREW_TOL = -0.10   # bottle/screw 相对 C2-fixed 的允许退化为一个 EPS_DP


def stage_b1(ids=None) -> dict:
    ids = ids or c11._CAND_IDS
    rows = []
    for cid in ids:
        s = summarize(cid, HARD_CATS, [0])
        if s is None:
            continue
        s["hard_pass"] = sum(1 for c in HARD_CATS if s["per_cat"].get(c, {}).get("pass"))
        s["weak_gain"] = max([s["per_cat"].get(c, {}).get("mean_dP_vs_c2", -9)] for c in ("cable", "hazelnut")
                             if s["per_cat"].get(c)) if False else None
        # 弱类相对 C2-fixed 的 preservation 改善
        gains = {}
        for c in ("cable", "hazelnut"):
            rs = [r for r in s["units"] if r["category"] == c and "dP_vs_c2fixed" in r]
            gains[c] = float(np.mean([r["dP_vs_c2fixed"] for r in rs])) if rs else float("nan")
        s["weak_gain"] = gains
        bottle_ok = bool(s["per_cat"].get("bottle", {}).get("pass")) if s["per_cat"].get("bottle") else False
        weak_ok = any(gains[c] >= B1_WEAK_GAIN for c in ("cable", "hazelnut") if np.isfinite(gains[c]))
        # robustness 不退回 Original：弱类 ΔR <= -EPS_RZ
        weak_r_ok = any(s["per_cat"][c]["mean_dR"] <= -EPS_RZ for c in ("cable", "hazelnut")
                        if c in s["per_cat"])
        s.update({"bottle_not_broken": bottle_ok, "weak_improved": weak_ok,
                  "weak_robust_kept": weak_r_ok,
                  "B1_advance": bool(bottle_ok and weak_ok and weak_r_ok)})
        rows.append(s)
    rows.sort(key=lambda r: (-r["hard_pass"], r["n_cata"], -r["worst_dP"], r["mean_dR"],
                             SIMPLE.get(r["candidate"], 9)))
    for i, r in enumerate(rows, 1):
        r["rank"] = i
    # Top-3 选择**按用户规定的排名优先级**（PASS 数 -> cata -> worst dP -> mean dR -> 简洁度），
    # 而不是额外过滤器（B1_advance 仅作描述性标注）。
    adv = [r["candidate"] for r in rows][:3]
    det = {"candidates": [r for r in rows], "advanced_top3": adv,
           "thresholds": {"B1_WEAK_GAIN_vs_c2fixed": B1_WEAK_GAIN, "weak_robust_kept": "dR <= -EPS_RZ",
                          "note": "阈值在 11B-1 完成 2/18 units 时写定，未查看候选数值"}}
    out_rows = [{k: v for k, v in r.items() if k not in ("units", "per_cat", "weak_gain")} |
                {"cable_gain_vs_c2": r["weak_gain"]["cable"], "hazelnut_gain_vs_c2": r["weak_gain"]["hazelnut"]}
                for r in rows]
    wcsv(E11 / "analysis" / "b1_hard_category.csv", out_rows)
    (E11 / "analysis" / "b1_hard_category.json").write_text(json.dumps(det, indent=2, ensure_ascii=False, default=str))
    return det


def stage_b2(ids) -> dict:
    """Top3 × 5 categories × seed0 → 5-category matrix；晋级需 >=4/5 PASS、cable 无 catastrophic、
    bottle/screw 相对 C2-fixed 退化不超过一个 EPS_DP。"""
    rows, det = [], {"candidates": []}
    for cid in ids:
        s = summarize(cid, CATS, [0])
        if s is None:
            continue
        n_pass = s["n_pass"]
        bottle_ok = s["per_cat"].get("bottle", {}).get("pass", False)
        screw_ok = s["per_cat"].get("screw", {}).get("pass", False)
        cable_cata = s["per_cat"].get("cable", {}).get("cata", 0) > 0
        deg = {}
        for c in ("bottle", "screw"):
            rs = [r for r in s["units"] if r["category"] == c and "dP_vs_c2fixed" in r]
            deg[c] = float(np.mean([r["dP_vs_c2fixed"] for r in rs])) if rs else float("nan")
        no_deg = all((not np.isfinite(deg[c])) or deg[c] >= B2_BOTTLE_SCREW_TOL for c in ("bottle", "screw"))
        adv = bool(n_pass >= 4 and not cable_cata and bottle_ok and screw_ok and no_deg)
        s.update({"B2_advance": adv, "cable_catastrophic": cable_cata,
                  "deg_vs_c2_bottle": deg.get("bottle"), "deg_vs_c2_screw": deg.get("screw")})
        det["candidates"].append(s)
    det["candidates"].sort(key=lambda r: (-r["n_pass"], r["n_cata"], -r["worst_dP"], r["mean_dR"]))
    top2 = [r["candidate"] for r in det["candidates"] if r["B2_advance"]][:2]
    det["advanced_top2"] = top2
    det["thresholds"] = {"n_pass_min": 4, "cable_no_catastrophic": True,
                         "bottle_screw_deg_tol_vs_c2": B2_BOTTLE_SCREW_TOL}
    for r in det["candidates"]:
        rows.append({"candidate": r["candidate"], "n_pass_5cat": r["n_pass"], "n_cata": r["n_cata"],
                     **{f"{c}_dR": r["per_cat"].get(c, {}).get("mean_dR") for c in CATS},
                     **{f"{c}_dP": r["per_cat"].get(c, {}).get("mean_dP") for c in CATS},
                     "worst_dP": r["worst_dP"], "B2_advance": r["B2_advance"]})
    wcsv(E11 / "analysis" / "b2_five_category.csv", rows)
    (E11 / "analysis" / "b2_five_category.json").write_text(json.dumps(det, indent=2, ensure_ascii=False, default=str))
    return det


def stage_c(ids) -> dict:
    """Top1-2 × 5 categories × seeds{0,1,2}（seed0 复用 B2）→ PASS/15 与冻结成功判据。"""
    det = {"candidates": [], "criteria": {
        "strong_advance": "PASS>=12/15 且 bottle>=3/3, screw>=2/3, grid>=2/3, hazelnut>=2/3, cable>=2/3, 0 catastrophic",
        "final_method_candidate": "PASS>=13/15 且 cable>=2/3 且 bottle 3/3"}}
    for cid in ids:
        s = summarize(cid, CATS, SEEDS)
        if s is None:
            continue
        per_cat_pass = {c: s["per_cat"].get(c, {}).get("pass", 0) for c in CATS}
        n_cata = s["n_cata"]
        strong = bool(s["n_pass"] >= 12 and per_cat_pass.get("bottle", 0) >= 3
                      and per_cat_pass.get("screw", 0) >= 2 and per_cat_pass.get("grid", 0) >= 2
                      and per_cat_pass.get("hazelnut", 0) >= 2 and per_cat_pass.get("cable", 0) >= 2
                      and n_cata == 0)
        final = bool(s["n_pass"] >= 13 and per_cat_pass.get("cable", 0) >= 2
                     and per_cat_pass.get("bottle", 0) >= 3)
        s.update({"per_cat_pass": per_cat_pass, "strong_advance": strong,
                  "final_method_candidate": final})
        det["candidates"].append(s)
    det["candidates"].sort(key=lambda r: (-r["n_pass"], r["n_cata"], -r["worst_dP"]))
    det["n_strong_advance"] = sum(1 for c in det["candidates"] if c["strong_advance"])
    det["n_final_candidate"] = sum(1 for c in det["candidates"] if c["final_method_candidate"])
    rows = []
    for r in det["candidates"]:
        rows.append({"candidate": r["candidate"], "PASS_15": r["n_pass"], "n_cata": r["n_cata"],
                     **{f"{c}_pass": r["per_cat_pass"].get(c) for c in CATS},
                     "mean_dR": r["mean_dR"], "mean_dP": r["mean_dP"], "worst_dP": r["worst_dP"],
                     "strong_advance": r["strong_advance"],
                     "final_method_candidate": r["final_method_candidate"]})
    wcsv(E11 / "analysis" / "c_multiseed.csv", rows)
    (E11 / "analysis" / "c_multiseed.json").write_text(json.dumps(det, indent=2, ensure_ascii=False, default=str))
    return det


def c2fixed_baseline_table() -> dict:
    """C2-fixed（Exp10）在同一判定下的 PASS/15，用于最终对照。"""
    s = summarize(C2F, CATS, SEEDS, pattern="{cid}")   # Exp10 参考点命名无 __cat 后缀
    return {"candidate": "C2_fixed(Exp10)", "n_pass": s["n_pass"], "n_cata": s["n_cata"],
            "per_cat_pass": {c: s["per_cat"].get(c, {}).get("pass", 0) for c in CATS},
            "worst_dP": s["worst_dP"], "mean_dP": s["mean_dP"]}


def _fig1(b1: dict) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 5.0), sharex=True, sharey=True)
    for ax, cat in zip(axes, HARD_CATS):
        ax.axhline(-EPS_DP, ls="--", c="#333", lw=1.1)
        ax.axvline(-EPS_RZ, ls=":", c="#333", lw=1.1)
        ax.scatter([0], [0], s=180, marker="s", color="#DD8452", edgecolors="black", zorder=6,
                   label="C2 fixed (baseline)")
        for r in b1["candidates"]:
            pc = r["per_cat"].get(cat)
            if not pc:
                continue
            ax.scatter([pc["mean_dR"]], [pc["mean_dP"]], s=130, marker="o",
                       color="#55A868" if r["B1_advance"] else "#4C72B0",
                       edgecolors="black", zorder=5)
            ax.annotate(r["candidate"].replace("E11_", ""), (pc["mean_dR"], pc["mean_dP"]),
                        textcoords="offset points", xytext=(5, 4), fontsize=7)
        ax.set_title(cat, fontsize=11)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("Δ preservation vs corrected Original")
    axes[1].set_xlabel("Δ robustness vs corrected Original")
    h, l = axes[0].get_legend_handles_labels()
    axes[0].legend(h, l, fontsize=8, loc="lower left")
    fig.suptitle("Experiment 11B-1 — fixed C2 vs adaptive candidates (seed 0); bands: Δd′ ≥ −0.10, Δ|Δz| ≤ −0.02",
                 fontsize=10.5)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(FIG / "fig1_fixed_vs_adaptive_plane.png", dpi=165, bbox_inches="tight")
    plt.close(fig)


def _fig2(c: dict, cid: str) -> None:
    s = next((x for x in c["candidates"] if x["candidate"] == cid), None)
    if s is None:
        return
    grid = np.zeros((len(CATS), len(SEEDS)))
    for r in s["units"]:
        grid[CATS.index(r["category"]), SEEDS.index(r["seed"])] = (
            1 if r["pass"] else (-1 if r["catastrophic"] else 0))
    fig, ax = plt.subplots(figsize=(5.6, 4.6))
    ax.imshow(grid, cmap=matplotlib.colors.ListedColormap(["#C44E52", "#DDDDDD", "#55A868"]),
              vmin=-1, vmax=1)
    for i in range(len(CATS)):
        for j in range(len(SEEDS)):
            lab = {1: "PASS", 0: "HOLD", -1: "FAIL"}[int(grid[i, j])]
            ax.text(j, i, lab, ha="center", va="center", fontsize=9, fontweight="bold")
    ax.set_xticks(range(len(SEEDS))); ax.set_xticklabels([f"seed{s}" for s in SEEDS])
    ax.set_yticks(range(len(CATS))); ax.set_yticklabels(CATS)
    ax.set_title("%s — 5 categories × seeds\nPASS=%d/15" % (cid.replace("E11_", ""), s["n_pass"]), fontsize=10.5)
    fig.tight_layout()
    fig.savefig(FIG / "fig2_heatmap_5cat_x_seed.png", dpi=170, bbox_inches="tight")
    plt.close(fig)


def _fig3(c: dict) -> None:
    base = c2fixed_baseline_table()
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.8))
    x = np.arange(len(CATS)); w = 0.22
    ax = axes[0]
    ax.bar(x - 1.5 * w, [base["per_cat_pass"][c] for c in CATS], w, label="C2 fixed (Exp10)",
           color="#DD8452", edgecolor="black", lw=0.7)
    for i, r in enumerate(c["candidates"][:2]):
        ax.bar(x + (i - 0.5) * w, [r["per_cat_pass"].get(c, 0) for c in CATS], w,
               label=r["candidate"].replace("E11_", ""), edgecolor="black", lw=0.7,
               color=["#55A868", "#4C72B0"][i])
    ax.set_xticks(x); ax.set_xticklabels(CATS, fontsize=9)
    ax.set_ylabel("PASS seeds (of 3)"); ax.set_ylim(0, 3.4)
    ax.set_title("per-category PASS count (target: all ≥2, cable ≥2)")
    ax.legend(fontsize=8); ax.grid(alpha=0.3, axis="y")
    ax = axes[1]
    ax.bar(x - 1.5 * w, [base["worst_dP"]] * len(CATS), w, color="#DD8452", alpha=0.35,
           edgecolor="black", lw=0.7, label="C2 fixed worst Δd′")
    for i, r in enumerate(c["candidates"][:2]):
        ax.bar(x + (i - 0.5) * w, [r["per_cat"].get(c, {}).get("worst_dP", np.nan) for c in CATS], w,
               color=["#55A868", "#4C72B0"][i], edgecolor="black", lw=0.7,
               label=r["candidate"].replace("E11_", "") + " worst Δd′")
    ax.axhline(CATA, ls="--", c="#C44E52", lw=1.2, label="catastrophic (−0.25)")
    ax.axhline(-EPS_DP, ls=":", c="#333", lw=1.1, label="tolerance (−0.10)")
    ax.set_xticks(x); ax.set_xticklabels(CATS, fontsize=9)
    ax.set_title("worst-case preservation per category"); ax.legend(fontsize=7.5)
    ax.grid(alpha=0.3, axis="y")
    fig.suptitle("Experiment 11 — C2 fixed vs Adaptive C2 (category consistency)", fontsize=10.5)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(FIG / "fig3_fixed_vs_adaptive_consistency.png", dpi=165, bbox_inches="tight")
    plt.close(fig)


def stage_ablation(cid: str = "E11_B2", cats=("bottle", "cable")) -> dict:
    """Ablation preview：Original / L3-only(=T1) / L2-only / Full，验证 L2+L3 互补性。"""
    det = {"candidate": cid, "per_category": {},
           "complementarity_rule": ("Full 的 robustness gain 优于 L2-only（L3 提供 robustness）"
                                    "且 Full 的 preservation 优于 L3-only（L2 提供 preservation）")}
    ok_all = []
    for cat in cats:
        o, _ = M(cat, 0, ORIG)
        t, _ = M(cat, 0, T1)
        l2, _ = M(cat, 0, f"{cid}_L2only__{cat}")
        full, _ = M(cat, 0, f"{cid}__{cat}")
        if None in (o, t, l2, full):
            continue
        def d(m):
            return (m["mean_abs_delta_z"] - o["mean_abs_delta_z"], m["mean_dprime"] - o["mean_dprime"])
        tR, tP = d(t); lR, lP = d(l2); fR, fP = d(full)
        comp = bool(fR < lR and fP > tP)
        ok_all.append(comp)
        det["per_category"][cat] = {"original": (o["mean_abs_delta_z"], o["mean_dprime"]),
                                    "L3_only_dR": tR, "L3_only_dP": tP,
                                    "L2_only_dR": lR, "L2_only_dP": lP,
                                    "full_dR": fR, "full_dP": fP, "complementary": comp}
    det["complementarity_supported"] = bool(ok_all) and all(ok_all)
    (E11 / "analysis" / "ablation_preview.json").write_text(json.dumps(det, indent=2, ensure_ascii=False))
    return det


def run_all() -> dict:
    b1 = stage_b1()
    b2 = stage_b2(["E11_B2", "E11_C2", "E11_A1"])
    cc = stage_c(b2["advanced_top2"] or ["E11_B2"])
    base = c2fixed_baseline_table()
    abl = stage_ablation()
    _fig1(b1)
    _fig2(cc, (b2["advanced_top2"] or ["E11_B2"])[0])
    _fig3(cc)
    rows = [{"rank": 1, "method": "Adaptive C2 (E11_B2)", "family": "B (category-adaptive β)",
             "PASS_15": cc["candidates"][0]["n_pass"], "cata": cc["candidates"][0]["n_cata"],
             **{f"{c}_pass": cc["candidates"][0]["per_cat_pass"].get(c) for c in CATS},
             "worst_dP": cc["candidates"][0]["worst_dP"],
             "strong_advance": cc["candidates"][0]["strong_advance"],
             "final_method_candidate": cc["candidates"][0]["final_method_candidate"]},
            {"rank": 2, "method": "C2 fixed (Exp10)", "family": "fixed β=0.25",
             "PASS_15": base["n_pass"], "cata": base["n_cata"],
             **{f"{c}_pass": base["per_cat_pass"].get(c) for c in CATS},
             "worst_dP": base["worst_dP"], "strong_advance": False, "final_method_candidate": False}]
    wcsv(E11 / "final_ranking.csv", rows)
    (E11 / "analysis" / "summary.json").write_text(json.dumps(
        {"b1": {"top3": b1["advanced_top3"],
                "rows": [{k: v for k, v in r.items() if k not in ("units", "per_cat", "weak_gain")}
                         for r in b1["candidates"]]},
         "b2": {"top2": b2["advanced_top2"],
                "rows": [{k: v for k, v in r.items() if k not in ("units", "per_cat")} for r in b2["candidates"]]},
         "c": {"rows": [{k: v for k, v in r.items() if k not in ("units", "per_cat")} for r in cc["candidates"]]},
         "c2fixed_baseline": base, "ablation": abl, "final_ranking": rows},
        indent=2, ensure_ascii=False, default=str))
    return {"b1": b1, "b2": b2, "c": cc, "base": base, "abl": abl, "ranking": rows}


if __name__ == "__main__":
    r = run_all()
    c0 = r["c"]["candidates"][0]
    print("=" * 120)
    print("Experiment 11 — C2 Adaptive Preservation | FINAL")
    print("=" * 120)
    print("%-26s%10s%10s%10s%10s"%("method", "PASS/15", "cata", "worst dP", "verdict"))
    for x in r["ranking"]:
        print("%-26s%10s%10d%+10.4f%10s" % (x["method"], "%d/15" % x["PASS_15"], x["cata"],
                                            x["worst_dP"],
                                            "FINAL" if x["final_method_candidate"] else
                                            ("STRONG" if x["strong_advance"] else "HOLD")))
    print("-" * 120)
    print("per-category PASS (adaptive vs fixed):",
          {c: (c0["per_cat_pass"].get(c), r["base"]["per_cat_pass"].get(c)) for c in CATS})
    print("ablation complementarity supported:", r["abl"]["complementarity_supported"])
    for cat, v in r["abl"]["per_category"].items():
        print("   %-7s L3only=%+.3f/%+.3f  L2only=%+.3f/%+.3f  Full=%+.3f/%+.3f  comp=%s"
              % (cat, v["L3_only_dR"], v["L3_only_dP"], v["L2_only_dR"], v["L2_only_dP"],
                 v["full_dR"], v["full_dP"], v["complementary"]))
    print("artifacts ->", ANA, FIG)
