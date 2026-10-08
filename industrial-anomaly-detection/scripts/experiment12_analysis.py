"""Experiment 12 — analysis（CPU-only；复用 Exp10/11 的参考点 raw）。

判定：PASS = (Δ|Δz| ≤ −EPS_RZ) 且 (Δd′ ≥ −EPS_DP)，相对同 (category,seed) 的 corrected Original-189；
      catastrophic = Δd′ ≤ −0.25；所有判据带沿用 Exp10/11 冻结值（未修改）。

Round-1 排名规则（在看结果前写定，见 README §6）：PASS 类数 → catastrophic 数 → cable Δd′ 改善
→ hazelnut Δd′ 改善 → worst Δd′ → ΔR → 简洁度。
硬性淘汰：bottle PASS→FAIL；新增 catastrophic；robustness 明显恶化；cable 与 hazelnut 都无改善。
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

E12 = ROOT / "results" / "experiment_12"
E11 = ROOT / "results" / "experiment_11_c2_adaptive"
E10 = ROOT / "results" / "experiment_10_preservation_recovery"
ROOTS = [E12 / "raw", E11 / "raw", E10 / "raw"]
ANA, FIG = E12 / "analysis", E12 / "figures"
CATS = ["bottle", "cable", "hazelnut", "screw", "grid"]
HARD = ["bottle", "cable", "hazelnut"]
SEEDS = [0, 1, 2]
EPS_RZ, EPS_DP, CATA = 0.02, 0.10, -0.25
ORIG = "P10_ORIG_a000"
C2F = "P10_C2_L2resid025_L3a025"
B2 = "E11_B2"
# 冻结阈值（Exp12 Round-1）
IMPROVE_MIN = 0.02      # 与 EPS_RZ 同量级：改善可判定门槛
CLEAR_IMPROVE = 0.10    # "明显改善"（= EPS_DP）
ROB_WORSE_MAX = 0.10    # 相对 B2 的 robustness 恶化上限


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


def M(cat, seed, cfg, name=None):
    m, d = A10.metrics(cat, seed, name or cfg, ROOTS)
    return m, d


def ev(cat, seed, cfg, name=None):
    """单元评估：相对该 (cat,seed) 的 corrected Original。"""
    m, _ = M(cat, seed, cfg, name)
    o, _ = M(cat, seed, ORIG)
    if m is None or o is None:
        return None
    dR = m["mean_abs_delta_z"] - o["mean_abs_delta_z"]
    dP = m["mean_dprime"] - o["mean_dprime"]
    return {"category": cat, "seed": seed, "cfg": cfg,
            "absdz": m["mean_abs_delta_z"], "dprime": m["mean_dprime"],
            "dR": dR, "dP": dP, "pass": bool(dR <= -EPS_RZ and dP >= -EPS_DP),
            "catastrophic": bool(dP <= CATA),
            "pixel_auroc": m["pixel_auroc"], "aupro": m["aupro"], "tau": m["tau_val"]}


def units_for(prefix, cats, seeds):
    return [ev(c, s, prefix, f"{prefix}__{c}") for c in cats for s in seeds]


def summary(prefix, cats, seeds):
    rows = [r for r in units_for(prefix, cats, seeds) if r]
    if not rows:
        return None
    per = {}
    for c in cats:
        rs = [r for r in rows if r["category"] == c]
        if rs:
            per[c] = {"n": len(rs), "pass": sum(r["pass"] for r in rs),
                      "cata": sum(r["catastrophic"] for r in rs),
                      "mean_dR": float(np.mean([r["dR"] for r in rs])),
                      "mean_dP": float(np.mean([r["dP"] for r in rs])),
                      "worst_dP": float(min(r["dP"] for r in rs))}
    return {"prefix": prefix, "units": rows, "per_cat": per,
            "n_pass": sum(r["pass"] for r in rows), "n_cata": sum(r["catastrophic"] for r in rows),
            "worst_dP": float(min(r["dP"] for r in rows)),
            "mean_dR": float(np.mean([r["dR"] for r in rows])),
            "mean_dP": float(np.mean([r["dP"] for r in rows]))}


def round1(prefixes=("E12_M1", "E12_M2", "E12_M3")) -> dict:
    base = summary(B2, HARD, [0])
    det = {"baseline_M0_B2": {c: base["per_cat"].get(c, {}) for c in HARD}, "candidates": []}
    for pf in prefixes:
        s = summary(pf, HARD, [0])
        if s is None:
            continue
        pc = s["per_cat"]
        cable_imp = pc.get("cable", {}).get("mean_dP", np.nan) - base["per_cat"].get("cable", {}).get("mean_dP", np.nan)
        haze_imp = pc.get("hazelnut", {}).get("mean_dP", np.nan) - base["per_cat"].get("hazelnut", {}).get("mean_dP", np.nan)
        bottle_pass = bool(pc.get("bottle", {}).get("pass"))
        new_cata = s["n_cata"] > base["n_cata"]
        rob_worse = any((pc.get(c, {}).get("mean_dR", 0.0) - base["per_cat"].get(c, {}).get("mean_dR", 0.0)
                         >= ROB_WORSE_MAX) or (pc.get(c, {}).get("mean_dR", 0.0) > 0.0) for c in HARD)
        no_gain = bool(cable_imp < IMPROVE_MIN and haze_imp < IMPROVE_MIN)
        killed = []
        if not bottle_pass:
            killed.append("bottle PASS->FAIL")
        if new_cata:
            killed.append("new catastrophic")
        if rob_worse:
            killed.append("robustness materially worse")
        if no_gain:
            killed.append("cable & hazelnut both no improvement")
        adv_a = bool(cable_imp >= CLEAR_IMPROVE and not rob_worse)
        adv_b = bool(pc.get("hazelnut", {}).get("pass") and cable_imp >= -IMPROVE_MIN)
        adv_c = bool(cable_imp >= IMPROVE_MIN and haze_imp >= IMPROVE_MIN and bottle_pass)
        det["candidates"].append({
            "prefix": pf, "n_pass": s["n_pass"], "n_cata": s["n_cata"],
            "cable_imp_vs_B2": float(cable_imp), "haze_imp_vs_B2": float(haze_imp),
            "worst_dP": s["worst_dP"], "mean_dR": s["mean_dR"], "bottle_pass": bottle_pass,
            "eliminated": bool(killed), "kill_reasons": killed,
            "ADVANCE_A": adv_a, "ADVANCE_B": adv_b, "ADVANCE_C": adv_c,
            "advance": bool((adv_a or adv_b or adv_c) and not killed),
            "per_cat": {c: pc.get(c, {}) for c in HARD}})
    alive = [c for c in det["candidates"] if not c["eliminated"]]
    comp = {c["prefix"]: c for c in det["candidates"]}
    alive.sort(key=lambda r: (-r["n_pass"], r["n_cata"], -r["cable_imp_vs_B2"], -r["haze_imp_vs_B2"],
                              -r["worst_dP"], r["mean_dR"]))
    det["top2"] = [r["prefix"] for r in alive if r["advance"]][:2]
    det["ranking"] = [r["prefix"] for r in alive]
    det["n_advance"] = sum(1 for c in det["candidates"] if c["advance"])
    rows = [{k: v for k, v in cand.items() if k != "per_cat"} |
            {f"{cat}_{k}": v for cat in HARD for k, v in cand["per_cat"].items()}
            for cand in det["candidates"]]
    wcsv(ANA / "round1_hard_category.csv", rows)
    (ANA / "round1.json").write_text(json.dumps(det, indent=2, ensure_ascii=False, default=str))
    return det


def reference_table() -> dict:
    """C2 家族在 bottle/cable/hazelnut seed0 上的横截面（全部复用已有 raw，0 GPU）。"""
    cfgs = [("Original", ORIG, None), ("T1 (L3-only)", "P10_T1_L2a000_L3a025", None),
            ("Fixed C2 (beta=0.25,g=1)", C2F, None), ("Adaptive B2 (M0)", B2, "E11_B2__{cat}"),
            ("E11_C2 (beta=0.25,g_C2)", "E11_C2", "E11_C2__{cat}"),
            ("Exp12 M1", "E12_M1", "E12_M1__{cat}"),
            ("Exp12 M2", "E12_M2", "E12_M2__{cat}"),
            ("Exp12 M3", "E12_M3", "E12_M3__{cat}")]
    out = []
    for label, cfg, pat in cfgs:
        row = {"config": label}
        for cat in HARD:
            r = ev(cat, 0, cfg, pat.format(cat=cat) if pat else None)
            row[f"{cat}_dR"] = round(r["dR"], 4) if r else None
            row[f"{cat}_dP"] = round(r["dP"], 4) if r else None
        out.append(row)
    wcsv(ANA / "family_cross_section.csv", out)
    return {"rows": out}


def sensitivity_note() -> dict:
    """C2 家族在 cable seed0 上的 Δd′ 跨度（稳定性诊断；不是拟合）。"""
    vals = {}
    for label, cfg, pat in [("FixedC2", C2F, None), ("B2", B2, None),
                            ("E11_C2", "E11_C2", "E11_C2__{cat}"), ("M1", "E12_M1", "E12_M1__{cat}"),
                            ("M2", "E12_M2", "E12_M2__{cat}"), ("M3", "E12_M3", "E12_M3__{cat}")]:
        r = ev("cable", 0, cfg, pat.format(cat="cable") if pat else None)
        if r:
            vals[label] = round(r["dP"], 4)
    return {"cable_seed0_dP_by_config": vals,
            "spread": round(max(vals.values()) - min(vals.values()), 4),
            "note": ("cable seed0 的 Δd′ 在 C2 家族内部的窄参数区间内跨度达 %.2f，"
                     "说明该 (category,seed) 的 preservation 对补偿强度/通道选择**高度敏感**；"
                     "因此 Exp11 的 cable 改善不宜单独作为机制性结论（仅作观察事实）。"
                     % (max(vals.values()) - min(vals.values())))}


def fig1(det: dict, ref: dict) -> None:
    labs, x, y, cols = [], [], [], []
    for r in ref["rows"][1:]:
        if any(r[f"{c}_{k}"] is None for c in HARD for k in ("dR", "dP")):
            continue
        labs.append(r["config"].split(" (")[0])
        x.append((r["cable_dR"] + r["hazelnut_dR"] + r["bottle_dR"]) / 3.0)
        y.append((r["cable_dP"] + r["hazelnut_dP"] + r["bottle_dP"]) / 3.0)
        cols.append("#C44E52" if r["config"].startswith("Exp12") else "#4C72B0")
    fig, ax = plt.subplots(figsize=(8.6, 5.6))
    ax.axhline(0, c="#AAA", lw=0.8); ax.axvline(0, c="#AAA", lw=0.8)
    ax.axhline(-EPS_DP, ls="--", c="#333", lw=1.1, label="preservation band −0.10")
    ax.axvline(-EPS_RZ, ls=":", c="#333", lw=1.1, label="robustness band −0.02")
    ax.scatter(x, y, s=150, c=cols, edgecolors="black", zorder=5)
    for l, xi, yi in zip(labs, x, y):
        ax.annotate(l, (xi, yi), textcoords="offset points", xytext=(6, 4), fontsize=8)
    ax.set_xlabel("mean Δ robustness over bottle/cable/hazelnut (seed0)")
    ax.set_ylabel("mean Δ preservation over bottle/cable/hazelnut (seed0)")
    ax.set_title("Experiment 12 — Category × Channel gate strength vs the C2 family\n"
                 "(red = Exp12 candidates; blue = existing references)")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG / "fig1_gate_strength_plane.png", dpi=170, bbox_inches="tight")
    plt.close(fig)


def fig2(ref: dict) -> None:
    cats = HARD
    rows = [r for r in ref["rows"]
            if (r["config"].startswith(("Exp12", "Adaptive", "Fixed")) or r["config"].startswith("E11_C2"))
            and all(r[f"{c}_{k}"] is not None for c in cats for k in ("dR", "dP"))]
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 4.6))
    xs = np.arange(len(cats)); w = 0.19
    for ax, key, ttl in ((axes[0], "dR", "Δ robustness vs corrected Original"),
                         (axes[1], "dP", "Δ preservation vs corrected Original")):
        for i, r in enumerate(rows):
            ax.bar(xs + (i - len(rows) / 2 + 0.5) * w, [r[f"{c}_{key}"] or 0 for c in cats], w,
                   label=r["config"].split(" (")[0], edgecolor="black", lw=0.6)
        ax.set_xticks(xs); ax.set_xticklabels(cats)
        ax.axhline(0, c="#333", lw=0.8)
        ax.set_title(ttl, fontsize=10.5); ax.grid(alpha=0.3, axis="y")
    axes[1].axhline(-EPS_DP, ls="--", c="#C44E52", lw=1.1)
    axes[1].axhline(CATA, ls=":", c="#8B0000", lw=1.1)
    axes[0].legend(fontsize=7.5)
    fig.suptitle("Experiment 12 — Category × Channel combination vs baselines (seed0)", fontsize=10.5)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(FIG / "fig2_candidate_comparison.png", dpi=165, bbox_inches="tight")
    plt.close(fig)
