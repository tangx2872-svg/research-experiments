"""Overnight Queue Q2 — analysis（CPU-only；复用 7A-O 冻结指标实现与 tier 规则）。

用法：
  python -u scripts/experiment7ao_q2_analysis.py --stage screen
  python -u scripts/experiment7ao_q2_analysis.py --stage final --promoted "Q2L0_l2org_l3uni"
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import experiment7ao_config as c7  # noqa: E402
import experiment7ao_q2 as q2  # noqa: E402
import experiment7ao_analysis as A  # noqa: E402

SUM, FIG, Q2REF = c7.SUM_DIR, c7.FIG_DIR, q2.Q2_REF


def coverage() -> dict:
    cov = defaultdict(set)
    for p in sorted(q2.Q2_RAW.glob("*/seed_*/config_*/info.json")):
        d = json.loads(p.read_text())
        if d.get("status") == "OK":
            cov[d["config"]].add((d["category"], int(d["seed"])))
    return cov


def q2_report(rows, lookup, tag: str) -> None:
    print(A.print_table(rows, tag))
    base = A.baseline_key_maps()["B2_uniform_040091275"]
    uni = A.eval_policy(lookup, base, q2.SCREEN_CATS, q2.SCREEN_SEEDS)
    out = []
    for r in rows:
        if r["type"] != "candidate":
            continue
        scope_c = c7.CATEGORIES if r["n_units"] > 3 else q2.SCREEN_CATS
        scope_s = c7.SEEDS if r["n_units"] > 3 else q2.SCREEN_SEEDS
        ev = A.eval_policy(lookup, {c: ("config", r["config"]) for c in scope_c}, scope_c, scope_s)
        for c in scope_c:
            d = round(ev["per_cat"][c]["mean_dprime"], 6)
            z = round(ev["per_cat"][c]["mean_abs_delta_z"], 6)
            out.append({"config": r["config"], "scope": f"{len(scope_c)}cat x {len(scope_s)}seed",
                        "layer_config": json.dumps(q2.Q2_SPECS[r["config"]]), "category": c,
                        "mean_defect_dprime": d, "mean_abs_delta_z": z,
                        "delta_dprime_vs_uniform": round(d - uni["per_cat"][c]["mean_dprime"], 6),
                        "delta_abs_z_vs_uniform": round(z - uni["per_cat"][c]["mean_abs_delta_z"], 6)})
    f = SUM / "q2_layer_representation.csv"
    with open(f, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out[0].keys()))
        w.writeheader(); w.writerows(out)
    print(f"[Q2] wrote {f.relative_to(ROOT)} ({len(out)} rows)")


def fig_q2(lookup, cfgs) -> None:
    uni = A.eval_policy(lookup, A.baseline_key_maps()["B2_uniform_040091275"], q2.SCREEN_CATS, q2.SCREEN_SEEDS)
    orig = A.eval_policy(lookup, A.baseline_key_maps()["B0_original"], q2.SCREEN_CATS, q2.SCREEN_SEEDS)
    pts = {"Uniform(0.4009)": (uni["mean_abs_delta_z"], uni["mean_dprime"], "#C44E52"),
           "Original(0)": (orig["mean_abs_delta_z"], orig["mean_dprime"], "#999999")}
    cols = ["#4C72B0", "#DD8452", "#55A868", "#CCB974"]
    for i, n in enumerate(cfgs):
        ev = A.eval_policy(lookup, {c: ("config", n) for c in q2.SCREEN_CATS}, q2.SCREEN_CATS, q2.SCREEN_SEEDS)
        pts[n] = (ev["mean_abs_delta_z"], ev["mean_dprime"], cols[i % 4])
    fig, axes = plt.subplots(1, 2, figsize=(13.4, 5.0))
    for k, (x, y, col) in pts.items():
        axes[0].scatter(x, y, s=190, color=col, edgecolors="black", linewidths=0.8, zorder=5)
        axes[0].annotate(k, (x, y), textcoords="offset points", xytext=(8, 6), fontsize=8)
    axes[0].set_xlabel("mean |ΔNormalScore_z|  (lower better)")
    axes[0].set_ylabel("mean defect d′  (higher better)")
    axes[0].set_title("Q2 — layer x representation plane (3 cat x seed0)")
    axes[0].grid(alpha=0.3)
    x = np.arange(len(cfgs)); w = 0.38
    d_dp, d_rz = [], []
    for n in cfgs:
        ev = A.eval_policy(lookup, {c: ("config", n) for c in q2.SCREEN_CATS}, q2.SCREEN_CATS, q2.SCREEN_SEEDS)
        d_dp.append(ev["mean_dprime"] - uni["mean_dprime"])
        d_rz.append(ev["mean_abs_delta_z"] - uni["mean_abs_delta_z"])
    axes[1].bar(x - w / 2, d_rz, w, label="Δ|Δz| vs Uniform", color="#C44E52", edgecolor="black", linewidth=0.5)
    axes[1].bar(x + w / 2, d_dp, w, label="Δd′ vs Uniform", color="#4C72B0", edgecolor="black", linewidth=0.5)
    axes[1].axhline(0, color="k", lw=0.9)
    for lv in (0.10, 0.02, -0.02):
        axes[1].axhline(lv, color="gray", ls=":", lw=1.0)
    axes[1].set_xticks(x); axes[1].set_xticklabels(cfgs, rotation=14, fontsize=8)
    axes[1].legend(fontsize=8); axes[1].grid(alpha=0.3, axis="y")
    axes[1].set_title("Δ vs Uniform (dotted = Tier A / B thresholds)")
    fig.suptitle("Q2 — Layer × Representation Composition", y=1.02)
    fig.tight_layout(); fig.savefig(FIG / "q2_layer_comparison.png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    print("[Q2] wrote figures/q2_layer_comparison.png")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="screen", choices=["screen", "final"])
    ap.add_argument("--promoted", default="")
    args = ap.parse_args()
    lookup, meta = A.build_lookup(extra=[("Q2", q2.Q2_RAW)])
    print(f"[Q2] lookup keys = {len(lookup)}")
    cov = coverage()
    print(f"[Q2] units on disk: { {k: sorted(v) for k, v in sorted(cov.items())} }")
    if args.stage == "screen":
        rows, uni = A.rank_rows(lookup, q2.Q2_ORDER, q2.SCREEN_CATS, q2.SCREEN_SEEDS,
                                A.baseline_key_maps()["B2_uniform_040091275"])
        print(A.print_table(rows, "[Q2 SCREEN] layer x representation, 3 categories x seed0, vs Uniform "
                                  f"(subset d′={uni['mean_dprime']:.4f} / |Δz|={uni['mean_abs_delta_z']:.4f})"))
        q2_report(rows, lookup, "[Q2 SCREEN] per-category")
        fig_q2(lookup, q2.Q2_ORDER)
        promoted = [r["config"] for r in rows if r["type"] == "candidate" and r["tier"] in c7.PROMOTE_TIERS]
        (SUM / "q2_promotion.json").write_text(json.dumps(
            {"promoted": promoted, "rule": "Q0 tier rule vs Uniform on the same subset", "tiers": c7.TIER},
            indent=2))
        print(f"[Q2] PROMOTED (mechanical): {promoted}")
        return
    winners = [w for w in (args.promoted.split(",") if args.promoted else []) if w]
    rows, uni = A.rank_rows(lookup, winners, c7.CATEGORIES, c7.SEEDS,
                            A.baseline_key_maps()["B2_uniform_040091275"])
    print(A.print_table(rows, f"[Q2 FINAL] full 5 categories x 3 seeds for {winners} vs Uniform "
                              f"(d′={uni['mean_dprime']:.4f} / |Δz|={uni['mean_abs_delta_z']:.4f})"))
    q2_report(rows, lookup, "[Q2 FINAL] per-category")
    fig_q2(lookup, winners)
    out = [{"config": r["config"], "family": "Q2", "n_units": r["n_units"],
            "mean_defect_dprime": r["mean_defect_dprime"], "mean_abs_delta_z": r["mean_abs_delta_z"],
            "delta_dprime_vs_uniform": r["delta_dprime_vs_uniform"],
            "delta_abs_z_vs_uniform": r["delta_abs_z_vs_uniform"],
            "worst_category_delta": r["worst_category_delta"],
            "negative_transfer_count": r["negative_transfer_count"], "tier": r["tier"],
            "layer_config": json.dumps(q2.Q2_SPECS.get(r["config"], {}))}
           for r in rows if r["type"] == "candidate"]
    c7.write_csv(SUM / "q2_ranking_full.csv", out)
    print("[Q2] wrote summary/q2_ranking_full.csv")


if __name__ == "__main__":
    main()
