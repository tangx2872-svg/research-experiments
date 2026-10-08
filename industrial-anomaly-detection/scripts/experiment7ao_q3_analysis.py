"""Overnight Queue Q3 — analysis（CPU-only）。

指标严格沿用 5A-H 冻结口径：
  mu_g / sigma_g  <- clean_good ;  Delta_z(cond) = mean((s_shift - mu_g)/sigma_g)
  d' <- h5.d_prime(defect_scores, mu_g, sigma_g)，逐 defect type 平均（与 illumination condition 无关）
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
import experiment7ao_q3 as q3  # noqa: E402
import experiment5a_h_analysis as h5  # noqa: E402

SUM, FIG = c7.SUM_DIR, c7.FIG_DIR
HIST_B0 = ROOT / "results" / "experiment_5a_h" / "raw"
HIST_UNI_ROOTS = [ROOT / "results" / "experiment_6b" / "raw_new",
                  ROOT / "results" / "experiment_6a" / "raw_new"]


def stress_raw_roots() -> list:
    roots = [q3.Q3_RAW]
    try:
        import experiment7ao_q4 as q4
        if q4.Q4_RAW.exists():
            roots.append(q4.Q4_RAW)          # Q4-B: stress-test seeds 1/2
    except Exception:
        pass
    return roots


def load_units() -> list:
    out = []
    for root in stress_raw_roots():
        out.extend(_load_root(root))
    return out


def _load_root(root: Path) -> list:
    out = []
    for ip in sorted(root.glob("*/seed_*/method_*/info.json")):
        d = json.loads(ip.read_text())
        if d.get("status") != "OK":
            continue
        rows = list(csv.DictReader(open(ip.parent / "per_image.csv", newline="")))
        sub = defaultdict(list)
        for r in rows:
            sub[r["condition"] if r["subset"] == "shift_good" else r["subset"]].append(r)
        out.append({"info": d, "sub": sub, "dir": ip.parent})
    return out


def unit_metrics(u: dict) -> dict:
    sub = u["sub"]
    cg = np.array([float(r["score"]) for r in sub["clean_good"]])
    mu_g, sd_g = float(cg.mean()), float(cg.std(ddof=1))
    tau = u["info"]["tau_val"]
    dfd = sub["clean_defect"]
    per_def = {}
    for dt in sorted({r["defect_type"] for r in dfd}):
        sd_ = np.array([float(r["score"]) for r in dfd if r["defect_type"] == dt])
        per_def[dt] = h5.d_prime(sd_, mu_g, sd_g)
    all_def = np.array([float(r["score"]) for r in dfd])
    conds = {}
    for name, fam, lv, sev in q3.CONDITIONS:
        ss = np.array([float(r["score"]) for r in sub[name]])
        z = (ss - mu_g) / sd_g if sd_g > 0 else np.full_like(ss, np.nan)
        conds[name] = {"family": fam, "level": lv, "severity": sev, "severity_x": q3.SEVERITY_X[sev],
                       "mean_delta_z": float(z.mean()), "abs_delta_z": float(abs(z.mean())),
                       "fpr_shift": float((ss > tau).mean()), "n": len(ss)}
    return {"method": u["info"]["method"], "category": u["info"]["category"], "seed": u["info"]["seed"],
            "tau_val": tau, "normal_clean_mean": mu_g, "normal_clean_std": sd_g,
            "fpr_clean": float((cg > tau).mean()), "mean_dprime": float(np.mean(list(per_def.values()))),
            "per_defect": per_def, "image_auroc": float(h5.image_auroc(cg, all_def)),
            "conditions": conds, "runtime": u["info"]["runtime_seconds"],
            "peak_vram": u["info"]["peak_gpu_memory_allocated_mb"]}


def _hist_ref(method: str, cat: str, seed: int):
    if method == "M0_original":
        p = HIST_B0 / cat / f"seed_{seed}" / "config_B0" / "per_image.csv"
        return p if p.exists() else None
    if method == "M1_uniform":
        for root in HIST_UNI_ROOTS:
            for d in sorted(root.glob(f"{cat}/seed_{seed}/config_*")):
                ip = d / "info.json"
                if not ip.exists():
                    continue
                info = json.loads(ip.read_text())
                if (info.get("status") == "OK"
                        and abs(float(info.get("alpha_l2", -1)) - c7.UNIFORM_ALPHA) < 1e-12
                        and abs(float(info.get("alpha_l3", -1)) - c7.UNIFORM_ALPHA) < 1e-12):
                    return d / "per_image.csv"
    return None


MEDIUM_NAMES = [c[0] for c in q3.CONDITIONS if c[3] == "medium"]


def equivalence_check() -> list:
    """medium 档（0.7/1.3）必须与历史 raw 逐位一致。"""
    res = []
    for u in load_units():
        m, cat, seed = u["info"]["method"], u["info"]["category"], u["info"]["seed"]
        ref = _hist_ref(m, cat, seed)
        if ref is None:
            res.append({"method": m, "category": cat, "seed": seed, "reference": "NO_HISTORY",
                        "n_keys": 0, "max_abs_delta_score": None})
            continue
        mine = {}
        for r in csv.DictReader(open(u["dir"] / "per_image.csv", newline="")):
            if r["subset"] == "shift_good" and r["condition"] in MEDIUM_NAMES:
                mine[(r["image_path"], r["condition"])] = float(r["score"])
        hist = {(r["image_path"], r["shift"]): float(r["score"])
                for r in csv.DictReader(open(ref, newline="")) if r["subset"] == "shift_good"}
        keys = set(mine) & set(hist)
        mx = max((abs(mine[k] - hist[k]) for k in keys), default=None)
        res.append({"method": m, "category": cat, "seed": seed, "reference": str(ref.parent.name),
                    "n_keys": len(keys), "max_abs_delta_score": mx})
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all", choices=["all", "equiv"])
    args = ap.parse_args()
    if args.stage == "equiv":
        print("[Q3] medium-condition equivalence vs historical raw (max|dscore|):")
        for r in equivalence_check():
            v = r["max_abs_delta_score"]
            print(f"  {r['method']:<12}{r['category']:<10}seed{r['seed']}  ref={r['reference']:<14}"
                  f"n={r['n_keys']:<4}max|dscore|=" + ("n/a" if v is None else f"{v:.3e}"))
        return
    per_unit = [unit_metrics(u) for u in load_units()]
    by_method = defaultdict(list)
    for u in per_unit:
        by_method[u["method"]].append(u)
    print(f"[Q3] units={len(per_unit)} methods={sorted(by_method)}")
    stress, summ = [], []
    for m, us in sorted(by_method.items()):
        for name, fam, lv, sev in q3.CONDITIONS:
            v = [u["conditions"][name] for u in us]
            stress.append({"method": m, "condition": name, "family": fam, "level": lv, "severity": sev,
                           "severity_x": q3.SEVERITY_X[sev],
                           "mean_delta_z": round(float(np.mean([x["mean_delta_z"] for x in v])), 6),
                           "abs_delta_z": round(float(np.mean([x["abs_delta_z"] for x in v])), 6),
                           "abs_delta_z_sd_over_categories":
                               round(float(np.std([x["abs_delta_z"] for x in v], ddof=1)), 6),
                           "fpr_shift": round(float(np.mean([x["fpr_shift"] for x in v])), 6),
                           "n_units": len(v)})
        abz = [float(np.mean([u["conditions"][c[0]]["abs_delta_z"] for u in us])) for c in q3.CONDITIONS]
        med = [float(np.mean([u["conditions"][c[0]]["abs_delta_z"] for u in us]))
               for c in q3.CONDITIONS if c[3] == "medium"]
        summ.append({"method": m, "n_units": len(us),
                     "mean_defect_dprime": round(float(np.mean([u["mean_dprime"] for u in us])), 6),
                     "image_auroc": round(float(np.mean([u["image_auroc"] for u in us])), 6),
                     "fpr_clean": round(float(np.mean([u["fpr_clean"] for u in us])), 6),
                     "mean_abs_delta_z_all12": round(float(np.mean(abz)), 6),
                     "mean_abs_delta_z_medium4": round(float(np.mean(med)), 6),
                     "worst_condition_abs_delta_z": round(float(np.max(abz)), 6),
                     "runtime_seconds": round(float(np.mean([u["runtime"] for u in us])), 1),
                     "peak_vram_mb": round(float(np.mean([u["peak_vram"] for u in us])), 1)})
    slopes = []
    for m, us in sorted(by_method.items()):
        for fam in q3.FAMILIES + ["pooled"]:
            xs, ys = [], []
            for name, f, lv, sev in q3.CONDITIONS:
                if fam != "pooled" and f != fam:
                    continue
                xs.append(q3.SEVERITY_X[sev])
                ys.append(float(np.mean([u["conditions"][name]["abs_delta_z"] for u in us])))
            b, a = np.polyfit(xs, ys, 1)
            pred = a + b * np.array(xs)
            ss_res = float(np.sum((np.array(ys) - pred) ** 2))
            ss_tot = float(np.sum((np.array(ys) - np.mean(ys)) ** 2))
            slopes.append({"method": m, "family": fam,
                           "slope_abs_delta_z_per_unit_deviation": round(float(b), 6),
                           "intercept": round(float(a), 6),
                           "r2": (round(1 - ss_res / ss_tot, 6) if ss_tot > 0 else None),
                           "n_points": len(xs)})
    for s in summ:
        if "M1_uniform" in by_method:
            ref = {r["condition"]: r["abs_delta_z"] for r in stress if r["method"] == "M1_uniform"}
            s["neg_transfer_count_vs_uniform"] = int(sum(
                1 for r in stress if r["method"] == s["method"] and r["abs_delta_z"] > ref[r["condition"]]))
    c7.write_csv(SUM / "q3_illumination_stress.csv", stress)
    c7.write_csv(SUM / "q3_degradation_slopes.csv", slopes)
    c7.write_csv(SUM / "q3_method_summary.csv", summ)
    c7.write_csv(SUM / "q3_equivalence_check.csv", equivalence_check())
    print("\n=== Q3 method summary ===")
    for s in summ:
        print("  " + json.dumps(s))
    print("\n=== Q3 degradation slopes ===")
    for s in slopes:
        print("  " + json.dumps(s))
    print("\n=== Q3 per-condition abs_delta_z ===")
    for m in sorted(by_method):
        row = {r["condition"]: r["abs_delta_z"] for r in stress if r["method"] == m}
        print(f"  {m:<12}" + " ".join(f"{k}={v:.4f}" for k, v in row.items()))
    cols = {"M0_original": "#999999", "M1_uniform": "#C44E52"}
    palette = ["#4C72B0", "#55A868", "#DD8452"]
    fig, axes = plt.subplots(1, 2, figsize=(12.6, 4.8))
    for ax, fam in zip(axes, q3.FAMILIES):
        for i, m in enumerate(sorted(by_method)):
            xs, ys, es = [], [], []
            for name, f, lv, sev in q3.CONDITIONS:
                if f != fam:
                    continue
                vals = [u["conditions"][name]["abs_delta_z"] for u in by_method[m]]
                xs.append(q3.SEVERITY_X[sev]); ys.append(float(np.mean(vals)))
                es.append(float(np.std(vals, ddof=1)))
            ax.errorbar(xs, ys, yerr=es, marker="o", lw=2.0, capsize=3,
                        color=cols.get(m, palette[i % 3]), label=m)
        ax.set_xlabel("|1 - level|   (0.1 mild / 0.3 medium / 0.5 strong)")
        ax.set_ylabel("mean |ΔNormalScore_z|")
        ax.set_title(f"{fam} perturbation")
        ax.set_xticks([0.1, 0.3, 0.5]); ax.grid(alpha=0.3); ax.legend(fontsize=8)
    fig.suptitle("Q3 — illumination severity stress-test (5 categories × seed0; error bars = s.d. over categories)",
                 y=1.03)
    fig.tight_layout(); fig.savefig(FIG / "q3_robustness_curves.png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    print("[Q3] wrote summary/q3_*.csv + figures/q3_robustness_curves.png")


if __name__ == "__main__":
    main()
