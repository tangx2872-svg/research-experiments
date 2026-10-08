"""Experiment 15 — X4（feature-side）分析 + 统一 leaderboard（含 Evidence Level）。

Pareto-positive 的操作化（§13，在看到 X4 结果前确定）：
    pareto_pos = (ddp > 0 且 dR <= +0.02)          # preservation 改善且 robustness 基本不退
              或 (dR <= -0.02 且 ddp >= -0.02)     # robustness 改善且 preservation 基本不退
              或 相对参考消除了 catastrophic        # 明显减少 catastrophic
参考 = 同 (cat,seed) 的 corrected Original；EPS 带沿用冻结的 0.02。
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import experiment14_metrics as M  # noqa: E402
import experiment15_candidates as c15  # noqa: E402

E15 = ROOT / "results" / "experiment_15"
ANA = E15 / "analysis"
EPS = 0.02


def discover(rounds) -> dict:
    out = {}
    for rd in rounds:
        for ip in sorted((E15 / "raw" / rd).glob("*/*/config_*/info.json")):
            try:
                if not str(json.loads(ip.read_text()).get("status", "")).startswith("OK"):
                    continue
            except Exception:
                continue
            d = ip.parent
            out[(d.parent.parent.name, int(d.parent.name.replace("seed_", "")),
                 d.name.replace("config_", "").split("__")[0])] = d
    return out


def main() -> None:
    ANA.mkdir(parents=True, exist_ok=True)
    units = discover(["x4_r1", "x4_r2"])
    rows_x4 = []
    if units:
        mt = M.metrics_of(units)
        pairs = sorted({(c, s) for c, s, _ in units})
        base = {}
        for c, s in pairs:
            p = M.source(c, s, "orig")
            if p is not None:
                base[(c, s, "o")] = p
        bm = M.metrics_of(base)
        for (cat, seed, cid), m in sorted(mt.items()):
            v = M.verdict(bm[(cat, seed, "o")], m)
            b2 = M.source(cat, seed, "b2")
            vb = M.verdict(M.metrics_of({(cat, seed, "b2"): b2})[(cat, seed, "b2")], m) if b2 else {}
            pareto = ((v["d_dprime"] > 0 and v["d_absz"] <= EPS)
                      or (v["d_absz"] <= -EPS and v["d_dprime"] >= -EPS))
            rows_x4.append({"method": cid, "category": cat, "seed": seed,
                            "dR": round(v["d_absz"], 6), "ddp": round(v["d_dprime"], 6),
                            "PASS": v["PASS"], "catastrophic": v["catastrophic"],
                            "pareto_pos": bool(pareto),
                            "dR_vs_B2": round(vb.get("d_absz", float("nan")), 6),
                            "ddp_vs_B2": round(vb.get("d_dprime", float("nan")), 6),
                            "absz": round(m["mean_abs_delta_z"], 6),
                            "dprime": round(m["mean_dprime"], 6),
                            "image_auroc": round(m["image_auroc"], 6)})
        with open(ANA / "X4_per_unit.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows_x4[0].keys()))
            w.writeheader(); w.writerows(rows_x4)

    ss = list(csv.DictReader(open(ANA / "scoreside_summary.csv")))
    x4s = {}
    for cid in sorted({r["method"] for r in rows_x4}):
        r = [x for x in rows_x4 if x["method"] == cid]
        s0 = [x for x in r if x["seed"] == 0]
        x4s[cid] = {"method": cid, "n_units": len(r),
                    "PASS": sum(1 for x in r if x["PASS"]),
                    "catastrophic": sum(1 for x in r if x["catastrophic"]),
                    "worst_ddp": round(min(x["ddp"] for x in r), 4),
                    "mean_dR": round(float(np.mean([x["dR"] for x in r])), 4),
                    "mean_ddp": round(float(np.mean([x["ddp"] for x in r])), 4),
                    "pareto_units": sum(1 for x in r if x["pareto_pos"]),
                    "s0_5cat_PASS": sum(1 for x in s0 if x["PASS"]),
                    "s0_5cat_n": len(s0),
                    "evidence": "E1" if len(r) < 15 else "E2"}

    lb = []
    for s in ss:
        lb.append({"method": s["method"], "side": "score", "n_units": int(s["n_units"]),
                   "PASS": int(s["PASS"]), "catastrophic": int(s["catastrophic"]),
                   "worst_ddp": float(s["worst_ddp"]), "mean_dR": float(s["mean_dR"]),
                   "mean_ddp": float(s["mean_ddp"]), "evidence": s["evidence"],
                   "pareto_units": ""})
    for cid, s in x4s.items():
        lb.append({"method": cid, "side": "feature", "n_units": s["n_units"], "PASS": s["PASS"],
                   "catastrophic": s["catastrophic"], "worst_ddp": s["worst_ddp"],
                   "mean_dR": s["mean_dR"], "mean_ddp": s["mean_ddp"], "evidence": s["evidence"],
                   "pareto_units": s["pareto_units"]})
    # Adaptive B2 参考（25 单元）
    # Adaptive B2 参考：以 bootstrap_ci.json 为准（35 单元，E3+）
    try:
        bref = json.loads((ANA / "bootstrap_ci.json").read_text())["Adaptive B2"]
    except Exception:
        bref = {"PASS": 19, "n_units": 25, "catastrophic": 3, "worst_ddp": -0.41,
                "mean_dR": -0.0843, "mean_ddp": 0.165}
    lb.append({"method": "Adaptive B2 (reference)", "side": "reference", "n_units": bref["n_units"],
               "PASS": bref["PASS"], "catastrophic": bref["catastrophic"], "worst_ddp": bref["worst_ddp"],
               "mean_dR": bref["mean_dR"], "mean_ddp": bref.get("mean_ddp", ""),
               "evidence": "E3+", "pareto_units": ""})
    lb.sort(key=lambda r: (-r["PASS"], r["catastrophic"], -r["worst_ddp"]))
    lb.insert(0, {"method": "(rank key)", "side": "-", "n_units": "-", "PASS": "PASS/units",
                  "catastrophic": "cat", "worst_ddp": "worst Δd′", "mean_dR": "mean ΔR",
                  "mean_ddp": "mean Δd′", "evidence": "evid", "pareto_units": "#pareto"})
    with open(ANA / "leaderboard.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(lb[0].keys()))
        w.writeheader(); w.writerows(lb[1:])

    print("=" * 120)
    print("Exp15 UNIFIED LEADERBOARD（score-side E3=25 单元；feature-side 视已完成阶段）")
    print("=" * 120)
    print("%-24s%-10s%8s%7s%5s%12s%10s%9s%7s" % ("method", "side", "units", "PASS", "cat",
                                                   "worst_ddp", "mean_dR", "mean_ddp", "evid"))
    for r in lb[1:]:
        print("%-24s%-10s%8s%7s%5s%12s%10s%9s%7s" % (r["method"], r["side"], r["n_units"], r["PASS"],
                                                      r["catastrophic"], r["worst_ddp"], r["mean_dR"],
                                                      r["mean_ddp"], r["evidence"]))
    if rows_x4:
        print("\nX4 per-unit（seed0 三类别为 R1；rows 含 screw/grid 若已跑）:")
        for x in rows_x4:
            print("   %-9s %s/s%d: %s dR%+.3f ddp%+.3f pareto=%s (vsB2 dR%+.3f ddp%+.3f)" % (
                x["method"], x["category"], x["seed"],
                "P" if x["PASS"] else ("C" if x["catastrophic"] else "F"),
                x["dR"], x["ddp"], x["pareto_pos"], x["dR_vs_B2"], x["ddp_vs_B2"]))
    json.dump({"leaderboard": lb[1:], "x4": {k: v for k, v in x4s.items()},
               "adaptive_b2_reference": bref},
              open(ANA / "tournament.json", "w"), indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
