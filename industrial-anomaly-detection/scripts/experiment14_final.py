"""Experiment 14 — 最终 Tier 评估（§15 冻结判据）+ 5x3 报告。

Tier S : >=13/15 且 0 catastrophic 且 cable>=2/3 且 hazelnut>=2/3 且 bottle 3/3 且 screw 3/3
Tier A : 12/15 且（catastrophic 1->0 或 worst Δd' 明显改善 或 cable/hazelnut consistency 改善）
Tier B : 11-12/15 且无明显优于 B2
Tier C : <=10/15 或明显 trade-off
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

E14 = ROOT / "results" / "experiment_14"
ANA = E14 / "analysis"


def load(p):
    return list(csv.DictReader(open(p))) if p.exists() else []


def main() -> None:
    rows = load(ANA / "AB_per_unit.csv") + load(ANA / "C_late_fusion_per_unit.csv")
    for r in rows:
        r["candidate"] = r["candidate"].split("__")[0]
        r.setdefault("family", "A/B")
        r.setdefault("catastrophic", r.get("catastrophic_vs_orig"))
        r["PASS_b"] = str(r["PASS_vs_orig"]) == "True"
        r["CAT_b"] = str(r.get("catastrophic")) == "True"

    # Adaptive B2 参考（15 单元，冻结历史）
    b2_units = {(c, s, "B2"): M.source(c, s, "b2") for c in M.CATS for s in M.SEEDS}
    o_units = {(c, s, "O"): M.source(c, s, "orig") for c in M.CATS for s in M.SEEDS}
    mb2, mo = M.metrics_of(b2_units), M.metrics_of(o_units)
    b2_rows = []
    for (c, s, _) in b2_units:
        v = M.verdict(mo[(c, s, "O")], mb2[(c, s, "B2")])
        b2_rows.append({"candidate": "Adaptive B2 (Exp11)", "category": c, "seed": str(s),
                        "PASS_b": v["PASS"], "CAT_b": v["catastrophic"], "ddp": v["d_dprime"],
                        "dR": v["d_absz"]})

    def summarize(rs, name):
        per_cat = {c: [x for x in rs if x["category"] == c] for c in M.CATS}
        return {"candidate": name, "n_units": len(rs),
                "PASS": sum(1 for x in rs if x["PASS_b"]),
                "catastrophic": sum(1 for x in rs if x["CAT_b"]),
                "worst_ddp": min(float(x.get("ddp_vs_orig", x.get("ddp"))) for x in rs),
                "mean_dR": float(np.mean([float(x.get("dR_vs_orig", x.get("dR"))) for x in rs])),
                "mean_ddp": float(np.mean([float(x.get("ddp_vs_orig", x.get("ddp"))) for x in rs])),
                "per_cat": {c: "%d/%d" % (sum(1 for x in per_cat[c] if x["PASS_b"]), len(per_cat[c]))
                            for c in M.CATS if per_cat[c]},
                "per_cat_cat": {c: sum(1 for x in per_cat[c] if x["CAT_b"]) for c in M.CATS if per_cat[c]}}

    summ = [summarize(b2_rows, "Adaptive B2 (Exp11)")]
    for cid in sorted({r["candidate"] for r in rows}):
        summ.append(summarize([x for x in rows if x["candidate"] == cid], cid))

    out = []
    for s in summ:
        pc, n = s["per_cat"], s["n_units"]
        full = (n == 15)
        cat_ok = (s["catastrophic"] == 0)

        def _rate(cat):
            v = pc.get(cat, "0/0")
            a, b = v.split("/")
            return (int(a), int(b))
        bottle, cable, haz, screw, grid = (_rate(c) for c in M.CATS)
        tier = "n/a (not 15 units)"
        if full:
            if (s["PASS"] >= 13 and cat_ok and cable[0] >= 2 and haz[0] >= 2
                    and bottle[0] == 3 and screw[0] == 3 and grid[0] == 3):
                tier = "S — FINAL METHOD CANDIDATE"
            elif s["PASS"] == 12 and (cat_ok or s["worst_ddp"] > -0.2852):
                tier = "A — SERIOUS FINAL CANDIDATE"
            elif s["PASS"] >= 11:
                tier = "B — archived"
            else:
                tier = "C — STOP"
        else:
            if s["PASS"] >= 12:
                tier = "A? (extrapolated; n=%d)" % n
            elif s["PASS"] >= 11:
                tier = "B? (extrapolated; n=%d)" % n
            else:
                tier = "C? (n=%d)" % n
        out.append({**{k: v for k, v in s.items() if k not in ("per_cat", "per_cat_cat")},
                    "bottle": "%d/%d" % bottle, "cable": "%d/%d" % cable, "hazelnut": "%d/%d" % haz,
                    "screw": "%d/%d" % screw, "grid": "%d/%d" % grid, "Tier": tier})
    with open(ANA / "final_leaderboard.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0].keys()))
        w.writeheader(); w.writerows(out)
    json.dump({"final": out, "frozen_rules":
               {"S": ">=13/15 + 0 catastrophic + cable>=2/3 + hazelnut>=2/3 + bottle 3/3 + screw 3/3",
                "A": "12/15 + (catastrophic 1->0 或 worst Δd' 改善 或 cable/hazelnut 一致性改善)",
                "B": "11-12/15 无明显优于 B2", "C": "<=10/15 或明显 trade-off"}},
              open(ANA / "verdict.json", "w"), indent=2, ensure_ascii=False)

    print("=" * 132)
    print("Exp14 FINAL — Tier 评估（5 cats × 3 seeds；Adaptive B2 为冻结参考）")
    print("=" * 132)
    print("%-26s%7s%7s%7s%10s%9s%9s   %-28s" % ("candidate", "units", "PASS", "cat", "worst_ddp",
                                                "mean_dR", "mean_ddp", "per-category PASS"))
    for s in out:
        print("%-26s%7d%7d%7d%10.4f%9.4f%9.4f   b%s c%s h%s s%s g%s" % (
            s["candidate"], s["n_units"], s["PASS"], s["catastrophic"], s["worst_ddp"],
            s["mean_dR"], s["mean_ddp"], s["bottle"], s["cable"], s["hazelnut"], s["screw"], s["grid"]))
    print("-" * 132)
    for s in out:
        print("  %-26s -> %s" % (s["candidate"], s["Tier"]))
    print("=" * 132)


if __name__ == "__main__":
    main()
