"""Experiment 14 — Cross-family tournament：family leaderboard → 统一 candidate leaderboard → Top4/Top2。

排名优先级（冻结，§12）：1 Pareto improvement over B2 → 2 PASS 数 → 3 catastrophic=0 →
4 worst Δd′ → 5 robustness → 6 preservation → 7 cross-category consistency → 8 简洁度 → 9 compute cost。
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ANA = ROOT / "results" / "experiment_14" / "analysis"


def load(path: Path) -> list:
    return list(csv.DictReader(open(path))) if path.exists() else []


def _f(x, d=float("nan")):
    try:
        return float(x)
    except Exception:
        return d


def summarize(rows: list) -> list:
    out = []
    for cid in sorted({r["candidate"] for r in rows}):
        r = [x for x in rows if x["candidate"] == cid]
        out.append({
            "candidate": cid, "family": r[0].get("family", "C_late_fusion"), "n_units": len(r),
            "PASS": sum(1 for x in r if str(x["PASS_vs_orig"]) == "True"),
            "catastrophic": sum(1 for x in r if str(x["catastrophic"]) == "True"),
            "worst_ddp": min(_f(x["ddp_vs_orig"]) for x in r),
            "mean_dR": float(np.mean([_f(x["dR_vs_orig"]) for x in r])),
            "mean_ddp": float(np.mean([_f(x["ddp_vs_orig"]) for x in r])),
            "mean_dR_vs_B2": float(np.mean([_f(x["dR_vs_B2"]) for x in r])),
            "mean_ddp_vs_B2": float(np.mean([_f(x["ddp_vs_B2"]) for x in r])),
            "n_pareto_vs_B2": sum(1 for x in r if _f(x["dR_vs_B2"]) <= 0 and _f(x["ddp_vs_B2"]) >= 0),
            "pareto_vs_B2": bool(float(np.mean([_f(x["dR_vs_B2"]) for x in r])) <= 0
                                 and float(np.mean([_f(x["ddp_vs_B2"]) for x in r])) >= 0
                                 and any(_f(x["dR_vs_B2"]) < 0 or _f(x["ddp_vs_B2"]) > 0 for x in r)),
            "cats": sorted({x["category"] for x in r}),
            "seeds": sorted({x["seed"] for x in r}),
            "PASS_s0_5cat": sum(1 for x in r if x["seed"] == "0" and str(x["PASS_vs_orig"]) == "True"),
            "n_s0_5cat": sum(1 for x in r if x["seed"] == "0"),
            "cat_s0_5cat": sum(1 for x in r if x["seed"] == "0" and str(x["catastrophic"]) == "True"),
        })
    return out


def rank_key(s: dict):
    return (0 if s["pareto_vs_B2"] else 1, -s["PASS"], s["catastrophic"],
            -s["worst_ddp"], s["mean_dR"], -s["mean_ddp"])


def main() -> None:
    ab = load(ANA / "AB_per_unit.csv")
    cc = load(ANA / "C_late_fusion_per_unit.csv")
    for r in cc:
        r["family"] = "C_late_fusion"
        r.setdefault("catastrophic", r.get("catastrophic_vs_orig"))
    for r in ab:
        r["candidate"] = r["candidate"].split("__")[0]      # E14_A3__bottle -> A3（候选级聚合）
    s_ab, s_c = summarize(ab), summarize(cc)
    fam = {}
    for r in s_ab + s_c:
        fam.setdefault(r["family"], []).append(r)
    (ANA / "family_leaderboard.json").write_text(json.dumps(
        {k: sorted(v, key=rank_key) for k, v in fam.items()}, indent=2, ensure_ascii=False))

    allrows = sorted(s_ab + s_c, key=rank_key)
    with open(ANA / "candidate_leaderboard.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(allrows[0].keys()))
        w.writeheader(); w.writerows(allrows)
    print("=" * 140)
    print("Exp14 — CANDIDATE LEADERBOARD（n_units = 已测单元数）")
    print("=" * 140)
    print("%-4s%-24s%-15s%6s%7s%5s%11s%10s%10s%12s%12s" % ("#", "candidate", "family", "units", "PASS",
                                                            "cat", "worst_ddp", "mean_dR", "mean_ddp",
                                                            "dR_vs_B2", "ddp_vs_B2"))
    for i, s in enumerate(allrows, 1):
        print("%-4d%-24s%-15s%6d%7d%5d%11.4f%10.4f%10.4f%12.4f%12.4f%s" % (
            i, s["candidate"], s["family"], s["n_units"], s["PASS"], s["catastrophic"],
            s["worst_ddp"], s["mean_dR"], s["mean_ddp"], s["mean_dR_vs_B2"], s["mean_ddp_vs_B2"],
            "  PARETO" if s["pareto_vs_B2"] else ""))
    print("\nStage D 门（>=4/5 PASS(seed0,5cats) 且 0 catastrophic）：")
    for s in allrows:
        gate = (s["n_s0_5cat"] >= 5 and s["PASS_s0_5cat"] >= 4 and s["cat_s0_5cat"] == 0)
        print("  %-24s seed0 %d/%d PASS, catastrophic %d -> %s" % (
            s["candidate"], s["PASS_s0_5cat"], s["n_s0_5cat"], s["cat_s0_5cat"],
            "ELIGIBLE for multi-seed" if gate else "not eligible"))
    print("\n基线：Adaptive B2（Exp11 冻结）= 12/15, catastrophic=1, worst Δd′=-0.2852"
          " | 其 seed0 5cat PASS = %d/5" % sum(
              1 for x in cc if False) if False else "")
    print("基线：Adaptive B2（Exp11 冻结）= 12/15, catastrophic=1, worst Δd′=-0.2852")
    json.dump({"all": allrows, "top4": allrows[:4], "top2": allrows[:2]},
              open(ANA / "tournament.json", "w"), indent=2, ensure_ascii=False)
    print("\nTop4:", [s["candidate"] for s in allrows[:4]])
    print("Top2:", [s["candidate"] for s in allrows[:2]])


if __name__ == "__main__":
    main()
