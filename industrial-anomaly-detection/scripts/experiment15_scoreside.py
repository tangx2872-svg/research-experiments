"""Experiment 15 — score-side 组合锦标赛（**0 GPU**，CPU-only）。

覆盖：
  C1–C5   : 冻结 fusion weight grid {0.20,0.35,0.50,0.65,0.80}（§17：grid 已冻结，不再细化）
  C6      : max(z_o, z_b)
  X5      : category-adaptive w（冻结 normal-only 规则，见 config/candidate_registry.json）
  X6c     : mean(z_C2, z_C6)（两个已验证 fusion rule 的保守组合）
  X2b     : mean(z_o, z_b, z_A6)（三路；A6 仅在 seed0 有单元 → E1）
来源：seeds0-2 = Exp10 Original + Exp11 B2；seeds3,4 = Exp14 E14_A0/E14_A1；A6 = Exp14 raw a1/d。
"""
from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import experiment14_metrics as M  # noqa: E402

E14 = ROOT / "results" / "experiment_14"
E15 = ROOT / "results" / "experiment_15"
CATS = M.CATS
W = {"C1": 0.20, "C2": 0.35, "C3": 0.50, "C4": 0.65, "C5": 0.80}
FROZEN = json.loads((E15 / "config" / "candidate_registry.json").read_text())
X5W = FROZEN["x5_frozen_rule"]["w_per_category"]


def src(cat: str, seed: int, which: str):
    """which in {o,b,a6}；seeds0-2 = Exp10/11；seeds3,4 = Exp14 o1 阶段；seeds5,6 = Exp15 o1ext 阶段。"""
    if which == "a6":
        for rd in ("a1", "d"):
            p = E14 / "raw" / rd / cat / f"seed_{seed}" / f"config_E14_A6__{cat}"
            if (p / "info.json").exists():
                return p
        return None
    if seed in M.SEEDS:
        return M.source(cat, seed, "orig" if which == "o" else "b2")
    pref = "E14_A0__" if which == "o" else "E14_A1__"
    cands = ([("o1_orig_34", E14 / "raw"), ("o1ext_orig", E15 / "raw")] if which == "o"
             else [("o1_b2_34", E14 / "raw"), ("o1ext_b2", E15 / "raw")])
    for rd, root in cands:
        pp = root / rd / cat / f"seed_{seed}" / f"config_{pref}{cat}"
        if (pp / "info.json").exists():
            return pp
    return None


def _z(rowsd: dict):
    v = [s for k, s in rowsd.items() if k[0] == "clean_good" and k[1] == "none"]
    mu, sd = float(np.mean(v)), float(np.std(v, ddof=1))
    return {k: (s - mu) / sd for k, s in rowsd.items()}


def rule_value(cid: str, cat: str, zo: float, zb: float, za: float | None):
    if cid in W:
        return W[cid] * zo + (1 - W[cid]) * zb
    if cid == "C6":
        return max(zo, zb)
    if cid == "X5":
        w = X5W[cat]
        return w * zo + (1 - w) * zb
    if cid == "X6c":
        return 0.5 * (0.35 * zo + 0.65 * zb) + 0.5 * max(zo, zb)
    if cid == "X2b":
        assert za is not None
        return (zo + zb + za) / 3.0
    raise ValueError(cid)


def fused(cid: str, cat: str, seed: int):
    po, pb = src(cat, seed, "o"), src(cat, seed, "b")
    pa = src(cat, seed, "a6") if cid == "X2b" else None
    if cid == "X2b" and (pa is None or not pa.exists()):
        return None
    if po is None or pb is None or not po.exists() or not pb.exists():
        return None
    ro, rb = M.rows(po), M.rows(pb)
    keys = [k for k in ro if k in rb]
    if cid == "X2b":
        ra = M.rows(pa)
        keys = [k for k in keys if k in ra]
    if not keys:
        return None
    zo, zb = _z(ro), _z(rb)
    za = _z(M.rows(pa)) if cid == "X2b" else None
    sub = defaultdict(lambda: defaultdict(list))
    for k in keys:
        subset, sh, dt, _ = k
        sub[subset][sh].append((dt, float(rule_value(cid, cat, zo[k], zb[k], za[k] if za else None)), 0.0, 0.0))
    tau = max(s for _, s, _, _ in sub["val"]["none"]) if sub["val"]["none"] else float("nan")
    return {"sub": sub, "info": {"tau_val": float(tau), "alpha_l2": -1.0, "alpha_l3": -1.0,
                                 "pixel_auroc": float("nan"), "aupro": float("nan"),
                                 "runtime_seconds": 0.0, "coreset_size": -1,
                                 "peak_gpu_memory_allocated_mb": 0.0}}


def main() -> None:
    (E15 / "analysis").mkdir(parents=True, exist_ok=True)
    units, pairs = {}, []
    for cat in CATS:
        for seed in list(M.SEEDS) + [3, 4, 5, 6]:
            e = fused("C1", cat, seed)
            if e is not None:
                pairs.append((cat, seed))
    for cid in list(W) + ["C6", "X5", "X6c", "X2b"]:
        for cat, seed in pairs:
            e = fused(cid, cat, seed)
            if e is not None:
                units[(cat, seed, cid)] = e
    mt = M.metrics_of(units)
    base = M.metrics_of({(c, s, "orig"): src(c, s, "o") for c, s in pairs})

    rows = []
    for (cat, seed, cid), m in sorted(mt.items()):
        v = M.verdict(base[(cat, seed, "orig")], m)
        rows.append({"method": cid, "category": cat, "seed": seed,
                     "dR": round(v["d_absz"], 6), "ddp": round(v["d_dprime"], 6),
                     "PASS": v["PASS"], "catastrophic": v["catastrophic"],
                     "absz": round(m["mean_abs_delta_z"], 6), "dprime": round(m["mean_dprime"], 6)})
    with open(E15 / "analysis" / "scoreside_per_unit.csv", "w", newline="") as f:
        wtr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wtr.writeheader(); wtr.writerows(rows)

    summ = []
    for cid in sorted({r["method"] for r in rows}):
        r = [x for x in rows if x["method"] == cid]
        n = len(r)
        summ.append({"method": cid, "n_units": n,
                     "PASS": sum(1 for x in r if x["PASS"]),
                     "PASS15": sum(1 for x in r if x["PASS"] and x["seed"] in (0, 1, 2)),
                     "catastrophic": sum(1 for x in r if x["catastrophic"]),
                     "worst_ddp": round(min(x["ddp"] for x in r), 4),
                     "mean_dR": round(float(np.mean([x["dR"] for x in r])), 4),
                     "mean_ddp": round(float(np.mean([x["ddp"] for x in r])), 4),
                     "n_seeds": len(sorted({x["seed"] for x in r})),
                     "evidence": ("E3+" if n >= 35 else "E3" if n >= 25 else "E2" if n >= 15 else "E1")})
    summ.sort(key=lambda s: (-s["PASS"], s["catastrophic"], -s["worst_ddp"]))
    with open(E15 / "analysis" / "scoreside_summary.csv", "w", newline="") as f:
        wtr = csv.DictWriter(f, fieldnames=list(summ[0].keys()))
        wtr.writeheader(); wtr.writerows(summ)

    # leaderboard.csv（供 launcher 显示 CURRENT LEADER）
    lb = [{"method": s["method"], "evidence": s["evidence"], "PASS": s["PASS"],
           "catastrophic": s["catastrophic"], "worst_ddp": s["worst_ddp"],
           "n_units": s["n_units"],
           "note": "score-side (0 GPU)"} for s in summ]
    with open(E15 / "analysis" / "leaderboard.csv", "w", newline="") as f:
        wtr = csv.DictWriter(f, fieldnames=list(lb[0].keys()))
        wtr.writeheader(); wtr.writerows(lb)

    # 参数敏感度曲线（w grid，25 单元）
    (E15 / "analysis" / "scoreside_per_seed.json").write_text(json.dumps(
        {cid: {str(sd): sum(1 for x in rows if x["method"] == cid and x["seed"] == sd and x["PASS"]) /
                     max(1, sum(1 for x in rows if x["method"] == cid and x["seed"] == sd))
               for sd in sorted({x["seed"] for x in rows if x["method"] == cid})}
         for cid in sorted({r["method"] for r in rows})}, indent=2))
    sens = [{"w_orig": W[c], "PASS": s["PASS"], "catastrophic": s["catastrophic"],
             "worst_ddp": s["worst_ddp"], "mean_dR": s["mean_dR"], "n_units": s["n_units"]}
            for c in ("C1", "C2", "C3", "C4", "C5") for s in summ if s["method"] == c]
    sens.sort(key=lambda x: x["w_orig"])
    json.dump(sens, open(E15 / "analysis" / "sensitivity_curve.json", "w"), indent=2)

    print("=" * 116)
    print("Exp15 score-side tournament（0 GPU）")
    print("=" * 116)
    print("%-6s%8s%8s%14s%11s%11s%10s" % ("method", "units", "PASS", "catastrophic", "worst_ddp", "mean_dR", "evid"))
    for s in summ:
        print("%-6s%8d%8d%14d%11.4f%11.4f%10s" % (s["method"], s["n_units"], s["PASS"],
                                                  s["catastrophic"], s["worst_ddp"], s["mean_dR"], s["evidence"]))
    print("\n参数敏感度（w_orig, %d 单元）：" % len(pairs))
    for x in sens:
        print("   w=%.2f  PASS %2d/%d  catastrophic %d  worst %+.4f  meanΔR %+.4f"
              % (x["w_orig"], x["PASS"], x["n_units"], x["catastrophic"], x["worst_ddp"], x["mean_dR"]))


if __name__ == "__main__":
    main()
