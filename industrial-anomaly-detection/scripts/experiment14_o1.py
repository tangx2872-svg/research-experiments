"""Experiment 14 — Optional O1: Top1 (C1) 在 seeds {3,4} 的稳定性确认（0 额外训练逻辑，纯 score 融合）。

来源（Exp14 本阶段新跑）：
  Original = raw/o1_orig_34/<cat>/seed_<s>/config_P10_ORIG_a000
  B2       = raw/o1_b2_34/<cat>/seed_<s>/config_E14_A1__<cat>   （A1 == B2 算子，已在 seeds0-2 与 Exp11 bit-exact 核验）
C1/C2 定义与冻结一致：z 各自 normal(clean_good) 标准化后 w*z_orig + (1-w)*z_B2。
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
ANA = E14 / "analysis"
W = {"C1": 0.20, "C2": 0.35, "C3": 0.50, "C4": 0.65, "C5": 0.80}


def _unit(cat: str, seed: int, which: str):
    if which == "orig":
        p = E14 / "raw" / "o1_orig_34" / cat / f"seed_{seed}" / f"config_E14_A0__{cat}"
    else:
        p = E14 / "raw" / "o1_b2_34" / cat / f"seed_{seed}" / f"config_E14_A1__{cat}"
    return p if (p / "info.json").exists() else None


def _fuse(cat: str, seed: int, w: float):
    po, pb = _unit(cat, seed, "orig"), _unit(cat, seed, "b2")
    if po is None or pb is None:
        return None
    ro, rb = M.rows(po), M.rows(pb)
    keys = [k for k in ro if k in rb]
    if not keys:
        return None

    def _st(rd):
        v = [s for k, s in rd.items() if k[0] == "clean_good" and k[1] == "none"]
        return float(np.mean(v)), float(np.std(v, ddof=1))
    mu_o, sd_o = _st(ro)
    mu_b, sd_b = _st(rb)
    sub = defaultdict(lambda: defaultdict(list))
    for k in keys:
        subset, shift, dt, _ = k
        val = w * (ro[k] - mu_o) / sd_o + (1.0 - w) * (rb[k] - mu_b) / sd_b
        sub[subset][shift].append((dt, float(val), 0.0, 0.0))
    tau = max(s for _, s, _, _ in sub["val"]["none"])
    return {"sub": sub, "info": {"tau_val": float(tau), "alpha_l2": -1.0, "alpha_l3": -1.0,
                                 "pixel_auroc": float("nan"), "aupro": float("nan"),
                                 "runtime_seconds": 0.0, "coreset_size": -1,
                                 "peak_gpu_memory_allocated_mb": 0.0}}


def main() -> None:
    rows = []
    for seed in (3, 4):
        o_units = {c: _unit(c, seed, "orig") for c in M.CATS}
        if any(v is None for v in o_units.values()):
            print(f"  seed {seed}: Original 单元不完整，跳过"); continue
        mo = M.metrics_of({(c, seed, "orig"): o_units[c] for c in M.CATS})
        for cid, w in W.items():
            fu = {c: _fuse(c, seed, w) for c in M.CATS}
            if any(v is None for v in fu.values()):
                continue
            mf = M.metrics_of({(c, seed, cid): fu[c] for c in M.CATS})
            for c in M.CATS:
                v = M.verdict(mo[(c, seed, "orig")], mf[(c, seed, cid)])
                rows.append({"candidate": cid, "category": c, "seed": seed,
                             "absz": round(mf[(c, seed, cid)]["mean_abs_delta_z"], 6),
                             "dprime": round(mf[(c, seed, cid)]["mean_dprime"], 6),
                             "dR_vs_orig": round(v["d_absz"], 6),
                             "ddp_vs_orig": round(v["d_dprime"], 6),
                             "PASS_vs_orig": v["PASS"], "catastrophic": v["catastrophic"],
                             "pixel_auroc": "NA", "aupro": "NA"})
    if not rows:
        print("no O1 data yet"); return
    with open(ANA / "O1_seeds34_per_unit.csv", "w", newline="") as f:
        wtr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wtr.writeheader(); wtr.writerows(rows)
    print("=" * 96)
    print("Optional O1 — Top1/Top2 在 seeds {3,4}（5 cats × 2 seeds = 10 单元/candidate）")
    print("=" * 96)
    print("%-6s%6s%8s%7s%12s%12s" % ("cand", "units", "PASS", "cat", "worst_ddp", "mean_dR"))
    for cid in sorted({r["candidate"] for r in rows}):
        r = [x for x in rows if x["candidate"] == cid]
        print("%-6s%6d%8d%7d%12.4f%12.4f" % (cid, len(r),
                                             sum(1 for x in r if x["PASS_vs_orig"]),
                                             sum(1 for x in r if x["catastrophic"]),
                                             min(x["ddp_vs_orig"] for x in r),
                                             float(np.mean([x["dR_vs_orig"] for x in r]))))
        det = " | ".join(f"{x['category']}/s{x['seed']}:" +
                         ("P" if x["PASS_vs_orig"] else ("C" if x["catastrophic"] else "F"))
                         for x in sorted(r, key=lambda z: (z["category"], z["seed"])))
        print("        " + det)
    print("=" * 96)


if __name__ == "__main__":
    main()
