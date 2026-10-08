"""Experiment 14 — Family C: Late / Score-level dual fusion（**0 GPU**；CPU-only）。

来源（严格复用，冻结）：
  Original  = Exp10 `P10_ORIG_a000`（corrected-189 + strict-V2）
  Robust    = Exp11 `E11_B2`（Adaptive B2，同一协议）
Calibration（冻结，normal-only）：z = (score - mu_g)/sd_g，mu_g/sd_g 取自**该路自己的 clean_good/none**。
  C1..C5: w*z_orig + (1-w)*z_B2 ; C6: max(z_orig, z_B2)（同量纲保守取大，预注册保留）。
tau 冻结规则沿用 runner：tau = max(融合 score over subset='val')。
pixel AUROC / AUPRO 不可由 score 融合重算（需要 pixel map）→ 标记 NA，不参与判据。
"""
from __future__ import annotations

import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import experiment14_metrics as M  # noqa: E402

OUT = ROOT / "results" / "experiment_14" / "analysis"
W_LIST = {"C1": 0.20, "C2": 0.35, "C3": 0.50, "C4": 0.65, "C5": 0.80}


def _fuse(cat: str, seed: int, w, mode: str):
    po, pb = M.source(cat, seed, "orig"), M.source(cat, seed, "b2")
    if po is None or pb is None:
        return None
    ro, rb = M.rows(po), M.rows(pb)
    keys = [k for k in ro if k in rb]
    if not keys:
        return None

    def _stats(rowsd):
        v = [s for k, s in rowsd.items() if k[0] == "clean_good" and k[1] == "none"]
        return float(np.mean(v)), float(np.std(v, ddof=1))

    mu_o, sd_o = _stats(ro)
    mu_b, sd_b = _stats(rb)
    sub = defaultdict(lambda: defaultdict(list))
    for k in keys:
        subset, shift, dt, _ = k
        zo = (ro[k] - mu_o) / sd_o
        zb = (rb[k] - mu_b) / sd_b
        val = max(zo, zb) if mode == "max" else w * zo + (1.0 - w) * zb
        sub[subset][shift].append((dt, float(val), 0.0, 0.0))
    tau = max(s for _, s, _, _ in sub["val"]["none"]) if sub["val"]["none"] else float("nan")
    info = {"tau_val": float(tau), "alpha_l2": -1.0, "alpha_l3": -1.0,
            "pixel_auroc": float("nan"), "aupro": float("nan"), "runtime_seconds": 0.0,
            "coreset_size": -1, "peak_gpu_memory_allocated_mb": 0.0}
    return {"sub": sub, "info": info}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    cands = [(k, W_LIST[k], "lin") for k in W_LIST] + [("C6", None, "max")]
    units = {}
    for cat in M.CATS:
        for seed in M.SEEDS:
            for cid, w, mode in cands:
                e = _fuse(cat, seed, w, mode)
                if e is not None:
                    units[(cat, seed, cid)] = e
    mt = M.metrics_of(units)
    base = M.metrics_of({(c, s, "orig"): M.source(c, s, "orig") for c in M.CATS for s in M.SEEDS})
    b2 = M.metrics_of({(c, s, "b2"): M.source(c, s, "b2") for c in M.CATS for s in M.SEEDS})

    rows = []
    for (cat, seed, cid), m in sorted(mt.items()):
        v_o = M.verdict(base[(cat, seed, "orig")], m)
        v_b = M.verdict(b2[(cat, seed, "b2")], m)
        rows.append({"candidate": cid, "category": cat, "seed": seed,
                     "absz": round(m["mean_abs_delta_z"], 6), "dprime": round(m["mean_dprime"], 6),
                     "tau": round(m["tau_val"], 6), "image_auroc": round(m["image_auroc"], 6),
                     "pixel_auroc": "NA", "aupro": "NA",
                     "dR_vs_orig": round(v_o["d_absz"], 6), "ddp_vs_orig": round(v_o["d_dprime"], 6),
                     "PASS_vs_orig": v_o["PASS"], "catastrophic_vs_orig": v_o["catastrophic"],
                     "dR_vs_B2": round(v_b["d_absz"], 6), "ddp_vs_B2": round(v_b["d_dprime"], 6),
                     "PASS_vs_B2": v_b["PASS"]})
    with open(OUT / "C_late_fusion_per_unit.csv", "w", newline="") as f:
        wtr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wtr.writeheader(); wtr.writerows(rows)
    summ = []
    for cid, w, mode in cands:
        r = [x for x in rows if x["candidate"] == cid]
        r1 = [x for x in r if x["seed"] == 0 and x["category"] in ("bottle", "cable", "hazelnut")]
        summ.append({"candidate": cid,
                     "form": "max(z_orig,z_B2)" if mode == "max" else f"{w}*z_orig+{1-w:.2f}*z_B2",
                     "PASS_s0_3cat": sum(1 for x in r1 if x["PASS_vs_orig"]),
                     "PASS/15": sum(1 for x in r if x["PASS_vs_orig"]),
                     "catastrophic/15": sum(1 for x in r if x["catastrophic_vs_orig"]),
                     "worst_ddp": round(min(x["ddp_vs_orig"] for x in r), 4),
                     "mean_dR": round(float(np.mean([x["dR_vs_orig"] for x in r])), 4)})
    with open(OUT / "C_late_fusion_summary.csv", "w", newline="") as f:
        wtr = csv.DictWriter(f, fieldnames=list(summ[0].keys()))
        wtr.writeheader(); wtr.writerows(summ)
    print("=" * 104)
    print("Family C — Late/Score Fusion (0 GPU; Exp10 Original + Exp11 B2)")
    print("=" * 104)
    print("%-6s%-26s%12s%9s%14s%11s%10s" % ("cand", "form", "PASS_s0/3", "PASS/15", "catastrophic/15",
                                            "worst_ddp", "mean_dR"))
    for s in summ:
        print("%-6s%-26s%12d%9d%14d%11.4f%10.4f" % (s["candidate"], s["form"], s["PASS_s0_3cat"],
                                                    s["PASS/15"], s["catastrophic/15"],
                                                    s["worst_ddp"], s["mean_dR"]))


if __name__ == "__main__":
    main()
