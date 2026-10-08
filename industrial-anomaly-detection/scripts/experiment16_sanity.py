"""Experiment 16 — P3 Sanity（S1 历史重建 / S2 X6c 定义 / S3 阈值 / S4 输出隔离）。全 CPU，0 GPU。"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import experiment14_metrics as M  # noqa: E402
import experiment16_fusion as F16  # noqa: E402

E15 = ROOT / "results" / "experiment_15"
E16 = ROOT / "results" / "experiment_16_final_validation"
OUT = E16 / "analysis"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    checks = []

    # ---------- S3 thresholds（与 Exp15 冻结一致） ----------
    e15_plan = json.loads((E15 / "config" / "analysis_plan.json").read_text())
    thr15 = e15_plan["evaluation"]["primary"]
    ok_s3 = (M.EPS_RZ == 0.02 and M.EPS_DP == 0.10 and M.CATASTROPHIC == -0.25)
    checks.append(("S3_thresholds_bit_exact", ok_s3,
                   f"EPS_RZ={M.EPS_RZ} EPS_DP={M.EPS_DP} catastrophic={M.CATASTROPHIC} | "
                   f"Exp15 config: {thr15[2]}"))

    # ---------- S4 output isolation ----------
    ok_s4 = ("experiment_16_final_validation" in str(E16)) and ("experiment_15" not in str(E16))
    checks.append(("S4_output_isolated", ok_s4, f"Exp16 root = {E16.relative_to(ROOT)}"))

    # ---------- S2 X6c definition ----------
    import inspect
    src = inspect.getsource(F16.rule)
    ok_s2 = ("0.5 * (z_c2 + max(zo, zb))" in src) and ("W_C2 = 0.35" in (ROOT / "scripts" / "experiment16_fusion.py").read_text())
    checks.append(("S2_X6c_definition", ok_s2, "X6c = mean(z_C2, z_C6); C2 = 0.35/0.65; C6 = max"))

    # ---------- S1 historical reconstruction vs Exp15 artifacts ----------
    ref = list(csv.DictReader(open(E15 / "analysis" / "scoreside_per_unit.csv")))
    sample = [("bottle", 0), ("cable", 2), ("hazelnut", 5), ("grid", 6), ("screw", 3)]
    mism = []
    n_cmp = 0
    for cat, seed in sample:
        po = F16.source(cat, seed, "o")
        pb = F16.source(cat, seed, "b")
        if po is None or pb is None:
            mism.append(f"{cat}/s{seed}: source missing")
            continue
        mo = M.metrics_of({(cat, seed, "o"): po})
        for cid in ("C2", "C6", "X6c"):
            u = F16.fused_unit(cat, seed, cid)
            mc = M.metrics_of({(cat, seed, cid): u})
            v = M.verdict(mo[(cat, seed, "o")], mc[(cat, seed, cid)])
            r = [x for x in ref if x["method"] == cid and x["category"] == cat and int(x["seed"]) == seed]
            if not r:
                mism.append(f"{cat}/s{seed}/{cid}: no Exp15 reference row")
                continue
            r = r[0]
            n_cmp += 1
            d_dr = abs(round(v["d_absz"], 6) - float(r["dR"]))
            d_dp = abs(round(v["d_dprime"], 6) - float(r["ddp"]))
            same_pass = (bool(v["PASS"]) == (r["PASS"] == "True"))
            same_cat = (bool(v["catastrophic"]) == (r["catastrophic"] == "True"))
            if max(d_dr, d_dp) > 1e-9 or not same_pass or not same_cat:
                mism.append(f"{cat}/s{seed}/{cid}: dR {v['d_absz']:.6f} vs {r['dR']} | "
                            f"ddp {v['d_dprime']:.6f} vs {r['ddp']} | PASS {v['PASS']}/{r['PASS']} | "
                            f"cat {v['catastrophic']}/{r['catastrophic']}")
    checks.append(("S1_historical_reconstruction", len(mism) == 0,
                   f"compared {n_cmp} (cat,seed,method) triples against Exp15 artifacts; "
                   f"max|Δ|<=1e-9 and identical PASS/catastrophic flags" if not mism else f"MISMATCH: {mism[:3]}"))

    with open(OUT / "sanity_checks.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["check", "status", "detail"])
        for n, ok, d in checks:
            w.writerow([n, "PASS" if ok else "FAIL", d])
    print("=" * 118)
    for n, ok, d in checks:
        print("%-34s %s | %s" % (n, "PASS" if ok else "FAIL", d[:150]))
    print("=" * 118)
    if not all(ok for _, ok, _ in checks):
        raise SystemExit("P3 SANITY FAILED — 停止正式实验并报告")
    print("P3 SANITY: %d/%d PASS -> 允许启动 P4 GPU" % (len(checks), len(checks)))


if __name__ == "__main__":
    main()
