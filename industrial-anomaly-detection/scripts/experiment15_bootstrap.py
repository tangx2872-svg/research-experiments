"""Experiment 15 — bootstrap CI（CPU-only；§16-O2）。

重采样单元 = (category, seed) 评估对（paired design：所有候选在同一组单元上评估）。
输出：PASS 率、mean ΔR、mean Δd′、worst Δd′ 的 95% CI，以及与 Adaptive B2 的**配对**差异 CI。
注意：worst-case 是极值统计量，bootstrap 分布仅作描述（不当作平滑参数的 CI）。
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
E15 = ROOT / "results" / "experiment_15"
ANA = E15 / "analysis"
B = 10000
RNG = np.random.RandomState(0)


def load() -> dict:
    rows = list(csv.DictReader(open(ANA / "scoreside_per_unit.csv")))
    d = {}
    for r in rows:
        d.setdefault(r["method"], {})[(r["category"], int(r["seed"]))] = {
            "PASS": str(r["PASS"]) == "True", "cat": str(r["catastrophic"]) == "True",
            "dR": float(r["dR"]), "ddp": float(r["ddp"])}
    return d


def ci(x: np.ndarray) -> tuple:
    return float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))


def main() -> None:
    d = load()
    # Adaptive B2 参考（25 单元）—— 从 Exp14 的 O1 参考 + 15 单元重建
    b2 = {}
    for cat in ["bottle", "cable", "hazelnut", "screw", "grid"]:
        for seed in (3, 4):
            b2[(cat, seed)] = None
    ref = json.loads((ROOT / "results" / "experiment_14" / "analysis" / "O1_b2_reference.json").read_text())
    for r in ref["B2_seeds34"]["per_unit"]:
        b2[(r["category"], int(r["seed"]))] = {"PASS": bool(r["PASS"]), "cat": bool(r["cat"]),
                                               "dR": float(r["dR"]), "ddp": float(r["ddp"])}
    import experiment14_metrics as M
    mo = M.metrics_of({(c, s, "o"): M.source(c, s, "orig") for c in M.CATS for s in M.SEEDS})
    mb = M.metrics_of({(c, s, "b"): M.source(c, s, "b2") for c in M.CATS for s in M.SEEDS})
    for c in M.CATS:
        for s in M.SEEDS:
            v = M.verdict(mo[(c, s, "o")], mb[(c, s, "b")])
            b2[(c, s)] = {"PASS": v["PASS"], "cat": v["catastrophic"],
                          "dR": v["d_absz"], "ddp": v["d_dprime"]}
    # seeds 5,6：Exp15 o1ext 阶段新跑的 Original(E14_A0) 与 B2 参考(E14_A1，与 B2 bit-exact)
    E15R = ROOT / "results" / "experiment_15" / "raw"
    o56, b56 = {}, {}
    for c in M.CATS:
        for s in (5, 6):
            po = E15R / "o1ext_orig" / c / f"seed_{s}" / f"config_E14_A0__{c}"
            pb = E15R / "o1ext_b2" / c / f"seed_{s}" / f"config_E14_A1__{c}"
            if (po / "info.json").exists() and (pb / "info.json").exists():
                o56[(c, s, "o")] = po
                b56[(c, s, "b")] = pb
    if o56:
        m5o = M.metrics_of(o56)
        m5b = M.metrics_of(b56)
        for (c, s, _) in o56:
            v = M.verdict(m5o[(c, s, "o")], m5b[(c, s, "b")])
            b2[(c, s)] = {"PASS": v["PASS"], "cat": v["catastrophic"],
                          "dR": v["d_absz"], "ddp": v["d_dprime"]}
    b2 = {k: v for k, v in b2.items() if v is not None}

    out = {}
    keys = sorted(b2)
    for name, dd in [("Adaptive B2", b2)] + [(k, v) for k, v in d.items()]:
        common = [k for k in keys if k in dd and k in b2]
        if len(common) < 5:
            continue
        P = np.array([dd[k]["PASS"] for k in common], dtype=float)
        C = np.array([dd[k]["cat"] for k in common], dtype=float)
        R = np.array([dd[k]["dR"] for k in common])
        D = np.array([dd[k]["ddp"] for k in common])
        n = len(common)
        p_bs, r_bs, d_bs, w_bs = [], [], [], []
        for _ in range(B):
            idx = RNG.randint(0, n, n)
            p_bs.append(P[idx].mean()); r_bs.append(R[idx].mean())
            d_bs.append(D[idx].mean()); w_bs.append(D[idx].min())
        rec = {"n_units": n, "PASS": int(P.sum()),
               "pass_rate": round(float(P.mean()), 4),
               "pass_rate_CI": [round(x, 4) for x in ci(np.array(p_bs))],
               "catastrophic": int(C.sum()),
               "mean_dR": round(float(R.mean()), 4), "mean_dR_CI": [round(x, 4) for x in ci(np.array(r_bs))],
               "mean_ddp": round(float(D.mean()), 4), "mean_ddp_CI": [round(x, 4) for x in ci(np.array(d_bs))],
               "worst_ddp": round(float(D.min()), 4),
               "worst_ddp_boot_CI": [round(x, 4) for x in ci(np.array(w_bs))]}
        if name != "Adaptive B2":
            dP = np.array([P[i] - float(b2[k]["PASS"]) for i, k in enumerate(common)])
            dD = np.array([D[i] - b2[k]["ddp"] for i, k in enumerate(common)])
            dR2 = np.array([R[i] - b2[k]["dR"] for i, k in enumerate(common)])
            dp_bs, dd_bs, dr_bs = [], [], []
            for _ in range(B):
                idx = RNG.randint(0, n, n)
                dp_bs.append(dP[idx].mean()); dd_bs.append(dD[idx].mean()); dr_bs.append(dR2[idx].mean())
            rec["vs_B2_paired"] = {
                "d_pass_rate": round(float(dP.mean()), 4),
                "d_pass_rate_CI": [round(x, 4) for x in ci(np.array(dp_bs))],
                "d_mean_ddp": round(float(dD.mean()), 4),
                "d_mean_ddp_CI": [round(x, 4) for x in ci(np.array(dd_bs))],
                "d_mean_dR": round(float(dR2.mean()), 4),
                "d_mean_dR_CI": [round(x, 4) for x in ci(np.array(dr_bs))],
                "ddp_advantage_significant": bool(ci(np.array(dd_bs))[0] > 0),
                "robustness_cost_significant": bool(ci(np.array(dr_bs))[1] < 0)}
        out[name] = rec
    json.dump(out, open(ANA / "bootstrap_ci.json", "w"), indent=2, ensure_ascii=False)
    print("=" * 118)
    print("Exp15 bootstrap CI（20000? no: %d resamples；paired over %d common units）" % (B, len(keys)))
    print("=" * 118)
    print("%-14s%8s%9s%18s%11s%12s%12s" % ("method", "units", "PASS", "pass-rate 95%CI",
                                           "worst_ddp", "mean_ddp", "mean_dR"))
    for k, v in out.items():
        print("%-14s%8d%9s%18s%11.4f%12.4f%12.4f" % (
            k, v["n_units"], "%d/%d" % (v["PASS"], v["n_units"]),
            "[%.2f, %.2f]" % tuple(v["pass_rate_CI"]), v["worst_ddp"], v["mean_ddp"], v["mean_dR"]))
    print("\nvs Adaptive B2（paired）:")
    for k, v in out.items():
        if "vs_B2_paired" in v:
            p = v["vs_B2_paired"]
            print("  %-8s ΔPASS-rate %+.3f %-16s | Δmean d′ %+.4f %-18s | Δmean ΔR %+.4f %-18s %s"
                  % (k, p["d_pass_rate"], "[%.2f,%.2f]" % tuple(p["d_pass_rate_CI"]),
                     p["d_mean_ddp"], "[%.3f,%.3f]" % tuple(p["d_mean_ddp_CI"]),
                     p["d_mean_dR"], "[%.3f,%.3f]" % tuple(p["d_mean_dR_CI"]),
                     "PRESERVATION ADV (sig)" if p["ddp_advantage_significant"] else ""))


if __name__ == "__main__":
    main()
