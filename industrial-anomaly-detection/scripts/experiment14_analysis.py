"""Experiment 14 — Family A/B 统一分析（GPU 单元）：指标 → PASS/catastrophic → 排名。

用法：python -u scripts/experiment14_analysis.py [--rounds a1 b1 ...] [--out analysis/AB_per_unit.csv]
基准（冻结）：同 (cat,seed) 的 corrected Original（Exp10 P10_ORIG_a000）；同时给出 vs Adaptive B2（Exp11 E11_B2）。
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import experiment14_metrics as M  # noqa: E402

E14 = ROOT / "results" / "experiment_14"


def discover(rounds) -> dict:
    """-> {(cat,seed,name): dir}（只取 info.json status OK）"""
    out = {}
    for rd in rounds:
        for ip in sorted((E14 / "raw" / rd).glob("*/*/config_*/info.json")):
            d = ip.parent
            cfg = d.name.replace("config_", "")
            seed = d.parent.name.replace("seed_", "")
            cat = d.parent.parent.name
            try:
                st = str(json.loads(ip.read_text()).get("status", ""))
            except Exception:
                continue
            if not st.startswith("OK"):
                continue
            out[(cat, int(seed), cfg)] = d
    return out


def family_of(name: str) -> str:
    if name.startswith("E14_A"):
        return "A_dual_path" if name[5] in "234" else "A_layerwise" if name[5] in "56" else "A"
    if name.startswith("E14_B"):
        return "B_tiny_inss"
    return "other"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", nargs="+", default=["a1", "b1"])
    ap.add_argument("--out", default="AB_per_unit.csv")
    a = ap.parse_args()
    units = discover(a.rounds)
    if not units:
        raise SystemExit("no finished units found")
    mt = M.metrics_of({k: v for k, v in units.items()})
    need = sorted({(c, s) for c, s, _ in units})
    base = {}
    for c, s in need:
        po, pb = M.source(c, s, "orig"), M.source(c, s, "b2")
        if po is not None:
            base[(c, s, "orig")] = M.ent(po)
        if pb is not None:
            base[(c, s, "b2")] = M.ent(pb)
    bm = M.metrics_of({k: v for k, v in base.items()})

    rows = []
    for (cat, seed, cfg), m in sorted(mt.items()):
        vo = M.verdict(bm[(cat, seed, "orig")], m) if (cat, seed, "orig") in bm else {}
        vb = M.verdict(bm[(cat, seed, "b2")], m) if (cat, seed, "b2") in bm else {}
        rows.append({"family": family_of(cfg), "candidate": cfg, "category": cat, "seed": seed,
                     "absz": round(m["mean_abs_delta_z"], 6), "dprime": round(m["mean_dprime"], 6),
                     "image_auroc": round(m["image_auroc"], 6),
                     "pixel_auroc": round(m.get("pixel_auroc", float("nan")), 6),
                     "aupro": round(m.get("aupro", float("nan")), 6),
                     "dR_vs_orig": round(vo.get("d_absz", float("nan")), 6),
                     "ddp_vs_orig": round(vo.get("d_dprime", float("nan")), 6),
                     "PASS_vs_orig": vo.get("PASS"), "catastrophic": vo.get("catastrophic"),
                     "dR_vs_B2": round(vb.get("d_absz", float("nan")), 6),
                     "ddp_vs_B2": round(vb.get("d_dprime", float("nan")), 6),
                     "PASS_vs_B2": vb.get("PASS")})
    (E14 / "analysis").mkdir(parents=True, exist_ok=True)
    with open(E14 / "analysis" / a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print("wrote", E14 / "analysis" / a.out, len(rows), "rows")

    # 汇总（R1 = 三类别 seed0；全部 = 所有已完成单元）
    print("=" * 122)
    print("Family A/B — per-candidate summary（基准 = 同 (cat,seed) corrected Original）")
    print("=" * 122)
    print("%-22s%-16s%9s%9s%14s%11s%11s%11s" % ("candidate", "family", "R1_PASS/3", "PASS/N",
                                                "catastrophic", "worst_ddp", "mean_dR", "mean_ddp"))
    for cfg in sorted({r["candidate"] for r in rows}):
        r = [x for x in rows if x["candidate"] == cfg]
        r1 = [x for x in r if x["seed"] == 0 and x["category"] in ("bottle", "cable", "hazelnut")]
        print("%-22s%-16s%9s%9s%14d%11.4f%11.4f%11.4f" % (
            cfg, family_of(cfg), f"{sum(1 for x in r1 if x['PASS_vs_orig'])}/{len(r1)}",
            f"{sum(1 for x in r if x['PASS_vs_orig'])}/{len(r)}",
            sum(1 for x in r if x["catastrophic"]),
            min(x["ddp_vs_orig"] for x in r), float(np.mean([x["dR_vs_orig"] for x in r])),
            float(np.mean([x["ddp_vs_orig"] for x in r]))))
    print("\nbaseline 参考：Adaptive B2 = 12/15（Exp11 冻结，catastrophic=1，worst Δd′=-0.2852）")
    print("per-category 明细：")
    for cfg in sorted({r["candidate"] for r in rows}):
        det = " | ".join(f"{x['category']}/s{x['seed']}:" +
                         ("P" if x["PASS_vs_orig"] else ("C" if x["catastrophic"] else "F")) +
                         f"(dR{x['dR_vs_orig']:+.3f},ddp{x['ddp_vs_orig']:+.3f})"
                         for x in sorted([y for y in rows if y["candidate"] == cfg],
                                         key=lambda z: (z["category"], z["seed"])))
        print(f"  {cfg}: {det}")


if __name__ == "__main__":
    main()
