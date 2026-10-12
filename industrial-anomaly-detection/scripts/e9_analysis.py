#!/usr/bin/env python
"""E9 - Round 1 ranking and report."""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np  # noqa: E402
import e9_common as E9C  # noqa: E402

OUT = ROOT / "results" / "e9"
RAW = OUT / "raw"
CATS = ["Bird", "Car", "Cube"]
ORDER = ["A0", "I01", "I02", "I03", "I04", "F01", "A1"]


def load_round1() -> dict:
    out = {}
    for cid in ORDER:
        for cat in CATS:
            p = RAW / f"{cid}_{cat}_seed0.json"
            if not p.exists():
                continue
            d = json.loads(p.read_text())
            if d.get("status") == "OK":
                out[(cid, cat)] = d
    return out


def main() -> int:
    data = load_round1()
    if not data:
        print("[E9] no round1 results yet")
        return 1

    rows = []
    agg = {}
    for cid in ORDER:
        cats = [c for c in CATS if (cid, c) in data]
        if not cats:
            continue
        per_cat = {}
        das, dds = [], []
        for cat in cats:
            d = data[(cid, cat)]
            da = d["d_auroc"]
            dd = d["d_dprime"]
            per_cat[cat] = da
            das.append(da)
            dds.append(dd)
        agg[cid] = {
            "mean_d_auroc": float(np.mean(das)),
            "median_d_auroc": float(np.median(das)),
            "mean_d_dprime": float(np.mean(dds)),
            "positive_categories": sum(1 for x in das if x > 0),
            "n_categories": len(cats),
            "worst_category": min(per_cat, key=per_cat.get),
            "worst_d": min(per_cat.values()),
            "per_category": per_cat,
        }
        rows.append({"candidate": cid, "name": E9C.CANDIDATES[cid]["name"],
                     "family": E9C.CANDIDATES[cid]["family"]})

    # ranking: prefer >=2/3 positive AND mean>0; then by mean
    def score(cid):
        e = agg[cid]
        return (e["positive_categories"] >= 2 and e["mean_d_auroc"] > 0, e["mean_d_auroc"])

    ranked = sorted(agg, key=lambda c: score(c), reverse=True)

    # write round1.csv
    with open(OUT / "round1.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["candidate", "name", "family", "mean_d_auroc", "median_d_auroc",
                    "mean_d_dprime", "positive_categories", "worst_category", "worst_d",
                    "Bird_dA", "Car_dA", "Cube_dA"])
        for cid in ranked:
            e = agg[cid]
            spec = E9C.CANDIDATES[cid]
            w.writerow([cid, spec["name"], spec["family"],
                        f"{e['mean_d_auroc']:+.4f}", f"{e['median_d_auroc']:+.4f}",
                        f"{e['mean_d_dprime']:+.4f}",
                        f"{e['positive_categories']}/{e['n_categories']}",
                        e["worst_category"], f"{e['worst_d']:+.4f}",
                        f"{e['per_category'].get('Bird', float('nan')):+.4f}",
                        f"{e['per_category'].get('Car', float('nan')):+.4f}",
                        f"{e['per_category'].get('Cube', float('nan')):+.4f}"])

    print("\n===== E9 ROUND-1 RANKING =====", flush=True)
    survivors = []
    for r, cid in enumerate(ranked, 1):
        e = agg[cid]
        spec = E9C.CANDIDATES[cid]
        verdict = ("ADVANCE" if e["positive_categories"] >= 2 and e["mean_d_auroc"] > 0
                   else ("HOLD" if e["mean_d_auroc"] > 0 else "STOP"))
        if verdict != "STOP":
            survivors.append(cid)
        print(f"{r}. {cid:4s} {spec['name']:34s} mean_dA={e['mean_d_auroc']:+.4f} "
              f"dd'={e['mean_d_dprime']:+.4f} pos={e['positive_categories']}/{e['n_categories']} "
              f"worst={e['worst_category']}({e['worst_d']:+.4f}) -> {verdict}", flush=True)

    n_survivor = sum(1 for cid in agg
                     if agg[cid]["positive_categories"] >= 2 and agg[cid]["mean_d_auroc"] > 0)
    json.dump({"ranking": ranked, "aggregates": agg, "survivors": survivors,
               "n_survivor": n_survivor}, open(OUT / "round1_ranking.json", "w"), indent=2)
    print(f"\n[E9] survivors (ADVANCE): {survivors if survivors else 'NO SURVIVOR'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
