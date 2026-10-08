"""Experiment 14 — 依据锦标赛结果机械生成 Stage D / E / O 计划（写入 plan.json）。

Stage D: Top4 × 5 categories × seed0（仅补缺失单元）
Stage E: Top2 × (categories × seeds {1,2})
Stage O1: Top1 × seeds {3,4}（若时间允许）
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
E14 = ROOT / "results" / "experiment_14"
CATS = ["bottle", "cable", "hazelnut", "screw", "grid"]


def spec_of(cid: str, cat: str) -> str:
    """Family C 无 GPU spec（score 融合）；A/B 为 E14_<cid>__<cat>。"""
    return f"E14_{cid}__{cat}" if cid.startswith(("A", "B")) else cid


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--top4", nargs="+", required=True)
    ap.add_argument("--top2", nargs="+", required=True)
    ap.add_argument("--top1", nargs="+", required=True)
    a = ap.parse_args()
    plan = json.loads((E14 / "plan.json").read_text())
    d_units, e1, e2, o1 = [], [], [], []
    for cid in a.top4:
        if cid.startswith("C"):      # score 融合：5×3 已全有，无需 GPU
            continue
        for cat in CATS:
            d_units.append(f"{cat}:0:{spec_of(cid, cat)}")
    for cid in a.top2:
        if cid.startswith("C"):
            continue
        for sd in (1, 2):
            for cat in CATS:
                (e1 if sd == 1 else e2).append(f"{cat}:{sd}:{spec_of(cid, cat)}")
    for cid in a.top1:
        if cid.startswith("C"):
            continue
        for sd in (3, 4):
            for cat in CATS:
                o1.append(f"{cat}:{sd}:{spec_of(cid, cat)}")
    plan["d"] = {"round": "d", "units": sorted(set(d_units))}
    plan["e_seed1"] = {"round": "e_seed1", "units": sorted(set(e1))}
    plan["e_seed2"] = {"round": "e_seed2", "units": sorted(set(e2))}
    plan["o1_seed3"] = {"round": "o1", "units": sorted(set(o1))}
    (E14 / "plan.json").write_text(json.dumps(plan, indent=2))
    print("stage d: %d units | e_seed1: %d | e_seed2: %d | o1(seeds3,4): %d"
          % (len(set(d_units)), len(set(e1)), len(set(e2)), len(set(o1))))


if __name__ == "__main__":
    main()
