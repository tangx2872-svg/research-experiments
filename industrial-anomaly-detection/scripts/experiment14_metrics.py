"""Experiment 14 — 统一指标/判据层（Family A/B/C 共用；零协议漂移）。

PASS（单 category-seed，冻结）：Δ|Δz| <= -EPS_RZ(0.02) 且 Δd' >= -EPS_DP(0.10)，相对同 (cat,seed) 的 corrected Original。
catastrophic（冻结）：Δd' <= -0.25。
所有指标由 experiment5a_h_analysis.unit_metrics 计算（与 5A-H/7A-O/9A/9B/10/11 同一实现）。
"""
from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import experiment5a_h_analysis as h5  # noqa: E402

E14 = ROOT / "results" / "experiment_14"
E10 = ROOT / "results" / "experiment_10_preservation_recovery" / "raw"
E11 = ROOT / "results" / "experiment_11_c2_adaptive" / "raw"
CATS = ["bottle", "cable", "hazelnut", "screw", "grid"]
SEEDS = [0, 1, 2]
EPS_RZ, EPS_DP, CATASTROPHIC = 0.02, 0.10, -0.25


def ent(path: Path) -> dict:
    info = json.loads((path / "info.json").read_text())
    sub = defaultdict(lambda: defaultdict(list))
    for r in csv.DictReader(open(path / "per_image.csv")):
        sub[r["subset"]][r["shift"]].append((r["defect_type"], float(r["score"]),
                                             float(r["clipped_high_ratio"]), 0.0))
    return {"sub": sub, "info": info}


def rows(path: Path) -> dict:
    return {(r["subset"], r["shift"], r["defect_type"], Path(r["image_path"]).name): float(r["score"])
            for r in csv.DictReader(open(path / "per_image.csv"))}


def find(roots, cat: str, seed: int, cfg: str):
    for root in roots:
        cands = sorted(Path(root).glob(f"*/{cat}/seed_{seed}/config_{cfg}"))
        if cands:
            return cands[0]
    return None


def source(cat: str, seed: int, method: str):
    if method == "orig":
        return find([E10], cat, seed, "P10_ORIG_a000")
    return find([E11], cat, seed, "E11_B2__" + cat)


def metrics_of(units: dict) -> dict:
    """值可以是 unit 目录（Path）或已构造好的 {"sub","info"}（Family C 融合后）。"""
    return h5.unit_metrics({k: (v if isinstance(v, dict) else ent(v)) for k, v in units.items()})


def verdict(base: dict, cand: dict) -> dict:
    drz = cand["mean_abs_delta_z"] - base["mean_abs_delta_z"]
    ddp = cand["mean_dprime"] - base["mean_dprime"]
    return {"d_absz": drz, "d_dprime": ddp,
            "PASS": bool(drz <= -EPS_RZ and ddp >= -EPS_DP),
            "catastrophic": bool(ddp <= CATASTROPHIC)}
