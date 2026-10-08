"""Experiment 15 — 新增候选 spec（仅两个层强度组合；其余全部复用 Exp14 冻结结果）。

X4a: L2 residual(beta_c)   + L3 alpha_in(0.125)   两个强度各自在 Exp14 独立验证（L2 来自 B2；0.125 来自 A3）
X4b: L2 residual(0.5beta_c)+ L3 alpha_in(0.25)    两个强度各自在 Exp14 独立验证（0.5beta_c 来自 A3；L3 来自 B2）
不新增超参数、不新增 fusion weight、不做连续 grid。
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import experiment11_candidates as c11  # noqa: E402
import experiment14_candidates as c14  # noqa: E402

CATS = ["bottle", "cable", "hazelnut", "screw", "grid"]
BETA = {c: float(c11.beta_table()[("E11_B2", c)]) for c in CATS}


def build() -> dict:
    out = {}
    for cat in CATS:
        out[f"E15_X4a__{cat}"] = {"l2": c14._resid(BETA[cat]), "l3": c14._alpha(0.125)}
        out[f"E15_X4b__{cat}"] = {"l2": c14._resid(0.5 * BETA[cat]), "l3": c14._alpha(0.25)}
    return out


ALL15_SPECS = build()


def register_into_env14() -> None:
    c14.ALL14_SPECS.update(ALL15_SPECS)


def verify() -> tuple:
    bad = [n for n, sp in ALL15_SPECS.items()
           if sp["l2"]["kind"] not in c14.FROZEN_KINDS or sp["l3"]["kind"] not in c14.FROZEN_KINDS]
    return (len(bad) == 0, f"n_specs={len(ALL15_SPECS)}; kinds frozen; layer strengths reused from Exp14; "
                           f"beta_c={ {k: round(v,4) for k,v in BETA.items()} }; violations={bad}")
