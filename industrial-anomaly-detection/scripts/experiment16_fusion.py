"""Experiment 16 — 冻结融合规则与统一评估（**定义必须与 Exp14/15 bit-exact 一致**）。

方法定义（从 Exp14/Exp15 原始冻结实现逐字复用）：
  Original        = PatchCore α=0（Exp14 `E14_A0` = l2/l3 alpha_in(0) → 严格 short-circuit）
  Adaptive B2     = L2 residual(β_c) + L3 alpha_in(0.25)（Exp14 `E14_A1`，与 Exp11 `E11_B2` bit-exact）
  C2              = 0.35*z_orig + 0.65*z_B2
  C6              = max(z_orig, z_B2)
  X6c             = mean(z_C2, z_C6)
z 校准（冻结）：z = (score - mu_g)/sd_g，mu_g/sd_g 取自**该路自己的 clean_good/none**。
判据（冻结）：PASS = ΔR <= -0.02 且 Δd' >= -0.10；catastrophic = Δd' <= -0.25（相对同 (cat,seed) Original）。
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
E16 = ROOT / "results" / "experiment_16_final_validation"
CATS = ["bottle", "cable", "hazelnut", "screw", "grid"]
SEEDS10 = list(range(10))
W_C2 = 0.35


def source(cat: str, seed: int, which: str):
    """Original(which='o') / B2(which='b') 的 unit 目录（按 seed 选择合法来源）。"""
    if which == "o":
        if seed in M.SEEDS:
            return M.source(cat, seed, "orig")
        pref, names = "E14_A0__", [("o1_orig_34", E14 / "raw"), ("o1ext_orig", E15 / "raw"),
                                   ("new10", E16 / "raw")]
    else:
        if seed in M.SEEDS:
            return M.source(cat, seed, "b2")
        pref, names = "E14_A1__", [("o1_b2_34", E14 / "raw"), ("o1ext_b2", E15 / "raw"),
                                   ("new10", E16 / "raw")]
    for rd, root in names:
        p = root / rd / cat / f"seed_{seed}" / f"config_{pref}{cat}"
        if (p / "info.json").exists():
            return p
    return None


def _z(rowsd: dict) -> dict:
    v = [s for k, s in rowsd.items() if k[0] == "clean_good" and k[1] == "none"]
    mu, sd = float(np.mean(v)), float(np.std(v, ddof=1))
    return {k: (s - mu) / sd for k, s in rowsd.items()}


def rule(cid: str, zo: float, zb: float) -> float:
    z_c2 = W_C2 * zo + (1 - W_C2) * zb
    if cid == "C2":
        return z_c2
    if cid == "C6":
        return max(zo, zb)
    if cid == "X6c":
        return 0.5 * (z_c2 + max(zo, zb))
    raise ValueError(cid)


def fused_unit(cat: str, seed: int, cid: str):
    po, pb = source(cat, seed, "o"), source(cat, seed, "b")
    if po is None or pb is None or not po.exists() or not pb.exists():
        return None
    ro, rb = M.rows(po), M.rows(pb)
    keys = [k for k in ro if k in rb]
    if not keys:
        return None
    zo, zb = _z(ro), _z(rb)
    sub = defaultdict(lambda: defaultdict(list))
    for k in keys:
        subset, sh, dt, _ = k
        sub[subset][sh].append((dt, float(rule(cid, zo[k], zb[k])), 0.0, 0.0))
    tau = max(s for _, s, _, _ in sub["val"]["none"]) if sub["val"]["none"] else float("nan")
    return {"sub": sub, "info": {"tau_val": float(tau), "alpha_l2": -1.0, "alpha_l3": -1.0,
                                 "pixel_auroc": float("nan"), "aupro": float("nan"),
                                 "runtime_seconds": 0.0, "coreset_size": -1,
                                 "peak_gpu_memory_allocated_mb": 0.0}}
