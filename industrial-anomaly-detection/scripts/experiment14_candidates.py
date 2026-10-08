"""Experiment 14 — 冻结的候选 spec 注册表（Family A dual-path + Family B tiny INSS）。

Family A 关键代数结论（已写入 registry，**不改变方法定义，只做等价改写**）：
    F = w*F_original + (1-w)*F_B2
      L2: F + (1-w)*beta_c*IN(F)*rms(F)      -> 既有 kind "residual"(lambda=(1-w)beta_c)
      L3: alpha_in((1-w)*0.25)               -> 既有 kind "alpha_in"
因此 Family A 无需新模型代码；A7 == A2 已删除。
Family B 需要新 kind "inss"（见 experiment14_model.py），basis 来自 Exp13-P 冻结资产（逐位校验 0.0）。
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
E14 = ROOT / "results" / "experiment_14"
CATS = ["bottle", "cable", "hazelnut", "screw", "grid"]
W = {"A2": 0.25, "A3": 0.50, "A4": 0.75}
B_PAIRS = {"B1": (8, 0.25), "B2": (8, 0.50), "B3": (16, 0.25),
           "B4": (16, 0.50), "B5": (32, 0.25), "B6": (32, 0.50)}


def _beta() -> dict:
    import experiment11_candidates as c11
    bt = c11.beta_table()
    return {c: float(bt[("E11_B2", c)]) for c in CATS}


def _resid(lam: float) -> dict:
    return {"kind": "residual", "lambda": float(lam), "norm": "instance",
            "scale_match": "per_sample_per_channel_rms(f)", "eps": 1e-5}


def _alpha(a: float) -> dict:
    return {"kind": "alpha_in", "alpha": float(a)}


def _inss(cat: str, k: int, lam: float) -> dict:
    z = np.load(E14 / "config" / f"inss_basis_{cat}_k{k}.npz")
    U = np.asarray(z["U"], dtype=np.float64).T          # (K, C) 便于逐行存储
    mu = np.asarray(z["mu"], dtype=np.float64)
    return {"kind": "inss", "K": int(k), "lam": float(lam),
            "U": [[float(x) for x in row] for row in U],
            "mu": [float(x) for x in mu], "layer": "layer2"}


def build_specs() -> dict:
    beta = _beta()
    out = {}
    for cat in CATS:
        for cid, w in W.items():                       # Family A: 线性融合（强度缩放）
            out[f"E14_{cid}__{cat}"] = {"l2": _resid((1.0 - w) * beta[cat]), "l3": _alpha((1.0 - w) * 0.25)}
        # A0 = w=1 端点（== Original；alpha_in(0) 严格 short-circuit → 与 plain PatchCore 逐位一致）
        out[f"E14_A0__{cat}"] = {"l2": _alpha(0.0), "l3": _alpha(0.0)}              # == Original
        # A1 = w=0 端点（== Adaptive B2 算子；**参考/等价检查，非竞争者**）
        out[f"E14_A1__{cat}"] = {"l2": _resid(beta[cat]), "l3": _alpha(0.25)}
        out[f"E14_A5__{cat}"] = {"l2": _alpha(0.0), "l3": _alpha(0.25)}            # == T1
        out[f"E14_A6__{cat}"] = {"l2": _resid(beta[cat]), "l3": _alpha(0.0)}       # == B2_L2only
        for cid, (k, lam) in B_PAIRS.items():          # Family B: tiny INSS (layer2 only)
            out[f"E14_{cid}__{cat}"] = {"l2": _inss(cat, k, lam), "l3": _alpha(0.0)}
    return out


ALL14_SPECS = build_specs()
FROZEN_KINDS = ("alpha_in", "residual", "energy", "concat_dual", "altnorm_strength",
                "residual_gated", "inss")


def verify_frozen() -> tuple:
    """R6 证据：kind 白名单 + Family A 与 registry 的代数等价数值核验 + INSS basis 与冻结资产逐位一致。"""
    beta = _beta()
    reg = json.loads((E14 / "config" / "candidate_registry.json").read_text())
    bad = []
    for name, sp in ALL14_SPECS.items():
        cid, cat = name.split("__")
        if sp["l2"]["kind"] not in FROZEN_KINDS or sp["l3"]["kind"] not in FROZEN_KINDS:
            bad.append(f"{name}:kind")
        if cid in W:
            exp = reg["families"]["A_dual_path"]["candidates"][cid]["spec"][cat]
            w = 1.0 - W[cid]
            if abs(sp["l2"]["lambda"] - exp["l2"]["lambda"]) > 1e-12 or abs(sp["l3"]["alpha"] - exp["l3"]["alpha"]) > 1e-12:
                bad.append(f"{name}:algebra")
            if abs(sp["l2"]["lambda"] - w * beta[cat]) > 1e-12:
                bad.append(f"{name}:beta")
        if cid in B_PAIRS:
            k, lam = B_PAIRS[cid]
            z = np.load(E14 / "config" / f"inss_basis_{cat}_k{k}.npz")
            U = np.asarray(sp["l2"]["U"], dtype=np.float64).T
            mu = np.asarray(sp["l2"]["mu"], dtype=np.float64)
            if (abs(U - np.asarray(z["U"], dtype=np.float64)).max() > 0
                    or abs(mu - np.asarray(z["mu"], dtype=np.float64)).max() > 0):
                bad.append(f"{name}:basis")
            if abs(sp["l2"]["lam"] - lam) > 1e-12 or sp["l2"]["K"] != k or sp["l3"]["alpha"] != 0.0:
                bad.append(f"{name}:k/lam")
    return (len(bad) == 0, f"n_specs={len(ALL14_SPECS)}; frozen kinds; "
                           f"FamilyA algebra+beta verified; INSS basis bit-identical to Exp13-P; "
                           f"violations={bad[:4]}")
