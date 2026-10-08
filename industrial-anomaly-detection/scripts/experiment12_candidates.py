"""Experiment 12 — Category × Channel adaptive compensation（**在读取任何 Exp12 target 结果前冻结**）。

假设：**Category-level 决定"补多少"（β_c），Channel-level 决定"补哪些通道"（g_k）**，二者机制正交，
组合应比单纯 category-level（Exp11 B2）更稳定地保护 defect information 同时保持 illumination robustness。

严格复用组件：
  * β_c   : Exp11 winner B2 的冻结值（scripts/experiment11_candidates.beta_table()，train/good only）
  * g_C2  : Exp11 E11_C2 的原始 channel gate（1 - clip(RATIO_j/p90_j, 0, 1)，train/good only）
  * L2 算子: experiment11_model.AdaptivePatchcoreModel 的 `residual_gated`（**零新代码**）
  * L3    : alpha_in(0.25) 固定

候选（M4 **按规则跳过**：C2 原 gate 本身已是连续 soft 映射（90% 通道 g>0，仅 10.2% 恰为 0），
任务书明确「已是 soft continuous gate 则不要为凑 M4 发明新公式」）：
  M0 = B2（g≡1）           —— 直接复用 Exp11，0 GPU
  M1 = B2 × (0.75 + 0.25*g_C2)   收缩 gate（轻度 channel 选择）
  M2 = B2 × (0.50 + 0.50*g_C2)   中等
  M3 = B2 × g_C2                 完整 channel 选择
  G1 = B2 且 g≡1（sanity 用；新代码路径必须逐位退化为 B2）
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
E12 = ROOT / "results" / "experiment_12"
CATS = ["bottle", "cable", "hazelnut", "screw", "grid"]
HARD = ["bottle", "cable", "hazelnut"]
B2_ID = "E11_B2"
L3_STEP = {"kind": "alpha_in", "alpha": 0.25}
_G_RULES = {"E12_M1": (0.75, 0.25), "E12_M2": (0.50, 0.50), "E12_M3": (0.00, 1.00)}


def _c11():
    import sys
    sys.path.insert(0, str(ROOT / "scripts"))
    sys.path.insert(0, str(ROOT / "experiments" / "2026-09-30_illumination_sensitivity_exploration" / "falpha_patchcore"))
    import experiment11_candidates as c11
    return c11


def b2_beta() -> dict:
    c11 = _c11()
    bt = c11.beta_table()
    return {c: float(bt[(B2_ID, c)]) for c in CATS}


def g_c2(cat: str) -> np.ndarray:
    c11 = _c11()
    return np.asarray(c11.channel_weights(cat)["E11_C2"], dtype=np.float64)


def gate(cid: str, cat: str) -> np.ndarray:
    if cid in _G_RULES:
        a, b = _G_RULES[cid]
        return a + b * g_c2(cat)
    if cid == "E12_G1":
        return np.ones_like(g_c2(cat))
    raise KeyError(cid)


def _spec(cid: str, cat: str) -> dict:
    beta = b2_beta()[cat]
    g = gate(cid, cat)
    return {"l2": {"kind": "residual_gated", "beta": float(beta),
                   "g": [float(x) for x in g], "norm": "instance", "eps": 1e-5},
            "l3": dict(L3_STEP)}


def build_specs(cats=None) -> dict:
    cats = cats or CATS
    out = {}
    for cid in ("E12_M1", "E12_M2", "E12_M3", "E12_G1"):
        for cat in cats:
            out[f"{cid}__{cat}"] = _spec(cid, cat)
    return out


SEVEN_AO_SPECS = build_specs()


def registry() -> dict:
    bt = b2_beta()
    reg = {"frozen_before_target_results": True,
           "hypothesis": ("Category-level (β_c) 决定补多少、Channel-level (g_k) 决定补哪些通道，"
                          "组合应优于单用 category-level（Exp11 B2）"),
           "m4_skipped_by_rule": ("Exp11 的 C2 gate 已是连续 soft 映射（1 - clip(RATIO_j/p90,0,1)；"
                                  "90% 通道 g>0，仅 10.2% 恰为 0）→ 按任务书不得为凑 M4 发明新公式"),
           "components_reused": {
               "beta": "Exp11 B2 frozen beta_table (train/good only)",
               "gate": "Exp11 E11_C2 original channel gate (train/good only)",
               "l2_operator": "experiment11_model residual_gated (zero new model code)",
               "l3": "alpha_in(0.25) fixed"},
           "candidates": [
               {"id": "M0", "spec_prefix": B2_ID, "gate": "g=1 (category-only)",
                "beta": bt, "units": 0, "note": "复用 Exp11 B2（15 units 已存在）"},
               {"id": "M1", "spec_prefix": "E12_M1", "gate": "g = 0.75 + 0.25*g_C2", "beta": bt,
                "gate_mean": {c: float(gate("E12_M1", c).mean()) for c in CATS}},
               {"id": "M2", "spec_prefix": "E12_M2", "gate": "g = 0.50 + 0.50*g_C2", "beta": bt,
                "gate_mean": {c: float(gate("E12_M2", c).mean()) for c in CATS}},
               {"id": "M3", "spec_prefix": "E12_M3", "gate": "g = g_C2", "beta": bt,
                "gate_mean": {c: float(gate("E12_M3", c).mean()) for c in CATS}},
               {"id": "G1(sanity)", "spec_prefix": "E12_G1", "gate": "g = 1（必须逐位等于 M0/B2）",
                "beta": bt, "units": 1}],
           "n_specs": len(SEVEN_AO_SPECS),
           "beta_by_category": bt,
           "gate_c2_mean": {c: float(g_c2(c).mean()) for c in CATS},
           "equivalence_checks": {"E12_G1__<cat> == E11_B2__<cat>": "required (Sanity B)"}}
    return reg


if __name__ == "__main__":
    E12.mkdir(parents=True, exist_ok=True)
    (E12 / "config" / "candidate_registry.json").write_text(
        json.dumps(registry(), indent=2, ensure_ascii=False, default=str))
    print(json.dumps({c: round(v, 4) for c, v in b2_beta().items()}, ensure_ascii=False))
    for cid in ("E12_M1", "E12_M2", "E12_M3"):
        print(cid, {c: round(float(gate(cid, c).mean()), 4) for c in CATS})
    print("specs:", len(SEVEN_AO_SPECS))
