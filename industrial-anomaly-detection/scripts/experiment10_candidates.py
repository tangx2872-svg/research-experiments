"""Experiment 10 — 候选注册表（**在读取任何 target result 之前冻结**）。

论文逻辑：T1 证明「normalization 换 robustness」有效，但 preservation 损失 −0.3544（bottle seed0, V2）。
今晚的问题：**能否把被 normalization 抑制的 defect-sensitive 信息补回来，同时保住 robustness gain？**

结构统一为：  Robust Representation  +  Defect-Preserving Information  →  Fusion

所有候选统一：**L2 保持 Original**（9B-R 证据：动 L2 会掉 preservation），只在 **L3** 或结构上做补偿。
全部 normal-only / deployable / training-free；参数在读取结果前冻结；不读 test/defect 数据。
"""

from __future__ import annotations

import experiment7ao_config as c7

# 历史冻结的 uniform-alpha 工作点（9A/9B uniform_alpha_frontier + 5A-H/6B/7A-O-Q4）
ALPHA_GRID = [0.0, 0.125, 0.2, 0.25, 0.3, 0.40091275, 0.5, 0.601369125, 0.8018255]


def _an(a: float) -> str:
    """完整精度命名（**不能**用 %g：它只保留 6 位有效数字，会把 0.40091275 截成 0.40091）。"""
    return repr(float(a)).replace(".", "")


NINE_B_CONFIGS: dict = {
    "P10_ORIG_a000": {"kind": "alpha", "alpha_l2": 0.0, "alpha_l3": 0.0},
    "P10_T1_L2a000_L3a025": {"kind": "alpha", "alpha_l2": 0.0, "alpha_l3": 0.25},
    "P10_D1_L3m9pow_g1": {"kind": "gate", "alpha_l2": -1.0, "alpha_l3": -1.0,
                          "gate_l2": ["const", {"value": 0.0}],
                          "gate_l3": ["m9_powmap", {"gamma": 1.0}]},
    "P10_D2_L3m8z_ab050_b100": {"kind": "gate", "alpha_l2": -1.0, "alpha_l3": -1.0,
                                "gate_l2": ["const", {"value": 0.0}],
                                "gate_l3": ["m8_zmap", {"a_bar": 0.5, "beta": 1.0}]},
    "P10_D3_L3m8z_ab025_b050": {"kind": "gate", "alpha_l2": -1.0, "alpha_l3": -1.0,
                                "gate_l2": ["const", {"value": 0.0}],
                                "gate_l3": ["m8_zmap", {"a_bar": 0.25, "beta": 0.5}]},
}
for _a in ALPHA_GRID:
    NINE_B_CONFIGS["P10_uniform_a%s" % _an(_a)] = {"kind": "alpha", "alpha_l2": _a, "alpha_l3": _a}

SEVEN_AO_SPECS: dict = {
    "P10_B1_L3concat_robust_a025": c7.spec(c7.alpha_step(0.0),
                                           {"kind": "concat_robust", "alpha": 0.25,
                                            "norm": "instance", "eps": 1e-5}),
    "P10_B2_L3concat_rscale_a025": c7.spec(c7.alpha_step(0.0),
                                           {"kind": "concat_robust_scaled", "alpha": 0.25,
                                            "norm": "instance", "eps": 1e-5}),
    "P10_C2_L2resid025_L3a025": c7.spec(c7.residual_step(0.25), c7.alpha_step(0.25)),
    "P10_C3_L2dual_L3a025": c7.spec(c7.dual_step(1.0), c7.alpha_step(0.25)),
    "P10_E1_L3energy025": c7.spec(c7.alpha_step(0.0), c7.energy_step(0.25)),
    "P10_E2_L3energy050": c7.spec(c7.alpha_step(0.0), c7.energy_step(0.50)),
}


CANDIDATES = [
    {"id": "P10_B1_L3concat_robust_a025", "family": "B", "backend": "7ao", "gpu": True,
     "defn": "L2 identity; L3 = cat([F, 0.75F+0.25*IN(F)]) —— Original block + robust block"},
    {"id": "P10_B2_L3concat_rscale_a025", "family": "B", "backend": "7ao", "gpu": True,
     "defn": "L2 identity; L3 = cat([F, 0.75F+0.25*IN(F)*rms(F)]) —— 尺度匹配版（仅 per-sample RMS）"},
    {"id": "P10_C2_L2resid025_L3a025", "family": "C", "backend": "7ao", "gpu": True,
     "defn": "L2 = F+0.25*IN(F)*rms(F)（scale-matched residual，非 alpha 插值）; L3 = 0.75F+0.25*IN(F)"},
    {"id": "P10_C3_L2dual_L3a025", "family": "C", "backend": "7ao", "gpu": True,
     "defn": "L2 = cat([F, IN(F)*rms(F)])（dual representation）; L3 = 0.75F+0.25*IN(F)"},
    {"id": "P10_D1_L3m9pow_g1", "family": "D", "backend": "9b", "gpu": True,
     "defn": "L2 identity; L3 per-channel gate w_c=(s_c/p90)^1（仅 train/good 统计）"},
    {"id": "P10_D2_L3m8z_ab050_b100", "family": "D", "backend": "9b", "gpu": True,
     "defn": "L2 identity; L3 gate a_c=clip(0.5+1.0*z_c)（仅 train/good 统计）"},
    {"id": "P10_D3_L3m8z_ab025_b050", "family": "D", "backend": "9b", "gpu": True,
     "defn": "L2 identity; L3 gate a_c=clip(0.25+0.5*z_c)（仅 train/good 统计）"},
    {"id": "P10_E1_L3energy025", "family": "E", "backend": "7ao", "gpu": True,
     "defn": "L2 identity; L3 = (F+0.25*IN(F)*rms(F)) * rms(F)/rms(mix) —— 能量保持"},
    {"id": "P10_E2_L3energy050", "family": "E", "backend": "7ao", "gpu": True,
     "defn": "L2 identity; L3 = (F+0.50*IN(F)*rms(F)) * rms(F)/rms(mix) —— 能量保持（更强）"},
]

EQUIVALENTS = [
    {"id": "P10_A1_resid025", "family": "A", "gpu": False, "effective": "alpha_L3 = 0.25*(1-0.25) = 0.1875",
     "proof": "F_R+lam(F_O-F_R) = (1-a)F+a*IN + lam*a(F-IN) = (1-a(1-lam))F + a(1-lam)IN  => 等价于 per-layer alpha a'=a(1-lam)"},
    {"id": "P10_A2_resid050", "family": "A", "gpu": False, "effective": "alpha_L3 = 0.25*(1-0.50) = 0.125",
     "proof": "同 A1（lam=0.5）"},
    {"id": "P10_A3_resid075", "family": "A", "gpu": False, "effective": "alpha_L3 = 0.25*(1-0.75) = 0.0625",
     "proof": "同 A1（lam=0.75）"},
]

REFERENCES = [
    {"id": "P10_ORIG_a000", "role": "corrected-189 Original（Stage 0.5 与 Stage 1 共用）"},
    {"id": "P10_T1_L2a000_L3a025", "role": "corrected-189 T1 reference（9B-R 已知最优 robustness）"},
]


def registry() -> dict:
    return {"alpha_grid_historical_frozen": ALPHA_GRID,
            "references": REFERENCES, "candidates": CANDIDATES, "equivalents_deduped": EQUIVALENTS,
            "n_candidates_gpu": sum(1 for c in CANDIDATES if c["gpu"]),
            "n_equivalents_no_gpu": len(EQUIVALENTS),
            "uniform_workpoints": ["P10_uniform_a%s" % _an(a) for a in ALPHA_GRID]}
