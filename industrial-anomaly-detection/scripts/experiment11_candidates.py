"""Experiment 11B — adaptive candidate registry（**在读取任何 Exp11 target result 之前冻结**）。

只用 train/good 的 normal 统计（`stats/normal_statistics.csv` + `stats/per_channel_<cat>.npz`，由 11A 生成）。
只改 **L2** 的 preservation 强度；**L3 恒为 alpha_in(0.25)**（robustness 支路不动）。
参考锚点 = **bottle**（已在 Exp10 中 3/3 PASS 的类别）→ 所有规则在 bottle 上退化为 C2-fixed 的 β=0.25。

Family A（RMS/响应幅度校准）：beta_c = clip(k / s_c, 0.05, 0.50)，k = 0.25 * s_bottle
   A1: s_c = mean_j(RATIO_j)；A2: s_c = median_j(RATIO_j)（robust）
   → 语义：让各类的「补偿项相对能量」β_c·RATIO_c 与参考类对齐
Family B（层敏感度比）：q = S_IN_L2/S_IN_L3；beta_c = clip(0.25 * (q_c/q_bottle)^p, 0.05, 0.50)
   B1: p=1（温和单调）；B2: p=3（更尖锐的单调映射）
Family C（channel-adaptive）：F_L2' = F + beta * g_j * IN(F)*rms(F)，beta=0.25
   C1: g_j ∝ 1/RATIO_j（对补偿响应弱的通道给更高权重），归一化到均值 1
   C2: g_j = 1 - clip(RATIO_j / p90_j, 0, 1) ∈ [0,1]
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "results" / "experiment_11_c2_adaptive"
CATS = ["bottle", "cable", "hazelnut", "screw", "grid"]
REF = "bottle"
BETA0, BMIN, BMAX = 0.25, 0.05, 0.50

_CAND_IDS = ["E11_A1", "E11_A2", "E11_B1", "E11_B2", "E11_C1", "E11_C2"]


def _load() -> dict:
    per = {}
    for c in CATS:
        z = np.load(EXP / "stats" / f"per_channel_{c}.npz")
        per[c] = {k: z[k].astype(np.float64) for k in ("rms", "inmag", "var", "ratio", "disp")}
    return per


_PC = None


def per_channel(cat: str) -> dict:
    global _PC
    if _PC is None:
        _PC = _load()
    return _PC[cat]


def beta_table() -> dict:
    """返回 {(candidate_id, category): beta}（category-level families）。"""
    ratio = {c: float(per_channel(c)["ratio"].mean()) for c in CATS}
    ratio_med = {c: float(np.median(per_channel(c)["ratio"])) for c in CATS}
    s_in2 = {c: float(per_channel(c)["inmag"].mean()) for c in CATS}
    # 重算 S_IN_L3（11A 的 CSV 已有，rms 无关）：从 csv 读
    import csv as _csv
    rows = {r["category"]: r for r in _csv.DictReader(open(EXP / "stats" / "normal_statistics.csv"))}
    q = {c: float(rows[c]["q_L2_over_L3"]) for c in CATS}
    k1, k2 = BETA0 * ratio[REF], BETA0 * ratio_med[REF]
    out = {}
    for c in CATS:
        out[("E11_A1", c)] = float(np.clip(k1 / ratio[c], BMIN, BMAX))
        out[("E11_A2", c)] = float(np.clip(k2 / ratio_med[c], BMIN, BMAX))
        out[("E11_B1", c)] = float(np.clip(BETA0 * (q[c] / q[REF]) ** 1, BMIN, BMAX))
        out[("E11_B2", c)] = float(np.clip(BETA0 * (q[c] / q[REF]) ** 3, BMIN, BMAX))
    return out


def channel_weights(cat: str) -> dict:
    """返回 {(candidate_id): g_vector(list[float])}（channel-level families）。"""
    r = per_channel(cat)["ratio"]
    g1 = (r.mean() / np.maximum(r, 1e-8))
    g1 = g1 / g1.mean()
    p90 = np.percentile(r, 90)
    g2 = 1.0 - np.clip(r / (p90 + 1e-12), 0.0, 1.0)
    return {"E11_C1": [float(x) for x in g1], "E11_C2": [float(x) for x in g2]}


def _resid(beta: float) -> dict:
    return {"kind": "residual", "lambda": float(beta), "norm": "instance",
            "scale_match": "per_sample_per_channel_rms(f)", "eps": 1e-5}


def _gated(beta: float, g: list) -> dict:
    return {"kind": "residual_gated", "beta": float(beta), "g": g, "norm": "instance", "eps": 1e-5}


L3_STEP = {"kind": "alpha_in", "alpha": 0.25}


def build_specs() -> dict:
    """构造 7A-O specs：{name: {'l2':..., 'l3':...}}，name = '<candidate>__<category>'。"""
    bt = beta_table()
    specs = {}
    for cid in _CAND_IDS:
        for cat in CATS:
            if cid in ("E11_C1", "E11_C2"):
                g = channel_weights(cat)[cid]
                l2 = _gated(BETA0, g)
            else:
                l2 = _resid(bt[(cid, cat)])
            specs[f"{cid}__{cat}"] = {"l2": l2, "l3": dict(L3_STEP)}
    return specs


SEVEN_AO_SPECS = build_specs()


# ---------------- Ablation preview（preview 专用；在读取 11C 结果前写定 ----------------
# 只在 Winner(E11_B2) × bottle/cable × seed0 上做：Original / L3-only / L2-only / Full
# L3-only = T1（L2 identity + L3 alpha_in(0.25)，Exp10 已有，复用）
# Full    = E11_B2（已有）
# L2-only = L2 用 E11_B2 的 beta_c，L3 设为 identity  -> 需要新增单元
def ablation_specs(cats=("bottle", "cable")) -> dict:
    bt = beta_table()
    out = {}
    for cat in cats:
        out[f"E11_B2_L2only__{cat}"] = {"l2": _resid(bt[("E11_B2", cat)]),
                                        "l3": {"kind": "alpha_in", "alpha": 0.0}}
    return out


SEVEN_AO_SPECS.update(ablation_specs())


def registry() -> dict:
    bt = beta_table()
    return {
        "frozen_before_target_results": True,
        "reference_anchor": REF,
        "beta_bounds": [BMIN, BMAX],
        "L3_fixed": L3_STEP,
        "candidates": [
            {"id": "E11_A1", "family": "A", "level": "category",
             "formula": "beta_c = clip(k1/RATIO_c, 0.05, 0.50), k1 = 0.25*RATIO_bottle, RATIO = mean_j(||IN(F)_j - F_j||/||F_j||)",
             "stat_source": "normal_statistics.csv RATIO_L2 (train/good only)",
             "rationale": "让各类的补偿项相对能量 beta_c*RATIO_c 与参考类(bottle)对齐",
             "beta_per_category": {c: bt[("E11_A1", c)] for c in CATS}},
            {"id": "E11_A2", "family": "A", "level": "category",
             "formula": "beta_c = clip(k2/median_j(RATIO_j), 0.05, 0.50), k2 = 0.25*median_j(RATIO_bottle,j)",
             "stat_source": "per_channel_<cat>.npz ratio (robust median)",
             "rationale": "同 A1 但用稳健中位数，抗离群通道",
             "beta_per_category": {c: bt[("E11_A2", c)] for c in CATS}},
            {"id": "E11_B1", "family": "B", "level": "category",
             "formula": "beta_c = clip(0.25*(q_c/q_bottle)^1, 0.05, 0.50), q = S_IN_L2/S_IN_L3",
             "stat_source": "normal_statistics.csv q_L2_over_L3",
             "rationale": "层敏感度比越高说明 L2 已被归一化压制越多 -> 线性补偿",
             "beta_per_category": {c: bt[("E11_B1", c)] for c in CATS}},
            {"id": "E11_B2", "family": "B", "level": "category",
             "formula": "beta_c = clip(0.25*(q_c/q_bottle)^3, 0.05, 0.50)",
             "stat_source": "同 B1（更尖锐的单调映射）",
             "rationale": "检验 sharpness 的影响（预注册的第二个单调版本）",
             "beta_per_category": {c: bt[("E11_B2", c)] for c in CATS}},
            {"id": "E11_C1", "family": "C", "level": "channel",
             "formula": "F_L2' = F + 0.25 * g_j * IN(F)*rms(F), g_j = (mean_j RATIO_j / RATIO_j) / mean(...)",
             "stat_source": "per_channel_<cat>.npz ratio",
             "rationale": "补偿响应弱的通道给更高权重（channel-level 精细化）",
             "g_mean": 1.0},
            {"id": "E11_C2", "family": "C", "level": "channel",
             "formula": "F_L2' = F + 0.25 * g_j * IN(F)*rms(F), g_j = 1 - clip(RATIO_j/p90_j, 0, 1)",
             "stat_source": "per_channel_<cat>.npz ratio",
             "rationale": "在归一化响应已很强的通道上**关闭**额外补偿",
             "g_range": [0.0, 1.0]},
        ],
        "equivalence_checks": {
            "E11_*_bottle_vs_C2_fixed": "所有规则在 bottle 上退化为 beta=0.25（C1/C2 的 g 在 bottle 上不退化，属 channel-level，不是等价）",
            "E11_A1_bottle == C2_fixed": True, "E11_A2_bottle_approx": True,
            "E11_B1_bottle == C2_fixed": True, "E11_B2_bottle == C2_fixed": True},
        "n_candidates": len(_CAND_IDS), "n_specs": len(SEVEN_AO_SPECS),
    }


if __name__ == "__main__":
    EXP.mkdir(parents=True, exist_ok=True)
    (EXP / "config" / "candidate_registry.json").write_text(
        json.dumps(registry(), indent=2, ensure_ascii=False, default=str))
    bt = beta_table()
    print("frozen beta table (candidate x category):")
    print("%-8s" % "cand" + "".join("%10s" % c for c in CATS))
    for cid in _CAND_IDS[:4]:
        print("%-8s" % cid + "".join("%10.4f" % bt[(cid, c)] for c in CATS))
    print("C1/C2: channel-level, beta=0.25 fixed; specs=%d" % len(SEVEN_AO_SPECS))
