"""
Experiment 1G — CASE A/B/C 判定（遵循 config.json case_rules + analysis_levels 分层规则）

LEVEL 1 (PRIMARY): 1E Δdefect_std ↔ 1G ΔNN-distance std 是否对应（Spearman）
LEVEL 2: NN-dispersion 变化是否来自 defect feature geometry（dispersion/norm）
LEVEL 3: feature geometry 是否与 channel variance 相关

判定（预注册，不可事后改）：
- CASE_A: 稳定链 1E↔NN-dispersion↔feature-geometry，非 background 同步变化 → 继续 channel/layer
- CASE_B: 1E↔NN-dispersion 对应好，但 feature dispersion/norm 解释不了 → 定位 NN geometry
- CASE_C: NN distance 对不上 + feature geometry 对不上 + background 乱 → STOP，切口降级
"""

import csv
import json
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "experiment_1g"
ANALYSIS = OUT / "analysis"

# 预注册阈值
RHO_THRESH = 0.60          # Spearman |ρ| 判定"对应好"的最低值
P_THRESH = 0.05            # 显著性
N_MIN = 5                  # 最少样本数（primary 9 个 defect，但按图聚合会更多）


def load_summary() -> list[dict]:
    return list(csv.DictReader(open(ANALYSIS / "defect_summary.csv", encoding="utf-8")))


def load_1e_response() -> dict:
    resp = {}
    for r in csv.DictReader(open(
            ROOT / "results/experiment_1e/analysis/defect_response_signatures_seed_summary.csv",
            encoding="utf-8")):
        resp[(r["category"], r["defect_type"])] = float(r["delta_defect_std_mean"])
    return resp


def spearman(xs, ys):
    xs = np.array(xs, dtype=float)
    ys = np.array(ys, dtype=float)
    keep = ~(np.isnan(xs) | np.isnan(ys))
    if keep.sum() < N_MIN:
        return np.nan, np.nan, int(keep.sum())
    return (*stats.spearmanr(xs[keep], ys[keep]), int(keep.sum()))


def main() -> None:
    summary = load_summary()
    resp = load_1e_response()

    # 对齐 1E 响应
    d1e = []
    dnn = []        # matched
    dnn_frozen = []
    dd   = []       # dispersion
    dnorm = []      # norm std
    dch   = []      # channel var
    for r in summary:
        key = (r["category"], r["defect_type"])
        if key not in resp:
            continue
        d1e.append(resp[key])
        dnn.append(float(r["delta_nn_std"]))
        dnn_frozen.append(float(r["delta_nn_std_frozen"]))
        dd.append(float(r["delta_dispersion"]))
        dnorm.append(float(r["delta_norm_std"]))
        dch.append(float(r["delta_channel_var"]))

    print("=" * 70)
    print("Experiment 1G — CASE 判定")
    print("=" * 70)

    r1, p1, n1 = spearman(d1e, dnn)
    r1f, p1f, _ = spearman(d1e, dnn_frozen)
    print(f"\n[LEVEL 1] 1E Δdefect_std vs 1G ΔNN-std (matched):  ρ={r1:+.3f} p={p1:.3f} n={n1}")
    print(f"[LEVEL 1] 1E Δdefect_std vs 1G ΔNN-std (frozen):   ρ={r1f:+.3f} p={p1f:.3f} n={n1}")

    level1_ok = (not np.isnan(r1)) and (abs(r1) >= RHO_THRESH) and (p1 < P_THRESH)
    print(f"[LEVEL 1] 对应成立? {level1_ok}  (要求 |ρ|≥{RHO_THRESH} 且 p<{P_THRESH})")

    # LEVEL 2: NN-dispersion 是否来自 feature geometry
    r2_disp, p2_disp, _ = spearman(dnn, dd)
    r2_norm, p2_norm, _ = spearman(dnn, dnorm)
    print(f"\n[LEVEL 2] ΔNN-std vs Δfeature-dispersion:  ρ={r2_disp:+.3f} p={p2_disp:.3f}")
    print(f"[LEVEL 2] ΔNN-std vs Δnorm-std:            ρ={r2_norm:+.3f} p={p2_norm:.3f}")
    level2_ok = (abs(r2_disp) >= RHO_THRESH and p2_disp < P_THRESH) or \
                (abs(r2_norm) >= RHO_THRESH and p2_norm < P_THRESH)
    print(f"[LEVEL 2] feature geometry 解释成立? {level2_ok}")

    # LEVEL 3: norm/channel variance
    r3, p3, _ = spearman(dnorm, dch)
    print(f"\n[LEVEL 3] Δnorm-std vs Δchannel-var:       ρ={r3:+.3f} p={p3:.3f}")

    # ---- CASE 判定（预注册规则）----
    print("\n" + "-" * 70)
    if level1_ok and level2_ok:
        verdict = "CASE_A"
        reason = ("稳定链 1E↔NN-dispersion↔feature-geometry 成立，"
                  "继续 channel/layer 机制（LEVEL 3）")
    elif level1_ok and not level2_ok:
        verdict = "CASE_B"
        reason = ("1E↔NN-dispersion 对应好，但 feature dispersion/norm 解释不了 → "
                  "定位到 NN geometry（bank geometry / neighbor switching）")
    else:
        verdict = "CASE_C"
        reason = ("NN-distance 对不上 1E 响应 → STOP，切口降级（1G 的 feature-space 假设不成立）")

    print(f"VERDICT: {verdict}")
    print(f"REASON: {reason}")
    print("-" * 70)

    # 写 verdict.json
    result = {
        "experiment": "1G",
        "verdict": verdict,
        "reason": reason,
        "level1": {"spearman_1e_vs_nnstd": r1, "p": p1, "n": n1,
                   "spearman_1e_vs_nnstd_frozen": r1f, "p_frozen": p1f,
                   "ok": bool(level1_ok)},
        "level2": {"spearman_nnstd_vs_dispersion": r2_disp, "p_disp": p2_disp,
                   "spearman_nnstd_vs_normstd": r2_norm, "p_norm": p2_norm,
                   "ok": bool(level2_ok)},
        "level3": {"spearman_norm_vs_channel": r3, "p": p3},
        "thresholds": {"rho": RHO_THRESH, "p": P_THRESH, "n_min": N_MIN},
    }
    with open(ANALYSIS / "verdict.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"\nverdict.json → {ANALYSIS / 'verdict.json'}")


if __name__ == "__main__":
    main()
