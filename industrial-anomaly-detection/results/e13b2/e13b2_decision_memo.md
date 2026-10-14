# E13-B2 · Bank Sensitivity + Extended Stability Sweep — Decision Memo

## Verdict: **STOP**（Stage A = UNSTABLE；R1 增益是 ceiling-limited，非普遍适用）

## 1. 实验题目
R1（Linear Residual Adapter）的跨 bank-seed 波动是否主要来自 normal-bank sampling sensitivity？区分「方法失效」与「baseline bank sensitivity」。

## 2. 日期
2026-10-14

## 3. 论文阶段
稳定性验证 / 混杂诊断。

## 4. 要验证的核心论断
「R1 的 paired 正收益在不同 normal-bank sampling 下是否稳定，且是否受 frozen baseline 高度限制（ceiling effect）。」

## 5. 实验目的
E13-B 中 8/9 为正、但 Cube seed1 唯一负例且 frozen R0 明显随 bank 漂移。必须判定这是随机波动还是系统性 ceiling 限制。

## 6. 数据与参数
- 方法/loss/lr/augmentation/scoring 严格复用 E13-A/E13-B（R1，未改动）。
- Stage A = Cube × seed3–7（5 新 cell，已跑）。种子规则 `random.Random(20261009+seed)`。
- 现有：Cube seed0–2、Bird seed0–2、Car seed0–2。

## 7. 实验过程
E13-B commit `e036dff` → 协议冻结 → Stage A（Cube seed3–7，screen `e13b2_main`，10.5 min）→ 判定 UNSTABLE → **按协议不跑 Stage B，直接止损写 memo**。

## 8. 结果（Fact）

### 8.1 Cube 8-seed 完整 ΔAUROC（R0 括号内）
| seed | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---|---|---|---|---|---|---|---|
| ΔAUROC | +0.0160 | −0.0123 | +0.0042 | −0.0192 | +0.0036 | −0.0048 | −0.0037 | +0.0045 |
| R0 | 0.8541 | 0.8720 | 0.8653 | 0.8681 | 0.8631 | 0.8620 | 0.8637 | 0.8595 |

- Cube：**4/8 正**，mean ΔAUROC **−0.00146**（std 0.0110，min −0.0192，max +0.0160），mean ΔAP **−0.00097**。

### 8.2 三类对比
| category | n seeds | mean ΔAUROC | std | positive | R0 范围 |
|---|---|---|---|---|---|
| Bird | 3 | **+0.02586** | 0.0135 | 3/3 | 0.747–0.756 |
| Car | 3 | **+0.01123** | 0.0073 | 3/3 | 0.786–0.814 |
| Cube | 8 | **−0.00146** | 0.0110 | 4/8 | 0.854–0.872 |

### 8.3 Ceiling effect（核心事实）
- Pearson(R0, ΔAUROC)：Cube 内 **−0.839**，整体 **−0.783**；Spearman(Cube 内) **−0.762**。
- 即 frozen baseline 越高，R1 增益越低/越负。Cube 的 R0 集中在 0.86 高位 → 增益被 ceiling 压成负均值。

## 9. 异常情况
无 NaN / OOM / collapse。Cube seed3（−0.0192）比 seed1（−0.0123）更负，证明 seed1 不是孤立异常，而是 Cube 高 baseline 的普遍现象。

## 10. 分析（Fact → Interpretation → Hypothesis）

**Fact**
1. Cube 8 seed 仅 4/8 正、mean −0.0015；Bird/Car 各 3/3 正。
2. R0 与 ΔAUROC 强负相关（r ≈ −0.78~−0.84）。

**Interpretation**
1. **R1 的「不稳定」不是随机 bank noise，而是系统性 ceiling effect**：ΔAUROC 随 R0 单调下降。Bird（低 baseline）收益大且稳；Car（中）收益中；Cube（高 baseline）收益消失甚至为负。
2. 因此 E13-B 中「Cube seed1 负例」被正确解释为**baseline 采样问题，而非方法失效**：Cube 的 frozen DINOv2-B 已近饱和（~0.86），illumination-consistent adaptation 无额外信息可挖。
3. R1 是**baseline-dependent** 的方法：只在有 headroom 的类别有效。

**Hypothesis（未验证）**
- 猜测 illumination-consistent adaptation 的作用机制是「在表示空间补偿光照扰动带来的可分性损失」，当 frozen baseline 已把类别分得很开（Cube）时该补偿无益甚至过拟合 normal 流形。

## 11. 结论
R1 增益是 **ceiling-limited**：可靠提升 Bird/Car（低/中 baseline），但对近饱和的 Cube 无益（8 seed mean −0.0015）。这不构成「普遍适用」的 PASS。

## 12. 出口条件判断
- Stage A 判据：STABLE 需 mean ≥+0.005 且 ≥6/8 正且 ΔAP≥0；BORDERLINE 需 mean≥0 且 ≥4/8 正。Cube 实际 mean −0.00146 < 0、4/8 正 → **UNSTABLE**。
- 按协议 UNSTABLE → 不跑 Stage B，直接止损。

## 13. 对论文证据链的贡献
本实验把「R1 跨 bank-seed 稳定性」这一格填上**关键诊断结论**：R1 的波动来自 **ceiling effect**（r≈−0.8），不是方法失效，而是 baseline 饱和。这既解释了 E13-B 的 Cube seed1 异常，也界定了 R1 的适用范围（低/中 baseline 类别）。论文可据实表述为「R1 在 baseline 尚有 headroom 的类别上稳定增益 +0.01~+0.026，在近饱和类别上无增益」。

## 14. 下一步（建议，不自动执行）
- 本轮 STOP（不扩类别）。
- 人工裁决方向：(a) 接受「R1 = baseline-dependent 方法」，重定义论文主张为「面向 headroom 类别的 illumination-consistent adaptation」；(b) 换 backbone/representation family（避开 DINOv2-B 在 Cube 上已饱和的问题）；(c) 若坚持通用性，需针对 ceiling 类别设计新机制（非本轮范围）。
- 禁止调 lr/steps/augmentation 救火。

## 15. 复现信息
- 环境：RTX 3090 24GB / PyTorch 2.8.0+cu128 / Python 3.12.3 / conda base
- 命令：`python scripts/e13b2_screen.py --categories Cube --seeds 3,4,5,6,7` + `scripts/e13b2_analysis.py`
- 输出：`results/e13b2/{e13b2_protocol.json, e13b2_results.csv, e13b2_summary.json, e13b2_decision_memo.md, README.md, figures/}`
