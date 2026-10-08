# Negative Results Registry（不许删除负结果）

> 由 overnight Q5（CPU-only）整理。每条都指向可核验的入口。

## N1 — 简单缺陷属性不足以解释 defect sensitivity
- 实验：`exp1f_mechanism_screening`（CASE_D）
- 含义：面积/对比度等简单属性筛选不能解释 sensitivity 差异 → 机制需要表示层解释。
- 入口：`experiments/exp1f_mechanism_screening/README.md`

## N2 — geometry 的"身份"从未被确立
- 实验：`experiment5d`（**literal CASE_D / 实质 CASE_B**，identity 未确立）
- 含义：5C 的收益不能被归因于 geometry 的**类别选择信息**；matched sensitivity control 反而更强
  （+0.1450/+0.1540，3/3 seeds，且由 grid 单类别驱动）。
- 入口：`experiments/experiment5d/README.md`；根 README §7

## N3 — geometry-guided gating 连续三次被削弱
- `experiment5c`（CASE_A）→ `experiment6a`（CASE_A，soft gating）→ `experiment6b`（**CASE_C**）
- 6B 结论：α_F=0.25 的可行点 **region length = 1（孤立点）**；放宽口径也只有 [0.20, 0.25]；
  相对 mean-α matched `Uniform(0.40091275)`，Δd′ +0.0842（< ε）而 \|Δz\| 反而差 +0.0195；
  收益 **60.1% 来自整体 normalization strength、仅 39.9% 来自 selective allocation**（兑换率差 5.7×）。
- 入口：`experiments/experiment6b/README.md`；commit `ad26ec5`

## N4 — 机制阶段（1J）的限制与冻结
- `experiment1j_b_transmission` 的 score 为 **proxy**（非 full-model 端到端），单 backbone；
  机制阶段在 1J 之后 **FROZEN**，未再推进（1J-C / 2A 未启动）。
- 入口：`experiments/experiment1j_b_transmission/README.md`；`experiments/experiment1j_feature_geometry/README.md`

## N5 — 7A-O：四类"简单可插拔模块"全部失败（今晚，92 GPU units）
- Family A（residual / energy-preserving fusion，5 configs）：全部"以 robustness 换 preservation"，
  **无一达到 Tier S/A/B**。
- Family B（dual representation，concat γ ∈ {0.25,0.5,1.0}）：同向、且 **被 A3_lam050 严格支配**；
  embedding dim 翻倍、峰值显存 ×2，并在 3-worker 下触发 **2 次 CUDA OOM**。
- Family C（layer-selective strength，4 configs）：与 Uniform 差异 ≤ 0.022 d′，**没有 layer 偏好信号**。
- Family D（LayerNorm-like / GroupNorm-like）：**被 B0_original 支配** → 明确否定。
- **A2（concat + fixed projection）= NOT IMPLEMENTED**：无训练固定投影只能是 α-interpolation 的伪装
  或无原则算子。
- **B2（grouped projection）= NOT RUN**：B1 已验证可跑，无需任意替代算子。
- 入口：`experiments/experiment7a_o/README.md`；`results/experiment_7a_o/summary/`；commit `f118e1e`

## N6 — Q2：layer × representation composition 同样无晋级（今晚，12 units）
- 0/4 配置达到 Tier S/A/B → STOP。
- 强非对称性：同一干预放在 **L3** 上会同时掉 d′ 与 robustness（`Q2L3` −0.0489 / +0.1184）。
- 入口：`results/experiment_7a_o_q2/README.md`；commit `7ffb3aa`

## N7 — 孤立点 / 非稳定区域清单
- 6A 的 `α_F = 0.25`（被 6B 判定为 region length = 1 的孤立点）
- 5A-H 的 `G2 = (L2 0.4527, L3 0.75)`（跨类别扩展后未能形成稳定规则）
- 7A-O 的全部 14 个候选（无一 Pareto-improving；仅 A3_lam050 相对 **弱** baseline 有支配关系）
