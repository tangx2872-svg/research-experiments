# Paper Evidence Map（Claim → 支持实验 → 支持数字 → 反例/限制 → 证据强度）

> 由 overnight Q5（CPU-only）整理；所有数字均可在对应 experiment README / summary 中核验。
> 证据强度标记：**SUPPORTED / PARTIALLY SUPPORTED / REJECTED / EXPLORATORY**。

## C1 — 光照/对比度扰动会显著改变 PatchCore 的 anomaly score，且不同缺陷类型受影响不同
- 支持实验：1B（phenomenon）、1C（multi-seed）、1D（size control）、1E（cross-category）
- 支持数字：见 `experiments/exp1c_multiseed/README.md`、`experiments/exp1d_size_control/README.md`、
  `experiments/exp1e_cross_category/README.md`（本文件不重复引用未核实的数字）
- 反例/限制：单 backbone（wide_resnet50_2）、n=5 categories、扰动为 synthetic photometric
- 强度：**SUPPORTED**

## C2 — 简单缺陷属性不足以解释上述差异
- 支持实验：1F（CASE_D）
- 强度：**REJECTED**（对"简单属性解释"这一替代解释的否定）

## C3 — normalization（InstanceNorm 混合）强度是 preservation ↔ robustness 的一维旋钮
- 支持实验：5A、5A-H（fixed/uniform α 网格）、6B（mean-α matched Uniform 对照）
- 支持数字（full panel 5 cat × 3 seeds，本会话核实）：α=0 → d′ 4.9852 / \|Δz\| 0.3105；
  α=0.5 → 4.6670 / 0.2082；**α=0.40091275 → 4.7940 / 0.2134**
- 反例/限制：该旋钮是**单一维度**，不能同时改善两轴（见 C5）
- 强度：**SUPPORTED**（机制层面）/ **EXPLORATORY**（最优 α 的具体数值）

## C4 — geometry-guided category-adaptive α 有价值
- 支持实验：5C（CASE_A）
- 反例：5D（identity 未确立，实质 CASE_B）→ 6A/6B（knee 是孤立点、收益 60.1% 来自 strength）
- 强度：**REJECTED**（作为独立机制；保留为"strength 效应"的从属解释）

## C5 — 简单 representation 模块（residual / dual / layer-selective / alternative-norm）可以推动
preservation–robustness frontier
- 支持实验：7A-O（92 units）、Q2（12 units）
- 支持数字（7A-O full panel vs Uniform）：A3_lam050 +0.5612 d′ / **+0.0619** \|Δz\|；
  B1_g100 +0.5104/+0.0717；D1 +0.1739/+0.1164；C3 +0.0219/+0.0115
- **反例（关键）**：**无任何 candidate 与 Uniform 相互 Pareto 支配**；所有模块只是沿既有 frontier
  移动工作点。唯一正面关系是 A3_lam050 **严格支配 B0_original**（+0.3700 / −0.0352）。
- 强度：**REJECTED**（作为"突破 frontier"的路径）/ **EXPLORATORY**（A3_lam050 相对弱 baseline 的支配关系）

## C6 — "只改 feature representation 就能提高 illumination robustness"这一方向本身
- 支持实验：6B（strength 主导）+ 7A-O（4 family 全败）+ Q2（layer × representation 无晋级）
- 含义：三条独立证据一致指向：**在 PatchCore + IN 框架内，representation-level 干预不能改变
  frontier 的形状，只能选择工作点**。
- 反例/限制：未测试"改变目标轴本身"的方案（inference-time illumination 校正 / 显式
  illumination-invariance 目标）——**尚未验证**。
- 强度：**PARTIALLY SUPPORTED**（否定 simple representation 路线；不否定所有可能的语言）

## C7 — μ 表述（可用于论文的限制声明）
- 上述结论的适用边界：single backbone（wide_resnet50_2）、PatchCore、coreset 0.1、kNN 9、
  MVTec AD 5 categories、synthetic photometric perturbation、seeds 0/1/2。
- 强度：**限制声明（必须随结论一起写）**
