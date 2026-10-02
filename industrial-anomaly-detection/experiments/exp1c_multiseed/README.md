# Experiment 1C: Multi-Seed Stability Validation

## Research Question

Does defect-specific sensitivity under α-IN remain stable across PatchCore random seeds?

## Hypothesis

The response pattern observed in Experiment 1B is not caused by random coreset sampling.

## Setup

- Dataset: MVTec AD bottle
- Model: PatchCore（wide_resnet50_2，layer2+layer3，coreset 0.1，num_neighbors 9，input 256）
- Seeds: 0, 1, 2
- Alpha: 0, 0.25, 0.5, 0.75, 1.0
- Defect types: broken_large / broken_small / contamination
- 阈值规则：tau_alpha = max(validation normal scores)（与 1B 一致）
- validation split：每 seed 从 209 train/good 划 20 张（seed 决定划分），189 进 bank
- 除 seed 外**所有条件与 1B 完全一致**；唯一允许变化的是 random seed

## 复用说明

严格复用 Experiment 1B 的 `experiment1b_defect_sensitivity.run_screening`（import 复用，
实验逻辑零改动），仅通过 `results_root` 参数按 seed 分目录：
`results/experiment_1c/seed_{N}/`。脚本：`scripts/experiment1c_multiseed.py`（驱动）、
`scripts/analyze_experiment1c.py`（跨 seed 稳定性分析）。

## Results

### d' 分离度（每个 seed × alpha）

| alpha | broken_large (s0/s1/s2) | broken_small (s0/s1/s2) | contamination (s0/s1/s2) |
|---:|---|---|---|
| 0 | 13.63 / 13.16 / 12.87 | 8.31 / 8.09 / 8.07 | 3.88 / 3.71 / 3.78 |
| 0.25 | 11.46 / 12.36 / 11.81 | 8.01 / 8.17 / 7.96 | 3.91 / 3.88 / 3.95 |
| 0.5 | 10.31 / 10.02 / 9.98 | 8.16 / 8.20 / 8.13 | 4.10 / 4.02 / 4.04 |
| 0.75 | 8.55 / 8.65 / 8.34 | 7.98 / 8.05 / 8.07 | 4.18 / 4.15 / 4.11 |
| 1.0 | 7.32 / 7.45 / 7.30 | 7.73 / 7.89 / 7.95 | 4.27 / 4.27 / 4.32 |

三个 seed 的曲线形态几乎重合（Figure 1）：
- broken_large 全部单调大幅下降（起点 12.9-13.6 → 终点 7.3-7.5）
- broken_small 全部基本平稳（8.0 附近微降）
- contamination 全部缓慢上升（3.7-3.9 → 4.3）

### Δd' = d'(α=1) − d'(α=0) 跨 seed 统计（sample std, ddof=1）

| defect | seed0 | seed1 | seed2 | mean | std |
|---|---:|---:|---:|---:|---:|
| broken_large | -6.31 | -5.71 | -5.57 | **-5.87** | **0.39** |
| broken_small | -0.58 | -0.20 | -0.12 | -0.30 | 0.24 |
| contamination | +0.40 | +0.56 | +0.54 | **+0.50** | **0.09** |

### Smoke 复现校验

seed=0 的 smoke test（每类 5 张）阈值 α=0 = 20.433，与 1B 全量记录的 validation max ~20.4
一致；validation split（seed=0）与 1B 完全相同。三个 seed 的 validation split 各不相同
（seed 机制生效），确认多 seed 验证真实覆盖了 split + coreset 两层随机性。

## Analysis

1. **现象跨 seed 高度稳定**：broken_large 的 Δd' 三个 seed 全部 < -5.5，std 仅 0.39
   （约为效应量的 7%）；contamination 三个 seed 全部为正（+0.40 ~ +0.56），std 仅 0.09。
2. **方向完全一致**：不存在协议中担忧的"seed0 降、seed1 升、seed2 平"的混乱模式；
   三类 defect 的相对排序（large 降 >> small 平 >> cont 升）在所有 seed 中保持不变。
3. **d' 曲线逐 α 对齐**：不仅端点（Δd'）稳定，整条 α-response 曲线在三个 seed 间几乎重合
   （Figure 1 中同色系三条线难以区分），说明 α 的作用模式本身是确定性的系统效应，
   随机性只带来小幅度水平抖动（±0.4 以内）。
4. **效应量层级**：|Δd'(large)| ≈ 12 × |Δd'(cont)|，两组效应的置信区间完全不重叠。

## Conclusion

**CONTINUE —— defect-specific α sensitivity 稳定存在，不是 coreset/feature 随机性的偶然产物。**

按预注册判断标准逐条核对：

- broken_large 三个 seed 全部明显下降：**通过**（-6.31 / -5.71 / -5.57）
- broken_small 变化较小：**通过**（|Δd'| ≤ 0.58）
- contamination 方向不同或不下降：**通过**（全部为正）
- broken_large std 较小：**通过**（0.39 << 1.0）

Experiment 1B 的观察正式获得多 seed 证据支持："α-IN 对不同 defect type 的差异化影响"
是稳定、可复现的系统性现象。

## Next Step

按协议进入 **Experiment 1D: Size Confound Analysis**——用 matched-area 比较彻底剥离
size confound，回答「broken_large 与 broken_small 的响应差异有多少来自缺陷面积本身」。

（注意：1B 已观察到类内 corr(area, delta) ≈ -0.73~-0.75 的 size 效应，1D 的核心问题是
控制面积后 type 效应是否仍然存在。）

## 文件

- 驱动脚本：`scripts/experiment1c_multiseed.py`（复用 1B `run_screening`，仅加 seed 循环与目录隔离）
- 分析脚本：`scripts/analyze_experiment1c.py`（d' 表、Δd' 统计、Figure 1/2、verdict）
- 结果：`results/experiment_1c/seed_{0,1,2}/`（结构与 1B 一致）+ `summary/`（dprime_all_seeds.json、
  delta_dprime_stability.csv、verdict.json）+ `figures/`（dprime_curves_per_seed.png、
  delta_dprime_stability.png）
- 配置：`configs/seed_{0,1,2}.yaml`
- 日志：`results/exp1c_smoke.log`（smoke）、`results/exp1c_full.log`（全量）
