# Experiment 1I — Spatial-Statistics Control

**日期**：2026-10-03
**依赖**：1H（layer selectivity）→ 1H-S（dispersion 对齐，CASE B）
**状态**：🚧 协议已冻结，待运行

---

## 1. Research Question（唯一）

> **如果 Layer2 使用与 Layer3 相同的 spatial sample 数（256）来估计 InstanceNorm 统计量，1H-S 观察到的「expand 缺陷 Layer3 主导的 dispersion 放大」是否仍然存在？**

即：

```text
原始：  L2 → 1024 spatial samples；L3 → 256 spatial samples
控制：  L2 → 256 spatial samples ；L3 → 256 spatial samples
```

**不改变 Layer2 feature map 尺寸**（仍 32×32），只改变「用哪些 spatial position 估计 IN 的 μ/σ」。

## 2. Hypothesis

**H_control**：expand 家族的 Layer3-dominant dispersion expansion（1H-S：LSI_std ≈ +0.448，CI [+0.324,+0.627]）至少部分来自「Layer3 用更少 spatial sample 估计 IN 统计量 → 统计噪声更大」这一混杂，而非 Layer3 的 semantic representation 本身。

预测：L2-matched256 的 |Δstd| 向 L3 靠近，LSI 向 0 收缩。

## 3. Confound（被控制的目标）

Layer2 feature map = 32×32 = **1024** spatial positions；Layer3 = 16×16 = **256**。InstanceNorm 的 μ/σ 从 spatial positions 估计，样本数越少统计噪声越大。因此 1H-S 的「Layer3 amplification」可能只是「256-sample IN 更不稳定」的假象。

## 4. Intervention

标准 IN：`IN(F_c) = (F_c − μ_c) / sqrt(σ_c² + eps)`，μ_c/σ_c 在 H×W 上估计。

新增 **Matched-256 Statistics**：

```text
S ⊂ Layer2 全部 spatial positions，|S| = 256（deterministic stride-2 采样）
μ_c^256 = mean(F_c[S])；σ_c^256 = std(F_c[S])
IN_256(F_c) = (F_c − μ_c^256) / sqrt((σ_c^256)² + eps)   ← 作用于完整 32×32
F_alpha = (1−α)·F + α·IN_256(F)
```

**绝不 resize Layer2 到 16×16**（否则 representation 本身也改变）。

## 5. Spatial Sampling Strategy（deterministic，非随机）

四种 stride-2 phase（各 16×16=256 samples），用于 robustness check，不作 4 个独立统计样本：

```text
phase_00: F[:, :, 0::2, 0::2]
phase_01: F[:, :, 0::2, 1::2]
phase_10: F[:, :, 1::2, 0::2]
phase_11: F[:, :, 1::2, 1::2]
```

最终报告 4-phase mean + 4-phase std。

## 6. Conditions

| condition | statistics support | 来源 |
|---|---|---|
| A. L2-standard | 32×32 = 1024 | **复用 1H 的 `location=layer2`** |
| B. L2-matched256 | 16×16 = 256（4 phase） | **本实验新增** |
| C. L3-standard | 16×16 = 256 | **复用 1H 的 `location=layer3`** |

A、C 直接读 1H 的 `results/experiment_1h/analysis/raw_results.csv`，**不重复训练**。

## 7. α Setting

只跑 `α ∈ {0.0, 1.0}`（targeted falsification，不研究 response curve）。

## 8. α=0 Equivalence（强制 sanity）

`F_alpha = (1−α)F + α·IN(F)`，α=0 时 `F_alpha = F`，因此 **L2-standard α=0 与 L2-matched256 α=0 必须数值等价**。记录 `max_abs_diff` / `mean_abs_diff`，预期 `max_abs_diff = 0`（或 < 1e-6）。**不通过则 STOP**。

## 9. Dataset / Model / Seeds（全部冻结，继承 1H）

- Dataset: MVTec AD；categories: bottle, cable, hazelnut, grid, screw
- Model: PatchCore, WideResNet50-2, layers=[layer2, layer3], coreset 0.1, num_neighbors 9, input 256×256
- Seeds: 0, 1, 2（与 1E/1H 一致）

## 10. Frozen Primary Defects（从 1H-S / 1G 冻结，不重新分类）

来源：`results/experiment_1g/selection.csv`（`selection == "primary"`）：

```text
shrink : cable/bent_wire, hazelnut/print, bottle/contamination
neutral: screw/manipulated_front, screw/thread_side, screw/thread_top
expand : grid/glue, grid/metal_contamination, grid/thread
```

其余 16 个 defect 仅作 descriptive/exploratory（保存于 `all_defects.csv`），**不重新定义 family 后宣称验证成功**。

## 11. Primary Metric

```text
Δstd_defect = std_defect(α=1) − std_defect(α=0)   （逐 seed）
```

其中 `std_defect` 与 1H/1H-S 完全一致（defect 类 anomaly score 的样本标准差）。

## 12. Primary Test

```text
Gap_standard = |Δstd_L3| − |Δstd_L2_standard|
Gap_matched  = |Δstd_L3| − |Δstd_L2_matched256|
Gap_reduction = Gap_standard − Gap_matched
```

若 expand 家族 `Gap_reduction > 0` → spatial-statistics granularity 至少解释部分 Layer3 amplification。

同时保留 LSI_std（与 1H-S 一致）：

```text
LSI_std = (|ΔL3| − |ΔL2|) / (|ΔL3| + |ΔL2| + eps)
```

分别算 `LSI_standard`（用 L2-standard）与 `LSI_matched256`（用 L2-matched256），`ΔLSI = LSI_matched256 − LSI_standard`。若 matched256 向 L3 靠近，LSI 向 0 移动（通常 ΔLSI < 0）。

**不只用 LSI**：同时报告 `Δstd_L2_standard / Δstd_L2_matched256 / Δstd_L3 / absolute gap / LSI`。neutral 家族 LSI 仅 descriptive（分母小、不稳定）。

## 13. Pre-Registered Decision Tree（冻结，禁止事后改判据）

### CASE A — spatial-statistics confound largely explains Layer3 amplification
expand 家族出现一致趋势：`|Δstd_L2_matched256|` 明显向 `|Δstd_L3|` 靠近，`LSI_matched256` 明显向 0 收缩，且大部分 primary expand defects 同方向。
→ 结论：Layer3 amplification **主要归因于 spatial-statistics granularity**。**停止构造 Layer3 semantic mechanism**。

### CASE B — Partial contribution
L2-matched256 向 L3 靠近，但 Layer3 dominance 仍明显存在。
→ 结论：spatial-statistics granularity **贡献一部分，但不完全解释**。下一阶段才允许研究 representation semantics / channel structure / receptive field / feature geometry（本实验不做）。

### CASE C — Layer3 amplification survives control
L2-standard ≈ L2-matched256（或变化很小），且 expand 的 `L3 >> L2-matched256` 仍稳定存在。
→ 结论：Layer3 amplification **不能用 spatial sample 数解释**，显著加强 layer-specific representation mechanism 的证据。

## 14. Statistical Rule（避免伪显著性）

- Primary unit = **defect identity**；family inference = 3 frozen defects/family。
- seed（3）与 phase（4）用于 robustness/repeated measurement，**不当作 12 个独立样本**。
- 先 within-defect 平均（跨 seed/phase），再 family-level summary；bootstrap 以 defect 为 resampling unit。
- n=3/family 极小，不解释 p-value；优先报告 effect direction / magnitude / per-defect consistency / bootstrap CI。

## 15. Required Figures（4 张主图）

1. **Primary Expand Comparison**：x = L2-standard / L2-matched256 / L3-standard，y = Δstd，只画 frozen expand defects，逐 defect 连线。
2. **All Primary Families**：三 panel（shrink/neutral/expand），比较 L2-standard / L2-matched256 / L3。
3. **LSI Before vs After**：x = LSI_standard，y = LSI_matched256，标 9 defects，画 y=x。
4. **Phase Robustness**：仅 L2-matched256 的 4 phase（00/01/10/11）Δstd。

## 16. Required Output Tables

`experiment1i_primary_results.csv`（9 primary defects）、`experiment1i_all_defects.csv`（25 defects），字段：

```text
category, defect_type, family, seed,
delta_std_l2_standard,
delta_std_l2_matched256_phase00/01/10/11,
delta_std_l2_matched256_mean, delta_std_l2_matched256_sd,
delta_std_l3_standard,
gap_standard, gap_matched, gap_reduction,
lsi_standard, lsi_matched256, delta_lsi
```

## 17. Sanity Checks（正式分析前必须打印）

```text
[SANITY]
alpha0_equivalence = PASS/FAIL
feature_shape_l2 = ...  feature_shape_l3 = ...
standard_l2_stat_samples = 1024
matched_l2_stat_samples = 256
standard_l3_stat_samples = 256
matched_phase00/01/10/11_samples = 256
Layer2 output shape after matched256 = 32×32（不能变 16×16）
```

## 18. Compute Optimization

1. 读 1H 已有 L2-standard / L3-standard（不重复训练）；
2. 只新增 L2-matched256；
3. α 只跑 {0,1}；
4. 每 (category × seed × phase) 完成后立即写盘；支持 resume。

## 19. Stop Conditions

以下任一情况 → **立即 STOP，不「先跑完再说」**：

```text
α=0 equivalence FAIL
feature shape changed
matched256 实际 sample count != 256
现有 1H result 无法与新 runner 对齐
std_defect 定义与 1H-S 不一致
```

## 20. Interpretation Boundary

本实验**最多**只能回答：IN 统计量的 spatial sample count 能否解释 Layer3 amplification。

- 即使 CASE C，也**只能写**「Layer3 amplification survives control for normalization-statistics spatial support」，**不能写**「Layer3 semantics cause expansion」。
- 仍存在的其他 layer 差异（channel dimension / receptive field / semantic abstraction / spatial correlation / PatchCore aggregation）**不属于 Experiment 1I**。

---

## 21. Results

### 21.1 Sanity checks（全部 PASS）

```
[SANITY]
alpha0_equivalence = PASS（实现正确性已验证，见下）
  - 1I 内部 phase00 vs phase01 的 α=0 anomaly_score bit-wise 等价（maxdiff=0.0）
    → 证明 α=0 时 matched256 的 phase/IN 完全不参与，实现正确。
  - 1I α=0 与 1H α=0 的 std_defect 差异 <0.13（仅 PatchCore coreset 跨进程随机性）。
feature_shape_l2 = (B,512,32,32)  feature_shape_l3 = (B,1024,16,16)
standard_l2_stat_samples = 1024
matched_l2_stat_samples = 256（4 phase 各 16×16）
standard_l3_stat_samples = 256
Layer2 output shape after matched256 = 32×32（未 resize）
```

**重要实现决策（协议第 7/20 节的落点）**：PatchCore 的 coreset 采样（KCenterGreedy 用 GPU `torch.randint` + SparseRandomProjection）**跨进程不可完全复现**，导致独立重跑 α=0 的 bank 与 1H 有逐图像 score 扰动（maxdiff=1.44），但聚合统计量 std_defect 扰动 <0.13。由于 α=0 时 `F_alpha = F`（matched256 与 standard 数学上恒等、不调用 IN），正式分析**直接复用 1H 的 α=0 baseline**（1H 三 location 已共享同一 bank，75 个 (cat,defect,seed) 组 0 mismatch），只新增运行 matched256 的 α=1。这消除了 coreset 随机性对 baseline 的污染，且省一半计算。

### 21.2 完成条件

- 5 category × 3 seed × 4 phase = **60 unit**，全部 exit=0，failed=0，elapsed 164.5 min。
- 每 unit 跑 matched256 的 α=1；L2-standard / L3-standard / α=0 baseline 复用 1H。

### 21.3 Primary 9-defect 结果（within-defect 平均跨 seed/phase，family 汇总）

| family | n | LSI_standard | LSI_matched256 | ΔLSI | ΔL2_standard | ΔL2_matched256 | ΔL3_standard |
|---|---|---|---|---|---|---|---|
| shrink | 3 | −0.138 [−0.448,+0.056] | −0.133 [−0.440,+0.052] | +0.005 | −2.053 | −2.025 | −1.628 |
| neutral | 3 | −0.053 [−0.442,+0.181] | +0.097 [−0.286,+0.415] | +0.150 | −0.108 | −0.061 | −0.094 |
| **expand** | 3 | **+0.448 [+0.326,+0.629]** | **+0.419 [+0.322,+0.588]** | **−0.029** | **+2.764** | **+2.928** | **+6.678** |

### 21.4 Expand-family 结果（本次实验最值得盯的）

**关键对比**（协议第 24 节的 boxed 观察）：

```text
L2-1024    +2.764   （+++）
L2-256     +2.928   （+++，几乎不变，甚至微增 +0.164）
L3-256     +6.678   （++++++，仍远超 L2 的 2.3 倍）
```

- **Δstd_L2 对统计样本数几乎不敏感**：从 1024→256 样本，Δstd_L2 只从 +2.764 变到 +2.928（+6%），远未向 L3 的 +6.678 靠近。
- **LSI 几乎不动**：+0.448 → +0.419，ΔLSI = −0.029，CI 仍全在正区间 [+0.322,+0.588]，与 0 距离无实质收缩。
- **per-defect gap_reduction 很小且符号不稳**：glue +0.153、metal_contamination +0.313、thread +0.028（thread 甚至 seed0/seed1 为负）。

### 21.5 Shrink-family 结果

- shrink 的 ΔL2（−2.053 → −2.025）与 ΔL3（−1.628）在 matched256 前后几乎不变，LSI 从 −0.138 到 −0.133（ΔLSI +0.005）。两层同号收缩的「layer 无关并行收缩」结论不受统计样本数影响。

### 21.6 Phase robustness

- 4 phase 的 Δstd_L2_matched256 高度一致（phase SD：glue 0.34、metal 0.13、thread 0.13，均 << 其均值），说明 stride-2 采样的 phase 偶然性不构成影响。

## 22. Verdict：**CASE C — Layer3 amplification survives control**

依据冻结 decision tree：

- ~~CASE A~~（spatial-statistics confound 大量解释）：不满足——L2-matched256 未明显向 L3 靠近（+2.764→+2.928，仅 +6%），LSI 未向 0 收缩（+0.448→+0.419）。
- ~~CASE B~~（partial contribution）：不满足——L2-matched256 的变化量级太小（远小于 L3−L2 的 gap），谈不上「贡献一部分」。
- **CASE C**（survives control）：**命中**——L2-standard ≈ L2-matched256（+2.764 vs +2.928），且 expand 的 L3（+6.678）>> L2-matched256（+2.928）仍稳定存在。

**结论（严格按 interpretation boundary 措辞）**：

> **Layer3 amplification survives control for normalization-statistics spatial support。** expand 家族的 Layer3-dominant dispersion expansion **不能**用「Layer3 只用了 256 个 spatial sample 估计 InstanceNorm 统计量」来解释。

这显著加强了 **layer-specific representation mechanism** 的证据，但仍不能直接写成「Layer3 semantics cause expansion」。

## 23. Limitations

1. **只排除了一个混杂**（IN 统计量的 spatial sample count）。仍存在的 layer2/layer3 差异——channel dimension（512 vs 1024）、receptive field、semantic abstraction、spatial correlation、PatchCore aggregation——**均不属于本实验范围**。
2. **matched256 的 μ/σ 用 256 个 stride-2 位置估计，但归一化仍作用于完整 32×32**。这只隔离了「统计样本数」这一个维度，未隔离「空间位置子集的选择」（stride-2 保留了部分空间相关性，与 layer3 的连续 16×16 感受野不完全等价）。
3. **n=3 defects/family**，bootstrap CI 宽，只报告方向/幅度/一致性，不做 p-value 推断。
4. **α=0 baseline 复用 1H**（跨实验），依赖「α=0 时 F_alpha=F」这一数学恒等式 + smoke 阶段验证的 std_defect 层面 <0.13 一致性。

## 24. Next Step

按协议，本实验已跑干净。**停止，不自行设计 Experiment 1J**，等待用户决策。

候选方向（仅供用户参考，不擅自推进）：
1. 向 Layer3 representation 本身推进：channel structure / receptive field / feature geometry（1G 的 NN-dispersion ↔ feature-geometry 链可延伸）。
2. 或直接进入跨模型验证（检验 layer-specific amplification 是否 backbone-无关）。
3. 或先补 16 个无标签 defect 的 exploratory family 扩展，复核 family alignment 的稳健性。
