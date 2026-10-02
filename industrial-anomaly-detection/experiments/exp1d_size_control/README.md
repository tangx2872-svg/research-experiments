# Experiment 1D — Size-Controlled Defect Sensitivity Analysis

## Research Question

控制 defect area 后，defect type 是否仍能解释 PatchCore α-IN sensitivity 的差异？

即：Experiment 1B/1C 观察到的三类缺陷 α-response 分化（broken_large 强降、broken_small 稳定、contamination 反向），究竟是"缺陷类型"本身造成的，还是只因为 broken_large 的缺陷面积大？

## Why Experiment 1D is needed

1B 同时发现类内 corr(defect_area, response_delta)：broken_large ≈ -0.747、broken_small ≈ -0.725、contamination ≈ +0.196，且粗回归 area-only R²=0.130 < type-only R²=0.292 < area+type R²=0.324。**DEFECT SIZE 是当前最大 confound**。1C 已排除 seed 随机性，1D 负责剥离 size。

## Four pre-registered questions（协议原文）

- **A.** defect area 能解释多少 α-response variation？
- **B.** 控制 area 后，defect type 是否还能提供额外解释？
- **C.** 面积相近的 defect samples 中，不同 defect type 是否仍表现出不同 α-response？
- **D.** contamination 的正向 response 是否仍无法被 size 单独解释？

## Data source

完全离线复用，未重新 fit/predict，未修改 PatchCore / α-IN / 任何实验逻辑：

- 逐样本 anomaly score：`results/experiment_1c/seed_{0,1,2}/raw/all_sample_scores.csv`（与 1B 同源，seed=0 已验证与 1B 一致）
- GT area：`results/experiment_1b/summary/area_analysis.csv`（由 `data/mvtec_ad/bottle/ground_truth/{type}/{stem}_mask.png` 计算，**非 predicted mask**）
- 数据审计：`results/experiment_1d/summary/data_audit.json` —— 63 样本（20/22/21），缺失 mask=0，area=0 的 defect=0，重复 sample=0，三 seed test samples 完全一致，area ∈ [0.0058, 0.2742]

## Response definition（冻结口径，未更改）

normal reference = **test/good 20 张**，每 (seed, alpha) 独立计算：

```
z_i(seed, alpha) = [score_i(seed, alpha) - mu_good(seed, alpha)] / sigma_good(seed, alpha)
```

- **PRIMARY（sample-level）**: `delta_z_i = z_i(alpha=1) - z_i(alpha=0)`
- **SECONDARY（sample-level）**: `slope_z_i`（z 对 5 个 α 线性拟合斜率）
- **GROUP-LEVEL CONTEXT**: 1B/1C 的 `d'(alpha) = (mu_D - mu_G) / pooled_std`，仅作群体分离度参照，**不作为单样本 sensitivity**
- 主分析用 per-image 三 seed mean；各 seed 单独作 robustness check；**未把 63×3 当 189 独立样本**

## Area definition

`defect_area_ratio = GT defect pixels / total image pixels`（冻结，未更改）。

## Metric decomposition sanity check（执行中追加的诊断，非 outcome 更换）

发现 Δz（1D PRIMARY）与 1C Δd' 方向不一致后，**未更换 PRIMARY**，而是按预注册口径继续，同时增加分解诊断（`summary/metric_decomposition.csv`、`figures/metric_decomposition.png`，脚本 `scripts/metric_decomposition_check.py`）。

对每个 (seed, defect_type, alpha) 输出 mu_good / std_good / mu_defect / std_defect / mean_gap / mean_sample_z / pooled_std / dprime。α=0 → 1 的变化（3-seed mean）：

| defect type | Δ mean_gap | Δ std_good | Δ std_defect | Δ mean_z | Δ d' |
|---|---:|---:|---:|---:|---:|
| broken_large | -8.44 | +0.17 | **+1.73** | -4.56 | -5.72 |
| broken_small | -3.40 | +0.17 | **-0.50** | -2.44 | -0.30 |
| contamination | -2.44 | +0.17 | **-2.30** | -1.95 | **+0.48** |

**诊断结论（4 问回答）：**

1. **broken_small Δz 降得多而 d' 只轻微下降**：其 defect 方差在 α 增大时收缩（Δσ_D=-0.50），pooled_std 变小托住了 d'；Δz 分母只有 σ_good，看不到该变化。
2. **contamination Δz 为负而 d' 为正**：contamination 的 defect 方差剧烈收缩（Δσ_D≈-2.3，三 seed 一致：-2.09/-2.43/-2.40），方差收缩效应强于 mean_gap 下降效应，d' 上升。
3. **分歧主要来自 defect variance**：mean_gap 三类同向下降；σ_good 三类同向上升且幅度几乎一致（+0.17）；**α-IN 对三类缺陷各自方差的影响方向不同**（large 增方差 / small 轻微缩 / contamination 强烈缩），这是类型分化的真正载体。
4. **三 seed 高度一致**，尤其 contamination 的 Δσ_D 与 Δd'>0 在所有 seed 下方向不变。

> 这本身是一个新发现：**"α-IN sensitivity" 不是单一统计量可描述的**——它同时作用于 normal variance、defect mean、within-defect variance 三个分量，且对不同 defect type 的作用结构不同。d' 把三个分量混在一起；Δz 只反映相对 contemporaneous good 分布的标准化距离。

## Statistical analysis

按协议顺序全部执行（脚本 `scripts/analyze_experiment1d_size_control.py`，统计规则在看到 outcome 前锁定，**执行中未更改任何规则**）：

1. area distribution + common support（Figure 1）
2. area vs Δz 相关（Pearson + Spearman，Figure 2）
3. OLS 回归：Model A `Δz ~ area`、Model B `Δz ~ type`、Model C `Δz ~ area + type`
4. matched-area comparison（greedy nearest-neighbor，无放回，caliper-bounded）
5. contamination targeted analysis（Figure 4）
6. residual plot（area-only 回归残差按 type，Figure 5）
7. multi-seed robustness（各 seed 独立重复主要分析）

## Matching rule（预注册，未调整）

- 方法：nearest-neighbor matching，仅基于 defect_area_ratio，无放回（greedy）
- **caliper = 全部 63 样本 area 的 MAD = 0.03744**（在看到任何 Δz 结果之前确定）
- 匹配前样本：large 20 / small 22 / cont 21；匹配后：

| pair | matched pairs | mean Δz_diff (A−B) | mean abs area diff |
|---|---:|---:|---:|
| broken_large vs broken_small | 5 | +0.89 ± 0.94 | 0.006 |
| broken_large vs contamination | 15 | **-2.23 ± 3.52** | 0.015 |
| broken_small vs contamination | 13 | +0.55 ± 2.98 | 0.021 |

- common support（面积区间重叠占较小 range 比例）：large-small 0.553 / large-cont 0.780 / small-cont 1.000，**三对均有足够 overlap，无需宣布 "not reliable"**
- large-small 仅 5 对（large 的面积分布整体大于 small），该对结论仅作参考

## Results

### Q1（Q-A）：area 分布差异大，但三对均有 common support

| type | n | mean | median | min | max |
|---|---:|---:|---:|---:|---:|
| broken_large | 20 | 0.117 | 0.119 | 0.041 | 0.274 |
| broken_small | 22 | 0.031 | 0.024 | 0.009 | 0.081 |
| contamination | 21 | 0.085 | 0.073 | 0.006 | 0.165 |

### Q2：area 与 Δz 的关系是 type-dependent 的

| group | n | Pearson r | Spearman ρ |
|---|---:|---:|---:|
| overall | 63 | -0.339 | -0.327 |
| broken_large | 20 | **-0.758** | -0.761 |
| broken_small | 22 | **-0.747** | -0.694 |
| contamination | 21 | **+0.211** | +0.195 |

两类 broken 类内均强负相关（面积越大掉得越多），contamination 类内**弱正相关**——area 与 sensitivity 的关系本身随 type 改变，这已提示 area 不是完整解释。

### Q3（Q-A）：area-only 解释有限

Model A `Δz ~ area`：**R² = 0.115**（adj 0.086）。

### Q4（Q-B）：type 提供显著额外解释力

| Model | R² | adj R² |
|---|---:|---:|
| A: area | 0.115 | 0.086 |
| B: type | 0.279 | 0.242 |
| C: area + type | 0.304 | 0.256 |

**ΔR²(type|area) = 0.188**（加入 type 后 R² 从 0.115 → 0.304，近乎三倍）
**ΔR²(area|type) = 0.025**（控制 type 后 area 几乎不再增加解释力）

不对称结论：**type 解释力无法被 area 替代，而 area 解释力大部分被 type 吸收。**

### Q5（Q-C）：matched-area 后 type 差异仍存在

- **large vs contamination（15 对，最可靠）**：同面积下 broken_large 的 Δz 比 contamination 低 **2.23**——面积相同时，contamination 仍显著更耐 α-IN。
- area-only 回归残差（Figure 5）：broken_large **-1.10**、broken_small -0.06、contamination **+1.11**——控制面积后 type 残差仍明显分化。

### Q6（Q-D）：contamination 不能被 size 单独解释

- 面积匹配后（28 对 contamination 相关对）：cont Δz mean = **-2.24** vs matched broken = **-3.18**，contamination 仍整体高于 matched broken。
- contamination 类内 area-Δz 相关为 **+0.21**（与两类 broken 的 -0.75 方向相反），area 单变量无法拟合出这种反向模式。
- **注意**：在 Δz 口径下 contamination 的 Δz 三 seed mean = **-1.95**（不是正数）——sample-level 上它也下降，只是显著慢于 broken 类。1C 中 contamination d' 上升是 group-level 现象，主要由其 defect 方差剧烈收缩驱动（见 decomposition 章节）。

### Multi-seed robustness

各 seed 独立重复：overall corr = -0.33 / -0.34 / -0.35；type 均值排序三个 seed 完全一致（broken_large 最负、contamination 最不负）。结论方向不依赖单一 seed。

## A/B 双口径诚实报告（协议要求）

- **A. Sample-level（Δz，PRIMARY）**：控制 area 后 defect type 仍提供显著额外解释力（ΔR²=0.188），matched-area 后 large vs contamination 仍差 2.23，contamination 残差 +1.11。三类 Δz 均为负，但分化结构稳定。
- **B. Group-level（d'，context）**：1B/1C 的 d' 分化（large 强降 / small 稳 / cont 上升）同样跨 seed 稳定，但 d' 混入了 within-defect variance 的变化——contamination 的 d' 上升主要由其 defect 方差收缩驱动。
- **二者不一致本身被记录为实验发现**：α-IN 对 defect 分布的作用至少有三个分量（mean separation / within-defect variance / normal variance），不同 defect type 的响应结构不同。未选择"更符合原假设"的指标作为真相。

## Limitations

1. n=63（单类别 bottle），回归仅 2-4 参数，CI 宽；未做 p 值为主的显著性声明，以 effect size 为主。
2. large-small 匹配仅 5 对，该对结论不可靠。
3. Δz 依赖 test/good（n=20）估计 μ/σ，good 分布估计噪声进入每个 z 值。
4. matched-area 是观察性控制，不能像随机实验一样完全排除残余 confound（如缺陷位置、纹理）。
5. Δz 与 d' 的 metric dependence 表明"敏感性"的操作化定义本身影响结论；本研究以 Δz 为预注册 PRIMARY，d' 仅作 group-level context。
6. 单类别、单 backbone（WideResNet-50）、单 detector（PatchCore），外推性未知。

## Verdict

**CASE_A — TYPE EFFECT REMAINS**

控制 defect area 后：
- ΔR²(type|area) = 0.188（type 仍提供显著额外解释力）
- matched-area 后 broken_large vs contamination 仍差 -2.23
- 残差分析：large -1.10 vs contamination +1.11，方向相反且分离清晰
- contamination 的响应模式（类内正相关 + 匹配后更耐）无法由 area 单独解释

同时以 Discussion 记录：group-level d' 的"contamination 上升"部分来自 within-defect variance 收缩，因此更准确的表述是 **defect type 影响 α-IN 对缺陷分布的完整作用结构（mean + variance）**，而非单一方向的"sensitivity 上升/下降"。

## Next Step

**Experiment 1E — Cross-Category Validation**（跨类别验证，cable/screw 等），检验 type effect 与 metric decomposition 结构是否跨类别成立。本实验不自动开始，等待确认。

## Files

- 脚本：`scripts/analyze_experiment1d_size_control.py`（主分析）、`scripts/metric_decomposition_check.py`（分解诊断）
- summary：`results/experiment_1d/summary/`（data_audit.json / area_statistics.csv / area_correlation.csv / regression_results.json / matched_pairs.csv / metric_decomposition.csv / verdict.json）
- figures：`results/experiment_1d/figures/`（5 张协议图 + metric_decomposition.png）
- tables：`results/experiment_1d/tables/sample_level_response.csv`
