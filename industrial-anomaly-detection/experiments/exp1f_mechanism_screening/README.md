# Experiment 1F — Offline Defect-Attribute Mechanism Screening

**日期**：2026-10-02
**状态**：已完成
**VERDICT**: 🔴 **CASE_D — NO_EXPLAINABLE_STRUCTURE**

---

## 0. 启动四句话（冻结于 `results/experiment_1f/config.json`）

1. **怀疑**：1E 的 defect-dependent reshaping 可能与缺陷的可解释视觉属性（纹理/对比度/形态）有系统关系。
2. **做什么**：完全复用 1E（不重新 fit），从原图+GT mask 提取预注册属性，defect-type level 聚合后与 response signature 关联。
3. **看什么**：PRIMARY = Δdefect_std（1E 中方向分化最强：13 POS / 11 NEG / 1 MIXED）。
4. **继续标准**：至少一个属性与 Δdefect_std 方向明确、效应不弱，控制 area、LOCO 后仍保持 → mechanism candidate；否则 CASE C/D。

## 1. 预注册属性（4 组，11 个）

| 组 | 属性 | 说明 |
|---|---|---|
| size（控制变量） | area_ratio, log_area_ratio | 已有 |
| contrast | intensity_contrast, lab_contrast, local_variance_ratio | defect ROI vs surrounding ring (dilate 15px) |
| frequency | gradient_mean_defect, gradient_contrast, laplacian_energy, hf_energy_ratio | Sobel/Laplacian；hf = 缺陷区/ring 梯度能量比 |
| morphology | compactness, elongation | mask 几何 |

**技术偏差记录（冻结，非事后调整）**：协议原 FFT HF-ratio（Σ(f>fc)|F|²/Σ|F|²）经诊断对自然图像**无区分度**——去均值后功率谱近似 1/f 幂律，f>0.25·nyquist 环带面积占比 >99%，该比值恒 ≈1.0；频谱质心频率同理（区分度 <0.003）。改为「缺陷区 vs ring 的梯度能量比」（协议 GradientContrast 的能量版），实测区分度良好（print≈16, crack≈5.5, cable_swap≈0.07）。

**Sensitivity check（预注册）**：ring 依赖属性在 dilation 10/15/20 px 下 rho 方向与量级全部稳定（local_variance_ratio: -0.30/-0.24/-0.25；hf_energy_ratio: -0.198/-0.187/-0.198）。

## 2. 数据流

```text
401 个唯一样本（5 cat × 25 defect type，1E sample_level alpha=0 去重）
  → sample_attributes_d{10,15,20}.csv
  → defect-type median + IQR 聚合
  → merge 1E seed_summary（Δmean_gap/Δstd/Δz/Δd'）+ seed_stability（符号）
  → merged_attributes_responses.csv（n=25）
```

ROI 正确性：3 张可视化人工校验（`results/experiment_1f/roi_check/`）——mask 精确覆盖缺陷，ring 落在周围正常区域。

## 3. 结果

### A1 — Spearman（attr vs Δdefect_std，n=25）

| attr | family | rho | 95% CI |
|---|---|---:|---|
| local_variance_ratio | contrast | -0.244 | [-0.561, +0.173] |
| log_area_ratio | size | -0.236 | [-0.602, +0.171] |
| hf_energy_ratio | frequency | -0.187 | [-0.574, +0.243] |
| laplacian_energy | frequency | -0.168 | [-0.552, +0.241] |
| elongation | morphology | +0.172 | [-0.232, +0.531] |
| ...（全部 11 个） | | \|rho\|≤0.24 | **全部跨零** |

**无任何属性达到非弱门槛（|rho|≥0.5 且 CI 不跨零）。**

### A2 — Partial Spearman（控制 log area）

频率族方向一致负：laplacian -0.361 (p=0.076)、local_var -0.294、gradient_contrast -0.264、hf -0.248。控制 area 后关系增强而非消失，但无一显著。

### A3 — Leave-One-Category-Out

频率族 LOCO 方向 5/5 保持（去 grid 后 laplacian -0.22、hf -0.23、local_var -0.25）——**不是 grid 单类驱动的假象**。对照：area 的弱相关去掉 grid 后归零（+0.04），再次确认 area 无解释力。

### A4 — Shrink vs Expansion 组对比

10 个属性的 Mann-Whitney 全部 p>0.19，bootstrap CI 全部跨零。两组在视觉属性上**不可区分**。

### A5 — Response PCA（exploratory）

PC1（49.7% 方差）= 方差收缩←→膨胀轴；k=2 聚类把 5 个 grid + broken_large + missing_cable 与其余 18 个分开。**响应空间结构真实存在，但**：

- k=2 cluster 属性画像：10 属性全部 p>0.19；
- 双变量 rank-OLS（上限 2 变量防过拟合）：最好 log_area+laplacian adj R²=0.141，其余 ≤0.078，纯 area adj R²=0.000；
- 对比 1E：defect identity 饱和 R²=1.0，category 0.69-0.80。

## 4. Verdict（按冻结规则）

| Case | 条件 | 判定 |
|---|---|---|
| A | 稳定非弱关系 + 组间差异 | ✗（A1 全弱 + A4 全 null） |
| B | 去掉某 category 后消失 | ✗（频率族 LOCO 5/5 保持方向） |
| C | 单属性弱但组合属性有结构 | ✗（multivariate follow-up：cluster 画像 null + adj R²≤0.141） |
| **D** | **无可解释结构** | **✓** |

**CASE_D 的准确含义**：1E 的 heterogeneity 真实且稳定（25/25 seed-consistent），但**没有证据表明它具有简单、可泛化的 defect-attribute mechanism**。机制不在图像空间的简单属性（size/contrast/frequency/morphology）里。

**诚实记录的方向性 hint**：频率族控制 area 后方向一致的弱负相关（最强 laplacian partial rho=-0.361, p=0.076，LOCO/dilation 稳定）。「缺陷相对周围越『高频细碎』→ α-IN 越倾向压缩其 defect 方差」这个方向值得记住，但它解释的方差比例太小，不构成机制。

## 5. 这意味着什么（1F 的科学产出）

1. **排除了一整类廉价解释**：不是大小（1D/1E 已排除）、不是颜色/明暗对比、不是形态、也不是简单的频率结构。25 个看似"乱七八糟"的点背后，**没有一根用手工视觉属性画得出的线**。
2. **把机制指向 representation space**：响应结构真实存在（PC1 49.7%）却完全不被图像属性解释 → 下一步若有 1G，应分析 IN 对 PatchCore feature 统计量的直接作用（如 patch 特征范数/方差/余弦结构在 α-IN 下的变化），而不是继续在图像空间找属性。
3. **方法学价值**：这是一次"预注册机制筛查"干净失败的范例——假设被诚实检验、负结果被完整记录，避免了事后挑指标讲故事的陷阱。

## 6. 按协议停止

不进入方法设计（Defect-Preserving Invariance 等），不做 representation-level 分析的自动启动。是否开启 1G（feature-space 机制分析）由人工决策。

## 7. 文件

- 脚本：`scripts/experiment1f_attribute_extractor.py`、`experiment1f_roi_check.py`、`experiment1f_analysis.py`、`experiment1f_verdict.py`、`experiment1f_figures.py`
- 数据：`results/experiment_1f/sample_attributes_d{10,15,20}.csv`、`analysis/`（merged / a1-a5 / verdict.json）、`roi_check/`、`figures/`
- 冻结协议：`results/experiment_1f/config.json`
