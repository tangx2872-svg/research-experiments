# CSEM-MISD Experiment 2 — 实验 Pipeline 设计（只设计，不实现）

> 本文件对应任务指令 Task 4 / Task 5。**当前阶段不写代码、不改模型、不训练。**
> 数据未下载，因此所有设计为预研规划，需在数据就绪后进入实现。

---

## Experiment Question

> 真实工业多光照条件下，不同缺陷（受限于标签，现阶段为**不同 specimen / 缺陷可见性**）
> 是否对 illumination/style invariant representation（α-IN）表现出不同敏感性？

---

## Variables

### 固定（fixed）

```text
Model:      PatchCore
Backbone:   wide_resnet50_2（沿用已有实验，便于与 MVTec Bottle 结果对比）
Dataset:    CSEM-MISD（gear 或 screw，待定）
预处理:     与现有实验一致（resize 256 + ImageNet normalize；灰度图需复制为 3 通道）
```

### 变化（varied）

```text
alpha:  0 / 0.5 / 1.0    （复用 F_alpha = (1-α)F + α·IN(F)，InstanceNorm affine=False）
illumination:  光照子集（从 108 个真实方向中选取代表性子集，待定）
```

---

## Evaluation（不直接比较 anomaly score）

> 教训（来自 Experiment 1）：跨 α 的绝对 anomaly score **不可直接比较**
> （InstanceNorm 改变特征/分数尺度）。必须用秩相关指标 + 同 α 内分离度 + 配对差值。

### 1. defect–normal separation（同 α 内）

- **image-level**：AUROC、AUPR（对单调尺度变换不敏感，跨 α 可比）。
- **pixel-level**：AUROC、AUPR（用静态缺陷 mask 评估定位）。
- **d-prime**（同 α 内）：
  `d' = (mean_score(defect) − mean_score(normal)) / std_score(normal)`，
  用于在绝对分数尺度不同的情况下比较"缺陷与正常的分离程度"。

### 2. degradation ratio（α=0 → α=1 的敏感度损失）

对每个 defect specimen（或每个光照条件）：

```text
degradation = 1 − (metric_{α=1} / metric_{α=0})
```

其中 metric 用 AUROC / AUPR / d'（秩或同 α 内指标，非绝对分数）。
用于量化"α 增大时缺陷可检测性的相对损失"。

### 3. illumination robustness

- 观察 **normal 样本的 score variance** 随 α 的变化：
  `var_normal(α) = Var(score of normal images across illuminations)`。
  若 α 增大使 normal 分数跨光照更稳定（方差下降），说明 α-IN 提高了 illumination robustness。
- 同时观察 **defect 样本跨光照的 score 变化模式**：同一 defect 在不同光照下的检出是否随 α 收敛/发散。

### 关键区别（相比 Experiment 1）

- Experiment 1 用**合成** brightness/gamma 扰动，CSEM-MISD 用**真实** 108 方向光照。
- CSEM-MISD 的 illumination 是离散方向，可做"同 specimen 跨光照配对"分析，比合成扰动更接近真实工业场景。

---

## Task 5：下一步开发计划（分阶段）

### Phase 1 — 数据整理

- **输入**：下载并解压 gear.tar.gz（需解决网络/代理问题，6.8GB）。
- **输出**：
  - `data/csem_misd/` 下的原始数据；
  - `docs/csem_dataset_analysis.md` 更新为**实测**统计（各目录完好/缺陷数量、光照编号核验、CSV 文件名核验）；
  - `scripts/prepare_csem_dataset.py`（**待数据就绪后再写**，当前不写）：
    把原始多光照栈拆成 MVTec AD 风格（train/good、test/good、test/defective、ground_truth）。
- **文件位置**：`data/csem_misd/`、`docs/`、`scripts/`。
- **预计难点**：下载速度/连接稳定性；核验官方 Train/Test 内完好/缺陷分布；灰度→3 通道适配。

### Phase 2 — PatchCore baseline

- **输入**：整理后的 CSEM-MISD 子集。
- **输出**：`results/csem_exp2_baseline/` 下的 baseline 指标（image/pixel AUROC、AUPR、d'）。
- **文件位置**：`scripts/`（复用/扩展 `experiment1_illumination_tradeoff.py` 的 fit 逻辑）。
- **预计难点**：金属镜面反射导致 normal 本身 variance 大；normal 样本量偏少；coreset 采样在 8GB 显存下的 OOM（沿用 barebones + 顺序执行 + 及时释放的既有经验）。

### Phase 3 — α-IN experiment

- **输入**：baseline 代码 + 数据集。
- **输出**：α=0/0.5/1.0 三档的逐 specimen × 逐光照分数记录（CSV）+ 指标。
- **文件位置**：`scripts/`、`results/csem_exp2/`。
- **预计难点**：跨 α 分数尺度不可比（用 d'/秩指标规避）；每 α 需独立 fit；多光照 × 多 α 的计算量控制。

### Phase 4 — Analysis

- **输入**：Phase 3 的分数记录。
- **输出**：
  - robustness（normal 跨光照方差）与 sensitivity（degradation ratio）的趋势分析；
  - 是否出现 defect-specific / specimen-specific 行为；
  - 与 Experiment 1（synthetic）结论的对照。
- **文件位置**：`results/csem_exp2/figures/`、`docs/`。
- **预计难点**：区分"缺陷可见性"与"镜面反射"对 anomaly score 的贡献（核心方法学难点）。

---

## 重要限制（当前阶段）

- ❌ 不下载大型数据自动训练
- ❌ 不修改 PatchCore
- ❌ 不写新模型
- ❌ 不添加新的 normalization 方法
- ❌ 不开始论文实验
- ✅ 当前只完成：数据适用性确认 + 实验设计

---

## 待用户决策的问题（阻塞项）

1. **数据获取**：是否由用户换网络/代理下载 gear.tar.gz（6.8GB）？当前本地无数据。
2. **研究粒度**：接受"specimen 级"（无 defect type 标签），还是先人工标注 defect type（35 个缺陷件）再做"defect 级"？—— 这决定了实验能否回答原研究问题。
3. **对象选择**：gear（缺陷最稀疏，最敏感）还是 screw（缺陷占比稍高，相对好检）？
4. **光照子集**：第一轮用全 108 个还是选代表性子集？
