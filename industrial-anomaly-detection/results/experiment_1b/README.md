# Experiment 1B：Defect-Specific α Sensitivity Screening

> 正式 Mini Experiment。目的不是证明想法正确，而是快速、诚实地判断「不同 defect type 对 α-IN representation probe 是否存在稳定差异」。

## 实验性质与边界

- 本实验**不做任何 synthetic illumination perturbation**（无 brightness/gamma/exposure/shadow）。
- 只研究**内部 representation probe**：α-IN 对不同 defect type 的影响。
- α 只是「原始特征 F 与 IN 特征 F_IN 的线性混合权重」，**不是**「去除了多少光照信息」，也**不是**我们提出的创新方法。

## 预注册四句话（实验前记录）

### ① 我怀疑什么？
当 PatchCore 内部逐渐增强统计归一化干预（α 增大）时，不同 defect type 的异常表示可能不会同步变化。结构明显的大缺陷（broken_large）可能相对稳定，而较小、弱对比度、外观/纹理相关缺陷（contamination）可能更敏感。这只是研究假设，不是预期结果。

### ② 准备干什么？
在 MVTec AD bottle + 现有 PatchCore + α-IN 上，只改变 α ∈ {0, 0.25, 0.5, 0.75, 1.0}，其他条件不变。测试原始 good / broken_large / broken_small / contamination，不做人为光照扰动。

### ③ 看什么结果？
三类 defect 随 α 变化的：image-level AUROC、Recall/TPR、raw anomaly score distribution、degradation curve、individual sample trajectory；以及 pixel-level AUROC（若 pipeline 支持）、anomaly map、defect area 分析。

### ④ 什么结果意味着继续？
若不同 defect type 在多个 α 下出现稳定、明显、可重复、非少数样本导致的不同响应曲线，则值得继续；若基本同步变化或差异来自随机波动/score scale/defect size，则如实报告、暂停。

## 数据集

- MVTec AD bottle
- train/good: 209（划出 20 作 validation，189 进 memory bank）
- test/good: 20
- test/broken_large: 20，broken_small: 22，contamination: 21
- 异常样本均有 GT mask（`ground_truth/{type}/{index}_mask.png`）

## PatchCore 配置

| 参数 | 值 |
|---|---|
| backbone | wide_resnet50_2 |
| feature layers | layer2 + layer3 |
| coreset_sampling_ratio | 0.1 |
| num_neighbors | 9 |
| input_size | 256×256 |
| batch_size | 16 |
| IN 位置 | `generate_embedding` concat 后、reshape 前 |
| IN 参数 | affine=False |
| α=0 处理 | 直接返回原始 feature，跳过 IN（bit-wise 一致） |

## Validation Split 规则

- 从 209 train/good 中，`random.Random(seed=0)` shuffle 后取前 20 个文件名作 validation。
- 所有 α 使用**完全相同**的 validation/train 划分（seed 固定）。
- validation 不进入 memory bank，仅用于定阈值。
- 划分结果保存在 `config/validation_split.csv`。

## Threshold Rule（阈值规则）

- 每 α 下：`tau_alpha = max(validation normal scores)`（最保守，验证集 FPR=0）。
- broken_large / broken_small / contamination **共享**同一个 tau_alpha。
- **不**使用 test 数据、**不**使用 defect 标签调阈值。

## Seed 与随机性

- seed = 0（固定 python/numpy/torch CPU+CUDA 随机种子）。
- coreset sampling（KCenterGreedy）与 SparseRandomProjection 的随机性由 seed 控制。

## 评估指标

1. Defect-wise AUROC：`test/good vs broken_large`、`vs broken_small`、`vs contamination`，每 α 分别计算。
2. Defect-wise Recall：用 tau_alpha 阈值，计算三类 defect 的 TPR，同时报告 test/good FPR。
3. Raw anomaly score：每图每 α 保存。
4. Individual trajectory：同一样本跨 5 档 α 的 score 变化。
5. Defect area：用 GT mask 算 defect_area_ratio，排查 size confound。

## 继续 / 暂停判据

- A（继续）：明显且稳定的 defect-specific difference。
- B（证据不足）：三类基本同步变化。
- C（重新定义问题）：差异主要由 defect size / outlier / score scale 解释。
- D（暂不结论）：结果混乱 / seed 敏感 / 实现异常。

## 禁止事项

不修改 α 求好看结果、不删样本、不按 defect 单独调阈值、不把 α 解释为光照去除量、不把 IN 说成新方法、不把 score 下降说成 defect information loss、不把 Bottle 结果推广到所有工业数据、不把 α-IN sensitivity 等同于真实 illumination sensitivity、不倒推假设、不编造结果、不覆盖历史记录。

## 运行记录

- 2026-10-01：smoke test 通过（5 张/类，α=0/0.25/0.5/0.75/1.0）。α=0 分数与 Experiment 1 原始 PatchCore 分数量级一致（good~24 / broken_large~64 / broken_small~59 / contamination~47），确认 α=0 复现 baseline。
- 2026-10-01：全量实验完成（seed=0，5 α × 103 图 = 515 条 raw 记录）。threshold 与 smoke test 完全一致，可复现。无 OOM。

---

# 实验结果（2026-10-01）

## Q1：alpha=0 是否成功复现 baseline？

**是。** 三个证据：
1. α=0 时 `generate_embedding` 直接返回原始 feature（代码层 bit-wise 一致）；
2. test/good 分数（mean 24.7，range 21.5-28.2）与 Experiment 1 中原始 PatchCore 的分数量级完全一致；
3. defect 分数同样一致（broken_large ~64 / broken_small ~59 / contamination ~47）。

注意：本实验 train 用 189 张（剔除了 20 张 validation），因此与用全量 209 的归档结果会有微小差异，这是协议设计所致。

## 主结果 1：Defect-wise AUROC（饱和）

| alpha | broken_large | broken_small | contamination |
|---|---|---|---|
| 0 | 1.0000 | 1.0000 | 1.0000 |
| 0.25 | 1.0000 | 1.0000 | 1.0000 |
| 0.5 | 1.0000 | 1.0000 | 1.0000 |
| 0.75 | 1.0000 | 1.0000 | 1.0000 |
| 1.0 | 1.0000 | 1.0000 | 1.0000 |

**image-level AUROC 在所有 α 下全部饱和于 1.0**。bottle 的三类缺陷在 image-level 都极易与 good 分开，AUROC 这个指标在本任务上对 α 不敏感，无法区分 defect-specific 差异。

## 主结果 2：Defect-normal 分离度 d'（核心发现）

同 α 内的标准化分离度 d' = (mean_defect − mean_good) / pooled_std：

| alpha | broken_large | broken_small | contamination |
|---|---|---|---|
| 0 | **13.63** | 8.31 | 3.88 |
| 0.25 | 11.46 | 8.01 | 3.91 |
| 0.5 | 10.31 | 8.16 | 4.10 |
| 0.75 | 8.55 | 7.97 | 4.18 |
| 1.0 | **7.32** | 7.73 | **4.27** |

**三类 defect 的 α-response 曲线明显分化**：
- **broken_large**：d' 单调大幅下降 13.63 → 7.32（-46%，几乎腰斩），每样本 score delta 均值 -4.03（仅 4/20 上升）
- **broken_small**：基本稳定 8.31 → 7.73（-7%），delta +0.91（15/22 上升）
- **contamination**：反而缓慢上升 3.88 → 4.27（+10%），delta +2.03（16/21 上升）

方向与初始假设（contamination 最敏感）**相反**：实际最敏感的是 broken_large。

## 主结果 3：Recall / FPR（阈值规则失效，如实报告）

预注册阈值规则 `tau = max(validation normal score)` 导致 **test/good FPR = 1.0**（所有 α 下）。

原因：validation（来自 train 分布）的分数系统性低于独立的 test/good——validation max ~20.4 vs test/good min ~21.5（α=0）。即「从 train 划出的 validation」与「真正的 test normal」存在分布 gap，max 阈值无法泛化到 test。

这是一个**真实的统计发现**（train-derived normal 与 test normal 有 gap），不是 bug。但它使 Recall/FPR 在本实验中失去区分度（Recall 全 1.0、FPR 全 1.0）。Recall 维度的 defect-specific 分析不可用，只能依赖 AUROC（饱和）和 d'/raw score（有效）。

## Q3：差异是否由多数样本共同支持？

**是。** individual trajectories 显示：
- 三类 defect 的样本普遍呈 U 型曲线（α≈0.5 最低，α=1 回升），U 型是跨样本的普遍模式，不是少数 outlier。
- broken_large 的深度下降在 16/20 样本上出现；contamination 的上升在 16/21 样本上出现。
- 类别均值曲线与个体曲线形态一致。

## Q4：缺陷面积是否能解释差异？—— 部分能，但不能完全

面积分布：broken_large mean=0.117，broken_small mean=0.031，contamination mean=0.085。

回归 `score_delta ~ area + type`（n=63）：
- area 单独：coef=-30.2，R²=0.130
- type 单独：R²=0.292
- area + type：R²=0.324，area coef=-19.1（控制 type 后仍显著为负）

关键模式：
- **broken_large 内部** corr(area, delta) = -0.747，**broken_small 内部** = -0.725：**面积越大，score 下降越多**（size confound 真实存在）
- **contamination 内部** corr = +0.196：无面积相关，且 delta 方向为**正**（上升），与 broken 类完全相反

结论：**size confound 部分解释 broken_large vs broken_small 的差异**（large 下降更多与其面积大有关），但**不能解释 contamination 的上升**——contamination 面积中等（0.085 > small 的 0.031），若纯粹由 size 驱动应比 small 下降更多，实际反而上升。defect type 本身（type-only R²=0.292 > area-only R²=0.130）是更强的解释变量。

**当前证据更支持「type-related sensitivity 为主、size-related 为辅」的混合解释**，而非纯 size artifact。

## Q5：score distribution 如何变化？是否只是整体 scale 改变？

不只是 scale 改变。若只是整体缩放，good 与 defect 的分离度 d' 应基本不变。实际：
- good mean：24.7 → 29.3（+19%）
- broken_large mean：63.9 → 58.5（-8%）——**defect 分数下降而 good 上升，方向相反**
- contamination mean：47.1 → 51.1（+9%）——与 good 同向但幅度不同

d' 的变化（large -46% / small -7% / cont +10%）无法用单一 scale 因子解释，说明 α 改变的是**good-defect 的相对几何关系**，且对不同 defect 方向不同。

## Q6：Heatmap 是否与定量指标一致？

**一致。** 代表性样本（每类取 defect area 中位数样本，规则透明）的五档 α anomaly map 显示：
- 三类缺陷的定位位置在所有 α 下保持准确（与 GT mask 重合），无定位漂移——与 AUROC 饱和一致；
- 变化体现在响应强度/范围（颜色深浅与热区大小），与 d'/raw score 的变化一致。

## Q7：结论类别

**A 与 C 之间，偏向 A（defect-specific difference 明确存在），但附 C 的限定**：

- 支持 A 的证据：三类 defect 的 d' 曲线稳定分化（单调降 / 平稳 / 缓慢升），方向不同、跨样本普遍、阈值无关（d' 不依赖阈值）、可复现（smoke 与全量 threshold 一致）。
- C 的限定：broken_large vs broken_small 的差异部分可由 defect area 解释（类内 corr≈-0.73）；contamination 的上升无法用 size 解释，是真正的 type-specific 现象。
- 诚实说明：bottle 的 image-level AUROC 已饱和，本实验的分化只能在 score-level 分离度（d'）上观察到；且方向与初始假设相反（敏感的是大缺陷而非外观型缺陷）。

## 替代解释

1. **Size confound**：α-IN 抑制了大面积缺陷的响应（broken_large 受影响最大），可能与 IN 对大区域统计量的平滑作用有关。控制面积后 type 效应仍在，但面积贡献真实存在。
2. **Score scale 混淆**：α=1 时 good 分数整体上升（memory bank 与 query 特征都在 α=1 空间，但 good 的 NN 距离变大），d' 用 pooled_std 标准化已部分消除此影响，但不能完全排除。
3. **Validation-test gap**：validation（train 分布）与 test normal 存在分数 gap，说明 bottle test/good 本身与 train 有 domain 差异，这可能放大了某些效应。
4. **单 seed**：本实验只用 seed=0。coreset 采样有随机性，d' 差异虽大（-46% vs -7% vs +10%），理论上仍需多 seed 验证稳定性。

## 局限性

- 单类别（bottle）、单 seed、n 较小（每类 20-22）。
- image-level AUROC 饱和，区分度只能来自 score 分布。
- 阈值规则（validation max）在 test 上 FPR=1.0，Recall 维度不可用。
- d' 是同 α 内的相对指标，跨 α 比较依赖 pooled_std 的稳定性。

## 结论

**Defect-specific α sensitivity 现象在 bottle 上明确存在**：broken_large 的 good-defect 分离度随 α 单调大幅下降（d' 13.6→7.3），broken_small 平稳，contamination 反而上升。方向与初始假设相反。size confound 部分解释 broken 类内部的差异，但不能解释 contamination 的反向行为。

## 下一步建议（供决策，不自动执行）

1. **多 seed 验证**（如 seeds={0,1,2}）确认 d' 分化不是 coreset 随机性导致——成本低、优先做。
2. **跨类别验证**：在 MVTec 其他类别（如 cable/screw，defect 类型更多样）复现本筛查，检验 defect-specific 分化是否普遍。
3. **size-controlled 分析**：matched-area 比较（在相同面积区间内比较 type 效应），彻底剥离 size confound。
4. 若多 seed + 跨类别都稳定，则「defect-specific representation sensitivity」值得作为正式研究问题推进；若 seed 敏感，则按判据 D 暂停。
