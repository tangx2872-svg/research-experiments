# 文献核查：CAP / FAPrompt、AnomalyDINO、PG-PSR 与 R1-ILA 的创新边界

**日期**：2026-10-30
**性质**：原论文 + 官方代码核对，**不运行**任何新方法。
**网络核查方式**：arXiv 摘要页 + ar5iv 全文 HTML（用于取章节号与逐字引用）；代码为本地克隆。

---

## 0. 结论摘要

| 对象 | 事实判定 | 对本文的影响 |
|---|---|---|
| **AnomalyDINO**（WACV 2025, Oral） | 已确认题名/作者/会议；官方代码已核对到 commit；**training-free**，patch 级最近邻 + 记忆库 + 旋转增广 + top-1% 均值聚合 | **最接近的已有设计**；本文 F0 与其同族但不同配置；R1 的增量必须表述为"给该族加一个 normal-only 线性残差适配器" |
| **CAP / FAPrompt**（ICCV 2025） | **"CAP" 是 FAPrompt 的一个模块名（Compound Abnormality Prompting module），不是方法名**；方法全名为 FAPrompt；**需要含异常样本与像素掩码的辅助训练集**，且**必须使用 CLIP 文本提示** | 与 R1-ILA 在监督信号、模态、任务设定上均不同；**不构成直接竞争**，但"轻量适配冻结 backbone"族谱相近 |
| **PG-PSR** | **仍无法唯一确认**（本轮两次检索均无异常检测文献对应；候选均不匹配） | **不得写入正文对比表**；若审稿人问及，只能答"未确认指代" |
| 是否存在已覆盖本文核心组合的工作 | **未发现**（详见 §5） | 增量可保留，但必须显式划界，不得写"首次" |

---

## 1. AnomalyDINO（已核验）

| 项 | 值 | 证据 |
|---|---|---|
| 题名 | AnomalyDINO: Boosting Patch-based Few-shot Anomaly Detection with DINOv2 | arXiv:2405.14529 摘要页 |
| 作者 | Simon Damm, Mike Laszkiewicz, Johannes Lederer, Asja Fischer | 同上 |
| 发表 | **WACV 2025（Oral）** | arXiv comments 字段；README |
| arXiv | 2405.14529（v1 2024-05-23；v3 2025-03-13） | 摘要页 |
| DOI | 10.48550/arXiv.2405.14529（arXiv 版） | 摘要页 |
| 代码 | `github.com/dammsi/AnomalyDINO`；本地 `third_party/AnomalyDINO` @ `b9d1c2648e3a5247437d4d953d907a8f3d994457` | `git rev-parse` |
| 许可 | 见 `third_party/AnomalyDINO/LICENSE`（DINOv2 Apache-2.0） | 本地文件 |
| 训练 | **training-free**（"does not require any additional data for fine-tuning or meta-learning"） | 摘要 |

**方法要点与原文位置（逐字）**

| 组件 | 位置 | 原文 |
|---|---|---|
| 记忆库 | §3.1 式(1) | "we collect the nominal patch features and store them in a memory bank" |
| 图像得分定义 | §3.1 式(4) | "s(x_test) := q({d_NN(p_1;M),…,d_NN(p_n;M)})" |
| 默认聚合 | §3.1 | "we define q as the average distance of the 1% most anomalous patches … q(D):=mean(H_0.01(D))"，即 top-1% 均值（文中亦称 99% 分位的 tail value at risk 经验估计） |
| 旋转增广 | §3.2 "ROTATION" | "In the 'agnostic' case, which we focus on in the main paper, we always augment the reference sample with rotations." |
| PCA 背景掩码 | §3.2 "MASKING"；§C.1 | "thresholding the first PCA component of the patch features"；是否启用由 masking test 逐对象决定（§C.1） |
| 聚合消融 | §4.2 "AGGREGATION STATISTIC"；§C.2 | "The mean of the 1% highest distances … improves over the standard choice (maximum …)" |
| backbone 选择 | §4.2 / §C.4 | 官方默认 `dinov2_vits14`（代码 `run_anomalydino.py --model_name` 默认值） |

**与 R1-ILA 的关系（关键）**：两者共享"冻结 DINOv2 + 正常参考记忆库 + patch 级最近邻"骨架。
本文 F0 与该骨架的差别是**配置**（ViT-B vs ViT-S；max vs top-1% 均值；无旋转增广；无 PCA 掩码）。
因此 **R1 − F0 才是适配器的效应；R1 − AnomalyDINO 是"配置 + 适配器"的混合效应，不能归因于适配器。**

---

## 2. FAPrompt / CAP（已核验，并纠正简称）

| 项 | 值 | 证据 |
|---|---|---|
| 题名 | Fine-grained Abnormality Prompt Learning for Zero-shot Anomaly Detection | CVF Open Access / arXiv:2410.10289 |
| 作者 | Jiawen Zhu, Yew-Soon Ong, Chunhua Shen, Guansong Pang | 同上 |
| 发表 | ICCV 2025, pp. **22241–22251** | CVF Open Access 条目 |
| arXiv | 2410.10289（v1 2024-10-14；v2 2025-10-02） | 摘要页 |
| 代码 | `github.com/mala-lab/FAPrompt`（MIT），最新提交 `506b6bf` | GitHub 条目 |
| **简称** | **方法名 = FAPrompt**；**CAP = Compound Abnormality Prompting module**（§3.3），**DAP = Data-dependent Abnormality Prior learning**（§3.4） | ar5iv 全文 |
| 训练监督 | **需要辅助训练集，含正常+异常样本、图像级标签与像素级异常掩码** | §3.1："D_train={X_train,Y_train} … both normal and anomalous samples"；"Y_train … contains the corresponding ground truth labels and pixel-level anomaly masks" |
| 文本提示 | **必需**（normal/abnormal prompts 经 CLIP text encoder） | §3.3/§3.5："These normal and abnormal text prompts are then processed by the CLIP's text encoder" |
| 零样本含义 | 仅指**不在目标数据集上训练** | "trained on the MVTecAD dataset by default and evaluated … without any further training or fine-tuning" |

**写作修正（必须执行）**：正文不得把该方法称为 "CAP"；应写 **FAPrompt**，需要时说明其 CAP/DAP 两个模块。
理由：CAP 是模块缩写，用它指代整篇方法属简称误用（与 PG-PSR 属同一类错误）。

**与 R1-ILA 的差别（四项全不同）**：模态（CLIP 视觉-语言 vs 纯视觉 DINOv2）、监督（异常样本+像素掩码 vs normal-only）、
适配形式（文本可学习 token vs 视觉线性残差）、任务（跨数据集 ZSAD vs 单一数据集多光照 AD）。

---

## 3. PG-PSR（未确认，维持原判）

本轮两次独立检索（`"PG-PSR" anomaly detection method abbreviation`、`"PG-PSR" OR "PGPSR" paper method`）**未找到任何异常检测文献使用该缩写**。
检索命中项均不匹配：视频异常检测的 "patch spatial relation (PSR)" 模块、旋转机械故障诊断的
"physics-guided periodic weighting sparse representation"、符号回归的 "Prior-Guided Symbolic Regression"。

**处置（不变）**：
- 不写入正文对比表；
- 若写作侧确指某方法，需提供全名/arXiv/官方代码；
- 审稿意见如提及，只能答"该缩写在 AD 文献中无法唯一对应"。

---

## 4. 与 R1-ILA 的七维对照

| 维度 | AnomalyDINO | FAPrompt (CAP+DAP) | **R1-ILA（本文）** |
|---|---|---|---|
| 训练数据 | 无（training-free） | 辅助集：正常+异常+像素掩码 | **仅正常图**（300 bank，无标签） |
| 配对来源 | 无（单图参考 + 旋转增广） | 无显式配对；DAP 从测试图自身取异常先验 | **同一正常图的合成光度扰动对**（brightness/contrast/gamma ×{0.85,1.15}） |
| 适配形式 | 无可训练参数 | 文本侧可学习 token（CLIP） | 视觉侧单层线性残差 `z' = z + Wz`，W∈R^{768×768}，零初始化、无 bias（589,824 参数） |
| 损失目标 | 无 | 像素 focal/dice + 图像 focal + CAP 正交约束 + DAP 先验正则 | `L = L_cons + 0.1·L_preserve`；`L_cons = mean(1 − cos(z'_aug, z'_clean))`，`L_preserve = 1 − mean cos(z', z_frozen)` |
| 表征保持 | 不适用 | 冻结 CLIP，仅训 token（隐式保持） | **显式锚定项** `L_preserve`，权重 0.1；`W=0` 时该损失处于极小值且梯度为 0 |
| 评分规则 | 1-NN patch 距离，**top-1% 均值** | CLIP 图文相似度（patch 级） | 1-NN patch cosine 距离，**max**（图像得分 = patch 最大值） |
| 光照验证条件 | 无（MVTec/VisA 单光照） | 无（跨数据集 ZSAD） | **M²AD 真实 12 视角 × 10 光照**；主表 + 工件级配对 bootstrap |

---

## 5. 哪些属于已有技术、本文剩余增量

**已有（不可声称新）**
1. 冻结 DINOv2 + patch 级最近邻 + 正常参考记忆库 —— AnomalyDINO §3.1。
2. 图像得分取 patch 距离的高分尾统计量 —— AnomalyDINO §3.1 式(4)、§C.2（其消融显示 top-1% 均值优于 max）。
3. 参考样本旋转增广、PCA 前景掩码 —— AnomalyDINO §3.2、§C.1。
4. 冻结 backbone + 轻量可训练适配（含文本侧 token） —— FAPrompt §3.3/§3.5。
5. 残差适配器用于异常检测 —— 见 `results/paper_supplement_20261023/closest_method_comparison.md` §4（ADRA, arXiv:2010.02310）：将其用于 CIFAR/STL 语义异常且需 outlier exposure。
6. 光度一致性/不变性损失用于低光照视觉任务 —— YOLA (NeurIPS 2024)，非 AD 域。

**本文剩余增量（可声称，但需限定）**
1. **normal-only、无异常样本、无像素掩码**的线性残差适配（FAPrompt 需要异常监督，ADRA 需要 outlier exposure）。
2. **显式表征锚定项** `L_preserve`（余弦保持，权重 0.1），用于约束零初始化残差在训练中不漂移；AnomalyDINO/FAPrompt 均无此显式项。
3. 在**真实多光照工业数据（M²AD）** 上按"同一工件跨光照"口径评测，并给出工件级配对区间（AnomalyDINO/FAPrompt 均未涉及该数据与协议）。
4. 佐证性发现：R1 使正常工件跨光照得分 SD 在 10/10 类下降（`results/e27_illum_paired/`），但**整体排序改善的区间跨零**——增量是"机制方向一致"，不是"显著提升"。

**不得声称**："首次将适配器用于异常检测"、"现有方法均未考虑光照"、"显著领先"。

---

## 6. 可直接使用的 related work 表述

> 近期工作已把"冻结 DINOv2 + patch 级最近邻 + 正常参考记忆库"确立为强基线：AnomalyDINO [Damm et al., WACV 2025]
> 在无任何训练的情况下，以记忆库最近邻距离的前 1% 均值作为图像得分，并在参考样本上使用旋转增广与可选的 PCA 前景掩码。
> 与之不同，本文不改变该检索与评分骨架，而是在其特征空间上加一个**仅用正常图训练**的线性残差适配器，
> 并用显式的余弦锚定项约束适配后的表征；监督信号仅为同一正常图的合成光度扰动对，不需要异常样本或像素掩码。
> 另一方面，FAPrompt [Zhu et al., ICCV 2025] 同样冻结主干并学习轻量提示，但其提示位于 CLIP 文本侧且训练依赖含异常样本与像素掩码的辅助集，
> 与本文"纯视觉 + normal-only"的设定不同。二者因此不构成同条件对照，本文的比较结论仅限于本协议下的受控配置。

---

## 7. 核查方法与局限

- 章节号与逐字引用来自 ar5iv 全文 HTML；**未取得 CVF 正式版页码**（AnomalyDINO 页码未核实，正文引用请用章节号）。
- FAPrompt 页码 22241–22251 来自 CVF Open Access 条目。
- 代码行为以本地克隆为准（AnomalyDINO），FAPrompt 仅核对仓库存在性与许可，**未克隆、未运行**。
- PG-PSR 的"未确认"是**检索结果**，不等于"该工作不存在"；如需写入正文，必须由写作侧提供指代证据。
