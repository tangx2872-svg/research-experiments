# MODULE CANDIDATE REGISTRY — E5-0 (Phase III — Module Composition Screening)

**日期**：2026-10-09 ｜ **检索日期**：2026-10-09 ｜ **性质**：**CPU-only 文献/资产审计**，**GPU work = 0**
**起点 HEAD**：`7e9baa1`（branch `research/industrial-anomaly`）
**上游状态**：
- `Phase III — Module Composition Screening`（由 E4-X 触发）
- **B2 = STOP / X6c = STOP**（`results/e4_x/verdict.json`，frozen view 120，M²AD Bird，seed 0）
  Original：AUROC **0.7677** / d′ **1.2068** / R_all **0.6754**

---

## 0. 核实等级约定（**强制标注，不得省略**）

| 标记 | 含义 |
|---|---|
| **VERIFIED** | 本轮已从一级来源直接确认 **标题 + venue/年份 + arXiv/DOI**（有代码则含代码入口） |
| **PARTIALLY VERIFIED** | 标题/年份已确认，但 **venue 或代码 URL 未从一级来源逐项确认** |
| **UNVERIFIED** | 仅见于二手列表，本轮**未**独立核实 |

> venue 若仅来自 curated awesome list，一律记 **PARTIALLY VERIFIED** 并注明来源为二手列表。
> **未核实字段留空或标 UNKNOWN，绝不填推测值。**

## 1. 已关闭 / 禁止重复的 family（**硬约束**）

来自 Exp5A–Exp16 与 E4-X 的结论，以下方向**已明确失败或已判定不可变现**，**不得**作为候选项重复：

| 已关闭 family | 依据 |
|---|---|
| α-IN 特征归一化及其所有变体（层选择 / 强度 / gate） | 5A-H、6A、6B、7A-O（CASE_C / 只能沿既有 frontier 移动） |
| L2 residual 强度重标定（C2 / Adaptive B2） | Exp10/11/14/16；**E4-X = STOP** |
| score-level 融合（C6 / X6c / late fusion） | Exp15/16；**E4-X = STOP** |
| illumination-nuisance 子空间投影（INSS / 显式 filter-out） | Exp13-P（**非 defect 特异**，机制叙事被否决）；Exp14 Family B STOP |
| category × channel gating | Exp12（M1/M2/M3 全部硬性淘汰） |
| dual-path 线性融合（代数上 ≡ 强度重参数化） | Exp14 Family A 结构性否定 |

**⇒ 候选池必须来自上述之外的技术路线。**

## 2. 本轮检索范围（记录，便于复现）

检索日期 **2026-10-09**。子领域：A industrial AD / B unsupervised AD / C domain generalization / D domain adaptation（仅借鉴无需 target anomaly label 的模块）/ E illumination robustness / F normalization-whitening / G decorrelation / H style invariance / I frequency-domain / J feature alignment / K multi-scale fusion / L calibration / M test-time adaptation / N normal-only adaptation / O memory-bank representation / P local-global decomposition。

**入口**：curated 列表 `M-3LAB/Awesome-Industrial-Anomaly-Detection`（二手，仅用于**发现**候选）＋ 逐候选一级来源核实（arXiv / 会议 openaccess / 官方 GitHub）。

**未发现（记录为检索缺口，不用臆造填补）**：
- **feature whitening / decorrelation 专门用于 AD 的工作**：在本轮检索范围内**未发现**可信条目 ⇒ **本 family 空缺**。
- 明确以 illumination-agnostic 命名的工作：**仅 PIAD（C01）**。

## 3. Initial Candidate Pool（17 项，其中 12 项通过审计）

### 3.1 输入级光照分解（Input-level photometric）

**C01 — PIAD-Retinex（Retinex 反射率–光照分解，输入级）**
- Paper: *PIAD: Pose and Illumination agnostic Anomaly Detection* ｜ 2025 ｜ **CVPR 2025** ｜ peer-reviewed **YES**
- Paper URL: `https://openaccess.thecvf.com/content/CVPR2025/html/Yang_PIAD_Pose_and_Illumination_agnostic_Anomaly_Detection_CVPR_2025_paper.html`
- Official code: `https://github.com/Kaichen-Yang/piad_baseline`（**MIT**）｜ Project: `https://kaichen-yang.github.io/piad/`
- 光照子模块来源：**URetinex-Net**（Retinex-based Deep Unfolding Network for Low-light Image Enhancement，NeurIPS 2022；PIAD README 致谢列出）
- **Verification: VERIFIED**
- Original task: 位姿+光照无关 AD（新数据集 + baseline）
- Core mechanism: 先把输入分解为**反射率图（illumination-invariant）**再送入 AD 前端；位姿部分用 3DGS + 可微位姿优化
- Why relevant: 光照在**输入端**被显式分解 —— 与已失败的**特征级** α-IN **插入点不同**，属不同技术路线
- Why defect-preserving: 缺陷主要存在于**反射率**分量，Retinex 保留反射率 ⇒ 理论上不抹掉缺陷
- Insertion point: **input**（`e1b.preprocess_for_model` 之前）
- Training: normal-only（Retinex 用预训练权重，不训练）｜ Extra params: **NO** ｜ Retrain backbone: **NO**
- Difficulty **LOW-MEDIUM** ｜ GPU cost **LOW**
- Existing industrial AD usage: PIAD 本身是工业 AD，但其主贡献是数据集+位姿；**Retinex 作为独立前端**在本轮检索范围内未见被广泛采用
- Novelty collision: **LOW-MEDIUM** ｜ Similarity to B2/X6c: **LOW**（插入点不同）
- Main advantage: 插入点全新、零训练、物理可解释
- Main risk: Retinex 可能引入伪影/放大噪声，反而损害 defect 细节；预训练权重来自低光增强域，可能引入域偏移
- Evidence 5 ｜ EngFeas 4 ｜ SciFit 5 ｜ Novelty 3 ｜ GPUEff 5 → **Score 88**

### 3.2 分布偏移下的不变表示

**C02 — FiCo（Filter or Compensate）**
- Paper: *Filter or Compensate: Towards Invariant Representation from Distribution Shift for Anomaly Detection* ｜ **AAAI 2025** ｜ peer-reviewed **YES**
- arXiv **2412.10115**（2024-12）｜ `https://arxiv.org/abs/2412.10115` ｜ AAAI `https://ojs.aaai.org/index.php/AAAI/article/view/32243`
- Official code: **PARTIALLY VERIFIED**（curated awesome list 标注有 code；本轮未取到一级 URL）
- **Verification: VERIFIED（venue+arXiv）；code PARTIALLY**
- Core mechanism: teacher–student（reverse distillation）框架下，针对分布偏移导致的**师生错位**，提出 **Filter**（滤除域移位信息）或 **Compensate**（补偿）得到不变表示
- Why relevant: 直接把"distribution shift → 不变表示"作为问题定义，与 illumination nuisance 同构
- Why defect-preserving: ⚠️ Filter 分支**本身有**"抹掉判别信息"的风险 —— 正是必须检验的点
- Insertion point: backbone / 特征级（layer2–3 之后）
- Training: normal-only（teacher–student 自监督）｜ Extra params **YES** ｜ Retrain **YES**
- Difficulty **HIGH** ｜ GPU **MEDIUM-HIGH**
- Novelty collision: **MEDIUM-HIGH** —— ⚠️ 其 **Filter 分支与本项目 Exp13-P（illumination subspace 投影 / filter-out）思想重合**，而 Exp13-P 已判定**非 defect 特异**
- Similarity to B2/X6c: **MEDIUM-HIGH**
- Main advantage: 问题定义高度对口；AAAI 级证据
- Main risk: 与已关闭路线思想上重合；需重训整个 teacher–student 框架
- Evidence 5 ｜ EngFeas 2 ｜ SciFit 5 ｜ Novelty 2 ｜ GPUEff 2 → **Score 70**

**C03 — AD under Distribution Shift**
- Title: *Anomaly Detection under Distribution Shift* ｜ 2023 ｜ **ICCV 2023**（per curated awesome list）｜ peer-reviewed **YES**
- Code: **PARTIALLY VERIFIED** ｜ **Verification: PARTIALLY VERIFIED**（机制细节本轮未核实）
- Insertion point: feature / backbone ｜ Difficulty **MEDIUM-HIGH** ｜ GPU **MEDIUM** ｜ Collision **MEDIUM**（与 C02 同族）
- Evidence 3 ｜ EngFeas 2 ｜ SciFit 4 ｜ Novelty 3 ｜ GPUEff 2 → **Score 60**

### 3.3 Memory-bank / 表示偏差抑制

**C04 — REB（Reducing Biases in Representation）**
- Paper: *REB: Reducing Biases in Representation for Industrial Anomaly Detection* ｜ Shuai LYU, Dongmei Mo, Wai Keung Wong ｜ 2023
- arXiv **2308.12577** ｜ Venue: **PARTIALLY VERIFIED**（本轮未确认正式发表 venue）｜ Code **PARTIALLY VERIFIED**
- **Verification: PARTIALLY VERIFIED**
- Core mechanism: 针对"预训练 CNN 特征 + KNN 距离度量"两阶段范式中**表示本身的偏差**做抑制
- Why relevant: 本项目核心观察即**光照在表示中形成 nuisance 方向**；REB 显式处理"表示偏差"
- Why defect-preserving: 目标是**纠偏**而非**替换**表示
- Insertion point: **memory bank / distance metric**
- Training: normal-only ｜ Extra params **UNKNOWN（待核实）** ｜ Retrain backbone **NO**
- Difficulty **MEDIUM** ｜ GPU **LOW-MEDIUM**
- Collision **LOW-MEDIUM** ｜ Similarity to B2/X6c: **MEDIUM**（若纠偏等价于特征统计归一化则接近已关闭路线 —— **E5-1 前必须核实机制**）
- Main advantage: 插入点在 **memory bank 侧**，本项目从未动过这一层
- Main risk: 机制未逐字核实，可能退化为另一种归一化
- Evidence 3 ｜ EngFeas 4 ｜ SciFit 4 ｜ Novelty 4 ｜ GPUEff 5 → **Score 78**

### 3.4 频域解耦 / 通道选择

**C05 — Omni-Frequency Channel-Selection（FD + CS）**
- Paper: *Omni-Frequency Channel-Selection Representations for Unsupervised Anomaly Detection* ｜ 2023 ｜ **IEEE TIP 2023**（ieeexplore doc 10192551）｜ peer-reviewed **YES**
- arXiv **2203.00259** ｜ `https://arxiv.org/abs/2203.00259`
- Code: **PARTIALLY VERIFIED**（awesome list 标注有 code；本轮未取到一级 URL）
- **Verification: VERIFIED（venue+arXiv）；code PARTIALLY**
- Core mechanism: **Frequency Decoupling（FD）** 把特征按频带拆开 + **Channel Selection（CS）** 按频带差异选择通道；动机是"正常/异常图像在频域分布上有显著差异"
- Why relevant: 光照变化主要落在**低频/全局亮度**分量，而缺陷细节多在高频 ⇒ 频域解耦**天然提供"分离 nuisance 与 defect"的掩码**
- Why defect-preserving: 显式保留高频通道 ⇒ 缺陷细节不被压制
- Insertion point: **post-concat**（嵌入层，`generate_embedding` 之后 / bank 之前）
- Training: normal-only（FD/CS 为可训练模块）｜ Extra params **YES（轻量）** ｜ Retrain backbone **NO**
- Difficulty **MEDIUM** ｜ GPU **LOW-MEDIUM**
- Collision: **LOW**（本项目从未做频域）｜ Similarity to B2/X6c: **LOW**
- Main advantage: family 完全新；机制上"低频=光照 / 高频=缺陷"的分离逻辑清晰
- Main risk: 频域分解对 256px 低分辨率可能不稳；FD/CS 需要训练，引入新的随机性
- Evidence 4 ｜ EngFeas 3 ｜ SciFit 4 ｜ Novelty 4 ｜ GPUEff 4 → **Score 76**

### 3.5 合成异常判别式适配（Synthetic-anomaly discriminative adaptor）

**C06 — SimpleNet**
- Paper: *SimpleNet: A Simple Network for Image Anomaly Detection and Localization* ｜ 2023 ｜ **CVPR 2023** ｜ peer-reviewed **YES**
- arXiv **2303.15140** ｜ CVF `https://openaccess.thecvf.com/content/CVPR2023/html/Liu_SimpleNet_A_Simple_Network_for_Image_Anomaly_Detection_and_Localization_CVPR_2023_paper.html`
- Official code: **PARTIALLY VERIFIED**（已知常有官方实现；本轮未从一级来源确认 URL）
- **Verification: VERIFIED（venue+arXiv）**
- Core mechanism: 在**冻结 backbone 特征**上加一个**轻量 adaptor**，用**特征空间合成异常**（高斯噪声）训练一个**判别器**，直接输出异常分数
- Why relevant: 判别器是**在特征空间学"什么算异常"**，理论上可学成对"光照导致的特征位移"不敏感（因为它只在 normal 特征上加噪）
- Why defect-preserving: 判别器显式以"偏离 normal 流形的方向"为目标 ⇒ 不做全局统计归一化
- Insertion point: **post-concat**（替/并列于 bank 的 NN 打分路径）
- Training: normal-only + **合成异常**（无需真实异常标签）｜ Extra params **YES（MLP 级）** ｜ Retrain backbone **NO**
- Difficulty **MEDIUM** ｜ GPU **LOW-MEDIUM**
- Collision: **HIGH**（SimpleNet 在工业 AD 中已被广泛使用）⚠️
- Similarity to B2/X6c: **LOW**
- Main advantage: 成本低、代码简单、与 PatchCore 可直接并列对比
- Main risk: **novelty collision 高**（作为"组合件"仍可用，但不能当作新颖性来源）
- Evidence 5 ｜ EngFeas 4 ｜ SciFit 3 ｜ Novelty 2 ｜ GPUEff 4 → **Score 70**

### 3.6 学习式 patch 表示投影

**C07 — ReConPatch**
- Paper: *ReConPatch: Contrastive Patch Representation Learning for Industrial Anomaly Detection* ｜ 2023 ｜ LG AI Research
- arXiv **2305.16713** ｜ Venue: **PARTIALLY VERIFIED**（本轮未确认正式发表 venue）｜ **Verification: PARTIALLY VERIFIED**
- Core mechanism: 在冻结 backbone 特征上学习一个**target-oriented 的 patch 表示投影**（对比式），增强特征的判别性
- Why relevant: 学习一个**与 nuisance 正交的判别子空间**，而非调整统计量
- Why defect-preserving: 对比目标以 patch 级判别为驱动 ⇒ 不必然压制缺陷方向
- Insertion point: **post-concat**（投影后再入 bank）
- Training: normal-only（对比学习）｜ Extra params **YES** ｜ Retrain backbone **NO**
- Difficulty **MEDIUM** ｜ GPU **LOW-MEDIUM**
- Collision **MEDIUM** ｜ Similarity to B2/X6c: **LOW-MEDIUM**
- Main advantage: 表示学习路线，本项目未做过；插入点清晰
- Main risk: 对比目标未显式约束 illumination ⇒ 能否降低 nuisance 属**假设**（未验证）
- Evidence 4 ｜ EngFeas 4 ｜ SciFit 3 ｜ Novelty 2 ｜ GPUEff 4 → **Score 66**

### 3.7 连续 memory 表示

**C08 — Continuous Memory Representation**
- Title: *Continuous Memory Representation for Anomaly Detection* ｜ **ECCV 2024**（per curated awesome list）｜ peer-reviewed **YES**
- Code: **PARTIALLY VERIFIED** ｜ **Verification: PARTIALLY VERIFIED**（机制细节本轮未核实）
- Core mechanism: 用**连续**（隐式/网格）表示替代离散 memory bank，缓解 KNN 采样与覆盖偏差
- Why relevant: 离散 bank 对光照造成的**连续位移**覆盖不足 ⇒ 连续表示可能更稳
- Insertion point: **memory bank**
- Training: normal-only ｜ Extra params **UNKNOWN** ｜ Retrain **NO**
- Difficulty **MEDIUM** ｜ GPU **MEDIUM**
- Collision **LOW-MEDIUM** ｜ Similarity to B2/X6c: **LOW**
- Main advantage: 直击"bank 覆盖度"这一本项目已观测到的真实约束（bank 1,200 图受显存所迫）
- Main risk: 机制未核实；实现复杂度中等
- Evidence 3 ｜ EngFeas 3 ｜ SciFit 3 ｜ Novelty 4 ｜ GPUEff 4 → **Score 66**

### 3.8 测试时 / 在线 normal 自适应

**C09 — On-The-Fly AD for Non-Stationary Image Distributions**
- Paper: *Unsupervised, Online, and On-The-Fly Anomaly Detection for Non-Stationary Image Distributions* ｜ McIntosh D., Branzan-Albu A. ｜ **ECCV 2024**（Milan）｜ peer-reviewed **YES**
- Code: **PARTIALLY VERIFIED**（awesome list 标注有 code）｜ **Verification: VERIFIED（venue+authors）；code PARTIALLY**
- Core mechanism: 面向**非平稳分布**的在线/即时（on-the-fly）无监督 AD，模型随到来的数据自适应
- Why relevant: 光照变化即"非平稳"；在线自适应用**当前批次的 normal 统计**做校准
- Why defect-preserving: 仅在 **normal 侧**自适应
- Insertion point: **test-time**
- Training: normal-only（在线，无标签）｜ Extra params **UNKNOWN** ｜ Retrain **NO**
- Difficulty **MEDIUM** ｜ GPU **LOW-MEDIUM**
- ⚠️ **协议冲突风险**：我们的 frozen 协议是**离线、单次评估**；引入在线自适应会**改变评测口径**，必须新预注册（否则与 Exp16/E4-D1 不可比）
- Collision **LOW-MEDIUM** ｜ Similarity to B2/X6c: **LOW**
- Main advantage: family 全新；直接针对"分布变化"
- Main risk: **评测协议必须重定义** —— 这是本轮最大的工程/科学耦合风险
- Evidence 4 ｜ EngFeas 2 ｜ SciFit 3 ｜ Novelty 4 ｜ GPUEff 3 → **Score 64**

### 3.9 匹配代价过滤（score 后处理）

**C10 — CostFilter-AD**
- Title: *CostFilter-AD: Enhancing Anomaly Detection through Matching Cost Filtering* ｜ **ICML 2025**（per curated awesome list）｜ peer-reviewed **YES**
- Code: **PARTIALLY VERIFIED** ｜ **Verification: PARTIALLY VERIFIED**（机制细节本轮未核实）
- Core mechanism: 对匹配**代价（cost）**做滤波以改善异常定位/判定
- Why relevant: 光照带来的代价噪声可被滤波抑制
- Insertion point: **distance metric / score aggregation**
- Difficulty **MEDIUM** ｜ GPU **LOW-MEDIUM**
- ⚠️ Collision/Similarity: **MEDIUM**（位于"score 后处理"层，与已关闭的 score-fusion family **相邻**，必须在 E5-1 前确认它作用在 cost map 而非我们的 Original/B2 分数融合上）
- Evidence 3 ｜ EngFeas 3 ｜ SciFit 3 ｜ Novelty 3 ｜ GPUEff 4 → **Score 62**

### 3.10 其余初始候选（未进 shortlist，记录备查）

| ID | Module | Family | Year/Venue | Verification | 未进 shortlist 原因 |
|---|---|---|---|---|---|
| C11 | Dinomaly（CVPR 2025，arXiv 2405.14325，`github.com/guojiajeremy/Dinomaly`） | 重建（基础模型特征） | CVPR 2025 | **VERIFIED** | 完整框架重训，成本 HIGH；且其定位是**多类统一模型**，非 illumination 模块 |
| C12 | ReContrast（arXiv 2306.02602） | 重建+对比 | 2023 | **PARTIALLY VERIFIED** | 与 C11 同族；成本 HIGH |
| C13 | MSFlow（2024） | 多尺度 normalizing flow | 2024 | **PARTIALLY VERIFIED** | flow 训练成本 MEDIUM-HIGH；family 与 C05 部分重叠 |
| C14 | PyramidFlow（CVPR 2023） | 多尺度 flow | CVPR 2023 | **PARTIALLY VERIFIED** | 同上 |
| C15 | FAIR: Frequency-aware Image Restoration（2023） | 频域恢复（图像级） | 2023 | **PARTIALLY VERIFIED** | 与 C05 同族（frequency）⇒ 受"每 family ≤2"约束，且 C05 证据更强 |
| C16 | GLAD（ECCV 2024） | 全局-局部扩散重建 | ECCV 2024 | **PARTIALLY VERIFIED** | 扩散模型成本 HIGH（≥12 GB 额外显存风险），24 GB 卡上与 PatchCore 共存困难 |
| C17 | Odd-One-Out（CVPR 2025）/ MuSc（ICLR 2024） | 零样本互评 | 2024/2025 | **PARTIALLY VERIFIED** | 属 zero-shot 范式，不做 normal-only 训练，与我们的评测口径不同 |

**初始候选数：17**（C01–C17）｜**通过审计进入评分：10**（C01–C10）

---

## 4. 统一评分（C01–C10）

**权重**：Scientific Fit **30%** ｜ Evidence Strength **20%** ｜ Engineering Feasibility **20%** ｜ Novelty Potential **20%** ｜ GPU Efficiency **10%**
`Score = (SciFit·30 + Evid·20 + EngFeas·20 + Novelty·20 + GPUEff·10) / 5`（各子项 1–5，满分 100）

| Rank | ID | Module | Family | SciFit | Evid | EngFeas | Novelty | GPUEff | **Score** |
|---:|---|---|---|---:|---:|---:|---:|---:|---:|
| 1 | **C01** | PIAD-Retinex（输入级光照分解） | input-photometric | 5 | 5 | 4 | 3 | 5 | **88** |
| 2 | **C04** | REB（表示偏差抑制，bank 侧） | memory-bank bias | 4 | 3 | 4 | 4 | 5 | **78** |
| 3 | **C05** | Omni-Frequency FD+CS | frequency | 4 | 4 | 3 | 4 | 4 | **76** |
| 4 | **C02** | FiCo（filter/compensate） | invariant-rep | 5 | 5 | 2 | 2 | 2 | **70** |
| 5 | **C06** | SimpleNet（合成异常判别） | synthetic-anomaly | 3 | 5 | 4 | 2 | 4 | **70** |
| 6 | **C07** | ReConPatch（patch 表示投影） | learned-projection | 3 | 4 | 4 | 2 | 4 | **66** |
| 7 | **C08** | Continuous Memory Representation | memory-bank repr | 3 | 3 | 3 | 4 | 4 | **66** |
| 8 | **C09** | On-The-Fly AD（非平稳分布） | test-time adaptation | 3 | 4 | 2 | 4 | 3 | **64** |
| 9 | C10 | CostFilter-AD | cost filtering | 3 | 3 | 3 | 3 | 4 | **62** |
| 10 | C03 | AD under Distribution Shift | invariant-rep | 4 | 3 | 2 | 3 | 2 | **60** |

## 5. Family Diversity 审计（**短名单硬约束**）

规则：**任何单一 family 最多占短名单 2 个位置**。

Top-8（按分数）的 family 分布：

| Family | 短名单内条目 | 计数 | 是否合规 |
|---|---|---:|---|
| input-photometric | C01 | 1 | ✔ |
| memory-bank（bias / repr） | C04, C08 | 2 | ✔（达上限） |
| frequency | C05 | 1 | ✔ |
| invariant-rep | C02 | 1 | ✔ |
| synthetic-anomaly | C06 | 1 | ✔ |
| learned-projection | C07 | 1 | ✔ |
| test-time adaptation | C09 | 1 | ✔ |

**结论**：Top-8 **恰好**满足 family 多样性约束（7 个 family，最大 2 席），**无需**为多样性做机械替换。
**明确排除**：C03（与 C02 同 family，且分数最低）、C10（分数第 9，且位于"score 后处理"层，与已关闭 family 相邻）。

## 6. Novelty Collision Audit（GO/HOLD 候选）

检索式（对每个候选）：`<module> anomaly detection` / `<module> industrial anomaly detection` / `<module> PatchCore` / `<module> MVTec` / `<module> illumination anomaly detection`。

**等级**：A 基本未用于工业 AD ｜ B 有类似思想但场景/组合不同 ｜ C 已有非常接近的方法 ｜ D 几乎完全撞车
> ⚠️ 表述纪律：只能写 **"在本轮检索范围内未发现明显相同工作"**，**不得**写成"全球没人做过"。

| ID | Module | 等级 | 判定依据（本轮检索范围内） |
|---|---|---|---|
| C01 | PIAD-Retinex | **B** | PIAD（CVPR 2025）本身即工业 AD 且含 illumination-agnostic 定位；但**"Retinex 作为独立输入前端 + PatchCore normal-only + 真实多光照评测"**在本轮检索范围内未发现明显相同工作。**其主贡献是数据集 + 位姿**，光照前端只是组件 |
| C04 | REB | **B** | "representation bias"在 KNN 类 AD 中已有明确工作（REB 本身）；但针对**真实光照 nuisance** 的 bank 侧纠偏，在本轮检索范围内未发现明显相同工作 |
| C05 | Omni-Frequency | **B** | FD/CS 已用于 AD（TIP 2023 原论文）；但**用频带分离照明 nuisance 与缺陷**这一**明确问题定义**，在本轮检索范围内未发现明显相同工作 |
| C02 | FiCo | **C** ⚠️ | 该论文**本身就是**"分布偏移下的不变表示"，且其 **Filter 分支与 Exp13-P 的 filter-out 思路重合**（Exp13-P 已判定非 defect 特异）⇒ **接近撞车**，只应作为**对照/基线**而非新颖性来源 |
| C06 | SimpleNet | **D** ⚠️ | SimpleNet 在工业 AD 中广泛使用 ⇒ **几乎完全撞车**。仅可作为**组合件/基线**，**不得**作为新颖性来源 |
| C07 | ReConPatch | **C** ⚠️ | 对比式 patch 表示投影在 AD 中已被提出（ReConPatch 本身）⇒ 接近撞车；作为组合件可用 |
| C08 | Continuous Memory | **B** | 连续 memory 表示已用于 AD（ECCV 2024 原论文）；但其"缓解离散 bank 对连续光照位移覆盖不足"的**用法**在本轮检索范围内未发现明显相同工作 |
| C09 | On-The-Fly AD | **B** | 非平稳分布在线 AD 已有工作；但作为**照明鲁棒性**手段 + 我们的离线 frozen 协议，属新组合 —— **且协议需重新预注册** |

**新颖性结论**：本轮 shortlist 中，**真正可承担"新颖性来源"的只有 C01 / C04 / C05 / C08（等级 B）**；
C02 / C07 为等级 C（接近撞车），**C06 为等级 D（几乎完全撞车）**。
⇒ 论文新颖性应建立在 **input-photometric × frequency × memory-bank 三族的组合逻辑**上，而非任何单一已被广泛使用的模块。

---

## 7. Code Availability Audit（shortlist）

| ID | 官方 GitHub | 本轮是否访问过 | license | PyTorch | 可摘取独立模块 | 结论 |
|---|---|---|---|---|---|---|
| C01 | `github.com/Kaichen-Yang/piad_baseline` | **是**（README 已读） | **MIT** | **是** | 位姿与 AD **脚本级分离**；`Retinex/` 目录存在，但**未声明为可插件** | **可用**；Retinex 部分需自行封装为 input transform |
| C04 | URL **未取到** | 否 | UNKNOWN | UNKNOWN | UNKNOWN | **需在 E5-1 前补齐核实** |
| C05 | URL **未取到**（awesome list 标"有 code"） | 否 | UNKNOWN | UNKNOWN | UNKNOWN | **需补齐核实** |
| C02 | URL **未取到**（awesome list 标"有 code"） | 否 | UNKNOWN | UNKNOWN | 需重训 teacher–student | 成本高，先核实再定 |
| C06 | URL **未取到** | 否 | UNKNOWN | UNKNOWN | 判别器为 MLP 级，易复现 | 复现成本低，但撞车 D |
| C07 | URL **未取到** | 否 | UNKNOWN | UNKNOWN | 投影层为线性/MLP 级，易复现 | 复现成本低 |
| C08 | URL **未取到** | 否 | UNKNOWN | UNKNOWN | UNKNOWN | **需补齐核实** |
| C09 | URL **未取到** | 否 | UNKNOWN | UNKNOWN | 需在线协议改造 | 协议耦合风险 |

**⚠️ 本轮重大缺口（必须如实记录）**：除 **C01** 外，其余 shortlist 候选的**官方代码 URL 均未从一级来源取到**
（本轮主要受检索预算限制：curated 列表中代码链接被渲染为占位文本；`raw.githubusercontent` 与 `github.com` 在本服务器**不可达**，只能经 Agent 的 web 抓取）。
⇒ **E5-1 正式开始前，必须先完成一次「代码 URL 补齐」子任务**（CPU-only，见 §10）。

**优先策略**（任务书 §8）：
- 优先：**有论文 + 有官方代码 + 模块独立 + 易移植** → 目前仅 **C01** 完全满足
- 降优先级：只有论文没有代码 / 复杂端到端重训 / 需要大型额外模型 / 需要异常标签 / 需要大量 target-domain 数据
  → 命中此类的：**C02（重训）**、**C09（协议改造）**

## 8. Composition Potential Matrix（shortlist 8×8）

**逻辑**：寻找 **A 解决 nuisance** ＋ **B 补偿 defect preservation** 的**互补对**，而不是"A 有效 + B 也有效"。

|  | C01 | C04 | C05 | C02 | C06 | C07 | C08 | C09 |
|---|---|---|---|---|---|---|---|---|
| **C01** input-photometric | – | **++** | **++** | + | **++** | + | + | 0 |
| **C04** memory-bank bias | **++** | – | + | **–** | + | **–** | **–** | + |
| **C05** frequency | **++** | + | – | 0 | **++** | + | + | 0 |
| **C02** invariant-rep | + | **–** | 0 | – | + | + | + | 0 |
| **C06** synthetic-anomaly | **++** | + | **++** | + | – | + | + | + |
| **C07** learned-projection | + | **–** | + | + | + | – | + | 0 |
| **C08** memory repr | + | **–** | + | + | + | + | – | + |
| **C09** test-time | 0 | + | 0 | 0 | + | 0 | + | – |

等级：**++ 强互补 ｜ + 可能互补 ｜ 0 无明显逻辑 ｜ – 可能冗余 ｜ –– 高度重复/冲突**

**++ 的理由（每对一句）**：
- **C01 × C04**：C01 在**输入端**消除照度分量（物理层），C04 在**bank 侧**纠正残余表示偏差（统计层）—— 阶段不重叠，一个减少 nuisance 来源，一个补偿其残留。
- **C01 × C05**：C01 压掉低频照度，C05 的 FD/CS 显式按频带选择通道并**保留高频** ⇒ 前者去除 nuisance、后者显式保护 defect 频带。
- **C01 × C06**：C01 负责 nuisance 抑制（无学习），C06 直接在**特征空间**学"偏离 normal 流形的方向"以**保住 defect 判别性** —— 正是"A 解 nuisance + B 补 preservation"的模板。
- **C05 × C06**：C05 提供频带分离的判别子空间，C06 在该子空间上做合成异常判别；两者作用对象不同（表征分解 vs 判别目标）。

**– / –– 的理由**：
- **C04 × C02**、**C04 × C07**、**C04 × C08**：四者都作用在**表示/bank 统计**上，很可能只是同一层级的重复干预（且 C02 的 Filter 分支与已判非特异的 Exp13-P 同思路）。
- **C04 × C08**：均为 memory-bank 侧 ⇒ family 重复，同时使用预期收益递减且难归因。
- **C02 × C04**：一个"滤除/补偿偏移方向"，一个"纠正表示偏差"—— 若两者都在压制同一 nuisance 方向，**存在相互抵消或不可归因**的风险。

---

## 9. E5-1 推荐名单（8 项，按优先级）

范围固定：**M²AD Bird ｜ seed 0 ｜ frozen view `120` ｜ illumination 01–10（700 图）**；**Original 复用 E4-X（0 GPU）**。

| 优先级 | ID | Module | Insertion point | 最小实现方案 | 需要 fit？ | 可复用 Original？ | 单 unit 估计 | 估计显存 | 证据等级 |
|---:|---|---|---|---|---|---|---|---|---|
| **P0** | **C01** | PIAD-Retinex | **input**（`preprocess_for_model` 之前） | 用**预训练** Retinex（URetinex-Net 权重）做反射率分解，其余 pipeline 完全不动 | 是（1,200 图 bank，含 Retinex 前处理） | 否（输入已变） | ~1,300–1,700 s | +~0.5 GB | VERIFIED（MIT 代码） |
| **P0** | **C04** | REB | **memory bank / distance** | 先核实其"表示偏差"的精确算子，再以最小改动接入 bank 构建或距离度量 | 是 | 否 | ~900–1,400 s | ≈baseline | PARTIALLY |
| **P1** | **C05** | Omni-Frequency FD+CS | **post-concat** | 在嵌入层后插入频带解耦 + 通道选择（轻量可训练），bank 仍 normal-only | 是（+模块训练） | 否 | ~1,200–1,600 s | +~0.5 GB | VERIFIED（venue/arXiv） |
| **P1** | **C06** | SimpleNet 判别器 | **post-concat** | 冻结 backbone 特征 + 轻量 adaptor，用**合成异常**训练判别器；与 NN 路径并列输出 | 是（+判别器训练） | 否 | ~1,100–1,500 s | ≈baseline | VERIFIED（venue/arXiv）；**撞车 D** |
| **P2** | **C08** | Continuous Memory | **memory bank** | 用连续表示替换离散 bank（实现待核实） | 是 | 否 | ~1,000–1,500 s | ≈baseline | PARTIALLY |
| **P2** | **C07** | ReConPatch 投影 | **post-concat** | 学习一个 patch 表示投影（对比式，仅 normal 训练） | 是（+投影训练） | 否 | ~1,100–1,500 s | ≈baseline | PARTIALLY |
| **P3** | **C02** | FiCo | backbone / 特征级 | **只作对照**：其 Filter 分支 ≈ 已被否决的 Exp13-P 思路 | 是（teacher–student 重训） | 否 | ~2,500 s+ | +2–4 GB | VERIFIED（venue/arXiv） |
| **P3** | **C09** | On-The-Fly AD | **test-time** | **需先写新预注册协议**（在线口径）；未解决前**不得**开跑 | 是（在线） | 否 | 待定 | 待定 | VERIFIED（venue）；**协议耦合** |

**P3 的说明**：C02 与 C09 **不是**为了"多凑候选"，而是分别承担 **(a) 已知失败思路的对照** 与 **(b) 协议边界探索**。
若 E5-1 预算紧张，**可先只跑 P0–P2 共 6 项**（C01、C04、C05、C06、C08、C07）。

### ⚠️ E5-1 开跑前的**强制前置子任务（CPU-only，不占 GPU）**

1. **代码 URL 补齐**：C04 / C05 / C06 / C07 / C08 的官方仓库、license、是否 PyTorch 逐项核实并写入本 registry（当前为 PARTIALLY）。
2. **C04 机制核实**：确认 REB 的"偏差抑制"**是否等价于特征统计归一化** —— 若是，则落入**已关闭 family**，必须**直接淘汰**。
3. **C10 排除确认**：若 C04 被淘汰，是否用 C10 补位需单独论证（C10 位于 score 后处理层，与已关闭 family 相邻）。
4. **C09 协议**：先写 `docs/E5_1_ONLINE_PROTOCOL.md` 并冻结，否则 C09 不进入 E5-1。

## 10. E5-1 预算估算（基于 E4-X / E4-D1 实测，**不做 micro-benchmark 线性外推**）

**实测锚点（E4-X，同 bank 规模、同 view）**：
- B2 fit（bank 1,200，coreset 主导）= **778.4 s**
- 评分 700 图 = **43.9 s**
- peak VRAM = **14,497 MB alloc / 18,258 MB reserved**（24,126 MB）
- peak RAM = **2,567 MB**

| 项 | 估算 |
|---|---|
| **GPU units** | **8**（每候选 1 个：1 fit + 1 score）＋ **2 备用**（重跑/失败重试）＝ **8–10** |
| **单 unit wall-clock** | **≈15–30 min**（= fit ≈13–21 min（含模块前处理/训练）＋ score ≈1–3 min） |
| **总 wall-clock** | **≈2.5–4.5 h**（8 候选，串行单 worker）；若只跑 P0–P2 的 6 项 ≈ **1.8–3.3 h** |
| **预计 peak VRAM** | **≈15–17 GB alloc / ≤20 GB reserved**（各候选均为轻量模块；**已排除**扩散/重建类 C11/C16） |
| **并发** | **强制 1 worker**（E4-D1/E4-X 已证明 4 workers 会 OOM） |
| **Original** | **复用 E4-X，0 GPU** |
| **X6c 类 score 融合** | 0 GPU（若需要，均为 CPU 后处理） |

**风险预算**：若某候选出现 OOM 或训练不收敛，按「记录错误 → 最多自动重试 2 次 → 跳过该 unit 并保留其余结果」处理，**不修改科学协议**。

## 11. 本轮状态总结（VERIFIED / PARTIALLY / UNVERIFIED）

| 项 | 数量 / 内容 |
|---|---|
| Initial candidates | **17**（C01–C17） |
| 进入评分审计 | **10**（C01–C10） |
| **VERIFIED**（venue 由**一级来源**本轮确认） | **5**：C01（CVPR 2025 openaccess）、C02（AAAI ojs 32243）、C05（IEEE TIP 2023, doc 10192551）、C06（CVF CVPR 2023 openaccess）、C09（ECCV 2024，作者主页确认 Milan） |
| **PARTIALLY VERIFIED** | **12**：C03、C04、C07、C08、C10、C11–C17（venue 或代码 URL 未逐项一级确认；C11 的 venue+代码已确认，但其机制细节未读） |
| **UNVERIFIED** | **0** —— 未被独立核实的项**未**进入候选池，**不以凑数方式补齐** |
| 有**一级确认官方代码** | **仅 C01**（`Kaichen-Yang/piad_baseline`，MIT）；C11 亦有一级代码（`guojiajeremy/Dinomaly`）但其未进审计组 |
| **STOP** | **0**（无候选因"与已关闭 family 重复"被直接淘汰——但 **C04 存在该风险，必须在 E5-1 前核实**，见 §9 前置任务 2） |
| **HOLD** | **2**：**C02**（其 Filter 分支与已判定非特异的 Exp13-P 同思路）、**C09**（改动评测口径，须先冻结新协议） |
| **GO** | **6**：**C01、C04、C05、C06、C07、C08** |
| **Shortlist 合计** | **8**（= GO 6 + HOLD 2），满足 family 多样性约束（7 family，最大 2 席） |

**GPU work performed this round: 0**（全程 CPU；GPU 保持空闲）。

## 12. 下一步

**NEXT = E5-1 — Broad Mini Screening**（尚未启动；需人工批准）。

启动前必须完成 §9 的 4 项 CPU 前置子任务。
