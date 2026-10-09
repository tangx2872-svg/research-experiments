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

---

# E5-0B VERIFICATION AUDIT（2026-10-09）

> **本文件向上不覆盖 E5-0 的任何原始判断**（§0–§12 原样保留）。本节只**追加**新证据、逐条标注「是否变化 / 为什么变化」，并保留全部 negative result。
> **GPU work = 0**（全程 CPU + web 一级来源核验）。起点 HEAD `53e808e`。

## B1. 与 E5-0 的差异总表（先给结论）

| ID | E5-0 原判断 | E5-0B 新证据 | 是否变化 | 变化原因 |
|---|---|---|---|---|
| C01 | VERIFIED（代码 MIT）｜insertion=input｜LOW-MEDIUM | PIAD 本体 CVPR 2025 + `Kaichen-Yang/piad_baseline` MIT **复核通过**；但**实际可移植件是其 Retinex 子模块（URetinex-Net, CVPR 2022）**，本轮出现**两个候选仓库**（`AndersonYong/URetinex-Net`、`imcy510/URetinex-Net`），官方归属**未一级确认**，license **UNKNOWN** | **部分变化** | 可移植件的**来源与许可**需在 E5-1 前钉死；机制与插入点判断不变 |
| C04 | PARTIALLY｜机制未核实；**若等价归一化则淘汰** | ① venue 升级为 **Knowledge-Based Systems 290(C), Elsevier, 2024-04-22, DOI 10.1016/j.knosys.2024.111563**（**VERIFIED**）②**机制审判已做**：REB = **DefectMaker（合成缺陷 + 自监督域适应）＋ LDKNN（局部密度 KNN）**，**不使用** mean/std/channel 归一化、**不使用** IN/LN/BN 统计 ⇒ **NOT closed-family** ③但**官方代码 URL 仍未取得**，且机制含**训练式生成/自监督流水线（非简单模块）** | **变化（重要）** | §5 机制审判**通过**；但**实现来源不足以进入 E5-1** ⇒ 降为 HOLD |
| C05 | VERIFIED（venue/arXiv）｜insertion=post-concat｜portable | **实质为 OCR-GAN 重建框架**：**FD 作用于输入图像**，**CS 作用于多个 encoder 之间**；**不是**可插拔的 feature transform；代码 URL 未取得 | **变化（重要）** | 按 §6 判据标记 **NOT_DIRECTLY_PORTABLE** |
| C06 | VERIFIED｜synthetic-anomaly｜**撞车 D** | 官方仓库确认：**`github.com/DonaldRR/SimpleNet`（PyTorch）**；结构 = 特征提取器 → **特征适配器** → 异常特征生成 + 判别器 ⇒ 适配器确为独立可插拔件 | **补充确认** | 机制/可用性不变；**ROLE 冻结为 SUPPORTING_COMPONENT** |
| C07 | PARTIALLY（venue 未确认） | ① venue 升级为 **WACV 2024**（arXiv Comments 明示 Accepted）⇒ **VERIFIED** ②机制确认：**对冻结 backbone 的 patch 特征训练 linear modulation（线性调制）**，用 **pairwise / contextual 相似度作伪标签**做对比学习，**不需要标注正负对** ③**代码：curated 多设定列表标注为 "Unofficial Code"** ⇒ 可能**无官方仓库** | **变化（重要）** | venue 升级；但**实现来源为第三方/非官方** |
| C08 | PARTIALLY | venue **ECCV 2024**、arXiv **2402.18293**、官方仓库 **`github.com/tae-mo/CRAD`**、项目页 `tae-mo.github.io/crad/`；机制 = **可学习特征网格 + 双线性插值**取代离散 bank（**无 NN 搜索、O(1)**），局部/全局网格 + 融合网络 + 特征精炼；**训练：normal-only、无异常样本、无分割标注**；骨干为预训练 EfficientNet-b4；50 epochs / batch 64 / A5000 | **升级** | venue + 机制 + 代码均取得一级证据 ⇒ **VERIFIED（机制）** |
| C02 | HOLD | 未扩大投入（按 §3）；维持：其 **Filter 分支与 Exp13-P 的 illumination-subspace 投影同思路**，而 Exp13-P 已判定**非 defect 特异** | 不变 | — |
| C09 | HOLD | 未扩大投入；维持：涉及 test-time adaptation，**改变 fixed-test 评测口径** | 不变 | — |

## B2. Per-candidate 填写表（E5-0B 要求字段）

### C01 — PIAD-Retinex（输入级 Retinex 光照分解）
- Paper title: *PIAD: Pose and Illumination agnostic Anomaly Detection*（本体）／ *URetinex-Net: Retinex-based Deep Unfolding Network for Low-light Image Enhancement*（**实际可移植件**）
- Authors: PIAD — Kaichen Yang 等（RICOH／UMich，项目页 `kaichen-yang.github.io/piad`）；URetinex-Net — **UNKNOWN（本轮未取全）**
- Year: PIAD **2025**；URetinex-Net **2022**
- Venue: PIAD **CVPR 2025**（openaccess 一级确认）；URetinex-Net **CVPR 2022**（第三方引述，非一级）
- Paper source: `openaccess.thecvf.com/content/CVPR2025/html/Yang_PIAD_Pose_and_Illumination_agnostic_Anomaly_Detection_CVPR_2025_paper.html`
- Official project page: `https://kaichen-yang.github.io/piad/`
- Official repository: **`https://github.com/Kaichen-Yang/piad_baseline`**（PIAD 本体）｜ owner **Kaichen-Yang**（作者本人）
- License: **MIT**（PIAD 本体，README+LICENSE 确认）
- Framework: **PyTorch + CUDA 自定义核 + 3D Gaussian Splatting**（PIAD 全流程）；PyTorch 版本 **UNKNOWN**
- Last repository activity: **UNKNOWN** ｜ Pretrained weights: **YES**（PIAD `model.zip`）
- Core implementation files: `pose_estimation.py`、`AUROC_TEST.py`、`Retinex/`、`retrieval/`、`gaussian_renderer/`
- Core classes/functions: **UNKNOWN（未读源码）**
- **Actual mechanism（依据论文/README，非营销语）**：把输入分解为**反射率（reflectance）**与**光照（illumination）**，用**反射率图**作为 AD 输入；位姿部分与本研究无关
- Input: RGB 图 ｜ Output: 反射率图（illumination-removed image）
- Need training **NO**（用预训练 Retinex）｜ Need anomaly samples **NO** ｜ Need target-domain samples **NO** ｜ Backbone retraining **NO**
- Can isolate module: **YES（UNCERTAIN on exact source repo）**
- PatchCore insertion point: **input**（`preprocess_for_model` 之前）
- Estimated LOC: **50–150** ｜ Expected GPU overhead: **LOW**（≈0.1–0.2 s/图；700 图 ≈1–3 min）
- License compatible: **YES（PIAD MIT）／UNCERTAIN（URetinex 独立仓库许可未确认）**
- **Verification: VERIFIED（论文/venue/许可）＋ PARTIALLY（实际 Retinex 件的仓库归属与许可）**

### C04 — REB
- Paper title: *REB: Reducing Biases in Representation for Industrial Anomaly Detection*
- Authors: **Shuai Lyu, Dongmei Mo, Wai Keung Wong**（香港理工大学）
- Year: **2024**（arXiv v1 2023-08-24；v2 2024-05-17）
- Venue: **Knowledge-Based Systems, Vol. 290, Issue C, Elsevier**，Published **2024-04-22** ｜ DOI **`10.1016/j.knosys.2024.111563`**
- Paper source: `https://arxiv.org/abs/2308.12577` ／ `https://www.sciencedirect.com/science/article/pii/S0950705124001989`
- Official project page: **none found** ｜ Official repository: **NOT FOUND this round**（摘要仅 "Code:this https URL"）
- Repository owner **UNKNOWN** ｜ License **UNKNOWN** ｜ Framework **UNKNOWN** ｜ Pretrained weights **UNKNOWN** ｜ Last activity **UNKNOWN**
- Core implementation files / classes: **UNKNOWN**
- **Actual mechanism（依据摘要 + 期刊关键词 "Synthetic defects; Local density; K-nearest neighbor"）**：
  (i) **DefectMaker**：**合成缺陷 + 自监督任务**做**域适应**，缓解预训练模型 domain bias；
  (ii) **LDKNN（local-density KNN）**：修正特征空间**局部密度偏差**的距离度量
- Input: 预训练特征 ｜ Output: 域适应后的表示 + 修正后的 KNN 距离/分数
- Need training **YES（DefectMaker 自监督）** ｜ Need anomaly samples **NO（合成）** ｜ Need target-domain samples **NO** ｜ Backbone retraining **NO**
- Can isolate module: **UNCERTAIN** ｜ Insertion point: **distance metric / memory bank**
- Estimated LOC: **150–300**（仅 LDKNN）／**>300**（含 DefectMaker） ｜ GPU overhead: **LOW-MEDIUM**
- License compatible: **UNKNOWN**
- **Verification: PARTIALLY VERIFIED**（venue 一级确认；代码与机制细节未确认）
- **Decision: HOLD**（见 B3）

### C05 — Omni-Frequency / OCR-GAN
- Paper title: *Omni-frequency Channel-selection Representations for Unsupervised Anomaly Detection*（方法名 **OCR-GAN**）
- Authors: **Yufei Liang, Jiangning Zhang, Shiwei Zhao, Runze Wu, Yong Liu, Shuwen Pan**
- Year: 2022(v1)/2023(v2) ｜ Venue: **IEEE TIP 2023** ｜ DOI **`10.1109/TIP.2023.3293772`**
- Paper source: `https://arxiv.org/abs/2203.00259`
- Official repository: **NOT FOUND this round** ｜ License **UNKNOWN** ｜ Pretrained weights **UNKNOWN**
- Framework: **GAN 重建框架（多 encoder）**
- **Actual mechanism**：**FD** 把**输入图像**解耦为不同频率分量，重建建模为**并行多频段图像复原**；**CS** 在**多个 encoder 之间**做频率交互、自适应选择通道
- Input: 输入图像（FD）／多 encoder 特征（CS） ｜ Output: 重建图 → 异常图
- Need training **YES（整套 GAN）** ｜ Need anomaly samples **NO** ｜ Need segmentation labels **NO** ｜ Backbone retraining **N/A（自建 encoder）**
- Can isolate module: **NO** ｜ Insertion point: **N/A** ｜ LOC: **>300（不可拆）**
- **Verification: VERIFIED（venue + arXiv + DOI）**
- **Decision: HOLD（§6 → NOT_DIRECTLY_PORTABLE）**（见 B4）

### C06 — SimpleNet
- Paper title: *SimpleNet: A Simple Network for Image Anomaly Detection and Localization*
- Authors: **Zhikang Liu, Yiming Zhou, Yuansheng Xu, Zilei Wang**（USTC）｜ Year **2023** ｜ Venue **CVPR 2023**
- Paper source: CVF openaccess（CVPR2023）｜ arXiv **2303.15140**
- Official repository: **`https://github.com/DonaldRR/SimpleNet`** ｜ owner **DonaldRR** ｜ License **UNKNOWN**（本轮未取到）
- Framework: **PyTorch**（README 明示）｜ Pretrained weights **YES（backbone）** ｜ Last activity **UNKNOWN**
- Core implementation files / classes: **UNKNOWN（未读源码）**
- **Actual mechanism**：冻结 backbone 提特征 → **特征适配器（adaptor）** 变换 → 特征空间**生成合成异常特征** → 训练**判别器**输出异常分数
- Need training **YES（adaptor + discriminator）** ｜ Need anomaly samples **NO（合成）** ｜ Backbone retraining **NO**
- Can isolate module: **YES** ｜ Insertion point: **post-concat** ｜ LOC **150–300** ｜ GPU overhead **LOW-MEDIUM**
- **Verification: VERIFIED（venue + 官方代码）**
- **ROLE FROZEN = SUPPORTING_COMPONENT** —— **不承担论文 novelty**；只能作为 baseline / defect-preservation compensator / composition component；**不得**写成 "our novel SimpleNet module"

### C07 — ReConPatch
- Paper title: *ReConPatch : Contrastive Patch Representation Learning for Industrial Anomaly Detection*
- Authors: **Jeeho Hyun, Sangyun Kim, Giyoung Jeon, Seung Hwan Kim, Kyunghoon Bae, Byung Jun Kang**（LG AI Research）
- Year 2023(v1)/2024(v3) ｜ Venue **WACV 2024**（arXiv Comments: "Accepted on WACV 2024"）
- Paper source: `https://arxiv.org/abs/2305.16713`
- Official repository: **NOT FOUND**；多设定列表（`Sunny5250/Awesome-Multi-Setting-UIAD`）将代码标注为 **"Unofficial Code"** ⇒ **可能无官方仓库**
- License **UNKNOWN** ｜ Framework **UNKNOWN** ｜ Pretrained weights **UNKNOWN**
- **Actual mechanism**：对**冻结预训练模型的 patch 特征**训练一个 **linear modulation（线性调制）**，用**对比表示学习**（**pairwise 相似度**与**contextual 相似度**作**伪标签**，无需标注正负对）得到 target-oriented、易分离表示
- Need training **YES（仅线性层）** ｜ Need anomaly samples **NO** ｜ Need target-domain samples **NO** ｜ Backbone retraining **NO**
- Can isolate module **YES（线性层）** ｜ Insertion point **post-concat** ｜ LOC **50–150** ｜ GPU overhead **LOW**
- **Verification: PARTIALLY VERIFIED（venue VERIFIED；实现来源 = 非官方/未确认）**

### C08 — CRAD（Continuous Memory Representation）
- Paper title: *Continuous Memory Representation for Anomaly Detection*（方法 **CRAD**）
- Authors: Joo Chan Lee*, Taejune Kim*, Eunbyung Park, Simon S. Woo, Jong Hwan Ko（成均馆大学）｜ Year **2024** ｜ Venue **ECCV 2024**
- Paper source: **arXiv 2402.18293** ｜ Project: `https://tae-mo.github.io/crad/`
- Official repository: **`https://github.com/tae-mo/CRAD`** ｜ owner `tae-mo`（与项目页一致）
- License **UNKNOWN**（论文页未提及；仓库页本轮 fetch 超时）｜ Framework PyTorch（推定）｜ Pretrained weights **UNKNOWN**
- **Actual mechanism（精确）**：用**可学习特征网格 $G$ + 双线性插值**表示正常特征 $\phi(v;G)$，**取代离散 memory bank** —— **不存离散条目、无最近邻搜索、O(1) 检索**；坐标由 **1×1 conv + tanh（局部）** 与 **GAP + linear（全局）** 生成，concat 后经卷积网络 ψ 融合 → 正常表示 $f^n$；测试时像素级 L2 距离 → 图上采样 → 图像分数 = **max of avg-pooled**；含**特征精炼**（MSE 阈值 + 余弦相似度加权）减少假阳性
- Need training **YES（网格 + 坐标生成 + 融合网 + 精炼；50 epochs, batch 64, AdamW）** ｜ Need anomaly samples **NO** ｜ Need segmentation labels **NO** ｜ Backbone retraining **NO（用预训练 EfficientNet-b4；"冻结"未明写）**
- Can isolate module: **UNCERTAIN** —— 官方实现自带 EfficientNet-b4 特征管线；移植到我们的 `wide_resnet50_2` layer2/3 属**需明示的适配**
- **PatchCore insertion point: memory bank（✓ 本项目从未探索过的插入点）** ｜ LOC **150–300** ｜ GPU overhead **MEDIUM**
- **Verification: VERIFIED（venue + arXiv + 官方仓库）；license UNKNOWN**

## B3. C04 强制机制审判（§5）

**问：所谓 representation bias suppression 在数学/代码层面是什么？**

逐项检查是否依赖已关闭 family 的算子：

| 已关闭算子 | REB 是否使用 | 依据 |
|---|---|---|
| mean subtraction | **NO** | 摘要未出现；机制被描述为 domain adaption + 局部密度 |
| variance / std normalization | **NO** | 同上 |
| channel-wise normalization / rescaling | **NO** | 同上 |
| InstanceNorm / LayerNorm / BatchNorm-style statistics | **NO** | 同上 |
| feature standardization | **NO** | 同上 |
| simple residual-strength scaling | **NO** | 同上 |

**⇒ 判定：`C04 ≠ closed-family`（机制审判通过）**

### WHY C04 IS NOT CLOSED-FAMILY NORMALIZATION

REB 的工作机理由两个**独立于特征归一化**的部分构成（依据摘要 + 期刊关键词 "Synthetic defects; Local density; K-nearest neighbor"）：

1. **DefectMaker（域适应）**：用**合成的、强多样性的缺陷**构造**自监督任务**，让预训练特征发生**适配（adaptation）**。
   它改变的是**表示本身的学习目标**（引入了一个训练过程），**不是**对一个既有特征张量做逐通道统计变换。
   ⇒ 与本项目已关闭的 α-IN / residual / dual-path（都是**对 feature tensor 的确定性重参数化**）在**操作类型**上不同。
2. **LDKNN（局部密度 KNN）**：修正的是**特征空间中局部密度的偏差**，即**检索/距离度量**层面。
   ⇒ 与本项目已关闭的 score-level fusion（作用在**已算好的分数**上）不同：LDKNN 作用在**特征空间的距离度量**上，
   是本项目**从未修改过**的组件。

**但仍必须声明两条风险（不改变"非 closed-family"的判定）**：
- ⚠️ **风险 1（实现来源不足）**：官方代码 URL **本轮未取得**；且 DefectMaker 是**训练式生成/自监督流水线**，
  不是"简单模块"⇒ **不满足 §17 的 from-paper 最小实现条件**。
- ⚠️ **风险 2（与 C06 重复）**：DefectMaker 的"**合成缺陷**"与 C06 SimpleNet 的"**特征空间合成异常**"在**思想上同源**；
  若两者同时进入 E5-1，其增益**很可能不可归因**（同一"合成异常"机制被计两次）。

**⇒ C04 最终判定：HOLD**（机制审判 PASS，但 §11 判据 C「实现来源充分核实」**FAIL**）。

## B4. C05 频域路线核验（§6）

逐项回答 §6 要求的问题（依据 IEEE TIP 2023 论文摘要，一级来源）：

| 问题 | 答案 |
|---|---|
| FD 到底如何分频？ | FD 把**输入图像**解耦为不同**频率分量**，把重建建模为**并行的多频段图像复原** |
| CS 到底做什么？ | 在**多个 encoder 之间**做**频率交互**，**自适应选择不同通道** |
| 是否需要训练？ | **是**（整套 OCR-GAN 端到端训练） |
| 是否依赖 segmentation labels？ | **否**（重建式无监督） |
| 是否能作为独立 feature transform？ | **否** —— CS 需要**多个 encoder** 才有"频率交互"的对象 |
| 是否需要修改 backbone？ | **N/A** —— 原方法**自建 encoder**，并非"在冻结 PatchCore 上插一层" |

**能否构造最小版本 `PatchCore feature → frequency decomposition → feature calibration → existing memory bank`？**
- **否。** 原论文的 FD 作用在**输入图像**而非特征，CS 作用在**多 encoder 之间**而非单一特征张量。
- 若要强行做成"特征层频域校准"，**等于自己发明一个新模块并冠以 OCR-GAN 之名** —— 这被 §6/§17 明令禁止。

**⇒ 标记 `NOT_DIRECTLY_PORTABLE`；Decision = HOLD。**

## B5. C02 / C09 处理（§10，仅必要核验，不扩大投入）

**C02 FiCo**
- 维持：其 **Filter 分支**（滤除域移位方向以得到不变表示）与本项目 **Exp13-P 的 illumination-subspace 投影（filter-out）** 在**目标与手段上一致**；
  而 Exp13-P 已用 normal 对照证明该方向**非 defect 特异**。
- **⇒ `C02 = HOLD_REDUNDANT`**，**不进入第一轮 GPU**。（若将来需要，只作为**对照/基线**，不得作为新颖性来源。）

**C09 On-The-Fly AD**
- 涉及 **test-time / online adaptation**，会**改变当前 fixed-test 评测口径**（我们所有历史结论都建立在离线单次评估上）。
- 未发现其"最小模块"能在**不改变协议**的前提下独立使用（本轮未扩大投入去证明）。
- **⇒ `C09 = HOLD_PROTOCOL`**。**本轮不为它新建复杂 online protocol**；第一轮保持 apples-to-apples。

## B6. E5-1 Eligibility Gate（§11）

判据：**A** 论文身份确认 ｜ **B** 机制理解 ｜ **C** 实现来源充分核实 ｜ **D** 非 closed-family 重复 ｜
**E** PatchCore 插入点明确 ｜ **F** 无需异常标签 ｜ **G** Mini 实现可行 ｜ **H** 预计运行时间合理

| ID | A | B | C | D | E | F | G | H | **判定** |
|---|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|---|
| **C01** | ✔ | ✔ | ▲ | ✔ | ✔ | ✔ | ✔ | ✔ | **PASS**（C 为 ▲：PIAD 本体 MIT 已确认，但其 Retinex 子件的仓库归属/许可未钉死） |
| **C04** | ✔ | ✔ | ✘ | ✔ | ✔ | ✔ | ▲ | ▲ | **HOLD**（C 失败：无官方代码且机制含训练式流水线，不满足 §17 最小实现条件） |
| **C05** | ✔ | ✔ | ✘ | ✔ | ✘ | ✔ | ✘ | ✘ | **HOLD**（NOT_DIRECTLY_PORTABLE） |
| **C06** | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ | **PASS**（ROLE = SUPPORTING_COMPONENT） |
| **C07** | ✔ | ✔ | ▲ | ✔ | ✔ | ✔ | ✔ | ✔ | **PASS**（C 为 ▲：仅非官方实现；机制简单、LOC 50–150 ⇒ 可**明示的**从论文最小实现） |
| **C08** | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ | ▲ | ▲ | **PASS（with disclosed adaptation）**（G/H 为 ▲：需移植到我们的 wide_resnet50_2 特征并另测训练时间） |
| C02 | ✔ | ▲ | ✘ | ✘ | ✔ | ✔ | ✔ | ✔ | **HOLD_REDUNDANT** |
| C09 | ✔ | ▲ | ✘ | ✔ | ✔ | ✔ | ✘ | ✘ | **HOLD_PROTOCOL** |

**✘ = FAIL ｜ ▲ = PASS with disclosure ｜ ✔ = PASS**

### 🔒 实现来源规则（本轮冻结，供后续轮次沿用）

- **有官方代码** → 实现来源 **VERIFIED**
- **仅有第三方/非官方代码，或代码 URL 未定位** → 实现来源 **UNVERIFIED**；该候选**仅当**同时满足
  (i) 机制在论文中被**完整而简单**地规定、(ii) 估计 LOC **< 150**、(iii) 不需要训练大型新网络
  时，才允许以「**明示的 from-paper 最小实现**」进入 E5-1。
- **否则 = HOLD**（§17：NO APPROXIMATE REIMPLEMENTATION WITHOUT DISCLOSURE）。

## B7. Composition Logic Audit（§13，重新核实）

每组必须回答：**A 解决什么 / B 解决什么 / 插入点是否不同 / 机制是否互补 / 是否冗余**。

### C01 × C04 →（**C04 已 HOLD，本轮不进入组合**）
- Module A (C01) solves: **input-level photometric nuisance**（照度分量从输入端被分解掉）
- Module B (C04) solves: **bank/distance-level 表示与局部密度偏差**
- Different insertion points: **YES**（input vs distance metric）
- Mechanistically complementary: **UNCERTAIN** —— REB 的 DefectMaker 也依赖**合成缺陷**，与 C06 同源
- Redundant: **NO**
- Composition recommendation: **+**（**本轮不执行**，待 C04 代码补齐后重评）

### C01 × C05 →（**C05 已 HOLD，NOT_DIRECTLY_PORTABLE**）
- Module A (C01) solves: input photometric nuisance ｜ Module B (C05) solves: **频段级重建能力**（非可插拔）
- Different insertion points: **N/A**（C05 无独立插入点）
- Mechanistically complementary: **UNCERTAIN** ｜ Redundant: **NO**
- Composition recommendation: **0 → 暂缓**（E5-0 曾记 **++**，**现下调**：C05 无法作为独立件，原互补逻辑失去载体）

### C01 × C06 →（两组均在册）
- Module A (C01) solves: **input photometric nuisance removal**（去除扰动来源）
- Module B (C06) solves: **defect-preservation compensation**（在特征空间学"偏离 normal 流形"的方向，保住判别性）
- Different insertion points: **YES**（input vs post-concat）
- Mechanistically complementary: **YES** —— 一个"减少 nuisance 来源"，一个"补偿 preservation"，**作用对象与阶段都不同**
- Redundant: **NO**
- Composition recommendation: **++**（**保留，列为本轮第一优先组合假设**）

### C05 × C06 →（**C05 已 HOLD**）
- Composition recommendation: **0 → 暂缓**（原 **++** 失去载体；C06 可与 C01 组成 ++）

### 新增（frozen 4 内部组合假设）
| 对 | A solves | B solves | 插入点不同 | 互补 | 冗余 | 推荐 |
|---|---|---|---|---|---|---|
| **C01 × C07** | input photometric nuisance | 表示判别性（线性调制） | YES（input / post-concat） | UNCERTAIN（C07 未显式约束 illumination） | NO | **+** |
| **C01 × C08** | input photometric nuisance | **bank 表示容量与覆盖**（连续网格取代离散 bank） | **YES（input / memory bank）** | **YES** —— C08 直击我们已观测到的真实约束（bank 仅 1,200 图受显存所迫） | NO | **++** |
| C06 × C07 | 合成异常判别 | 对比式表示投影 | NO（均在 post-concat） | **UNCERTAIN** | **YES**（两者都在"学更好的特征表示"） | **–** |
| C06 × C08 | 合成异常判别 | bank 表示 | YES（post-concat / memory bank） | UNCERTAIN | NO | **+** |
| C07 × C08 | 表示投影 | bank 表示 | YES（post-concat / memory bank） | UNCERTAIN（两者都改变"正常"的表示） | **可能 YES** | **0** |

**⚠️ 组合纪律**：`++` 的组合必须写成"**A 解 nuisance + B 补 preservation/容量**"，**不得**写成"A 有效 + B 也有效 = 随机叠加"。

## B8. E5-1 FROZEN CANDIDATES（最终冻结名单）

范围固定：**M²AD Bird ｜ seed 0 ｜ frozen view `120` ｜ illumination 01–10（700 图）**；**Original 复用 E4-X（0 GPU）**；**1 worker 强制**。

| # | Module | Insertion point | Implementation source | 预计 runtime/unit | 预计 VRAM | **Role** |
|---|---|---|---|---|---|---|
| 1 | **C01 PIAD-Retinex** | **input** | PIAD `github.com/Kaichen-Yang/piad_baseline`（MIT）的 `Retinex/` 子模块；**Retinex 权重的确切来源与许可待钉死** | ≈18 min（fit 1,200 图含 Retinex 前处理 ≈16 min ＋ score 700 图 ≈2 min） | ≈15 GB（+0.5） | **NOVELTY-BEARING 候选**（input-photometric 族） |
| 2 | **C06 SimpleNet 判别器** | **post-concat** | `github.com/DonaldRR/SimpleNet`（官方，PyTorch） | ≈17 min（fit ≈13 min ＋ adaptor/discriminator 训练 ≈3 min ＋ score ≈1 min） | ≈15 GB | **`ROLE = SUPPORTING_COMPONENT`**（baseline / defect-preservation compensator） |
| 3 | **C07 ReConPatch 线性调制** | **post-concat** | **无官方仓库**；仅非官方实现 ⇒ 以**明示的 from-paper 最小实现**（线性层 + 对比损失，LOC 50–150） | ≈15 min（fit ≈13 min ＋ 线性层对比训练 ≈1 min ＋ score ≈1 min） | ≈15 GB | **NOVELTY-BEARING（弱）**（learned-projection 族） |
| 4 | **C08 CRAD 连续记忆表示** | **memory bank** ✓（本项目**从未探索**的插入点） | `github.com/tae-mo/CRAD`（官方）；**需明示适配**到我们的 `wide_resnet50_2` layer2/3 特征 | ≈40 min（网格/融合网训练 50 epochs ≈35 min ＋ score ≈3 min；**须实测**） | ≈16–19 GB | **NOVELTY-BEARING（强）**（memory-bank 族） |

### ⚠️ 必须随冻结名单一起声明的三条纪律

1. **C07 的实现来源为非官方** ⇒ E5-1 中必须**逐条披露**：我们实现的是**论文所载的线性调制 + 对比伪标签**，**不是**复现某个官方仓库。
2. **C08 需要明示适配** ⇒ 官方实现自带 EfficientNet-b4 管线；我们把它接到**自己的 wide_resnet50_2 特征**上。
   这属**已披露的适配**，**不得**声称"原样复现 CRAD"。
3. **C06 只作 SUPPORTING_COMPONENT**，**不得**出现在论文的 novelty 陈述中。

## B9. VACANCY 报告（§12 要求）

目标阵容原为 **5–6 个**（E5-0 §9 曾列 8 项）；本轮核验后**只有 4 项**通过 §11 判据：

```
VACANCY = 1–2 slots
```

淘汰原因（**不自动补位**）：
- **C05 淘汰（NOT_DIRECTLY_PORTABLE）**：E5-0 曾给 **76 分 / 第 3 名**，本轮因**机制实质是 GAN 重建框架**而 HOLD ⇒ **这是 E5-0 的一处实质性误判，已如实记录**。
- **C04 HOLD（代码/实现来源不足）**：E5-0 曾给 **78 分 / 第 2 名**；机制审判**通过**，但**无法在不臆造的前提下实现**。
- **C02 / C09** 维持 HOLD（redundant / protocol）。

**关于 C10（CostFilter-AD）是否补位**：**本轮不作决定**。
理由：C10 位于 **"score / cost 后处理"层**，与已关闭的 score-fusion family **相邻**；且本轮**未核实**其机制是否作用在 cost map 而非我们的 final score。
⇒ **由下一轮单独决定**（§12 明确禁止自动补位）。

## B10. E5-1 预算估算（基于真实实现重新估计）

| 项 | 估算 |
|---|---|
| **GPU units** | **4**（每候选 1：1 训练/fit + 1 score）＋ **1 备用** ＝ **4–5** |
| **Estimated runtime** | **≈1.5–2.5 h**（4 × 15–40 min；C08 是主导项且**未实测**） |
| **Expected peak VRAM** | **≈16–19 GB alloc**（≪ 24 GB；**1 worker**；C08 因不需要离散 bank 可能更低） |
| Original | **复用 E4-X，0 GPU** |
| X6c 类 score 融合 | 0 GPU（CPU 后处理） |
| 失败处理 | 记录 → 最多重试 2 次 → 跳过该 unit 并保留其余结果，**不改科学协议** |

> 说明（§16）：本轮实测预算**低于** E5-0 的 2.5–4.5 h，因为**候选从 8 个降到 4 个**。
> **不为了凑满时间增加无意义实验。**

## B11. E5-0B 最终判定表

| Candidate | Verification | Mechanism | Closed-family? | Portable? | **Decision** |
|---|---|---|---|---|---|
| **C01** PIAD-Retinex | **VERIFIED**（PIAD）＋ PARTIALLY（Retinex 件归属/许可） | Retinex 反射率–光照分解，**input 级** | **NO** | **YES** | **PASS** |
| **C04** REB | **PARTIALLY**（venue VERIFIED；代码 NOT FOUND） | DefectMaker（合成缺陷+自监督域适应）＋ LDKNN（局部密度 KNN） | **NO**（机制审判 **PASS**） | **UNCERTAIN** | **HOLD**（实现来源不足） |
| **C05** Omni-Frequency | **VERIFIED**（TIP 2023 + DOI） | OCR-GAN：FD 作用于**输入图像**、CS 作用于**多 encoder 之间** | NO | **NO**（NOT_DIRECTLY_PORTABLE） | **HOLD** |
| **C06** SimpleNet | **VERIFIED**（CVPR 2023 + 官方代码） | 冻结特征 → adaptor → 合成异常 → 判别器 | NO | **YES** | **PASS** — `ROLE = SUPPORTING_COMPONENT` |
| **C07** ReConPatch | **PARTIALLY**（venue WACV 2024 VERIFIED；**代码非官方**） | 冻结 patch 特征的 **linear modulation** + 对比伪标签 | NO | **YES** | **PASS**（须披露 from-paper 实现） |
| **C08** CRAD | **VERIFIED**（ECCV 2024 + arXiv + 官方仓库） | **可学习特征网格 + 双线性插值**取代离散 bank（O(1)、无 NN 搜索） | NO | **UNCERTAIN → YES with disclosed adaptation** | **PASS**（须披露适配） |
| C02 FiCo | VERIFIED | Filter ≈ Exp13-P 投影 | — | — | **HOLD_REDUNDANT** |
| C09 On-The-Fly | VERIFIED | test-time 自适应 | — | — | **HOLD_PROTOCOL** |

## B12. 本轮保留的 negative result（不得删除）

1. **E5-0 对 C05 的评分（76 分 / 第 3 名）被本轮推翻** —— 高分会掩盖"不可插拔"这一致命工程事实。
   记录：**评分不能替代可移植性核验**。
2. **E5-0 对 C04 的 78 分（第 2 名）** 同样偏高：机制虽非 closed-family，但**实现来源不足**使其无法进入 E5-1。
3. **C02 的 Filter 分支与 Exp13-P 同思路** —— 这是本项目**第二次**在外部文献中遇到"滤除 nuisance 方向"的路线，
   而 Exp13-P 已证明该方向**非 defect 特异** ⇒ 该路线的**负面证据被加强**，不得因外部论文采用而翻案。
4. **C06 的 novelty collision = D** —— 维持 E5-0 判定，**不得**因为"要用它"而软化。

## B13. 下一步

**NEXT = E5-1 — Broad Mini Screening（4 candidates）**，**尚未启动**，需人工批准。

**E5-1 开跑前的剩余待办（CPU-only）**：
1. 钉死 **C01 的 Retinex 权重来源与许可**（PIAD `Retinex/` vs URetinex-Net 独立仓库）。
2. 决策本文件 **B9 的 VACANCY**（是否补位、C10 是否值得）—— **由人工决定**。
