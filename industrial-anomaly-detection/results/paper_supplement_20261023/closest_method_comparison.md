# 近似方法差异对照（closest_method_comparison）

**日期**：2026-10-23
**性质**：文献层面的一手/官方资料核对，**非实验对比**。所有"本项目是否已运行"均为事实陈述，不构成性能领先声明。
**约束**：本轮**不**在预算外自动复现任何外部方法；没有官方代码/无法唯一确认的方法，如实标注"未确认"。

---

## 0. 结论摘要（先读）

| 方法 | 与 R1-ILA 关系 | 能否直接对比 | 官方代码/许可 |
|---|---|---|---|
| **PIAD**（CVPR 2025） | **同问题域**（光照无关 AD），技术路线完全不同 | 概念对照（强相关），**不可直接换数字比** | 项目页有，代码/数据"after acceptance"（尚未公开验证） |
| **AnomalyDINO**（WACV 2025） | **同 backbone + 同评分族**，training-free，无 adapter | **最可比的 baseline**，但协议不同（few-shot MVTec/VisA，非 M²AD 光照协议） | 有官方代码（Apache-2.0，DINOv2；数据 MVTec CC BY-NC-SA） |
| **CAP（FAPrompt）**（ICCV 2025） | 都是"轻量适配 + 冻结 backbone"，但 CLIP vs DINO、prompt vs linear residual | 仅方法族对照 | 有官方代码（mala-lab/FAPrompt） |
| **PG-PSR** | **无法唯一确认**（见 §5） | 不写进对比表正文 | 未确认 |

**核心判断**：R1-ILA 的最强"近邻"不是残差适配（ADRA，见之前的文献核查），而是**同用 DINOv2 + patch-NN + 正常参考的 AnomalyDINO**——差别在于 R1-ILA 多了一个**可训练的线性残差适配器 + 光照一致性损失**，而 AnomalyDINO 是纯 training-free。这既是 novelty 的来源，也是需要诚实划界的地方。

---

## 1. PIAD — Pose and Illumination agnostic Anomaly Detection（CVPR 2025）

| 维度 | PIAD | R1-ILA |
|---|---|---|
| 问题 | 姿态 + 光照无关 AD（PAD 的推广） | 光照变化下的 AD（不处理姿态） |
| 核心机制 | 3D Gaussian Splatting（3DGS）重建 + 合成 reflectance 图 + 相机姿态估计/优化 | 单层线性残差 `z' = z + Wz`，normal-only 训练 |
| 是否需要多视角/三维 | **是**（COLMAP 相机标定 + 多视角采集 + foreground mask） | **否**（单视角 view120 即可） |
| 是否需要配对采集 | 需训练/查询在同类场景，靠 3D 重建对齐 | 不需要 |
| backbone 冻结 | 是（3DGS 重建与深度特征比较，非 DINO） | 是（DINOv2-B 冻结，仅训 W） |
| 适配目标 | 重建 reflectance（光照无关量） | 让 `z'` 在光度扰动下保持一致 |
| 评分 | RGB/reflectance 光度比较 + 多尺度深度特征 | 1-NN-max（patch 级 cosine 距离） |
| 数据需求 | 需标定（COLMAP）、mask | 只需正常图 + 标签 |

**与本项目关系**：这是**概念上最直接的同域竞争**（都在做"光照不一致下检测异常"），但技术路线、数据采集要求、可部署性差异巨大。PIAD 需要 3D 重建和相机标定，R1-ILA 是单视角轻量线性适配。**论文 related work 必须显式引用 PIAD 并划界**：R1-ILA 不依赖 3D/姿态/配对采集，代价是只处理光照、且增益幅度小且条件性。

**官方代码状态**：项目页 https://kaichen-yang.github.io/piad/ 存在，但数据集/代码标注"released after acceptance"；**本轮未验证其是否已开源**。**未确认**可复现性。

---

## 2. AnomalyDINO — Boosting Patch-based Few-shot AD with DINOv2（WACV 2025）

| 维度 | AnomalyDINO | R1-ILA |
|---|---|---|
| backbone | DINOv2（frozen） | DINOv2-B（frozen） |
| 评分 | patch 级 deep NN（memory bank 最近邻距离） | 1-NN-max（patch 级 cosine 距离） |
| 适配/训练 | **training-free**（无任何可训练模块） | **有** 768×768 线性残差 W（~59 万参数，normal-only 训练） |
| 一致性损失 | 无 | 有（光照一致性 L_cons + 0.1×保持 L_preserve） |
| 预处理 | PCA masking（zero-shot 分割）+ 旋转增广 reference | 亮度/对比度/gamma 光度增广 |
| 任务设定 | few-shot（1/2/4/8/16 shots）MVTec/VisA | 全量 normal bank（300）+ 多光照 M²AD |
| 数据 | MVTec-AD、VisA | M²AD、RobustAD |

**与本项目关系**：这是**最直接的 baseline 候选**——两者共享"frozen DINOv2 + patch-NN + 正常参考"的技术骨架。R1-ILA 相对它的增量 = ①一个可训练线性残差适配器、②光照一致性损失、③在**真实多光照工业数据（M²AD）**上的评测。AnomalyDINO 官方在 MVTec/VisA 上，**没有**在 M²AD 上报告。

**关键诚实点**：R1-ILA 的 F0 baseline 本质上就是"不带 PCA masking 的 AnomalyDINO 式 DINOv2 patch-NN"，所以 R1-ILA 相对 F0 的增益，可以直接解读为"给 AnomalyDINO 式 baseline 加一个光照一致性线性适配器带来的变化"——这是一个**强对比叙事**，但必须注意协议差异（AnomalyDINO 有 PCA masking + 旋转增广，R1-ILA 无）。

**官方代码状态**：github.com/dammsi/AnomalyDINO（Apache-2.0，DINOv2；数据 MVTec CC BY-NC-SA 4.0、VisA CC BY 4.0）。**已确认存在**。

---

## 3. CAP — Compound Abnormality Prompt（FAPrompt，ICCV 2025）

| 维度 | CAP / FAPrompt | R1-ILA |
|---|---|---|
| backbone | CLIP（frozen，VLM） | DINOv2-B（frozen，vision-only） |
| 适配机制 | learnable text prompt（normal/anomaly 分解 prompt） | linear residual W |
| 是否用文本/语言 | **是**（需类别 prompt，语言先验） | **否**（纯视觉） |
| 训练监督 | focal/dice 局部 + focal 全局 + CAP/DAP 约束（**需异常示例/分割监督**） | normal-only（**无异常示例**） |
| 任务设定 | zero-shot（跨 19 数据集） | normal-only 训练 |

**与本项目关系**：仅"冻结 backbone + 轻量适配"这一族谱上相近，但**监督信号、模态、任务设定都不同**。FAPrompt 需要文本 + 异常监督（localization），R1-ILA 是纯视觉 + normal-only。**不是直接竞争**，但 related work 可提一句"轻量适配冻结 backbone 的思路已有先例（如 prompt learning），本文的不同在于纯视觉 + normal-only + 线性残差"。

**官方代码状态**：mala-lab/FAPrompt（GitHub）。**已确认存在**。

---

## 4. 与 ADRA / YOLA 的关系（承接之前的文献核查）

之前在《文献创新性核查报告》已确认：
- **ADRA**（arXiv 2020）：占"冻结 backbone + 残差适配器"架构前提，但用于 CIFAR/STL 语义异常 + Outlier Exposure（需异常代理监督）。
- **YOLA**（NeurIPS 2024）：占"光照一致性损失 + 光度扰动"损失设计，但用于低光照目标检测，非 AD。

R1-ILA 相对这两者的差异：**纯 normal-only（无异常代理）+ 单层线性（非逐层 1×1）+ 表示锚定项 L_preserve（防塌缩）+ 真实多光照工业 AD 域**。这与本表 §2 的 AnomalyDINO 划界共同构成 novelty 的边界。

---

## 5. PG-PSR —— 无法唯一确认

指令提及 "PG-PSR"，但经检索**未能在异常检测文献中唯一确认该缩写**。候选（均不确认为指令所指）：

| 候选 | 含义 | 域 |
|---|---|---|
| **PG-LSR** | Periodic-pattern Guided Low-rank/Sparse（HKUST-GZ 2024，周期性图案 untrained AD） | 周期性纹理 AD |
| **PGSR** | Pixel-Grounded Super-Resolution（超分辨率，非 AD） | SR |
| **PPR** | Patch Position Regression（MLR 2022，位置回归 AD） | 3D 体素 AD |

**处置**：**未确认 = 不写入对比正文、不冒称已对比**。若写作侧确指某方法，请提供其全名/arXiv/官方代码，我再补一行。符合指令"没有官方代码就明确写未确认"。

---

## 6. 本项目是否已运行（事实陈述）

| 方法 | 本项目是否已运行 | 说明 |
|---|---|---|
| PIAD | **否** | 需 3DGS + COLMAP + 多视角，超出本方法范式；未运行 |
| AnomalyDINO | **否** | 未运行官方代码；但 F0（frozen DINOv2-B + 1-NN-max）在技术上等价于"去掉 PCA masking/旋转增广的 AnomalyDINO" |
| CAP/FAPrompt | **否** | 未运行 |
| ADRA / YOLA | **否** | 未运行 |
| PG-PSR | **否**（且未确认指代） | — |

**不构成任何"实验领先"声明**：本表只说明"本项目没有跑这些方法"，以及 R1-ILA 与它们的**概念差异**，不隐含性能比较。

---

## 7. 若需补对照：唯一推荐的最小可执行方案

**最值得补的对比 = AnomalyDINO**（同 backbone + 同评分族 + training-free，是 R1-ILA 相对 F0 的最干净外部锚点）。

**最小方案**（仅当确认官方实现 + 许可 + 公平协议后）：
1. 用 AnomalyDINO 官方 `run_anomalydino.py`，在 M²AD view120 上跑 training-free 基线（不启用 PCA masking，或用官方默认 agnostic 预处理，二选一并**明确记录**）。
2. 与 R1-ILA 的 F0/R1 在同一 10 类 × 3 seed × 300-bank 协议下对比。
3. **公平性前提**：AnomalyDINO 的 bank 构建、评分、增广需与 E22 协议对齐；否则只能报告"非严格可比"。

**资源估算（粗）**：DINOv2-B 前向（训练集 ~300 图 + 测试集 ~700 图/类 × 10 类）约 2–4 GPU 小时（RTX 3090）；需 M²AD 原图 + DINOv2 权重（本地缺失，见 HANDOFF）。

**结论**：本轮**不执行**（缺数据 + 缺权重 + 需先确认公平协议），仅作为下一阶段的可选动作，标记"未运行、未确认可复现"。

---

## 8. 引用信息（供论文 related work 使用）

- PIAD: Yang et al., "PIAD: Pose and Illumination agnostic Anomaly Detection", CVPR 2025, pp. 4734–4743.
- AnomalyDINO: Damm et al., "AnomalyDINO: Boosting Patch-based Few-shot Anomaly Detection with DINOv2", WACV 2025, arXiv:2405.14529.
- FAPrompt/CAP: Zhu et al., "Fine-grained Abnormality Prompt Learning for Zero-shot Anomaly Detection", ICCV 2025.
- ADRA: Deecke et al., "Deep Anomaly Detection by Residual Adaptation", arXiv:2010.02310 (2020).
- YOLA: NeurIPS 2024（低光照目标检测，见《文献创新性核查报告》）。
