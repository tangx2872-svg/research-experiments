# 投稿前最后一轮补充 · 最终报告

**日期**：2026-10-30
**范围（严格执行，未扩大）**：①1 个新增基线实验（AnomalyDINO 官方实现）②1 项已有得分分析（真实光照同工件配对）③文献核查（CAP/FAPrompt、AnomalyDINO、PG-PSR）
**未做**：不修改 R1 方法/超参/划分/评分规则；不新增其他主干、数据集、消融、扰动；不补跑多视角 seed1/2；不下载 PatchCore 原图；不运行 CAP / PG-PSR 等新方法。
**GPU 预算**：估算 2.1 GPU·h → 实际约 **1.4 GPU·h**（< 4 GPU·h 上限，未请示替代方案）。

---

## 0. 执行摘要

| 交付 | 状态 | 一句话结论 |
|---|---|---|
| AnomalyDINO 完整基线 | **完成**（30/30 cell，身份与 bank 哈希全部核对通过） | 官方实现（本协议配置）宏平均 AUROC **0.8036**；R1 **0.8464**，R1−AD = **+0.0428 [+0.0251, +0.0602]**；但差距主要来自 backbone/聚合配置，**不能归因于适配器** |
| 真实光照配对分析 | **完成**（仅用已有得分，0 GPU） | 正常工件跨光照得分 SD 在 **10/10 类下降**，宏平均 **−0.0080 [−0.0089, −0.0072]**；9/10 类同期 AUROC 上升，**Doll 为唯一例外** |
| 文献核查 | **完成** | PG-PSR **仍无法唯一确认**（不得写入正文）；CAP 是 FAPrompt 的**模块名**（不得当方法名）；已确认方法与 R1 的七维差异 |

---

## 1. 交付物一：AnomalyDINO 官方实现完整比较

### 1.1 出处与可运行性（已核验）

| 项 | 值 |
|---|---|
| 论文 | AnomalyDINO: Boosting Patch-based Few-shot Anomaly Detection with DINOv2 |
| 作者 / 发表 | S. Damm, M. Laszkiewicz, J. Lederer, A. Fischer；**WACV 2025 (Oral)**；arXiv:2405.14529 |
| 代码 | `third_party/AnomalyDINO`，commit `b9d1c2648e3a5247437d4d953d907a8f3d994457`，`git status` 干净（未修改） |
| 训练 | **training-free**（无任何可训练参数） |

### 1.2 官方实现的保留项与协议调整（完整 8 条见 `results/e26_anomalydino/protocol.md` §4）

**保留**：`DINOv2Wrapper.prepare_image/extract_features`、`augment_image`（8 角度旋转）、
`L2_normalized` k=1、官方 `mean_top1p`（前 1% patch 距离均值）。

**调整**：A1 目录布局（符号链接）；A2 full-shot 300 张（= 该 seed 的 bank）；
A3 masking 关闭（**官方对自定义数据集的 fallback `agnostic_no_mask`，非本文选择**）；A4 rotation 保持官方默认 True；
A5 评测改用 `image_anomaly + detectable != False`；A6 faiss-CPU → torch 精确 1-NN（**max|Δscore| = 1.8e-07**，官方 faiss 路径逐图校验）；
A7 torch.hub 离线寻址；A8 输入改用 `l0_448` 缓存（见 1.4）。

**结果命名**：一律称"**官方实现的本协议配置**"，不称"AnomalyDINO 原论文数值"，也不把 F0 称为 AnomalyDINO。

### 1.3 主要结果（10 类 × 3 seed，每类先 3-seed 平均再类间等权）

| 方法 | AUROC | AP |
|---|---:|---:|
| AnomalyDINO（官方实现·本协议配置） | 0.80357 | 0.85918 |
| F0（本文受控基线） | 0.83881 | 0.89396 |
| **R1（本文）** | **0.84641** | **0.90173** |
| R1 − AD | **+0.04284** [+0.02508, +0.06018] | **+0.04254** [+0.02726, +0.05870] |
| AD − F0 | −0.03525 | −0.03477 |

逐类别见 `results/e26_anomalydino/baseline_category_summary.csv`；R1 正向 7/10，其中 **6 类 CI 排除 0**
（Bird +0.115、Car +0.070、Dice +0.064、**Ring +0.061（边际，下界仅 +0.0008）**、Teapot +0.076、Tube +0.058；
Cube +0.039 的下界 −0.009 跨零，故不计入）；**AnomalyDINO 领先 3/10**（Doll −0.009、Holder −0.017、Motor −0.028，
即 R1−AD 为负），**这 3 类 CI 均跨零**。

开发 3 类 +0.0746 [+0.0408, +0.1106]；扩展 7 类 +0.0292 [+0.0080, +0.0503]（描述性，不作为独立验证）。

### 1.4 输入模式偏差（必须披露）

服务器仅 Bird 保留 1024² 原图，其余 9 类只有 `l0_448`。缓存与官方的几何处理一致（短边 448 + BICUBIC + antialias，
M²AD 原图统一 1024²），但缓存以 **PNG uint8** 落盘，多一次 8-bit 量化。Bird 三 seed 实测：

| seed | max\|Δscore\| | ΔAUROC（缓存−原图） | ΔAP |
|---:|---:|---:|---:|
| 0 | 0.0105 | −0.00085 | +0.00166 |
| 1 | 0.0146 | −0.00089 | −0.00058 |
| 2 | 0.0186 | −0.00085 | +0.00055 |

**⇒ 指标层偏差 ≤ 0.0017，不影响结论；但正文应说明输入为 448 缓存。**

### 1.5 结论边界（关键）

本比较**不能**证明"R1 的方法优于 AnomalyDINO"，因为三处官方组件未对齐：
backbone（ViT-S/14 vs ViT-B/14）、图像得分聚合（top-1% 均值 vs max）、参考旋转增广。
**即使完全不训练，F0 也已高出 0.0352** —— 差距主要来自配置。可支持的表述：
"在与一个独立发表的完整方法按同一数据协议比较时，本文方法未处劣势"。

---

## 2. 交付物二：真实光照下的同工件配对得分变化

口径：正常测试工件在 10 个真实光照下得分的**样本标准差**（唯一主要指标）；类内工件等权 → 3 seed → 10 类等权；
工件级配对 bootstrap，3 seed 共用抽样，3000 次。**事后探索性**。

| 层级 | ΔSD (R1−F0) | 95% CI | 方向一致性 |
|---|---:|---|---|
| **10 类宏平均** | **−0.008003** | **[−0.008856, −0.007165]** | **10/10 类下降** |
| 开发 3 类 | −0.010982 | [−0.012558, −0.009459] | 3/3 |
| 扩展 7 类 | −0.006726 | [−0.007735, −0.005755] | 7/7 |

逐类别 ΔSD：Bird −0.0171、Motor −0.0119、Car −0.0105、Holder −0.0092、Doll −0.0070、Teapot −0.0061、
Ring −0.0060、Cube −0.0053、Dice −0.0045、Tube −0.0024。

**配套检查（与 AUROC/AP 并列）**：9/10 类为"SD↓ + AUROC↑"；**Doll 是唯一"SD↓ 但 AUROC↓"的类别**
（ΔSD −0.0070，ΔAUROC −0.0385）；不存在"SD↑ 而 AUROC↓"或"SD↓ 且无排序变化"的隐含类别。
正常与异常得分**均整体下移**（−0.042 至 −0.174），两侧 SD **同时收缩** → 属**整体压缩**，
不得表述为"专门消除了光照漂移"或"证明因果机制"。

异常工件覆盖（detectable 过滤后 10 光照齐全）：16/37 ~ 44/50，逐类见 `results/e27_illum_paired/anomaly_illum_coverage.csv`。

---

## 3. 交付物三：文献核查与创新边界

完整版见 `results/e28_lit_review/LITERATURE_CHECK.md`。

| 对象 | 核验结果 | 处置 |
|---|---|---|
| **AnomalyDINO** | 题名/作者/WACV 2025 (Oral)/arXiv 2405.14529/code 全部确认；§3.1 式(1) 记忆库、式(4) 得分、§3.1 明确 q = 前 1% 均值；§3.2 "In the 'agnostic' case … we always augment the reference sample with rotations" | 写作引用 §3.1/§3.2/§4.2（CVF 页码未核实，用章节号） |
| **FAPrompt** | ICCV 2025 pp. **22241–22251**；**CAP = Compound Abnormality Prompting module（§3.3），方法名是 FAPrompt**；训练**需要**含异常样本 + 像素掩码的辅助集（§3.1/§3.5），且**依赖 CLIP 文本提示** | 正文不得用 "CAP" 指代该方法 |
| **PG-PSR** | 两轮独立检索均**无法在 AD 文献中唯一确认**；候选（PSR 视频 AD 模块、physics-guided periodic sparse representation、Prior-Guided Symbolic Regression）均不匹配 | **不写入正文对比**；审稿问及只能答"无法唯一对应" |

**最可能影响创新判断的已有设计**：AnomalyDINO §3.1（骨架同源：冻结 DINOv2 + 记忆库 + patch 1-NN + 尾统计量聚合）。
**本文剩余增量**：normal-only（无异常样本/无像素掩码）的线性残差适配 + 显式余弦锚定项 `L_preserve` +
真实多光照工业数据上的同工件配对评测。**不得声称"首次"或"现有方法均未考虑光照"。**

---

## 4. 可直接粘贴的正文建议

**（4a）外部基线段落**

> We additionally ran the official AnomalyDINO implementation (WACV 2025) under our protocol: the same 300 normal
> references per category and seed, the same test images, and the same label/detectable filtering, with masking
> disabled (the official fallback for custom datasets) and the official rotation augmentation and top-1% patch
> aggregation retained. Averaged over ten categories (three seeds each, equal weight per category), it reaches
> AUROC 0.8036 / AP 0.8592, versus 0.8464 / 0.9017 for our method (ΔAUROC +0.0428, 95% CI [+0.0251, +0.0602],
> specimen-level paired bootstrap). Because the two systems differ in backbone capacity (ViT-S/14 vs ViT-B/14),
> score aggregation (top-1% mean vs max) and reference augmentation, this comparison should be read as
> "not at a disadvantage under a shared protocol", not as evidence that the adapter alone accounts for the gap:
> the untrained baseline F0 already exceeds AnomalyDINO by 0.0352 AUROC.

**（4b）光照机制段落（补充分析小节）**

> Across the ten categories, the adapter reduces the dispersion of normal-image scores within a specimen across
> the ten real illuminations (mean ΔSD = −0.0080, 95% CI [−0.0089, −0.0072], paired by specimen with a shared
> resample across seeds; all ten categories decrease). In nine of ten categories this coincides with a higher
> AUROC, Doll being the exception. Since both class means shift downward and both class standard deviations shrink,
> we interpret this as a global compression of the score distribution rather than as illumination-specific
> invariance, and we make no causal claim. This analysis is post-hoc and exploratory.

**（4c）related work 划界**

> AnomalyDINO (Damm et al., WACV 2025) establishes frozen DINOv2 patch-level nearest neighbour retrieval with a
> normal memory bank as a strong training-free baseline (§3.1); FAPrompt (Zhu et al., ICCV 2025) learns lightweight
> prompts but on the CLIP text side and with an auxiliary training set containing anomalous samples and pixel masks
> (§3.1, §3.5). Our setting differs from both: the only supervision is pairs of photometric perturbations of the
> same normal image, and adaptation is a single linear residual in the visual feature space.

---

## 5. 论文可支持 / 不可支持

| 可支持 | 不可支持（必须删除或改写） |
|---|---|
| R1 在同协议下不低于独立发表的完整外部方法（7/10 类正向，宏平均 CI 不含 0） | "优于 AnomalyDINO 的方法设计"、"SOTA"、"显著领先"、"首创" |
| R1 降低同工件跨真实光照的得分离散度（10/10 类，CI 不含 0） | "消除了光照漂移"、"证明因果机制"、"改善固定阈值性能" |
| 9/10 类离散度下降与排序改善同向，Doll 为例外 | "所有类别一致改善" |
| 扩展 7 类（未参与选型）方向一致、幅度更小 | 把开发 3 类的 +0.075 当作独立验证 |
| 与 FAPrompt 在监督/模态/任务上均不同 | 与 CAP/PG-PSR 做过对比实验 |

---

## 6. 收口判断（四项）

1. **外部基线比较是否补齐？** **是。** 30/30 cell，官方实现未改，身份 + bank 哈希逐 cell 核对通过；
   逐图得分、配置、脚本、日志、比较条件表齐备。**残余限制**：输入为 448 缓存（指标偏差 ≤0.0017），
   且与 F0/R1 存在三处配置不对齐 → 只能作"未处劣势"表述。
2. **光照证据是否补齐到可准确表述？** **是。** 主指标单一（跨光照 SD）、口径预注册、10 类全列、
   区间不含 0、9/10 与排序指标同向、明确标注事后探索。**不能**升级为判别性能或机制结论。
3. **是否存在影响论文有效性的阻断问题？** **无新增阻断。**
   既有已知限制（`results/paper_ready/EVIDENCE_MAP.md` 记录的 D_shift 命名、Doll d′ 口径、
   train/val 共享工件 039、E21 仅 seed0、E25 test0 归档）仍需在写作时如实处理，但不影响本轮结论成立。
4. **是否建议进入全文修订与润色？** **建议进入。** 本轮范围内实验停止，不再追加。

---

## 7. 产物与校验

| 目录 | 内容 |
|---|---|
| `results/e26_anomalydino/` | 协议、2 脚本、30 个 cell 逐图得分、6 份分析 CSV、等价性验证、运行日志、README |
| `results/e27_illum_paired/` | 协议、2 脚本、7 份 CSV、README |
| `results/e28_lit_review/` | 文献核查报告（含章节级引用与对照表） |
| `results/final_supplement_20261030/` | 本报告 + 打包件 |

打包：`results/final_supplement_20261030/e26_e27_e28_bundle_20261030.tar.gz`

- 内容：`e26_anomalydino/` + `e27_illum_paired/` + `e28_lit_review/`，共 **63 个真实文件**，896 KB（本报告不入包，避免哈希自指；其校验值单独列于 `SHA256SUMS.txt`）。
- 已排除 `e26_anomalydino/work/`：该目录是 33,970 个指向 M²AD 数据集目录的符号链接，仅运行期脚手架，打包时既不携带内容也没有复现价值（打包件内 symlink 数已核验为 0）。
- 完整性已核验：解包到临时目录后与源目录逐文件 `diff -r`，结果完全一致（`EXTRACT_DIFF_CLEAN`）。
- 校验值（同目录 `SHA256SUMS.txt`）：打包件 `e26_e27_e28_bundle_20261030.tar.gz` 的 SHA256 记录如下；本报告自身的 SHA256 亦在该文件中（报告内无法自指嵌入）。

| 文件 | SHA256 |
|---|---|
| `e26_e27_e28_bundle_20261030.tar.gz` | `e51693f3389124c117e946a5b45a4a209dd94fd8b2dea826ec269c9dd2fb4cee` |

注意：打包件是已跟踪文件的副本，按仓库惯例 `.tar.gz` 不入 Git（体积冗余、可重打），仅 `SHA256SUMS.txt` 入库作为校验锚点。
