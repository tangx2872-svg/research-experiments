# E26 · AnomalyDINO 官方实现完整基线（预注册协议）

日期：2026-10-30
阶段：论文投稿前最后一轮补充 — 外部基线比较
状态：**预注册，先于任何 E26 结果**

---

## 1. 目的

检验 R1 相对**完整外部方法基线**的表现，补齐目前主要依赖 F0（本文受控基线）比较的不足。

论文要证明的话：本文的适配器 R1 相对一个独立发表的、完整的外部方法仍有竞争力（或至少不劣）。
本实验排除的替代解释：R1 相对 F0 的提升仅仅来自"我们自己的 baseline 太弱"。

## 2. 边界（不得越界）

- 不修改 R1 方法、超参数、数据划分、评分规则；不根据本次结果重选配置。
- 不新增其他主干、数据集、消融、扰动实验；不补跑多视角 seed1/2；不下载 PatchCore 原图。
- 正负结果全部保留；结果不理想不构成追加实验的理由。
- 不改动既有代码、日志、结果；全部新增产物写入 `results/e26_anomalydino/`。

## 3. 官方来源核验

| 项 | 值 |
|---|---|
| 论文 | AnomalyDINO: Boosting Patch-based Few-shot Anomaly Detection with DINOv2 |
| 作者 | Simon Damm, Mike Laszkiewicz, Johannes Lederer, Asja Fischer |
| 发表 | WACV 2025；arXiv:2405.14529 |
| 代码 | `third_party/AnomalyDINO`（本地克隆） |
| commit | `b9d1c2648e3a5247437d4d953d907a8f3d994457` |
| 许可证 | 见 `third_party/AnomalyDINO/LICENSE` |

官方算法核心（逐字对应 `src/detection.py`）：

1. 参考特征：`model.prepare_image` → `model.extract_features` → `compute_background_mask`；
   参考图旋转增广 `augment_image`（8 个角度 `[0,45,...,315]`）。
2. 检索：`faiss.IndexFlatL2` + `faiss.normalize_L2`，`knn_metric="L2_normalized"`，`knn_neighbors=1`；
   距离换算 `distances = L2_squared / 2`（= 1 − cosine similarity）。
3. 图像得分：`mean_top1p(distances)` = 全部 patch 距离的**前 1% 均值**（1024 patch → 前 10 个）。

## 4. 为匹配本文协议所作的调整（逐条记录）

| # | 官方设定 | 本实验设定 | 理由 | 影响 |
|---|---|---|---|---|
| A1 | MVTec/VisA 目录布局 | 每 (类别, seed) 生成符号链接树 `{cat}/train/good`, `{cat}/test/good`, `{cat}/test/anomaly` | M²AD 无官方布局 | 仅数据寻址；图像像素不变 |
| A2 | `--shots k`（few-shot，前 k 张） | `n_ref_samples = -1`（full-shot），参考图 = 该 seed 的 **300 张 bank 图**（与 F0/R1 完全相同的 300 张） | 与本文 300-bank 协议一致 | 参考集规模远大于官方 few-shot 设定 |
| A3 | 官方 `agnostic` 对 MVTec 逐对象选择 masking | **masking = False**（官方对未知数据集的 fallback：`src/utils.py:172-196` 打印 "custom dataset ... default setting ('agnostic_no_mask')"） | 官方代码对自定义数据集的默认分支，非本文选择 | 无 PCA 掩码；参考图与测试图均不掩码 |
| A4 | `rotation=True`（`agnostic_no_mask` 默认） | **保持 True** | 官方默认 | 参考集 300×8 = 2400 张 |
| A5 | 官方评测读目录标签（`test/good` vs `test/<type>`） | 标签取自官方元数据 `image_anomaly`，并按 `detectable != False` 过滤 | 本文评价口径（与 F0/R1 完全一致） | 仅评测；得分不受影响 |
| A6 | `faiss-gpu` | 环境仅有 faiss 1.15.1 (CPU)，**不允许 pip/conda 安装** → 用 PyTorch fp32 精确 1-NN 替代 | 依赖约束 | 数学等价；已做等价性验证（见 §7） |
| A7 | `torch.hub.load('facebookresearch/dinov2', ...)` | 改为从本地缓存目录 `source='local'` 载入同一 repo 与权重 | 本轮 GitHub 网络不稳，hub 的在线校验会阻塞 130 s 并可能直接失败 | 纯离线寻址，权重与代码完全相同；不改变数值 |

**命名**：本结果一律称"**官方实现的本协议配置**"（official implementation under this protocol），
不得称"AnomalyDINO 原论文数值"，也不得把本文 F0 称为 AnomalyDINO。

## 5. 运行配置（冻结）

| 项 | 值 |
|---|---|
| backbone | `dinov2_vits14`（官方默认） |
| 分辨率 | 448（官方默认，短边 resize + BICUBIC + ImageNet 归一化 + 裁到 14 的倍数） |
| 检索 | L2_normalized, k=1 |
| 聚合 | mean of top 1% patch distances |
| masking | False |
| rotation | True（8 角度） |
| 类别 | 10 类全部 |
| 视角 | view120 |
| seed | 0 / 1 / 2（决定 300 张 bank 图像） |
| 参考集 | 与 `results/e22/cells/{Cat}_seed{s}.json` 的 F0/R1 完全相同 |

不搜索任何超参数、不挑选增强或评分方式。

## 6. 成本估算（先于启动，实测基准）

实测（RTX 3090，本机）：特征提取 47.6 ms/张（含 PNG 解码）；2.46M×384 银行精确 1-NN 检索 140 ms/张。

| 项 | 单 cell | ×30 cell |
|---|---|---|
| 构建银行（2400 张） | 114 s | 57 min |
| 测试提取（700 张） | 33 s | 17 min |
| 检索（700 张） | 98 s | 49 min |
| 合计 | **245 s** | **≈ 2.05 GPU·h** |

**结论：约 2.1 GPU·h < 4 GPU·h 上限，不需要预先请示替代方案。** 另加等价性验证与 bootstrap（CPU）约 15 min。

**实测校正（首个 cell 完成后回填）**：单 cell = 银行 76 s + 测试 115 s ≈ **191 s**（低于估计的 245 s），
30 cell ≈ **96 min ≈ 1.6 GPU·h**。差异主要来自 2.46M 银行检索实测优于基准值。

## 7. 强制核验项

1. **样本身份**：AnomalyDINO 评测集合必须与 R1 cell 的 `labels`/`detectable_raw` **逐图同集合**（n_ok/n_ng 完全相等），否则该 cell 作废。
2. **后端等价性**：在缩小设定（30 张参考 × 8 旋转、10 张测试）上同时跑官方 `run_anomaly_detection`（faiss-CPU）与本次 torch 后端，报告 max|Δscore|。
3. **官方代码未被修改**：`git -C third_party/AnomalyDINO status --porcelain` 为空。
4. **输入模式偏差量化**：Bird 三个 seed 同时保留"1024² 原图"与"l0_448 缓存"两条输入路径的结果，
   报告 `max|Δscore|`、`ΔAUROC`、`ΔAP`（`scores_originput/` vs `scores/`）。
   若 |ΔAUROC| > 0.01，则 9 个仅缓存类别的结果必须降级为"近似"表述。

## 8. 出口条件

- 若 AnomalyDINO（本协议配置）在 10 类宏平均上 **高于 R1** → 论文必须收缩"相对外部方法更优"的任何表述，只保留受控比较结论。
- 若 R1 高于该基线 → 可作为一条外部比较证据，但不得表述为"SOTA"或"显著领先"。
- 无论方向，均不追加新实验。
