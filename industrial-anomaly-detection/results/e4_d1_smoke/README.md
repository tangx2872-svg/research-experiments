# E4-D1 — M²AD Bird × seed0 × Original PatchCore GPU Smoke

**日期**：2026-10-08
**论文阶段**：Phase II — Paper Validation ｜ `METHOD SEARCH = CLOSED` ｜ `FROZEN METHOD = X6c`
**实验编号**：E4-D1 ｜ **性质**：**pipeline smoke + 原始表征**，**不是 E4-A**
**起点 HEAD**：`14a3863`（branch `research/industrial-anomaly`）
**协议冻结**：`docs/E4_D1_PROTOCOL.md`（**先于任何 GPU 运行**写定）
**数据**：M²AD，category **Bird**（官方 `meta_unsupervised.json`，SHA256 见 `data/m2ad/DATASET_LOCAL_INFO.md`）
**seed**：0 ｜ **GPU**：RTX 3090 24 GB，**1 worker** ｜ **并发**：单进程

---

## 1. 要验证的东西（本实验服务于哪一句）

> **M²AD 的完整执行链（loader → fit → embedding → coreset → inference → score → GT/mask 对齐 → metric → 落盘）能否跑通，
> 真实 runtime / 显存 / 内存 / IO 是多少，以及 10 个 illumination 与 12 个 view 下的原始表现如何。**

**本实验不验证**：X6c 是否有效、trade-off 是否存在、hypothesis supported/rejected。
**明确未运行**：Adaptive B2、X6c、其他 seed、其他 category、任何阈值/PASS/catastrophic 判定。

---

## 2. 配置（逐字来自冻结协议）

| 项 | 值 |
|---|---|
| method | **Original**：`FAlphaDualLayerPatchcore(alpha_l2=0.0, alpha_l3=0.0)` |
| backbone / layers | `wide_resnet50_2` / `["layer2","layer3"]` |
| coreset_sampling_ratio / num_neighbors | `0.1` / `9` |
| IMAGE_SIZE / 归一化 | `256` / ImageNet mean,std（`experiment1b.preprocess_for_model`） |
| Bank | 官方 `train`，30 normal specimen 按字典序 `[::3][:10]` → 10 specimen × 12 view × 10 illum = **1,200 图** |
| Val（仅报 τ） | `[1::3][:2]` → 2 specimen × 120 = **240 图** |
| Scoring | 官方 `test` **全量**：20 normal + 50 NG specimen × 12 × 10 = **8,400 图** |

## 3. Sanity checks — **9 / 9 PASS**

| ID | 检查 | 结果 |
|---|---|---|
| S1 | 官方 metadata 与磁盘一致 | PASS — 12,000 记录全部存在 |
| **S2** | **α=0 与 plain PatchcoreModel 等价** | **PASS — `max|Δ| = 0.000e+00`（逐位相等）** |
| S3 | 计数 bank/val/scoring | PASS — 1,200 / 240 / 8,400 |
| S4 | view × illumination 笛卡尔积 | PASS — 70 specimen 全部 120/120，无缺失 |
| S5 | mask 对齐 | PASS — 抽样 10 对，尺寸全一致 |
| S6 | score 有限 | PASS — 8,400 / 8,400 finite |
| S7 | 取值域 | PASS — illuminations `01..10`，views 12 个 |
| S8 | 输出隔离 | PASS — 无既有产物被覆盖 |
| S9 | 无 B2/X6c | PASS — `alpha=(0.0, 0.0)`；未加载任何 fusion 模块 |

> 运行时曾因**本脚本自身 bug** 触发 2 次 ABORT（S3 未限定 `split=="test"`；S9 用源码文本匹配导致误报），已修复后重跑。**sanity 机制按预期拦住了错误**。

## 4. 真实 runtime 与资源（FACT）

| 阶段 | 秒 | 说明 |
|---|---:|---|
| datamodule setup | 0.24 | symlink farm 已缓存 |
| **fit（bank 1,200）** | **774.22** | **完全由 coreset 选择主导**：122,880 候选项 @ ≈167 it/s |
| val（240 图，算 τ_val） | 14.52 | τ_val = **48.1887**（mean 40.446，std 2.990） |
| **scoring（8,400 图）** | **541.72** | **15.5 img/s** |
| analysis（AUROC/d′/pixel/AUPRO） | 41.77 | |
| **总计** | **1,378.15 s = 22.97 min** | |

| 资源峰值 | 值 |
|---|---|
| **peak VRAM allocated** | **14,497 MB**（占 24,126 MB 的 60%） |
| **peak VRAM reserved** | **18,258 MB**（76%） |
| **peak RAM (RSS)** | **34,685 MB** |
| GPU 利用率（fit 期间） | 100% |

**⚠️ 与 micro-benchmark 预估的重大偏差（必须记录）**：micro-benchmark 以 200 图 bank fit 用时 24.9 s 线性外推得「1,200 图 ≈150 s」，实测为 **774 s（5.2×）**。
**原因**：PatchCore 的 coreset 选择代价随**已选集合规模**增长，因此对 bank 规模是**超线性**的，小规模测速**不可线性外推**。
→ 后果：本次 smoke 实际耗时 ~23 min（预估 8–10 min）。

## 5. 结果（FACT）

### 5.1 总体（Bird, seed0, Original；test 全量）

| 指标 | 值 |
|---|---|
| image AUROC | **0.7633** |
| d′（池化，Exp16 同款公式） | **1.2658** |
| pixel AUROC | **0.9638** |
| **AUPRO** | **NaN — FAILED**（根因见 §6） |
| n_good / n_defect | 2,400 / 6,000 |

### 5.2 逐 illumination（`illumination_metrics.csv`）

| illum | image AUROC | d′ | good_mean | defect_mean |
|---|---:|---:|---:|---:|
| 01 | 0.7658 | 1.2815 | 39.092 | 46.737 |
| 02 | 0.7681 | 1.2973 | 38.541 | 46.653 |
| **03** | **0.8260** | **1.4361** | 37.478 | 46.777 |
| 04 | 0.7615 | 1.2825 | 39.542 | 47.543 |
| 05 | 0.7746 | 1.2939 | 40.503 | 48.508 |
| 06 | 0.7443 | 1.2342 | 40.039 | 47.712 |
| 07 | 0.7659 | 1.2382 | 40.023 | 47.762 |
| 08 | 0.7463 | 1.2200 | 39.700 | 47.239 |
| 09 | 0.7411 | 1.2353 | 40.280 | 48.042 |
| **10** | **0.7327** | 1.2240 | 40.354 | 48.265 |

**汇总**：AUROC mean **0.7626**、sd **0.0262**、range **0.0933**（0.7327–0.8260）；d′ mean 1.2743、range 0.2160。
**排序（AUROC 降序）**：`03 > 05 > 02 > 07 > 01 > 04 > 08 > 06 > 09 > 10`

### 5.3 逐 view（`view_metrics.csv`）

| view | AUROC | d′ | good_mean | defect_mean |
|---|---:|---:|---:|---:|
| 000 | 0.7702 | 1.3370 | 39.766 | 47.006 |
| 030 | 0.7662 | 1.2497 | 40.449 | 46.273 |
| 060 | 0.8221 | 1.4688 | 39.516 | 45.440 |
| **090** | **0.8303** | **1.4757** | 38.892 | 46.218 |
| 120 | 0.7677 | 1.2068 | 39.597 | 45.067 |
| 150 | 0.7216 | 1.1290 | 39.825 | 45.950 |
| **180** | **0.7157** | 1.2280 | 39.594 | 47.810 |
| 210 | 0.7528 | 1.3304 | 39.652 | 48.924 |
| 240 | 0.7641 | 1.3921 | 39.419 | 50.227 |
| 270 | 0.7537 | 1.3760 | 38.798 | 49.996 |
| 300 | 0.7355 | 1.3379 | 39.111 | 49.227 |
| 330 | 0.7556 | 1.3005 | 40.042 | 48.146 |

**汇总**：AUROC mean **0.7630**、sd **0.0344**、range **0.1145**（0.7157–0.8303）；d′ mean 1.3193、range 0.3467。

### 5.4 跨 illumination 的 score 离散度（固定 specimen × view，10 个光照）

`dispersion_pairs.csv`（每个 (specimen, view) 一条，`score_std` = 10 个光照下 score 的样本标准差）

| 类别 | 配对数 | mean std | median | p90 | max |
|---|---:|---:|---:|---:|---:|
| Good（normal） | 240 | **1.9484** | 1.7993 | 2.9815 | 5.1520 |
| NG（defect） | 600 | **1.7770** | 1.6912 | 2.6563 | 4.7740 |

### 5.5 原始分数水平

- Good（test normals）score 约 37.5–40.5；defect 约 45.1–50.2（跨 view 的 defect_mean range 5.16）
- τ_val = 48.19 落在 defect 分布内部 ⇒ 用 val-max 作阈值时 FPR 会很高（**仅记录，不作判定**）

---

## 6. 异常情况（必须如实记录）

### 6.1 AUPRO = NaN（根因**已确认**，非推测）

日志：`[warn] aupro failed: tuple index out of range`

**实验验证的根因（FACT）**：

1. Bird 的 NG 图 **6,000 张中有 1,450 张没有 mask**（`image_anomaly=0`，即该视角看不到缺陷）；有 mask 的 4,550 张 = **75.8%**，与论文"平均约 75% 异常图可检出"一致。
2. 本 runner 对无 mask 的 NG 行传入 `np.zeros((1,1))` 作为占位。
3. `experiment5a_h_runner.resize_mask` 内 `np.asarray(m).squeeze()` 会把 `(1,1)` 压成 **0 维**，随后 `m.shape[:2]` 抛 `IndexError: tuple index out of range`（已验证复现）。
4. **冻结 helper 从未遇到这种情况**：MVTec 的每张 test defect 图都有 mask。

⇒ 这是 **reused-helper 与 M²AD "detectable 子集" 的不兼容**，**不是**执行链断裂。`pixel_auroc_from_maps` 因为自己 resize 而正好未受影响（把无 mask 行当作全负像素）。

**对未来的影响（必须带进 E4-A 协议）**：M²AD 需要显式的「无 mask NG」处理策略（跳过 / 计为全负 / 仅用 detectable 子集），且 `resize_mask` 需先 reshape 到 2D。**本轮未修改任何冻结 helper。**

### 6.2 micro-benchmark 外推失败

见 §4 —— 200 图 fit 的线性外推低估 5.2×。**结论：GPU 预算只能用目标 bank 规模实测，不能用小规模测速外推。**

### 6.3 工程适配（均**不改变方法**）

| ID | 适配 | 原因 | 影响科学含义 |
|---|---|---|---|
| E1 | `Folder` 用 symlink farm 适配显式图像列表 | anomalib 2.6.2 `Folder` 对每个条目 glob → 传文件列表报 `Found 0 DirType.NORMAL images in <file>` | 否 |
| **E2** | **bank 由 3,600 降为 1,200 图** | PatchCore 的 `embedding_store` 无界 + `torch.vstack`；实测 **12.485 MB / bank 图** ⇒ 3,600 图峰值 ≈45 GB，24 GB 必 OOM | **是** —— bank 覆盖度小于 MVTec 全量 train。**E4-A 必须显式规定 bank 规模；本 smoke 不构成对该选择的验证** |
| E3 | `Engine(num_sanity_val_steps=0, limit_val_batches=0)`；τ_val 自行计算 | `val_split_mode="none"` 时 Lightning 仍请求 `val_dataloader` → `AttributeError: 'Folder' object has no attribute 'val_data'` | 否 |
| E4 | `val_split_mode="same_as_test"` + 240 张 val 图 | 保证 `val_dataloader` 存在 | 否 |

**未修改**：α、权重、阈值、coreset ratio、num_neighbors、backbone、layers、IMAGE_SIZE、归一化、任何 fusion 规则。

---

## 7. 分析（INTERPRETATION — 明确区分于 §5 的 FACT）

1. **执行链完整可用**：loader → fit → coreset → inference → score → GT/mask 对齐 → metric → 落盘 全链路在 M²AD 上跑通，9/9 sanity PASS，α=0 等价性**逐位相等**。
2. **illumination 与 view 都显著调制表现，量级相近**：AUROC 极差分别为 **0.0933**（illumination）与 **0.1145**（view）。
   ⚠️ **未做任何显著性检验**（smoke 范围）。参考：单 view 单元 n_good=200 / n_defect=500，AUROC 的粗略标准误 ~0.02，因此 12 个 view 上 0.1145 的极差看似超出噪声，但**这不是统计结论**。
3. **illumination 与 view 的最优/最差不相同**：illum 03 最好 / 10 最差；view 090 最好 / 180 最差。⇒ 两者不是同一个混淆因素的不同投影。
4. **Good 的跨光照 score 离散度 ≥ NG**（1.948 vs 1.777）：说明光照同时移动 normal 与 defect 分数，并非只影响一侧。**在 E4-B 的指标设计里这一点必须考虑。**
5. **原始 AUROC 仅 0.76**：Bird 缺陷极小（E4-D0 实测 mask 前景占比 0.005%–0.46%），与论文报告的量级相符（M2AD-Invariant 上 SOTA 亦大幅退化）。**这不构成对 X6c 的任何评价。**
6. **τ_val=48.19** 高于绝大多数 clean-good 分数，落在 defect 分布内部 ⇒ 「用 train 的 val-max 当阈值」在 M²AD 上不适用；**E4-A 的 FPR 类指标需要重新考虑口径**（属协议问题，非方法问题）。

## 8. 假设（HYPOTHESIS — 尚未验证）

- H1：view 差异部分来自"缺陷在该视角的可见性"（`detectable`），而非成像质量本身。**未验证**（本轮未按 detectable 分层）。
- H2：illumination 03 之所以最好，可能与其光照方向更贴近 bank 的成像条件。**未验证**（bank 用了全部 10 个光照，故该解释不直接成立）。
- H3：把 bank 从 1,200 扩到全量 3,600 会提高 AUROC。**完全未验证**，且当前显存不可行。

## 9. 结论与出口条件

| 出口条件（`docs/E4_D1_PROTOCOL.md` §7） | 结果 |
|---|---|
| 完整执行链跑通 | **PASS** |
| 真实 runtime / VRAM / RAM / IO 已实测 | **PASS** |
| 10 illumination + 12 view 原始表现已导出 | **PASS** |
| sanity checks | **9/9 PASS**（运行时曾 2 次 ABORT，均因本脚本 bug 且已修复） |
| 未运行 B2 / X6c / 额外 seed / 额外 category | **PASS** |
| 未修改方法或阈值 | **PASS** |

> ### `VERDICT = PIPELINE FEASIBILITY PASS`
> 本实验**只**评价 pipeline feasibility。**不输出** X6c 成功/失败、hypothesis supported/rejected。

## 10. 对论文证据链的贡献

- 完成/推进的格子：**E4 的「可执行性」门** —— 从「数据结构已核实（E4-D0）」推进到
  **「冻结 Original 路径在 M²AD 上可端到端执行，且 runtime/显存/内存成本已实测」**。
- **不产生任何方法证据**，不改变 Exp16 的任何数字。
- **下一格**：由人工决定 **E4-D2**。

## 11. 成本外推（FACT-based，粗估；用于人工决策）

基于 Bird 实测（fit 774 s @1,200 bank；scoring 15.5 img/s）：

| 情形 | scoring 图数/方法 | 估算 |
|---|---:|---|
| 完整冻结协议（`clean_good` 2,400 + `clean_defect` 6,000 + 4×shift 9,600）= 18,000 图 | 18,000 | ≈19.4 min/方法 |
| 每 (category, seed)：Original + Adaptive B2 = 2 fits + 2 scannings | — | **≈65 min**（2×774 s fit + 2×19.4 min score） |
| E4-A Mini（3 cat × 3 seeds = 9 对） | — | **≈9.7 h** |
| E4-A Full（10 cat × 3 seeds = 30 对） | — | **≈32.5 h** |

**两个可显著降本的方向（均需新预注册，不得就地改协议）**：
① 固定 view（10 illumination）⇒ scoring 降至约 1/12；
② bank 规模与 coreset 代价必须联合标定（coreset 超线性，是 fit 的主导项）。

**显存**：bank 1,200 → peak 14.5 GB alloc / 18.3 GB reserved。**E4-A 沿用 1 worker**；3 workers 需重新标定 bank 规模。

## 12. 本轮未做

未运行 Adaptive B2 / X6c；未扩展 seed / category；未修改任何方法、权重、阈值；
未修改任何冻结 helper（`resize_mask` / `compute_aupro` 保持原样）；未启动 E4-A / E4-B / E4 Mini。
