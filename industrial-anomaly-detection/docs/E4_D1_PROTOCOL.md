# E4-D1 — M²AD Bird × seed0 × Original PatchCore GPU Smoke — PROTOCOL (FROZEN)

**冻结时间**：2026-10-08（**先于任何 GPU 运行**；仅由 micro-benchmark 的**速度/显存**数据定标，不含任何模型结果）
**起点 HEAD**：`14a3863`（branch `research/industrial-anomaly`）
**性质**：**pipeline smoke + 原始表征**，**不是 E4-A**。不产生方法结论。
**GPU**：1 worker（单进程），RTX 3090 24 GB

---

## 0. 目标（只做这些）

1. 打通完整执行链：loader → fit(memory bank) → embedding → coreset → inference → score extraction → GT/mask 对齐 → metric pipeline → artifact 落盘
2. 实测真实 runtime、峰值 VRAM / RAM、IO
3. 导出 **10 个 illumination** 与 **12 个 view** 的原始表现
4. Sanity checks

**明确不做**：不跑 Adaptive B2、不跑 X6c、不扩展 seed、不扩展 category、**不修改方法或阈值**。

---

## 1. FROZEN 配置

| 项 | 值 | 来源 |
|---|---|---|
| Dataset | **M²AD**，category **Bird** | E4-0C / E4-D0 |
| 官方 metadata | `data/m2ad/jsons_extract/meta_unsupervised.json` | 官方 `jsons.zip` |
| Metadata SHA256 | `bbeaa75c231c981e443c818859a701fcc0daf1769065abd91631f7e676725980`（zip） | E4-D0 |
| seed | **0** | 任务书 |
| method | **Original**：`FAlphaDualLayerPatchcore(alpha_l2=0.0, alpha_l3=0.0)` | `scripts/experiment5a_model.py`（α=0 严格 short-circuit ⇒ 与 plain PatchCore 逐位等价） |
| backbone | `wide_resnet50_2`，layers `["layer2","layer3"]` | 冻结协议 |
| coreset_sampling_ratio | `0.1` | 冻结协议 |
| num_neighbors | `9` | 冻结协议 |
| IMAGE_SIZE | **256** | `experiment1b_defensive_sensitivity.IMAGE_SIZE` |
| 归一化 | ImageNet mean/std | `experiment1b.preprocess_for_model` |
| 阈值 | **不适用**（Original-only smoke 不做 PASS/catastrophic 判定） | — |
| 并发 | **1 worker**（任务书 §13） | — |

## 2. 子集规则（结果无关，运行前记录）

### 2.1 Bank（memory bank 训练集）

官方 `train` split（Bird = 30 个 normal specimen × 120 = 3,600 图）。按 specimen id 字典序**等距抽样 10 个**：

```text
bank_specimens = sorted(train_specimens)[::3][:10]      # 索引 0,3,6,…,27 → 跨全范围
bank = 上述 10 个 specimen 的全部 12 views × 10 illuminations = 1,200 图
```

**为什么等距而不是取前 10 个**：M²AD 每类含 2 个视觉子类；取连续前缀可能整段落在同一子类里。等距抽样在**不看任何模型输出**的前提下覆盖全范围。

### 2.2 Val（仅用于报告 τ_val）

```text
val_specimens = sorted(train_specimens)[1::3][:2]        # 索引 1, 4
val = 2 个 specimen × 12 × 10 = 240 图
```

`τ_val = max(val scores)`（冻结协议同款定义；**仅作报告**，不参与任何判定）。

### 2.3 Scoring（原始表征）

官方 `test` split **全量**：20 normal specimen + 50 NG specimen，全部 **12 views × 10 illuminations** = **8,400 图**。

## 3. 输出指标

| 输出 | 定义 |
|---|---|
| **per-illumination image AUROC** | 对每个 `illumination ∈ 01..10`：以该光照下的 test normals（`object_anomaly=0`）为负类、NG（`object_anomaly=1`）为正类 |
| **per-illumination d′** | 同光照下 `d′ = (mean(defect) − mean(good)) / sqrt((std_defect² + std_good²)/2)`（Exp16 同款公式，池化、不分 defect type） |
| **per-view image AUROC / d′** | 同上，对每个 `view ∈ 000..330` 聚合全部 10 光照 |
| **cross-illumination dispersion** | 对每个 (specimen, view)：10 个光照下 score 的 `std` / `range`；并汇总其分布（normal 与 NG 分开） |
| **overall pixel AUROC + AUPRO** | test 全集（good + 有 mask 的 defect），复用 `experiment1h_runner.pixel_auroc_from_maps` 与 `experiment5a_h_runner.compute_aupro` —— 用来验证 mask 对齐链路 |
| **overall image AUROC** | test 全集 |
| **runtime / VRAM / RAM** | 分阶段计时；`torch.cuda.max_memory_allocated/reserved`；`resource.ru_maxrss` |

## 4. 工程适配（必须记录；**均不改变方法**）

| ID | 适配 | 原因（实测） | 是否影响科学含义 |
|---|---|---|---|
| **E1** | `Folder` datamodule 需要**目录**，故用确定性 **symlink farm** 适配显式图像列表 | anomalib 2.6.2 `Folder` 对每个条目做 glob → 传文件列表报 `Found 0 DirType.NORMAL images in <file>` | 否（不改像素/变换） |
| **E2** | **bank 由 3,600 图降为 1,200 图** | anomalib PatchCore 的 `embedding_store` 是**无界 list**，最后 `torch.vstack`；256 px 下 **12.485 MB / bank 图**（实测）⇒ 3,600 图峰值 ≈45 GB，**24 GB 必 OOM** | **是** —— bank 覆盖度小于 Exp16 的 MVTec 全量 train。**E4-A 的正式协议必须显式规定 bank 规模**，本 smoke 不构成对该选择的验证 |
| **E3** | `Engine(num_sanity_val_steps=0, limit_val_batches=0)`；τ_val 由 runner 用冻结的 `e1b.predict_one` 自行计算 | `val_split_mode="none"` 时 Lightning 仍请求 `val_dataloader` → `AttributeError: 'Folder' object has no attribute 'val_data'` | 否 |
| **E4** | `val_split_mode="same_as_test"` 并提供 240 张 val 图作为 `normal_test_dir` | 保证 `val_dataloader` 存在 | 否 |

**未修改**：α、权重、阈值、coreset ratio、num_neighbors、backbone、layers、IMAGE_SIZE、归一化、任何 fusion 规则。

## 5. Sanity checks（运行中执行）

| ID | 检查 | 通过条件 |
|---|---|---|
| S1 | 官方 metadata 与磁盘文件一致（Bird） | 加载记录数 == 12,000 |
| S2 | α=0 等价性 | `FAlphaDualLayerPatchcoreModel(α=0,0).generate_embedding == PatchcoreModel.generate_embedding`，`max|Δ| < 1e-6` |
| S3 | bank / val / scoring 计数 | 1,200 / 240 / 8,400 |
| S4 | view × illumination 笛卡尔积完整 | 每个 specimen 120 个 (view, illum) 全覆盖 |
| S5 | mask 对齐 | 抽样 mask 与图像同分辨率（1024×1024）且非空 |
| S6 | score 有限 | 全部 8,400 个 score 为 finite |
| S7 | illumination / view 取值域 | `{01..10}` / 12 个 view |
| S8 | 输出隔离 | 写入 `results/e4_d1_smoke/`，不触碰任何历史 result 目录 |
| S9 | 无 B2/X6c | 进程中不存在 B2/X6c 的 α 或融合调用 |

**任一 sanity FAIL ⇒ 停止并报告，不继续。**

## 6. 进度显示（任务书强制）

运行中每个进度 tick 打印：

```text
completed / total
percentage
elapsed
dynamic ETA
images/s
GPU memory allocated / reserved
```

## 7. 出口条件

- 本 smoke **只**输出 `PIPELINE FEASIBILITY`，**不得**输出 X6c 成功/失败、hypothesis supported/rejected。
- 完成后 **STOP**，等待人工决定 E4-D2。
- 不自动启动 E4-A / E4-B / E4 Mini。
