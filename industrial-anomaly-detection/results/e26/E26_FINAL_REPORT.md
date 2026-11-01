# E26 Pre-submission P0 Audit

**日期**：2026-11-01
**范围**：E26-A（重复审计）· E26-B（checkpoint 审计）· E26-C（复现性信息提取）· E26-D（数值一致性核验）
**性质**：**只读审计组**。不训练、不重跑 cell、不改冻结代码、不改任何历史结果；仅 E26-D 的脚本可重复执行且只写本组目录。
**父提交**：`417d883`（branch `research/industrial-anomaly`）；本组文件在该提交之上一次性纳入版本控制。
**本文件是 E26 的唯一总入口**；四份子报告见 §8。

---

## 1. Overall Verdict

```text
E26 STATUS: CLOSED
VERDICT: READY_FOR_MANUSCRIPT_REVISION
EXPERIMENT REOPEN: NO
NEXT STAGE: MANUSCRIPT CLEANUP AND SUBMISSION
```

前提逐条成立：

| 前提 | 状态 | 证据 |
|---|---|---|
| E26-A 无实质近重复风险 | ✅ | 145 对同名文件：SHA-256 相等 0 对、像素相等 0 对、`VERY_LIKELY_NEAR_DUPLICATE` 0 对；SSIM 最大 0.8139、pHash Hamming 最小 8 |
| E26-B checkpoint 实现与冻结协议无重大冲突 | ✅ PASS_A | 实现 = 停止时刻权重；冻结协议与论文面向材料对该点**沉默**，无相反规定（附写作义务，见 §3） |
| E26-C 复现信息已提取 | ✅ | `reproducibility_manifest.json`（§4），关键字段已与实时环境/冻结配置/权重哈希核对一致 |
| E26-D 数值一致性 1249/1249 PASS | ✅ | 最大数值差 5.4488e-06；7 项 1 ulp 全部机械归因（§5） |

**含义**：论文主表与核心辅助表（E21 多视角、E22 十类主表、E25 RobustAD 多 seed、E26-AnomalyDINO 补充基线）中已发布的数值，均可由仓库内归档的逐图分数机械复算。进入写作修订阶段，**不再开启实验**。

---

## 2. E26-A · Duplicate Audit（RobustAD MetalParts train/test0）

| 项 | 值 |
|---|---:|
| 匹配判据 | 两侧 `metadata.jsonl` 的 `file_name` 字符串完全相等（沿用 E25 已记录判据，未重新设计） |
| train 条目 / test0 条目 | 231 / 171 |
| 检查 pair 数 | **145**（normal 104 + anomaly 41；两侧标签一致 145/145；磁盘缺失 0；尺寸一致 145/145） |
| SHA-256 完全重复 | **0 / 145** |
| 字节数相同 | 0 / 145 |
| 像素级完全相等 | **0 / 145** |
| 感知哈希 | pHash Hamming min **8**（median 20，max 36）；dHash Hamming min **2**（仅 1 对 `MP_012`，其 pHash=10、SSIM=0.740） |
| SSIM（原生分辨率，145/145 未触发 resize） | min 0.6601 / median 0.7515 / **max 0.8139** |
| 辅助分类 | EXACT 0 · VERY_LIKELY_NEAR_DUPLICATE **0** · SIMILAR_SCENE_OR_OBJECT 17 · NO_OBVIOUS 128 |
| LPIPS | **SKIPPED**（冻结环境未装 `lpips`，按环境隔离原则不引入新依赖） |
| Top-20 目视复核 | 两张 contact sheet（`top01_10` / `top11_20`，约 11 MB，**仅本地保存、不入库**）：全部为同型号金属板的两次不同拍摄（高光、位姿、可见缺陷不同），未见同帧/重编码/裁切/轻微几何扰动级重复 |

**对论文的影响**

- 不影响 E25 的任何聚合数字：MetalParts test0 仍按原协议 `valid=0` 排除，本审计未改变该判定、未触发重跑。
- 可加强的表述（补充材料）：145 对同名文件已完成字节级 + 感知哈希 + SSIM 审计，0 对完全重复、0 对近重复；该 split 仍按原协议保守排除。
- **不得过度声称**：本审计只回答"图像是否重复"，**不能**回答"采集对象是否独立"。同一物理工件在不同光照/位姿下的两次拍摄在本套指标下呈中等相似度（SSIM 0.66–0.81），与"近邻帧"无法在概念上完全区分；文件名重合本身仍是数据集组装层面的关联信号。
- 可引用结论：**未发现足以表明外部验证结果由明显图像近重复驱动的证据。**

---

## 3. E26-B · Checkpoint Audit

三方对照（静态代码审计 + 步数统计，0 GPU）：

| 维度 | 内容 | 是否明确 |
|---|---|---|
| **Actual implementation** | 使用 **early-stop 触发时刻的当前权重**；`best_val` 仅用于"何时停"与报告，**从不**影响选权重（`e13b_screen.train_r1` 内无 `deepcopy` / `state_dict` 快照 / `torch.save` / `load_state_dict`） | 明确（代码 44–90 行） |
| **Frozen protocol** | 只规定 ≤300 step / patience 30 / 每 5 步在 60 张留出正常图上评估 val consistency，**未规定**停止后是否回到最佳点 | 沉默 |
| **Manuscript description** | 仓库内论文面向材料同样只写 "patience 30（每 5 step 验证）"，未出现 best checkpoint / restore / model selection | 沉默 |

**裁决：PASS_A**（实现做了一件事，协议与论文材料都没有相反规定或相反宣称；不构成协议违背，也不属于"论文误写"）。

**附带写作义务（不是重跑义务）**：因为 100% 的 cell 都实际触发早停，正文必须写明"不恢复最佳验证权重"：

> Early stopping monitors a validation consistency loss on 60 held-out normal images (evaluated every 5 steps, patience 30); **the parameters at the stopping iteration are used for evaluation — no best-validation checkpoint is restored.**

**量化**（2026-11-01 复核一致）：

| 结果集 | cell 数 | `steps_used` 范围 | 中位数 | 跑满 300 步 | 实际早停 |
|---|---:|---|---:|---:|---:|
| E22 | 30 | 44–124 | 59 | 0 | 30 / 30 |
| E25 | 39 | 49–164 | 74 | 0 | 39 / 39 |
| E24 | 51 | 44–219 | 59 | 0 | 51 / 51 |

附带发现（不影响任何数字）：`report["steps_used"]` 在早停时少记 1 步（报告层 off-by-one）；`best_val` 是"死变量"，不能支撑 "model selection on validation" 的说法。

---

## 4. E26-C · Reproducibility Audit

已确认的关键配置（完整机器可读版本：`results/e26/reproducibility/reproducibility_manifest.json`）：

| 项 | 值 |
|---|---|
| **Backbone** | `dinov2_vitb14`（patch 14），frozen，`get_intermediate_layers(n=1, norm=True)`；权重 `/root/.cache/torch/hub/checkpoints/dinov2_vitb14_pretrain.pth`，SHA-256 `0b8b82f8…fad8c73`，346,378,731 B（本次已重算核对一致） |
| 输入 / 特征 | 短边 448 BICUBIC→448×448，1024 patch token，768 维，ImageNet 归一化 |
| **Detector** | 1-NN-max：特征 L2 归一化后余弦距离，图像分数 = 1024 patch 距离的 **max**；纯 torch matmul（无 FAISS/sklearn） |
| **Bank** | M2AD：300 张正常参考图/类/seed；RobustAD：180 张 source-train 正常图 |
| **Optimizer** | AdamW，lr = 1e-3；betas (0.9, 0.999) / eps 1e-8 / weight_decay 0.01 均为 **torch 默认**（代码只显式传 lr） |
| **Adapter** | `z' = z + Wz`，W: 768×768 **zero-init**，无 bias，589,824 参数 |
| **Loss** | `L = L_cons + BETA·L_preserve`，BETA = 0.1；`L_cons`/`L_preserve` 均为 1 − 平均 patch 余弦 |
| **Augmentation** | brightness / contrast / gamma × {0.85, 1.15} → **6 个固定变体张量**（预先算好、跨 step 复用）；每 step 每图均匀抽一个变体索引；扰动在 ImageNet 归一化**之前** |
| **训练预算** | ≤300 step，每 5 步评估，patience 30（评估量 = val consistency 项，非总损失）；train 240 / val 60（M2AD），120 / 60（RobustAD） |
| **Seeds** | 0 / 1 / 2（seed 只控制适配器初始化与训练 RNG；M2AD bank 随 seed 重抽，RobustAD bank 跨 seed 固定） |
| **Checkpoint policy** | 无 best-restore；评分用停止时刻权重（见 §3） |
| **Commit / config / 结果路径** | 审计 HEAD `417d883`；冻结配置 `results/e17/final_method.json`；主结果 `results/e22/main_table_10cat_3seed.csv` + `results/e25/multiseed_analysis.json` + `results/paper_ready/paper_main_table.csv` |
| 环境 | Linux / Python 3.12.3 / PyTorch 2.8.0+cu128 / torchvision 0.23.0+cu128 / CUDA 12.8 / cuDNN 91002 / numpy 2.3.2 / RTX 3090 24 GB（conda base） |

**本次只读复核**：Python / PyTorch / numpy 版本一致；骨干权重 SHA-256 与字节数一致；`final_method.json` 的 BETA 0.1 / LR 1e-3 / AdamW / 300 / 30 / 5 / 8 / 240 / 60 / 589,824 参数与 manifest 逐项一致。

---

## 5. E26-D · Numerical Consistency

| 指标 | 值 |
|---|---|
| 检查总数 / 通过 / 失败 | **1249 / 1249 / 0** |
| 判定 | **PASS** |
| 最大数值差（非类别项） | **5.4488e-06** |
| 需要整个 ulp 的项 | 7（全部机械归因） |
| 运行成本 | 纯 CPU，1.7 s，GPU 0 |

按类别：`full_precision` **343** · `published_6dp` **646** · `published_5dp` **152** · `categorical_exact` **105** · `cross_source_1e-5` **3**。

覆盖（14 个发布位置，单来源单归属）：`e22/cells`（120）· `e22/main_table_10cat_3seed.csv`（80）· `paper_ready/paper_main_table.csv`（80）· `e25/cells`（117）· `e25/multiseed_analysis.json`（58）· `e25/multiseed_summary.csv`（78 + 39）· `e25/e25_summary.csv`（78 + 39）· `e21/e21_partial.json`（48）· `paper_ready/paper_multiview_table.csv`（72 + 24）· `e26_anomalydino/baseline_30cells.csv`（180）· `baseline_category_summary.csv`（200）· `baseline_macro.csv`（30 + 3）· 跨工件 E21↔E22（3）。

**7 项 1 ulp 的机械解释**（差异量级 5.2–5.5e-06，均不构成数值分歧）：

| 归因 | 数量 | 项 |
|---|---:|---|
| 打印值 = 复算值**截断**（向零） | 4 | `paper_ready-Dice-dAUROC_std` 0.00077 · `Motor-R1_AP_mean` 0.918 · `Ring-F0_AUROC_std` 0.011 · `Teapot-dAP_mean` 0.01069 |
| **双舍入**（先 6 位、后打印 5 位） | 3 | `paper_multiview_table`：`Car-330-R1_AP` 0.78368 · `Cube-120-F0_AUROC` 0.8541 · `Cube-330-R1_AP` 0.94076 |

**审计框架灵敏度（非恒真证据）**：开发期两版确实报 FAIL——① `is_source` 语义取错 → 6 失败（PCB 域均值被算成 +0.013948，真实值 +0.018352）；② `e25_summary.csv` 误按 3-seed 比对 → 26 失败（由此反推出 Fact 1）。反例目录 `run1_prefix_bug/` 仅本地保留、不入库，其信息已在 `results_consistency/REPORT.md` 文字记录。

---

## 6. 三条口径事实（永久记录，引用前必看）

### Fact 1 — E25 summary 的 seed 语义

`results/e25/e25_summary.csv` **不是三种子汇总表**。其中 `auroc_r1` / `ap_r1` / `d_auroc` / `d_ap` 对应 **seed-0**（13/13 split 与 seed-0 复算一致，0/13 与 3-seed 均值一致；其 `d_auroc_ci_*` 也是 seed-0 的配对 bootstrap CI）。

真正的三种子结果来源是 `results/e25/multiseed_summary.csv`（`*_mean` / `*_std`）。

论文当前引用的是三种子结果（已核对一致）。因此：

> **不得将 `e25_summary.csv` 当作论文最终三种子结果表使用。**

### Fact 2 — 发布精度不是"Python 默认 round"

部分 `paper_ready` 表存在**截断**与**6 位→5 位双舍入**。因此未来复现核验**不得**简单要求"按 Python 默认 `round` 后字符串完全一致"，而应依据发布脚本的真实机械规则核验（E26-D 用 `rounds_to_published` / `truncates_to_published` / `double_rounds_to_published` 三个字段显式判定）。

这些差异（≤5.5e-06）**不代表实验数值冲突**。

### Fact 3 — E22 与 E25 的 seed 方差含义不同

- **E25**：`bank_hash` 跨 seed 相同（PCB `08821b36…` / MetalParts `b4b51729…`）→ F0 baseline 跨 seed **逐位相同**（差 0.0）→ E25 的 seed std 主要反映**适配器训练随机性**（`d_auroc_std` 范围 0.000217–0.012405）。
- **E22**：每个 seed **重新抽 300-bank** → F0 也随 seed 变化（跨 seed 最大差 0.028464 AUROC / 0.028881 AP）→ 其 seed std 同时含 **bank 抽样方差**。

> 二者 seed std 的统计含义不同，**不可直接横向解释或并列比较**。

**聚合外部结果时必须使用**：

```text
valid == 1 AND is_source == 0
```

`is_source` 与 `valid` 是**正交字段**。例如 `MetalParts:test0` 的 `condition_group = source_reference_INVALID`：`is_source = 1` 但 `valid = 0`，**不能进入目标域均值**（E25 目标集 = 11 个 target split：PCB 5 + MetalParts 6）。

---

## 7. AnomalyDINO 的处理裁决

`results/e26_anomalydino/` 中已核验的数值：macro AUROC **0.803565**、R1 − AnomalyDINO **+0.042844**、AnomalyDINO − F0 **−0.035247**（dev_3cat 0.743570 / ext_7cat 0.829278）。这些值**未传播**到 `results/paper_ready/` 或 `PROJECT_STATE.md`。

本轮**不**自动写入论文正文。

**裁决：`KEEP_AS_SUPPLEMENTARY_SYSTEM_REFERENCE`**

理由：当前论文主线不是 AnomalyDINO 对比；增加该结果可能改变正文叙事与比较边界。

固定表述（写入时照抄）：

> AnomalyDINO 补充基线已经完成数值复算与一致性核验，但当前未被纳入 paper_ready 主叙事；这是有意的证据分层，而非遗漏导致的数值问题。

除非后续论文明确需要该比较，否则不再传播。

---

## 8. 子报告与产物

| 部分 | 报告 | 关键数据文件 |
|---|---|---|
| E26-A | `results/e26/robustad_duplicate_audit/REPORT.md` | `pair_manifest.csv`（145 行）· `all_pairs.csv`（145 行 × 34 列）· `top20.csv` · `audit_summary.json`（两张 contact sheet 仅本地） |
| E26-B | `results/e26/checkpoint_audit/REPORT.md` | `step_stats.json` |
| E26-C | 无独立 md；字段表见 §4 | `results/e26/reproducibility/reproducibility_manifest.json` |
| E26-D | `results/e26/results_consistency/REPORT.md` | `consistency_checks.csv`（1249 行）· `consistency_report.json` |
| 脚本 | `scripts/e26_a_duplicate_audit.py` · `e26_b_checkpoint_audit.py` · `e26_c_reproducibility.py` · `e26_d_results_consistency.py` | — |

**版本控制策略**：只入库脚本、报告与小体积 CSV/JSON。排除两张 contact sheet（约 11 MB，仅供人工目视）与 `results_consistency/run1_prefix_bug/`（开发期反例）。数据集、权重、bank、checkpoint 一律不入库。

---

## 9. 复现 / 复核入口

```bash
cd /root/autodl-tmp/research-experiments/industrial-anomaly-detection

# E26-D：可重复执行，只写 results/e26/results_consistency/，期望 1249/1249 PASS 且退出码 0
python scripts/e26_d_results_consistency.py

# E26-A：只读图像审计（约 2 分 15 秒，纯 CPU）；会重写本目录产物
python scripts/e26_a_duplicate_audit.py

# E26-B：静态代码审计 + 步数统计；会重写 step_stats.json
python scripts/e26_b_checkpoint_audit.py

# E26-C：重写 reproducibility_manifest.json（会把 working_tree_dirty 置为该次运行时的真实状态）
python scripts/e26_c_reproducibility.py
```

**2026-11-01 提交前复核结果**（只读方式，未重写 A/B/C 产物）：A 的 CSV 明细与 summary JSON 计数一致（145 / 0 / 0 / 17 / 128，SSIM max 0.8139，pHash min 8）；B 的步数统计与报告逐项一致且最终路径无 checkpoint 恢复符号；C 的环境版本、权重 SHA-256、冻结超参与 manifest 一致；D 重跑 1249/1249 PASS、退出码 0。

---

## 10. STOP — E26 不再扩展

E26 至此 **CLOSED**。不再进行：

- E24 消融表重新审计；
- bootstrap CI 全量复算；
- 新 baseline；新 seed；新数据集；unseen illumination；CAP / PG-PSR；新 backbone；
- 再次调整 E26-D 的覆盖范围。

已足够支撑的论断：

> 论文主表及核心辅助表中的发布数值，能够由仓库归档结果机械复算。

审计**不**追求"仓库里每一个历史 CSV 全覆盖"。

```text
E26 STATUS: CLOSED
VERDICT: READY_FOR_MANUSCRIPT_REVISION
EXPERIMENT REOPEN: NO
NEXT STAGE: MANUSCRIPT CLEANUP AND SUBMISSION
```
