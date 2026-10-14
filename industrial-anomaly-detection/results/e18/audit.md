# E18-P0 — Final Baseline Benchmark · READ-ONLY AUDIT

日期：2026-10-14 · 阶段：`METHOD SEARCH CLOSED → FINAL BENCHMARK AND PAPER CONVERGENCE`
**本轮未启动任何 GPU 训练。**

## 1. 环境与 Git

| 项 | 值 |
|---|---|
| branch | `research/industrial-anomaly` |
| HEAD | `28d02b3`（`exp: freeze final illumination adaptation in E17`） |
| ahead/behind | **0 / 0**（E16 `52b2b74`、E17 `28d02b3` 均已 push） |
| git status | 6 项历史遗留（`monitor/exp_monitor.py`、`results/e8/status.json`、`monitor/events.jsonl`、`scripts/_lockprobe.txt`、`scripts/_probe2.py`、`scripts/e11_p4_runner.py`）—— 与 E18 无关 |
| GPU | RTX 3090，util 0%，1 MiB，0 进程（空闲） |
| disk free | 16 GB |

**最终方法唯一权威配置**：`results/e17/final_method.json`（R1-ILA，BETA=0.1）。E18 不读取任何手工复制的参数。

## 2. 统一评价协议（E11+ 线，即主表要用的协议）

| 项 | 定义 |
|---|---|
| 数据集 | M²AD，view **120**，seed **0** |
| 测试集 | view120 全部 **700** 图；**official 口径** = `image_anomaly` 标签 + `detectable != False` 过滤 |
| 过滤后规模 | Bird **653**（320 ok / 333 ng，剔除 47）·Car **656**（300 / 356，剔除 44）·Cube **678**（260 / 418，剔除 22） |
| bank | 300 normal，`E8.bank_for(pool, 0)`，`random.Random(20261009)`，sha256(seed0) = `33381cd3004bfe68` |
| 指标 | image-level **AUROC** + **AP**（sklearn `roc_auc_score` / `average_precision_score`） |
| 实现 | `scripts/e11_p2_runner.py::official_metrics`（E11-P2 起，E13/E14/E15/E16 全部沿用） |

## 3. ⚠️ 发现 A：view120 存在两套标签协议（关键）

M²AD 的 view120 元数据同时带 `image_anomaly` 与 `object_anomaly` 两个标签字段，二者在 view120 上**不一致**：

| category | view120 n | `image_anomaly` | `object_anomaly == 0` | 不一致张数 |
|---|---|---|---|---|
| Bird | 700 | 320 good / 380 ng | **200** good / 453 ng | 120 |
| Car | 700 | 300 / 400 | 200 / 456 | 100 |
| Cube | 700 | 260 / 440 | 200 / 478 | 60 |

- **E7 线（PatchCore/CLAHE 等，git `af56dd3`）用 `object_anomaly`**：E7-1 文档写明 "700 = 200 Good + 500 NG"。
- **E11 线（F0 / AnomalyDINO / SoftPatch / EfficientAD / R1-ILA）用 `image_anomaly` + `detectable` 过滤**。

→ **两套标签下的 AUROC 不可直接比较**。这正是本审计最有价值的发现（见 §4）。

## 4. ⚠️ 发现 B：PatchCore 的官方口径数字必须重算（已完成，0 GPU）

E7-1 **P00 = Original PatchCore**（`wide_resnet50_2` layer2+layer3，input 256，coreset 0.1，k=9，max-pool），
历史报告 image AUROC **0.78089** / AUPR 0.91723 —— 但那是 **`object_anomaly`** 口径。

**可重算性已逐项证明**（`scripts/e18_recompute_official.py`，全部断言通过）：
1. **bank 逐位一致**：E7-1 的 "1200 池的 25% 子集 + `BANK_SUBSET_SEED=20261009`" 与
   `E8.bank_for(pool, 0)` **完全相同**（sha256 = `33381cd3004bfe68`，断言通过）。
2. **测试集一致**：均为 view120 全部 700 图。
3. **(specimen, illumination) → 元数据 1:1**：700/700 命中，**0 未匹配**，无重复键。
4. **分数与标签无关**：标签只进入指标计算，不进入分数。
5. **身份校验**：用旧标签重算，**逐位复现历史 0.78089 / 0.9172272557040083** → 证明映射的是同一次运行。

**重算结果（official 口径，全精度）**：

| method | Bird AUROC | Bird AP | n |
|---|---|---|---|
| PatchCore（official scoring） | **0.6931775525525525** | **0.7230433828349869** | 653（320/333，剔除 47） |

→ 历史 0.78089 是**标签口径造成的虚高**（+0.0877）；official 口径下 PatchCore 在 Bird 上为 **0.69318**。
产物：`results/e18/recomputed/patchcore_bird_official.json`。

## 5. 各基线逐项审计

### 5.1 PatchCore — `RECOMPUTE`（Bird 已完成重算，Car/Cube 缺）
- 官方实现：`scripts/e7_pipelines.py::run_patchcore`（frozen E4-D1/E5-1 配置）。
- 唯一有 official 口径数字的是 **Bird（0.69318）**；**Car/Cube 从未以 official 口径跑过**。
- 另有一条**同协议、同 bank、同打分**的替代行：**E11-P2 的 "resnet (PatchCore emb) + 1-NN-max"**
  （Bird 0.68454 / Car 0.62247 / Cube 0.60857，official 口径全精度）→ 可作为
  **"PatchCore 表示 + 本文统一 scoring"** 的 representation-only 对照行，**REUSE**。

### 5.2 AnomalyDINO（F1）— `RECOMPUTE`（缺 Cube）
- 实现：`third_party/AnomalyDINO` + `e7_pipelines.run_dinov2`；官方 `mean_top1p`（DINOv2-448，L2，kNN k=1）。
- Bird 0.70214 / 0.72234；Car 0.75684 / 0.78277（official 口径，全精度）。
- **Cube 未跑**：E14 预注册 stageB 规则 "only for Stage A advanced families"，F1 在 Bird 即跌破 −0.015 → STOP，未进 Cube。

### 5.3 SoftPatch（F2）— `RECOMPUTE`（缺 Cube）
- 实现：`third_party/SoftPatch` + `e7_pipelines.run_softpatch`（官方 `WeightedGreedyCoresetSampler` soft weighting）。
- Bird 0.64396 / 0.57510；Car 0.50219 / 0.55541（official 口径，全精度）。
- **Cube 未跑**，原因同 §5.2。

### 5.4 EfficientAD — `EARLY_STOP`
- 实现：`third_party/EfficientAD`，PDN-**small**，input 256，70000 steps，`pretrain_penalty=False`（无 ImageNet）。
- **Bird 0.61206 / 0.62196**（official 口径，n=653，wall 1750.6 s）。
- **Car/Cube 未跑**：E15 Stage-A 规则 `advance if dAUROC >= −0.010`，Bird dAUROC = **−0.13519** → STOP。
- **不得补推测值，不得称其为完整三类别 benchmark。**

### 5.5 F0（DINOv2-B + 1-NN-max）— `REUSE` ✅
- Bird 0.74725 / 0.78863 · Car 0.78552 / 0.84207 · Cube 0.85410 / 0.91686（official 口径）。
- 3 个 seed 在 E16 已有完整 9 cell（含 per-image 分数）。

### 5.6 Ours = R1-ILA — `REUSE` ✅
- seed0：Bird 0.75786 / 0.79911 · Car 0.80511 / 0.85502 · Cube 0.87010 / 0.92856。
- 3-seed mean ΔAUROC（vs F0）：**Bird +0.02586 / Car +0.01123 / Cube +0.00265**（E16，9/9 cell）。
- 配置冻结于 `results/e17/final_method.json`。

## 6. 全精度可获取性（逐项）

| 方法 | 权威记录 | 精度 |
|---|---|---|
| PatchCore (official) | `e18/recomputed/patchcore_bird_official.json`（本次重算） | 全精度 ✅ |
| PatchCore (unified) | `e11/p2_raw_results.json` | 全精度 ✅ |
| AnomalyDINO / SoftPatch | `e14/e14_partial.json` | 全精度 ✅ |
| EfficientAD | `e15/e15_partial.json` | 全精度 ✅ |
| F0 / Ours | `e16/e16_partial.json`（含 per-image 分数 + bootstrap CI） | 全精度 ✅ |
| E7-1 P00 原始 | `e7_1/raw/P00.json`（5 位小数） + `P00_per_image.csv`（全精度分数） | 展示低精度，分数全精度 ✅ |

**未使用任何低精度展示 CSV 作为权威记录。**

## 7. 缺失单元与成本

| 缺失单元 | 原因 | 是否可补 | ETA |
|---|---|---|---|
| PatchCore official · Car / Cube | 从未以 official 口径运行 | ✅ 可补 | ~5 min |
| AnomalyDINO · Cube | E14 预注册 Stage-B 门未达 | ✅ 可补 | ~3 min |
| SoftPatch · Cube | 同上 | ✅ 可补 | ~1.5 min |
| EfficientAD · Car / Cube | E15 预注册 Stage-A STOP | ⚠️ 需单独批准 | ~58 min（29 min/类） |

**最小补跑（不含 EfficientAD）≈ 10 min GPU。**
**含 EfficientAD 完整三类别 ≈ 68 min GPU。**

## 8. 不可直接比较 / 必须标注的项

1. **E7 线全部 image 指标**（`object_anomaly` 口径）与 E11+ 线不可直接比较 —— 已对 PatchCore 完成重算修正。
2. **EfficientAD 只有 Bird** —— 不能声称三类别完整结果。
3. **AnomalyDINO / SoftPatch 缺 Cube** —— 主表将出现空单元。
4. **各方法 native scoring 不同**（PatchCore coreset+kNN / AnomalyDINO mean_top1p / SoftPatch soft-weighted kNN / EfficientAD discrepancy / F0 & Ours 1-NN-max）——
   这是**方法自身定义**，**不得**为了表面统一而篡改；主表须并列标注 native scoring，
   并另附 "统一 scoring" 的 representation-only 对照（E11-P2 PatchCore-emb 行）。
5. **单 seed（seed0）** —— 除 F0/Ours（E16 有 3 seed）外，其余基线均为 seed0。
   主表中基线 vs Ours 是**同 seed、同 bank、同测试集**的公平比较，但**未做 seed 方差**。
