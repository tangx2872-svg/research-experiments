# E21 — M²AD Multi-View Generalization Validation（冻结协议）

**日期**：2026-10-15
**执行者**：实验执行 AI（协议由策划侧确定；执行侧不得改变研究问题、方法、BETA、训练协议、评价指标或判据）
**状态**：预注册（本文件在 GPU 运行前定稿）
**方法冻结**：R1-ILA，BETA=0.1（`results/e17/final_method.json`）；禁止重新搜索 BETA / 增加模块 / 依据 E21 结果调参

---

## 0. 论文导航

```text
📍 PAPER STAGE : Phase II — Paper Validation / Evidence Expansion（METHOD SEARCH = CLOSED）
🎯 CLAIM      : R1-ILA 的收益并非仅存在于 view120 单一拍摄视角；在多个已有拍摄视角下，
                其整体异常检测性能仍保持正向收益，并表现出一定的跨视角稳定性。
🧪 WHY E21    : 现有 A1/A2/A3 全部基于 view120 单视角；E21 用已有 4 个缓存视角
                （090/120/240/330）扩展评估，检验收益的跨视角总体正向性。
🚦 EXIT       : Round 1 (seed0) 按 §7 判据 GO → 自动补 seed1/2；HOLD → 停止回报待裁决；
                STOP → 停止、归档负结果、不救火。
```

**允许表述**（无论结果如何）：多视角结果表明，R1-ILA 的性能收益并非仅局限于原始 view120 条件，在已有多个拍摄视角下仍表现出总体正向趋势。
**禁止表述**：viewpoint invariant / universally robust / all views improved / all categories significant / causal proof / generalizes to arbitrary industrial datasets / 每个 illumination group 都改善。跨数据集泛化仍需路线 B（E20-B）。

---

## 1. 协议审计结论（运行前，2026-10-15）

**审计问题**：E16 的训练数据是否绑定 view120？

**审计结果（事实，实测）**：
- bank pool = `E8` 规则 `sorted(train_specs)[::3][:10]`（10 个 train specimens）的**全部图** = 10 × 12 views × 10 illum = 1200 图；bank = `E8.bank_for(pool, seed)` 抽 300。
- 实测 bank（seed0/1/2，Bird/Car/Cube 共 9 次）的 view 分布**横跨全部 12 views**（每 view 18–32 图；train240 每 view 14–29；val60 每 view 2–9）。
- bank ∩ test specimens = 0（specimen 级隔离，无泄漏）。
- **结论：训练数据不绑定 view120**。adapter 不是 view-specific；直接测试 090/240/330 属于 **"同一 12-view bank/训练设置下的多视角评估"**（每个测试 view 与原始 view120 设置在结构上相同：同 specimen 隔离、bank 含该 view 的其他 specimen 图）。
- 该 setting **不是** zero-shot 视角迁移（那需要 view-specific bank 或留出 view），**也不是** per-view 训练。本协议不改变训练 view、不重建 bank。

---

## 2. 实验设计（冻结）

| 项 | 值 |
|---|---|
| 类别 | Bird, Car, Cube |
| 视角（测试条件） | 090 / 120 / 240 / 330（E8 冻结 4-view 集；0 获取成本，l0_448 已缓存） |
| seed | Round 1 = seed0；GO 后补 seed1/2 |
| 方法 | F0（frozen DINOv2-B, 1-NN-max）· R1-ILA（BETA=0.1） |
| 训练次数 | **每 (category, seed) 只训练 1 个 adapter**，然后在 4 个 view 上分别评估（view 是测试条件，不是调参轴） |
| 输入 | `data/m2ad/e7_cache/l0_448/<cat>/{Good,NG}/<spec>/A<view>_I<illum>.png`（448 BICUBIC） |

### 2.1 训练协议（逐字继承 E16/E17，实现上直接复用同一函数）

- frozen final-layer patch tokens（`get_intermediate_layers(n=1, norm=True)`），1024×768；prep：Resize(448, BICUBIC, antialias) → ToTensor → 可选 aug → ImageNet normalize（`e13a_screen.prep_image`）。
- R1：`z' = z + Wz`，W zero-init，trained params = 589824；normal-only；`L = L_cons + 0.1·L_preserve`（BETA=0.1）。
- AdamW lr=1e-3，max_steps=300，patience=30，eval_every=5，batch=8，train/val=240/60（bank 内按序切分），aug = {brightness, contrast, gamma} × {0.85, 1.15}（6 变体）。
- 训练器 = `e13b_screen.train_r1`（torch.manual_seed(seed) + np.random.default_rng(seed)）；调用序列与 `e16_qualify.run_cell` 完全一致。
- F0：bank 300 图 patch 特征 L2 → 1-NN-max 评分（`e11_p2_runner.nn_image_scores`）。

### 2.2 标签与过滤（冻结）

- 标签 = official `image_anomaly`（1=anomaly, 0=good）；**禁止** legacy `object_anomaly`。
- 过滤 = `detectable != False`（继承 E18/E19 的 `official_metrics`）。
- view120 的 (raw=700, kept=653/656/678) 必须与 E16/E18/E19 完全一致（运行前数据审计已 PASS，见 `data_audit.csv`）。

### 2.3 view120 复用规则

- **view120 结果复用 E16/E18/E19 历史值**（`e16_partial.json` 的 per-image scores；AUROC/AP 见 S1/S2 目标值），CSV 中标记 `source=REUSED_VERIFIED`。
- E21 运行中仍会提取 view120 的 700 图（作为 cell 的复现校验，见 §3），但**报告值以历史为准**；复现差值单独记录。

---

## 3. 复现要求（S1–S4，全部 PASS 才允许继续；失败即 STOP）

| 检查 | 内容 | 容差 |
|---|---|---|
| S1 | E21 复算 view120 F0 AUROC/AP vs 历史：Bird 0.74725 / Car 0.78552 / Cube 0.85410（AUROC 目标值取自 E18/E19 主表） | ≤ 1e-6 |
| S2 | E21 复算 view120 R1-ILA seed0 AUROC/AP vs 历史：0.75786 / 0.80511 / 0.87010 | ≤ 1e-6 |
| S3 | (category, specimen, view, illumination) 唯一映射，无重复/错标（test 4 views） | 0 duplicates |
| S4 | 测试 view 改变不改变 bank / seed / adapter 参数：每个 cell 记录 `bank_hash`（image_path 排序后 sha256）与 `adapter_hash`（W float32 bytes 的 sha256）；与 E16 路径的构建代码比对（同函数同顺序） | 记录并核对 |

> S1/S2 的"历史目标值"以 `results/e16/e16_partial.json` 的 **f0/r1 全精度值**为唯一 ground truth（E18/E19 主表即由它生成）：
> F0 = Bird 0.7472503754 / Car 0.7855243446 / Cube 0.8540945896；R1 = 0.7578641141 / 0.8051123596 / 0.8700956938。
> **展示精度注记（执行前发现，2026-10-15）**：协议原文 "Cube 0.85410" 是 0.8540945896 的 4 位小数展示；其 5 位小数形式为 0.85409。运行 preflight 以 1e-5（展示精度）核对协议值、以 1e-6（实验复现容差）核对 e16_partial，两者均已通过。判据与容差未变。

---

## 4. Round 1 运行与产物

- Round 1 = seed0 × 3 categories × 4 views。新计算 = 090/240/330（view120 仅作复现校验，不重报）。
- 每 cell 记录（`round1_seed0.csv`）：category, view, seed, n_good, n_anomaly, AUROC_F0, AUROC_R1, ΔAUROC, AP_F0, AP_R1, ΔAP, runtime_s, bank_hash, adapter_hash, label_protocol, source(REUSED_VERIFIED|NEW)。
- 原始产物：`e21_partial.json`（每 cell：4 views 的 per-image scores + labels + specimen + illumination + repro 差值 + hashes；原子写入，支持 resume）。

## 5. 统计（冻结口径）

- **cell**：AUROC/AP（`official_metrics`）。
- **per-category 4-view pooled**：合并该 category 的 4-view per-image (score, label)（同一 score 空间）计算标准 AUROC/AP。
- **per-view 3-category pooled**：3 个 category 的 view-AUROC 的 **unweighted mean**（跨 category 分数尺度不可比，故不做样本合并；样本数 653–678 近似相等）。
- **overall pooled（主判据口径）**：3 个 category 的 4-view pooled ΔAUROC（和 ΔAP）的 unweighted mean；同时报告辅助口径：12-cell 平均、样本加权平均。
- **bootstrap（§13E）**：paired image bootstrap，`n_boot=3000`，固定 RNG seed=20261015+cell 序号：
  - per-cell：该 cell 的 test images 重采样，ΔAUROC 的 95% CI；
  - per-category pooled：该 category 4-view 合并样本重采样；
  - overall：**stratified**（每 category 内独立重采样 → 重算 3 个 pooled Δ → unweighted mean）的 95% CI。
  - **声明**：paired image bootstrap 仅反映 image sampling uncertainty，不等价于 cross-seed uncertainty（后者需 seed1/2）。

## 6. GPU/资源预算

- 每 cell ≈ 提取 4,900 图（bank 300 + aug 1800 + test 2800）+ 训练 300 步内 + 4-view 评分；按 E16 实测外推 ≈ 6–8 min/cell。
- Round 1（3 cells）≈ 20–30 min GPU；若 GO，seed1/2（6 cells）≈ 40–60 min。
- 磁盘新增 < 2 GB（仅 JSON/CSV/图）。禁止下载/解压原图；禁止复制 l0_448。

## 7. 判据（冻结，逐字执行）

**GO**（→ 自动补 seed1/2）：
- overall pooled ΔAUROC > 0；且 overall pooled ΔAP > 0；
- 至少 2/3 category 的 pooled ΔAUROC > 0；
- 无 catastrophic regression：任一 category×view 的 ΔAUROC ≥ -0.03。

**HOLD**（停止，不补 seed1/2，回报待裁决）：
- overall pooled 为正，但只有 1/3 category pooled 为正；或出现单个明显负 view（未达 -0.03）；或 AUROC/AP 明显冲突。

**STOP**（停止、归档负结果、不救火）：
- overall pooled ΔAUROC ≤ 0 且 ΔAP ≤ 0；或 ≥2 categories pooled ΔAUROC ≤ 0；或任一 cell ΔAUROC < -0.03。

## 8. 产物清单

`results/e21/{protocol.md, data_audit.csv, e21_partial.json, round1_seed0.csv, all_seeds.csv(若 GO), analysis_{summary.csv, bootstrap_ci.csv}, figures/*.png+pdf, verification.csv, README.md}` + `scripts/e21_multiview_validation.py`（新脚本，不改历史 runner）。

## 9. 六项历史遗留文件（禁止触碰）

`monitor/exp_monitor.py` · `results/e8/status.json` · `monitor/events.jsonl` · `scripts/_lockprobe.txt` · `scripts/_probe2.py` · `scripts/e11_p4_runner.py`
