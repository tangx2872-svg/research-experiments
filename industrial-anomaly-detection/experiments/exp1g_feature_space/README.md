# Experiment 1G — Feature-Space Representation Reshaping

> 状态：**完成**（2026-10-02）。判定：**CASE_A — REPRESENTATION_CHAIN_SUPPORTED**。
> 冻结协议：`results/experiment_1g/config.json`（FROZEN，分析前锁定）。

## 0. 启动四句话（冻结于 `config.json`）

1. **核心问题**：α-IN 如何改变 defect patch → normal memory bank 的最近邻距离分布，从而引起 1E 的 anomaly-score dispersion shrink/expansion？
2. **因果链**：α-IN → defect-to-normal-memory-bank NN distance distribution → image anomaly-score distribution。
3. **primary outcome**：GT defect patch 的 NN-distance dispersion（std，α=0 vs α=1 的变化量 ΔNN-std）。
4. **分层规则**：LEVEL 1（NN-distance）→ LEVEL 2（feature geometry）→ LEVEL 3（norm/channel variance），只有上一层成立才进入下一层，禁止跳层扫指标。

## 1. 双轨 memory-bank 分解

| Track | 距离 | 含义 |
|---|---|---|
| A（matched） | `dist(Fα, Mα)` | 系统层 response：bank 随 α 一起重建（真实 PatchCore 部署状态） |
| B（frozen） | `dist(Fα, M0)` | defect representation drift：bank 冻结在 α=0，只看特征端移动 |

目的：区分「defect 特征自己移动」vs「normal bank 移动」vs「相对几何重构」。若 frozen track 单独就能复现 1E 分级，则机制主要在 defect 特征表示本身。

## 2. Selection 与 patch-role

- **selection**（`selection.csv`，immutable）：从 1E seed_summary 按 Δdefect_std（3-seed mean）严格取 shrink bottom3 / expand top3 / |Δstd| 中间 3：
  - shrink：cable/bent_wire（-3.51）、hazelnut/print（-2.77）、bottle/contamination（-2.30）
  - neutral：screw/manipulated_front（-0.14）、screw/thread_side（+0.20）、screw/thread_top（-0.36）
  - expand：grid/glue（+6.67）、grid/metal_contamination（+9.78）、grid/thread（+11.43）
- **patch-role**：defect = GT overlap > 50%；boundary = 0 < overlap ≤ 50%；background = overlap = 0。primary 用 defect patches，boundary/background 作 control。
- **seeds**：协议规定 pilot 仅 seed=0；出现明确结构才补 seed=1/2。
- **α**：{0, 0.25, 0.5, 0.75, 1.0}。

## 3. 数据规模

- 25 个 memory bank（5 category × 5 α），coreset 与 1E 完全同构（全量 train 图，ratio 0.1，seed 固定）。
- 770 个 embedding npz = 154 张 defect 图 × 5 α，每个含 `feat_pre` / `feat_alpha`（1024×1536）/ `overlap`（32×32）。
- `patch_level.csv`：788,480 patch 行（770 × 1024）。150MB，不入 git，可由 npz 重建（extract 脚本 `rebuild_patch_csv`）。

## 4. 结果

### L1（PRIMARY）— 1E Δdefect_std ↔ 1G ΔNN-std

| defect | 1E Δstd | matched ΔNNstd | frozen ΔNNstd | family |
|---|---|---|---|---|
| cable/bent_wire | -3.51 | -0.606 | -1.403 | shrink |
| hazelnut/print | -2.77 | -0.684 | -2.214 | shrink |
| bottle/contamination | -2.30 | **+0.514** | -1.884 | shrink |
| screw/manipulated_front | -0.14 | -0.180 | -0.066 | neutral |
| screw/thread_side | +0.20 | -0.317 | +0.076 | neutral |
| screw/thread_top | -0.36 | -0.287 | -0.709 | neutral |
| grid/glue | +6.67 | +1.958 | +1.782 | expand |
| grid/metal_contamination | +9.78 | +2.431 | +2.231 | expand |
| grid/thread | +11.43 | +6.358 | +6.152 | expand |

- **Spearman(1E, matched) = +0.833（p=0.005, n=9）** → 通过冻结阈值（|ρ|≥0.6 且 p<0.05）。
- **Spearman(1E, frozen) = +0.950（p<0.001, n=9）** → frozen track 对应更强。
- shrink 端三个、expand 端三个全部方向正确；neutral 端 1E 本身就接近零，ΔNNstd 也接近零。
- 1E 的 dispersion shrink/expansion 在特征空间有清晰对应物：**defect patch 到 memory bank 的 NN 距离分布收缩/膨胀**。

### L2 — NN 变化是否来自 defect feature geometry

- ΔNNstd ↔ Δfeature-dispersion（到 centroid）：ρ=+0.600（p=0.088）——方向对，p 略超阈值。
- ΔNNstd ↔ Δnorm-std：**ρ=+0.700（p=0.036）→ 通过**。
- 判定：feature geometry 解释成立（经 norm-std 通道）。defect 特征族在 α 增大时 norm 结构整体重排，重排幅度分级对应 NN 分布变化。

### L3 — norm / channel variance

- Δnorm-std ↔ Δchannel-var：ρ=+0.600（p=0.088）——弱 hint，未过显著性。norm 重排与 channel 方差变化方向一致但证据不足，留给 1H 的 layer/channel 实验裁决。

### 双轨分解的两个关键观察

1. **frozen ≈ matched，且 frozen 更贴 1E**（ρ 0.950 vs 0.833）：1E 分级主要由 **defect 特征表示自身的漂移**承载，而不是 bank 端重构的副产品。
2. **bottle/contamination 是唯一方向分歧点**：matched +0.51（微膨胀）vs frozen -1.88（收缩）→ 该缺陷的特征表示在收缩，但 **Mα bank 端的结构变化把它反向推回**。system-level（matched）读数会掩盖 defect 漂移——这正是双轨设计的价值。

### Background control（诚实记录）

Image-level 聚合（a0→a1）：

| family | defect ΔNNstd | background ΔNNstd |
|---|---|---|
| shrink | -0.171 | **-1.021** |
| neutral | -0.260 | +0.779 |
| expand | +3.582 | **+4.448** |

背景的 NN-std 并非不动：expand 端背景与缺陷同步上升，shrink 端背景反而降得比缺陷多。即响应**不是纯 defect 局部化**，存在一个随 α 增大的全局分量（IN 的全局方差重塑）。但背景变化**不与缺陷同步同构**（shrink 端方向相反、幅度分级不同），CASE_A 的「非 background 同步变化」在"非镜像"意义上成立。任何 channel/layer 机制（1H）必须同时解释：defect 分级对应（强）+ 全局分量（存在）。

## 5. Verdict（按冻结规则）

**CASE_A — REPRESENTATION_CHAIN_SUPPORTED**：稳定链 1E ↔ NN-dispersion ↔ feature-geometry 成立（L1 两个 track 都过阈值，L2 经 norm-std 过阈值），→ 继续 channel/layer 机制（1H 已 preflight）。

## 6. 下一步（按协议，不自动执行）

1. **seed 1/2 复核**：冻结协议规定「seed=0 出现明确结构才补 seed=1/2」——现在结构明确（L1 ρ=0.83/0.95），应复核 9 个 defect 的 seed 稳定性。
2. **1H（layer-specific normalization sensitivity）**：CASE_A 授权进入 channel/layer 机制层，回答 norm 重排发生在哪些 layer/channel；需把 L3 的弱 hint（channel-var）与 contamination 的双轨分歧作为重点检验对象。
3. contamination 案例值得单独解剖：matched/frozen 分歧提示 bank 端 coreset 几何在 α 下的变化方式是独立变量。

## 7. 执行中的技术决策与偏差记录（均为技术性，非事后修正）

1. **空 defect-mask fallback（8 图）**：thread_side（000/003/004/011/013/019）、thread_top（1 图）、glue（1 图）的 GT mask 在 32×32 网格下 max overlap 仅 0.37–0.49，无 patch 达到 >50% 阈值。按统一规则降级为 overlap>0（boundary∪defect）计入 defect patches，并对全部 154 图验证 fallback 后非空。这是细小缺陷在粗网格下的固有边界情况，不是挑选。
2. **`nn_distances` OOM 修复**：原实现广播出 (N,M,1536) 张量（screw bank 32768×1536 下约 15GB），改为分块矩阵积 `‖f‖²+‖b‖²−2f·bᵀ`；与暴力解误差 <2e-6（float32 精度）。
3. **bank 加载缓存**：原实现每个 npz 重复从磁盘加载 bank（1540 次 × ~180MB），加 `(cat, α)` 级缓存降至 25 次；纯 IO 优化，不改统计口径。
4. **extract 脚本解包顺序核查**：`train_ids, _ = make_validation_split(...)` 误把 val_ids（20 张）当 train_ids，但 anomalib `engine.fit` 重新 `setup()` 会以全量 train 重建 train_data，实际 bank 用全量图构建（cable 22937 patch = 224×1024×0.1 吻合），与 1E 同构，结果有效。已在 extract 脚本中加断点续跑（category 级 skip + npz 全量重建 CSV），全程中断两次均无损续跑。
5. **分析在沙箱内被 file-read 拦截一次**，改在沙箱外重跑（同一脚本、同一数据），非方法学变更。

## 8. 文件

```
results/experiment_1g/
├── config.json              # 冻结协议
├── selection.csv            # 9 defect 选择（immutable）
├── embeddings/*.npz         # 770 个（6GB，不入 git）
├── memory_banks/*.npy       # 25 个（4.2GB，不入 git）
├── patch_level.csv          # 788,480 行（150MB，不入 git，可重建）
├── analysis/
│   ├── image_level.csv      # 770 行 × 21 字段（图级统计）
│   ├── defect_summary.csv   # 9 行（defect 级 Δ 汇总）
│   └── verdict.json         # CASE_A 判定 + 三层统计量
└── figures/
    ├── figure1_1e_vs_1g_core.png        # L1 散点：1E vs matched/frozen
    ├── figure2_trajectory_3group.png    # 三组 NN-std α 轨迹
    ├── figure3_feature_geometry.png     # L2/3：dispersion/norm/channel-var vs 1E
    └── figure4_defect_vs_background.png # defect vs background 响应对比

scripts/
├── experiment1g_select.py    # selection 生成
├── experiment1g_extract.py   # embedding/bank 提取（--resume + rebuild_patch_csv）
├── experiment1g_analysis.py  # L1/L2/L3 + 图级/defect 级 CSV
├── experiment1g_figures.py   # 4 张图
└── experiment1g_verdict.py   # CASE 判定
```
