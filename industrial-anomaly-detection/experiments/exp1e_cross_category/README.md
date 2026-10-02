# Experiment 1E — Formal Cross-Category Pilot

**解释实验**（非模型优化）。预注册协议驱动：验证 1B/1C/1D 在 Bottle 上发现的 defect-dependent α-IN response heterogeneity 是否跨 MVTec AD 类别成立。

- **日期**：2026-10-02
- **状态**：已完成，**VERDICT: CASE_A — CROSS_CATEGORY_HETEROGENEITY_SUPPORTED**
- **前置**：1B（screening）→ 1C（multi-seed stability）→ 1D（size control）→ 1D metric decomposition → 本实验

---

## 研究问题

α-IN 引起的 defect-dependent representation response（特别是 Δdefect_std 的方向分化与"方差收缩托高 d'"模式）是否：

1. 在 bottle 以外的类别稳定出现（3 seeds 符号一致）？
2. 不能被 defect area（size confound）单独解释？

## 预注册设计（冻结于 results/experiment_1e/config.json，执行中零更改）

- **模型**：PatchCore，backbone=wide_resnet50_2，layers=[layer2, layer3]，coreset_sampling_ratio=0.1，num_neighbors=9，image 256²
- **α-IN**：F_alpha = (1-α)F + α·IN(F)，InstanceNorm affine=False，位置 generate_embedding concat 后 reshape 前（与 1B/1C/1D 完全一致）
- **α grid**：[0, 0.25, 0.5, 0.75, 1.0]（完整五档，非端点）
- **seeds**：[0, 1, 2]（validation split / coreset 初始化均随 seed 变化）
- **validation**：每 category 从 train/good 划 20 张（固定 seed），不进 memory bank
- **z 标准化**：contemporaneous category×seed×α 的 test/good 分布（ddof=1）
- **d'**：(μ_D − μ_G) / sqrt((σ_D² + σ_G²)/2)（pooled std）
- **area**：GT mask 像素 / 全图像素（原始分辨率）；log_area_ratio = log(area_ratio + 1e-9)
- **类别 × 运行顺序**：grid → cable → screw → hazelnut 串行（仅资源管理顺序，无科学含义）；bottle 复用不重训

## 数据（5 categories，25 defect types，mask 零缺失）

| category | train good | test good | defect types | defect samples |
|---|---:|---:|---:|---:|
| bottle | 209 | 20 | 3 | 63 |
| grid | 264 | 21 | 5 | 57 |
| cable | 224 | 58 | 8 | 92 |
| screw | 320 | 41 | 5 | 119 |
| hazelnut | 391 | 40 | 4 | 70 |

## 执行记录

### Phase 0 — Git checkpoint
commit `2b49f05`（基础设施 + smoke test），工作树干净，作为 Pilot 起点。

### Phase 1 — 冻结配置
`results/experiment_1e/config.json`（FROZEN）。执行期间无任何配置修改。

### Phase 2 — Bottle 重建 + 1D equivalence check（不重训）
从 1C raw（3 seed × 5 α 逐样本 score）+ 1B area 重建统一 1E schema。**Equivalence check 9/9 全 PASS**：broken_large / broken_small / contamination × 3 seeds 的 Δz / Δdefect_std / Δd' 与 1D 确认结果完全一致（阈值 1e-6）。

### Phase 3–7 — 正式 Pilot（60 fits + bottle 15 复用 = 75 conditions）
串行执行，每 category 完成后自动 sanity checkpoint（completeness / sample count / finite values / good normalization mean(z)≈0 std(z)≈1 / defect discovery / area 0<ratio≤1 / 重复与错位检测）：

| category | 3 seeds 耗时 | sanity |
|---|---|---|
| grid | 10.3 min × 3 | PASS |
| cable | 9.5 min × 3 | PASS |
| screw | 14.5 min × 3 | PASS |
| hazelnut | 20.3 min × 3 | PASS |

无 FAIL、无 OOM、无中断；resume/skip 机制未触发（无中断场景）。

### Phase 8 — 资源记录（每 category×seed×α，resource.csv）

| category | coreset size | peak GPU MB |
|---|---:|---:|
| grid | 27 033 | 3 265 |
| cable | 22 937 | 2 785 |
| screw | 32 768 | 3 937 |
| hazelnut | 40 038 | 4 789 |
| bottle (1C) | ~21 400 | 未测（历史） |

峰值显存用 `torch.cuda.max_memory_allocated()` 实测（非 available VRAM）。8 GB 显存内全程安全。

## 结果

### Seed stability（25/25 defect types 至少一个维度 3/3 符号一致）

| dimension | POS 3/3 | NEG 3/3 | MIXED |
|---|---:|---:|---:|
| Δmean_gap | 6 | 18 | 1 |
| Δdefect_std | 13 | 11 | 1 |
| Δz | 2 | 19 | 4 |
| Δd' | 6 | 19 | 0 |

响应普遍稳定——现象不是随机产物。关键在**方向分化**：

### Category 内方向分化（heterogeneity within category）

| category | 出现方向分化的维度 |
|---|---|
| bottle | Δdefect_std, Δd' |
| grid | Δmean_gap |
| cable | Δdefect_std, Δz, Δd' |
| screw | Δmean_gap, Δdefect_std |
| hazelnut | Δmean_gap, Δdefect_std, Δd' |

5/5 类别内部都存在 defect type 间方向不一致的稳定响应；3 个新增类别（cable/screw/hazelnut）在方差维度（Δdefect_std 或 Δd'）分化，grid 全类型同向（Δmean_gap 与 Δdefect_std 全正）但幅度差异极大（metal_contamination +10.4 vs bent +1.1）。

### 方差收缩模式跨类别复现（Δz<0 & Δstd<0 & Δd'>0，全部 3/3 稳定）

| pattern instance | Δz | Δdefect_std | Δd' |
|---|---:|---:|---:|
| bottle/contamination | -1.95 | -2.30 | +0.48 |
| cable/bent_wire | -1.07 | -3.51 | +2.19 |
| hazelnut/print | -4.63 | -2.77 | +0.70 |

1D 发现的"α-IN 收缩 defect 方差 → sample-level z 下降但 group-level d' 反升"模式在两个新增类别出现稳定复现实例（hazelnut print 为 smoke test 预测的正式确认）。注意 cable/bent_wire 的 Δd'=+2.19 主要由 mean_gap 下降（-7.26）与方差收缩共同驱动，机制归属留给 1F。

### Area control（Phase 15，25 defect types，response = 3-seed mean signature）

| response | Model A: log(area) | Model B: +category | Model C: +defect identity |
|---|---:|---:|---:|
| Δz | R²=0.015 | R²=0.692 (ΔR²=0.677) | R²=1.000 (饱和) |
| Δdefect_std | R²=0.079 | R²=0.796 (ΔR²=0.717) | R²=1.000 (饱和) |
| Δd' | R²=0.052 | R²=0.246 (ΔR²=0.194) | R²=1.000 (饱和) |

**area-only 解释力接近零（R²=0.015–0.079）**。area contributes 但远不 sufficient——与 1D 的"area contributes, but is area sufficient to explain cross-category response heterogeneity?"答案一致：不充分。Model C R²=1.0 是 25 个 one-hot 自由度的饱和拟合，无诊断价值，仅表明异质性载体在 defect-level identity。

## 判定：CASE_A — CROSS_CATEGORY_HETEROGENEITY_SUPPORTED

四条件全部满足：

1. **A1** ≥2 个新增 category 存在方向分化：3 个（cable/screw/hazelnut，grid 幅度分化）
2. **A2** 稳定响应覆盖率：25/25
3. **A3** area-only 不充分：R² ≤ 0.079
4. **A4** 方差收缩模式跨类别复现：bottle/contamination + cable/bent_wire + hazelnut/print

**正确表述**：α-IN 的 defect-dependent response heterogeneity 跨类别成立；不同 defect type 的响应差异主要由 defect identity 而非 area 驱动；1D 的 metric decomposition 结构（mean/variance/normalization 三分量受不同方向作用）在跨类别数据中再次出现。

## 科学解释边界（Phase 16 声明）

- hazelnut print 的正式结果确认了 smoke test 的方向，但**机制解释（texture/structure/frequency/contrast attribution）属于 Experiment 1F，本实验不做**。
- cable/bent_wire、hazelnut hole 的 Δd'>0 与 bottle contamination"像"但不等同——不预设同一 mechanism。
- slope 仅为描述指标（trajectory_slopes.csv），非线性 trajectory 保留在 alpha_trajectory.csv。

## 局限

- 单 backbone（wide_resnet50_2）、单 detector（PatchCore）、单 image size（256²）外推性未知
- bottle 为重建数据（1C raw + 1B area），非同批重训；equivalence check 缓解但无法完全消除批次差异
- 每 category n_defect 10–25，seed summary 的 std 为 n=3 估计
- grid 的 texture defect 方差普遍放大（Δstd up to +11.4），可能与 texture/structure 先验有关，留待 1F
- Model C 饱和（R²=1.0）不可解释为"identity 完全解释"

## 文件

- 脚本：`scripts/experiment1e_runner.py`（单 condition）、`scripts/experiment1e_orchestrator.py`（串行+sanity+resume）、`scripts/experiment1e_bottle_reconstruct.py`（bottle 重建+equivalence）、`scripts/experiment1e_analysis.py`（Phase 10–13）、`scripts/experiment1e_figures_area.py`（Phase 14–15）、`scripts/experiment1e_verdict.py`（Phase 17）
- 冻结配置：`results/experiment_1e/config.json`
- 数据盘点：`results/experiment_1e/dataset_inventory.{csv,md}`
- 原始结果：`results/experiment_1e/<category>/seed_<seed>/{sample_level.csv, group_level.csv, resource.csv, config.json}`
- 统一分析：`results/experiment_1e/analysis/`（group_level_metrics / signatures / seed_summary / seed_stability / alpha_trajectory / trajectory_slopes / area_control_results / **verdict.json**）
- 图：`results/experiment_1e/figures/figure1~4`

## Next Step（等用户决策）

**停止。不自动进入 Experiment 1F。**

人工判断的问题：方差收缩模式（print/contamination/bent_wire 同型）与 grid texture 方差放大是否值得机制级研究（texture vs structure defect 先验、frequency analysis、clustering）。若继续，1F 应预注册 mechanism attribution 协议；若停止，当前链条（1→1B→1C→1D→1E）已构成完整的 defect-dependent α-IN response 现象学描述。
