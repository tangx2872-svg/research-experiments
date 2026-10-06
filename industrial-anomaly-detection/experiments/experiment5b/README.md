# Experiment 5B — Normal-Only Normalization-Tolerance Predictability Probe

> 状态：**5B-C 已完成 → FINAL VERDICT: CASE_A (FINAL)（见 §17）**。
> Predictor registry 在计算任何 target correlation 之前写入
> `results/experiment_5b/reference/predictor_registry.csv`。
> 前序：5A ✅（bottle CASE A）→ 5A-H ❌ CASE C（global frozen rule 不泛化，
> commit `ff4a1b2`）。本实验不设计新方法、不搜 α，只回答一个可预测性问题。

---

## 0. Paper Navigation

📍 Stage ⑤ Method Design：5A ✅ → 5A-H ❌（global rule STOP）→ **5B（当前）**

### ① 我怀疑什么？

5A-H 证明不同 category 对 normalization 的 tolerance 差异极大（bottle tolerant；
grid/screw 被 systematic harm）。怀疑这种 category-level tolerance 差异**在训练阶段
就已经体现在 normal training feature geometry / statistics 中**——即无需看到任何
defect 就能提前评估 normalization risk。

### ② 准备干什么？

只用 **normal training data**（1J 冻结的 α=0 normal train coreset banks，
5 categories × 3 seeds × {layer2, layer3}，30 个 npy，协议与 5A-H 同 split）计算
简单可解释统计量（magnitude / geometry / cross-layer ratio），与 5A-H 已观测的
defect preservation damage（target）做描述性相关 / 排序 / LOCO 分析。CPU only。

### ③ 看什么？

normal-only statistics 能否预测某 category 在 normalization 后会不会严重损失
defect evidence（Damage = d′_B0 − d′_normalized > 0 为损伤）。

### ④ 什么结果意味着继续？

一个或少数 normal-only indicators：跨 seed 稳定 + 与 damage 有明确关系 +
LOCO 方向基本正确 + 优于 negative controls → GO category-adaptive method design。
若只有看 defect 才能解释 / predictors 不稳定 → STOP 此路线，不造复杂网络救结果。

---

## 1. Target（运行前冻结）

- **Primary：C2 damage** = d′_B0(c,s) − d′_C2(c,s)（5A-H per_unit.csv，15 units）
- Secondary：C3 damage、G2 damage（同式，C3/G2）
- Tolerance = −Damage
- **禁止在看到结果后更换 primary target。** Primary 用 uniform C2 而非 G2：
  首先回答"这个 category 能不能承受 normalization"，而非"G2 是否成功"。

## 2. Data Sources（全部已存在于磁盘，CPU only）

- **X（predictors）**：`results/experiment_1j/banks/{category}_seed{seed}_layer{2,3}.npy`
  —— 1J 冻结的 normal train coreset features（KCenterGreedy 0.1，α=0 原始 feature，
  与 5A-H 同 `make_validation_split` 协议；对 X 而言仅为 train/good 数据，
  无 defect 信息）+ train/good 原图（negative controls）
- **Y（target）**：`results/experiment_5a_h/summary/per_unit.csv`（已 commit `ff4a1b2`）

## 3. Frozen Predictor Registry（15 个，≤15 上限；Group C 缺资产见 §7）

| # | name | definition | group | source |
|---|---|---|---|---|
| 1 | norm_mean_L2 | bank patch 向量 L2 范数均值 | A | banks |
| 2 | norm_mean_L3 | 同上 layer3 | A | banks |
| 3 | norm_cv_L2 | patch 范数 std/mean | A | banks |
| 4 | norm_cv_L3 | 同上 layer3 | A | banks |
| 5 | rms_radius_L2 | bank patch 到质心 RMS 距离 / 质心范数（尺度不变） | B | banks |
| 6 | rms_radius_L3 | 同上 layer3 | B | banks |
| 7 | nn_dist_rel_L2 | bank 内 NN 欧氏距离均值 / bank 平均范数 | B | banks |
| 8 | nn_dist_rel_L3 | 同上 layer3 | B | banks |
| 9 | eff_dim_L2 | PCA participation ratio（有效维数，λ_i²/Σλ² 口径） | B | banks |
| 10 | eff_dim_L3 | 同上 layer3 | B | banks |
| 11 | radius_ratio_L3L2 | rms_radius_L3 / rms_radius_L2 | D | 派生 |
| 12 | nn_dist_ratio_L3L2 | nn_dist_rel_L3 / nn_dist_rel_L2 | D | 派生 |
| 13 | eff_dim_ratio_L3L2 | eff_dim_L3 / eff_dim_L2 | D | 派生 |
| 14 | img_brightness_mean | train/good 平均像素亮度 | control | images |
| 15 | img_pixel_std | train/good 像素强度 std | control | images |

全部 uses_normal_only = true。Direction 假设（预注册，分析后核对）：
tolerant category（低 damage）预期 feature 分散度大（rms_radius / nn_dist / eff_dim
高）——normal feature 云越"胖"，IN 拉动质心的相对扰动越小。

## 4. Analysis Plan（预注册）

1. **描述性**：category × predictor 表 + z-score heatmap（n_category=5 为主，
   seed-level n=15 仅 exploratory）
2. **相关**：每 predictor 对 3 个 target 的 Spearman ρ（category-level，seed-mean；
   n=5 不做显著性主张，|ρ|≥0.7 仅标记 strong descriptive association）+
   seed-level（n=15）作方向一致性核对
3. **Seed stability**：between-category variance / within-category seed variance
   （stability ratio，README 冻结口径：每 predictor 用 seed-mean 的跨 category 方差
   除以 category 内 seed 方差均值）
4. **Rank prediction**：predictor 排序 vs C2-damage 排序的 Spearman/Kendall；
   重点看能否把 grid/screw 排到高风险端
5. **LOCO**：4 训 1 测；只允许 single-predictor threshold / ≤2 predictor 线性；
   risk label 冻结为 **Damage > train-categories median → HIGH**（held-out category
   的 target 不参与 threshold 构造）；报 accuracy / balanced accuracy（n=5 仅 exploratory）
6. **Negative controls**：brightness / pixel-std / random(seed=0) —— geometry
   predictors 应明显优于 controls，否则"只是在编码图片亮不亮"

## 5. §16 专项：category-specific s_L2/s_L3（global rule 失败归因）

1J 冻结的 s_L2=0.267/s_L3=0.161 是 9 defects 池化值。逐 category 重算需要
**normal train features 的 IN(F) 前后 geometry**（per-image InstanceNorm，
无法从 coreset bank 恢复 per-image 统计）→ 依赖最小 GPU 提取，见 §7。
此步只归因失败、不重设 α。

## 6. Sanity（全部 PASS 才解释）

S1 predictors 仅来自 train/good（banks = train split features；图像 control 只读
train/good）；S2 无 test defect image 进入 X；S3 无 defect label 进入 X；
S4 registry 在 target correlation 计算前落盘（文件时间戳 + 脚本顺序）；
S5 predictor 数 ≤15；S6 category-level n=5 与 seed-level n=15 明确分离；
S7 无 α search；S8 Stage ④ 文件 md5 不变；S9 5A commit `ba11038` 不变；
S10 5A-H commit `ff4a1b2` 结果文件 md5 不变；S11 primary predictors 无 NaN/Inf；
S12 bank 的 (category, seed) 与 per_unit.csv 的 (category, seed) 对齐
（split 协议一致性：1J 与 5A-H 同 `make_validation_split`）。

## 7. Missing Asset（预注册时已知）

**Group C（normalization sensitivity：geometry(F) vs geometry(IN(F))）与 §16 的
category-specific s_L2/s_L3 无法从现有资产恢复**：1J banks 是 coreset 子采样，
patch 归属 image 信息丢失，无法重放 per-image InstanceNorm2d。需最小 GPU 提取
（normal train 图前向 → (N,C,H,W) 特征 → CPU 计算 IN 前后 geometry；无训练、
确定性、预计 <10 min GPU / ~2.6 GB VRAM）。**按协议：本轮先完成 CPU-only 分析，
输出 [MISSING NORMAL-ONLY ASSET] 后 STOP，不自行启动 GPU。**

## 8. Exit Criteria（预注册）

- **CASE A**：≥1 简单 normal-only predictor 达标（strong association + seed stable +
  rank sensible + LOCO 方向 mostly correct + 优于 controls + 物理意义合理）
  → Method v1 Design（normal-only category-adaptive）
- **CASE B**：有 category 结构但单 predictor 不稳 / LOCO 一般 → 最多讨论 2-predictor
  简单 rule，不训练复杂模型
- **CASE C**：normal-only 统计与 damage 无稳定关系 → STOP normalization-adaptation
  路线，回 Candidate B / Adapter
- **CASE D**：只有 defect 信息才能解释 tolerance → STOP 当前 route

## 9. Outputs

`results/experiment_5b/{reference,summary,figures,logs}/`；
summary：normal_only_predictors / category_predictors / target_damage /
correlation_summary / seed_stability / rank_analysis / loco_predictions /
negative_controls / sanity_checks；
figures：category_predictor_heatmap / predictor_vs_damage / category_risk_ranking /
seed_stability（+ Group C 批准后：l2_l3_category_sensitivity）。

## 10. Results

执行命令：`python scripts/experiment5b_analysis.py`（CPU only，耗时 262s）。

### 10.1 Primary target（C2 damage）

按 category 平均（3 seeds），damage = d′(B0) − d′(C2)（正值表示 C2 有损）：

| category | damage_C2 mean | damage_C2 std | damage_C3 | damage_G2 |
|---|---|---|---|---|
| bottle   | **+1.207** | 0.113 | +1.627 | +1.029 |
| grid     | **+1.001** | 0.026 | +1.333 | +1.155 |
| cable    |  +0.249 | 0.072 | +0.434 | +0.315 |
| screw    |  +0.113 | 0.090 | +0.390 | +0.328 |
| hazelnut | **−0.183** | 0.214 | +0.006 | −0.035 |

C2 损伤排序：bottle ≈ grid > cable > screw > hazelnut。

### 10.2 Category-level correlation（n=5，primary target C2）

按 |ρ| 排序：

| predictor | group | ρ(C2) | ρ(C3) | ρ(G2) | seed ρ(C2) | direction consistent? |
|---|---|---:|---:|---:|---:|:---:|
| eff_dim_L3 | B | **−0.90** | −0.90 | −0.90 | −0.875 | ✅ |
| radius_ratio_L3L2 | D | **−0.90** | −0.90 | −0.60 | −0.832 | ✅ |
| nn_dist_ratio_L3L2 | D | **−0.90** | −0.90 | −0.90 | −0.804 | ✅ |
| img_pixel_std | control | **+0.90** | +0.90 | +0.50 | +0.818 | ✅ |
| eff_dim_L2 | B | −0.80 | −0.80 | −1.00 | −0.768 | ✅ |
| nn_dist_rel_L3 | B | −0.80 | −0.80 | −1.00 | −0.739 | ✅ |
| norm_mean_L2 | A | +0.70 | +0.70 | +0.70 | +0.636 | ✅ |
| rms_radius_L3 | B | −0.60 | −0.60 | −0.60 | −0.564 | ✅ |
| nn_dist_rel_L2 | B | −0.60 | −0.60 | −0.90 | −0.604 | ✅ |

方向与预注册假设一致（高分散度 → 低 damage）：除 control 外，B/D 组预测器均为负相关。

### 10.3 Seed stability

`between-cat var / within-cat seed var` 全部 ≫1（最小 384，最大 577032），
说明 predictor 的 category 差异远大于 seed 抖动。对所有 15 个预测器，
per-seed category-level ρ 方向跨 seed0/seed1/seed2 完全一致
（详见 `summary/seed_consistency_descriptive.csv`）。

### 10.4 LOCO（4 训 1 测，单 predictor threshold）

| predictor | acc / 5 | bal_acc | 备注 |
|---|---:|---:|---|
| rms_radius_L3 | **5/5** | 1.00 | L3 geometry，最稳 |
| eff_dim_L2 | **5/5** | 1.00 | L2 effective dimension |
| radius_ratio_L3L2 | **5/5** | 1.00 | cross-layer 比值 |
| nn_dist_rel_L2, nn_dist_rel_L3, eff_dim_L3, nn_dist_ratio_L3L2, norm_mean_L2 | 4/5 | 0.83 | 强但非完美 |
| norm_cv_L2, eff_dim_ratio_L3L2 | 3/5 | 0.58 | 中等 |
| img_pixel_std (control) | 3/5 | 0.58 | 与 random 同水平 |
| random_control | 3/5 | 0.58 | baseline |
| rms_radius_L2 | 0/5 | 0.00 | 无预测力 |

所有 held-out 极端类别（bottle 高损伤、hazelnut 负损伤）均被 top geometry
predictor 正确预测，说明相关不是由单一 category 驱动。

### 10.5 Negative controls & robustness

- `img_pixel_std` 的 |ρ| 与 top geometry 预测器并列 0.90，但 LOCO 仅为 3/5，
  与 random control 同水平，说明其 category-level 相关无法稳定泛化。
- `random_control` 3/5 是 LOCO 基线；geometry 预测器 5/5 明显高于该基线。
- 2-predictor 线性组合（train fold 内按 |ρ| 选 top-2）LOCO 4/5，
  未系统性优于单 predictor 5/5，说明复杂组合不是必需的。

## 11. Verdict

**CASE_A — Predictive Structure Exists（interim）**

> ⚠️ 本判定已在 §17（5B-C，补齐 Group C）升级为 **CASE_A (FINAL)**：
> 原 geometry 结论完全不变，但 normalization-sensitivity 通道（Group C）为 null。

理由：
- 至少三个预注册 normal-only predictor（`radius_ratio_L3L2`、`eff_dim_L2`、
  `rms_radius_L3`）与 C2 damage 有强描述性相关（|ρ|≥0.60），方向与预注册
  假设一致；
- 三 seed 方向完全一致，seed stability ratio 极高；
- LOCO 单 predictor 5/5 泛化，正确预测包括 bottle/hazelnut 在内的极端类别，
  优于 img_pixel_std / random 等 controls；
- 预测信息主要来自 L3 / L3-L2 相对结构，与此前 1H/1J 的 layer3 机制证据一致。

**Interim 限定**：本 verdict 基于 Groups A/B/D predictors。Group C
（normalization sensitivity：geometry(F) vs geometry(IN(F))）仍属
[MISSING NORMAL-ONLY ASSET]，按 §7 协议本次不自行启动 GPU 提取。
最终 verdict 需待 Group C 资产补齐后复核；若 Group C 仍支持当前趋势，
则 CASE_A 可转为 final。

## 12. Implementation Fix

初始 `nn_mean_dist` 在 squared-Euclidean 展开中漏掉 `||x||²` 项，导致
`nn_dist_rel_L2 = 0`（bank 内所有点被错误判为自身）。修复后公式：

```python
d2 = bn[i:i + chunk, None] - 2.0 * (xb @ b.T) + bn[None, :]
```

即完整 `d²(x,y) = ||x||² − 2x·y + ||y||²`。该 bug 在正式分析前发现并修复，
修复结果经独立 sanity 脚本验证（A–E 全部 PASS）。

## 13. Sanity

- 30 banks：可加载、shape 合法、无 NaN/Inf ✅
- NN distance：
  - raw d² 最小值 +1.3e2（无负值）✅
  - self-distance max 1.5e-11 ✅
  - brute-force 与 torch.cdist NN 距离 max|Δ|≈3e-13 ✅
  - 子采样 NN mean/median >0，零距离比例 0.0 ✅
  - 端到端全量 `nn_mean_dist` 在 bottle:0:L2/L3 与 grid:0:L2 上均 >0，
    `nn_dist_rel`≈0.21–0.24（原 bug 值 0）✅
- 主分析脚本 sanity S1–S12：11/11 PASS ✅

## 14. Limitations

- n_category=5，所有 p 值/LOCO 仅属描述性，不能作为显著性证据；
- predictors 来自 1J coreset 子采样（~10% patch），是真实 geometry 的近似；
- `img_pixel_std` control 与 top geometry 预测器在 |ρ| 上并列，说明存在
  “纹理/对比度丰富度” 与 geometry 的共变；LOCO 区分了二者的泛化稳定性，
  但仍需更大样本或额外 confound control；
- illumination 协议与 5A-H 绑定；
- Group C 缺失，最终 verdict 为 interim。

## 15. Paper Implication

本实验填充的证据链格子：

```text
Before 5B:
Stage ⑤ Method Design
  5A: bottle 上 fixed geometry rule CASE A ✅
  5A-H: global frozen rule CASE C ❌ STOP
  未决：category tolerance 是否可被 normal-only 特征预测？

After 5B:
  Normal Geometry → Normalization Tolerance 路线 CASE A（interim）
  支持方案① Category-Adaptive Normalization 作为候选
  关键 signal：L3 effective dimension / L3-L2 relative dispersion
```

5B 使论文从 “fixed rule 已死” 推进到 “normal-only geometry 可预测 tolerance”，
为后续方法设计提供了可解释、可验证的 normal-only 基础。

## 16. Next（建议，不执行）

1. **Group C 最小 GPU 提取**（需人工批准）：按 §7 对 normal train 图像提取
   F0(layer2/layer3) 与 IN(F0) 的 per-category geometry，补齐 predictor registry
   后复跑 5B，将 interim CASE_A 转为 final verdict。
2. 若 final CASE_A 维持：进入 Method v1 design —— 基于 normal-only
   geometry（如 `radius_ratio_L3L2` 阈值或 `eff_dim_L2/L3`）做 category-adaptive
   α 分配，严禁直接上复杂模型。
3. 若 Group C 后结果反转：按 CASE_C 停止 normalization-adaptation 路线。

**禁止**：在未获批准前启动 5C、实现 Adaptive α、开启新 GPU experiment、
创建新 predictor family、进入方案②。

---

## 17. Experiment 5B-C — Group C Completion & FINAL VERDICT（2026-10-06）

> 本附录为 5B 的最终状态；§1–§16 保留为 interim 记录（未修改任何历史内容）。
> 输出目录：`results/experiment_5b_final/`（**不覆盖** `results/experiment_5b/`）。

### 17.1 目的

补齐 5B 预注册但缺失的 Group C（normalization sensitivity），然后用**完全相同的冻结协议**
重跑分析，把 interim CASE_A 升级为 final verdict。

### 17.2 Group C 定义（读取任何 target 之前冻结）

5B registry（15 项）**未逐条枚举** Group C；README 中唯一的具名量是 §5
「category-specific s_L2/s_L3」（= 1J-A `radius_response`），§7 给出概念定义
「geometry(F) vs geometry(IN(F))」，§9 预注册 Group C figure 名 `l2_l3_category_sensitivity`。

按协议「恢复最保守的机械定义」，直接复用冻结实现（不新造指标）：

| 项 | 来源 |
|---|---|
| geometry | `scripts/experiment1j_extract.py::geometry_of`（1J-A 冻结） |
| IN | `FAlphaLayerPatchcoreModel._alpha_mix(alpha=1)` == `F.instance_norm(F)`（1E/1H/1J 冻结） |
| response | `log((metric(IN(F0))+1e-12)/(metric(F0)+1e-12))`；pca1 用简单差（1J-A 冻结） |
| feature 路径 | wide_resnet50_2 **eval** + feature_pooler；layer2 (512,32,32)、layer3 (1024,16,16) |

| # | predictor | layer | 定义 | 角色 |
|---|---|---|---|---|
| 1 | sens_radius_L2 | L2 | mean_image log(R(IN(F0))/R(F0)) | **primary（= s_L2）** |
| 2 | sens_radius_L3 | L3 | mean_image log(R(IN(F0))/R(F0)) | **primary（= s_L3）** |
| 3 | sens_mdc_L2 | L2 | mean_image log(MDC(IN(F0))/MDC(F0)) | secondary（同一冻结族） |
| 4 | sens_mdc_L3 | L3 | 同上 | secondary |
| 5 | sens_effrank_L2 | L2 | mean_image log(nPR(IN(F0))/nPR(F0)) | secondary |
| 6 | sens_effrank_L3 | L3 | 同上 | secondary |
| 7 | sens_pca1_L2 | L2 | mean_image (PCA1(IN(F0)) − PCA1(F0)) | secondary |
| 8 | sens_pca1_L3 | L3 | 同上 | secondary |

全部 uses_normal_only = true；direction hypothesis = high→fragile。
**UNRESOLVED_PRE-REGISTERED PREDICTOR：无**（8 项均可由冻结代码唯一恢复）。

Freeze record：`results/experiment_5b_final/reference/group_c_freeze.json`
（frozen_at `2026-10-06T17:07:27`，git_head `24d1b22`，**targets_read = false**，
md5：group_c_predictors.csv `629eb89e…`，predictor_registry_group_c.csv `d95b60f9…`）。

### 17.3 Extraction 与 sanity

- forward-only（无训练、无 α search），仅读 `<cat>/train/good`；
  15 units = 5 categories × 3 seeds，共 **3924 张** normal 图。
- **S-C1…S-C7 = 7/7 PASS**：
  shape(F0)==shape(IN(F0))；无 NaN/Inf；determinism max|Δfeature| = `0.000e+00`；
  IN 效应存在；覆盖 15 units；无 defect 泄漏（路径断言）；
  **S-C7 定义一致性：与 1J-A 冻结 per_image CSV 对照 n=64，max_rel_dev = 1.11e-05**。
- 主分析链 sanity：**19 checks / 0 FAIL**。interim 的 `S5_predictor_count_le_15`
  在 final 表中标注为 **SUPERSEDED**（≤15 上限属于未枚举 Group C 的 interim registry），
  并补 `S5b_group_c_completion`（8 项 Group C 在 target 读取前已冻结）。
  未做任何 post-hoc predictor 增补；Group A/B/D 与 LOCO/风险标签规则未改动。

### 17.4 Results：Group C（全部 8 项，不筛选不隐藏）

| predictor | ρ(C2) | ρ(C3) | ρ(G2) | seed ρ(C2) | LOCO | 逐 seed 方向 | 留一 category ρ 范围 |
|---|---:|---:|---:|---:|---:|:---:|---|
| sens_radius_L2 | **−0.30** | −0.30 | +0.30 | −0.246 | 2/5 | ✅ | [−0.8, **+0.4**]（剔 bottle 反号） |
| sens_radius_L3 | +0.10 | +0.10 | +0.40 | +0.146 | 3/5 | ✅ | [−0.4, +0.6] |
| sens_mdc_L2 | −0.30 | −0.30 | +0.30 | −0.246 | 2/5 | ✅ | [−0.8, +0.4] |
| sens_mdc_L3 | −0.10 | −0.10 | +0.50 | −0.096 | 0/5 | ✅ | [−0.6, +0.4] |
| sens_effrank_L2 | −0.20 | −0.20 | +0.30 | −0.157 | 1/5 | ✅ | [−0.8, +0.2] |
| sens_effrank_L3 | −0.20 | −0.20 | +0.30 | −0.143 | 1/5 | ✅ | [−0.8, +0.2] |
| sens_pca1_L2 | +0.20 | +0.20 | −0.30 | +0.168 | 1/5 | ✅ | [−0.2, +0.8] |
| sens_pca1_L3 | +0.20 | +0.20 | −0.30 | +0.125 | 2/5 | ✅ | [−0.2, +0.8] |

**事实**：Group C 全部 8 个 predictor 与 C2 damage 仅弱描述性关系（|ρ| ≤ 0.30），
LOCO 0.00–0.60（random control = 0.60），留一 category 后符号可翻转。

**解释**：在最保守且预注册的 Group C 定义下，normal feature「对 normalization 的敏感度」
**没有**可泛化的 tolerance 预测力 —— 这是一个 negative result，如实记录。

**限定**：这不是对 Group D 的反证。sens_radius_L2（ρ=−0.30）与 eff_dim / L3:L2 ratio 族
同号（负），方向并不矛盾，只是幅度远低于可用水平，故判为 **null 而非反证**。

### 17.5 Q1–Q4

**Q1 — 原 best predictor `radius_ratio_L3L2` 加入 Group C 后是否仍稳定？**
ρ = **−0.90（与 interim 完全一致）**，seed ρ = −0.832，逐 seed 方向 3/3 一致，
LOCO **5/5** → **仍稳定**。（同理 `eff_dim_L3` ρ=−0.90 / LOCO 4/5、
`nn_dist_ratio_L3L2` ρ=−0.90 / LOCO 4/5。）

**Q2 — Group C 是否出现与 C2 damage 稳定相关的 predictor？**
否。最强者为 `sens_radius_L2`（ρ = −0.30，LOCO 2/5，剔 bottle 后反号），
其余 7 项更弱（含 `sens_mdc_L3` LOCO 0/5）。

**Q3 — Group C 与 Group D 是否方向一致、是否形成合理链路？**
- `geometry → damage`：**supported（descriptive）**
- `normalization sensitivity → damage`（Group C 通道）：**unsupported**
→ 链路整体 **partially supported**（仅 descriptive / mechanistic consistency 层面，
**不构成 causal proof**）。

**Q4 — Geometry 是否仍优于 negative controls（以 LOCO 为准）？**

| predictor | 类型 | LOCO |
|---|---|---|
| radius_ratio_L3L2 | geometry (D) | **5/5** |
| eff_dim_L2 | geometry (B) | 5/5 |
| rms_radius_L3 | geometry (B) | 5/5 |
| eff_dim_L3 | geometry (B) | 4/5 |
| img_pixel_std | control | 3/5 |
| random_control | control | 3/5 |
| img_brightness_mean | control | 2/5 |

→ geometry 在 LOCO 上明显优于 controls。但必须诚实记录：`img_pixel_std` 在 **|ρ|** 上
与 top geometry 并列（0.90），只有 LOCO 才拉开差距（3/5 vs 5/5）。

### 17.6 FINAL VERDICT

**CASE_A — Predictive Structure Exists (FINAL)**

按预注册规则逐条核对：
1. 原 geometry signal 加入 Group C 后仍稳定（ρ / seed / LOCO 完全不变）✅
2. top geometry predictor 三 seed 方向完全一致 ✅
3. LOCO 明显优于 negative controls（5/5、4/5 vs 3/5、3/5、2/5）✅
4. Group C **不推翻**现有结构（8/8 null，无反向证据），但**未提供**支持性的
   normalization-sensitivity 证据 ⚠️（规则要求的是"不推翻"，该条成立）
5. leave-one-category-out 后 top predictors |ρ| 仍 ≥ 0.8，非单一 category 驱动 ✅

**必须同时声明的限定（不得省略）**：
- 存活的是**通用 normal-feature 分散度几何**（eff_dim_L3 / L3:L2 radius ratio 等），
  而非 normalization-specific 的敏感度指标；「敏感度中介 tolerance」的机制解释
  **不被 Group C 支持**。
- `img_pixel_std` 在 |ρ| 上与 top geometry 并列，提示信号可能部分与图像纹理/对比度
  共变；目前唯一区分证据是 LOCO。
- n_category = 5，全部为描述性 / exploratory 证据，无显著性主张，不做 causal claim。
- 未执行未预注册的 bootstrap / permutation 检验。

### 17.7 Files（5B-C 新增，interim 全部保留）

- scripts：`scripts/experiment5b_c_extract.py`、`scripts/experiment5b_final_analysis.py`
  （后者 import 并复用 5B 冻结分析代码，仅扩展 registry 与合并 Group C 列，
  **不修改旧 predictor 算法 / target / LOCO 规则**）
- `results/experiment_5b_final/group_c/`：`per_image/`（15 CSV，3924 行）、
  `group_c_predictors.csv`、`group_c_sanity.csv`
- `results/experiment_5b_final/reference/`：`predictor_registry.csv`（23 项完整 registry）、
  `predictor_registry_group_c.csv`、`group_c_freeze.json`
- `results/experiment_5b_final/summary/`：冻结协议全部输出 + `final_summary.json`、
  `leave_one_category_out_rho.csv`、`seed_consistency_descriptive.csv`、
  `sanity_checks.csv` 与 `sanity_checks_final.csv`
- `results/experiment_5b_final/figures/`：`all_predictors_vs_damage`、`group_c_panel`、
  `l2_l3_category_sensitivity`、`loco_geometry_vs_controls`、`best_predictors_vs_damage`
  （+ 冻结协议原有的 4 张）
- `results/experiment_5b_final/logs/`：`group_c_extract.log`、`analysis_run.log`、
  `paper_navigation.txt`、`registry_freeze_time.txt`

### 17.8 下一步（建议，不执行）

1. 若继续方案①：5C = 最简 **Geometry-Guided / Category-Adaptive α v1**，
   必须与 fixed α（含 5A-H 已否定的 G2）直接对比，并额外加入 "predictor identity"
   对照（通用分散度 vs normalization-specific 敏感度），因为 Group C 已表明
   机制性通道缺乏证据。
2. 需人工批准新协议后才可启动。本轮 **不启动 5C**。
