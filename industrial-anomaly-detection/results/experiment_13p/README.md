# Experiment 13-P — Illumination-Nuisance Subspace Feasibility Probe

**日期**：2026-10-08 ｜ **环境**：AutoDL RTX 3090 24GB / conda `base` / PyTorch 2.8.0+cu128 / Anomalib 2.6.2
**起始 HEAD**：`ae9aec5`（Exp12）｜ **冻结物**：`config/{README,config,protocol_freeze,augmentation_registry,analysis_plan}.json` + `config/sha256_frozen.json`

> **状态：PROTOCOL FROZEN（在读取任何 13-P target statistic 之前冻结）。**
> 本轮是 **Feasibility Probe**：不追求 AUROC 提升，不允许围绕结果调参，**不是**正式方法实验。

## 1. Motivation — 为什么探索这条新路线

当前冠军是 **Adaptive B2（12/15 PASS）**，但 Exp12 已关闭 **Category × Channel gating** 路线
（M1/M2/M3 全部触发硬性淘汰；见 §18 Exp12）。因此需要**换一条与 C2/B2 family 机制不同的路线**。

一条自然的假设是：**illumination nuisance 在 feature 空间中只占少数方向**。若是这样，就可以用
**纯 normal/train 数据**（在受控 illumination perturbation 下）估计该 subspace，再把它抑制掉，
从而在不牺牲（甚至提升）defect preservation 的前提下获得 robustness —— 这就是候选方法 **INSS
（Illumination-Nuisance Subspace Suppression）**。

**但这条路线只有在以下条件成立时才值得开发**：subspace ① 明显低维；② 在不同 augmentation / category /
sampling seed 下稳定；③ 与此前 defect-sensitive representation 有可测量的（且**特异于 defect** 的）关系。
本轮**只做这个可行性检验**；不通过就 STOP，不开发 INSS。

## 2. Literature-gap framing（本轮的定位）

- 工业异常检测的 robustness 文献多依赖 **augmentation-invariant training** 或 **test-time augmentation**，
  需要训练或额外推理；纯 test-time、training-free 的 **nuisance-subspace** 路线在 MVTec/illumination
  场景下缺少**可行性证据**（该 subspace 是否真的低维、稳定、且与 defect 信息纠缠）。
- 本文此前的工作（Exp1/1H/1I）已确认 **α-IN 在 robustness 与 preservation 之间存在 trade-off**，
  并提示「过度 invariance 会损害 defect information」。**INSS 提供了一种解释该 trade-off 的可能机制**
  （defect 方向与 illumination 方向 overlap）—— 但本轮强调：**overlap 只能作为 exploratory 机制证据，
  不得据此宣称因果**，且**必须通过 specificity control**（见 §14/§17）。

## 3. Protocol freeze（摘要；完整定义见 `config/`）

| 项 | 冻结值 |
|---|---|
| 数据 | MVTec AD 5 类（bottle/cable/hazelnut/screw/grid）的 **train/good** 估计 subspace；test defect 仅用于**冻结之后**的只读 overlap diagnostic |
| Augmentations | **仅 4 个基础 photometric**：T1 gamma 0.8、T2 gamma 1.2、T3 brightness ×0.8、T4 brightness ×1.2（几何/resize/crop 不变；无 blur/noise/color-jitter/spatial） |
| 采样 | N = **min(64, n_train_good)**（**基于 smoke 耗时在读取任何统计量前决定**：实测 0.12 s/image，全量 15 units 仅 103 GPU·s，故取 64）；primary seed = 0；stability seeds = {0,1,2} |
| 特征 | `wide_resnet50_2` 的 **layer2 / layer3**，**pooled / pre-concat** 空间（与 `generate_embedding` 输入一致）；layer2 = 32×32 = 1024 位置/图，layer3 = 16×16 = 256 位置/图 |
| paired difference | `ΔF_l,k(x,u) = F_l(T_k(x))[u] − F_l(x)[u]` —— **同一图、同一空间位置严格对应** |
| PCA | **PRIMARY_centered**（任务书 §8：先 center 再 PCA）与 **SECONDARY_uncentered**（= 9A `nuisance_basis` 定义）**都预注册**；K grid = {1,2,4,8,16,32}；`eigh` 精确解；协方差**在线累加** `sum_d`/`sum_ddT`（float64），逐 perturbation 分别累加 |
| 指标 | A: EV@K / K80；B: gamma(T1+T2) vs brightness(T3+T4) 子空间，K=8，`mean cos²(principal angles) = ‖UᵀV‖_F²/K`；C: 逐类 + 跨类子空间相似度；D: seed 稳定性；E: defect overlap `‖P d‖²/‖d‖²` + 100 个随机子空间对照 |
| 判据 | 见 `config/analysis_plan.json` 的 `verdicts`（CASE A/B/C/D），**不允许事后修改** |

**复用的历史资产（P0）**：9A 已有 `results/experiment_9a_screening/cache/nuisance_basis_<cat>_seed0_k16_*.npz`
（bottle/cable，`U2 (512,16)` / `U3 (1024,16)`）与 9B 的 `illum_sensitivity_*.npz`；其 **illumination level 是 0.7/1.3**，
与本轮冻结的 **0.8/1.2 不同** → **不作为本轮 subspace**（协议不同），仅作历史参照。
本轮的 covariance 线上累加方式沿用 9A `estimate_nuisance_basis` 的实现模式（并扩展为逐 perturbation 分离，以支持 B 的 gamma/brightness 对照）。

## 4. Data & 实际采样

每类实际 sampled filenames 落盘于 `raw/<cat>/seed_<s>/meta.json` 的 `sampled_filenames` +
`sampled_filenames_sha256`；**全部 15 units 的 n_used = 64** ✓（5 类 × 3 seeds）。
`n_train_good`：bottle 209 / cable 224 / hazelnut 391 / screw 320 / grid 264（均 ≥64）。

## 5. Feature extraction（成本）

15 units（5 类 × 3 seeds）× 64 图 × 5 variants（clean + T1..T4）= **4,800 次 backbone 前向**，
实测 **103.0 GPU·s（1.72 GPU-min）**，墙钟 **2:08**，**peak VRAM ≈ 0.5–1 GB**，failed = 0；
并发 4（smoke 后确认远低于 18GB，无需升 5）。

---

# 结果

## 6. Primary A — Low-dimensionality（seed0；PRIMARY_centered）

| cat / layer | EV@1 | EV@2 | EV@4 | **EV@8** | EV@16 | EV@32 | **K80** |
|---|---|---|---|---|---|---|---|
| bottle / layer2 | 0.323 | 0.457 | 0.585 | **0.698** | 0.787 | 0.873 | 18 |
| bottle / layer3 | 0.172 | 0.327 | 0.448 | 0.555 | 0.668 | 0.776 | 38 |
| cable / layer2 | 0.258 | 0.369 | 0.471 | 0.564 | 0.675 | 0.790 | 34 |
| cable / layer3 | 0.067 | 0.131 | 0.233 | **0.316** | 0.438 | 0.589 | 79 |
| hazelnut / layer2 | 0.267 | 0.379 | 0.487 | 0.562 | 0.657 | 0.774 | 37 |
| hazelnut / layer3 | 0.071 | 0.139 | 0.231 | 0.352 | 0.474 | 0.609 | 75 |
| screw / layer2 | 0.478 | 0.534 | 0.614 | 0.682 | 0.763 | 0.852 | 22 |
| screw / layer3 | 0.152 | 0.255 | 0.400 | 0.483 | 0.591 | 0.719 | 51 |
| **grid / layer2** | 0.754 | 0.774 | 0.798 | **0.821** | 0.853 | 0.895 | **5** |
| grid / layer3 | 0.249 | 0.451 | 0.746 | **0.784** | 0.841 | 0.882 | 10 |

**读数**：**EV@8 ≥ 0.70 只出现在 2–3/10 个 category×layer**（grid 两层、bottle/layer2 0.698 差一点）
→ **不满足 CASE A 的「大多数」条件**。K80 跨度很大：**layer2 = 5–37**，**layer3 = 10–79**。
**layer2 全部优于同类别 layer3**（0.698>0.555、0.564>0.316、0.562>0.352、0.682>0.483、0.821>0.784）→
illumination response 在**较浅的 layer2 更集中**。
`SECONDARY_uncentered`（9A 定义，含平均响应方向）与本表**数值几乎相同**（差异 ≤0.007）→
**去掉/保留平均 illumination 方向不改变结论**（caveat：center 会移除平均响应方向，本轮两者都报，结论一致）。

## 7. Primary B — Cross-augmentation stability（gamma vs brightness，K=8，seed0）

| cat | layer2 | layer3 |
|---|---|---|
| bottle | 0.605 | 0.446 |
| cable | 0.606 | 0.380 |
| grid | **0.958** | 0.653 |
| hazelnut | **0.924** | 0.753 |
| screw | 0.831 | 0.724 |

**读数**：**8/10 ≥ 0.60**；**layer2 全部 ≥ 0.60（5/5）**，layer3 为 3/5（cable 0.380 最低）。
→ **gamma 与 brightness 确实激活高度相似的特征方向（尤其在 layer2）** ✓ 满足 CASE A 的该条门槛（多数）。

## 8. Primary C — Cross-category structure（K=8，seed0，PRIMARY）

跨类子空间相似度区间为 **0.335–0.686**（layer2：cable-grid 0.335 最低、screw-grid 0.686 最高；
layer3：0.459–0.656）。→ **部分共享、无通用子空间** → 属于 pre-registered 的
**Case 1：category-specific illumination subspace**（而不是 Case 2 shared / Case 3 unstable）。
热图见 `figures/fig3_cross_category_layer{2,3}.png`。

## 9. Diagnostic D — Sampling-seed stability（seeds {0,1,2}）

| cat / layer | EV@8 (seed0/1/2) | K80 | mean pairwise subspace sim | min |
|---|---|---|---|---|
| bottle / layer2 | 0.698/0.699/0.700 | 18/18/18 | **0.9996** | 0.9993 |
| bottle / layer3 | 0.555/0.556/0.555 | 38/38/38 | 0.9995 | 0.9994 |
| cable / layer2 | 0.564/0.566/0.564 | 34/34/35 | 0.9835 | 0.9755 |
| cable / layer3 | 0.316/0.315/0.315 | 79/79/79 | 0.9790 | 0.9737 |
| hazelnut / layer2 | 0.562/0.559/0.562 | 37/38/37 | 0.9929 | 0.9918 |
| hazelnut / layer3 | 0.352/0.345/0.349 | 75/76/75 | 0.9948 | 0.9934 |
| screw / layer2 | 0.682/0.683/0.689 | 22/22/21 | 0.9051 | 0.8662 |
| screw / layer3 | 0.483/0.485/0.491 | 51/51/50 | 0.8658 | 0.8386 |
| grid / layer2 | 0.821/0.814/0.816 | 5/6/6 | 0.9809 | 0.9764 |
| grid / layer3 | 0.784/0.778/0.785 | 10/11/10 | 0.8408 | 0.7856 |

**读数**：**seed 稳定性极好** —— EV@8 逐 seed 变化 ≤0.007，K80 几乎不变，子空间相似度 **0.84–0.9996**
（最低是 grid/layer3 0.841 与 screw 0.87/0.91，也属高稳定）。→ **不存在「PCA 偶然性」问题** ✓

## 10. Diagnostic E — Defect overlap vs random-subspace control（冻结 K=8 subspace）

定义：`d = F_patch − μ_normal(category, layer)`，`overlap = ‖Uᵀd‖²/‖d‖²`（`μ_normal` 来自 train/good）；
random control = 100 个同维（K=8，ambient C）随机正交子空间。

| cat / layer | **good（normal 对照）** | defect 均值 | **contrast（defect − good）** | random 子空间均值 | percentile |
|---|---|---|---|---|---|
| bottle / layer2 | 0.7958 | 0.7809 | **−0.0149** | 0.0159 | 100 |
| bottle / layer3 | 0.5958 | 0.5709 | −0.0249 | 0.0077 | 100 |
| cable / layer2 | 0.6775 | 0.6733 | −0.0042 | 0.0152 | 100 |
| cable / layer3 | 0.5528 | 0.5473 | −0.0055 | 0.0078 | 100 |
| grid / layer2 | 0.9641 | 0.9551 | −0.0090 | 0.0167 | 100 |
| grid / layer3 | 0.9515 | 0.9443 | −0.0072 | 0.0080 | 100 |
| hazelnut / layer2 | 0.7866 | 0.7714 | −0.0152 | 0.0153 | 100 |
| hazelnut / layer3 | 0.6170 | 0.6008 | −0.0161 | 0.0078 | 100 |
| screw / layer2 | 0.7881 | 0.7904 | **+0.0023** | 0.0156 | 100 |
| screw / layer3 | 0.5406 | 0.5394 | −0.0012 | 0.0077 | 100 |

**读数（本轮最重要的 negative 结果）**
1. **overlap 远高于随机子空间**（0.32–0.96 vs ≈0.008–0.017，percentile = 100，z ≈ 96–600）——
   即 illumination subspace 确实落在「patch 相对 normal centroid 的偏差」的主方向上。
2. **但这个 elevation 完全不是 defect 特异的**：**normal（test/good）对照的 overlap 与 defect 几乎相同**，
   9/10 个 category×layer 的 `contrast = defect − good` 落在 **±0.02** 内（唯一 1 个 > 0.02 的也没有），
   多个类别 defect **略低于** good。
3. 物理解释（与估计方式自洽）：illumination subspace 由 illumination 引起的 ΔF 的**高方差方向**估计而来，
   而 `d = F − μ_normal` 的主要能量本来就在这些高方差方向里 —— 对 normal 与 defect **同样成立**。
   → 因此 **「overlap 高」不能作为 defect entanglement 的证据**；本轮的 specificity control **未通过**。

**关于统计的诚实 caveat**：E 的 percentile/z 是在 **patch 聚合**层面、对 **random-subspace null** 计算的
（patch 之间不独立 → 该 p 值对「跨图像总体」而言是 anti-conservative）。因此本报告**只把它当作 effect-size /
percentile 陈述**，不当作总体显著性检验；关键判读依赖 **defect-vs-good 内部对照**（上表第 3 列）。

## 11. Verdict（按冻结判据机械判定）

| CASE | 判据（`config/analysis_plan.json`，冻结） | 实际 | 结论 |
|---|---|---|---|
| **A** STRONG | 大多数 category×layer `EV@8 ≥ 0.70` **且** sim ≥ 0.60 **且** seed 稳定 **且** defect overlap 相对 random 有 elevation | EV@8 ≥ 0.70 仅 **2/10**（grid l2/l3；bottle l2 0.698 差一点）；sim 8/10 ✓；seed 稳定 ✓✓；overlap 相对 random 有 elevation **但非 defect 特异** | **❌ 不成立** |
| **B** CATEGORY-SPECIFIC | 类内 EV@8 较高 **且** aug stability 较高 **且** cross-category 低 | **layer2**：EV@8 0.56–0.82、sim 5/5 ≥0.60、cross-cat 0.34–0.69（低-中）；layer3 明显更弱 | **✅ 成立（以 layer2 为主）** |
| **C** LOW-DIM, NO DEFECT LINK | subspace 稳定低维 **且** defect overlap 与 random 无明显差异 | subspace 稳定（seed sim ≥0.84）；defect overlap 相对 random **显著高**（字面判据不满足），但 **defect vs normal 无差异（非特异）** → **defect link 实质不成立** | **⚠️ 内容成立（机制叙事被否决）** |
| **D** NO STABLE SUBSPACE | EV@8 普遍低 **且** sim 低 **且** seed 稳定差 | EV@8 0.32–0.82（非普遍低）；sim 8/10 ≥0.60；seed sim 0.84–0.9996 | **❌ 不成立** |

> ### 最终：**`B — CATEGORY-SPECIFIC FEASIBILITY → recommend category-specific INSS Mini`**
> **并附加强制 caveat（CASE C 的机制限制）**：本轮 **defect-overlap diagnostic 未通过 specificity 检验**
> （normal 与 defect overlap 相同），因此**不得**把「defect directions 与 illumination subspace 纠缠」
> 作为机制结论或论文卖点；INSS 若继续，只能以**纯粹的 nuisance-suppression / robustness** 为动机。

## 12. Limitations（必须与结论一起读）

1. **仅 5 类**（MVTec 子集），N=64/类、sampling seed 3 个；不构成跨数据集证据。
2. **仅 4 个 photometric 扰动（±0.8/±1.2）**。历史 9A/9B 用的是 **0.7/1.3**，两者**协议不同**，本报告未混用；
   强度依赖（subspace 是否随 level 变化）**未测**，且按冻结规则**不允许**为好看而新增 level。
3. **layer3 的 paired difference 在 pooled(16×16) 空间**计算（与历史 9A/9B `_pooled_features` 一致）；
   `generate_embedding` 里的 layer3 是在 concat 前被**上采样**到 32×32，二者不是同一空间（上采样为线性算子，
   不改变通道协方差结构，但严格来说本轮的 layer3 subspace 属 **pre-concat pooled** 定义）。
4. **E 的 p 值**为 patch 聚合 + random-subspace null（见 §10 caveat），**不是**跨图像总体检验；
   判读依赖 defect-vs-good 内部对照。
5. **PRIMARY(centered) 与 SECONDARY(uncentered) 数值几乎一致**（≤0.007），说明结论不依赖 center 与否；
   但 center 会移除平均 illumination 响应方向，这在解释「subspace 覆盖了多少 illumination 效应」时需说明。
6. 本轮**不涉及任何模型/scoring**：只用 backbone 特征做几何分析，`AUROC/AUPRO` 等**未被使用**（符合 Probe 定位）。

## 13. Paper implication

- **支持**：「illumination 引起的 normal feature 变化在**较浅层（layer2）**相对集中，且 gamma 与 brightness
  激活**相似方向**，在给定 sampling 下**高度可复现**」——这为「用 normal-only 数据估计 nuisance subspace」
  提供了**可行的几何前提**（category-specific，非通用）。
- **否决**：「该 subspace 与 defect-sensitive 方向**特异性地**纠缠」——本轮 specificity control 失败
  （normal 与 defect overlap 相同）。因此**不能**用 INSS 解释 α-IN 的 robustness–preservation trade-off；
  解释 trade-off 需要**另一条**机制论证。
- 对方法候选的意义：INSS 即便继续，其价值定位只能是 **robustness/nuisance suppression**（与 preservation 增益无关），
  因此**优先级低于**继续打磨 Adaptive B2，或转向 pre-registered 的下一候选
  （Exp12 的建议：**Dual-Path Original + Robust Feature**）。

## 14. Next step（**仅建议；本轮到此 STOP**）

1. **不自动启动 INSS 正式实验**（按 §16 禁止事项 9/10 与 §24 STOP 点）。
2. 若要继续 INSS：只允许 **category-specific、layer2-only、K∈{8,16,32}** 的**极小** Mini
   （动机 = nuisance suppression），且**不得**声称 defect entanglement。
3. **更推荐**：把 **Adaptive B2（12/15）** 作为当前最佳候选提交人工决策；方法侧换
   **Dual-Path Original + Robust Feature** 作为下一条 pre-registered 路线。
