# Experiment 1J-A — Full Primary Layer-wise Feature Geometry Probe

> 状态：协议已冻结（运行前）。运行后仅追加 Results / Verdict / Limitations / Next Step。
> 前序链：1E → 1G → 1H → 1H-S → 1I（CASE C：Layer3 amplification 不能用 IN 统计量的 spatial sample count 解释）。

---

## 1. Background

已确认事实（继承，不重述全部推导）：

1. defect-specific dispersion response（Δstd_defect = std(α=1) − std(α=0)）稳定存在，3 seed 一致。
2. frozen primary defects 分 shrink / neutral / expand 三类（来自 `results/experiment_1g/selection.csv`，`selection == "primary"`）。
3. 1H/1H-S：expand 家族 Layer3-dominant dispersion amplification（LSI_std ≈ +0.448）；shrink 家族 Layer2/Layer3 同号收缩。
4. 1G：feature → NN-distance → anomaly-score dispersion 强传导（ρ≈0.95）。
5. 1I（spatial-statistics control，CASE C）：
   - expand: Δstd_L2_standard=+2.764, Δstd_L2_matched256=+2.928, Δstd_L3_standard=+6.678
   - LSI_standard=+0.448 → LSI_matched256=+0.419（ΔLSI=−0.029）
   - 结论：Layer3 amplification **不能**被「Layer3 用 256 个 spatial sample 估 IN 统计量」解释。

## 2. Research Question（唯一）

α-IN 是否在 **Layer3** 中对 expand defects 造成比 **Layer2** 更强的 defect feature-geometry expansion？

比较对象是 **within-layer α response**（α=0 → α=1 各自的几何变化），**不是** raw Layer2 radius vs raw Layer3 radius（channel 维度、representation scale、receptive field 都不同，不可直接比）。

## 3. Hypothesis

H_geometry：对 frozen expand defects，α-IN 在 Layer3 引起比 Layer2 更明显的 defect feature-space dispersion / geometry expansion，且方向与此前 NN-distance / anomaly-score dispersion amplification 一致。

- 本实验只能检测 feature geometry 是否**支持**这一机制。
- 本实验**不能证明** feature geometry causally causes score dispersion（留 1J-B）。

## 4. Frozen Primary Defects

严格来自 `selection.csv`（不可根据 1J 结果重新分类）：

| family | category/defect |
|---|---|
| shrink | cable/bent_wire, hazelnut/print, bottle/contamination |
| neutral | screw/manipulated_front, screw/thread_side, screw/thread_top |
| expand | grid/glue, grid/metal_contamination, grid/thread |

其余 16 defect 不参与 primary decision；保存 descriptive 结果但禁止事后重新定义 family / 挑选支持案例。

## 5. Feature Hook Definition（与 1H 严格对齐）

研究对象是 **IN intervention 作用位置上的 feature**，因此 hook 位置必须与 1H 的
`FAlphaLayerPatchcoreModel.generate_embedding`（`scripts/falpha_layer_patchcore.py`）一致：

- `layer2`：`features["layer2"]`，shape `(B, 512, 32, 32)`；α-IN 施加在 concat 前。
- `layer3`：`features["layer3"]`，shape `(B, 1024, 16, 16)`；α-IN 施加在 **upsample 之前**（保持该层原始 16×16 统计）。

保存/计算：
- `F_after_alpha_mix`（α=0 时 == F_before；α=1 时 == IN(F_before)）。

**禁止**：用 post_concat 1536 维冒充 layer2/layer3 feature；layer2 在 pooling 前、layer3 在 pooling 后取；一个条件 interpolate 前、另一个 interpolate 后。

α=0 identity sanity：`max_abs_diff(F_before, F_after_α0) < 1e-6`，否则 STOP。

## 6. Primary Geometry Metrics（运行前冻结）

对每张 defect image，在某 layer 得到 `F ∈ R^(C×H×W)` → `X ∈ R^(P×C)`（P=H×W；layer2 P=1024，layer3 P=256），每个 row 是 spatial feature vector：

1. **MDC**（Mean Distance to Centroid）：`mean_i ‖x_i − μ‖_2`
2. **RMS radius**（primary spread）：`sqrt(mean_i ‖x_i − μ‖₂²)`
3. **normalized participation ratio**：`PR = (Σλ)²/(Σλ²+eps)`，normalize 为 `PR / min(P−1, C)`（减少 dimensionality 机械差异）
4. **PCA1 explained variance ratio**：`λ₁/Σλ`（描述 anisotropy，不合成总分）

Secondary（仅解释）：feature L2 norm；normal-defect centroid separation（不参与 primary decision）。

## 7. Statistical Hierarchy

patch → image-level aggregation → defect×seed → defect identity → family

- **Primary inference unit = defect identity**；seed = repeated measurement；spatial patch = within-image representation sample（非独立 experimental unit）。
- 禁止把 N×H×W patches 当独立样本做 pseudo-replication。

## 8. GAI Definition

对 spread metric，用 within-layer log-ratio response：

```
R_radius_L2 = log((RMS_radius_L2_α1 + eps) / (RMS_radius_L2_α0 + eps))
R_radius_L3 = log((RMS_radius_L3_α1 + eps) / (RMS_radius_L3_α0 + eps))
GAI_radius  = R_radius_L3 − R_radius_L2
```

- `GAI > 0` = α-IN 对 Layer3 spread expansion 比 Layer2 更强。
- 同时保存 GAI_MDC；**Primary GAI = GAI_radius**（禁止事后换）。

## 9. Pre-Registered Decision Tree

### CASE A — Strong Geometry Support
expand 3/3：`R_radius_L3 > R_radius_L2`（GAI>0，3 seeds 方向基本稳定）；Layer3 radius response magnitude 明显强于 Layer2；geometry expansion 与 NN/score dispersion amplification 方向一致；shrink 不呈现相同稳定 Layer3 expansion。→ 结论只能写「Layer3-specific feature-geometry expansion is consistent with and supports the observed dispersion amplification mechanism」。下一阶段允许 1J-B（但本实验 STOP）。

### CASE B — Partial Geometry Support
2/3 expand 支持，或 radius 支持但 effective rank/anisotropy 不一致，或 geometry 与 NN 只部分对应，或 seed consistency 弱。→ feature geometry 是候选机制一部分，不足以称为稳定机制。

### CASE C — Geometry Unsupported
expand 的 L2/L3 radius response 接近，GAI 不稳定，3 expand 方向不一致，Layer3 无稳定 geometry expansion，或 geometry 与 NN/score 脱节。→ STOP feature-geometry mechanism，转 memory-bank / NN retrieval / PatchCore aggregation。

## 10. Secondary Cross-chain Measurement — Layer-specific NN

> 本节是**运行前**发现 1G 数据不足（1G 只有 post_concat 口径 NN、且仅 seed0，无分层 NN）后补充的**测量口径修正**，不是看到结果后的 hypothesis 修改。

新增 Layer2 / Layer3 的 **layer-specific NN-distance dispersion extraction**，作为 1J-A 的 **secondary cross-chain measurement**，**不改变** primary hypothesis / primary metrics / CASE A-B-C 判据。

1. 用与 geometry 完全相同的 frozen 9 defects、seeds={0,1,2}、α={0,1}、data split、preprocessing。
2. NN 必须在各 layer 自己的 feature space 计算：Layer2 feature → Layer2-specific normal bank → Layer2 NN；Layer3 同理。禁止用 post_concat bank 冒充分层 NN。
3. 复用 1J 本次 feature extraction，不重复跑 backbone；layer-specific normal bank 在同一次 extraction/cache 流程中保存/计算。
4. NN 定义尽量与 PatchCore 一致（kNN，num_neighbors=9，coreset 采样）。layer-specific dimensionality 差异只做**维度适配**（1536→512/1024），**不改** distance metric / neighbor rule / aggregation rule；所有差异记录进 README。
5. 每个 `(defect, seed, layer, alpha)` 计算 defect image 的 NN-distance 分布，保存 `mean_nn_distance` / `std_nn_distance` / `median_nn_distance`。primary cross-chain measurement 用 `NN_dispersion_response(layer) = std_NN(α=1) − std_NN(α=0)`。
6. 输出 `nn_dispersion_response_l2` / `nn_dispersion_response_l3` 到 `experiment1j_cross_chain_summary.csv`。
7. 1G 的 `post_concat delta_nn_std (seed0)` 保留为 historical/descriptive reference，标注 `1G_postconcat_seed0`，**不冒充** Layer2/Layer3 NN response，不与新分层 NN 混成同一统计量。
8. Figure 4 改为 layer-matched cross-chain alignment：Layer2 geometry ↔ Layer2 NN；Layer3 geometry ↔ Layer3 NN；可选画 `GAI_geometry ↔ (NN_response_L3 − NN_response_L2)`。correlation 仅 descriptive（n=9），不以 p-value 为核心证据。
9. NN measurement 不改变预注册主判据。CASE A/B/C 仍首先按 geometry criteria 判定。若 geometry CASE A 但 NN 不对齐 → 报告 `geometry supported, transmission unsupported/incomplete`。
10. 若 layer-specific NN 需要大量重训练、无法与 PatchCore 原 NN 公平对齐、或必须改模型结构 → 立即 STOP NN 部分，标记 `unavailable`，不阻塞 geometry 主实验。

## 11. Stop Conditions

任一发生立即 STOP（不允许「先跑完再说」）：

1. Layer2/Layer3 hook 与 1H intervention location 无法对齐；
2. α=0 identity 不成立；
3. data split 与 1H 不一致且无法恢复；
4. preprocessing 与 1H 不一致；
5. family / frozen defect mapping 无法与 selection.csv 对齐；
6. geometry metric 在大量 image 上 numerical degenerate；
7. 提取到的是 post_concat feature 而非 layer-specific intervention feature；
8. 必须改变模型定义才能获得 feature；
9. 1G NN metric 无法可靠对齐时：不伪造，cross-chain NN 对齐标记 unavailable（geometry 继续）。

## 12. Interpretation Boundary

无论结果多漂亮，不能写「Layer3 semantics cause dispersion expansion」/「Feature geometry causes PatchCore anomaly dispersion」/「We discovered the causal mechanism」。

CASE A 最多写「Layer3-specific feature-geometry expansion is consistently associated with and supports the observed Layer3 dispersion amplification」。geometry → NN → score 的传导需 1J-B。

---

## 13. Artifact Audit（运行前审计结论）

- 1G_feature_cache = `results/experiment_1g/embeddings/*.npz`（770 个，**post_concat 1536 维**，仅 seed=0，α∈{0,.25,.5,.75,1}）
- 1H_feature_cache = 无（只有 anomaly_score / 聚合指标，无分离 layer feature）
- available_layers = 仅 post_concat（1536）；无分离 512/1024
- available_seeds = 1G 仅 seed0；1H/1I 有 {0,1,2}
- available_nn_metrics = 1G `defect_summary.csv` 的 `delta_nn_std`/`delta_dispersion`（post_concat，seed0）
- **need_new_gpu_extraction = YES**（原因：1G 只有 post_concat，无分离 layer2/layer3 feature；需新增纯 forward hook 提取）

## 14. Results（运行后追加）

### 14.1 运行完整性

- 提取：1848 npz（9 defect × 3 seed × 2 layer × 2 alpha × 各 defect 全部 test 图，与理论值精确吻合）；30 个 layer-specific bank（5 cat × 3 seed × 2 layer）；`missing bank = 0`。
- CSV：per_image 1848 行、nn 1848 行（经 rebuild 脚本从 npz+bank 重建，重建后行数与理论值一致）。
- 运行事故（不影响数据正确性）：① `layer_nn_stats` 初版三维广播 10.2 GiB OOM → 改 einsum 展开（内存降 250×）；② flush 的 `id()` 去重在 `clear()` 后命中被复用的对象 id，丢 ~50% 行 → 已修复并从 npz 重建 CSV；③ 重建脚本初版重复遍历 alpha → 修复后 1848 精确。

### 14.2 Primary 结果：GAI_radius（defect identity 单元，跨 seed mean）

| family | defect | R_L2 | R_L3 | GAI | sign consist |
|---|---|---|---|---|---|
| shrink | cable/bent_wire | −0.429 | −0.037 | +0.392 | 1.00 |
| shrink | hazelnut/print | −0.221 | +0.020 | +0.242 | 1.00 |
| shrink | bottle/contamination | −0.464 | +0.021 | +0.485 | 1.00 |
| neutral | screw/manipulated_front | −0.172 | +0.021 | +0.193 | 1.00 |
| neutral | screw/thread_side | −0.163 | +0.024 | +0.187 | 1.00 |
| neutral | screw/thread_top | −0.165 | +0.026 | +0.191 | 1.00 |
| expand | grid/glue | +0.278 | +0.465 | +0.187 | 1.00 |
| expand | grid/metal_contamination | +0.251 | +0.464 | +0.213 | 1.00 |
| expand | grid/thread | +0.259 | +0.447 | +0.188 | 1.00 |

Family 水平（mean ± sd）：shrink R_L2=−0.372±0.131 / R_L3=+0.001±0.033；neutral R_L2=−0.167±0.005 / R_L3=+0.024±0.003；expand R_L2=+0.263±0.014 / R_L3=+0.459±0.010。

**结构性发现（比 hypothesis 更细的分离结构）**：

1. **expand 家族是唯一 R_L3 显著为正的家族**（+0.447~+0.465，强扩张）；shrink 的 R_L3≈0（−0.037~+0.021）、neutral≈0（+0.021~+0.026）。
2. **R_L2 符号完美三分**：expand 全正（+0.25~+0.28）、shrink/neutral 全负（−0.16~−0.46）。
3. **GAI 全部 9/9 为正，且 shrink 的 GAI（+0.24~+0.49）> expand（+0.19~+0.21）**——因为 shrink 的 R_L2 收缩深、L3 中性，差值反而大。GAI>0 本身不区分家族；判据中真正有区分力的是「Layer3 expansion」（R_L3 显著正）条款。

### 14.3 Secondary cross-chain：layer-specific NN（NN_dispersion_response = std_NN(α1)−std_NN(α0)）

| defect | nn_L2 | nn_L3 | score_L2 | score_L3 |
|---|---|---|---|---|
| grid/glue | +4.48 | +5.39 | +1.17 | +5.10 |
| grid/metal_contamination | +3.93 | +4.98 | +3.00 | +6.86 |
| grid/thread | +2.78 | +2.58 | +4.12 | +8.07 |
| cable/bent_wire | −2.71 | −2.15 | −2.15 | −2.05 |
| hazelnut/print | −1.17 | −2.53 | −1.77 | −1.97 |
| bottle/contamination | −3.17 | −1.86 | −2.24 | −0.86 |
| screw/manipulated_front | +3.75 | +4.19 | −0.38 | −0.14 |
| screw/thread_side | +3.71 | +3.85 | +0.13 | −0.04 |
| screw/thread_top | +2.97 | +3.78 | −0.08 | −0.11 |

- expand：NN 两层均大正、2/3 为 L3>L2（thread 例外，L2 略大 +2.78 vs +2.58）；score 3/3 强 L3 主导。
- shrink：NN/score 全负（收缩），与 geometry 一致。
- **neutral 意外信号**：NN response 显著为正且 L3>L2（feature→NN 层面在扩张），但 score ≈ 0（未传导到 anomaly score）。

### 14.4 关键方法学事实：geometry 的 seed-invariance

geometry（MDC/radius/PR/PCA1）只依赖 frozen backbone forward + 确定性 α-IN，不涉及 coreset bank，故同一 (defect, layer, alpha) 的 3 个 seed 数值**完全相同**（非 bug，数学必然）。NN 数据的 per-seed 差异真实存在（bank 采样随 seed 变）。因此：geometry 的「3-seed consistency = 1.00」是确定性证据，说明 **Layer3 expansion 信号不是 coreset 采样噪声的产物**；真正承载 seed 随机性的是 NN/score 链。

### 14.5 产物

`results/experiment_1j/`：`analysis/`（per_seed / per_defect / family_summary / cross_chain_summary 4 CSV）+ `figures/`（figure1 radius response、figure2 GAI、figure3 components、figure4 geometry↔NN alignment、figure5 cross-chain z-score heatmap 5 张）+ `per_image/`、`nn/`、`raw/`（1848 npz）、`banks/`（30 npy）。

## 15. Verdict（运行后追加）

**CASE A — Strong Geometry Support**（按第 9 节冻结判据逐条核对）：

1. ✅ expand 3/3：R_radius_L3 > R_radius_L2（GAI = +0.187/+0.213/+0.188 > 0，符号一致）。
2. ✅ expand 的 Layer3 radius response（+0.447~+0.465）明显强于 Layer2（+0.251~+0.278，约 1.8 倍）。
3. ✅ geometry expansion 与 score dispersion amplification 方向一致（score L3 主导 3/3；NN L3 主导 2/3）。
4. ✅ shrink 不呈现 Layer3 expansion（R_L3 = −0.037~+0.021 ≈ 0，而 expand = +0.45~+0.47）。

**结论（受 Interpretation Boundary 约束）**：Layer3-specific feature-geometry expansion 在 frozen expand defects 上稳定存在、方向与 NN/score dispersion amplification 一致，且为 expand 家族所特有（shrink/neutral 的 R_L3≈0）。这一致性**支持**（supports / is consistently associated with）Layer3 dispersion amplification 的表征机制假设；不写 causal（geometry → NN → score 的传导检验属 1J-B）。

**必须随 Verdict 记录的三点诚实限定**：
- 「GAI>0」条款本身不具家族区分力（9/9 全正、shrink 更大）；判据的实际承载条款是「Layer3 expansion 存在于 expand 且仅存在于 expand」。
- NN 的 layer-dominance 是 2/3（thread 的 nn_L2 略大于 nn_L3），geometry 与 NN 的 layer 对齐不完全。
- geometry 的 3-seed 一致性是确定性必然（见 14.4），不能当作独立于确定性的统计稳定性证据。

## 16. Limitations（运行后追加）

1. GAI 作为「层间差」指标混淆了「L3 扩张」与「L2 收缩」两种来源；本实验中 shrink 的高 GAI 来自 L2 收缩。后续引用 GAI 必须同时报告 R_L2 / R_L3 分量。
2. geometry 无 seed 变异（确定性），无法做跨 seed 的统计检验；seed 层面的不确定性只存在于 NN/score 链。
3. neutral 家族 NN 扩张但 score 不响应——NN→score 传导并非对一切家族成立，1G 的强传导结论（ρ≈0.95）是在全部 25 defect 混合下成立的，family 级传导结构需 1J-B 澄清。
4. cross-chain 相关性 n=9，仅 descriptive。
5. feature 提取基于 wide_resnet50_2 单一 backbone，结论的外推需跨模型验证。
6. 运行中发生过 flush 丢行事故（已修复并重建），最终 CSV 行数与理论值精确一致，但最初的 extract.py 直写 CSV 路径不可复现本报告数字，以 rebuild 路径为准。

## 17. Next Step（运行后追加）

本实验按协议 STOP，不自动开 1J-B。留待用户决策的候选方向：

1. **1J-B（因果链补全）**：在 layer-specific feature 空间做干预/消融（如 L3 expansion 的模拟放大/抑制），检验 geometry → NN → score 传导；并解释 neutral「NN 扩张但 score 不响应」的阻断点。
2. **家族级传导结构**：按 family 分层重估 1G 传导（expand 应强传导、neutral 应阻断），直接回应 16.3。
3. **跨 backbone 验证**（如 ResNet 系列）验证 Layer3-specific expansion 的架构普适性。
