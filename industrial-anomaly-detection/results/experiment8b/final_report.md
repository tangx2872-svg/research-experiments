# Experiment 8B — Final Report (Defect–Illumination Feature Separability Probe)

- 日期：2026-10-07　论文阶段：Phase 5（Solution Selection）
- 冻结协议：`results/experiment8b/reference/protocol_freeze.json` sha256 `5f2bbd54cbc9964d…`
- 预注册修正：A1（S13 判据，见 experiments/experiment8b/README.md §9，早于任何 target result）
- 判定：**CASE_B**，Plan C = **HOLD**

## 1. 要验证的核心论断

> Defect sensitivity and illumination sensitivity may be heterogeneously distributed across
> pretrained feature dimensions, leaving a potentially selectable subspace that is
> simultaneously defect-sensitive and illumination-stable.

（hypothesis，不是结论；8B 只负责证伪或支持它。若成立 → Plan C；若不成立 → Plan A。）

## 2. 数据与参数（冻结）

- 表示：α=0 PatchCore pretrained（wide_resnet50_2, layer2+layer3, layer3 不 upsample）；
  复现 `feature_extractor → feature_pooler(AvgPool2d(3,1,1)) → concat` 的历史约定。
- 15 cells = 5 categories × 3 seeds；split 与历史一致（val 20 / train 其余）。
- D_c：defect-region（mask overlap>0.5，空则 top-1 overlap）per-image patch mean →
  per-image equal weight → /σ_eff；I_c：paired illumination（12 conditions，
  brightness/gamma × {0.5,0.7,0.9,1.1,1.3,1.5}）same-image same-location mean|Δf| → /σ_eff。
- σ_eff = sqrt(σ²+ε²)，ε = 1e-3·median(σ)；selection 在 layer 内做（s = z_D − z_I，ratio 25%/50%）。

## 3. 实验过程

- Phase A（GPU 特征提取，无 fit/无 bank）：15/15 cells，累计 524 s；
  defect 图 1203 张；fallback（无 overlap>0.5 格点，改用 top-1）l2 = 24、l3 = 393。
- Phase B（score-level probe，复用 5A-H 完整评估路径 + channel mask）：420/420 units OK，单 unit 累计 19.67 GPU·hour（sequential 口径）；
  wall-clock ≈ 6.64 h（3 workers 并行，10:53 → 17:31）。**预注册估计为 ~2 h，实际偏高 3.3×**，
  原因：hazelnut/screw 单 unit 达 260 s（历史口径 46–111 s）且 CPU 已过载（load 15/14），
  追加 worker 无收益（已实测，未改协议）。
- S13（新提取 vs 1J 历史资产）：240/240 通过（corr ≥ 0.9999999）。

## 4. 结果

### 4.1 事实 — metric level（channel 空间，oracle/diagnostic）

- 每 cell 1536 channel（layer2 512 + layer3 1024），共 23040 行。
- PROPOSED25：**15/15** cells `z_D ≥ +2`、**15/15** cells `z_I ≤ −2`、
  directional 15/15；mean z_D = 18.64、mean z_I = -9.94。
- 子集均值（15 cells 平均）：D̄_sel/D̄_full = 2.36×，Ī_sel/Ī_full = 0.65×。
- 相关性：pooled Spearman(|D|, I) = 0.0666（layer2 0.1509，layer3 0.0231）。
- Pareto（layer 内非支配集）：平均占比 layer2 0.0161、layer3 0.0063（≈ i.i.d. 的 ln(n) 期望，说明 frontier 本身不是特殊结构）。
- 分布：D_abs 重尾（p50 0.46 / p99 9.39 / max 251.8，top 1% channel 占 34% 质量）；I_mean 尾部温和（p99 0.56 / max 2.79）。
- 变体对照（ratio 25%）：D-ONLY 的 D̄ = 2.66× 但 Ī = 1.00×；I-ONLY 的 Ī = 0.42× 但 D̄ = 1.14×；PARETO 作为 mask 失效（D̄ 0.15×、Ī 4.08×，因其为极值点集合）。
- defect-type control（§15）：150 行，selected_better = 100% — 每个 (category, seed, layer, defect type) 上选中的 channel 平均 |D| 都不低于全体 channel。

### 4.2 事实 — score level（Phase B probe，oracle-diagnostic）

- S4 等价性：FULL mask 与历史 α=0（5A-H config_B0）逐图对比 183 rows/cell，max|Δscore| = 0 → **PASS**（全 4203 行）。
- PROPOSED25 vs FULL：win 5/15；
  mean Δd′ = -0.2285, mean Δ|Δz| = -0.1545。
- PROPOSED25 vs RANDOM-mean(10 draws)：win 12/15；
  mean Δd′ = 0.2442, mean Δ|Δz| = -0.2377。
- C6b = FAIL（(i) False / (ii) True / (iii) False / (iv) True）。

#### 各 mask 的 score-level 汇总（15 cells 平均）

| mask | n | d′ | \|Δz\| | Δd′ vs FULL | Δ\|Δz\| vs FULL | win vs FULL | win vs RANDmean |
| --- | --- | --- | --- | --- | --- | --- | --- |
| PROPOSED50 | 15 | 5.1744 | 0.1495 | +0.1892 | -0.1610 | 11/15 | 15/15 |
| D-ONLY50 | 15 | 5.0142 | 0.3467 | +0.0290 | +0.0362 | 2/15 | 6/15 |
| I-ONLY50 | 15 | 4.8489 | 0.1393 | -0.1363 | -0.1712 | 6/15 | 12/15 |
| D-ONLY25 | 15 | 4.8351 | 0.2899 | -0.1501 | -0.0206 | 0/15 | 5/15 |
| PROPOSED25 | 15 | 4.7567 | 0.1560 | -0.2285 | -0.1545 | 5/15 | 12/15 |
| I-ONLY25 | 15 | 4.4108 | 0.1438 | -0.5744 | -0.1667 | 5/15 | 6/15 |
| PARETO25 | 15 | 1.1712 | 3.9865 | -3.8141 | +3.6760 | 0/15 | 0/15 |
### 4.2b 事实 — score-level 的 category 结构（负迁移）

| ratio | category | Δd′ vs FULL | Δ\|Δz\| vs FULL | cells d′-loss |
| --- | --- | --- | --- | --- |
| PROPOSED25 | bottle | -0.3777 | -0.3206 | 3/3 |
| PROPOSED25 | cable | -0.9089 | -0.1181 | 3/3 |
| PROPOSED25 | grid | -0.0848 | -0.1574 | 3/3 |
| PROPOSED25 | hazelnut | +0.0405 | -0.1074 | 1/3 |
| PROPOSED25 | screw | +0.1883 | -0.0688 | 0/3 |
| PROPOSED50 | bottle | +0.5579 | -0.2781 | 0/3 |
| PROPOSED50 | cable | -0.2741 | -0.1482 | 3/3 |
| PROPOSED50 | grid | +0.1926 | -0.1576 | 0/3 |
| PROPOSED50 | hazelnut | +0.1519 | -0.0977 | 1/3 |
| PROPOSED50 | screw | +0.3176 | -0.1232 | 0/3 |

- **cable 是唯一系统性负迁移 category**（两个 ratio 下 d′ 均下降）；其余四类在 50% 上 d′ 均上升。
- |Δz| 在 15/15 cells 上从不因选择而变差（两个 ratio 均成立）。

### 4.2c 事实 — 与历史干预族在同一 harness 下的对比

8B 的 FULL mask 与 7A-O 的 `B0_original` 在同一 15 cells 上数值一致（d′ 4.985 vs 4.9852，|Δz| 0.3105 vs 0.3105），因此下表可严格跨实验比较：

| config | 来源 | d′ | \|Δz\| | 说明 |
| --- | --- | --- | --- | --- |
| B0_original (α=0) | 7A-O | 4.9852 | 0.3105 | 无干预参考 |
| B2_uniform α=0.4009 | 7A-O | 4.7940 | 0.2134 | 历史强 baseline（CASE_C 的对照） |
| B1_fixed α=0.5 | 7A-O | 4.6670 | 0.2082 | 历史 best fixed |
| C1_l2_030_l3_050 | 7A-O Q2 | 5.2508 | 0.1616 | layer-selective（对 α=0 亦为 Pareto 改善） |
| A3_lam050 | 7A-O | 5.3552 | 0.2753 | residual 族最佳 mean |
| **8B PROPOSED25 (primary)** | 8B | 4.7567 | 0.1560 | 稳健性最好之一，但 d′ < α=0 |
| **8B PROPOSED50 (secondary)** | 8B | 5.1744 | 0.1495 | d′ > α=0 且 \|Δz\| < α=0 |

- 结论（事实层）：channel-selection 在 **50% ratio** 上给出的点 (5.1744, 0.1495) 严格支配 α=0，
  但并未严格支配历史上已经存在的 C1_l2_030_l3_050 (5.2508, 0.1616) —— 两者互不支配。
  因此“选择 channel 能推动 frontier”这一说法**不能**说成第一次；
  它能说的是：在不做任何 representation intervention 的前提下，仅靠 channel 子集也能达到同一 frontier 区域。

### 4.3 事实 — 稳定性与 layer control（metric level）

- category（seeds winning）：['bottle:3/3', 'cable:3/3', 'grid:3/3', 'hazelnut:3/3', 'screw:3/3']；pooled-3-seed ranking 在 5/5 categories 仍 win。
- seed（categories winning）：['0:5/5', '1:5/5', '2:5/5']。
- L2-only / L3-only 各自 win：15/15、15/15（C7 = True）。
- 剔除 degenerate channel（σ<1e-6，43/23040）后：z_D≥2 15/15、z_I≤−2 15/15（结论不依赖退化 channel）。

## 5. Sanity（S1–S13）


| ID | status | detail |
| --- | --- | --- |
| S1_feature_shape | PASS | layer2=(512,32,32), layer3=(1024,16,16) 全部 cell 一致（extract 侧 assert） |
| S2_channel_ordering | PASS | 与 1J 逐 channel 对齐（S13 corr≈1.0, 同 index 同 channel） |
| S3_layer_identity | PASS | layer3 未 upsample；与 1J 命名一致（同 pooler 约定） |
| S4_original_score_equivalence | PASS | FULL mask vs 历史 α=0（5A-H config_B0）逐图逐条件：4203 行，max_rel_diff = 0.0e+00 |
| S5_image_mask_alignment | PASS | top-5% patch-NN 格点与 mask overlap>0.5 的重叠 z>3：n=60/60 rows(15 cells x 2 img x 2 layers)，z_min=4.51 > 3，但绝对 overlap 最小仅 0.047（screw/manipulated_front layer3），已记录 |
| S6_paired_illumination_alignment | PASS | paired identity 条件 Δ≡0（15/15 cells，layer2+layer3，逐位相等） |
| S7_no_test_leakage | PASS | ranking=oracle-diagnostic；score-level 结果标 oracle_probe_only=true |
| S8_D_finite | PASS | n=23040 |
| S9_I_finite | PASS | I_mean/brightness/gamma 全部有限 |
| S10_scale_normalization_valid | PASS | sigma_eff=sqrt(sigma^2+eps^2)，eps=1e-3*median(sigma) 已冻结 |
| S11_coverage_complete | PASS | cells=15/15, channel rows=23040 |
| S12_random_baseline_reproducible | PASS | 同 seed null 逐位一致 |
| S13_source_consistency | PASS | n=240 全部 corr>=0.9999 且 max|Δ|<=0.05 |

primary sanity FAIL count (excl. PENDING) = 0

（说明：S5 在首版分析脚本中因键名取值错误曾被判 FAIL；
经 `scripts/experiment8b_s5_alignment.py` 全 15 cells 复算后为 PASS，证据见 `sanity/S5_alignment.csv`。属分析脚本 bug，不涉及数据/协议。）


## 6. 异常情况 / 修正记录（全部保留）

- **运行时偏差（如实上报）**：Phase B 实际 6.64 h wall-clock，预注册估计 ~2 h（3.3×）。
  原因见 §3；未因此改动 unit 数、ratio 或 mask 集，运行期间 0 failed、未中断。
- **随机基线抽样噪声**：score-level 的 RANDOM 对照只有 10 draws/ratio（预注册值），
  其 mean 的标准误不可忽略；这是 secondary（50%）信号需要 confirmatory 实验的原因之一。
- **A1（S13 判据）**：首次 S13 FAIL（corr 0.894）来自 8B 提取器漏掉 `feature_pooler`；
  修正后 corr ≈ 1.0；但因 GPU batch 组成相关非确定性（实测 6.5e-3）bit-exact 不可达，
  判据改为数值容差。触发早于任何 8B 目标结果，且不触碰 D/I 定义与 CASE 阈值。
- layer3 mask overlap>0.5 覆盖不足：393/1203 张 defect 图触发 top-1 fallback（layer2 24/1203）；已在 defect_type_summary 中逐项报告。
- PARETO 作为 selection mask 与预期（§11“优先 Pareto ranking”）不符：
  它是一个极值点集合，不是“好区域”，因此在 score-level 上必然劣于 FULL。已如实报告，
  未回改协议、未把它换成别的 mask 重跑。
- D_abs 重尾（top 1% channel 占 34% 质量）⇒ “D̄ ×2.36” 不能解读为校准后的效应量；
  真正的证据是 I 轴下降、ρ ≈ 0、两 layer 同时成立以及 score-level probe。
- PARETO 作为 mask 在 score level 上崩溃（d′ 1.17、|Δz| 3.99）与其 metric-level 性质一致：
  非支配集是极值点集合（平均 D̄ 0.15×、Ī 4.08×），把 frontier 当“好区域”使用是错误用法。

## 7. 分析（事实 / 解释 / 假设分离）

**事实**：
1. metric level：15/15 cells 同时 D↑（z_D>0）与 I↓（z_I<0）；
   pooled Spearman(|D|, I) = 0.0666（近似无关）。
2. Layer identity 不能解释该现象：L2-only 与 L3-only 各自独立 win（C7 = True）。
3. Score level：见 §4.2（probe 的 d′ / |Δz| 与 FULL / RANDOM 对照）。

**解释**（基于结果的推断，仍属解释）：
- D 与 I 在 pretrained channel 上近似独立 ⇒ 存在可选择的 channel 子空间；
  但它在 PatchCore score 上是否可变现，由 C6b 决定（见 §8）。
- Pareto frontier 大小 ≈ i.i.d. 的 ln(n) 期望 ⇒ 该 2D 结构本身没有额外的“特殊点聚集”。

**假设**（未被实验验证，不得写成结论）：
- 若 score-level 无法变现，最可能的机制是 PatchCore 的 kNN score 由少数高范数/高方差
  channel 主导，与 D/I 的 channel 级排序不一致。
- normal-only proxy（8B-B）能否近似识别 desirable channel，本轮未做（协议禁止）。

**预注册 secondary 信号（不参与 CASE，但必须记录）**：
- PROPOSED50 相对 FULL 的 15-cell 均值为 Δd′ = +0.1892、Δ|Δz| = −0.1610，逐 cell 胜 11/15，
  对 matched-size random（10 draws）15/15 胜，且 Δd′ < 0 的 cell 只在 cable（3/3）与 hazelnut（1/3）。
- 该信号**不改变判定**：C6b 的 (i)(iii) 冻结在 primary ratio 25%，此处不得事后改用 50% 升级 verdict。
- 用 7A-O 的历史 mean-based tier 规则独立复评 PROPOSED50（对照 Uniform 0.4009、要求 negTr = 0）：
  Δd′ = +0.380、Δ|Δz| = −0.064 满足 tier A 幅度，但 cable 造成 1 个 category 负迁移 → 只有 tier B。
  即两套独立判据都指向“弱但真实（CASE_B）”，这提高了 CASE_B 结论的可信度。

## 8. 出口条件判断（机械执行）

| 条件 | 结果 |
| --- | --- |
| C1 | True |
| C2 | True |
| C3 | True |
| C4 | True |
| C5 | True |
| C6a | True |
| C6b | False |
| C7 | True |

→ **CASE_B**，Plan C = **HOLD**（判定顺序见 experiment8b/README.md §4.2）。

**Negative results（必须记录）**：
- 预注册的“prioritize Pareto ranking”方案作为 selection mask **失败**（0/15 cells 胜 FULL 或 RANDOM，
  d′ 1.1712、|Δz| 3.9865），失败原因已定位：非支配集是极值点集合。
- D-ONLY（只用 defect sensitivity 排序）**不能**带来稳健性收益（Δ|Δz| = +0.0362，11/15 cells 更差），
  说明“D 高”本身与 illumination robustness 无关，必须显式使用 I 轴。
- I-ONLY50 有稳健性收益（12/15 胜 random）但 d′ 明显下降（−0.1363），
  说明只压 illumination sensitivity 不够，D 轴必须同时进入排序。
- primary ratio 25% 的 PROPOSED 在 d′ 上**输给** FULL（−0.2285，10/15 cells），即 C6b(i)(iii) FAIL。
- cable 是唯一系统性负迁移 category（PROPOSED25 Δd′ = −0.909，PROPOSED50 Δd′ = −0.274）。
- metric-level directional cells = 15/15。

## 9. 对论文证据链的贡献

- 本实验**关闭**了「是否存在 defect-sensitive + illumination-stable 的 pretrained feature 子空间」
  这一前置问题：metric level 上是 **YES**（ρ ≈ 0.067、15/15 cells、两个 layer 独立成立），
  但在预注册的 primary ratio（25%）下该子空间**无法**在 PatchCore score 上同时保值与增益 → CASE_B。
- 它不影响已冻结的 7A-O 结论（simple representation intervention 无法推动 frontier）；
  它新增的是：**channel 子集选择**在 50% ratio 上给出了一个相对 α=0 的 Pareto 改善信号，
  且仅靠 channel 选择（无任何 representation intervention）即可进入历史 frontier 区域。
- 所有 score-level 结果均为 **oracle diagnostic**（ranking 使用 test defect mask），不构成方法性能主张（§13）。

## 10. 下一步（仅建议，不自动执行）

按 CASE_B 的规则：**Plan C 暂缓，只允许一次非常小的 confirmatory experiment**。候选（尚未批准）：

1. **8C-a（ratio 确认，最小）**：只用已完成的 50% ratio 单元，把 random baseline 从 10 draws 扩到 ≥100 draws，
   并在 cable 上单独报告；目的是确认 PROPOSED50 相对 matched-size random 的 15/15 优势不是抽样噪声。
   注意：这仍不是方法主张（ranking 仍为 oracle），只回答“50% 信号是否稳定”。
2. **8C-b（leakage-free 前置，真正关键）**：检验能否用 **normal-only** 代理（synthetic anomaly /
   feature perturbation / RealNet-style）近似复现 8B 的 desirable channel 排序；
   若复现度低，则 Plan C 在方法层面不可实现，应转 Plan A。8B 本轮按协议未开发任何 proxy。
3. **不建议**：直接进入 8C 的大规模 feature-selection 方法开发（违反 CASE_B 的“禁止立即大规模方法开发”）。

另外需要注意：cable 的系统性负迁移提示 Plan C 若启动，必须内建 category-adaptive 机制，
否则会在 cable 上重演 7A-O 的 negative transfer 问题（此为解释/建议，不是本实验结论）。

