# Industrial Anomaly Detection

工业视觉异常检测研究项目（MVTec AD + PatchCore）。研究主线：**Defect-Preserving Illumination Robustness**
—— 面向光照扰动的特征归一化（α-IN，`F_α = (1−α)F + α·IN(F)`）可能同时**抑制 defect-relevant 信息**；
本项目用 1B→1J-B 建立机制证据，再用 5A→5C 把机制结论推进为可验证的方法。

> **当前状态**：Stage ⑧ Method Identity / Matched-Control Validation **已完成**（Experiment 5D：
> 预注册判定链 literal = CASE_D，实质性结论 = **CASE_B — Geometry Useful but Identity Weak**；CASE 归属待人工裁决）。
> **STOP，不启动 5E / v2 / v3 / 方案②，等待人工 review。**

---

## 0. 当前状态快照（TL;DR）

| 项目 | 内容 |
|---|---|
| 论文阶段 | ① Problem ✅ · ② Theory/Literature ✅ · ③ Phenomenon ✅ · ④ Mechanism ✅ FROZEN · ⑤ Method Design ✅ · ⑥ Improvement Screening ✅ · ⑦ Method Validation ✅ · **⑧ Method Identity / Matched-Control Validation ✅（当前）** · ⑨ Method v2 / benchmark ⏸ · ⑩ Writing ⏸ |
| 研究载体 | PatchCore（`wide_resnet50_2`，layer2+layer3，coreset 0.1，k=9）+ α-IN 层级干预 |
| 数据规模 | MVTec AD 五类：bottle / cable / grid / hazelnut / screw × seeds {0,1,2}；train/good 209 / 224 / 264 / 391 / 320 |
| 扰动协议 | 冻结 synthetic photometric：brightness 0.7/1.3、gamma 0.7/1.3（`apply_photometric`） |
| 主指标（冻结） | robustness = `mean abs(ΔNormalScore_z)`（越低越好）；preservation = `mean defect d′`（越高越好）；ε = 0.10（d′ 非劣带，5A-H §2 冻结） |
| **Stage ④ 结论** | 机制证据链闭合：defect-specific α-IN 响应具有 **layer-specific 几何载体**；`geometry → NN` 为 intervention 级证据（`geometry → score` 为 proxy 级）。**机制探索已 FROZEN** |
| **Stage ⑤ 结论** | 固定的「几何引导层级 α 规则」G2 在 bottle 有效，但**跨 5 类不泛化 → CASE C，STOP frozen rule** |
| **Stage ⑥ 结论** | **仅用 normal training features 的几何即可预测 category 的 normalization tolerance（CASE_A FINAL）**；但 normalization-sensitivity 通道（Group C）**全部 null** → 机制性中介解释**未确立** |
| **Stage ⑦ 结论** | 用该几何做 adaptive α：**激进重分配（v1A）不优于 best fixed α**（事实上的 CASE_D）；**保守门控形式（v1B）达成 harm avoidance（CASE_B）**：negative transfer 0/5、high-damage recovery +0.824 |
| **Stage ⑧ 结论** | matched-control 证伪实验（零 GPU，sanity 18/18）：10 个同 budget gate 的 Δd′ 跨度 −0.033…+0.330 → **「随便 gate 都差不多」（CASE_C）被排除**；GC 在 10 个 gate 中 preservation / worst-category harm / negative transfer 均 **1/10**、**Pareto-efficient**，但 **robustness 10/10（最差）**；GC vs matched sensitivity control Δd′ **+0.1450 / +0.1540（≥ε、3/3 seeds）**，但优势**由单一类别（grid）驱动** → **literal CASE_D / 实质 CASE_B**，identity 未确立 |
| 允许声称 | ① normal-only feature geometry 含有 category-level normalization tolerance 的预测信息（n=5，descriptive）；② 把该几何用于**门控式 safe normalization** 可避免高脆弱类别的归一化损伤；③ **gate 选择本身携带大量信息**（10 个同 budget gate 的 Δd′ 相差 0.36），且 geometry 的 top-2 = 唯一最优 gate |
| 不允许声称 | 因果机制已被证明；"adaptive α 提升平均性能"；**geometry predictor identity 已确立**（5D：优势为单类别驱动、robustness 为 10 个 gate 中最差）；结论可外推到其他 backbone / detector / 数据集 / 真实光照 |
| 最新 commit | `31150bd`（5D）；关键历史 commit：`fdd6fa9`(5C) · `5d6ee98`(5B-C) · `24d1b22`(5B) · `ff4a1b2`(5A-H) · `ba11038`(5A) · `1a0245e`(docs 总账) |

---

## 1. 论文路线

```text
① Problem（现实问题）                          ✅
② Theory / Literature                          ✅
③ Phenomenon（α-IN 的 defect-specific 现象）    ✅
④ Mechanism Validation                         ✅ FROZEN / REPRODUCED（2026-10-06）
⑤ Method Design                                ✅（5A 有条件成功 → 5A-H 否定固定规则）
⑥ Improvement Method Screening                 ✅（5B / 5B-C：CASE_A FINAL）
⑦ Method Validation                            ✅（5C：CASE_B — Safe Normalization 方向）
⑧ Method Identity / Matched-Control Validation ✅（5D：literal CASE_D / 实质 CASE_B — identity 未确立）
⑨ Method v2 / 更大规模验证                      ⏸ 需人工批准（未启动）
⑩ Full Experiments / Benchmark                 ⏸
⑪ Writing                                      ⏸
```

---

## 2. 证据链（每一格：结论 + 关键数字 + 入口）

```text
【③ 现象】1B → 1E
  α-IN 对 defect 的影响存在稳定、跨 seed、跨 5 类别的异质性；
  缺陷面积不足以解释（1D matched-area 后 type 仍有 ΔR²=0.188）
        ↓
【③-排除】1F
  图像空间简单属性（size/contrast/frequency/morphology）无法解释（CASE_D，|ρ|≤0.24）
        ↓
【④ 传导】1G
  feature → NN-distance → score dispersion 强传导（matched ρ=+0.833，frozen ρ=+0.950）
        ↓
【④ 层级结构】1H / 1H-S
  shrink 家族由 layer2-IN 驱动（3/3）、neutral/expand 由 layer3-IN 驱动（6/6，45 units）；
  dispersion 口径下 expand 是真正 L3-dominant（LSI_std=+0.448），shrink 为双层同号
        ↓
【④ 排除混杂】1I
  L3 放大不能用 IN 统计量 spatial sample count 解释（matched-256 后 LSI 仅 −0.029，CASE C）
        ↓
【④ 几何载体】1J-A
  仅 expand 家族出现 Layer3 RMS-radius expansion（R_L3≈+0.45）；geometry 为确定性证据
        ↓
【④ 干预级证据】1J-B
  主动控制 Layer3 radius（β∈{0,0.5,1}）→ NN/score 单调联动（expand 3/3，TRR_NN +0.64/+0.82/+1.39）
        ↓
【复现审计 2026-10-06】1848 NPZ / 30 banks / 1386 intervention rows 全部重建；
  deterministic 指标舍入级一致（max|Δ|=0.000477），stochastic ≤0.03 → REPRODUCED
        ↓
【⑤ 方法设计】5A → 5A-H
  5A（bottle/seed0）：几何引导层级 α 规则 G2 进入 Pareto 前沿，有条件成功；
  5A-H（5 类 × 3 seeds × 5 configs = 75 runs）：**CASE C — bottle-specific**，
  excl-bottle 的 preservation 显著变差（Wilcoxon p=0.0024）→ **固定规则 STOP**
        ↓
【⑥ 方法筛选】5B → 5B-C
  仅用 normal training features 的几何预测 normalization tolerance：
  `radius_ratio_L3L2` / `eff_dim_L3` 与 C2 damage ρ=−0.90、LOCO 5/5、跨 seed 方向一致 → **CASE_A (FINAL)**；
  但补齐的 Group C（normalization sensitivity）8 个 predictor **全部 null** → 机制性解释不成立
        ↓
【⑦ 方法验证】5C
  把该几何变成 category-adaptive α：激进重分配（v1A）≈ best fixed α（0/15 pair-win、3/5 类别 d′ 损失 > ε）；
  保守门控（v1B：仅把 bottle/grid 的 α 降到 0）→ 负迁移 0/5、high-damage 恢复 +0.824 → **CASE_B（Harm Reduction Only）**
        ↓
【⑧ 方法身份验证】5D
  matched-control 证伪：全部 policy 同 budget（gate 2/5、mean α=0.30、α∈{0,0.5}），CPU-only 重组 5A-H frozen raw；
  10 个 exhaustive gate 的 Δd′ 跨度 −0.033…+0.330 → 「随便 gate 都差不多」被排除（6/10 无 harm，但只有 GC 同时 preservation 最大）；
  GC（bottle+grid，= 唯一最优 gate）在 10 个 gate 中 preservation / worst-harm / neg-transfer 均 1/10、Pareto-efficient，
  但 robustness 10/10（最差）；vs matched sensitivity control（SC-A bottle+cable / SC-B grid+screw）Δd′ +0.1450 / +0.1540
  （≥ε、3/3 seeds）却由单一类别（grid）驱动（LOCO 去掉 grid 后 −0.0117）→ literal CASE_D / 实质 **CASE_B：identity 未确立**
```

---

## 3. 实验总索引

| 编号 | 问题 | 关键结果 | 判定 | 状态 |
|---|---|---|---|---|
| [5D](experiments/experiment5d/README.md) | 5C 的收益是否来自 geometry 的**类别选择信息**（而非「随便 gate」） | CPU-only 重组：10 个同 budget gate 的 Δd′ 跨度 −0.033…+0.330（CASE_C 被排除）；GC preservation / worst-harm / neg-transfer 均 **1/10** 且 Pareto-efficient，但 **robustness 10/10**；vs matched sensitivity control **+0.1450/+0.1540（≥ε、3/3 seeds）** 且由 **grid 单类别**驱动 | **literal CASE_D / 实质 CASE_B**（identity 未确立） | DONE `31150bd` |
| [5C](experiments/experiment5c/README.md) | 5B 的预测结构能否真正改善 normalization policy | adaptive α 未在两轴同时优于 best fixed（pair-win 0/15）；v1B 零负迁移 + high-damage 恢复 +0.824 | **CASE_B**（Harm Reduction Only） | DONE `fdd6fa9` |
| [5B-C](experiments/experiment5b/README.md#17-experiment-5b-c--group-c-completion--final-verdict2026-10-06) | 补齐 Group C 后 verdict 是否改变 | 原 geometry 信号不变；Group C 8/8 null（\|ρ\|≤0.30，LOCO ≤ random） | **CASE_A (FINAL)** | DONE `5d6ee98` |
| [5B](experiments/experiment5b/README.md) | normal-only geometry 能否预测 tolerance | `radius_ratio_L3L2` ρ=−0.90 / LOCO 5/5；Group C 资产缺失 | CASE_A (interim) | DONE `24d1b22` |
| [5A-H](experiments/experiment5a_h/README.md) | 冻结几何规则是否跨类别泛化 | 否：G2 仅在 bottle 有效；screw 全缺陷受损；grid 对任何 α 极敏感 | **CASE_C**（STOP） | DONE `ff4a1b2` |
| [5A](experiments/experiment5a/README.md) | bottle 上几何引导 α 是否值得继续 | G2 在 Pareto 前沿、逃出 fixed-α 前沿 | CASE_A（conditional） | DONE `ba11038` |
| [1J-B](experiments/experiment1j_b_transmission/README.md) | geometry 是中介还是伴随现象 | β 干预下 NN/score 单调联动（TRR_NN 0.64–1.39） | CASE A（intervention 级） | REPRODUCED |
| [1J-A](experiments/experiment1j_feature_geometry/README.md) | layer3 是否有几何扩张 | 仅 expand 家族 R_L3≈+0.45 | CASE A | REPRODUCED |
| [1I](experiments/experiment1i_spatial_statistics_control/README.md) | L3 放大是否统计粒度假象 | 否（matched-256 后 LSI 几乎不动） | CASE C | DONE |
| [1H-S](experiments/exp1hs_dispersion_layer/README.md) | dispersion 口径下的层偏好 | expand = L3-dominant（LSI_std +0.448） | CASE B | DONE |
| [1H](experiments/exp1h_layer_selectivity/README.md) | α-IN 是否有层级选择性 | shrink=L2 驱动、neutral/expand=L3 驱动 | 层级选择性成立 | DONE |
| [1G](experiments/exp1g_feature_space/README.md) | 机制在哪个空间传导 | feature→NN→score（ρ≈0.95） | CASE_A | DONE |
| [1F](experiments/exp1f_mechanism_screening/README.md) | 图像空间属性能否解释 | 不能（\|ρ\|≤0.24） | CASE_D | DONE |
| [1E](experiments/exp1e_cross_category/README.md) | 现象是否跨类别 | 25/25 types 3/3 seed 稳定 | CASE_A | DONE |
| [1D](experiments/exp1d_size_control/README.md) | size confound 是否解释响应 | 控制面积后 type 仍有 ΔR²=0.188 | CASE_A | DONE |
| [1C](experiments/exp1c_multiseed/README.md) | 现象是否 seed 稳定 | Δd′ large −5.87±0.39 / cont +0.50±0.09 | CONTINUE | DONE |
| [1B](results/experiment_1b/README.md) | 是否存在 defect-specific α 响应 | d′ large 13.63→7.32、small 稳、cont 反升 | 偏 A（含 C 限定） | DONE |
| [1](results/experiment1_illumination_tradeoff/README.md) | synthetic 光照 × α-IN 苗头 | robustness↑ 与 pixel-level cost 温和 trade-off | CASE A（苗头） | DONE |

**Stage ④ = FROZEN**：`geometry → NN` = intervention-level；`geometry → score` = proxy-level + natural correlation。

> **机制阶段停止声明（仍然有效）**：Mechanism exploration is frozen. Do NOT continue adding
> 1J-C / 2A / additional mechanism probes，除非 Stage ⑦ 之后暴露出具体的证据缺口。

---

## 4. Stage ⑤ 方法设计：5A / 5A-H

### 4.1 [5A](experiments/experiment5a/README.md) — 用 Stage ④ 的 sensitivity 排序做 α 分配（bottle/seed0，11 configs）

- **规则**（冻结，`results/experiment_5a/geometry_rule.json`，md5 `dd08ed5e…`）：inverse-sensitivity max-rescale
  —— `α_L3 = A`，`α_L2 = A × (s_L3/s_L2) = 0.603651 A`，其中 `s_L2 = 0.267031 > s_L3 = 0.161193`
  （1J-A family-level `|radius_response|`）。
- **配置**：B0(0,0) / B1(0.25) / B2(0.5) / B3(0.75) / B4(1.0) + 几何组 G1–G3 + 等 budget 均值对照
  C1(0.400913) / C2(0.601369) / C3(0.801826)。
- **结果**：G2 = (0.45273825, 0.75) 在 bottle 上 4/4 shifts 优于等 budget 对照 C2，并"逃出"fixed-α Pareto 前沿
  → **CASE A（conditional）**。
- **限定**：bottle 上 image AUROC 饱和、`FPR@τ_val = 1.0`，结论由 d′ / \|Δz\| 承载；单类别、单 seed。

### 4.2 [5A-H](experiments/experiment5a_h/README.md) — 跨类别验证（5 类 × 3 seeds × 5 configs = 75 runs，sanity 23/23）

- **目的**：只验证冻结 G2 是否泛化，零 α search、零新方法。
- **主要结果**：
  - excl-bottle 12 units：robustness wins **8/12**，但 median Δ = **−0.007**、Wilcoxon **p = 0.29**（不显著、量级微小）；
  - preservation non-inferior 仅 **4/12**，Wilcoxon **p = 0.0024**（G2 对 bottle 以外类别**系统性损伤** defect evidence）；
  - beyond frontier 仅 **1/12**；**12/25** defect types 退化（screw 的全部缺陷类型受损）；
  - **grid 对任何 α 都极敏感**：d′ B0 3.136 → B2 2.364 / C2 2.134 / C3 1.802（同时 \|Δz\| 改善）；
  - **hazelnut** 在 B2 下 d′ 反而略升（5.964 → 6.175）；screw/C3 最差（2.593 → 2.204）；
  - `FPR@τ_val` 退化（train/good 与 test/good 系统性分布 gap）是**全部 5 类**共同现象。
- **判定**：**CASE C — BOTTLE-SPECIFIC → STOP frozen G2**（5A 的优势是 bottle/seed0 sweet spot，不是通用规则）。
- **副产物（5C 的 baseline 来源）**：冻结的 `ε = 0.10` PAIR-WIN 规则 + 4 个 uniform α 的全 5 类结果
  （B0=0 / B2=0.5 / C2=0.601369125 / C3=0.8018255），构成后续「best fixed α」与 α grid 的唯一合法来源。

### 4.3 该阶段的教益

> **"用机制结论直接写死一条全局规则" 失败**：sensitivity 排序（由 defect 特征估计）不能跨 category 复用。
> 这直接引出 Stage ⑥ 的问题：**normal training data 本身是否含有 tolerance 信息**。

---

## 5. Stage ⑥ 方法筛选：5B / 5B-C

### 5.1 [5B](experiments/experiment5b/README.md) — predictive structure probe（CPU-only）

- **问题**：只看 normal training features 的 geometry，能否预测某 category 对 normalization 的 tolerance？
- **X**：1J 冻结的 30 个 normal coreset bank（5 类 × 3 seeds × {layer2, layer3}，α=0 原始特征，KCenterGreedy 0.1）+
  train/good 图像统计（negative control）。
- **Y（primary）**：`C2 damage = d′(B0) − d′(C2)`（来自 5A-H，同一 split 协议）；secondary：C3 / G2 damage。
- **预注册 predictor**：15 个（Group A magnitude / Group B geometry / Group D cross-layer ratio / controls）。
- **Category damage（3 seeds 平均）**：bottle **+1.207** > grid **+1.001** > cable +0.249 > screw +0.113 > hazelnut **−0.183**。

| predictor | group | ρ(C2) | ρ_seed | LOCO |
|---|---|---:|---:|---:|
| eff_dim_L3 | B geometry | **−0.90** | −0.875 | 4/5 |
| radius_ratio_L3L2 | D cross-layer | **−0.90** | −0.832 | **5/5** |
| nn_dist_ratio_L3L2 | D cross-layer | **−0.90** | −0.804 | 4/5 |
| img_pixel_std | control | +0.90 | +0.818 | 3/5 |
| nn_dist_rel_L3 | B geometry | −0.80 | −0.739 | 4/5 |
| eff_dim_L2 | B geometry | −0.80 | −0.768 | **5/5** |
| rms_radius_L3 | B geometry | −0.60 | −0.564 | **5/5** |
| random_control | control | — | — | 3/5 |

- **其它检查**：15/15 predictor 逐 seed 方向一致；leave-one-category-out（对全部 15 个 predictor 都报告）后 top predictor \|ρ\| 仍 ≥ 0.8；
  geometry 预测器在 LOCO 上明显优于 controls（5/5 vs 3/5、2/5）。
- **实现记录**：首次运行的 NN 距离展开式漏 `||x||²` 导致 `nn_dist_rel = 0`，修复后 sanity A–E 全 PASS（属实现 bug，非科研结果）。
- **判定**：**CASE_A (interim)** —— 预测结构存在，但 Group C（normalization sensitivity）资产缺失，机制通道未验证。

### 5.2 [5B-C](experiments/experiment5b/README.md#17-experiment-5b-c--group-c-completion--final-verdict2026-10-06) — 补齐 Group C 后的 FINAL verdict

- **新增资产**：8 个 normalization-sensitivity predictor（`sens_radius/mdc/effrank/pca1` × L2/L3），
  复用 1J-A 冻结 `geometry_of` 与 1E/1H/1J 冻结 IN（`_alpha_mix(α=1) == instance_norm`），
  forward-only、只读 normal train 图（3924 张），定义在读取 target 前写入 `group_c_freeze.json`（`targets_read=false`）。
- **sanity**：S-C1…S-C7 **7/7 PASS**（含与 1J-A 冻结 per_image CSV 的定义一致性对照 `max_rel_dev = 1.11e-05`）；
  主分析链 **19 checks / 0 FAIL**。

| Group C predictor | ρ(C2) | LOCO | 留一 category ρ 范围 |
|---|---:|---:|---|
| sens_radius_L2 | −0.30 | 2/5 | [−0.8, **+0.4**]（剔 bottle 反号） |
| sens_radius_L3 | +0.10 | 3/5 | [−0.4, +0.6] |
| sens_mdc_L2 | −0.30 | 2/5 | [−0.8, +0.4] |
| sens_mdc_L3 | −0.10 | **0/5** | [−0.6, +0.4] |
| sens_effrank_L2/L3 | −0.20 | 1/5 | [−0.8, +0.2] |
| sens_pca1_L2/L3 | +0.20 | 1/5、2/5 | [−0.2, +0.8] |

- **结论**：**CASE_A (FINAL)** —— 原 geometry 信号（`eff_dim_L3` / `radius_ratio_L3L2`，ρ=−0.90、LOCO 5/5）
  加入 Group C 后**完全不变**；但 **Group C 8/8 null**（\|ρ\|≤0.30、LOCO ≤ random 0.60、留一可反号）
  → **「normalization 敏感度中介 tolerance」的机制解释不被支持**，存活的是**通用 normal-feature 分散度几何**。
- **限定**：n = 5 categories，全部 descriptive；`img_pixel_std` 在 \|ρ\| 上与 top geometry 并列（0.90），
  仅 LOCO 可区分（3/5 vs 5/5）。

---

## 6. Stage ⑦ 方法验证：[5C](experiments/experiment5c/README.md) — Geometry-Guided Category-Adaptive α v1

### 6.1 冻结设计（运行前写入 `geometry_policy_freeze.json`，sha256 `05dc6c37…`）

| 项 | 值 |
|---|---|
| primary predictor | `radius_ratio_L3L2`（5B FINAL：ρ=−0.90，LOCO 5/5） |
| control predictor | `sens_radius_L2`（Group C primary，5B：ρ=−0.30、LOCO 2/5） |
| α grid（只能取自历史） | `[0.0, 0.5, 0.601369125, 0.8018255]`（= 5A-H B0/B2/C2/C3；唯一具备 5 类 × 3 seeds 全覆盖的 uniform α） |
| best fixed α | **0.5（B2）**：用 5A-H 冻结 PAIR-WIN 规则（ε=0.10）统计，B2 在 **7/15** units 战胜 α=0，远高于 C2/C3（各 3/15） |
| conservative ceiling | **0.5 = best fixed α**（零新参数的机械规则） |
| mapping | `α(r) = grid[min(K−1, ceil(r·K/n)−1)]`，r = predictor 升序 rank，n=5；rank-monotonic、deterministic、category-name blind |
| 方向 | higher geometry → higher α |

| policy | predictor | α 分配（bottle / grid / cable / hazelnut / screw） | mean α |
|---|---|---|---|
| A. Original | — | 0 / 0 / 0 / 0 / 0（复用 5A-H B0） | 0.000 |
| B. Best Fixed | — | 0.5 ×5（复用 5A-H B2） | 0.500 |
| **C. Geometry Full (v1A)** | `radius_ratio_L3L2` | 0 / **0.5** / 0.6014 / 0.8018 / 0.8018 | 0.541 |
| **D. Geometry Conservative (v1B)** | `radius_ratio_L3L2` | 0 / **0** / 0.5 / 0.5 / 0.5 | 0.300 |
| E. Sensitivity-Guided | `sens_radius_L2` | 0 / 0.8018 / 0.5 / 0.6014 / 0.8018 | 0.541 |

### 6.2 执行与等价性

- **45/45 config runs 完成**（3 policies × 5 categories × 3 seeds，3 workers，约 40 min）；A/B 复用 5A-H 冻结结果。
- **raw score 级等价性**：5C 与 5A-H 在相同 α 上的 **33 个 (cat, seed, α) key、9273 行 per-image 分数逐位一致**
  （`max|Δscore| = 0.000e+00`，`max|Δτ_val| = 0.0`）——因为 5A-H 已覆盖全部 4 个 uniform α × 5 类，
  **5C 本质是冻结 α grid 上的重组**，同时构成一条强复现性检查。
- **sanity S1–S15 全 PASS**（含 6 个历史冻结文件 md5 前后不变）。

### 6.3 主要结果（5 categories × 3 seeds 平均）

| policy | mean α | mean defect d′ ↑ | mean \|ΔNormalScore_z\| ↓ | gain vs best fixed (d′) | Δrobust vs best fixed |
|---|---:|---:|---:|---:|---:|
| A. Original（α=0） | 0.0000 | 4.9852 | 0.3105 | +0.3182 | +0.1023 |
| B. Best Fixed（α=0.5） | 0.5000 | 4.6670 | 0.2082 | 0（参照） | 0（参照） |
| C. Geometry Full（v1A） | 0.5410 | 4.7019 | 0.2277 | +0.0349 | **+0.0195（更差）** |
| D. Geometry Conservative（v1B） | 0.3000 | **4.9967** | 0.2643 | **+0.3297** | **+0.0560（更差）** |
| E. Sensitivity-Guided | 0.5410 | 4.6679 | 0.2364 | +0.0009 | +0.0282 |

逐 category（d′ / \|Δz\|，3 seeds 平均）：

| category | A Original | B Best Fixed | C GF | D GC | E SG |
|---|---|---|---|---|---|
| bottle | 8.179 / 0.442 | 7.302 / 0.286 | 8.179 / 0.442 | 8.179 / 0.442 | 8.179 / 0.442 |
| grid | 3.136 / 0.224 | 2.364 / 0.100 | 2.364 / 0.100 | **3.136 / 0.224** | 1.802 / 0.105 |
| cable | 5.055 / 0.331 | 5.008 / 0.271 | 4.806 / 0.238 | 5.008 / 0.271 | 5.008 / 0.271 |
| hazelnut | 5.964 / 0.214 | 6.175 / 0.151 | 5.958 / 0.138 | 6.175 / 0.151 | 6.147 / 0.143 |
| screw | 2.593 / 0.341 | 2.487 / 0.232 | 2.203 / 0.221 | 2.487 / 0.232 | 2.203 / 0.221 |

**Harm analysis（相对 best fixed α=0.5）**：

| policy | mean gain | worst category | high-damage recovery (bottle+grid) | negative transfer | pair-win units |
|---|---:|---|---:|---:|---:|
| C. GF（v1A） | +0.0349 | screw **−0.2831** | +0.4384 | **3/5** | **0/15** |
| D. GC（v1B） | +0.3297 | cable **0.0000** | **+0.8243** | **0/5** | 0/15 |
| E. SG | +0.0009 | grid **−0.5617** | +0.1576 | 2/5 | 2/15 |

**Seed 稳定性**（d′ mean ± std over 3 seeds；robustness 同向）：

| policy | d′ s0/s1/s2 | mean ± std | \|Δz\| mean ± std |
|---|---|---|---|
| A Original | 5.020 / 4.956 / 4.980 | 4.9852 ± 0.0321 | 0.3105 ± 0.0164 |
| B Best Fixed | 4.664 / 4.670 / 4.667 | 4.6670 ± 0.0032 | 0.2082 ± 0.0071 |
| C GF | 4.739 / 4.677 / 4.690 | 4.7019 ± 0.0327 | 0.2277 ± 0.0066 |
| D GC | 5.019 / 5.018 / 4.953 | 4.9967 ± 0.0378 | 0.2643 ± 0.0075 |
| E SG | 4.698 / 4.696 / 4.609 | 4.6679 ± 0.0511 | 0.2364 ± 0.0097 |

- 方向在 3/3 seeds 一致：GF/GC 的 d′ 全胜 best fixed，但 robustness 全败（seed_pair_wins = [0,0,0]）→ **无 seed 反转**。
- **LOCO**：去掉任一类别、用剩余 4 类重算 rank→α 后，gain 仍为正（+0.098 … +0.142），robustness 代价 +0.026…+0.034
  → **不是单一 category 驱动**。
- **Predictor identity（GF vs SG，同 budget 0.5410）**：聚合上 GF 两轴都不劣（Δd′ **+0.0341**、Δ\|Δz\| **−0.0087**），
  但优势集中在 **grid 单类别**（+0.562），在 cable（−0.202）/hazelnut（−0.189）落后
  → **identity directionally consistent，但未确立**（且 v1B 没有 matched sensitivity 对照）。

### 6.4 判定与限定

**CASE_B — Harm Reduction Only**（driver = v1B / Geometry Conservative）：

- **v1A（激进全范围）实质是 CASE_D**：Δd′ +0.035 但 robustness 更差 +0.0195、pair-win **0/15**、
  3/5 类别 d′ 损失 > ε（screw −0.283 / hazelnut −0.217 / cable −0.202）→ **"更激进地重分配 α" 不优于 best fixed α**。
- **v1B（保守门控）成立**：等于「best fixed α 保持不变，仅把 bottle + grid 的 α 降到 0」→
  high-damage recovery **+0.824**、worst-category gain **0.000**、negative transfer **0/5**；
  且 5 类平均在两轴上都不劣于 Original PatchCore（Δd′ +0.0115、Δ\|Δz\| −0.0462，后者 3/3 seeds 一致）。
- **限定**：n=5；v1B 的收益只来自 2 个类别且以这两类的 robustness 为代价；predictor identity 未确立；
  未做未预注册的显著性检验；不做 causal claim。

---

## 7. Stage ⑧ 方法身份验证：[5D](experiments/experiment5d/README.md) — Matched-Control Gate Validation

**实验目的**：检验 5C 的 GC 收益是否真的来自 geometry 提供的**类别选择信息**，而不是「随便关两个类别」。
5D 是**证伪实验**，唯一自由量是「**谁**决定哪 2 个 category 的 α=0」。

**为什么必须做**：5C 只证明了「保守 gate 可以避免 fragile 类别的 normalization 损伤」，
未排除「任何 ranking 只要把 fragile 类别降到 0 就能拿到同样收益」这一替代解释。

**Frozen protocol**（`results/experiment_5d/reference/policy_freeze.json`，sha256 `30150bc4…`；
运行前冻结、`target_results_read=false`；**全部 policy 同 budget**：gate 2/5、α∈{0, 0.5}、mean α = 0.30）

| 角色 | policy | predictor | orientation | gate | α 分配（bottle/cable/grid/hazelnut/screw） |
|---|---|---|---|---|---|
| primary method | **GC** | `radius_ratio_L3L2` | low→fragile（5C 冻结，逐类一致 hard assertion） | bottle+grid | 0 / 0.5 / 0 / 0.5 / 0.5 |
| primary matched control | **SC-A** | `sens_radius_L2` | low→fragile（5C 冻结 mechanical rule） | bottle+cable | 0 / 0 / 0.5 / 0.5 / 0.5 |
| secondary control | SC-B | `sens_radius_L2` | high→fragile（5B registry 假设） | grid+screw | 0.5 / 0.5 / 0 / 0.5 / 0 |
| reference / baseline | A / B | — | — | — | α=0 全体（5A-H B0）/ α=0.5 全体（5A-H B2） |
| exhaustive controls | 10 个 gate | — | 全部 C(5,2) | 任意 2 类 | **EVALUATION-ONLY**，禁止用其结果替换冻结 policy |

**Protocol ambiguity（运行前发现并上报，人类裁决）**：`sens_radius_L2` 的 conservative direction 在历史记录中
**不唯一** —— 5B registry 预注册 `high->fragile`，而 5B 实测 ρ(C2)=−0.30（负）与 5C 冻结的 rank→α 机械方向
都指向 `low->fragile`。裁决：**两者都预注册**，SC-A 为 primary、SC-B 对称全量报告；不依据 target 结果选方向。

**执行**：**零 GPU**（CPU-only 重组 5A-H 已冻结 raw per-image 分数，30/30 condition key 齐备）；
sanity **S1–S18 = 18/18 PASS**；5C↔5A-H 同 α raw 等价性 `max|Δscore| = 0`。

**关键结果**

| policy | gate | mean α | defect d′ ↑ | \|Δz\| ↓ | Δd′ vs fixed | Δ\|Δz\| | worst-cat gain | neg transfer | HD recovery |
|---|---|---|---|---|---|---|---|---|---|
| A. Original | — | 0.00 | 4.9852 | 0.3105 | +0.3183 | +0.1023 | −0.2107 | 1 | +0.8243 |
| B. Best Fixed | — | 0.50 | 4.6670 | 0.2082 | 0 | 0 | 0.0000 | 0 | 0 |
| **GC** | bottle+grid | 0.30 | **4.9967** | 0.2643 | **+0.3297** | +0.0560 | **0.0000** | **0** | **+0.8243** |
| SC-A | bottle+cable | 0.30 | 4.8517 | 0.2514 | +0.1847 | +0.0432 | 0.0000 | 0 | +0.4384 |
| SC-B | grid+screw | 0.30 | 4.8427 | 0.2547 | +0.1757 | +0.0465 | 0.0000 | 0 | +0.3859 |

- **Exhaustive**：10 个同 budget gate 的 Δd′ 跨度 **−0.0328 … +0.3297**（6/10 做到 worst-cat gain = 0，
  4/10 因 gate 了 hazelnut 产生 **−0.2107** harm）。GC 排 **1/10**（次优 bottle+screw +0.1967，gap +0.133 > ε），
  且**只有 GC 自己**落在 GC − ε 之内。
- **GC 位置**：preservation **1/10**、worst-category harm **1/10**、negative transfer **1/10**、
  **robustness 10/10（最差）**、**Pareto-efficient**。
- **GC vs matched controls**：Δd′ **+0.1450**（SC-A）/ **+0.1540**（SC-B），worst-harm 与 neg-transfer 均相同（0 / 0），
  **3/3 seeds 方向一致**；但 robustness 更差（+0.0128 / +0.0095），且优势**由单一类别驱动**
  （LOCO 去掉 grid → −0.0117；去掉 bottle → GC−SC-B −0.0267）。
- 结构性事实：gate 排序完全由「被 gate 类别的 α=0 收益」决定
  （bottle +0.877 > grid +0.772 > screw +0.107 > cable +0.047 > hazelnut −0.211）；
  **geometry 的 top-2 = 唯一最优 gate**，两个 sensitivity orientation 都漏掉最优组合的一半
  （SC-A 把 grid 排最后，SC-B 把 bottle 排最后）。

**Final verdict**

- **预注册判定链（literal）：CASE_D** —— 且**仅**由 LOCO「单类别驱动」条件触发；CASE_D 其余三条显式为 false。
- **预注册缺陷披露（未修改协议、未重跑）**：**(D1)** CASE_D 的 driver 判据用了严格 `> 0` 而非冻结 ε 带，
  在 **−0.0117（0.12 ε）** 上触发，对任意一对 gate 近乎必然触发；**(D2)** CASE_A 条件 5（seed-level pair-win，
  蕴含 `Δ|Δz| < 0`）与条件 2（容许 `Δ|Δz| ≤ +0.02`）互相矛盾 → **CASE_A 结构性不可达**（实测 0/3）。
  按 §18 原文口径（seed-level **d′ 方向** ≥2/3）应为 **true**（实测 3/3）。
- **实质结论 = CASE_B（Geometry Useful but Identity Weak）**：CASE_A 条件 1–7 成立、**仅条件 8（非单一 category 驱动）
  不成立**；CASE_C 两条判据均显式排除（`|Δd′| = 0.1450 ≥ ε`、n_ge = 1）；CASE_D 的实质含义与
  「GC 是 10 个同 budget gate 中唯一同时做到无 harm 且 preservation 最大」不符。
- **最终 CASE 归属请人工裁决**（literal = CASE_D / 实质 = CASE_B）。**不得**据此声称 CASE_A。

**对论文证据链的贡献 / 假设的支持与否定**

- **支持**：gate 选择携带大量信息 —— 10 个同 budget gate 的 Δd′ 相差 0.36；geometry 的 top-2 = 唯一最优 gate
  （= target 侧 high-damage 子集 {bottle, grid}，仅用 normal images 复现）。
- **否定**：**CASE_C（"随便 gate 都差不多"）被排除**；同时否定「GC 在两轴同时更优」（GC 的 robustness 是
  10 个 gate 中最差）。
- **削弱**：**geometry identity 的排他性证据弱** —— 优势为单类别（grid）驱动、n=5、gate size=2，
  且 GC 的 gate 恰好等于历史 target-derived high-damage 子集。

**是否停止方案①**：**不由本实验单独决定**。5D 排除了 CASE_C，但未能确立 identity（CASE_A 不可声称），
实质落在 CASE_B。按 §27 **STOP，等待人工 review**。

---

## 8. 已冻结的结论与边界（Fact / Interpretation / Hypothesis 分离）

**事实（实验直接观察到）**

1. α-IN 的 defect 响应存在稳定异质性，且不能被缺陷面积或图像空间简单属性解释（1B–1F）。
2. 该异质性在表征层传递：feature → NN-distance → score dispersion（1G，ρ≈0.95）。
3. 响应具有 layer-specific 结构：shrink=L2 驱动、neutral/expand=L3 驱动（1H，45 units）；expand 家族出现
   Layer3 RMS-radius expansion（1J-A）；主动控制 radius 可使 NN/score 单调联动（1J-B）。
4. 固定几何规则 G2 只在 bottle 有效，跨类别会系统性损伤 defect evidence（5A-H，p=0.0024）。
5. 仅用 normal training geometry 即可预测 category 的 C2 damage（5B/5B-C，ρ=−0.90、LOCO 5/5、跨 seed 一致）。
6. normal feature 的 normalization sensitivity（Group C）与 damage 无稳定关系（8/8 null）。
7. 5C：adaptive α 未在两轴同时优于 best fixed（pair-win 0/15、0/15、2/15）；conservative 门控形式
   可做到 negative transfer 0/5 与 high-damage 完全恢复。
8. 5D（matched-control 证伪，零 GPU，sanity 18/18）：10 个同 budget gate（gate 2/5、mean α=0.3、α∈{0,0.5}）的
   Δd′ 跨度 **−0.033…+0.330**；GC 在 10 个 gate 中 preservation / worst-category harm / negative transfer
   均 **1/10** 且 **Pareto-efficient**，但 **robustness 10/10（最差）**；GC 相对 matched sensitivity control
   Δd′ **+0.1450 / +0.1540**（≥ε、**3/3 seeds**），但优势**由单一类别（grid）驱动**（LOCO 去掉 grid 后 −0.0117）。

**解释（基于事实的推断，且受限定）**

- 机制位于**表征空间的 layer3 几何**，而非图像空间属性。
- 5B 的预测信息是**通用分散度几何**，不是 normalization-specific 敏感度。
- 5C 表明 geometry 的可用价值是**"哪些类别不该被 normalization"的门控信息**，而非"更强的 α 分配"。

**假设（尚未被验证）**

- ~~v1B 的收益是否必须依赖 geometry rank~~ → **5D 已部分回答**：**不是**「随便 gate 都差不多」
  （CASE_C 被排除），但 geometry 相对 matched sensitivity control 的优势**由单一类别驱动**，
  **predictor identity 仍未确立**（CASE_A 不可声称）。
- "高脆弱类别 → α=0" 的规则是否能泛化到 MVTec 其余类别、其他 backbone/detector、真实光照。

---

## 9. 下一步（建议，均未启动，需人工批准）

0. **5D CASE 归属裁决（人工，必须）**：literal CASE_D vs 实质 CASE_B —— 见 §7 与
   [5D README §15](experiments/experiment5d/README.md)。
1. **⑨ Method v2（Safe Normalization 形式化）**：若继续方案①，结论口径须固定为「gate 选择携带信息 +
   geometry 复现最优 gate，但 identity 未确立」，并需要**扩大类别数**（使 gate size 与类别数不成比例）
   来解除「单类别驱动」这一设计性限制；仍须预注册 matched control。
2. **限制声明先行**：任何 v2 结论都必须带着限定（v1A 失败 / identity 未确立 / GC robustness 为 10 gate 中最差 / n=5）。
3. **不做**：post-hoc 调参救 v1A、根据 exhaustive 结果挑新 gate、调 threshold / gate size / α、新造 predictor、
   无新证据的机制深挖（1J-C/2A）、启动 5E / v2 / v3 / 方案②（均需人工批准）。

---

## 10. 复现与工程约定

- 2026-10-06 曾在 AutoDL 完成一次**全量资产恢复复现审计**（1848 NPZ / 30 banks / 1386 intervention rows），
  详见 [OVERNIGHT_REPORT.md](OVERNIGHT_REPORT.md)；此后 5B / 5B-C / 5C 均在新环境内完成并通过等价性检查
  （5C 与 5A-H 同 α 的 raw 分数逐位一致）。**5D 为纯 CPU 历史重组**（零 GPU、零新 inference），
  同样复现 `max|Δscore| = 0`（33 keys / 9273 行）。
- **Git 约定**：`results/` 默认忽略，仅对 `experiment_5a/`、`experiment_5a_h/`、`experiment_5b/`、
  `experiment_5b_final/`、`experiment_5c/`、`experiment_5d/` 开白名单；数据集、大 NPZ/bank/checkpoint、
  运行日志（`*.log`）不入库。
- **实验纪律**：每个正式实验先写预注册 README（四问 + 出口判据）→ 最小 sanity → 运行 → 结果/判定/限定
  写回 README 与根 README → commit（push 需授权）。预注册规则若在运行后发现缺陷，**如实披露并保留原判定**，
  不得回改协议（5D §15.1 为例）。

---

## Environment

### 当前正式实验环境（AutoDL 服务器，2026-10-05 验证 READY）

- Linux（AutoDL 容器）
- Conda **base** 环境（`/root/miniconda3`），Python 3.12.3
- PyTorch 2.8.0 + CUDA 12.8
- NVIDIA GeForce RTX 3090 24GB
- Anomalib 2.6.2
- OpenCV 5.0.0

运行 anomalib 模型时必须设置（服务器直连 huggingface.co 不可达）：

```bash
export HF_ENDPOINT=https://hf-mirror.com
```

### 历史环境（Windows 本地，已退役，仅作记录）

- Windows 11 / Python 3.11 / PyTorch 2.11.0 + CUDA 12.8 / RTX 5060 Laptop GPU
- Conda 环境 `industrial-ad`（服务器上不存在该环境）

---

## Project Structure

```text
industrial-anomaly-detection/
├── data/                         # 数据集（不上传 Git）
│   └── mvtec_ad/
│
├── experiments/                  # 正式模型实验
│
├── notebooks/                    # 学习、分析与 Baseline Notebook
│   └── 01_patchcore_bottle.ipynb
│
├── results/                      # 实验结果、可视化图片、模型权重等
│   └── patchcore_bottle_broken_large.png
│
└── scripts/                      # 通用训练、测试和数据处理脚本
```

其中：

- `data/`：存放 MVTec AD 等实验数据，不提交至 Git；
- `notebooks/`：保存模型学习、Baseline 复现和结果分析过程；
- `results/`：保存正式实验结果和可视化图片；
- `experiments/`：后续用于更加规范的模型对比实验；
- `scripts/`：后续逐步抽离可重复使用的训练、测试和数据处理代码。

---

## Dataset

当前使用：

### MVTec AD

MVTec AD 是工业视觉异常检测常用公开数据集，包含多种工业物体和纹理类别。

当前正式使用（Stage ③ 起）：

```text
MVTec AD
├── bottle
├── cable
├── grid
├── hazelnut
└── screw
```

| category | train/good | 说明 |
|---|---:|---|
| bottle | 209 | Stage ③ 起首个类别（1B–1D 均在此） |
| cable | 224 | 1E 起加入（跨类别验证） |
| grid | 264 | 1E 起加入；对 α-IN 最敏感 |
| hazelnut | 391 | 1E 起加入；部分 α 下 defect d′ 反而略升 |
| screw | 320 | 1E 起加入；strong α 下全缺陷类型受损 |

通用约定：

- 训练阶段**仅使用正常样本**（train/good 划 20 张作 validation，其余进 memory bank）；
- 测试集包含正常样本与异常样本，异常样本提供像素级 Ground Truth Mask；
- 随机性由 `make_validation_split(category, seed)` 控制，固定 seeds {0, 1, 2}；
- 早期（2026-09-24 ~ 2026-10-01）曾只用 bottle 跑通流程，历史记录见文末附录。

---

## Models

计划逐步实验以下工业异常检测方法：

- [x] PatchCore
- [ ] PaDiM
- [ ] EfficientAD
- [ ] 其他近期工业异常检测方法

当前第一个重点 Baseline：

### PatchCore

当前配置：

```text
Backbone: Wide ResNet-50-2
Feature Layers: layer2 + layer3
Coreset Sampling Ratio: 0.1
```

PatchCore 的基本思路是：

```text
正常训练图片
        ↓
预训练 CNN 提取局部特征
        ↓
layer2 + layer3 多尺度特征
        ↓
Coreset Sampling
        ↓
建立正常特征 Memory Bank
        ↓
测试图片特征
        ↓
与正常特征进行最近邻比较
        ↓
异常分数
        ↓
Anomaly Map
        ↓
Predicted Mask
```

与传统监督目标检测不同，PatchCore 不需要提前收集大量不同类型的缺陷样本。

它主要学习：

> **正常产品的局部特征应该是什么样子。**

测试时，如果某个区域的特征与正常 Memory Bank 中的特征差异较大，则该区域会获得更高的异常分数。

---

## Research Roadmap

研究路线已收敛为一条论文主线（详见上文 §1 / §2）：

- [x] 1. 实验环境（AutoDL / conda base / PyTorch 2.8.0+cu128 / anomalib 2.6.2）
- [x] 2. MVTec AD 五类数据集 + 冻结 split 协议
- [x] 3. PatchCore baseline 跑通并与 anomalib 指标对齐
- [x] 4. 现象层：α-IN 的 defect-specific 响应（1B–1E）
- [x] 5. 排除层：图像空间属性无法解释（1F）
- [x] 6. 机制层：传导链 / 层级结构 / 几何载体 / 干预验证（1G–1J-B，FROZEN）
- [x] 7. 方法设计：几何引导 α 规则的探索与否定（5A / 5A-H）
- [x] 8. 方法筛选：normal-only geometry 的预测性（5B / 5B-C，CASE_A FINAL）
- [x] 9. 方法验证：adaptive α vs fixed α（5C，CASE_B）
- [ ] 10. 方法 v2（Safe Normalization 门控形式）+ 更大规模验证（未启动，需批准）

当前原则：

> **先冻结协议，再跑实验；先做小实验，再决定方向；失败/否定的结果同样入库。**

---

# 附录 A：历史详细日志（2026-09-24 → 2026-10-02）

> 以下为分阶段实验的**原始详细记录**（环境搭建 / baseline 跑通 / 1 号实验 / 1B–1E）。
> 结论摘要已并入上文 §2–§5；1F–1J-B、5A–5C 的完整记录见各自 `experiments/*/README.md`。

## 2026-09-24｜环境搭建与异常检测入门

### 已完成

- [x] 创建 `industrial-ad` Conda 环境
- [x] 配置 PyTorch + CUDA
- [x] RTX 5060 Laptop GPU 测试成功
- [x] 安装 OpenCV / Jupyter
- [x] 安装 Anomalib 2.6.2
- [x] 创建工业异常检测实验目录
- [x] 明确使用 MVTec AD 作为第一阶段实验数据集
- [x] 初步理解工业异常检测与普通监督分类 / YOLO 检测的区别
- [x] 理解重建式异常检测的基本思想
- [x] 明确当前主线优先放在工业视觉异常检测

### 当日里程碑

> **完成工业异常检测实验环境搭建，并建立对工业异常检测任务的基本认识。**

---

## 2026-09-25｜PatchCore Bottle Baseline

### 已完成

#### 1. 数据集

- [x] 加载 MVTec AD Bottle 数据集
- [x] 确认训练集包含 209 张正常图片
- [x] 确认测试集包含 83 张图片
- [x] 理解训练集、测试集和 Ground Truth Mask 的作用

#### 2. PatchCore 模型

- [x] 确定 PatchCore 作为第一个重点 Baseline
- [x] 使用 `wide_resnet50_2` 作为 Backbone
- [x] 使用 `layer2 + layer3` 提取局部特征
- [x] 设置 `coreset_sampling_ratio=0.1`
- [x] 初步理解 PatchCore 的特征式异常检测思路

#### 3. Memory Bank

- [x] 使用正常 Bottle 图片执行 PatchCore Fit
- [x] 完成正常样本局部特征提取
- [x] 完成 Coreset Sampling
- [x] 建立正常特征 Memory Bank
- [x] 理解 PatchCore Fit 与普通神经网络训练的区别

#### 4. Test / Predict

- [x] 完成 Bottle 测试集 Test
- [x] 完成测试图片 Predict
- [x] 获取 Anomalib `ImageBatch` 预测结果
- [x] 学习读取以下模型输出：
  - `image`
  - `gt_mask`
  - `anomaly_map`
  - `pred_mask`
  - `pred_score`
  - `pred_label`

#### 5. 可视化

- [x] 选择 `broken_large` 异常样本
- [x] 完成 Original Image 可视化
- [x] 完成 Ground Truth Mask 可视化
- [x] 完成 PatchCore Anomaly Map 可视化
- [x] 完成 Predicted Mask 可视化
- [x] 完成图像反归一化，恢复正常 RGB 显示
- [x] 将四联图保存至 `results/`

实验结果文件：

```text
results/patchcore_bottle_broken_large.png
```

#### 6. Notebook 整理

- [x] 清理重复的 Fit / Predict / 可视化代码
- [x] 按完整实验流程重新组织 Notebook
- [x] 添加实验目的和方法说明
- [x] 添加各阶段 Markdown 实验记录
- [x] 添加实验结果与总结

当前 Notebook：

```text
notebooks/01_patchcore_bottle.ipynb
```

Notebook 当前结构：

```text
实验说明
   ↓
1. 实验环境与依赖
   ↓
2. MVTec AD Bottle 数据集
   ↓
3. PatchCore 模型
   ↓
4. Engine
   ↓
5. Fit：建立 Memory Bank
   ↓
6. Test：模型性能评估
   ↓
7. Predict：生成预测结果
   ↓
选择 broken_large 样本
   ↓
8. 异常检测结果可视化
   ↓
9. 实验结果与总结
```

### 当日里程碑

> **完成第一个 PatchCore + MVTec AD Bottle Baseline 的完整实验闭环。**

已经从：

```text
了解工业异常检测
```

推进到：

```text
数据集
→ 模型
→ Fit
→ Memory Bank
→ Test
→ Predict
→ Anomaly Map
→ Predicted Mask
→ 实验结果保存
```

---

## 2026-10-01｜Experiment 1: Synthetic Illumination × α-IN 机制筛查

### 实验性质（边界声明）

**Synthetic Illumination Mechanism Screening / Sanity Check**。

brightness / gamma 等简单数字变换**不能**模拟真实工业光照（specular reflection / highlight / shadow / local contrast / material response / defect visibility / illumination direction 均未覆盖）。因此本实验只能回答：

> 在简单 photometric perturbation 下，α-IN 是否表现出值得进一步验证的现象？

**不能**回答"在真实工厂光照变化下该方法是否有效"。任何结论只能表述为 "Synthetic illumination screening suggests..."。

### 启动前记录（四句话，预注册）

**① 我怀疑什么？**
增强特征归一化（α-IN）可能减少 PatchCore 对简单 photometric variation 的敏感性，使正常产品在亮度/曝光/gamma 改变后不易被误判为异常；但过强归一化也可能削弱真实缺陷相关的外观/纹理/局部对比度信息，且这种损失可能具有 defect-type dependence（structural defects 如 broken_* 相对稳定，appearance-related 如 contamination 更敏感）。**仅为待验证假设。**

**② 准备干什么？**
MVTec AD bottle + PatchCore，固定除 α 外全部条件，`F_alpha = (1-α)F + α·IN(F)`（InstanceNorm affine=False，复用 2026-09-30 归档实现），α ∈ {0, 0.25, 0.5, 0.75, 1.0}。A 组：正常 test 图 + 简单 synthetic photometric perturbation；B 组：原始真实 defect 图（不做人为光照修改）。

**③ 看什么结果？**
Robustness side：perturbation 后正常图 anomaly score 升高多少，随 α 是否减弱；Sensitivity side：各 defect type 的 score / 检测性能随 α 的变化。保留 defect-type level 结果，不只看 overall AUROC。

**④ 什么结果意味着继续？**
若 α 增大同时出现"perturbation 导致的正常样本响应下降"与"真实 defect 响应/检测能力稳定下降"，则存在值得用真实光照数据验证的 trade-off 苗头；若 α 对 photometric perturbation 无帮助或 defect sensitivity 完全不变或趋势混乱，则停止围绕 α 调参。

### 实验设置

- **Dataset**: MVTec AD Bottle（train 209 / test good 20, broken_large 20, broken_small 22, contamination 21）
- **Model**: PatchCore，backbone `wide_resnet50_2`，layers layer2+layer3，coreset_sampling_ratio=0.1，num_neighbors=9，seed=0（与归档 baseline 相同）
- **α**: 0, 0.25, 0.5, 0.75, 1.0（每 α 独立 fit 一次）
- **Synthetic perturbations**（只施加于正常 test 图）：
  - `brightness 0.7 / 1.3`：线性缩放后 clamp 到 [0,1]（brightness 1.3 时约 37% 像素被裁剪饱和）
  - `gamma 0.7 / 1.3`：output = input^(1/γ)，即 **gamma 0.7 变暗、gamma 1.3 变亮**（已实测确认：mean 0.54→0.47 / 0.54→0.60，无裁剪）
  - `original`：原图
- **评分**：原始 PatchCore pred_score（max-NN 距离）。跨 α 的绝对分数**不可直接比较**（IN 改变特征/分数尺度），跨 α 只比较：同 α 内配对差值、秩相关指标（AUROC/AUPR）、同 α 内 defect-normal 分离度。
- **阈值**：α=0 正常图 original 的 max score = 28.173（recall 全部 100%，饱和，不具区分力）。

### 结果

**α=0 baseline 校验：通过。** image AUROC=1.0000、pixel AUROC=0.98557、pixel F1=0.72704、pixel AUPR=0.77112，与 2026-09-30 归档 alpha=0 结果完全一致。

**Robustness side（正常图，配对 delta = perturbed − original，跨 α 可比）：**

| condition | α=0 | α=0.25 | α=0.5 | α=0.75 | α=1.0 |
|---|---:|---:|---:|---:|---:|
| brightness 0.7（变暗） | +1.28 | +0.81 | +0.80 | +0.86 | +1.04（U 型回升） |
| brightness 1.3（变亮，37% 裁剪） | +1.18 | +0.95 | +0.73 | +0.63 | +0.65（单调下降） |
| gamma 0.7（变暗） | +1.28 | +0.81 | +0.78 | +0.70 | +0.70（单调下降） |
| gamma 1.3（变亮） | +0.31 | +0.22 | +0.17 | +0.11 | +0.01（趋近 0，α=1 时 Wilcoxon p=0.31 不显著） |

α=0 时所有扰动都显著推高正常图分数（Wilcoxon p<1e-4，18-19/20 样本为正），即 baseline 确实受光度扰动影响；变暗类扰动（约 +1.28）远强于平滑变亮（+0.31）。

**Sensitivity side（真实 defect 图，无扰动）：**

| 指标 | α=0 | α=0.25 | α=0.5 | α=0.75 | α=1.0 |
|---|---:|---:|---:|---:|---:|
| image AUROC | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| pixel AUROC | 0.98557 | 0.98541 | 0.98481 | 0.98376 | 0.98218 |
| pixel AUPR | 0.77112 | 0.77044 | 0.76287 | 0.74929 | 0.73129 |
| 分离度 d'（broken_large） | 17.6 | 15.1 | 14.0 | 13.2 | 11.5 |
| 分离度 d'（broken_small） | 16.3 | 14.6 | 14.0 | 13.7 | 12.4 |
| 分离度 d'（contamination） | 14.8 | 13.2 | 12.7 | 12.6 | 11.5 |

原始 defect 绝对分数呈 U 型（先降后升），但这是分数尺度混淆：α=1 时正常图 original 分数本身从 24.7 涨到 29.3（整体分数膨胀）。同 α 内的分离度 d' 与秩相关 pixel 指标才是跨 α 可比的量。

**六问回答：**

- **Q1** synthetic photometric shift 是否让 baseline normal score 上升？——**是**，α=0 下四种扰动均显著推高（+0.31 ~ +1.28）。
- **Q2** α 增大后 photometric sensitivity 是否下降？——**4 个条件中 3 个单调下降**（brightness 1.3、gamma 0.7、gamma 1.3；gamma 1.3 在 α=1 几乎被完全吸收）；**brightness 0.7 例外**，呈 U 型（α=1 回升到 +1.04）。
- **Q3** α 增大后真实 defect sensitivity 是否下降？——**image-level 无变化**（AUROC 饱和于 1.0）；**pixel-level 单调轻度下降**（pixel AUROC 0.9856→0.9822，pixel AUPR 0.7711→0.7313）；同 α 内 defect-normal 分离度 d' 单调下降（broken_large 17.6→11.5 最明显）。
- **Q4** defect type 趋势是否不同？——**没有明显分化**：三类 defect 的 d' 下降幅度接近（-3.3 ~ -6.1），未出现"contamination 独降、broken 稳定"的 defect-specific 行为。
- **Q5** 是否出现 robustness↑ + defect sensitivity↓ 的稳定趋势？——**苗头存在但温和**：robustness 改善（3/4 条件）伴随 pixel-level 指标单调轻度下降；image-level 检测在该单类别饱和设置下无可见代价。
- **Q6** 是否由极少数样本驱动？——**不是**：各条件下 13-19/20 样本 delta 为正，top-3 样本只贡献总正增量的 25-46%，且逐样本轨迹（fig4）显示 α=1 的分数上移是普遍模式。

### Observations

- α-IN 对平滑光度变换（gamma 类）的吸收效果最好；对带裁剪饱和的 brightness 1.3 和变暗类 brightness 0.7 仍有残余敏感性，后者在 α=1 回升，说明 IN 并未消除所有光度敏感成分。
- 变暗类扰动（+1.28）远强于平滑变亮（+0.31）：扰动强度本身不对称。
- α=1 出现整体分数膨胀（normal original 24.7→29.3），三类 defect 同步 U 型回升——这是跨 α 比较绝对分数时的主要混淆，已在分析中用同 α 内配对差值和秩指标规避。
- 本次未出现 OOM；RTX 5060 Laptop 8GB 顺序执行 5 个 α 正常完成（每 α 约 1.5 分钟）。
- 实现问题：Anomalib 在 Windows 下复用同一 default_root_dir 时第二次 fit 会因版本目录清理失败报 `SHFileOperationW 0x2`，已改为每 α 独立目录规避。

### Conclusion（Case 判定）

**属于 Case A（trade-off 苗头存在，需真实光照验证），附带两点保留**：(1) defect-side cost 主要体现在 pixel-level 指标与同 α 分离度上，image-level 在当前饱和设置下无可见变化；(2) 未观察到 defect-specific 分化，Case B 不成立。依据当前结果**不支持**继续围绕 α 精细调参（如 0.1/0.2/0.3 扫描）。

### Next Step

- 本探针的第一轮低成本筛查已完成，**停止扩展 synthetic 实验**（不做 defect × illumination 二维实验、不加扰动种类）。
- 若继续该方向：需要真实 multi-illumination 数据验证 trade-off 是否在真实光照下存在。此前调查的两个来源受阻（CSEM-MISD 下载失败；MVTec AD 2 光照划分不满足需求），需先解决数据问题再决定是否重启。
- 若不解决数据问题：当前探针暂停，与 2026-09-30 归档状态一致。

### 文件

- 脚本：`scripts/experiment1_illumination_tradeoff.py`（主实验）、`scripts/analyze_experiment1.py`（汇总+图）、`scripts/analyze_experiment1_supplementary.py`（Q6+分离度）
- 结果：`results/experiment1_illumination_tradeoff/`（raw_results.csv 815 条、summary_results.csv、metrics.json、analysis_summary.json、supplementary_analysis.json、figures/fig1-fig4）
- 复用实现：`experiments/2026-09-30_illumination_sensitivity_exploration/falpha_patchcore/falpha_patchcore.py`（未修改）

---

## 2026-10-01｜Experiment 1B: Defect-Specific α Sensitivity Screening

### 实验性质（边界声明）

**正式 Mini Experiment**，目的不是证明想法正确，而是快速、诚实地判断「不同 defect type 对 α-IN representation probe 是否存在稳定差异」。

- 本实验**不做任何 synthetic illumination perturbation**（无 brightness/gamma/exposure/shadow）。
- 只研究**内部 representation probe**：α-IN 对不同 defect type 的影响。
- α 只是「原始特征 F 与 IN 特征 F_IN 的线性混合权重」，**不是**「去除了多少光照信息」，也**不是**提出的创新方法。

### 预注册四句话（实验前记录）

- **① 怀疑什么？** 当 α 增大时，不同 defect type 的异常表示可能不同步变化：结构明显的大缺陷（broken_large）可能相对稳定，而较小、弱对比度、外观/纹理相关缺陷（contamination）可能更敏感。
- **② 干什么？** 在 MVTec AD bottle + 现有 PatchCore + α-IN 上，只改变 α ∈ {0, 0.25, 0.5, 0.75, 1.0}，其余条件不变，测试原始 good / broken_large / broken_small / contamination。
- **③ 看什么？** 三类 defect 随 α 的 image-level AUROC、Recall/TPR、raw score distribution、individual trajectory、d' 分离度，以及 pixel-level AUROC、anomaly map、defect area 分析。
- **④ 什么结果意味着继续？** 若不同 defect type 在多个 α 下出现稳定、明显、可重复、非少数样本导致的不同响应曲线，则继续；否则如实报告、暂停。

### 实验设置

- **Dataset**：MVTec AD bottle（train/good 209 → 划 20 作 validation、189 进 memory bank；test/good 20、broken_large 20、broken_small 22、contamination 21）
- **Model**：PatchCore，backbone `wide_resnet50_2`，layer2+layer3，coreset_sampling_ratio=0.1，num_neighbors=9，input_size 256×256
- **IN 位置**：`generate_embedding` concat 后、reshape 前；affine=False；α=0 直接返回原始 feature（bit-wise 一致）
- **Seed**：0（固定 python/numpy/torch CPU+CUDA 随机种子）
- **阈值规则**：`tau_alpha = max(validation normal scores)`（最保守，验证集 FPR=0），三类 defect 共享，不用 test 数据/defect 标签调阈值

### 结果（六问回答）

- **Q1 α=0 是否复现 baseline？** —— **是**。α=0 直接返回原始 feature（代码层 bit-wise 一致），test/good 分数（mean 24.7，range 21.5-28.2）与 Experiment 1 原始 PatchCore 分数量级完全一致。
- **Q2 是否出现 defect-specific 分化？** —— **是，且方向与假设相反**。同 α 内标准化分离度 d'：

| alpha | broken_large | broken_small | contamination |
|---|---:|---:|---:|
| 0 | **13.63** | 8.31 | 3.88 |
| 0.25 | 11.46 | 8.01 | 3.91 |
| 0.5 | 10.31 | 8.16 | 4.10 |
| 0.75 | 8.55 | 7.97 | 4.18 |
| 1.0 | **7.32** | 7.73 | **4.27** |

  - **broken_large**：d' 单调大幅下降 13.63→7.32（-46%，几乎腰斩）
  - **broken_small**：基本稳定 8.31→7.73（-7%）
  - **contamination**：反而缓慢上升 3.88→4.27（+10%）

  初始假设（contamination 最敏感）被推翻，实际最敏感的是 broken_large。

- **Q3 是否由少数样本驱动？** —— **否**。broken_large 的下降在 16/20 样本出现、contamination 的上升在 16/21 样本出现；类别均值曲线与个体曲线形态一致（普遍 U 型）。
- **Q4 是否 size confound？** —— **部分，但不完全**。broken_large 类内 corr(area, delta)=-0.747、broken_small=-0.725（面积越大下降越多，size 效应真实存在）；但 contamination 面积中等（0.085 > small 的 0.031）却反向上升，无法用 size 解释。回归 type-only R²=0.292 > area-only R²=0.130，defect type 是更强解释变量。
- **Q5 是否只是 scale 改变？** —— **否**。good mean 24.7→29.3（+19%）而 broken_large 63.9→58.5（-8%），方向相反，d' 变化无法用单一 scale 因子解释。
- **Q6 heatmap 是否一致？** —— **一致**。三类缺陷定位在所有 α 下保持准确（与 GT 重合，无漂移），变化在响应强度/范围。

### 如实记录的问题

- image-level AUROC 在所有 α 下全部饱和于 1.0，无区分度；分化只能靠 score-level 分离度 d' 观察。
- 预注册阈值规则导致 test/good FPR=1.0：validation（train 分布）分数系统性低于独立 test/good（val max ~20.4 < test min ~21.5），是「train-derived normal 与 test normal 存在分布 gap」的真实统计发现，非 bug，但使 Recall/FPR 失去区分度。
- 单类别、单 seed、每类 n=20-22，样本量小。

### Conclusion（Case 判定）

**A 与 C 之间，偏向 A（defect-specific difference 明确存在），附 C 限定**：三类 d' 曲线稳定分化（单调降 / 平稳 / 缓慢升）、跨样本普遍、阈值无关、可复现；但 broken_large vs broken_small 的差异部分可由 defect area 解释，contamination 的上升则是真正 type-specific 现象。

### Next Step（供决策，不自动执行）

1. **多 seed 验证**（seeds={0,1,2}）确认 d' 分化非 coreset 随机性——成本低，优先。
2. **跨类别验证**（cable/screw 等 defect 更多样的类别）检验分化是否普遍。
3. **size-controlled / matched-area 分析**彻底剥离 size confound。
4. 若多 seed + 跨类别稳定，则「defect-specific representation sensitivity」可作为正式研究问题推进；若 seed 敏感，按判据 D 暂停。

### 文件

- 脚本：`scripts/experiment1b_defect_sensitivity.py`（主实验）、`scripts/analyze_experiment1b.py`（汇总+图）
- 结果：`results/experiment_1b/`（README、config/、summary/、figures/、heatmaps/、raw/all_sample_scores.csv；`raw/anomaly_maps.npz` 约 89MB 不上传 Git）
- 数据集侦察：`docs/msc_dataset_analysis.md`、`docs/csem_dataset_analysis.md`、`docs/msc_download_guide.md`、`datasets/MSC-AD/access.md`（MSC-AD / CSEM-MISD / BGA 多光照数据集调查）

---

## 2026-10-02｜Experiment 1C: Multi-Seed Stability Validation

### 实验性质（边界声明）

**稳定性验证实验**。目的不是发现新结果，而是给 Experiment 1B 的 defect-specific α-IN response「办身份证」：排除 PatchCore coreset sampling / feature randomness 导致的偶然现象。

- 保持 1B **所有实验条件不变**（dataset/model/α-IN/阈值规则/split 规则），唯一允许变化的是 random seed。
- 复用 1B 的 `run_screening`（import 复用，实验逻辑零改动），仅按 seed 分结果目录。

### 预注册四句话（实验前记录）

- **① 怀疑什么？** 1B 观察到的 broken_large 大幅下降 / broken_small 稳定 / contamination 反向，可能只是 seed=0 的随机波动，需要排除 coreset sampling 随机性。
- **② 干什么？** 固定除 seed 外全部条件，seed ∈ {0, 1, 2}，重复 1B 全流程（每 seed 独立 validation split + 每 α 独立 fit）。
- **③ 看什么？** 每 seed × alpha × defect 的 d'；Δd' = d'(α=1) − d'(α=0) 的跨 seed mean/std；三类 defect 曲线是否跨 seed 保持形态。
- **④ 什么结果意味着继续？** broken_large 三个 seed 全部明显下降、broken_small 变化小、contamination 方向不同且 std 小 → 稳定存在，进入 1D；若 seed 间方向混乱 → 1B 可能是随机产物，重新设计。

### 实验设置

- **Dataset**：MVTec AD bottle（每 seed：209 train/good → 20 validation + 189 bank；test 20 good + 20/22/21 defect）
- **Model**：PatchCore，wide_resnet50_2，layer2+layer3，coreset 0.1，num_neighbors 9
- **Seeds**：0, 1, 2（固定 python/numpy/torch CPU+CUDA）
- **α**：0, 0.25, 0.5, 0.75, 1.0（每 α 独立 fit）
- **阈值规则**：tau_alpha = max(validation normal scores)，与 1B 一致

### Smoke 复现校验

seed=0 smoke（每类 5 张）的 α=0 阈值 = 20.433，与 1B 全量 validation max ~20.4 一致；seed=0 的 validation split 与 1B 完全相同。三个 seed 的 validation split 各不相同，确认多 seed 真实覆盖 split + coreset 两层随机性。

### 结果

**d' 曲线（三 seed 形态几乎重合，Figure 1）：**

| alpha | broken_large (s0/s1/s2) | broken_small (s0/s1/s2) | contamination (s0/s1/s2) |
|---:|---|---|---|
| 0 | 13.63 / 13.16 / 12.87 | 8.31 / 8.09 / 8.07 | 3.88 / 3.71 / 3.78 |
| 1.0 | 7.32 / 7.45 / 7.30 | 7.73 / 7.89 / 7.95 | 4.27 / 4.27 / 4.32 |

**Δd' = d'(α=1) − d'(α=0) 跨 seed 统计（sample std, ddof=1）：**

| defect | seed0 | seed1 | seed2 | mean | std |
|---|---:|---:|---:|---:|---:|
| broken_large | -6.31 | -5.71 | -5.57 | **-5.87** | **0.39** |
| broken_small | -0.58 | -0.20 | -0.12 | -0.30 | 0.24 |
| contamination | +0.40 | +0.56 | +0.54 | **+0.50** | **0.09** |

### 分析要点

- **现象跨 seed 高度稳定**：broken_large 三个 seed Δd' 全部 < -5.5，std 0.39（约效应量 7%）；contamination 三个 seed 全部为正，std 0.09。
- **方向完全一致**：不存在「seed0 降 / seed1 升 / seed2 平」的混乱模式；三类相对排序（large 降 >> small 平 >> cont 升）在所有 seed 中不变。
- **整条曲线逐 α 对齐**：不仅端点稳定，α-response 曲线在三 seed 间几乎重合，说明 α 的作用是确定性系统效应，随机性只带来 ±0.4 内的水平抖动。
- **效应量层级**：|Δd'(large)| ≈ 12 × |Δd'(cont)|，两组置信区间完全不重叠。

### Conclusion（Case 判定）

**CONTINUE —— defect-specific α sensitivity 稳定存在，不是 coreset/feature 随机性的偶然产物。** 预注册判断标准四条全部通过：

1. broken_large 三 seed 全部明显下降 ✅（-6.31 / -5.71 / -5.57）
2. broken_small 变化较小 ✅（|Δd'| ≤ 0.58）
3. contamination 方向不同或不下降 ✅（全部为正）
4. broken_large std 较小 ✅（0.39 << 1.0）

### Next Step

按协议进入 **Experiment 1D: Size Confound Analysis**：matched-area 比较彻底剥离 size confound，回答「控制面积后 type 效应是否仍存在」（1B 已知类内 corr(area, delta) ≈ -0.73~-0.75，size 效应真实存在）。

### 文件

- 驱动脚本：`scripts/experiment1c_multiseed.py`（seed 循环 + 目录隔离）
- 分析脚本：`scripts/analyze_experiment1c.py`（d' 表、Δd' 统计、Figure 1/2、verdict）
- 目录：`experiments/exp1c_multiseed/`（configs/seed_{0,1,2}.yaml + README）
- 结果：`results/experiment_1c/seed_{0,1,2}/` + `summary/`（verdict.json 等）+ `figures/`（dprime_curves_per_seed.png、delta_dprime_stability.png）；`raw/`（含大体积 npz）与 `logs/` 不上传 Git

---

## 2026-10-02｜Experiment 1D: Size-Controlled Defect Sensitivity Analysis

**解释实验**（非模型优化）。完整复用 1B/1C 逐样本 score 与 GT area，零 fit/predict，零模型/口径修改。核心问题：控制 defect area 后，defect type 是否仍能解释 α-IN sensitivity 的差异？

### 预注册口径（冻结，执行中未更改）

- normal reference = test/good 20 张，每 (seed, alpha) 独立 μ/σ
- PRIMARY（sample-level）：`Δz = z(α=1) - z(α=0)`，`z = (score - μ_good) / σ_good`
- SECONDARY：`slope_z`；GROUP-LEVEL CONTEXT：1B/1C 的 d'（不作单样本 sensitivity）
- 主分析 per-image 三 seed mean；禁止把 63×3 当 189 独立样本
- area = GT defect pixels / total pixels；matched-area caliper = 全样本 area 的 MAD（0.03744，看结果前锁定）

### 数据审计

63 样本（20/22/21），缺失 mask=0、area=0 defect=0、重复=0、三 seed test samples 完全一致、三对 type 面积 common support 充足（overlap 0.553/0.780/1.000）。无需重新运行任何模型部分。

### Metric decomposition sanity check（执行中追加的诊断）

Δz 与 1C Δd' 方向分歧（如 contamination Δz<0 而 Δd'>0）不是 bug：二者回答不同问题。分解（3-seed mean，α=0→1）显示 **defect variance 响应是类型分化的真正载体**：

| type | Δmean_gap | Δstd_good | Δstd_defect | Δmean_z | Δd' |
|---|---:|---:|---:|---:|---:|
| broken_large | -8.44 | +0.17 | **+1.73** | -4.56 | -5.72 |
| broken_small | -3.40 | +0.17 | **-0.50** | -2.44 | -0.30 |
| contamination | -2.44 | +0.17 | **-2.30** | -1.95 | **+0.48** |

α-IN 同时作用于 mean separation / within-defect variance / normal variance 三个分量，且对不同 type 的作用结构不同。d' 混合三者；Δz 只反映相对 contemporaneous good 分布的标准化距离。"敏感性"不是单一统计量可完全描述的——记录为实验发现。

### 六问回答

- **Q1 area 分布**：large(mean .117) > cont(.085) > small(.031)，但三对均有 common support。
- **Q2 area-Δz 关系**：type-dependent——两类 broken 类内强负相关（-0.76/-0.75），contamination 类内**弱正相关**（+0.21）。
- **Q3 area-only**：R²=0.115，解释有限。
- **Q4 +type**：R²→0.304，**ΔR²(type|area)=0.188**；反向 ΔR²(area|type)=0.025。type 不可被 area 替代，area 大部分被 type 吸收。
- **Q5 matched-area**：large vs contamination 15 对，同面积 Δz 仍差 **-2.23**；残差 large **-1.10** / small -0.06 / cont **+1.11**，控制面积后 type 分化清晰。
- **Q6 contamination**：面积匹配后仍整体高于 matched broken（-2.24 vs -3.18，28 对）；类内正相关与 broken 类反向，无法由 area 单独解释。注意 Δz 口径下其 Δz mean=-1.95（也下降，只是显著慢于 broken 类）；1C 中 d' 上升是 group-level 方差收缩驱动的现象。

### Multi-seed robustness

各 seed 独立重复主要分析：overall corr -0.33/-0.34/-0.35，type 均值排序三 seed 完全一致（large 最负、cont 最不负）。方向不依赖单一 seed。

### 判定：CASE_A — TYPE EFFECT REMAINS

控制 area 后 type 仍提供显著额外解释力；contamination 的响应模式无法由 size 单独解释。限定：d' 的 group-level 上升部分来自 within-defect variance 收缩，准确表述为 **defect type 影响 α-IN 对缺陷分布的完整作用结构（mean + variance）**。

### 局限

n=63 单类别；large-small 匹配仅 5 对（不可靠）；Δz 依赖 n=20 good 估计 μ/σ；matched-area 是观察性控制；单 backbone/detector 外推性未知。

### Next Step

**Experiment 1E — Cross-Category Validation**（cable/screw 等），检验 type effect 与 decomposition 结构是否跨类别成立。未自动开始，等待确认。

### 文件

- 脚本：`scripts/analyze_experiment1d_size_control.py`、`scripts/metric_decomposition_check.py`
- 目录：`experiments/exp1d_size_control/`（README + 全部口径/规则/结果记录）
- 结果：`results/experiment_1d/summary/`（audit/statistics/correlation/regression/matched_pairs/metric_decomposition/verdict）、`figures/`（6 张）、`tables/sample_level_response.csv`

---

## 2026-10-02｜Experiment 1E: Formal Cross-Category Pilot

**解释实验**（非模型优化）。验证 1B–1D 在 Bottle 上发现的 defect-dependent α-IN response heterogeneity 是否跨 MVTec AD 类别成立。预注册协议驱动，配置冻结后零修改。

### 设计（冻结于 results/experiment_1e/config.json）

- 5 categories（bottle/grid/cable/screw/hazelnut）× 3 seeds × 5 α = **75 conditions**；bottle 复用 1C raw + 1B area 重建（不重训），新增 60 fits 串行运行
- α-IN / backbone / coreset / z / d' 口径与 1B/1C/1D 完全一致；z 用 contemporaneous category×seed×α test/good（ddof=1）
- 每 category 完成后自动 sanity checkpoint（completeness/count/finite/good normalization/discovery/area/重复错位）——4 类全部 PASS，无 OOM 无中断
- 资源实测：coreset 22.9k–40.0k patches，peak GPU 2.8–4.8 GB（`max_memory_allocated()` 实测）

### Bottle 重建 equivalence check

9/9 PASS（3 defect types × 3 seeds 的 Δz/Δdefect_std/Δd' 与 1D 确认值完全一致，阈值 1e-6）。

### 核心结果

**Seed stability**：25/25 defect types 至少一个 response 维度 3/3 seeds 符号一致——现象普遍稳定，非随机产物。方向分化是关键：

- **Δdefect_std**：13 POS vs 11 NEG vs 1 MIXED——方差响应强烈类型分化
- 5/5 类别内部出现方向分化（cable/hazelnut 在 3 个维度分化；grid 全类型同向但幅度差 10 倍）

**方差收缩模式跨类别复现**（Δz<0 & Δdefect_std<0 & Δd'>0，全部 3/3 稳定）：

| pattern | Δz | Δstd | Δd' |
|---|---:|---:|---:|
| bottle/contamination | -1.95 | -2.30 | +0.48 |
| cable/bent_wire | -1.07 | -3.51 | +2.19 |
| hazelnut/print | -4.63 | -2.77 | +0.70 |

1D 发现的"α-IN 收缩 defect 方差 → sample z 降但 group d' 反升"模式有两个新增类别的正式确认实例（hazelnut print 即 smoke test 预测的正式验证）。

**Area control**（Phase 15，25 types）：

| response | A: log(area) | B: +category |
|---|---:|---:|
| Δz | R²=0.015 | R²=0.692 |
| Δdefect_std | R²=0.079 | R²=0.796 |
| Δd' | R²=0.052 | R²=0.246 |

area-only 解释力接近零 → **area contributes but is NOT sufficient**；异质性载体在 defect identity（Model C 饱和 R²=1.0 为 one-hot 饱和拟合，仅作方向参考）。

### 判定：CASE_A — CROSS_CATEGORY_HETEROGENEITY_SUPPORTED

A1（≥2 新增类别方向分化：cable/screw/hazelnut）+ A2（稳定覆盖 25/25）+ A3（area 不充分）+ A4（方差收缩模式复现 3 例）全部满足。

### 边界与局限

不做机制归因（texture/structure/frequency/clustering 属 1F）；单 backbone/detector/size；bottle 为重建数据（equivalence check 缓解）；grid texture 类方差普遍放大（Δstd 至 +11.4）现象记录待 1F 解释。

### Next Step

**停止，不自动进入 1F。** 人工判断：方差收缩模式与 grid texture 方差放大是否值得机制级研究。

### 文件

- 脚本：`scripts/experiment1e_{runner,orchestrator,bottle_reconstruct,analysis,figures_area,verdict}.py`
- 实验记录：`experiments/exp1e_cross_category/README.md`
- 结果：`results/experiment_1e/`（config.json 冻结配置、`<category>/seed_*/` 原始 schema、`analysis/`（含 verdict.json）、`figures/figure1~4`）

---

> ℹ️ 本节为 **2026-09-25 首个 baseline 里程碑**的历史记录（bottle / broken_large 四联图），
> 保留作为项目起点证据；研究阶段与当前结论见上文 §0–§8。

## PatchCore Bottle Result

当时选择 `broken_large` 异常样本进行可视化。

结果包括：

1. **Original Image**
   - 输入模型的 Bottle 图片；

2. **Ground Truth**
   - 数据集提供的真实缺陷区域；

3. **PatchCore Anomaly Map**
   - PatchCore 输出的连续像素级异常分数；
   - 高响应区域表示该位置与正常特征差异较大；

4. **Predicted Mask**
   - 根据异常阈值得到的最终异常区域。

实验中可以观察到：

> PatchCore Anomaly Map 的主要高响应区域与 Ground Truth 缺陷位置基本一致。

同时，Predicted Mask 能够定位主要缺陷区域，但预测边界与 Ground Truth 仍存在一定差异。

### Result Image

![PatchCore Bottle Result](results/patchcore_bottle_broken_large.png)

---

> ⚠️ **以下三节（Current Progress / Next Step / Current Goal）为 2026-09-25 baseline 阶段的原始记录，已过时。**
> 当前阶段、结论与下一步请以 **§0 当前状态快照 / §8 已冻结结论 / §9 下一步** 为准。

# （历史）Current Progress — 2026-09-25 阶段快照

当时已经完成：

## 阶段 1：工业异常检测入门 ✅

理解：

- 工业异常检测任务
- 正常 / 异常样本
- Ground Truth
- 重建式异常检测
- 特征式异常检测

## 阶段 2：实验环境搭建 ✅

完成：

- Conda
- PyTorch
- CUDA
- Anomalib
- Jupyter
- MVTec AD

## 阶段 3：PatchCore Baseline ✅

完成：

- Dataset
- Model
- Fit
- Memory Bank
- Test
- Predict
- Anomaly Map
- Predicted Mask
- 结果保存
- Notebook 整理

## 阶段 4：实验分析与 Baseline 扩展 🚧

接下来开始从：

> **“把模型跑起来”**

逐渐进入：

> **“分析为什么得到这样的结果，并设计自己的实验。”**

---

# （历史）Next Step — 2026-09-25 阶段

> 已过时；这些 baseline 阶段的小任务后来被并入 1B 起的正式实验链。

当时的原则是不急着直接修改模型。

优先完成以下几个小任务：

1. **记录并理解 PatchCore 的评价指标**
   - Image AUROC
   - Image F1
   - Pixel AUROC
   - Pixel F1

2. **分析图像级检测与像素级定位的区别**
   - 为什么 Image-level 指标很高；
   - 为什么 Pixel-level F1 相对较低；
   - 结合 Ground Truth 和 Predicted Mask 分析误差。

3. **完成第一个 Mini Experiment**
   - 修改 `coreset_sampling_ratio`
   - 例如比较：
     - 0.05
     - 0.10
     - 0.20
   - 观察检测性能、运行时间和 Memory Bank 规模变化。

4. **逐步扩展 Baseline**
   - PatchCore
   - PaDiM
   - EfficientAD

最终逐渐形成：

```text
Baseline 复现
      ↓
指标分析
      ↓
参数实验
      ↓
模型对比
      ↓
发现问题
      ↓
提出小型改进
      ↓
论文方向探索
```

---

## （历史）Current Goal — 2026-09-25

当时短期目标：

> **从“能够跑通 PatchCore”推进到“能够独立分析 PatchCore 实验结果，并完成第一个小型对比实验”。**

暂时不追求复杂模型改进。

优先保证：

**每学一个方法，都留下一个能够运行、能够解释、能够复现的实验成果。**