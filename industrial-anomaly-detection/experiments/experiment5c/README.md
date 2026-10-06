# Experiment 5C — Geometry-Guided Category-Adaptive α v1

> 状态：**已完成（45/45 runs，sanity S1–S15 全 PASS）→ FINAL VERDICT: CASE_B — Harm Reduction Only**（driver = v1B / Geometry Conservative）。
> 协议冻结于运行前；Results / Verdict / Limitations / Next 为运行后追加。
> 前序：5A ✅（bottle CASE A conditional，`ba11038`）→ 5A-H ❌ CASE C（fixed G2 STOP，`ff4a1b2`）
> → 5B ✅（`24d1b22`）→ 5B-C ✅ **CASE_A (FINAL)**（`5d6ee98`）。
> 本实验**不新增 α grid、不做 α search、不训练 mapping 模型**。

---

## 0. Paper Navigation

📍 **Stage ⑦ — 从「预测性发现」进入「方法验证」**。Stage ⑥ 结论：`Normal Geometry → Normalization Tolerance = CASE_A (FINAL)`
（`radius_ratio_L3L2` ρ=−0.90、LOCO 5/5、seed 稳定；Group C sensitivity 全 null，机制性解释未确立）。

### ① 我怀疑什么？

一个固定 normalization strength α 不适合所有工业类别。5A-H 已证明 category-level tolerance 差异极大
（bottle/grid 被 strong α 系统性伤害，hazelnut 几乎无损甚至受益）。5B/5B-C 已证明：**仅用 normal training
images 的 geometry（`radius_ratio_L3L2`）就能预测哪个类别"怕"normalization**。

因此怀疑：**可以仅根据 normal training images 的 geometry，为不同 category 分配不同 α，从而比统一 fixed α
更安全或更有效。**

### ② 这个实验准备干什么？

构造最简单的 **Geometry-Guided Category-Adaptive α v1**：只用 normal training images 计算 geometry，
用 5B 已冻结的 predictor `radius_ratio_L3L2`，经**同一条 rank→α 映射**为每个 category 生成 α，
然后与 Original PatchCore / Best Fixed α / Sensitivity-Guided α control 直接比较。**零新 α、零拟合。**

### ③ 我要看什么？

1. adaptive α 是否优于 best fixed α；2. 是否降低 high-damage categories 的 normalization damage；
3. 是否不会靠牺牲 low-damage categories 换平均提升；4. 是否跨 3 seeds 稳定；
5. 是否不是单一 category 驱动；6. Geometry-Guided 是否优于 Sensitivity-Guided control。

### ④ 什么结果意味着继续？

- **PASS**：相比 best fixed α 稳定改善 + 跨 seed + 非单类别驱动 + high-damage harm 明显下降
  + 优于 Sensitivity control → 进入完整方法实验。
- **PARTIAL**：总体平均不明显，但 bottle/grid 等 high-damage categories 退化明显减少、low-damage 类别无恶化
  → 保留 “safe normalization / harm avoidance” 方向。
- **FAIL**：≈ fixed α / 差于 fixed α / 完全由单一 category 驱动 / Sensitivity 同样有效 → **方案①停止**，
  不得无限调参救结果。

---

## 1. Frozen Inputs（全部复用历史冻结结果，零新资产）

| 用途 | 文件 |
|---|---|
| primary predictor / secondary / control | `results/experiment_5b_final/summary/normal_only_predictors.csv`（每 category×seed 一行） |
| predictor 注册与冻结记录 | `results/experiment_5b_final/reference/predictor_registry.csv`、`predictor_registry_group_c.csv`、`group_c_freeze.json` |
| historical fixed-α 与 target | `results/experiment_5a_h/summary/per_unit.csv`（B0/B2/C2/C3/G2，75 runs） |
| 5B 相关性/LOCO 证据 | `results/experiment_5b_final/summary/correlation_summary.csv`、`loco_predictions.csv` |

predictor 角色（**冻结，禁止事后更换**）：
- **primary**：`radius_ratio_L3L2`（5B：ρ=−0.90，LOCO 5/5，seed 稳定）
- **secondary / identity check**：`eff_dim_L3`（ρ=−0.90，LOCO 4/5）——只做 assignment 层面的 identity 核对，**不作为替代方案**
- **negative mechanistic control**：`sens_radius_L2`（Group C primary；5B：ρ=−0.30，LOCO 2/5）

## 2. Historical α grid（机械恢复，禁新增）

5A-H 中在 **5 categories × 3 seeds 全覆盖** 的 uniform α（同一 layer-wise α-IN 干预族）：

```text
α_grid = [0.0, 0.5, 0.601369125, 0.8018255]        # 来自 B0 / B2 / C2 / C3（exact 值）
K = 4
```

排除项（并记录理由）：`α ∈ {0.25, 0.75, 1.0}`（5A 的 B1/B3/B4）**只在 bottle/seed0 跑过**，
无跨类别覆盖，纳入会需要新跑 baseline 且造成不公平比较 → 不纳入。G2 = (0.45273825, 0.75) 是
**层间非均匀** config，不属于 uniform fixed α → 不纳入 α_grid。

## 3. Best fixed α（冻结规则，零新自由参数）

用 5A-H **已冻结**的 PAIR-WIN 准则（5A-H §2：ΔRobustness < 0 AND ΔPreservation ≥ −ε，**ε = 0.10**，d′ 绝对单位）
对每个 uniform α 统计其相对 α=0（B0）取胜的 (category, seed) unit 数：

| fixed α | config | wins vs B0（/15） | 5-cat mean d′ | 5-cat mean \|Δz\| |
|---|---|---|---|---|
| 0.5 | B2 | **7** | 4.6670 | 0.2082 |
| 0.601369125 | C2 | 3 | 4.5078 | 0.1895 |
| 0.8018255 | C3 | 3 | 4.2272 | 0.1902 |
| (G2, 非 uniform) | G2 | 3 | 4.4269 | 0.1806 |

→ **best fixed α = 0.5（config B2）**；tie-break 规则：取更小的 α（更保守）。
→ **conservative ceiling = best fixed α = 0.5**（§10 规定的「不需要新阈值」机械规则：`α_max = best fixed α`）。
（说明：ε=0.10 直接复用 5A-H 冻结值，**未新增任何阈值**。）

## 4. Mapping（冻结：rank-based monotonic）

对每个 seed，将 5 个 category 按 predictor 值**升序**排名 r=1..5（r=1 最脆弱/最不耐受），
再机械映射到 legal α 集合：

```text
α(r) = GRID[ min(K-1, ceil(r * K / n) - 1) ]      n = 5, K = |legal set|
```

- 与 rank 单调不减（higher geometry → higher α）
- deterministic、category-name blind、target-label blind
- **GF / GC / SG 共用同一个 rank→α 函数**，只允许两处不同：legal α set（GC 用 ceiling）与 predictor（SG 用 control）

## 5. Policies（冻结；A/B 复用 5A-H，不重跑）

| policy | predictor | legal α set | 说明 |
|---|---|---|---|
| **A. Original** | — | α=0 | = 5A-H B0（复用） |
| **B. Best Fixed** | — | α=0.5 全体 | = 5A-H B2（复用） |
| **C. Geometry Full (GF, v1A)** | `radius_ratio_L3L2` | full grid | 直接检验 geometry-guided hypothesis |
| **D. Geometry Conservative (GC, v1B)** | `radius_ratio_L3L2` | α ≤ 0.5 | safe ceiling = best fixed α |
| **E. Sensitivity-Guided (SG)** | `sens_radius_L2` | full grid | predictor-identity control（与 GF **同 budget**） |

## 6. Policy Assignment（冻结；见 `summary/policy_assignment.csv`）

| category | `radius_ratio_L3L2`(seed-mean) | geom rank | α (GF full) | α (GC ceiling 0.5) | `sens_radius_L2` | sens rank | α (SG full) |
|---|---:|---:|---:|---:|---:|---:|---:|
| bottle | 0.885 | 1（最脆弱） | 0.0 | 0.0 | −0.465 | 1 | 0.0 |
| grid | 0.899 | 2 | 0.5 | 0.0 | +0.290 | 5 | 0.8018255 |
| cable | 1.118 | 3 | 0.601369125 | 0.5 | −0.424 | 2 | 0.5 |
| hazelnut | 1.160 | 4 | 0.8018255 | 0.5 | −0.209 | 3 | 0.601369125 |
| screw | 1.354 | 5（最耐受） | 0.8018255 | 0.5 | −0.166 | 4 | 0.8018255 |

- mean α：GF = **0.5410**，GC = 0.300，SG = **0.5410**（GF 与 SG **budget 相同**，仅 allocation 不同
  → 这是 predictor-identity 的核心对照）
- 逐 seed rank 完全一致（3/3 seeds 同序），故三 policy 的 α 分配逐 seed 相同（mapping 仍按 seed 机械计算，不硬编码）。
- GF 的语义：把强 normalization 从 bottle/grid（high-damage）挪到 cable/hazelnut/screw（low-damage），
  bottle 给 α=0。

## 7. Metrics（复用 5A-H 冻结口径，禁新造主指标）

| 类型 | 名称 | 方向 |
|---|---|---|
| primary robustness | `mean_abs_delta_z`（= mean \|ΔNormalScore_z\|，4 photometric shifts） | 越低越好 |
| primary preservation | `mean_dprime`（clean test 全集 defect d′） | 越高越好 |
| damage 口径 | `damage(α) = d′(B0) − d′(α)`（5B 的 C2 damage 同式） | 越低越好 |
| secondary（仅解释） | image AUROC / pixel AUROC / AUPRO / FPR@τ_val | — |

**Harm 指标（本实验新增的**分析**量，非新主指标）**：
- `mean_gain` = adaptive − best_fixed（per category / 5-cat mean）
- `worst_category_gain` = min over categories of gain
- `high_damage_recovery`：**预先指定** subset = {bottle, grid}（依据 5A-H/5B 历史 damage 排序，
  **不看 5C 结果**）的 gain
- `negative_transfer_count` = 相比 best fixed 明显下降的 category 数（tie band 复用冻结 ε=0.10）
- `win / tie / loss`：Geometry vs Fixed、Geometry vs Sensitivity（tie = |Δd′| ≤ 0.10）

## 8. Seeds / Categories（不变）

`categories = [bottle, cable, grid, hazelnut, screw]`；`seeds = [0, 1, 2]`（与 1E / 5A-H / 5B 完全一致）。

## 9. Statistics

- per-unit（15）与 per-category（5，mean±std over 3 seeds）paired 差；
- **LOCO robustness**：每次去掉一个 category，仅用剩余 4 类的 predictor rank 重算 rank→α 映射
  （n=4 → indices [0,1,2,3]），在被保留的 4 类上重算结论；**不使用 held-out category target 重新拟合**；
- **Predictor identity**：GF vs SG 直接配对（同 budget）；
- 不做未预注册的显著性检验；n=5 只报 descriptive。

## 10. Sanity（全部 PASS 才解释结果）

```text
S1  predictor 仅用 normal train（5B 冻结资产；无 test 图通路）
S2  无 test label / C2 damage 参与 α 分配
S3  mapping 单调（rank↑ → α 不减）
S4  mapping deterministic（重算两次完全一致）
S5  所有 category 使用同一 mapping 函数
S6  每个被分配的 α ∈ 历史 grid
S7  seeds 未变（0/1/2）
S8  categories 未变（5 类）
S9  fixed baseline 在 adaptive 评估前已冻结（freeze JSON + 5A-H 复用）
S10 Group C control 使用完全相同 mapping
S11 无 target leakage
S12 无 post-hoc predictor switching（primary 硬编码为 radius_ratio_L3L2）
S13 无 post-hoc α tuning（所有 α ∈ grid）
S14 复用历史结果文件 md5 前后不变（5A-H/5B-Final 关键 CSV）
S15 所有输出 finite（无 NaN/Inf）
```

## 11. Outputs

`results/experiment_5c/{reference,raw,summary,figures,logs}/`；
tables：`policy_assignment` / `fixed_alpha_reference` / `per_seed_results` / `per_category_results` /
`method_comparison` / `harm_analysis` / `geometry_vs_sensitivity` / `loco_robustness` /
`sanity_checks` / `final_summary`；
figures：`geometry_to_alpha_policy` / `method_comparison` / `per_category_gain_vs_fixed` /
`geometry_vs_sensitivity_control` / `harm_reduction`。

## 12. Exit Criteria（预注册）

- **CASE_A — Geometry-Guided Policy Works**：GF 稳定优于 best fixed（多 seed 一致、非单 category 驱动、
  harm 未明显恶化）+ GF 优于 SG → 方案①进入完整实验。
- **CASE_B — Harm Reduction Only**：平均性能无明显提高，但 high-damage/worst-case 明显改善、
  low-damage 基本保持、negative transfer 减少 → 转为 Geometry-Guided Safe Normalization 继续。
- **CASE_C — Adaptive Helps, Geometry Identity Unclear**：adaptive 优于 fixed，但 GF ≈ SG → 可研究
  adaptive normalization，但不能声称 5B geometry 是关键原因。
- **CASE_D — No Useful Improvement**：GF ≈ fixed 或 < fixed，或严重依赖单 category → **方案①停止**，不再调参。

## 13. Results（运行后）

### 13.1 执行与等价性（关键事实）

- **45/45 config runs 完成**（C/D/E = 3 policies × 5 categories × 3 seeds，3 workers 并行，约 40 min）。
- **raw score 级等价性（S14b）**：5C 与 5A-H 在相同 α 上的 **33 个 (category, seed, α) key、9273 行
  per-image 分数逐位一致**（max|Δscore| = 0.000e+00，max|Δτ_val| = 0.0）。
  原因：5A-H 已对全部 4 个 historical uniform α × 5 类做过测量，因此 **5C 的每个 policy 都是在冻结
  α grid 上的"重组（recombination）"**，不是新条件。这也使 5C 同时成为一条强复现性检查。
- 表内数值：adaptive 三 policy 用**本次 raw 重算**；A/B 用 5A-H 冻结 `per_unit.csv`（该 CSV 按 4 位小数
  存盘，summary 级差异 ≤5e-5 已核对为舍入，raw 分数证明一致）。
- **sanity S1–S15 全部 PASS**（含 S14：6 个历史冻结文件 md5 复核前后不变）。

### 13.2 Method comparison（5 categories × 3 seeds 平均）

| policy | mean α | mean defect d′ ↑ | mean \|ΔNormalScore_z\| ↓ | gain vs best fixed (d′) | Δrobust vs best fixed |
|---|---:|---:|---:|---:|---:|
| A. Original（α=0） | 0.0000 | 4.9852 | 0.3105 | +0.3182 | +0.1023 |
| B. Best Fixed（α=0.5） | 0.5000 | 4.6670 | 0.2082 | 0（参照） | 0（参照） |
| C. Geometry Full（v1A） | 0.5410 | 4.7019 | 0.2277 | **+0.0349** | **+0.0195** |
| D. Geometry Conservative（v1B） | 0.3000 | **4.9967** | 0.2643 | **+0.3297** | **+0.0560** |
| E. Sensitivity-Guided | 0.5410 | 4.6679 | 0.2364 | +0.0009 | +0.0282 |

### 13.3 Per-category（3 seeds 平均；d′ / \|Δz\|）

| category | α (GF/GC/SG) | A Original | B Best Fixed | C GF | D GC | E SG |
|---|---|---|---|---|---|---|
| bottle | 0 / 0 / 0 | 8.179 / 0.442 | 7.302 / 0.286 | 8.179 / 0.442 | 8.179 / 0.442 | 8.179 / 0.442 |
| grid | 0.5 / 0 / 0.8018 | 3.136 / 0.224 | 2.364 / 0.100 | 2.364 / 0.100 | 3.136 / 0.224 | 1.802 / 0.105 |
| cable | 0.6014 / 0.5 / 0.5 | 5.055 / 0.331 | 5.008 / 0.271 | 4.806 / 0.238 | 5.008 / 0.271 | 5.008 / 0.271 |
| hazelnut | 0.8018 / 0.5 / 0.6014 | 5.964 / 0.214 | 6.175 / 0.151 | 5.958 / 0.138 | 6.175 / 0.151 | 6.147 / 0.143 |
| screw | 0.8018 / 0.5 / 0.8018 | 2.593 / 0.341 | 2.487 / 0.232 | 2.203 / 0.221 | 2.487 / 0.232 | 2.203 / 0.221 |

### 13.4 Harm analysis（相对 best fixed α=0.5）

| policy | mean gain | worst category | high-damage recovery (bottle+grid) | negative transfer | pair-win units | win/tie/loss |
|---|---:|---|---:|---:|---:|---|
| C. GF（v1A） | +0.0349 | screw **−0.2831** | +0.4384 | **3/5** | **0/15** | mixed |
| D. GC（v1B） | +0.3297 | cable **0.0000** | **+0.8243** | **0/5** | 0/15 | mixed（3 tie + 2 win） |
| E. SG | +0.0009 | grid **−0.5617** | +0.1576 | 2/5 | 2/15 | mixed |

- **D（GC）**：逐 category 相对 best fixed 的 gain = bottle **+0.877**、grid **+0.772**、cable/hazelnut/screw
  **恰好 0**（因为 GC 在低损伤类别上分配的 α 就是 best fixed 的 0.5）。代价是 bottle/grid 的 robustness
  变差（+0.156 / +0.124），**但没有任何 category 的 preservation 变差**（worst-category gain = 0.000）。
- D（GC）相对 **Original（α=0）**：5 类平均 Δd′ **+0.0115**、Δ\|Δz\| **−0.0462** → **在两个轴上都不劣于
  原始 PatchCore**（逐类别：bottle/grid 完全相同；cable −0.047 d′/−0.060 \|Δz\|；hazelnut +0.211/−0.063；
  screw −0.107/−0.109）。
- **C（GF）**：把 normalization 从 bottle（α=0）挪到 cable/hazelnut/screw（α 升高），preservation 净 +0.035
  但 robustness 净变差 +0.0195，且 3/5 类别的 d′ 损失 > ε（screw −0.283、hazelnut −0.217、cable −0.202）。
  **按 5A-H 冻结 PAIR-WIN 判据，GF 对 best fixed 0/15 取胜。**
- **E（SG）**：与 GF 同 budget（mean α 0.5410）但 allocation 不同（保护 bottle+cable，把 grid 推到 0.8018），
  grid 的 d′ 比 GF 低 0.562 → SG 是三条 policy 中唯一拿到 2/15 pair-win 的（都来自 bottle：把 α 从 0.5 降到 0），
  但总体最差类别损失最大（grid −0.562）。

### 13.5 Predictor identity（GF vs SG，同 budget）

| category | α_geom | α_sens | Δd′ (GF−SG) | Δ\|Δz\| (GF−SG) | outcome |
|---|---:|---:|---:|---:|---|
| bottle | 0.0 | 0.0 | 0.0000 | 0.0000 | tie |
| grid | 0.5 | 0.8018 | **+0.5617** | −0.0044 | GF better |
| cable | 0.6014 | 0.5 | **−0.2022** | −0.0333 | mixed |
| hazelnut | 0.8018 | 0.6014 | **−0.1892** | −0.0059 | mixed |
| screw | 0.8018 | 0.8018 | 0.0000 | 0.0000 | tie |

- 5 类聚合：GF 在两个轴上都不劣于 SG（Δd′ **+0.0341**，Δ\|Δz\| **−0.0087**）。
- 但优势**集中在 grid 单一类别**（+0.562），同时在 cable/hazelnut 上落后（−0.202 / −0.189）。
  → **geometry identity：directionally consistent，但不 robust（单类别驱动），不能声称已确立。**

### 13.6 LOCO（去掉一个 category，用剩余 4 类重算 rank→α，n=4）

| 去掉 | 剩余 4 类 mean gain | 剩余 4 类 mean Δrobust |
|---|---:|---:|
| bottle | +0.1153 | +0.0261 |
| cable | +0.1415 | +0.0342 |
| grid | +0.1415 | +0.0342 |
| hazelnut | +0.0979 | +0.0278 |
| screw | +0.1144 | +0.0273 |

去掉任一类别后，结论形状不变（preservation gain 为正、robustness 略差）→ **不是单一 category 驱动**；
mapping 在 4 类上仍单调合理（未使用 held-out 的 target 重拟合）。

### 13.7 事实 / 解释 / 假设（分离）

- **事实**：GF/GC/SG 均未在两个轴上同时优于 best fixed（pair-win 0/15、0/15、2/15）；GC 无任何 preservation
  负迁移且 high-damage recovery +0.824；GC 聚合上不劣于 Original；GF 与 SG 的差别集中在 grid。
- **解释**：geometry rank 的可用价值**主要体现在"知道哪些类别不该被 normalization"**（bottle/grid），
  而不是"把 normalization 更激进地重新分配"；一旦把所有类别都推强（v1A），净收益消失。
- **假设（未验证）**：GC 的收益来自"把 fragile 类别的 α 降到 0"这一条规则本身，而非 geometry rank 的精确排序
  （matched sensitivity-conservative 对照未预注册、未运行，见 Limitations）。

## 14. Verdict（运行后）

**CASE_B — Harm Reduction Only**

按 §25 条件对**两个预注册 geometry 变体**逐一机械核对：

| §25 条件 | v1A（GF, full-range） | v1B（GC, conservative） |
|---|:---:|:---:|
| 平均性能稳定优于 best fixed（两轴同时） | ✗（Δd′ +0.035，但 Δ\|Δz\| **+0.0195** 更差） | ✗（Δd′ +0.330，但 Δ\|Δz\| **+0.0560** 更差） |
| 两轴 pair-win 复合判据 | **0/15** | **0/15** |
| 多 seed 一致 | ✓（3/3 seeds 同样 0 win，稳定失败/稳定表现） | ✓ |
| 非单一 category 驱动（LOCO） | ✓（去掉任一类别 gain 仍正） | ✓ |
| high-damage / worst-case 明显改善 | 半：HD +0.438，但 worst = screw **−0.283** | ✓ HD **+0.824**，worst = **0.0000** |
| low-damage 类别基本保持 | ✗（hazelnut −0.189、cable −0.202、screw −0.283 > ε） | ✓（cable/hazelnut/screw gain 恰好 **0**） |
| negative transfer 减少 | ✗（**3/5**） | ✓（**0/5**） |

- **CASE_A 排除**：没有任何 geometry 变体在两轴上稳定优于 best fixed（Δrobust 均为正、pair-win 0/15），
  且 GF vs SG 的 identity 优势集中在单一类别。
- **CASE_C 排除**：CASE_C 要求 adaptive 至少在 robustness 轴上优于 fixed，而 GF/GC/SG 在 robustness 轴均劣于
  best fixed α=0.5（这是"budget 相近、分配不同"的必然结果）。
- **CASE_B 成立（driver = v1B / GC）**：平均性能没有在两轴上同时提高，但
  high-damage 恢复 +0.824、worst-category 无损失、negative transfer 0/5、low-damage 类别完全保持。
- 单独看 v1A 则接近 **CASE_D**：GF ≈ best fixed（+0.035 d′ / +0.0195 robustness），0/15 pair-win，
  3/5 类别 d′ 损失 > ε。**必须明确记录这一点，不能只报 v1B。**

**必须同时声明的限定（不得省略）**

1. **v1A（激进全范围）失败**：把 normalization 更激进地重分配，并没有击败 best fixed α；geometry 排序的
   收益**不能**通过"整体推强 α"实现。
2. **v1B 的收益是"安全的减法"**：GC = 在 best fixed α 基础上，仅把 **bottle + grid** 的 α 降到 0。
   它的收益结构完全来自这 2 个高脆弱类别（+0.877 / +0.772），其余 3 类恰好不变；代价是这 2 类的
   illumination robustness 变差（+0.156 / +0.124）。因此这是 **harm avoidance / safe normalization**，
   不是"更好的 normalization allocation 提升平均性能"。
3. **predictor identity 未确立**：GF vs SG（同 budget）聚合上 GF 略优（Δd′ +0.0341、Δ\|Δz\| −0.0087），
   但优势集中在 grid（+0.562），同时在 cable/hazelnut 落后（−0.202 / −0.189）。
   v1B 没有 matched sensitivity-conservative 对照（该 policy 未预注册、未运行）→ 无法排除
   "任何 ranking 只要把 fragile 类别降到 0 就能拿到同样收益"。
4. n_category = 5，全部为 descriptive/exploratory，不做显著性主张、不做 causal claim。
5. 5C 的 45 个 run 与 5A-H 在相同 α 上 **raw 分数逐位一致**（S14b），说明本实验是**历史 α 条件的重组**，
   其证据强度受 5A-H 已有的 α×category 覆盖限制。

## 15. Limitations（运行后补充）

- n = 5 categories，且 GC 的收益只来自 2 个类别（bottle/grid），泛化性未验证。
- 5A-H summary CSV 按 4 位小数存盘；本实验以 raw 分数做等价性判据（max|Δscore| = 0）。
- 未执行未预注册的 bootstrap / permutation / 显著性检验。
- illumination 协议（brightness/gamma 0.7/1.3）与 5A-H 绑定，不外推到其他扰动。
- primary metric 沿用 5A-H 冻结口径（mean d′ / mean |ΔNormalScore_z|），未新造"对 adaptive 有利"的指标。
- **matched sensitivity-conservative control 缺失**：v1B 的 predictor identity 只能间接由 v1A 对 SG 的结果支持。

## 16. Next（建议，不执行）

1. 若继续方案①，方向应是 **Geometry-Guided Safe Normalization**（把 geometry rank 用于"是否施加
   normalization"的门控，而非"施加多强"的连续分配），并在协议中**预注册 matched sensitivity control**。
2. 建议先在 MVTec 其余类别上检验"高脆弱类别 → α=0"这一条规则是否泛化（当前证据只覆盖 5 类、2 类贡献收益）。
3. 需人工批准新协议后才可启动；**本轮不启动 5D**，不实现 v2/v3，不做 post-hoc 调参。
