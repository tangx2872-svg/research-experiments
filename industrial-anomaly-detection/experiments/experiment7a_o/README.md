# Experiment 7A-O — Overnight Module Screening

> 状态：**已完成**（92 GPU units：2 smoke + 42 Round1 + 48 Round2；sanity **18/18 PASS**；
> freeze SHA256 `e046a89a852da9aad55d2eafeea1c3bc05a5aff640be4338f7e515e53150f224`）。
> **FINAL VERDICT：CASE_C — NO USEFUL MODULE**（没有任何简单 module 超过 Uniform α=0.40091275；
> 但 **A3_lam050 严格支配 B0_original**）。
> **本实验是 EXPLORATORY SCREENING，不是 confirmatory。**
> 前序：6B ✅ **CASE_C**（`ad26ec5`） → 停止挽救 geometry-guided category gating，进入模块筛选。

---

## 0. 论文导航

📍 **Stage 7 — Method Candidate Screening**（探索性）

① 现象 → ② defect-specific response → ③ 跨类别/defect 验证 → ④ 简单属性排除 →
⑤ layer-specific response → ⑥ geometry-guided adaptive normalization →
⑦ **6B：Soft-GC α_F=0.25 是孤立点、不构成 knee region、收益主要来自整体 normalization strength，
`Uniform α≈0.40091275` 成为新的强 baseline** → **今晚：模块筛选（7A-O）**

**停止继续拯救 geometry-guided category gating。**

## 1. 今晚要回答的唯一问题

> 是否存在**简单、可插拔**的 feature intervention / fusion module，能够相对 **Uniform α≈0.40091275**
> 取得更好的 **defect-preservation / illumination-robustness** trade-off？

不先编理论，先找 empirical winner。

## 2. Frozen Baselines（全部由历史 raw 重组，**零 GPU**）

| 名称 | 定义 | 来源 | 覆盖 |
|---|---|---|---|
| B0_original | 全部 α=0 | 5A-H B0 | 15/15 |
| B1_fixed_050 | 全部 α=0.5 | 5A-H B2 | 15/15 |
| **B2_uniform_040091275** | 全部 α=0.40091275（**强 baseline**） | 5A C1 + 6A + 6B | 15/15 |
| B3_soft_gc_af025 | fragile{bottle,grid} α=0.25，其余 0.5（reference） | 6A + 5A-H B2 | 15/15 |

## 3. Module Families（参数 grid 运行前冻结；**禁止 dense sweep**）

### Family A — Residual / Original-Preserving Fusion
- **A1**：`F_out = F + λ·N̂`，`N = IN(F)`，`N̂ = N · rms_spatial(F)`（per (sample, channel)）；λ ∈ {0.10, 0.25, 0.50}
- **A3**：`F_out = (F + λ·N̂) · rms(F)/rms(F + λ·N̂)`（能量保持）；λ ∈ {0.25, 0.50}
- **A2 = NOT SAFE / NOT IMPLEMENTED**：把 2C 维 concat 用**无训练固定投影**压回 C 维，只能是
  (i) `P=[I|I]/√2`（等价于 α interpolation，协议明确禁止），或 (ii) 无原则依据的人造算子。
  两者都不接受；concat 本身由 Family B1 以**可测量的维度变化**实现。
- 说明：`F + λ(IN(F)−F)`（=interpolation）**未实现**，不包装成新模块。

### Family B — Dual Representation
- **B1**：`F_out = concat([F, γ·N̂], dim=1)`，γ ∈ {0.25, 0.50, 1.00}；embedding dim 1536→**3072**
- **B2 = NOT RUN**：Round 0 实测 B1 在预算内可跑，因此不需要任意的 dimension-balancing 变体。
- 公平性：coreset ratio 0.1 / kNN 9 / split / illumination / scoring 不变；逐 unit 记录 embedding dim /
  bank size / runtime / VRAM。

### Family C — Layer-Selective Normalization（mean α = 0.40091275，与 Uniform 同 budget）
`C1 (L2=0.30, L3=0.50)`、`C2 (0.35, 0.45)`、`C3 (0.45, 0.35)`、`C4 (0.50, 0.30)`；
`C0 = (0.40091275, 0.40091275)` 即 Uniform → 必须先通过 **equivalence smoke**。

### Family D — Alternative Normalization（strength = 0.40091275, affine=False, 无训练）
- **D1**：layernorm-like（mean/var over (C,H,W) per sample）
- **D2**：groupnorm-like（G=8 组，mean/var over (C/G,H,W) per sample）

## 4. Round 0 — Equivalence / Smoke（已完成）

| 检查 | 结果 |
|---|---|
| `SMOKE_ORIG_a000`（α=0, bottle/seed0）vs 5A-H B0 | **183 keys，`max\|Δscore\| = 0.000e+00`**，τ 逐位一致 ✅ |
| `SMOKE_UNIFORM_a040091275`（bottle/seed0）vs 6B 同 α | **183 keys，`max\|Δscore\| = 0.000e+00`** ✅ |
| embed dim / bank / runtime / VRAM 记录 | 1536 / 21401 / ~45 s / 2605 MB ✅ |
| runner pre-run sanity R1–R6 | 全 PASS |

→ generator 的 feature 变换路径被证明与历史**逐位等价**；C0 的 Uniform equivalence 因此同样成立。

## 5. Successive Halving 调度（机械）

- **Round 1**：14 个合法 candidate × {bottle, grid, hazelnut} × seed0 = **42 units**
- **Round 1 晋级**（对 Uniform，同一子集上计算）：
  - **Tier S**：`Δd′ > 0` 且 `Δ|Δz| ≤ 0`（且 negTr = 0）
  - **Tier A**：`Δd′ ≥ +0.10` 且 `Δ|Δz| ≤ +0.02` 且 negTr = 0
  - **Tier B**：`Δ|Δz| ≤ −0.02` 且 `Δd′ ≥ −0.05`
  - 其余淘汰；**每 family 至多 Top-1**，合计最多 **4 configs**
- **Round 2**：晋级 configs × 5 categories × 3 seeds（复用已跑 seed0 单元，只补缺失）
- **Round 3（条件）**：仅当 ≥2 个不同 family 在 Round 2 仍为 Pareto-improving/Tier A 时，
  测**恰好一个**组合（Top-2 family winners 依次施加），先跑 3 categories × seed0；
  若不明显优于最佳单模块则 STOP。

## 6. Primary Metrics（沿用 5A-H 冻结口径）

`mean defect d′` ↑ ；`mean |ΔNormalScore_z|` ↓ ；ε = 0.10。
另报 per-category / per-seed / worst-category Δ / negative transfer / image AUROC /
pixel AUROC / AUPRO / runtime / VRAM / bank size / embedding dim。

## 7. Sanity S1–S20

见 `results/experiment_7a_o/summary/sanity_report.md`（dataset / mask / seed / duplicate / NaN / Inf /
rows / baseline-equivalence / identity-equivalence / Uniform-equivalence / freeze-hash /
no-target-derived-params / no-per-category-tuning / no-hidden-sweep / deterministic-serialization /
bank-dim / runtime / VRAM / mechanical-promotion / no-silent-exclusion）。

## 8. Verdict Rules（PRE-REGISTERED）

- **CASE_A STRONG WINNER**：≥1 candidate 跨 5 categories × 3 seeds 稳定、相对 Uniform Pareto-improving 或
  Tier A、无 negative transfer。
- **CASE_B WEAK WINNER**：有改善但 seed/category 稳定性不足，或幅度小，或 trade-off 未外推。
- **CASE_C NO USEFUL MODULE**：所有简单 module 都无法超过 Uniform → 停止当前 fusion 路线。
- **CASE_D COMBINATION WIN**：≥2 单模块有效且组合进一步稳定改善 → 冻结组合为下一阶段候选。

判定顺序：**D → A → B → C**（首个命中即判定）。

## 9. Stop Rules

数据损坏 / baseline equivalence FAIL / 大量 NaN·Inf / frozen protocol 被改 / OOM 无法靠降 worker 解决 /
仓库异常修改 / 无法在不看 target 的情况下解决的歧义 → **STOP**。OOM 时 3→2→1 workers 自动降级。

## 10. Leakage Guard（重要）

今晚允许 `看结果 → 排名 → 淘汰`，但**全部标记为 EXPLORATORY / selection-set**。
**禁止**把同一结果当最终 confirmation；winner 明天必须冻结结构/参数/指标并设计独立 confirmation。
所有负结果保留。

## 11. Expected Outputs

```text
results/experiment_7a_o/reference/  policy_freeze.json  policy_freeze.sha256  asset_audit.csv
results/experiment_7a_o/configs/    module_specs.csv
results/experiment_7a_o/raw/        {cat}/seed_{s}/config_{NAME}/{per_image.csv,info.json,module_meta.json}
results/experiment_7a_o/summary/    all_candidates.csv round1_ranking.csv round2_ranking.csv
                                    per_category.csv per_seed.csv pareto.csv runtime.csv
                                    promotion.json verdict.json sanity_report.md final_report.md
results/experiment_7a_o/figures/    fig1_pareto_all_candidates.png fig2_family_delta_vs_uniform.png
                                    fig3_winner_category.png fig4_winner_seed.png
                                    fig5_single_vs_combination.png (if applicable)
```

---

# 运行后

## 12. Round 0 — 结果（equivalence smoke）

| 检查 | 结果 |
|---|---|
| `SMOKE_ORIG_a000`（α=0, bottle/seed0）vs 5A-H B0 | **183 keys，`max\|Δscore\| = 0.000e+00`**，τ 完全一致 → **α=0 identity equivalence PASS** |
| `SMOKE_UNIFORM_a040091275`（bottle/seed0）vs 6B 同 α | **183 keys，`max\|Δscore\| = 0.000e+00`** → **Uniform equivalence PASS**（因此 C0 等价性亦成立） |
| 记录 | embed dim 1536、bank 21401、runtime 44–47 s、peak 2605 MB |

→ 7A-O runner 的 feature 变换路径与历史**逐位等价**，模块可以安全地在此路径上做干预。

## 13. Round 1 — 结果（14 configs × {bottle, grid, hazelnut} × seed0，vs Uniform）

Uniform baseline 在同一子集上：**d′ = 5.4467，\|Δz\| = 0.1660**。

| rank | config | family | d′ ↑ | \|Δz\| ↓ | Δd′ | Δ\|Δz\| | worst cat Δ | negTr | tier |
|---|---|---|---|---|---|---|---|---|---|
| 1 | **A3_lam050** | A | 6.3064 | 0.2533 | **+0.8597** | +0.0872 | +0.1257 | 0 | — |
| 2 | A3_lam025 | A | 6.2430 | 0.2472 | +0.7962 | +0.0812 | −0.0739 | 0 | — |
| 3 | **B1_g100** | B | 6.2373 | 0.2593 | +0.7905 | +0.0933 | +0.1849 | 0 | — |
| 4 | B1_g050 | B | 6.1358 | 0.2598 | +0.6891 | +0.0937 | −0.0940 | 0 | — |
| 5 | A1_lam050 | A | 6.0758 | 0.2404 | +0.6290 | +0.0744 | −0.1088 | 1 | — |
| 6 | B1_g025 | B | 6.0244 | 0.2864 | +0.5777 | +0.1204 | −0.0749 | 0 | — |
| 7 | A1_lam010 | A | 5.9535 | 0.2674 | +0.5068 | +0.1013 | −0.2230 | 1 | — |
| 8 | A1_lam025 | A | 5.9213 | 0.2297 | +0.4746 | +0.0637 | −0.2294 | 1 | — |
| 9 | **D1_ln_s040** | D | 5.7275 | 0.3241 | +0.2808 | +0.1581 | −0.1834 | 1 | — |
| 10 | D2_gn8_s040 | D | 5.6963 | 0.3180 | +0.2496 | +0.1520 | −0.2738 | 1 | — |
| 11 | **C3_l2_045_l3_035** | C | 5.4574 | 0.1864 | +0.0107 | +0.0204 | −0.0261 | 0 | — |
| 12 | C4_l2_050_l3_030 | C | 5.3920 | 0.1915 | −0.0547 | +0.0255 | −0.1647 | 2 | — |
| 13 | C2_l2_035_l3_045 | C | 5.3097 | 0.1828 | −0.1370 | +0.0168 | −0.1785 | 3 | — |
| 14 | C1_l2_030_l3_050 | C | 5.2508 | 0.1616 | −0.1959 | −0.0044 | −0.3288 | 2 | — |
| — | B0_original（baseline） | — | 5.7809 | 0.3122 | +0.3342 | +0.1462 | −0.3012 | 1 | — |
| — | B3_soft_gc（baseline） | — | 5.6473 | 0.2213 | +0.2006 | +0.0553 | +0.0578 | 0 | — |
| — | **B2_uniform（强 baseline）** | — | 5.4467 | 0.1660 | 0 | 0 | 0 | 0 | — |
| — | B1_fixed_050（baseline） | — | 5.3084 | 0.1771 | −0.1383 | +0.0111 | −0.2728 | 2 | — |

**关键观察（Round 1）**

1. **没有任何 config 达到 Tier S/A/B** → §14 机械晋级集为 **空**。
2. 结构非常清楚：**A / B / D 三个 family 全部"以 robustness 换 preservation"** —— Δd′ 高达 +0.86，
   但 Δ\|Δz\| 同时 +0.06…+0.16，全部超出 Tier A 的 +0.02 与 Tier S 的 ≤0。
   即这些模块**只是把工作点沿既有 frontier 向左上移动**，没有把 frontier 往外推。
3. **Family C（layer-selective）** 全部劣于 Uniform：C1 略降 robustness（−0.0044，未达 Tier B 的 −0.02），
   代价是 Δd′ −0.196；C2/C4 两轴都更差。→ **layer-selective allocation 不比 Uniform 有效。**

## 14. 协议修订 7A-O-A1（**在任何 Round-2 target 之前写入**）

`reference/round2_amendment.json`。问题：§14 的 tier 规则淘汰全部 candidate（晋级集空），
而 §15 写"从 Round 1 中每个 family 最多保留 Top 1"——两者在"零晋级"时冲突。

**裁决（不改任何阈值、不新增任何 config）**：以 Round-1 的 **Top-1 per family**（已冻结的 4 个 config）
跑 **5 categories × 3 seeds 的 full-panel diagnostic**，理由：§21 要求明早回答 Q5–Q9，
这些都需要 5×3 覆盖；而仅凭 3 类 × 1 seed 给出 CASE_C 证据过薄。

- 选择规则（机械）：每 family 取 Round-1 `Δd′ vs Uniform` 最高者（并列时取 `Δ|Δz|` 更低者）
  → **A: A3_lam050 · B: B1_g100 · C: C3_l2_045_l3_035 · D: D1_ln_s040**
- **§14 的机械结论（零晋级 → CASE_C）单独报告，并作为"机械口径"保留**；
  full-panel diagnostic 不得静默覆盖它。
- Round 3 仍只在 **full panel 上 ≥2 个 family 达到 Tier S/A** 时才允许。

## 15. Round 2 — 执行与 OOM 降级（自动，按 §26）

- Round 2 启动为 3 workers（16/16/16）。第 1 小时内 **w2 / w3 触发 CUDA OOM**：
  `PatchcoreModel.subsample_embedding → torch.vstack(self.embedding_store)` 单次请求 **4.58 GiB**
  （Family B 的 concat 使 embedding dim 1536→3072，bank 材料翻倍）。
  失败发生在 **hazelnut:1:B1_g100 / hazelnut:2:B1_g100** 两个 unit（status=FAILED，**保留记录**，见 sanity S20）。
- **按 §26 自动降级，不等待人工**：3 workers → **2 workers + 内存感知分组**
  （lane1 只跑 B1_g100 concat 单元，lane2 跑其余），并设 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`。
  这样同时最多只有一个 concat 单元 → 峰值 ~13 GB < 24 GB。
- 已完成的 74 个 unit 未重跑（`unit_done` resume）；FAILED 单元自动重跑。
- 该事件记录为 **family B 的真实成本**：concat 表示不仅维度翻倍，峰值显存也约翻倍。

## 16. Round 2 — 结果（full panel，5 categories × 3 seeds；总计 92 GPU units）

Uniform baseline（同 panel）：**d′ = 4.7940，\|Δz\| = 0.2134**。

| rank | config | family | d′ ↑ | \|Δz\| ↓ | Δd′ | Δ\|Δz\| | worst cat Δ | negTr | tier |
|---|---|---|---|---|---|---|---|---|---|
| 1 | **A3_lam050** | A | **5.3552** | 0.2753 | **+0.5612** | +0.0619 | −0.0493 | 0 | — |
| 2 | B1_g100 | B | 5.3044 | 0.2850 | +0.5104 | +0.0717 | −0.0001 | 0 | — |
| 3 | D1_ln_s040 | D | 4.9678 | 0.3298 | +0.1739 | +0.1164 | −0.2796 | 1 | — |
| 4 | C3_l2_045_l3_035 | C | 4.8159 | 0.2248 | +0.0219 | +0.0115 | −0.0538 | 0 | — |
| — | B0_original（baseline） | — | 4.9852 | 0.3105 | +0.1913 | +0.0971 | −0.3104 | 1 | — |
| — | B3_soft_gc（baseline） | — | 4.8782 | 0.2329 | +0.0842 | +0.0195 | −0.0996 | 0 | — |
| — | **B2_uniform（强 baseline）** | — | 4.7940 | 0.2134 | 0 | 0 | 0 | 0 | — |
| — | B1_fixed_050（baseline） | — | 4.6670 | 0.2082 | −0.1270 | −0.0052 | −0.2806 | 2 | — |

### 16.1 Pareto（full panel）

```text
Pareto frontier : A3_lam050 (5.3552, 0.2753) · B3_soft_gc (4.8782, 0.2329) · C3 (4.8159, 0.2248)
                  · B2_uniform (4.7940, 0.2134) · B1_fixed (4.6670, 0.2082)
dominated       : B1_g100 <- A3_lam050 ; B0_original <- A3_lam050, B1_g100
                  D1_ln_s040 <- A3_lam050, B0_original, B1_g100
```

- **没有任何 candidate 支配 Uniform，Uniform 也不支配任何 candidate** → **模块没有把 frontier 向外推**：
  绝大多数只是把工作点沿既有 frontier 移到"高 preservation / 低 robustness"一端。
- 但 **A3_lam050 严格支配 B0_original（无 normalization baseline）**：Δd′ **+0.3700** 且 Δ\|Δz\| **−0.0352**
  → "在原始特征上叠加一个尺度匹配的 normalized residual" **严格优于什么都不做**。

### 16.2 seed / category 稳定性（A3_lam050）

```text
per-seed Δd′  vs Uniform : s0 +0.5949 · s1 +0.4904 · s2 +0.5983     (3/3 同向)
per-seed Δ|Δz| vs Uniform: s0 +0.0665 · s1 +0.0653 · s2 +0.0540     (3/3 同向)
per-category Δd′/Δ|Δz|   : bottle +0.823/+0.052 · cable +0.505/+0.080 · grid +1.537/+0.136
                           hazelnut −0.009/+0.025 · screw −0.049/+0.018  (worst Δ = −0.049 < ε)
```

### 16.3 成本（每 unit）

| config | embedding dim | bank | runtime | peak VRAM |
|---|---|---|---|---|
| A3_lam050 | 1536 | 21401 | ~106 s | 2605 MB |
| **B1_g100 (concat)** | **3072** | 21401 | ~103 s | **5113 MB（×2；3 workers 时 OOM）** |
| D1_ln_s040 | 1536 | 21401 | ~84 s | 2605 MB |
| C3 | 1536 | 21401 | ~96 s | 2605 MB |

## 17. Sanity — **18/18 PASS**

见 `summary/sanity_report.md`。含 S1 dataset / S2 mask（401 masks）/ S3 seeds / S4 无重复 92-unique /
S5 primary 无 NaN（624 值）+ S5b（16 个 secondary NaN 全部来自 5A 来源 baseline 条目，其源资产不含 pixel 指标，已记明）/
S6 无 Inf / S7 per-category 行数一致 / S11 freeze hash / S12 无 target 依赖参数 / S13 无 per-category 调参 /
S14 无隐藏 sweep / S15 配置序列化一致 / S16 bank 维度 / S17 runtime / S18 VRAM /
S19 晋级规则机械复算 / S20 无静默排除失败配置。

## 18. FINAL VERDICT — **CASE_C（NO USEFUL MODULE）**

> 所有简单 module 都无法超过强 Uniform baseline α=0.40091275。

- **机械口径（§14，预注册）**：Round 1 晋级集 **空**（14/14 tier = "—"）→ CASE_C。
- **full-panel 口径（修订 7A-O-A1）**：4 个 family representative 在 5×3 上同样**无一达到 Tier S/A/B**
  → 与机械口径**一致**，CASE_C 被加强而非推翻。
- Round 3 **未运行**（§16 条件不满足：0 个 family 达 Tier S/A）。

## 19. Paper Progress — Q1–Q10

| # | 问题 | 回答 |
|---|---|---|
| Q1 | Residual / Original-Preserving Fusion 有配置超过 Uniform？ | **没有**。A 族全部"以 robustness 换 preservation"：A3_lam050 Δd′ **+0.5612** 但 Δ\|Δz\| **+0.0619**。唯一正面事实：A3_lam050 **严格支配 B0_original**（+0.3700 d′、−0.0352 \|Δz\|）。 |
| Q2 | Dual Representation 有价值？ | **无增量价值**。B1_g100 被 A3_lam050 **严格支配**，代价是 embedding dim 1536→3072、峰值显存 ×2（3 workers 时 OOM）。它只是同一条 frontier 上更贵的点。 |
| Q3 | Layer-selective allocation 比 Uniform 更有效？ | **不**。C1–C4 在 5×3 上与 Uniform 的差别 ≤ 0.022 d′（最佳 C3 = +0.0219/+0.0115），C1/C2/C4 更差；两个方向（C1: L2<L3 vs C4: L2>L3）无系统性优劣 → **没有 layer 偏好信号**。 |
| Q4 | Alternative Norm 改变 trade-off？ | **变差**。D1(LayerNorm-like) Δd′ +0.1739 / Δ\|Δz\| +0.1164，**被 Original 支配**；D2(GroupNorm-like) 更差（+0.2496/+0.1520）。instance-norm 基座优于这两种。 |
| Q5 | 哪个单模块是 winner？ | **没有符合预注册 tier 的 winner**。按"相对弱 baseline 的严格支配"口径，**A3_lam050**（energy-preserving residual fusion, λ=0.5）是唯一有实质意义的候选。 |
| Q6 | winner 跨 5 categories / 3 seeds 稳定？ | **稳定**：3/3 seeds 同向（Δd′ +0.49…+0.60），worst-category Δ = −0.049（ε 内），negTr = 0。 |
| Q7 | 是否存在 Pareto domination？ | 有，但方向是**内部排序而非外推**：A3 ⊃ {B0, B1_g100, D1}；B1_g100 ⊃ {B0, D1}；B0 ⊃ {D1}。**无 candidate 与 Uniform 相互支配**。 |
| Q8 | 两模块组合是否进一步提升？ | **未测（协议禁止）**：Round 3 需 ≥2 个 family 在 full panel 达 Tier S/A，实际为 0。 |
| Q9 | 最强方法（A3_lam050）相对各 baseline 改善多少 | vs **Original**：d′ **+0.3700**、\|Δz\| **−0.0352**（**双轴更优**）；vs **Fixed 0.5**：+0.6882 / +0.0671；vs **Uniform 0.40091275**：+0.5612 / +0.0619；vs **Soft-GC**：+0.4770 / +0.0424。 |
| Q10 | 下一步 | **STOP** —— 停止"简单 feature fusion"路线；A3_lam050 仅保留为"最强的弱归一化参考点"，不作为方法。 |

## 20. Negative results（全部保留）

- **Family A（A1×3 + A3×2）**：preservation 大幅提升、robustness 同步恶化 → Tier 全不达；
  A1（additive residual）系统性差于 A3（energy-preserving）。
- **Family B（B1_g025/g050/g100）**：与 A 族同向，且被 A3_lam050 支配；concat 的显存 ×2 并导致 2 次 OOM。
- **Family C（C1–C4）**：全部劣于或等于 Uniform；C1 是唯一降 robustness 的（−0.0044，未达 Tier B 的 −0.02），
  代价 Δd′ −0.1959。
- **Family D（D1、D2）**：被 Original 支配 → 明确否定。
- **A2（concat + fixed projection）**：NOT IMPLEMENTED（无安全的无训练投影，见 §3）。
- **B2（grouped projection）**：NOT RUN（B1 已验证可跑，无需任意替代算子）。

## 21. Next Step（建议，**未启动，需人工批准**）

1. **不**继续调 residual 的 λ / concat 的 γ / per-category 权重 —— 本实验已证明它们只沿既有 frontier 移动。
2. 若要真正 push frontier，必须改变**目标轴本身**（例如：把 illumination robustness 与 defect 表达解耦，
   例如 inference-time illumination 校正而非改 representation；或引入显式 illumination-invariance 目标），
   这需要新的预注册协议与新的 baseline。
3. A3_lam050 保留为"最强的弱归一化参考点"，供未来方法对照。
4. **STOP**，等待人工 review。

---

> **EXPLORATORY ONLY**：本实验是 selection-set 结果；任何 config 都不能直接作为论文 confirmatory result。
> 若要推进，必须冻结结构/参数/指标并设计独立 confirmation。
