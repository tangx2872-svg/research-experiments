# Experiment 14 — Afternoon Broad Method Screening（BROAD FIRST, DEEP LATER）

**日期**：2026-10-08 ｜ **窗口**：11:00 → 目标 14:00 ｜ **环境**：AutoDL RTX 3090 24GB / conda `base` / PyTorch 2.8.0+cu128 / Anomalib 2.6.2
**起始 HEAD**：`66488f3`（Exp13-P）｜ **冻结物**：`config/{config,candidate_registry,analysis_plan,reuse_matrix}.json` + `config/sha256_frozen.json`

> **状态：PROTOCOL FROZEN（在读取任何 Exp14 target result 之前冻结候选与判据）。**
> 原则：**广度优先、快速淘汰、只把算力给赢家**。不为跑满时间制造无意义实验。

## 1. Purpose

在多个**有明确依据**的方法 family 中，用**统一协议**快速寻找是否存在明显优于当前冠军
**Adaptive B2（12/15 PASS，1 catastrophic，worst Δd′ = −0.2852）** 的 robustness–preservation trade-off。
本轮的产出目标是 family leaderboard / candidate leaderboard / Top1–2 / 是否存在值得进入最终完整实验的方法；
**若全部失败，则明确记录 negative result**（这本身是重要结果）。

## 2. Candidate families（冻结于 `config/candidate_registry.json`）

| Family | 形式 | 依据 | 候选 |
|---|---|---|---|
| **A — Dual-Path Original + Robust** | `F = w·F_original + (1−w)·F_B2`（layer2/3 pre-concat）；另有 layer-wise 组合 | Original 保 preservation、B2 保 robustness；两者可对齐 | A2/A3/A4（w=0.25/0.50/0.75）、A5（L2 orig + L3 B2）、A6（L2 B2 + L3 orig） |
| **B — Tiny category-specific INSS** | `F' = F − λ·P_illum(F − μ)`（**仅 layer2**） | Exp13-P：illumination subspace 低维、稳定、category-specific | B1–B6（K∈{8,16,32} × λ∈{0.25,0.50}） |
| **C — Late / Score-level dual fusion** | `score = w·z_orig + (1−w)·z_B2`（normal-only calibration） | 保留两条完整 representation，只在 evidence 层融合 | C1–C5（w=0.20…0.80）、C6（max） |

**关键代数发现（不改变方法定义，只做等价改写；已写入 registry 并强制核验）**
`w·F_original + (1−w)·F_B2`（pre-concat，两层）**严格等价于**：
- L2：`F + (1−w)·β_c·IN(F)·rms(F)` → 既有 kind `residual(λ=(1−w)·β_c)`
- L3：`alpha_in((1−w)·0.25)`
因此 **Family A 不需要任何新模型代码**，且 **A7（`F_B2+0.25(F_orig−F_B2)`）与 A2 严格代数等价 → 按规则删除、不重复运行**。
→ Family A 实质上是 **B2 残差强度的重参数化扫描**（这一点本身就是本轮的结构性结论）。

## 3. Protocol（零漂移）

| 项 | 值 |
|---|---|
| train | **corrected-189**（Exp10 冻结：`Engine.fit` 内 `dm.setup` 后重新过滤，`n_train_ids=189`, `kept_all_train_ids=True` 已核验） |
| RNG / coreset | **strict-V2**（逐字复用 `experiment9c_rng`：R1 `Engine.fit` 入口 + R2 `select_coreset_idxs` 入口 replay；证据落盘） |
| 指标 | `experiment5a_h_analysis.unit_metrics`（与 5A-H/7A-O/9A/9B/10/11 同一实现） |
| PASS（单 cat-seed） | `Δ\|Δz\| ≤ −EPS_RZ(0.02)` **且** `Δd′ ≥ −EPS_DP(0.10)`，相对**同 (cat,seed) 的 corrected Original** |
| catastrophic | `Δd′ ≤ −0.25`（Exp10 Stage3 冻结定义） |
| 排名优先级 | 1 Pareto over B2 → 2 PASS 数 → 3 catastrophic=0 → 4 worst Δd′ → 5 robustness → 6 preservation → 7 cross-category 一致性 → 8 简洁度 → 9 算力 |
| 禁则 | 不围绕 cable seed2 / hazelnut seeds1,2 调参；看结果不新增 weight；不改 PASS 阈值；不只报最好 seed；不删类别；不用 test defect 校准 |

## 4. REUSE MATRIX（`config/reuse_matrix.json`）

| 资产 | 覆盖 | 用途 | GPU |
|---|---|---|---|
| Exp10 `P10_ORIG_a000`（Original-189） | **15/15** | 基准 + Family C 左路 | 0 |
| Exp11 `E11_B2`（Adaptive B2） | **15/15** | 冠军基准 + Family C 右路 | 0 |
| Exp10 `P10_C2_L2resid025_L3a025`（Fixed C2） | **15/15** | Tier 比较基线 | 0 |
| Exp13-P `illum_subspace_*_k8.npz` + `raw/*/stats.npz` | 5 类 | Family B 的 basis（K=8/16/32）+ μ（**K=8 与冻结资产逐位一致，maxabs=0.0**） | 0 |
| Exp10 `P10_T1_L2a000_L3a025`（≡ A5） | 3/15 | A5 的跨路径等价 sanity 对照 | 0 |
| Family A/B 候选 | — | 需要新 GPU 单元（见 §6） | full |

**→ Family C 完全 0 GPU；Family A/B 各 3 类 × seed0 起步。**

---

# 结果（运行中，2026-10-08）

## 5. 成本

| 阶段 | 单元 | 说明 |
|---|---|---|
| Round 1（Family A2–A6 + B1–B6 × bottle/cable/hazelnut × seed0） | **33** | 首次并发 4 → hazelnut 触发 **CUDA OOM**（见 §9），修复后补跑 2 单元 |
| Stage D（Family A 幸存者 A2/A3/A6 × screw/grid/hazelnut） | **7** | 3 workers |
| Stage E（A3 × 5 cats × seeds{1,2}） | **10** | Family C 的 5×3 已由历史 score 直接构成（**0 GPU**） |
| Family C（C1–C6） | **0** | 复用 Exp10 Original + Exp11 B2 的 per-image score |
| **合计（至 Stage E）** | **50 GPU units** | **avg 158.4 s/unit**、failed 0（OOM 单元已补跑）、peak VRAM 22.6 GB |
| Optional O1（Top1=C1 的 seeds 3,4：Original 10 + B2参照 10 + A1 等价检查 3） | **23** | 运行中 |

## 6. Round 1（3 cats × seed0）与淘汰

| candidate | family | PASS/3 | worst Δd′ | mean ΔR | ΔR_vs_B2 | Δd′_vs_B2 | 处置 |
|---|---|---|---|---|---|---|---|
| **A3** | A (strength-scaled B2) | **2/3** | −0.065 | −0.037 | +0.061 | +0.079 | **存活** |
| A6 | A (L2 B2 + L3 orig) | 1/3 | +0.023 | −0.039 | +0.058 | +0.119 | 存活（唯一 bottle Pareto） |
| A2 | A (w=0.25) | 1/3 | −0.196 | −0.068 | +0.029 | −0.015 | 淘汰（cable preservation −0.196 而 robustness 仅 −0.028） |
| A4 / A5 | A | 0/3 | −0.105 / −0.140 | +0.002 / −0.013 | +0.100 / +0.085 | +0.095 / +0.076 | 淘汰（bottle PASS→FAIL，仅 trade-off） |
| **B1–B6** | B (INSS) | 最佳 B3 1/3 | −0.194…+0.111 | −0.004…+0.052 | +0.093…+0.150 | −0.075…+0.207 | **STOP FAMILY B**（见 §7） |
| A7 | A | — | — | — | — | — | 未运行：与 A2 **严格代数等价** |

## 7. Family B（Tiny INSS）—— 本轮第一个 negative result

**观测模式（bottle/cable/hazelnut × seed0，18 单元，0 failed）**：
- **λ = 0.50（B2/B4/B6）**：`Δd′ > 0`（preservation 改善，bottle +0.18、cable +0.27/hazelnut +0.09 等）
  **但 `ΔR > 0`（robustness 比 Original 更差）**；
- **λ = 0.25（B1/B3/B5）**：`ΔR ≈ 0`（−0.026…+0.004，**几乎没有 robustness 收益**），preservation 多数持平或略差。

→ **投影掉 illumination 子空间并不改善 robustness**，强度越大反而使模型对 illumination 更敏感
（0 catastrophic，但方向与目标相反）。按冻结规则「全部候选只是单边 trade-off → STOP FAMILY B」：
**未扩展 screw/grid，Family B 全家族淘汰**。
这与 Exp13-P 的结论一致且互补：13-P 说明该子空间**不含 defect 特异信息**，14 进一步说明**移除它也不带来 nuisance 收益**
（很可能因为该子空间同时承载 kNN 距离结构中的有效方差）。

## 8. Stage D（5 cats × seed0 门）与 Stage E（multi-seed）

**Stage D 门（≥4/5 PASS 且 0 catastrophic）**：C1 **5/5** ✓、C2 4/5 ✓、C3 4/5 ✓、**A3 4/5** ✓；
C4/C6 3/5 ✗、A2/A6 3/5 ✗、Family B 全部 ✗。

**Stage E（种子 1,2 确认）**：
- **Family C（C1/C2）**：5×3 已由历史 score 直接构成（0 GPU）→ 见 §10。
- **A3**：补 seeds 1,2 → **5×3 = 11/15，但 catastrophic = 2，worst Δd′ = −0.4713** →
  **比 Adaptive B2（1 catastrophic，−0.2852）更差** → Tier B 归档。
  这一结果是 **Stage E 的价值证明**：A3 在 seed0 的五类门是 4/5 且 0 catastrophic，若不做多 seed 确认会被误判为安全。

## 9. 错误报告

`ERROR_REPORT.json`：Stage a1 中 `hazelnut:0:E14_A5/A6__hazelnut` 触发 **CUDA OOM**
（hazelnut `n_train_good=391`，单进程峰值 ≈7.4 GiB，4 workers ≈29.6 GiB > 24 GiB）。
处置：**降并发至 3（该修复用 2）**、**只重跑失败单元**、**协议/阈值零改动**；两单元随后成功，
其余 48 单元 0 failed。已成功的单元结果未被覆盖（runner 仅在成功时写 `info.json`）。

## 10. 最终 leaderboard（5 cats × 3 seeds，统一判据）

| # | candidate | family | units | **PASS/15** | catastrophic | **worst Δd′** | mean ΔR | mean Δd′ | bottle / cable / hazelnut / screw / grid | Tier |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | *Adaptive B2（冠军参考，Exp11 冻结）* | *C2-adaptive* | 15 | *12* | *1* | *−0.2852* | *−0.0807* | *+0.1902* | *3/3 · 2/3 · 1/3 · 3/3 · 3/3* | *基准* |
| **1** | **C1 (0.20·z_orig + 0.80·z_B2)** | **C late-fusion** | 15 | **12** | **0** | **−0.1339** | −0.0660 | **+0.2133** | 3/3 · 2/3 · 1/3 · 3/3 · 3/3 | **A — SERIOUS FINAL CANDIDATE** |
| 2 | C2 (0.35/0.65) | C | 15 | 11 | 0 | −0.0464 | −0.0538 | +0.2113 | 3/3 · 1/3 · 1/3 · 3/3 · 3/3 | B — archived |
| 3 | A3（(1−w)=0.50 强度缩放 B2） | A | 15 | 11 | **2** | **−0.4713** | −0.0606 | +0.1517 | 2/3 · 1/3 · 2/3 · 3/3 · 3/3 | B — archived |
| 4 | C3 (0.50/0.50) | C | 15 | 10 | 0 | +0.0140 | −0.0399 | +0.1919 | 3/3 · 0/3 · 1/3 · 3/3 · 3/3 | C — STOP |
| 5 | C6 (max) | C | 15 | 10 | 0 | −0.1414 | −0.0414 | +0.1636 | 3/3 · 1/3 · 0/3 · 3/3 · 3/3 | C |
| 6 | C4 (0.65/0.35) | C | 15 | 9 | 0 | +0.0297 | −0.0266 | +0.1544 | 3/3 · 0/3 · 0/3 · 3/3 · 3/3 | C |
| 7 | C5 (0.80/0.20) | C | 15 | 5 | 0 | +0.0263 | −0.0137 | +0.0994 | 3/3 · 0/3 · 0/3 · 0/3 · 2/3 | C |
| 8 | A6（L2 B2 + L3 orig） | A | 5 | 3 | 0 | +0.0233 | −0.0564 | +0.2733 | 1/1 · 0/1 · 0/1 · 1/1 · 1/1 | 淘汰（Stage D 门 3/5） |
| 9 | A2 (w=0.25) | A | 5 | 3 | 0 | −0.1956 | −0.0739 | +0.1508 | 1/1 · 0/1 · 0/1 · 1/1 · 1/1 | 淘汰（门 3/5） |
| 10 | B3 (K=16, λ=0.25) | B | 3 | 1 | 0 | −0.1195 | −0.0040 | −0.0370 | 1/1 · 0/1 · 0/1 · — · — | STOP（Family B 全家族） |
| 11–17 | B1/B2/B4/B5/B6/A4/A5 | B/A | 3 | 0 | 0 | — | — | — | — | STOP / 淘汰 |

### 10.1 关键对比：**C1 vs Adaptive B2**

| | Adaptive B2 | **C1** |
|---|---|---|
| PASS/15 | 12 | **12**（相同） |
| per-category PASS | 3/3 · 2/3 · 1/3 · 3/3 · 3/3 | **完全相同** |
| catastrophic | **1**（cable seed2） | **0** ✓ |
| worst Δd′ | −0.2852 | **−0.1339**（减半）✓ |
| mean Δd′（preservation） | +0.1902 | **+0.2133** ✓ |
| mean ΔR（robustness） | **−0.0807** | −0.0660（略逊） |
| GPU 成本 | 每单元 1 次完整 fit | **0**（纯 score 融合，复用现有 detector） |

**解读（严格按冻结判据，不作事后放宽）**：C1 与 B2 的 **PASS 集合完全相同**；C1 的增益**不是新增 PASS**，
而是**把 B2 唯一的 catastrophic（cable seed2，Δd′≈−0.285）压到 −0.134**，并整体抬升 preservation（mean Δd′ +0.023），
代价是 robustness 增益略减（mean ΔR 从 −0.081 升到 −0.066）。
也就是：**C1 用少量 robustness 换取显著更好的 downside 安全性** —— 这正是 Tier A 规则所描述的情形
（「12/15 但 catastrophic 从 1→0、worst Δd′ 明显改善」），因此 **C1 定为 SERIOUS FINAL CANDIDATE**。
**注意不是 threshold-sensitive 翻转**：PASS 集合与 B2 完全一致，没有任何单元靠阈值投机翻转。

## 11. 最终 Tier 判定

| candidate | PASS/15 | catastrophic | worst Δd′ | Tier |
|---|---|---|---|---|
| **C1** | 12 | **0** | −0.1339 | **A — SERIOUS FINAL CANDIDATE** |
| C2 / A3 | 11 | 0 / 2 | −0.046 / −0.471 | B — archived（C2 未超过 B2；A3 downside 更差） |
| C3 / C4 / C5 / C6 / A2 / A6 / Family B | ≤10 或门未过 | — | — | **C — STOP** |

**未出现 Tier S（≥13/15）**。**未出现 ≥13/15**。**catastrophic 在本轮所有 C-family 候选上均为 0**（B2 为 1）。

## 12. 协议等价与证据（本轮方法学验证）

| 检查 | 结果 |
|---|---|
| Exp14 路径 vs Exp10 路径：`E14_A5`（L2 α=0 + L3 α=0.25）vs `P10_T1_L2a000_L3a025` | bottle s0：**max\|Δscore\| = 0.0（bit-exact）**，tau/coreset 相同 |
| Exp14 路径 vs Exp11 路径：`E14_A1`（= B2 算子）vs `E11_B2` | bottle/cable/hazelnut s0：**max\|Δscore\| = 0.0（bit-exact）**，tau 与 coreset 完全相同 |
| corrected-189 train | `train_filter_evidence`：`n_train_ids=189`、`kept_all_train_ids=True`（全部单元） |
| strict-V2 RNG | 每单元落盘 `fit_replay_evidence`（R1）+ `rng_evidence`（R2）+ embedding/bank sha256 |

→ 本轮所有新单元与历史 Exp10/Exp11 **在同一协议下逐位可比**；这使 Family C 的 0-GPU 复用与 Family A/B 的新单元可以直接放在同一张表上比较。

## 13. 图表（`figures/`）

- **Fig1** `fig1_pareto_scatter.png` — (a) robustness–preservation Pareto 散点（含 Original/Fixed C2/Adaptive B2 三星 + 全部候选）；(b) **决策视图**：PASS 数 vs worst-case Δd′（标出 catastrophic 线与 B2 的基线）
- **Fig2** `fig2_family_leaderboard.png` — family 级 leaderboard 与 downside risk
- **Fig3** `fig3_top_heatmap.png` — Top5 候选的 category × seed Δd′ 热图
- **Fig4** `fig4_pass_matrix.png` — PASS/HOLD/FAIL 矩阵
- **Fig5** `fig5_seed_stability.png` — per-seed PASS 稳定性

---

# Optional O1（seeds 3,4）—— 稳定性确认结果（**不改变按冻结规则产生的 winner**）

**做法**：为 Top1/Top2（C1/C2，及 C3–C5 一并给出）在 **unseen seeds {3,4}** 上补跑其所需的两条 source 路径
（`E14_A0` = Original，`E14_A1` = B2 算子；A0/A1 是冻结线性 family 的 w=1/w=0 端点，非竞争者），
共 **23 GPU units**（3 + 10 + 10，0 failed），再做**同一冻结定义**的 score 融合与判据。
`E14_A1` 在 seeds0–2 与 Exp11 `E11_B2` **逐位一致**（3 类 max\|Δscore\|=0.0），因此 A1 可作 B2 在 seeds3,4 的参考。

## O1.1 seeds {3,4}（每候选 10 单元）

| candidate | PASS/10 | catastrophic | worst Δd′ |
|---|---|---|---|
| **C1** | **6/10** | **1**（cable s4） | −0.2694 |
| **C2** | **8/10** | **0** | −0.1817 |
| C3 | 7/10 | 0 | −0.1108 |
| C4 | 4/10 | 0 | −0.0574 |
| C5 | 3/10 | 0 | −0.0215 |
| **Adaptive B2（同种子同参考）** | **7/10** | **2**（bottle s4 −0.258、cable s4 −0.410） | **−0.4100** |

## O1.2 合并 25 单元（5 cats × seeds 0–4）—— 目前最可靠的估计

| method | **PASS/25** | catastrophic | **worst Δd′** | mean ΔR |
|---|---|---|---|---|
| *Adaptive B2（冠军）* | *19/25 (76%)* | *3 (12%)* | *−0.4100* | *−0.0843* |
| **C2 (0.35/0.65)** | **19/25 (76%)** | **0** | **−0.1817** | −0.0544 |
| **C1 (0.20/0.80)** | 18/25 (72%) | 1 | −0.2694 | −0.0670 |
| C3 (0.50/0.50) | 17/25 | 0 | −0.1108 | −0.0403 |
| C4 (0.65/0.35) | 13/25 | 0 | −0.0574 | −0.0263 |
| C5 (0.80/0.20) | 8/25 | 0 | −0.0215 | −0.0135 |

## O1.3 结论与 caveat（**重要**）

1. **按 §17 冻结规则，Optional 阶段不得反向改变 winner** → **official winner 仍为 C1（Tier A）**，
   因为 winner 是在 pre-registered 阶段（Stage D/E，seeds 0–2）机械确定的，O1 只用于**确认稳定性**。
2. **但 O1 的确认结果是「Top1 未通过、Top2 通过」**：
   - C1 在 unseen seeds 上退化（6/10，出现 1 个 catastrophic）；
   - **C2 反而更稳（8/10，0 catastrophic）**；
   - 合并 25 单元后 **C2 与 B2 的 PASS 数完全相同（19/25），却把 catastrophic 从 3 降到 0，
     worst Δd′ 从 −0.410 改善到 −0.182**。
3. **因此本轮给出的建议（recommendation，不是规则修改）**：若下一轮只允许冻结一个候选，
   应在 **C2** 而非 C1 上做完整验证 —— 依据是 **downside 安全性 + 跨 unseen seed 稳定性**，
   而不是 PASS 数的边际差异。C1 相对 B2 的优势（catastrophic 1→0）在 25 单元尺度上由 C2 以更强形式保持。
4. **对冠军本身的重要修正**：Adaptive B2 的 catastrophic 率在 25 单元上是 **3/25（12%）**，
   高于此前 15 单元观察到的 1/15（§Exp11）—— **B2 的 downside 风险此前被低估**。
   （仅此为新增事实；B2 仍是本轮唯一在 5×3 上达到 12/15 的非融合方法。）
5. **O3（免费 ablation）**：Family C 的两个端点即 w=1 → **Original**（5×3 = 0 PASS 基准）与 w=0 → **B2**，
   两者均已在表中；C1/C2 的增益来自**中间权重**，而非端点。
