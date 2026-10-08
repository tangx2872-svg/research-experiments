# Experiment 15 — Winner Expansion & Combination Tournament

**日期**：2026-10-08 ｜ **起点**：`6efae31`（Exp14）｜ **窗口**：12:24 → 目标 90–120 min（软上限 120 / 硬上限 135）
**冻结物**：`config/{candidate_registry,analysis_plan,reuse_matrix}.json` + `config/sha256_frozen.json`
**原则**：不为「服务器还有时间」多调参；只检验 **Exp14 赢家是否稳定、不同有效模块是否真的互补、能否跨 seed 泛化**。
失败模块不救；Category×Channel 不复习；Exp13-P 的 defect-entanglement 叙事不复活；Mahalanobis/whitening 只作成熟 baseline（本轮未用）。

## 1. 输入：Exp14 的结论（不改写）

| | 方法 | 5×3 PASS | catastrophic | worst Δd′ | 25 单元（seeds 0–4） |
|---|---|---|---|---|---|
| 规则 winner | C1 = 0.20·z_o + 0.80·z_B2 | 12/15 | 0 | −0.134 | 18/25, 1 cat, worst −0.269 |
| 推荐 | C2 = 0.35/0.65 | 11/15 | 0 | −0.046 | 19/25, 0 cat, worst −0.182 |
| 冠军参考 | Adaptive B2 | 12/15 | 1 | −0.285 | 19/25, **3 cat**, worst −0.410 |
| 已否定 | Family A（Dual-Path，代数 ≡ B2 强度重参数化）、Family B（Tiny INSS 全家族 trade-off） | — | — | — | — |

## 2. 分支决策：**PATH A**

§3 的 PATH S 要求 `≥13/15 + 0 catastrophic` —— **Exp14 最高 12/15，未达**；
但 Top1 完全命中 §4 的显式示例（`12/15 + 0 catastrophic` 且 `worst Δd′ 明显优于 B2`，−0.134 vs −0.285）
→ **PATH A：Winner × Complementary Module Tournament（最多 6 个组合）**。

## 3. 组合合法性审计（先判断，再决定是否运行）

| 组合 | 判定 | 依据 |
|---|---|---|
| **X1** Top feature + B2 | **SKIP（数学重复）** | Exp14 已证 Family A 的线性融合严格等价于 B2 残差强度重参数化；`E14_A1` 与 `E11_B2` 在 3 类上 **bit-exact（max\|Δscore\|=0.0）** → 与 B2 组合无新信息 |
| **X2** Top feature + best late fusion | **已实现** | Exp14 的 C1/C2 本身就是「Top feature detector（Original 与 B2）+ late fusion」；新增 **X2b = mean(z_o, z_B2, z_A6)**（A6 = Exp14 唯一 bottle Pareto 的 feature method，仅 seed0 有单元） |
| **X3** Dual-Path winner + mild INSS | **FORBIDDEN** | §6 前提「INSS 至少出现 Pareto-positive candidate」**不满足**（Exp14 Family B 全家族单边 trade-off） |
| **X4** Layer-specialized | **A5/A6 已跑（不重跑）**；新增 **X4a = L2 residual(β_c) + L3 α=0.125**、**X4b = L2 residual(0.5β_c) + L3 α=0.25** | 每个层强度都在 Exp14 被独立验证过：L2 满强度来自 B2、L2 半强度与 L3 α=0.125 来自 A3、L3 α=0.25 来自 B2 → 组合**不含新超参数** |
| **X5** Normal-stat calibrated fusion | **RUN（0 GPU）** | 冻结 normal-only 规则：`w_orig(c) = clip(0.35·(0.25/β_c), 0.20, 0.50)` —— β_c 越大（B2 越激进 → catastrophic 风险越高）→ 给更多 Original 权重。**只用 train/good 统计，不用任何 test/defect 信息**，看到结果前冻结 |
| **X6** Conservative ensemble | **RUN（0 GPU）** | X6a = mean（≡ C3，已有）、X6b = max（≡ C6，已有）、**X6c = mean(z_C2, z_C6)**（两个已验证 rule 的保守组合，无新超参数） |

**X5 的冻结权重**：bottle 0.350、cable 0.297、hazelnut 0.384、screw 0.404、grid 0.200（`config/candidate_registry.json`）。
**§19 held-out category：SKIPPED** —— 本机只有 bottle/cable/grid/hazelnut/screw 五个类别（= 方法开发用过的全部），
规则要求不得临时下载 → 泛化证据改用 **seeds 0–6（E3+）**。

## 4. Round 1（GPU，3 类 × seed0）：X4a / X4b 结果与淘汰

| candidate | 形式 | PASS/3 | worst Δd′ | mean ΔR | mean Δd′ | vs B2 |
|---|---|---|---|---|---|---|
| X4a | L2 residual(β_c) + L3 α=0.125 | **0/3** | −0.241 | −0.071 | −0.022 | bottle：dR +0.005 / ddp −0.176（**双双更差**） |
| X4b | L2 residual(0.5β_c) + L3 α=0.25 | **1/3**（hazelnut） | −0.103 | −0.059 | −0.024 | bottle：+0.060 / −0.038；cable：+0.042 / −0.040（**双双更差**） |

**判定：两者均不晋级**（§13 要求 Pareto-positive 或明显减少 catastrophic；X4a/X4b 在 bottle/cable 上被 B2 支配，
仅 hazelnut 单点较好）→ **不触发 Round 2（screw/grid）**，不做 Round 3。GPU 因此省下 ~8 units。

## 5. Score-side 组合锦标赛（0 GPU）

覆盖 C1–C5（冻结 w grid）+ C6(max) + X5(category-adaptive) + X6c(max/mean 组合) + X2b(三路)。
来源：seeds0–2 = Exp10 Original + Exp11 B2；seeds3,4 = Exp14 `E14_A0`/`E14_A1`；seeds5,6 = Exp15 `o1ext` 阶段（新跑，见 §7）。

## 6. Score-side 结果（**35 单元 = 5 类 × seeds 0–6，E3+**）

| method | 形式 | PASS/35 | catastrophic | **worst Δd′** | mean ΔR | mean Δd′ | vs B2（paired bootstrap） |
|---|---|---|---|---|---|---|---|
| **X6c** | mean(z_C2, z_C6) | **27 (77.1%)** | **0** | **−0.1097** | −0.0471 | 0.1931 | ΔPASS +0.114 [−0.03,+0.29]；Δd′ +0.028 [−0.012,+0.070]；ΔR +0.034 [0.020,0.049] |
| C6 | max(z_o, z_b) | 26 (74.3%) | 0 | −0.1414 | −0.0413 | 0.1560 | ΔPASS +0.086 [−0.09,+0.26]；Δd′ −0.009 [−0.048,+0.035] |
| C2 | 0.35·z_o+0.65·z_b | 26 (74.3%) | 0 | −0.1817 | −0.0547 | **0.1972** | ΔPASS +0.086 [−0.06,+0.23]；Δd′ +0.032 [−0.009,+0.074] |
| X5 | category-adaptive w | 26 (74.3%) | 0 | −0.2110 | −0.0530 | **0.2099** | Δd′ **+0.045 [0.013,0.077]（显著）** |
| C1 | 0.20·z_o+0.80·z_b | 25 (71.4%) | 1 | −0.2694 | −0.0663 | 0.1949 | Δd′ **+0.030 [0.006,0.054]（显著）** |
| C3 | 0.50/0.50（≡ X6a mean） | 23 (65.7%) | 0 | −0.1108 | −0.0410 | 0.1815 | ΔPASS ±0.000 |
| *Adaptive B2* | *冠军参考* | *23 (65.7%)* | ***4 (11.4%)*** | ***−0.4100*** | −0.0814 | 0.1650 | — |
| C4 / C5 | 0.65 / 0.80 | 18 / 11 | 0 | −0.0574 / −0.0215 | −0.027 / −0.014 | 0.148 / 0.096 | ΔPASS −0.143 / −0.343 |
| X2b | mean(z_o, z_b, z_A6) | 3/5 (E1) | 0 | +0.0815 | −0.0528 | 0.2424 | 三路加入 A6 无增益（A6 已在 Exp14 被 Stage D 门淘汰） |

**关键读数**
1. **X6c 是本轮 Top1**：35 单元上 **27/35 vs B2 的 23/35（+4 单元，+11.4pp）**，且 **catastrophic 0 vs 4**。
   但 **ΔPASS 的 bootstrap CI 含 0 → PASS 优势不显著**；同时 **robustness 代价显著**（Δmean ΔR +0.034 [0.020,0.049]，
   即 fusion 的平均 robustness 增益小于 B2）。
2. **唯一在统计上清晰分离的是尾部风险**：worst Δd′ 的 bootstrap CI
   **X6c [−0.110, −0.033] vs B2 [−0.410, −0.258] → 不重叠**；B2 的 catastrophic 率在 35 单元上是 **4/35 = 11.4%**。
3. **X5（category-adaptive，normal-only 规则）没有超过固定权重**：PASS 相同（26），preservation 显著更高（+0.045），
   但 worst Δd′ 比 C2 更差（−0.211 vs −0.182）→ **自适应选择规则不优于固定 w=0.35**（诚实的 negative）。
4. **X2b（三路 + A6）无价值**：seed0 仅 3/5，与 Exp14 中 A6 未过 Stage D 门一致。
5. **C6（max）是并列组中最简单的单一规则**，但它的 mean Δd′ **低于 B2**（0.156 vs 0.165）→ 只有尾部安全收益。

## 7. 参数敏感度（§17/§18）与泛化

**冻结的 coarse grid（禁止细化）**：w_orig ∈ {0.20, 0.35, 0.50, 0.65, 0.80}，35 单元：

| w_orig | PASS/35 | catastrophic | worst Δd′ | mean ΔR |
|---|---|---|---|---|
| 0.20 | 25 | 1 | −0.2694 | −0.0663 |
| **0.35** | **26** | **0** | −0.1817 | −0.0547 |
| 0.50 | 23 | 0 | −0.1108 | −0.0410 |
| 0.65 | 18 | 0 | −0.0574 | −0.0270 |
| 0.80 | 11 | 0 | −0.0215 | −0.0139 |

- **曲线平滑、单调、最优点是一个宽峰（w≈0.20–0.50 区间内 PASS ≥23）** → **不是尖锐孤立峰** → **无 HIGH OVERFITTING RISK** ✓
- 同时 worst Δd′ 随 w 单调改善：**这是标准的 robustness–preservation frontier 重参数化**，
  说明 fusion weight 的作用是「在 frontier 上移动」，而不是发现新的 frontier。
- **泛化证据级别**：E3+（5 类 × 7 seeds = 35 单元）。**E4（held-out category）不可得** —— 本机只有方法开发用过的 5 个类别，
  按 §19 不允许临时下载 → 已明确记录为 limitation。
- **§17 结论**：唯一允许的敏感度分析（w 的 coarse grid）已完成且稳定；**未新增任何 fusion weight**。

## 8. 自动停止规则核对（§21）

| 规则 | 状态 |
|---|---|
| S1（≥13/15 + 0 catastrophic 且参数稳定 → 转 stability validation） | **未触发**（最高 27/35 ≈ 11.6/15 等价，未达 13/15） |
| S2（所有候选均不超过 B2 → STOP） | **未触发**：X6c 在 PASS（27 vs 23）与尾部安全（0 vs 4 catastrophic）上优于 B2 |
| S3（性能并列 → 取更简单者） | **已应用**：X6c/C6/C2/X5 的 PASS 与 catastrophic 完全相同（26–27、0），差异在尾部与代价上；据此按「简单性 + robustness 代价」给出 C2 作为并列组中的**推荐**（见 §10） |

## 9. 最终排名（按 Exp14 冻结排名键机械排序）

键：(1) Pareto-over-B2 → (2) PASS 数 → (3) catastrophic=0 → (4) worst Δd′ → (5) robustness → (6) preservation

| Rank | Method | PASS/35 | cat | worst Δd′ | mean ΔR | mean Δd′ | Evidence | Verdict |
|---|---|---|---|---|---|---|---|---|
| **1** | **X6c = mean(z_C2, z_C6)** | **27** | **0** | **−0.1097** | −0.0471 | 0.1931 | **E3+** | **Top1（HOLD）** |
| 2 | C6 = max(z_o, z_b) | 26 | 0 | −0.1414 | −0.0413 | 0.1560 | E3+ | Top2（并列组，最简单一规则） |
| 3 | C2 = 0.35/0.65 | 26 | 0 | −0.1817 | −0.0547 | **0.1972** | E3+ | 并列组（**最低 robustness 代价**，Exp14 推荐） |
| 4 | X5 = category-adaptive w | 26 | 0 | −0.2110 | −0.0530 | **0.2099** | E3+ | 并列组（preservation 显著更优但不更安全） |
| 5 | C1 = 0.20/0.80 | 25 | 1 | −0.2694 | −0.0663 | 0.1949 | E3+ | 被同族其他点支配 |
| 6 | *Adaptive B2（冠军参考）* | *23* | *4* | *−0.4100* | −0.0814 | 0.1650 | *E3+* | *被 #1–#5 在尾部安全上支配* |
| 7–9 | C4 / C5 / C3 | 18 / 11 / 23 | 0 | — | — | — | E3+ | 归档 |
| 10 | X2b（三路 + A6） | 3/5 | 0 | +0.0815 | — | — | E1 | 淘汰 |
| 11–12 | **X4a / X4b**（层强度组合） | 0/3 · 1/3 | 0 | −0.2412 · −0.1031 | −0.071 · −0.059 | −0.022 · −0.024 | E1 | **淘汰（R1 未晋级）** |

**注（并列组）**：X6c / C6 / C2 / X5 的 PASS（26–27）与 catastrophic（0）几乎完全相同，bootstrap CI 全部重叠
→ **它们之间的差异不可由本轮数据区分**；机械键按 PASS 与 worst Δd′ 选出 X6c，Occam 规则（§21-S3）则会指向 C6（单一 `max` 规则）
或 C2（单一固定权重、robustness 代价最低、preservation 次高）。

## 10. 最终建议

> ### 🔶 **B — HOLD WINNER — one final validation needed**

**为什么不是 A（FREEZE）**：① 未达 §3 的 Tier S 等价条件；② **PASS 优势不显著**（ΔPASS +0.114，CI [−0.03,+0.29]）；
③ 存在**统计显著的 robustness 代价**（Δmean ΔR +0.034 [0.020,0.049]）；④ **E4（held-out category）不可得**；
⑤ Top1 与同族 3 个候选在统计上不可区分 —— 冻结一个「并列组中的某一个」需要外部判据。
**为什么不是 C（STOP）**：X6c 在 **PASS（27 vs 23）** 与**尾部安全（0 vs 4 catastrophic；worst Δd′ 的 bootstrap CI 不重叠）**上
确实优于冠军 B2 —— 不是「没有方法超过 B2」的情形。

**一次最终验证需要回答的三件事（建议）**
1. **尾部差异的统计强度**：4/35 vs 0/35 的事件数太少。建议在**同一 5 类上扩到 5 类 × 10 seeds（50 单元）**，
   或（更强）在**另一个数据集**（VisA / MVTec-LOCO / BTAD，需人工确认可获取性）上做 held-out（E4）。
2. **robustness 代价是否可接受**：fusion 的 mean ΔR 比 B2 高 0.034（显著）——需人工判定该代价是否值得换取尾部安全。
3. **并列组内的选择**：X6c（尾部最安全）/ C2（代价最低、preservation 最高、单规则）/ C6（最简单）三选一，
   当前数据**无法机械区分**；建议在扩充数据后按同一冻结键重排，或按 Occam 直接取 **C2**。

## 11. 成本、工程与错误

- **GPU**：26 units（X4 R1 6 + o1ext 20；其中 3 个 OOM 单元降并发重跑成功）= **4308 GPU·s ≈ 1.20 GPU·h**；
  score-side 全部 **0 GPU**（复用 Exp10/11/14 与本次 o1ext 的 per-image score）。
- **墙钟**：≈ 45 min（含 4 个分析阶段、5 张图、bootstrap 10000 次重采样）→ **在 90–120 min 目标内提前完成，未为跑满时间新增实验**。
- **错误**：`ERROR_REPORT.json` —— `o1ext_b2_56` 阶段 4-worker 并发在 hazelnut/screw/grid 触发 **CUDA OOM**
  （root cause 与 Exp14 相同：单进程 ~7.4 GiB × 4 > 24 GiB）→ 按规则**降并发至 2** + **只重跑 3 个失败单元** + **协议零改动**；
  其余 23 单元 0 failed。
- **协议**：完全复用 Exp14 的 strict-V2 + corrected-189 + `unit_metrics`；新增 spec 的 R6 核验把「层强度均来自 Exp14 已验证值」
  纳入证据（`R6_no_target_dependent_params` PASS）。

## 12. Limitations（必须与结论一起读）

1. **无 held-out 数据集**（E4 缺失）：所有结论仅在 MVTec 的 5 个类别上成立，且这 5 类正是方法开发所用类别。
2. **无 Tier S**：最高 27/35 ≈ 11.6/15 等价，未达 13/15。
3. **尾部差异 n=4 事件**：catastrophic 4→0 的效应量大，但事件数少，CI 宽。
4. **significant robustness cost**：fusion 的 robustness 增益显著小于 B2（这是「买保险」的代价，不是免费改进）。
5. **并列组不可区分**：X6c/C6/C2/X5 的差异在噪声内。
6. **X4（层强度组合）为 negative**：不排除更系统的层强度搜索有价值，但本轮按 §13 规则未扩展，也未做连续 grid。
7. **X5 的规则虽然 normal-only，但「β_c 越大越保守」这一映射本身是设计选择**；它未超过固定权重，故不构成新方法主张。
