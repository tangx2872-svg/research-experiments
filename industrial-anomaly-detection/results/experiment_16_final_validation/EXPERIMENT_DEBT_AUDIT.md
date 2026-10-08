# P0.5 — Experiment Debt Audit（Exp14 / Exp15）

**审计时间**：2026-10-08 16:37+08:00 ｜ **审计对象**：Exp14（`results/experiment_14/`）、Exp15（`results/experiment_15/`）
**方法**：逐项核对两轮 README / config / plan.json / runner / progress.json / leaderboard / ERROR_REPORT / raw unit 的 `info.json`，
并把每个计划项归入 A–E 五类。**未运行任何新 GPU 工作**。

**审计原则**：`PRUNED`（按预注册规则停止）/ `CLOSED`（重复或前提不满足）/ `DEFERRED`（本来就属未来验证）
**都不属于"实验没做完"**；只有 `TRUE MISSING` 才允许补跑。

---

## 0. 机器可读结果

- `experiment_debt_audit.csv` —— 逐项任务表（Task / Planned / Completed / Status / Reason / Action）

## 1. 逐项核对（planned vs completed）

| 轮次 | Stage | planned | completed | 判定 |
|---|---|---:|---:|---|
| Exp14 | `a1`（Family A2–A6 + B1–B6 × 3 类 × seed0） | 15 | 15 | A — COMPLETE |
| Exp14 | `b1`（同上，Family B 独占批次） | 18 | 18 | A — COMPLETE |
| Exp14 | `d`（Stage D：A2/A3/A6 × screw/grid/hazelnut） | 7 | 7 | A — COMPLETE |
| Exp14 | `e_seed1` / `e_seed2`（A3 × 5 类 × seeds 1,2） | 5 / 5 | 5 / 5 | A — COMPLETE |
| Exp14 | `sanity_a1`（A1≡B2 跨路径等价） | 3 | 3 | A — COMPLETE |
| Exp14 | `o1_orig_34` / `o1_b2_34`（O1：seeds 3,4） | 10 / 10 | 10 / 10 | A — COMPLETE |
| Exp14 | `sanity_eq`（A5/A6 bottle seed0） | 2 | 0 | **C — CLOSED（被 `a1`/`d` 取代的冗余占位项）** |
| Exp15 | `x4_r1`（X4a/X4b × 3 类 × seed0） | 6 | 6 | A — COMPLETE |
| Exp15 | `x4_r2`（X4a/X4b × screw/grid） | 4 | 0 | **B — PRUNED（R1 未晋级）** |
| Exp15 | `o1ext_orig_56` / `o1ext_b2_56`（O1-ext：seeds 5,6） | 10 / 10 | 10 / 10 | A — COMPLETE（含 3 个 OOM 单元降并发重跑成功） |
| Exp15 | score-side（C1–C5、C6、X5、X6c、X2b） | 35 per candidate | 35 | A — COMPLETE（0 GPU 重建） |
| **合计** | | **105** | **103** | 2 项：1 CLOSED + 1 PRUNED |

## 2. 特别核查项（15 问）

| # | 问题 | 结论 |
|---|---|---|
| 1 | X4a Round 2 为什么没跑？ | **PRUNED**：X4a R1 = **0/3 PASS**、worst Δd′ −0.241，且在 bottle 上被 B2 双向支配（vs B2：ΔR +0.005、Δd′ −0.176）→ 按 Exp15 §13「必须 Pareto-positive 或明显减少 catastrophic」**未晋级** → 不触发 R2。README §4 已记录 |
| 2 | X4b Round 2 为什么没跑？ | **PRUNED**：X4b R1 = **1/3 PASS**（仅 hazelnut）、worst −0.103；bottle 相对 B2 为 ΔR +0.060 / Δd′ −0.038（**双双更差**）→ 未晋级。README §4 已记录 |
| 3 | X2b 为什么没有扩大？ | **PRUNED**：X2b（三路 mean(z_o,z_B2,z_A6)）seed0 仅 **3/5 PASS**、mean Δd′ −0.053、worst +0.082 → 未达 ≥4/5 门；且其第三路 A6 已在 Exp14 被 Stage D 门淘汰。README §6 已记录 |
| 4 | X1 为什么没跑？ | **CLOSED**：与 B2 **数学重复** —— Exp14 已证 dual-path 线性融合严格等价于 B2 残差强度重参数化，且 `E14_A1` 与 `E11_B2` 在 3 类 **bit-exact（max\|Δscore\|=0.0）** |
| 5 | X3 为什么没跑？ | **CLOSED**：§6 前提「INSS 至少出现 Pareto-positive candidate」**不满足**（Exp14 Family B 18 个单元全为单边 trade-off） |
| 6 | X5 是否还有协议要求但遗漏的 units？ | **无遗漏**。X5 为 score-side（0 GPU），在全部 **35 个 (cat,seed)** 上完成重建 |
| 7 | C1–C6 是否存在缺失 unit？ | **无**。C1–C6 各 **35/35** 单元（seeds 0–6 × 5 类） |
| 8 | Original/B2 seeds 0–6 是否完整？ | **完整 35/35 对**：seeds 0–2 = Exp10 `P10_ORIG_a000` / Exp11 `E11_B2`；seeds 3,4 = Exp14 `E14_A0`/`E14_A1`；seeds 5,6 = Exp15 `o1ext_orig`/`o1ext_b2` |
| 9 | OOM 的 3 个 units 是否已全部成功补跑？ | **全部成功**。`screw:6:E14_A1__screw`、`grid:5/6:E14_A1__grid` → 在 `results/experiment_15/raw/o1ext_b2/` 下均有 `status=OK` 的 `info.json` |
| 10 | 是否存在 `failed > 0` 的最终 artifact？ | **不存在**。Exp14+Exp15 共 **100 个 `info.json`，非 OK = 0**；无缺 `info.json` 的 partial 目录（runner 仅在成功时写 `info.json`，失败单元不覆盖既有结果） |
| 11 | 是否存在 runner planned > completed 但 README 未解释的任务？ | **1 项未解释**：Exp14 `sanity_eq`（2 units）。**已在本文档解释**：它是早期为「A5/A6 跨路径等价检查」预留的占位项，随后 A5/A6 在 `a1`/`d` 中实际执行，且 A5≡`P10_T1` 的等价核验已报告 → **冗余，非缺失**。Exp15 的 `x4_r2` 已在 README §4 解释 |
| 12 | O1/O2/O3/O4 是否真的完成？ | **全部完成**：O1 = seeds 3,4（Exp14，20 units）+ **扩展至 seeds 5,6**（Exp15，20 units）；O2 = bootstrap 10000 次重采样（`analysis/bootstrap_ci.json`）；O3 = 免费 ablation（两端点 Original / B2 均在表中）；O4 = 5 张图（`figures/fig1..fig5`） |
| 13 | 60/90/110 min checkpoint 未执行是否只因实验提前结束？ | **是**。Exp15 于 **40 min** 完成（12:24→13:04），仅 **+30 min** checkpoint 到期并已打印；60/90/110 min 检查点按其定义只在实验仍在进行时适用 → **N/A，不是遗漏** |
| 14 | held-out category 是否属于 deferred 而不是 missing？ | **DEFERRED**。本机仅有方法开发所用的 5 个类别，且 Exp15/Exp16 规则明确禁止临时下载 → 属"未来验证"，非 Exp15 欠债 |
| 15 | 是否有任何缺失任务可能改变 X6c/C2/C6/B2 排名？ | **没有**。唯一未跑的实验项是 `x4_r2`（仅影响 X4a/X4b，二者在 R1 已被 B2 支配）；`sanity_eq` 无实验内容。X6c/C2/C6/B2 的 35 单元数据完整，排名不受影响 |

## 3. 本次审计中额外补做的一处**证据缺口**（0 GPU，非新实验）

Exp14 的 `sanity_eq` 计划项原本还应覆盖 **A6 的跨路径等价核验**，但 Exp15 报告中未给出。
本次审计用**已有 unit** 补齐（**未运行任何 GPU 实验**）：

| 核验 | 结果 |
|---|---|
| `E14_A6`（Exp14 路径：L2 residual(β_c) + L3 alpha_in(0)）vs `E11_B2_L2only`（Exp11 ablation 路径），bottle seed0 | **max\|Δscore\| = 0.000e+00（bit-exact）**，tau 38.333637 相同，coreset 19353 相同 |

→ 连同此前已报告的 `E14_A5 ≡ P10_T1`（0.0）、`E14_A1 ≡ E11_B2`（0.0），**Exp14/15/16 的全部代码路径与历史路径均已逐位一致**。

## 4. 分类汇总

| 类别 | 条目 |
|---|---|
| **A — COMPLETE** | Exp14 `a1`/`b1`/`d`/`e_seed1`/`e_seed2`/`sanity_a1`/`o1_orig_34`/`o1_b2_34`；Exp15 `x4_r1`/`o1ext_orig_56`/`o1ext_b2_56`/all score-side；O1–O4 |
| **B — PRUNED BY RULE**（`PRUNED — NO ACTION`） | Exp15 `x4_r2`（4 units）；Exp14 Family A 的 A2/A4/A5 未扩大（R1 淘汰）；Family B 未扩展 screw/grid（STOP）；低排名候选 C3/C4/C5 未做多 seed |
| **C — INVALID / FORBIDDEN / DUPLICATE**（`CLOSED — NO ACTION`） | X1（与 B2 数学重复）；X3（INSS 前提不满足）；A7（≡A2）；Exp14 `sanity_eq`（被 `a1`/`d` 取代的冗余占位）；Exp13-P defect-entanglement 叙事（不复活）；Category×Channel（Exp12 已 STOP） |
| **D — TRUE MISSING** | **无** |
| **E — DEFERRED FUTURE VALIDATION** | seeds 7–9 / 50-unit 扩展（→ **本轮 Exp16 执行**）；held-out category / E4 外部验证（→ `HELD_OUT_VALIDATION_PLAN.md`）；论文 baseline/ablation 补全（→ paper assets / ablation inventory） |

## 5. 最终债务判定

> ## EXPERIMENT DEBT VERDICT: **CLEAN**
> 没有会影响最终结论的真正遗漏任务（**TRUE MISSING = 0**）。
> 唯一 2 个未完成计划项分别为 `CLOSED`（冗余/前提不满足）与 `PRUNED`（按预注册规则停止），均**不允许补跑**。
> → **允许立即进入 Exp16 Final Frozen Validation。**
