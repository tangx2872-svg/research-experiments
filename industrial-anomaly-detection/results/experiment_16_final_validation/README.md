# Experiment 16 — Final Frozen Validation & Method Freeze Decision

**日期**：2026-10-08 ｜ **起点 HEAD**：`3dba831`（Exp15）｜ **LOCAL AHEAD OF ORIGIN: 10 commits**（**未 push**）
**环境**：AutoDL RTX 3090 24GB / conda `base` / PyTorch 2.8.0+cu128 / Anomalib 2.6.2
**冻结文件**：`PRE_RUN_PROTOCOL.md`（在读取任何 Exp16 target result 之前写入）

> **本轮不是方法搜索**，而是 Exp14–Exp15 筛选阶段的**最终冻结验证**。Exp15 的 verdict 是 `HOLD — one final validation needed`；
> Exp16 **不允许再 HOLD**，必须在 A/B/C 三选一。

---

## 0. Purpose（唯一核心问题）

> **X6c 的 defect-preservation tail-safety 优势能否在扩大 seed（新增 unseen block D = seeds 7–9）后继续成立，
> 并据此决定最终方法是否可以冻结。**

## 1. P0.5 — 实验债务审计（`EXPERIMENT_DEBT_AUDIT.md` + `experiment_debt_audit.csv`）

逐项核对 Exp14/15 的 README / config / plan.json / runner / progress / leaderboard / ERROR_REPORT / raw unit：

| 分类 | 数量 | 条目 |
|---|---:|---|
| **A — COMPLETE** | 23 | Exp14 `a1`/`b1`/`d`/`e_seed1`/`e_seed2`/`sanity_a1`/`o1_orig_34`/`o1_b2_34`；Exp15 `x4_r1`/`o1ext_orig_56`/`o1ext_b2_56`/全部 score-side；O1–O4 |
| **B — PRUNED BY RULE** | 3 | Exp15 `x4_r2`（X4 未晋级）；Family A 低排名候选未扩大；Family B 未扩展 screw/grid |
| **C — CLOSED** | 5 | X1（与 B2 数学重复）；X3（INSS 前提不满足）；A7（≡A2）；Exp14 `sanity_eq`（被 `a1`/`d` 取代的冗余）；Exp13-P defect-entanglement 叙事 |
| **D — TRUE MISSING** | **0** | — |
| **E — DEFERRED** | 3 | seeds 7–9 扩展（→ 本轮执行）；held-out 数据集 / E4；论文 baseline/ablation 补全 |

**关键核查结论**：① **Exp14+Exp15 共 100 个 `info.json`，非 OK = 0**，无 partial 目录；② OOM 的 3 个单元**全部补跑成功**；
③ Original/B2 在 5 类 × seeds 0–6 **全覆盖（35/35）**；④ 唯一未完成计划项为 `sanity_eq`（CLOSED，冗余）与 `x4_r2`（PRUNED）；
⑤ 60/90/110 min checkpoint 未执行**仅因 Exp15 在 40 min 完成**（定义上不适用）。
**额外补齐的证据缺口**（0 GPU，用已有 unit）：`E14_A6` ≡ Exp11 `E11_B2_L2only` → **max\|Δscore\| = 0.0（bit-exact）**。

> ## EXPERIMENT DEBT VERDICT: **CLEAN** → 立即进入 Exp16。

## 2. P1 — 冻结协议（摘要；完整见 `PRE_RUN_PROTOCOL.md`）

| 项 | 值 |
|---|---|
| Categories | bottle / cable / hazelnut / screw / grid（**5 类，禁增删**） |
| Seeds | **0–9** → **5 类 × 10 seeds = 50 units** |
| Methods | 仅 5 个：Original、Adaptive B2、C2、C6、**X6c** |
| 阈值 | `EPS_RZ=0.02`、`EPS_DP=0.10`、catastrophic `Δd′≤−0.25`（与历史**完全一致**） |
| 方法定义 | 逐字取自 Exp14/15：C2 = `0.35·z_o+0.65·z_B2`；C6 = `max(z_o,z_B2)`；**X6c = `mean(z_C2, z_C6)`**；z 用各自 clean_good 的 μ/σ |

## 3. P2 — 复用审计（`reuse_manifest.csv`，250 行）

```
REUSED (baseline units) = 70      [5 cats × 7 seeds × {Original, B2}]
NEW GPU UNITS           = 30      [seeds 7,8,9 × 5 cats × {Original, B2}]
SCORE-SIDE UNITS        = 150     [5 cats × 10 seeds × {C2, C6, X6c}] —— 0 GPU
```

## 4. P3 — Sanity（`analysis/sanity_checks.csv`）

| check | 结果 |
|---|---|
| **S1** historical reconstruction（5 个 (cat,seed) × {C2,C6,X6c} = 15 个三元组，与 Exp15 artifact 对比） | **PASS** —— `max\|Δ\| ≤ 1e-9` 且 PASS/catastrophic 标志完全一致 |
| **S2** X6c 定义 = `mean(z_C2, z_C6)`（代码级核验） | **PASS** |
| **S3** 阈值与 Exp15 config **bit-exact** | **PASS**（EPS_RZ 0.02 / EPS_DP 0.10 / catastrophic −0.25） |
| **S4** 输出隔离（只能写入 Exp16 目录） | **PASS** |

**4/4 PASS → 允许启动 GPU。**

## 5. P4 — GPU 补跑

| 项 | 值 |
|---|---|
| 单元 | **30**（seeds 7,8,9 × 5 类 × {Original (`E14_A0`), B2 (`E14_A1`) }） |
| 并发 | **3 workers**（Exp15 曾在 4 workers 下 OOM；本轮实测 peak **14.7 / 24 GB**，保留余量） |
| 结果 | **30/30 OK，0 failed**，avg **120 s/unit**，墙钟 **22.1 min** |
| 协议证据 | 每单元均落盘 `train_filter_evidence`（corrected 过滤，`kept_all_train_ids=True`）、`fit_replay_evidence`（R1）、`rng_evidence`（R2，`rng_after_restore` 哈希）、`bank_evidence`（bank SHA256） |

## 6. P5 — 最终 50-unit 表（`analysis/final_50unit_table.csv`）

| method | PASS/50 | PASS rate | catastrophic | cat rate | mean Δd′ | median Δd′ | **worst Δd′** | mean ΔR | complexity |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Original | 0/50 | 0.0% | 0 | 0.0% | 0.0000 | 0.0000 | 0.0000 | 0.0000 | low |
| *Adaptive B2* | *30/50* | *60.0%* | ***8*** | ***16.0%*** | *0.1438* | *0.0763* | ***−0.4944*** | *−0.0811* | low |
| C2 | 35/50 | 70.0% | 0 | 0.0% | **0.1857** | **0.1129** | −0.1817 | −0.0545 | low（单固定权重） |
| C6 | 38/50 | 76.0% | 0 | 0.0% | 0.1468 | 0.0511 | −0.2142 | −0.0455 | low（单规则 max） |
| **X6c** | **38/50** | **76.0%** | **0** | **0.0%** | 0.1831 | 0.0991 | **−0.1416** | −0.0493 | medium（两规则平均） |

**读数**：X6c 与 C6 并列最高 PASS（38/50 = 76%），比 B2（30/50 = 60%）**多 8 个单元（+16pp）**；
**catastrophic：B2 = 8 (16%)，C2/C6/X6c 全部 = 0**；**worst Δd′：X6c −0.142 为全表最佳**（B2 已到 −0.494）。

## 7. P6 — Seed-block 稳定性（`analysis/seed_block_stability.csv`）

| Block | n | Adaptive B2 | C2 | C6 | **X6c** |
|---|---:|---|---|---|---|
| A（seeds 0–2） | 15 | 12/15（cat 1，worst −0.285） | 11/15 | 10/15 | **11/15（cat 0，worst −0.040）** |
| B（seeds 3–4） | 10 | 7/10（cat 2，worst −0.410） | 8/10 | 9/10 | **8/10（cat 0，worst −0.110）** |
| C（seeds 5–6） | 10 | 4/10（cat 1，worst −0.341） | 7/10 | 7/10 | **8/10（cat 0，worst −0.048）** |
| **D（seeds 7–9，unseen）** | 15 | **7/15（cat 4，worst −0.494）** | 9/15 | 12/15 | **11/15（cat 0，worst −0.142）** |

**关键回答**：**X6c 在 unseen block D 中没有崩溃**（11/15 = 73%，与其整体 76% 相当；worst 仍为 −0.142，远高于 catastrophic 线）；
而 **Adaptive B2 在 block D 崩溃**（7/15 = 47%，**4 个 catastrophic**，worst −0.494）。
（副作用发现：C6 在 block D 反而是最高 PASS（12/15），但其 worst −0.214 差于 X6c 的 −0.142。）

## 8. P7 — 统计（`analysis/statistics.json`；bootstrap 10000，重采样单元 = (cat,seed)，paired）

| 比较 | ΔPASS rate | Δmean Δd′ | Δmean ΔR |
|---|---|---|---|
| **X6c vs B2** | **+0.160 [0.020, 0.300]（显著）** | **+0.0393 [0.002, 0.078]（显著）** | **+0.0318 [0.020, 0.044]（显著代价）** |
| X6c vs C2 | +0.060 [−0.020, 0.140]（ns） | −0.0026 [−0.015, 0.010]（ns） | +0.0052 [0.001, 0.010]（极小代价） |
| X6c vs C6 | 0.000 [−0.100, 0.100]（相同） | **+0.0363 [0.024, 0.049]（显著优于 C6）** | −0.0038 [−0.008, 0.000]（X6c 略更鲁棒） |

**Tail analysis（worst 1 / 2 / 3 / worst-5 均值）**

| method | worst 1 | worst 2 | worst 3 | **worst-5 均值** |
|---|---|---|---|---|
| Adaptive B2 | −0.4944 | −0.4119 | −0.4100 | **−0.4012** |
| C2 | −0.1817 | −0.1816 | −0.1303 | −0.1431 |
| C6 | −0.2142 | −0.1499 | −0.1414 | −0.1471 |
| **X6c** | **−0.1416** | **−0.1097** | **−0.1059** | **−0.0954** |

**Tail-5 均值的配对 bootstrap**：X6c − B2 = **+0.306 [0.289, 0.331]（显著）**；
X6c − C2 = **+0.048 [0.034, 0.062]（显著）**；X6c − C6 = **+0.052 [0.041, 0.063]（显著）**
→ **X6c 的 tail-safety 优势相对两个更简单的候选都是统计显著的**。

## 9. P8 — Robustness cost 决策（`analysis/robustness_cost.csv`）

| Method | Robustness（mean \|Δz\|，越低越好） | Preservation（mean d′，越高越好） | Tail safety（worst Δd′） | catastrophic | ΔR vs B2 | Complexity |
|---|---:|---:|---:|---:|---:|---|
| Adaptive B2 | **0.2332** | 5.1302 | −0.4944 | 8 (16%) | 0 | low |
| C2 | 0.2598 | **5.1721** | −0.1817 | 0 | +0.0266 [0.018, 0.035] | low |
| C6 | 0.2689 | 5.1332 | −0.2142 | 0 | +0.0357 [0.020, 0.051] | low |
| **X6c** | 0.2650 | 5.1695 | **−0.1416** | **0** | +0.0318 [0.020, 0.044] | medium |

**回答「tail-safety 改善是否值得 robustness cost」**：X6c 保留了 B2 鲁棒性增益的 **61%**
（B2 mean ΔR −0.0811 → X6c −0.0493；让出 39%），换来 **catastrophic 率 16% → 0%** 与
**worst Δd′ −0.494 → −0.142（3.5× 改善）**，且 preservation 与 B2 无显著差异（+0.039，其实更好）。
→ **在"缺陷保持 + 尾部安全"优先的工业场景下，该代价是可接受的**；同时 X6c 相对 C2/C6 的额外复杂度
（两规则平均）由**统计显著的 tail 优势**支撑（+0.048 / +0.052）。

## 10. 最终 verdict

> # `A — FREEZE X6c`
> **依据（对 CASE A 每条判据逐一核对）**：
> ① X6c 在 50 单元上稳定（38/50 = 76%，全表并列最高）；② **unseen block D（seeds 7–9）无崩溃**（11/15，worst −0.142）；
> ③ catastrophic **8 → 0**（B2 的 16% 归零）；④ **worst/tail Δd′ 优势显著**（worst-5 均值 +0.306 vs B2、+0.048 vs C2、+0.052 vs C6，全部 CI 不含 0）；
> ⑤ robustness cost **未恶化到不可接受**（保留 61% 的鲁棒性增益）；⑥ 相对 C2/C6 的**额外复杂度有可测的 tail 收益支撑**。
>
> **未选择 B（SIMPLE WINNER）的原因**：CASE B 的前提是「X6c 与 C2/C6 统计不可区分且额外复杂度无足够收益」——
> 实测 X6c 的 **worst-5 均值显著优于 C2 与 C6**、且 mean Δd′ 显著优于 C6，故前提不成立。
> **未选择 C（STOP）的原因**：X6c 相对 B2 的 PASS 优势已**统计显著**（+0.160，CI 不含 0），并非"优势消失"。
>
> **保留意见（诚实记录）**：C2 仍是「最简 + 最低 robustness cost（+0.0266）」的备选；若下游优先最小复杂度，
> 可人工改选 C2（其 tail 为 −0.182，仍远优于 B2 的 −0.494）。**方法搜索阶段在本轮正式结束。**

## 11. Limitations

1. **无 held-out 数据集（E4 缺失）**：所有结论限于 MVTec 5 类（且是方法开发所用类别）。
   已交付 `HELD_OUT_VALIDATION_PLAN.md`（推荐 **MPDD**：数据集本身以光照/油污变化为设计变量，规模小、anomalib 原生支持），
   但**本轮未下载、未运行**。
2. **robustness cost 是真实的**（+0.0318，统计显著）：X6c 换来了尾部安全，而不是全面支配 B2。
3. **catastrophic 的事件数仍有限**（B2 侧 8 个事件）：16% 的估计 CI 较宽（约 6–27%）。
4. **X6c 的复杂度略高于 C2/C6**（两规则平均），其正当性依赖 tail 指标；若审稿人只关心均值指标，则 C2 更简洁。
5. **X6c 的定义来源于经验组合**（Exp15 的 X6 家族），不是从第一原理推导；其可解释性弱于 C2（单一固定权重）。

## 12. Next step

1. **方法冻结**：X6c 作为最终方法候选（`A — FREEZE X6c`），**方法海选阶段结束**（禁止再开新候选）。
2. **E4 held-out 验证**（唯一实质缺口）：按 `HELD_OUT_VALIDATION_PLAN.md` 执行（需人工批准与下载）。
3. **论文阶段**：使用 `results/paper_assets/`（主表草稿 / evidence map / ablation inventory / Fig1–5）与根 README 总账。
4. **不再新增方法实验**；后续只允许 E4 与统计功效扩展（seeds 10–19）。
