# OVERNIGHT FINAL REPORT — Experiment 7A-O + Queue Q1–Q5

**Session**：2026-10-06 22:17:05 → 2026-10-07 02:09（wall-clock **≈ 3 h 52 min**）
**协议**：`Experiment 7A-O` 主指令 + `ADDENDUM — Overnight Continuation Queue`
**性质**：**EXPLORATORY / selection-set**。今晚选出的任何配置**不得**直接作为论文 confirmatory result。

---

## 0. 一页式摘要

```text
Total runtime        : 3 h 52 min wall-clock (budget 8-10 h; early finish allowed by section 17)
GPU runtime          : 7.11 GPU-hours (182 completed units with recorded runtime; mean 140.6 s/unit)
New GPU units        : 170 protocol units (+12 accidental stress runs, quarantined, unused)
Completed queues     : Q0, Q2, Q3, Q4, Q5
Failed queues        : none  (Q1 SKIPPED by rule: Q0 = CASE_C -> no winner)

Best baseline        : Uniform alpha=0.40091275   d' = 4.7940 | |dz| = 0.2134  (5 cat x 3 seeds)
                       strongest competitor alpha=0.20: d' 5.0094 | |dz| 0.2777
Best single module   : NONE qualifies (0 / 14 frozen configs reached Tier S/A/B)
                       highest preservation: A3_lam050 (energy-preserving residual, lambda=0.5)
Best composition     : Q2L2_l2res50_l3uni (+0.5793 d' / +0.0429 |dz| vs Uniform, 3 cat x seed0; NOT a winner)

Strongest preservation : A3_lam050  d' 5.3552 (+0.5612 vs Uniform) but |dz| 0.2753 (+0.0619)
Strongest robustness   : Uniform 0.40091275  |dz| 0.2134 (12-condition stress mean 0.3480)

Pareto winner        : Uniform 0.40091275 stays non-dominated; NO candidate dominates it
                       A3_lam050 strictly dominates B0_original (+0.3700 d' AND -0.0352 |dz|)
Seed stability       : A3_lam050 3/3 seeds same direction (dd' +0.49...+0.60); Q3 curves over 3 seeds
Category stability   : A3_lam050 worst-category dd' = -0.049 (within eps = 0.10), negTr = 0

Main positive finding : two ancillary assets are positive:
                        (i) alpha=0.20 Pareto-dominates alpha=0 (d' 5.0094 > 4.9852 AND |dz| 0.2777 < 0.3105)
                        (ii) Uniform degrades ~33% more slowly than Original as illumination severity grows
                             (pooled slope 1.886 vs 2.825 over 12 conditions)
Main negative finding : all 14 simple representation modules fail to push the frontier (7A-O CASE_C);
                        layer-selective strength (Q0 family C) and layer x representation composition
                        (Q2) also fail to push it
Most surprising       : the L2/L3 intervention-slot asymmetry is very strong -- the same residual
                        intervention helps at L2 (+0.5793/+0.0429) but hurts on BOTH axes at L3
                        (-0.0489/+0.1184); and alpha >= 0.80 is strictly dominated by alpha ~ 0.60

Recommended next exp : no further tuning of the current feature-intervention family. To move the
                       frontier, change the objective axis itself (inference-time illumination
                       correction, or an explicit illumination-invariance objective), under a new
                       pre-registered protocol + new baselines.
Paper progress       : 7A-O completes one evidence cell ("simple representation modules cannot push
                       the frontier"); Q3/Q4 add a severity-response curve and a completed
                       strength-response map
Git HEAD             : <FINAL_HEAD>
origin/main          : <FINAL_ORIGIN>
GPU status           : idle (0 % utilization)
```

---

## 1. Queue 总览与冻结哈希

| Queue | 状态 | GPU units | 关键结论 |
|---|---|---|---|
| **Q0** 7A-O module screening | **DONE** | 92 | **CASE_C — NO USEFUL MODULE**；Round1 晋级集为空，full panel 5×3 一致 |
| **Q1** winner local robustness | **SKIPPED** | 0 | 启动条件（存在 CASE_A/B/D winner）不满足 |
| **Q2** layer × representation composition | **DONE (STOP)** | 12 | 0/4 达 Tier；**强 L2/L3 非对称**（L2 有效、L3 有害） |
| **Q3** illumination stress-test | **DONE** | 10 (+20 in Q4-B) | 12 conditions；medium 档与历史 `max|Δscore|=0`；slope −33% |
| **Q4** strong-baseline extended validation | **DONE** | 56 | 9 个 α 点补齐 15/15；α=0.20 支配 α=0；α≥0.8 被 α≈0.60 支配 |
| **Q5** CPU paper assets | **DONE** | 0 | registry / evidence map / negative results / figure inventory / main table |

```text
Q0 policy_freeze   : e046a89a852da9aad55d2eafeea1c3bc05a5aff640be4338f7e515e53150f224
Q2 protocol_freeze : c76ee8f358e6d159e78df7fff8f9dc30e295df175167141c44fd9de3ee3c0d7c
Q3 protocol_freeze : 6c4ee3d07c081c23026a82fb3a426494675f76947785e0b32eb8e256144a91ec
Q4 protocol_freeze : 36c4115262fa70fb112a189a0363d317a7605d5e05c1f8cec7fa89db59c2bab2
```

## 2. Q0 — 7A-O 结果（full panel 5 cat × 3 seeds）

smoke：α=0 与 Uniform 相对历史 **`max|Δscore| = 0.000e+00`**；sanity **18/18 PASS**；
Round 1（3 cat × seed0）**14/14 candidate tier = "—"**。Uniform 基线：d′ 4.7940 / \|Δz\| 0.2134。

| config | family | d′ | \|Δz\| | Δd′ | Δ\|Δz\| | worst cat Δ | negTr | tier |
|---|---|---|---|---|---|---|---|---|
| A3_lam050 | A | **5.3552** | 0.2753 | **+0.5612** | +0.0619 | −0.0493 | 0 | — |
| B1_g100 | B | 5.3044 | 0.2850 | +0.5104 | +0.0717 | −0.0001 | 0 | — |
| D1_ln_s040 | D | 4.9678 | 0.3298 | +0.1739 | +0.1164 | −0.2796 | 1 | — |
| C3_l2_045_l3_035 | C | 4.8159 | 0.2248 | +0.0219 | +0.0115 | −0.0538 | 0 | — |

**Pareto**：`A3_lam050 ⊃ {B0_original, B1_g100, D1}`；**无 candidate 与 Uniform 相互支配** → **CASE_C**。
A2 = NOT IMPLEMENTED（无安全的无训练固定投影）；B2 = NOT RUN。
协议歧义 `7A-O-A1`（§14 空晋级 vs §15 Top-1-per-family）在**任何 Round-2 数据之前**写入
`reference/round2_amendment.json`，以 4 个已冻结 config 跑 full-panel diagnostic，
结论与机械口径一致（CASE_C 被加强而非推翻）。

## 3. Q2 / Q3 / Q4 结果

**Q2**（3 cat × seed0；Uniform 子集 d′ 5.4467 / \|Δz\| 0.1660）

| config | L2 | L3 | d′ | \|Δz\| | Δd′ | Δ\|Δz\| | 兑换率 |
|---|---|---|---|---|---|---|---|
| Q2L2 | residual λ=0.5 | Uniform | 6.0260 | 0.2089 | +0.5793 | +0.0429 | **13.5** |
| Q2L1 | Uniform | Original | 5.6712 | 0.2725 | +0.2245 | +0.1065 | 2.11 |
| Q2L0 | Original | Uniform | 5.5686 | 0.2132 | +0.1218 | +0.0472 | 2.58 |
| Q2L3 | Uniform | residual λ=0.5 | 5.3979 | 0.2844 | −0.0489 | +0.1184 | — |

**Q3**（2 methods × 5 cat × 3 seeds × 12 conditions）

| method | d′ | \|Δz\| 12-cond | \|Δz\| medium-4 | worst cond | slope pooled | slope brightness | slope gamma |
|---|---|---|---|---|---|---|---|
| M0_original | 4.9852 | 0.5071 | 0.3105 | 2.3286 | 2.825 | 3.786 | 1.865 |
| M1_uniform | 4.7940 | **0.3480** | **0.2134** | **1.4463** | **1.886** | 2.442 | 1.329 |

medium 档在 5×3 上精确重现 canonical baseline（0.3105 / 0.2134）→ 扩展管道与冻结口径无缝衔接。

**Q4-A**（9 个历史 α × 5 cat × 3 seeds，全部 15/15）

| α | 0.0 | 0.125 | **0.20** | 0.25 | 0.30 | **0.4009** | 0.50 | 0.601 | 0.802 |
|---|---|---|---|---|---|---|---|---|---|
| d′ | 4.9852 | 4.9649 | **5.0094** | 4.9320 | 4.8831 | 4.7940 | 4.6670 | 4.5078 | 4.2272 |
| \|Δz\| | 0.3105 | 0.2867 | 0.2777 | 0.2503 | 0.2394 | 0.2134 | 0.2082 | **0.1895** | 0.1902 |
| AUROC | 0.9866 | 0.9876 | **0.9877** | 0.9869 | 0.9864 | 0.9827 | 0.9757 | 0.9710 | 0.9597 |

支配关系（本 9 点内）：**α=0.20 支配 α=0 与 α=0.125**；**α=0.6014 支配 α=0.8018**。
跨来源一致性 sanity：新 α 上"历史 bottle/grid"与"新增 cable/hazelnut/screw"的 \|Δz\| 量级一致 → 无来源伪影。

## 4. Sanity / 异常（全部保留记录）

- **OOM**：Family B 的 concat 使 embedding 1536→3072、峰值显存 ×2；3-worker 下 2 个单元 OOM →
  按 §26 **自动降级为 2 workers + 内存感知分组**（B1 独立 lane），未等待人工；失败单元保留并重跑成功。
- **参数顺序 bug**（首轮 Q4）：`--units` 约定 `category:seed:CONFIG` 被误传为 `CONFIG:category:seed`，
  w1/w2 立即退出、w3 把 4 个 α-point 交给 stress-runner → **12 个误跑单元**，
  已隔离至 `results/experiment_7a_o_q3/accidental_stress_runs/`（**未删除、未用于任何分析**）。
  修正后 Q4-A 36/36、Q4-B 20/20 完成。
- **S5 检查口径修正**：拆为 primary（必须有限）+ secondary（NaN 全部来自 5A 来源、其源资产不含 pixel 指标）；
  **S7 修正**为 per-category 行数一致性。二者均为检查实现缺陷，非数据问题（已在 sanity_report.md 注明）。
- **无数据损坏、无 baseline mismatch**：所有 equivalence 检查 `max|Δscore| = 0`。

## 5. 明确未做的事（§27 / FINAL RULE）

未修改任何 threshold / ε / metric / category / seed；未做 dense α sweep；未做 per-category tuning；
未新增 dataset / backbone / illumination family；未把非 winner 塞进 winner 槽位；
所有负结果保留（`summary/negative_results.md`）。**未启动**新方法线、Final baseline + ablation、Writing。

## 6. 明早建议

1. **停止**当前 feature-intervention 家族的调参（λ / γ / layer 分配 / norm 类型）。
2. 若要继续 push frontier，必须改**目标轴**（inference-time illumination 校正，或显式
   illumination-invariance 目标），并配新预注册协议 + 新 baseline。
3. 可直接用于论文的两个新资产：**Q4-A 的 α 强度响应图**（含 α=0.20 ⊃ α=0）与
   **Q3 的 severity 退化曲线 + slope**（Uniform slope 比 Original 低 ~33%）。
4. 5D 的 CASE 归属裁决与方法学方向裁决仍需**人工**（根 README §11/§12）。

---

> 详细逐队列记录：`experiments/experiment7a_o/README.md`、
> `results/experiment_7a_o_q2/README.md`、`results/experiment_7a_o_q3/README.md`、
> `results/experiment_7a_o_q4/README.md`、`results/experiment_7a_o/summary/`。
