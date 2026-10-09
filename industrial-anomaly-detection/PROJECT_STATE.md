# PROJECT_STATE — 科研项目快速恢复入口

> **本文件是唯一「快速恢复入口」**。任何新会话 / 新研究者接手本项目时，**先读本文件**。
> 最后更新：2026-10-08 ｜ HEAD：`3615f84` ｜ 依据：Exp16 冻结验证 + `docs/RESEARCH_RECOVERY_AUDIT.md`
> **所有数字均从正式 artifact 重新读取确认**（路径逐项标注）；未在本文件出现的结论不属于项目正式结论。

---

## 1. CURRENT PAPER STAGE

| 项 | 状态 |
|---|---|
| 阶段 | **Measurement Validity Audit**（2026-10-09 由 E5-FAILURE-AUDIT 的指标病理触发）；前一阶段 Phase III Module Composition 已 STOP（E5-1 = `0 GO / 1 HOLD / 3 STOP`） |
| Method discovery / search | **CLOSED（已正式结束）** |
| 冻结方法 | **X6c** — 内部 `A — FREEZE X6c`；但 **E4-X 外部（M²AD Bird）已 KILLED**，见 §13 |
| 内部验证 | **完成**（5 类 × 10 seeds = 50 单元） |
| Module composition | **E5-1 BROAD SCREENING 已完成（2026-10-09）** ⇒ `0 GO / 1 HOLD / 3 STOP` ⇒ 协议 §17 **case C**：**STOP Module Composition 分支**（见 §13） |
| 主 E4 数据集 | **M²AD**（E4-0C winner；E4-D0 integrity **PASS**；E4-D1 Original smoke **PASS**） |
| 下一步 | **人工裁决**（E6-A 已完成，CPU-only，**GPU 全程空闲**）：① 冻结 Axis Y = **`M5_A`**（门用**并列校正 AUROC** + `Δd′ ≥ −0.10`），伴轴 `M3_good`、反向对照 `M4_ng`；② 先解释 **C01 的指标符号分歧**（`M5_A`/`M3_*` 说更好，`M1_R_all`/`M4_*` 说更差）；③ `R_all` 已降为 SECONDARY，**不得再作首要鲁棒性指标**。**禁止自动启动 GPU、禁止 multi-view/multi-seed、禁止 rescue C06/C07/C08、禁止调参、禁止补 C10。** |

---

## 2. CORE RESEARCH QUESTION

工业异常检测（PatchCore 类）在**照明变化**下会退化。常见的补救是引入 **instance normalization（α-IN）**：
它对光照更鲁棒，但会**改变特征分布并损害缺陷保持（defect preservation）**。
本项目研究：**illumination robustness 与 defect preservation 之间的 trade-off 到底由什么决定，
以及能否在不牺牲 preservation 的前提下获得鲁棒性**（不是「找更大 backbone」，而是表征/融合层面的机制与轻量方法）。

---

## 3. CURRENT EVIDENCE CHAIN（每格：结论 + 实验编号）

```
Problem           照明变化导致 normal 分布漂移                 ← Exp1
   ↓
defect-dependent  α-IN 的代价依缺陷类型/类别而不同（非均匀）    ← Exp1B / 1C（seed 稳定）
   ↓
confound exclusion size(面积)不能解释该差异（控制后 type 仍显著）← Exp1D（ΔR²=0.188, CASE_A）
   ↓
cross-category    现象跨 5 类成立（25/25 types 3/3 seed 稳定） ← Exp1E（CASE_A）
   ↓
simple-property   图像空间属性无法解释异质性（|ρ|≤0.24）       ← Exp1F（CASE_D）；8B（CASE_B）
   ↓
layer-specific    层级选择性：shrink=L2 驱动、neutral/expand=L3 ← Exp1G(链条) / 1H / 1H-S / 1I(控制)
   ↓
transmission      geometry → NN → score 干预级传导成立          ← Exp1J-A / 1J-B（CASE A，已复现）
   ↓
method screening  5A→5A-H(固定规则 STOP) → 5B(CASE_A 但机制 null) → 5C/5D(identity 未确立)
                  → 6A/6B(soft gating CASE_C) → 7A-O/Q2(无模块晋级) → 9A/9B(无候选晋级)
                  → 9C(评测协议 V2：噪声地板 → 0) → 9B-R(V2 下重审)
                  → 10(Fixed C2 = 首个 FRONTIER-BREAK) → 11(Adaptive B2 = 12/15)
                  → 12(Category×Channel STOP) → 13-P(INSS 机制叙事被否决)
                  → 14(Family A 代数否定 / Family B STOP / late fusion 胜出)
   ↓
X6c               C2 与 C6 的保守融合：唯一在 unseen seed block 不退化的候选 ← Exp15（PATH A）
   ↓
Exp16 final       50 单元冻结验证：X6c 38/50、0 catastrophic；B2 30/50、8 catastrophic → FREEZE
```

---

## 4. FROZEN METHOD

### X6c（正式定义 —— 引用实现，不得凭记忆重写）

```
z_o  = (score_Original  − μ_o)/σ_o      # μ/σ 取自 Original 路径自己的 clean_good/none
z_b  = (score_AdaptiveB2 − μ_b)/σ_b     # 同上，各自路径独立校准（normal-only）
z_C2 = 0.35 * z_o + 0.65 * z_b
z_C6 = max(z_o, z_b)
X6c  : score = mean(z_C2, z_C6)          # 即 0.5*z_C2 + 0.5*z_C6
```

| 组件 | 正式定义 | 实现路径 | 冻结配置 |
|---|---|---|---|
| **X6c** | `mean(z_C2, z_C6)` | `scripts/experiment16_fusion.py::rule("X6c")` | Exp15 `config/candidate_registry.json`（X6 家族） |
| **C2** | `0.35·z_o + 0.65·z_b`（固定权重，**不再细化**） | 同上 `rule("C2")`；`W_C2 = 0.35` | Exp15 §7 敏感度曲线（w grid 冻结 {0.20,0.35,0.50,0.65,0.80}） |
| **C6** | `max(z_o, z_b)` | 同上 `rule("C6")` | Exp15 `config/candidate_registry.json`（X6b） |
| **Adaptive B2** | L2 `residual(β_c)` + L3 `alpha_in(0.25)`（β_c = `clip(0.25·(q_c/q_ref)³, 0.05, 0.50)`，normal-only） | `scripts/experiment14_candidates.py::_resid/_alpha`；`scripts/experiment11_candidates.py::beta_table()` | Exp11 `config/` |
| **Original** | PatchCore α=0（`alpha_in(0)` 严格 short-circuit） | `scripts/experiment14_candidates.py::ALL14_SPECS["E14_A0__*"]` | Exp10/14 |

**注意**：C2/C6/X6c 都是 **score-level（late fusion）** 方法，**不改变表征**、不需要训练、**0 GPU**（只需两条现成 detector 的 per-image score）。

---

## 5. FINAL INTERNAL VALIDATION（Exp16，全部从 artifact 重读）

**来源**：`results/experiment_16_final_validation/analysis/final_50unit_table.csv`、`statistics.json`、`seed_block_stability.csv`

| 项 | X6c | Adaptive B2 | C2 | C6 |
|---|---|---|---|---|
| 规模 | 5 categories × **10 seeds** = **50 units** | 同 | 同 | 同 |
| **PASS** | **38/50 = 76.0%** | 30/50 = 60.0% | 35/50 = 70.0% | 38/50 = 76.0% |
| **catastrophic** | **0/50** | **8/50 = 16.0%** | 0 | 0 |
| **worst Δd′** | **−0.1416** | **−0.4944** | −0.1817 | −0.2142 |
| mean Δd′ | 0.1831 | 0.1438 | 0.1857 | 0.1468 |
| mean ΔR（越低越鲁棒） | −0.0493 | −0.0811 | −0.0545 | −0.0455 |

**统计（bootstrap 10000，resampling unit = (category,seed)，paired）**

| 比较 | ΔPASS rate | Δmean Δd′ | Δmean ΔR |
|---|---|---|---|
| X6c vs B2 | **+0.160 [0.020, 0.300]（显著）** | **+0.0393 [0.002, 0.078]（显著）** | **+0.0318 [0.020, 0.044]（显著代价）** |
| X6c vs C2 | +0.060 [−0.020, 0.140]（ns） | −0.0026（ns） | +0.0052（极小代价） |
| X6c vs C6 | 0.000（相同） | **+0.0363 [0.024, 0.049]（X6c 显著更好）** | −0.0038（X6c 略更鲁棒） |

**Tail evidence（Δd′，越大越安全）**

| method | worst 1 | worst 2 | worst 3 | **worst-5 均值** |
|---|---|---|---|---|
| B2 | −0.4944 | −0.4119 | −0.4100 | −0.4012 |
| C2 | −0.1817 | −0.1816 | −0.1303 | −0.1431 |
| C6 | −0.2142 | −0.1499 | −0.1414 | −0.1471 |
| **X6c** | **−0.1416** | **−0.1097** | **−0.1059** | **−0.0954** |

worst-5 均值配对 bootstrap：X6c−B2 **+0.306 [0.289,0.331]**、X6c−C2 **+0.048 [0.034,0.062]**、X6c−C6 **+0.052 [0.041,0.063]**（**全部显著**）。

**unseen seed block D（seeds 7–9，Exp16 新跑）**：X6c **11/15（worst −0.142，0 catastrophic）**；
Adaptive B2 **7/15（4 catastrophic，worst −0.494）** → **X6c 不崩溃，B2 崩溃**。

---

## 6. WHAT IS FROZEN（E4 及以后**不得修改**）

依据：`results/experiment_16_final_validation/PRE_RUN_PROTOCOL.md`（sha256 见 `config/sha256_frozen.json`）

1. **方法定义**：X6c / C2 / C6 的公式与权重（`0.35/0.65`、`0.5/0.5`）—— 禁止重新调权重。
2. **score normalization**：`z = (score − μ_g)/σ_g`，μ_g/σ_g 取自**该路自己的 clean_good/none**（normal-only；禁止用 defect/test 校准）。
3. **阈值**：`EPS_RZ = 0.02`、`EPS_DP = 0.10`、`catastrophic: Δd′ ≤ −0.25`（与 5A-H/7A-O/Exp10–16 完全一致）。
4. **PASS 定义**：相对**同 (category, seed)** 的 corrected Original：`ΔR ≤ −0.02 且 Δd′ ≥ −0.10`。
5. **评价指标**：`experiment5a_h_analysis.unit_metrics`（mean|ΔNormalScore_z|、mean defect d′、per-shift Δz、image AUROC…）。
6. **协议**：corrected-189 train filter + strict-V2 RNG（R1 `Engine.fit` 入口、R2 `select_coreset_idxs` 入口 replay）。
7. **候选选择逻辑**：Exp16 已关闭 method search；**不得新增候选**（X7/X8/…），不得复活已关闭方向（§7）。
8. **统计口径**：resampling unit = (category, seed)；bootstrap ≥10000；报告 point estimate + 95% CI（不只报 p 值）。

---

## 7. CLOSED / FAILED DIRECTIONS（一句话原因，避免重复实验）

| 方向 | 关闭原因 |
|---|---|
| **X1**（Top feature + B2） | 与 B2 **数学重复**（dual-path 线性融合 ≡ B2 残差强度重参数化，已 bit-exact 验证） |
| **X2b**（三路 mean: o+B2+A6） | 无增益（seed0 3/5）；第三路 A6 已在 Exp14 Stage D 门淘汰 |
| **X3**（Dual-Path + mild INSS） | 前提不满足 —— Exp14 Family B（INSS）**无任何 Pareto-positive 候选** |
| **X4a/X4b**（层强度组合） | R1 **0/3 · 1/3**，且在 bottle/cable 上被 B2 双向支配 → 未晋级（未触发 R2） |
| **X5**（normal-only category-adaptive 融合） | 未超过固定权重（PASS 相同、worst 更差）→ honest negative |
| **C1**（Exp14 规则 winner） | 在 unseen seeds 上退化（12/15 → 6/10）→ 属 **seed 选择效应**，已归档 |
| Family A（Dual-Path 线性融合） | **代数上**只是 B2 强度重参数化，无法产生 Pareto 改进 |
| Family B（Tiny INSS，layer2 投影） | 全家族单边 trade-off；λ=0.5 时 robustness **反而更差** → STOP |
| Category × Channel gating（Exp12） | M1/M2/M3 全部触发硬性淘汰 → 0 ADVANCE |
| INSS / defect-entanglement 叙事（Exp13-P） | illumination subspace 与 defect 方向**非特异**（normal 对照相同）→ 机制叙事被否决 |
| Geometry gating 系列（5C/5D/6A/6B） | identity 未确立（5D）；6B 为 CASE_C（strength 效应） |
| 机制扩张（1J-C / 2A / 新 mechanism probe） | 机制阶段 **FROZEN**（除非 Stage ⑦ 之后暴露具体证据缺口） |
| Mahalanobis / whitening / frequency enhancement | 仅允许作**成熟 baseline/probe**，**不得**包装为原创主路线 |

---

## 8. CURRENT LIMITATIONS

1. **方法开发与验证全部集中在 MVTec 5 类**（bottle/cable/hazelnut/screw/grid）——**且正是方法开发所用类别**，存在过拟合风险。
2. **尚无真正的 held-out external dataset validation（E4）** → 所有结论目前限定为 MVTec-5-specific。
3. **robustness–preservation trade-off 真实存在**：X6c 的 robustness 增益**显著小于** B2（Δmean ΔR +0.0318，CI [0.020,0.044]）；
   X6c 只保留 B2 鲁棒性增益的 **61%**。
4. **X6c 不是 free lunch**：它用少量 robustness 换尾部安全（catastrophic 16% → 0%、worst Δd′ 3.5× 改善）。
5. **tail 事件数有限**：B2 侧 catastrophic 8/50 → 16% 的估计 CI 较宽（约 6–27%）。
6. **X6c 复杂度略高于 C2/C6**（两规则平均），其正当性依赖 tail 指标；若下游只看均值指标，C2 更简洁（tail −0.182，仍远优于 B2 的 −0.494）。
7. **X6c 源自经验组合**（Exp15 X6 家族），非第一原理推导，可解释性弱于 C2。

---

## 9. NEXT REQUIRED STEP

> # `E4 — Frozen Held-out External Validation`
> **这不是「继续找新方法」**，而是把**已冻结的 X6c**（连同 Original / Adaptive B2 / C2 作为对照）
> 放到**从未参与方法开发的外部数据集**上验证。**禁止新增候选、禁止调权、禁止改阈值。**
>
> 计划文件：**`HELD_OUT_VALIDATION_PLAN.md`**（推荐数据集 **MPDD**：其设计变量即光照/油污变化，规模小、anomalib 原生支持；
> 备选 VisA）。**该文件仅为计划，未下载、未运行**（下载需人工批准 + 许可确认）。

---

## 10. E4 FAILURE POLICY（**提前记录**；本次审计新建）

如果 E4 **失败**（X6c 未能在外部数据集上保持 catastrophic/worst-tail 优势）：

1. **禁止立即重新调 X6c**（禁止改权重、加新融合、改阈值、删 seed/类别）。
2. **必须先判定失败类别**（并逐项给出证据）：

| 失败类型 | 判定证据 | 正确处置 |
|---|---|---|
| **dataset / domain mismatch** | E4 数据的 anomaly 类型/背景与 MVTec 差异极大（如逻辑异常） | 结论限定为「MVTec-like 表面缺陷域」；**不修改方法** |
| **illumination relevance 不足** | E4 数据集无照明扰动设计 / 光照变化不构成主要变量 | E4 证据**无效**（不能证伪也不能证实）→ 重选数据集，而非调方法 |
| **method generalization failure** | 数据/协议兼容、照明相关，但 X6c 的 tail 优势消失 | **这是有效的负面结论**：把方法结论降级为 MVTec-specific，并在论文中报告 |
| **metric / protocol incompatibility** | 阈值/指标/EPS 带无法在本数据上合法应用（如缺陷类型无 mask） | 先修协议（冻结新协议并留 sha256），**再**比较；不得临时改阈值 |
| **implementation issue** | seed/RNG/runner/spec 错误、复现失败、sanity FAIL | 修 bug 后**原样重跑**；不得借机改方法 |

3. **只有出现新的、充分的科学证据**（例如：另一条独立机制路线在 ≥5 类 × ≥3 seeds 上通过冻结判据）**才允许重新开放 method search**。
4. 任何情况下：**不得改写既有 negative result、不得删除不利 seed、不得把解释写成事实。**

---

## 11. DO NOT DO

- ❌ 不继续 X7 / X8 / X9 方法海选（method search 已 **CLOSED**）
- ❌ 不重新调 X6c 权重（`0.35/0.65`、`0.5/0.5` 已冻结）
- ❌ 不为了 E4 结果修改 threshold（0.02 / 0.10 / −0.25 已冻结）
- ❌ 不删除不利 seed 或类别（含 catastrophic 单元）
- ❌ 不覆盖/软化 negative result（Family A/B、X1/X2b/X3/X4/X5/C1、12、13-P、5C/5D/6A/6B 等）
- ❌ 不把「解释」写成「事实」（README 已有 Fact / Interpretation / Hypothesis 分离约定，见根 README §23）
- ❌ 不重跑已有 protocol-compatible 历史结果；不覆盖 Exp14/15/16 原始 artifact
- ❌ 不处理/破坏已知的 Git 分叉（local 与 `origin/main` 已分叉，含无关 LMS 提交；**禁止 push / merge / rebase / reset**，需人工决策）
- ❌ 不临时下载数据集（E4 下载需人工批准）

---

## 12. NEXT RECOVERY INSTRUCTION

> 新 AI / 新会话接手本项目时，请首先阅读：
> 1. **`PROJECT_STATE.md`**（本文件）
> 2. **根 `README.md`**（实验总账：§0 快照 / §1 论文路线 / §2 证据链 / §3 实验总索引 / §4+ 各实验详节）
> 3. **`PAPER_EVIDENCE_MAP.md`**（论文叙事 × 实验/图表证据对照 + 负结果与协议可信度清单）
> 4. **当前阶段对应的 Experiment README**（当前 = `results/experiment_16_final_validation/README.md`；
>    配套 `PRE_RUN_PROTOCOL.md`、`EXPERIMENT_DEBT_AUDIT.md`、`docs/RESEARCH_RECOVERY_AUDIT.md`）
>
> **在读取上述文件之前，不应设计新的实验。**

---

## 13. E4 / 主数据集状态（2026-10-08 更新，E4-D0）

```text
Primary E4 dataset = M²AD
Dataset selection  = CLOSED
Integrity audit    = PASS   (E4-D0)
Original GPU smoke = PASS   (E4-D1, pipeline feasibility only)
Candidate kill test= DONE   (E4-X: B2 STOP, X6c STOP)
E5-1 broad screening= DONE  (0 GO / 1 HOLD / 3 STOP -> protocol §17 case C)
E5-2A branch read  = CASE C (0 GO) -> E5-2A NOT EXECUTABLE (no survivor)
E5-FAILURE-AUDIT   = DONE   (CPU-only, GPU idle)
Module composition = STOPPED (this branch)
E6-A metric audit  = DONE   (CPU-only): R_all DEMOTED to SECONDARY; TWO-AXIS PARETO recommended;
                     C01 re-evaluated as SUPPORTS FOLLOW-UP (with sign-disagreement caveat)
Next               = human review required (no GPU work authorised, no rescue, no C10)
```

### E6-A 指标有效性审计结果（`docs/E6_A_METRIC_PROTOCOL.md` / `results/e6_a/`，2026-10-09，**CPU-only，GPU = 0**）

协议冻结于 commit `77e81ee`（**先于任何指标结果与解盲**）；盲映射 `BLIND_SEED=20261009`；bootstrap 1000（`BOOTSTRAP_SEED=20261009`）。Runtime **221 s**。

- **验收 9/9 PASS**：冻结值全部机械复现（Original `R_all` 0.675427 / C01 0.696170 / C01 `R_ratio` 1.030711 / Original AUROC 0.767660 / d′ 1.206757 / `M3_good` 0.462616 / C01 0.319159 / `M3_ng` 0.112001 / C01 0.098092）。
- **锦标赛**：**KEEP = `M4_ng`(92.0) / `M5_A`(90.25) / `M5_C`(90.25)**；SECONDARY 8（**含 `M1 R_all` 86.50**）；REJECT = `M7_std`(T6 fatal) / `M7_raw` / `M2_raw_var`(scale-sensitive)。
- **发现 1 — 冻结检测指标有并列 bug**：`image_auroc` 不对并列取平均秩 ⇒ 全常量分数得 **AUROC 1.0**（校正 0.5）。**真实 7 方法上 max|diff| = 0.000e+00 ⇒ 历史数值全部成立**；但 `AUROC≥0.5` **单独不足以**挡住 constant collapse。
- **发现 2 — 指标符号分歧**：`M3_*`/`M5_A`/`M5_C` 说 C01 **更鲁棒**（−31%/−37.6%/−7.7%/−5.4%）；`M1_R_all`(+3.1%)、`M7_std`(+3.1%)、`M4_good`(+15.8%)、`M4_ng`(+1.6%) 说**更差**。`M7` 两种归一化**互相矛盾**且都被 REJECT。
- **发现 3 — 冗余**：`M5_A`≡`M5_C`（ρ=+1.0000）、`M3_ng`≡`M3_all`、`R_ratio`≡`M7_std` ⇒ 保留 **`M5_A`**。
- **推荐协议 = TWO-AXIS PARETO**：Axis X = AUROC/d′（门用**并列校正 AUROC** + 保留 `Δd′ ≥ −0.10`）；Axis Y = **`M5_A`**（伴轴 `M3_good`，反向对照 `M4_ng`）。**禁止加权综合分**。**`R_all` 降为 SECONDARY，不再作首要鲁棒性指标**。
- **C01 = `SUPPORTS FOLLOW-UP`**（§11 冻结规则全满足），**附条件**：须先解释 `M4_ng`/`M1_R_all` 的反号，不得当作无保留胜利。
- **B2/X6c 历史 STOP 不变**；**C06/C07/C08 仍 STOP**（均未过检测门）。
- **限定**：1 类别 × 1 view × 1 seed × 7 方法 × 1 backbone，且 7 方法中 3 个为坍缩检测器 ⇒ 指标排序部分由"如何处理坍缩"决定。**未启动任何 GPU 工作**。

### E5-FAILURE-AUDIT 结果（`docs/E5_FAILURE_AUDIT.md`，2026-10-09，**CPU-only，GPU = 0**）

由 E5-2A 协议 §2 **CASE C** 机械触发（`SURVIVORS (GO) = []`；安全门 6/6 PASS；GPU 全程空闲 1 MiB / 0%）。

**失败机理已定位（不只是记录）**：

| 候选 | 插入点 | 判定 | ΔAUROC | d′ | R_ratio | σ 比 | ρ vs Original | NG 光照方差占比 |
|---|---|---|---|---|---|---|---|---|
| **Original** | 参考 | REFERENCE | 0 | 1.20676 | 1.0000 | 1.000 | +1.000 | **0.1120** |
| C01 | input | HOLD | +0.01277 | 1.30552 | 1.0307 | 0.947 | **+0.904** | 0.0981 |
| C06 | post-concat | STOP | −0.27946 | −0.05534 | 0.7529 | 0.031 | **−0.343** | **0.3649** |
| C07 | post-concat | STOP | −0.21211 | 0.14363 | 0.9962 | 0.322 | +0.480 | **0.4000** |
| C08 | memory | STOP | −0.30569 | −0.11035 | 0.4652 | 4.422 | **−0.402** | 0.1745 |

- **Q1**：**C06/C08 反向了异常排序**（ρ −0.343 / −0.402）；C06 分数尺度坍缩 0.031×，C08 反而放大 **4.422×** ⇒ **C08 低 `R_ratio` 不是"压扁分数"**（更正 README 条目 13 的简写），而是"对光照与缺陷都不响应"的表示。C07 保住一半排序但 d′ 崩 −88%。C01 与 Original ρ=+0.904 ⇒ 基本是同一探测器。
- **Q2**：C06/C07/C08 逐光照 ΔAUROC **10/10 全负** ⇒ **全局失败**，非光照驱动；且三者把 NG 光照依赖放大到 **1.6–3.6×**。
- **Q3**：三类插入点**并非全灭**——input HOLD 未失败，post-concat 2/2 失败，memory 失败。**真正分界是"是否需要拟合"：三个需拟合的模块全败，唯一免训练模块存活。**
- **Q4**：在冻结包络内，**已拟合的特征/记忆级模块无一改善 trade-off**（负结果）；但**不等于** module insertion 普遍不可行（C01 即反例）。
- **Q5**：下一步优先级 = ① input-level / 免训练方向（先解释 C01 指标符号矛盾）→ ② 先重验鲁棒性指标 → ③ 仅允许免训练或不能被"忽略输入"满足的目标 → ④ C06/C07/C08 及调优变体明确降级禁止重跑。
- **新指标风险（本轮最重要的产出）**：C01 的**原始尺度**光照方差占比 0.463 → **0.319（−31%）**，而冻结首要指标 `R_all` 却报 **+3.1% 恶化** —— **符号相反**。且 `R_all` 可被"对光照与缺陷都不响应"的表示刷低（C08）。
  ⇒ **`R_all` 不得单独作为鲁棒性目标**，必须与 Δd′/ΔAUROC 联读，且需先解决 C01 的符号矛盾。
- **限定**：1 类别 × 1 view × 1 seed × 1 backbone × 4 候选；**负结果**，非"不可能"证明。
- **E5-2A 状态**：**未执行**（无 survivor）；**未创建** `docs/E5_2A_FROZEN_PROTOCOL.md`（为空 survivor 冻 view 属空流程）；接力调度器 `scripts/run_e5_2a_after_e5_1.sh` 已就绪且 **CASE C 下拒绝启动 GPU**（默认 dry-run）。

### E5-1P + E5-1 Broad Mini Screening 结果（`docs/E5_1_FROZEN_PROTOCOL.md` / `results/e5_1/`，2026-10-09）

协议先冻结于 commit `e96cd5a`（`sha256 32159221…6035d`，**先于任何候选结果**）。**`0 GO / 1 HOLD / 3 STOP`**。

**E5-1P 关键发现**：E4-D0 时 `github.com` 在本机不可达，本次复测**已可达** ⇒ 三个官方仓库实克隆
（`piad_baseline@67b816b`、`SimpleNet@351a2b8`、`CRAD@b5a1c472`，置于已 gitignore 的 `third_party/`）。
**C01 provenance 零发明解决**：PIAD 自身路径 `Retinex.RL_separate.Separate` → 反射率图 R；`Retinex/LICENSE = MIT © 2022 AndersonYong`。

| 候选 | 插入点 | AUROC | ΔAUROC | d′ | Δd′ | R_all | R_ratio | 判定 |
|---|---|---|---|---|---|---|---|---|
| Original（复用 E4-X，0 GPU） | — | 0.76766 | — | 1.20676 | — | 0.67543 | 1.0000 | REFERENCE |
| **C01 PIAD-Retinex** | input | **0.78043** | **+0.01277** | **1.30552** | **+0.09876** | 0.69617 | 1.0307 | **HOLD** |
| C06 SimpleNet | post-concat | 0.48820 | −0.27946 | −0.05534 | −1.26210 | 0.50851 | 0.7529 | **STOP** |
| C07 ReConPatch | post-concat | 0.55555 | −0.21211 | 0.14363 | −1.06313 | 0.67286 | 0.9962 | **STOP** |
| C08 CRAD | memory | 0.46197 | −0.30569 | −0.11035 | −1.31711 | 0.31423 | 0.4652 | **STOP** |

**方法论发现（须写入 limitation）**：**`R_all` 可被"压扁分数"平凡最小化** —— C08 `R_ratio=0.465` 看似最鲁棒，
实则 AUROC **0.462（低于随机）**、d′ **−0.110（负）**；C06 判别器 loss 收敛到恰好 **1.0000**（`2·θ` margin 下界）。
**三个表示级插入点全部同时降低 AUROC 与 d′**；仅 input-photometric 未失败。`R_ratio` 必须与 `Δd′`/`ΔAUROC` 联读。

**Cost**：总 GPU wall **7510 s ≈ 2 h 05 min**（C01 776 s / C08 5569 s / C07 640 s / C06 524 s）。
**Leakage = NO**；**协议偏离 5 项（D1–D5）全部先于读取结果并已 logged**（无方法级修改）。

**结论**：`Module composition branch STOPPED. No rescue for C06/C07/C08. No automatic E5-2. No C10 top-up.`

### E5-0B 结果（追加于 `docs/MODULE_CANDIDATE_REGISTRY.md` §B1–B13，2026-10-09，CPU-only，GPU = 0）

对 shortlist 做一级来源核验 + 实现审计。**E5-0 的两处实质误判被推翻并保留 negative result**：

- **C05（原 76 分/第 3 名）→ HOLD**：实为 **OCR-GAN 重建框架**（FD 作用于**输入图像**、CS 作用于**多 encoder 之间**）⇒ **NOT_DIRECTLY_PORTABLE**。教训：**评分不能替代可移植性核验**。
- **C04（原 78 分/第 2 名）→ HOLD**：**机制审判通过**（REB = DefectMaker 合成缺陷+自监督域适应 ＋ **LDKNN 局部密度 KNN**；**不使用** mean/std/channel 归一化或 IN/LN/BN ⇒ **非 closed-family**；venue = **Knowledge-Based Systems 290(C) 2024**），但**官方代码 NOT FOUND** 且机制含训练式流水线 ⇒ 实现来源不足。
- 核实升级：C07 = **WACV 2024**（代码为 **Unofficial**，疑无官方仓库）；C08 = **ECCV 2024**（arXiv 2402.18293，官方仓库 `tae-mo/CRAD`，**可学习特征网格+双线性插值取代离散 bank**，O(1)/无 NN 搜索，normal-only）；C06 官方仓库 `DonaldRR/SimpleNet`。

**E5-1 FROZEN CANDIDATES = 4**：

| # | Module | Insertion point | Runtime/unit | Role |
|---|---|---|---|---|
| 1 | C01 PIAD-Retinex | **input** | ≈18 min | NOVELTY-BEARING（input-photometric） |
| 2 | C06 SimpleNet | post-concat | ≈17 min | **SUPPORTING_COMPONENT（不承担 novelty）** |
| 3 | C07 ReConPatch | post-concat | ≈15 min | NOVELTY-BEARING（弱）；须披露 from-paper 实现 |
| 4 | C08 CRAD | **memory bank**（从未探索的插入点） | ≈40 min | NOVELTY-BEARING（强）；须披露适配到 wide_resnet50_2 |

**HOLD**：C04、C05、C02（**HOLD_REDUNDANT**：Filter ≈ Exp13-P，已证非 defect 特异）、C09（**HOLD_PROTOCOL**）。**STOP：无。**
**VACANCY = 1–2 slots**（不自动补位；C10 是否补位由下一轮决定）。
**Composition**：C01×C06 = **++**；C01×C08 = **++**；C06×C07 = **–（冗余）**。
**E5-1 预算**：GPU units **4（+1 备用）**，runtime **≈1.5–2.5 h**，peak VRAM **≈16–19 GB**，1 worker。

### E5-0 结果（`docs/MODULE_CANDIDATE_REGISTRY.md`，2026-10-09，CPU-only，GPU = 0）

Phase III 首个任务：候选模块文献/资产审计。
初始候选 **17**（C01–C17）→ 审计 **10**（C01–C10）→ **Shortlist 8**（GO 6 + HOLD 2），family 多样性合规。
**VERIFIED 5 / PARTIALLY VERIFIED 12 / UNVERIFIED 0**；除 C01（PIAD-Retinex，MIT 代码）外，其余候选**官方代码 URL 未一级确认** ⇒ E5-1 前必须先做代码补齐子任务。

Shortlist（Score）：**C01 PIAD-Retinex 88**（input-photometric）｜C04 REB 78（bank bias）｜C05 Omni-Frequency 76（frequency）｜C02 FiCo 70（HOLD，与 Exp13-P 同思路）｜C06 SimpleNet 70（**撞车 D**）｜C07 ReConPatch 66｜C08 Continuous Memory 66｜C09 On-The-Fly 64（HOLD，协议耦合）。

**硬约束**：已关闭 family（α-IN 及变体 / residual 重标定 / score 融合 / illumination subspace 投影 / category×channel gating / dual-path 代数）**一律禁止重复**。
**新颖性结论**：可承担新颖性来源的只有 C01/C04/C05/C08（等级 B）；论文新颖性须建立在 **input-photometric × frequency × memory-bank 的组合逻辑**上。
**E5-1 预算**：GPU units **8（+2 备用）**，wall-clock **≈2.5–4.5 h**，peak VRAM **≈15–17 GB**，**1 worker**，Original 复用 E4-X。

### E4-X 结果（`results/e4_x/`，协议 `docs/E4_X_PROTOCOL.md`）

M²AD **Bird × seed 0 × frozen view 120 × illumination 01–10**（700 张评分图）。
**Original 复用 E4-D1（0 GPU）**；B2 = 1 GPU unit（fit 778 s）；X6c = score-level 融合（0 GPU）。总 13.7 min，sanity 12/12 PASS。

| Method | Image AUROC | d′ | R_all ↓ | Decision |
|---|---:|---:|---:|---|
| Original | 0.7677 | 1.2068 | 0.6754 | REFERENCE |
| B2 | 0.7680 | 1.1833 | 0.6900 | **STOP** |
| X6c | 0.7689 | 1.2059 | 0.6786 | **STOP** |

**`B2 STOPPED. X6c STOPPED. No rescue experiment is authorized.`**
限定：单类别 × 单 seed × 单 view（不外推）；bank 1,200 图（显存所迫，非全量 train）；阈值是 screening band（`eps_det=0.03` / `delta_robust=0.05`，机械取自 E4-D1 实测变异）**不是统计结论**；AUPRO 仍为 KNOWN ISSUE（协议禁止本轮修复）。
**E4-X 之后禁止继续围绕 X6c 自动实验。**

### E4-D1 实测（`results/e4_d1_smoke/`，协议 `docs/E4_D1_PROTOCOL.md`）

- 仅 **Original（α=0）**，Bird × seed0；**未运行 B2/X6c**，未改方法与阈值。**Sanity 9/9 PASS**，α=0 等价性 `max|Δ|=0.000e+00`。
- runtime：fit(bank 1,200) **774.2 s** ＋ val 14.5 s ＋ scoring(8,400 图) **541.7 s @15.5 img/s** ＋ analysis 41.8 s = **1,378 s / 23.0 min**。
- 峰值：VRAM **14,497 MB alloc / 18,258 MB reserved**（24 GB 卡）；RAM **34.7 GB**。
- 原始表现：image AUROC **0.7633**、d′ **1.2658**、pixel AUROC **0.9638**；illumination AUROC range **0.0933**、view AUROC range **0.1145**；跨光照 score std：Good 1.948 / NG 1.777。
- **未决问题（E4-A 协议必须先解决）**：① **AUPRO NaN** —— M²AD 有 1,450/6,000 无 mask NG 图，`resize_mask` 对 `(1,1)` 占位 mask 抛 `IndexError`（根因已实验确认）；需显式「无 mask NG」策略。② **bank 规模未定** —— 显存所迫本 smoke 用 1,200 图，`embedding_store` 无界使全量 3,600 图不可行（峰值 ≈45 GB）。③ **coreset 代价对 bank 规模超线性**（小规模测速不可外推）。④ τ_val 口径不适配（τ_val 落在 defect 分布内）。
- **成本外推**：每 (category, seed) ≈65 min ⇒ E4-A Mini（3×3）≈9.7 GPU-h、Full（10×3）≈32.5 GPU-h。

| 项 | 状态 |
|---|---|
| E4 数据集选择 | **CLOSED** — `docs/E4_DATASET_SELECTION_CRITERIA.md`（预注册判据）+ `docs/E4_LL_IAD_M2AD_PROTOCOL_AUDIT.md`（E4-0C，M²AD WIN） |
| M²AD 完整性审计 | **PASS** — `docs/E4_M2AD_INTEGRITY_AUDIT.md`（E4-D0，0 GPU，未运行模型） |
| 本地数据 | `data/m2ad/`（gitignored）：official metadata（`meta_unsupervised.json`）＋ category **Bird**（12,910 文件 / 15 GB，SHA256 = 官方 etag） |
| 已完成实测校验 | 真实树 vs metadata **0 不匹配**；parser `view=[1:4]`/`illumination=[6:8]` **0/119,760 偏差**；illumination **01–10 完整均衡**；每 specimen **12×10=120 配置 100/100 完整**；**same-view 配对 Good/NG 均 1.000**；`DEFECT PAIRING = LEVEL 4`；mask **20/20** 匹配；**官方 split specimen 重叠 = 0** |
| 关键约定 | **`REFERENCE ILLUMINATION = UNDEFINED`** —— 官方未定义哪档是常规照明，**禁止**擅自指定 illumination 01；E4-B 指标不得沿用 Exp16 冻结 ΔR 的语义，须另立预注册 |
| 尚未解决（开工前人工确认） | M²AD 数据集 license（HF tag `apache-2.0`，论文/README 未声明）；`df` 报 used 36 G / avail 15 G 而内容实为 21.6 GB（疑 XFS prjquota 记账滞后，未验证） |
| 成本警示 | Bird 单类 ≈ 21,600 图/unit（≈ MVTec 单类 58×）⇒ E4-A Full 粗估 **15–37 GPU-h**；固定 view 可降至 ~1/12。**正式 runtime 待 E4-D1 实测** |
| 被取代/搁置的候选 | `MPDD` = optional generic external-generalization dataset（见 `HELD_OUT_VALIDATION_PLAN.md` §5）；`MVTec AD 2` = pending/暂不作主 E4；`LL-IAD` = UNKNOWN 暂不使用 |

**补充提醒（仓库结构）**：Exp1C–8B 的正式实验文档位于 **`experiments/<exp>/README.md`**，其 artifacts 位于 `results/<exp>/`；
Exp9A 及以后实验的 README 位于 **`results/<exp>/README.md`**。查找实验文档时两个目录都要看。
