# PROJECT_STATE — 科研项目快速恢复入口

> **本文件是唯一「快速恢复入口」**。任何新会话 / 新研究者接手本项目时，**先读本文件**。
> 最后更新：2026-10-08 ｜ HEAD：`3615f84` ｜ 依据：Exp16 冻结验证 + `docs/RESEARCH_RECOVERY_AUDIT.md`
> **所有数字均从正式 artifact 重新读取确认**（路径逐项标注）；未在本文件出现的结论不属于项目正式结论。

---

## 1. CURRENT PAPER STAGE

| 项 | 状态 |
|---|---|
| 阶段 | **Phase II — Paper Validation** |
| Method discovery / search | **CLOSED（已正式结束）** |
| 冻结方法 | **X6c**（`results/experiment_16_final_validation/`，verdict `A — FREEZE X6c`） |
| 内部验证 | **完成**（5 类 × 10 seeds = 50 单元） |
| 下一步 | **E4 — 冻结的 held-out 外部验证**（计划已写，**未执行**） |

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
Integrity audit    = PASS
Next               = E4-D1 Original-only GPU smoke
```

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
