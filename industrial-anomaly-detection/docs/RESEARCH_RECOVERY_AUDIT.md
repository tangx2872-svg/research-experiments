# RESEARCH RECOVERY AUDIT

**目的**：确保**在没有 ChatGPT/Codex 对话上下文**的情况下，一个初次接触本项目的研究者/新 AI，**仅凭 Git 仓库文件**
即可恢复：研究问题、实验历史、当前结论、失败路线、冻结方法与下一步。
**审计时间**：2026-10-08 ｜ **审计 HEAD**：`3615f84` ｜ **审计方式**：只读（未运行 GPU 实验、未修改任何历史数据）
**恢复入口**：[`PROJECT_STATE.md`](../PROJECT_STATE.md) → [`README.md`](../README.md) → [`PAPER_EVIDENCE_MAP.md`](../PAPER_EVIDENCE_MAP.md) → 阶段对应 Experiment README

---

## 0. 审计结论（TL;DR）

| 项 | 结论 |
|---|---|
| README 覆盖 | **35/35 实验项均有正式 README**（0 MISSING；少数维度 PARTIAL，且**全部可由根 README 总账恢复**） |
| 冻结协议可恢复性 | **可**（Exp14/15/16 有 `PRE_RUN_PROTOCOL` / `config/*.json` / sha256；Exp1–13 有 `config.json` + 判据段） |
| 关键数字可追溯性 | **PASS（6/6 抽查）** —— README 声明 ↔ CSV/JSON artifact 数字一致 |
| Negative results 可恢复性 | **可**（每个方法实验 README 均有淘汰/否定段；见 §4 负结果索引） |
| 最大**科学**缺口 | **无 held-out 外部数据集验证（E4）** → 结论限定为 MVTec 5 类 |
| 最大**文档**风险 | ① README 双目录约定（Exp1C–8B 在 `experiments/`，Exp9A+ 在 `results/`）；② 根 README §0 TL;DR 为单行 4032 字符 |
| 推荐动作 | 已建立 `PROJECT_STATE.md`；在根 README 顶部加入恢复入口指针（见 §6） |

---

## 1. P0 — 仅凭仓库回答的 15 个问题

> 每问标注**回答来源**；若仓库无法可靠回答则标 `RECOVERY GAP`。

| # | 问题 | 仓库回答（来源） | 状态 |
|---|---|---|---|
| 1 | 论文研究问题是什么？ | 工业视觉中 **illumination 变化下的 robustness** 与 **defect preservation** 的 trade-off：instance normalization（α-IN）提升光照鲁棒性的同时是否损害缺陷可分性，以及如何在不牺牲 preservation 的前提下获得鲁棒性（根 README §1 论文路线、§2 证据链） | COMPLETE |
| 2 | 最初观察到什么现象？ | Exp1：synthetic illumination × α-IN 存在「robustness↑ / pixel-level cost」的温和 trade-off（苗头）；Exp1B：**defect-specific α 响应**（d′ large 13.63→7.32、small 稳定、contamination 反升） | COMPLETE |
| 3 | Exp1–Exp16 分别解决什么？ | 根 README §3 实验总索引（**33 条**，含 5B-C/1H-S）＋每个实验 README 的 Why/Question | COMPLETE |
| 4 | 哪些实验支持核心假设？ | 根 README §2 证据链（逐格：Problem → defect-dependent response(1C/1D/1E) → confound exclusion(1D) → cross-category(1E) → simple-property failure(1F/8B) → layer-specific(1G/1H/1H-S/1I/1J) → method screening(5A–13-P) → X6c(14/15) → Exp16） | COMPLETE |
| 5 | 哪些假设被否定？ | 见本文件 §4 负结果索引（1F simple attributes、5A-H 固定规则、5B Group C 机制、5C/5D/6A/6B geometry identity、7A-O/Q2 模块晋级、8B defect-illumination separability、9A/9B 无候选晋级、12 Category×Channel、13-P defect entanglement、14 Family A/B、15 X1/X2b/X3/X4/X5/C1） | COMPLETE |
| 6 | 哪些方法已淘汰？ | 同上（每个实验 README 的 Verdict/淘汰段） | COMPLETE |
| 7 | 为什么淘汰？ | 每个实验 README 明确给出机械判据（CASE_X / STOP / 0 ADVANCE / 被支配） | COMPLETE |
| 8 | 当前 frozen candidate？ | **X6c = `mean(z_C2, z_C6)`**（Exp16 `A — FREEZE X6c`） | COMPLETE |
| 9 | X6c 精确定义？ | `results/experiment_16_final_validation/PRE_RUN_PROTOCOL.md` §2 ＋ `scripts/experiment16_fusion.py`（实现）＋ Exp15 `config/candidate_registry.json`（X6 家族定义） | COMPLETE |
| 10 | X6c 为什么胜出？ | Exp16 README §10（CASE A 六条判据逐条核对）＋ §7 seed-block（unseen block D 不崩溃）＋ §8 统计显著性 | COMPLETE |
| 11 | 最强统计证据？ | bootstrap 10000（paired）：**ΔPASS +0.160 [0.020, 0.300]**（vs B2，显著）、**tail-5 均值 +0.306 [0.289,0.331]**（vs B2）、+0.048/+0.052（vs C2/C6）→ `results/experiment_16_final_validation/analysis/statistics.json` | COMPLETE |
| 12 | 最大 limitation？ | **无 held-out 外部数据集验证（E4）**；所有结论仅基于 MVTec 5 类（且为方法开发所用类别）；另：robustness cost 显著、X6c 非 free lunch | COMPLETE |
| 13 | 下一步为什么是 E4？ | `HELD_OUT_VALIDATION_PLAN.md`（推荐 MPDD：数据集设计变量即光照/油污变化；规模小、anomalib 原生支持；**计划文件，未下载**）＋ Exp16 README §12 | COMPLETE |
| 14 | 哪些参数/协议已冻结？ | Exp16 `PRE_RUN_PROTOCOL.md` §1–§2（阈值 EPS_RZ 0.02 / EPS_DP 0.10 / catastrophic −0.25；方法定义；z 校准口径）＋ `config/sha256_frozen.json` | COMPLETE |
| 15 | E4 失败应如何解释？ | **原仓库无此策略文件** → 已在本次审计中于 `PROJECT_STATE.md` §10 建立（**E4 FAILURE POLICY**） | **RECOVERY GAP → 已补齐** |

### 仍然存在的 RECOVERY GAP（诚实记录）

| # | Gap | 说明 | 处置 |
|---|---|---|---|
| G1 | **E4 失败策略** | 原仓库只写「下一步是 E4」，未定义失败后的解释纪律 | 本次在 `PROJECT_STATE.md` §10 补齐（禁止立即调 X6c；先分类失败原因） |
| G2 | **Q2/Q3/Q4 未进 §3 总索引** | 三个 overnight 子队列只在 §10.1 与各自 README 记录 | 保留现状（§10.1 可导航）；`PROJECT_STATE.md` §7 提及 |
| G3 | **README 双目录约定无说明** | Exp1C–8B 的正式 README 在 `experiments/<exp>/README.md`，其 artifacts 在 `results/<exp>/`；`results/<exp>/` 下**没有** README（早期实验） | 本次在根 README 顶部与本文 §6 显式说明 |
| G4 | **§0 TL;DR 单行 4032 字符** | 可读性差，新人难以解析 | 本次在根 README 顶部加入精简恢复入口；**不改写历史行** |
| G5 | **Exp1B 的「四句话」预注册 vs 通用叙事** | 已存在于 `results/experiment_1b/README.md`（§预注册四句话） | 无缺口（审计初判为误报，已纠正） |
| G6 | 少量实验缺显式「Next Step」段（如 1B、Q4 有、15 无） | 后续实验 README 的开头「前序」段可恢复 | 标 PARTIAL（可由后继实验恢复） |

---

## 2. P1 — README Coverage Audit（Exp1–Exp16）

**图例**：`COMPLETE` = 该维度在仓库中可直接读到；`PARTIAL` = 需从根 README/后继实验/artifact 推导；
`MISSING` = 不可恢复；`N/A` = 该实验结构上不适用。
**维度**：A=Why｜B=Question｜C=Frozen Protocol｜D=Results（含数字/路径）｜E=Negative Results｜F=Interpretation｜G=Paper Role｜H=Exit Decision｜I=Next Step

| Experiment | README（路径） | A | B | C | D | E | F | G | H | I | Recoverable |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1B | `results/experiment_1b/README.md` | ✅ | ✅ | ✅ | ✅ | N/A | ✅ | ◐ | ✅ | ◐ | YES |
| 1C | `experiments/exp1c_multiseed/README.md` | ✅ | ✅ | ✅ | ✅ | N/A | ✅ | ◐ | ✅ | ✅ | YES |
| 1D | `experiments/exp1d_size_control/README.md` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ◐ | ✅ | ◐ | YES |
| 1E | `experiments/exp1e_cross_category/README.md` | ✅ | ✅ | ✅ | ✅ | N/A | ✅ | ◐ | ✅ | ✅ | YES |
| 1F | `experiments/exp1f_mechanism_screening/README.md` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ◐ | ✅ | ✅ | YES |
| 1G | `experiments/exp1g_feature_space/README.md` | ✅ | ✅ | ✅ | ✅ | ◐ | ✅ | ◐ | ✅ | ✅ | YES |
| 1H | `experiments/exp1h_layer_selectivity/README.md` | ✅ | ✅ | ✅ | ✅ | N/A | ✅ | ◐ | ◐ | ✅ | YES |
| 1H-S | `experiments/exp1hs_dispersion_layer/README.md` | ✅ | ✅ | ✅ | ✅ | N/A | ✅ | ◐ | ✅ | ◐ | YES |
| 1I | `experiments/experiment1i_spatial_statistics_control/README.md` | ◐ | ✅ | ✅ | ✅ | N/A | ✅ | ◐ | ✅ | ✅ | YES |
| 1J-A | `experiments/experiment1j_feature_geometry/README.md` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ◐ | ✅ | ✅ | YES |
| 1J-B | `experiments/experiment1j_b_transmission/README.md` | ✅ | ✅ | ✅ | ✅ | N/A | ✅ | ◐ | ✅ | ◐ | YES |
| 5A | `experiments/experiment5a/README.md` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ◐ | ✅ | ✅ | YES |
| 5A-H | `experiments/experiment5a_h/README.md` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ◐ | ✅ | ✅ | YES |
| 5B / 5B-C | `experiments/experiment5b/README.md` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ◐ | ✅ | ✅ | YES |
| 5C | `experiments/experiment5c/README.md` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ◐ | ✅ | ✅ | YES |
| 5D | `experiments/experiment5d/README.md` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ◐ | ✅ | ✅ | YES |
| 6A | `experiments/experiment6a/README.md` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ◐ | ✅ | ✅ | YES |
| 6B | `experiments/experiment6b/README.md` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ◐ | ✅ | ✅ | YES |
| 7A-O | `experiments/experiment7a_o/README.md` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | YES |
| 7A-O-Q2 | `results/experiment_7a_o_q2/README.md` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | YES |
| 7A-O-Q3 | `results/experiment_7a_o_q3/README.md` | ◐ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | YES |
| 7A-O-Q4 | `results/experiment_7a_o_q4/README.md` | ✅ | ✅ | ✅ | ✅ | ◐ | ✅ | ✅ | ✅ | ✅ | YES |
| 8B | `experiments/experiment8b/README.md` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | YES |
| 9A | `results/experiment_9a_screening/README.md` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | YES |
| 9B | `results/experiment_9b_screening/README.md` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | YES |
| 9C | `results/experiment_9c_rng_calibration/README.md` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | YES |
| 9B-R | `results/experiment_9b_r_strict_replay/README.md` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | YES |
| 10 | `results/experiment_10_preservation_recovery/README.md` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | YES |
| 11 | `results/experiment_11_c2_adaptive/README.md` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | YES |
| 12 | `results/experiment_12/README.md` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | YES |
| 13-P | `results/experiment_13p/README.md` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | YES |
| 14 | `results/experiment_14/README.md` | ✅ | ◐ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | YES |
| 15 | `results/experiment_15/README.md` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ◐ | YES |
| 16 | `results/experiment_16_final_validation/README.md` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | YES |

**统计**：**0 MISSING**；`◐`（PARTIAL）共 22 处，**全部可由根 README §2 证据链 / §3 索引 / 后继实验 README / artifact 恢复**。
**说明**：`◐` 集中在两类：① **G=Paper Role**（Exp1C–6B 写在「论文导航」约定之前 → 由根 README §2/§3 恢复）；
② **I=Next Step**（部分实验以**后继实验的开头「前序」段**承载，而非自身结尾）。
**本次未补造任何数字**：凡 README 未写明的历史细节，本审计一律不改写；无法确定者按 P2 规则标记 `UNKNOWN`。

---

## 3. P2 — 不伪造历史（本次审计遵守的边界）

- **允许**：从已有 `README` / `config.json` / `summary/*.csv` / `analysis/*.json` / `figures` / git log 中**读取**确定事实。
- **禁止**：按当前叙事反向推测当时发生了什么；为让论文故事更漂亮而修改 negative result。
- **本次审计中发现的「疑似缺失」全部经二次核验**：Exp1B（README 实际存在于 `results/experiment_1b/`）、
  Exp1C/1D/1E/1G/1H/1J/1J-B/Q3/Q4/15 的「缺 Next/Question」均为**关键词启发式误报**，实际内容存在（已逐个人工核verify）。
- **未发现**任何 README 中的数字与其 artifact 冲突（见 §5 traceability）。
- **未发现**需要标 `UNKNOWN — historical artifact insufficient` 的历史关键事实。
  （若未来发现，请按此格式标注，**不得补造**。）

---

## 4. Negative Results 索引（恢复用，按阶段）

| 阶段 | 被否定/淘汰 | 原因（README 原文要点） | 入口 |
|---|---|---|---|
| 机制 | 1F 简单图像属性 | \|ρ\|≤0.24，无法解释异质性（CASE_D） | `experiments/exp1f_mechanism_screening/README.md` |
| 机制 | 1H-S dispersion 口径 | 仅 partial alignment（CASE_B） | `experiments/exp1hs_dispersion_layer/README.md` |
| 方法设计 | 5A-H 固定层级规则 | CASE C — bottle-specific；excl-bottle preservation 显著变差（p=0.0024）→ STOP | `experiments/experiment5a_h/README.md` |
| 机制解释 | 5B Group C predictors | 8 个 normalization-sensitivity predictor **全部 null** → 机制性解释不成立 | `experiments/experiment5b/README.md` |
| 方法验证 | 5C 激进重分配 | 0/15 pair-win → CASE_B（Harm Reduction Only） | `experiments/experiment5c/README.md` |
| 方法身份 | 5D matched-control | GC 的 preservation 优势由单一类别（grid）驱动（LOCO 去掉 grid 后 −0.0117）→ identity 未确立 | `experiments/experiment5d/README.md` |
| 模块筛选 | 7A-O / Q2 | 无模块晋级（Q2：layer×representation 组合 → STOP） | `experiments/experiment7a_o/README.md`、`results/experiment_7a_o_q2/README.md` |
| 方法选型 | 8B defect–illumination separability | **CASE_B → Plan C = HOLD**（缺陷方向与 illumination 方向不可分性未成立） | `experiments/experiment8b/README.md` |
| 海选一轮 | 9A M1–M6 | 无方法晋级（M3/M4 = STOP；M6 = oracle） | `results/experiment_9a_screening/README.md` |
| 海选二轮 | 9B M7–M11 | 无候选晋级；发现 coreset 噪声地板 0.0202/0.0542 | `results/experiment_9b_screening/README.md` |
| 评测协议 | 9C（V1 问题） | 噪声地板归零（V2），V1 结果冻结保留 | `results/experiment_9c_rng_calibration/README.md` |
| 组合 | 12 Category × Channel | M1/M2/M3 全部触发硬性淘汰 → 0 ADVANCE → **STOP 该组合** | `results/experiment_12/README.md` |
| 新 family | 13-P INSS | subspace 存在但不含 **defect 特异**信息 → **否决 defect-entanglement 叙事** | `results/experiment_13p/README.md` |
| 新 family | 14 Family A（Dual-Path） | **代数等价**于 B2 强度重参数化（bit-exact 已证）→ 结构性无法 Pareto | `results/experiment_14/README.md` |
| 新 family | 14 Family B（Tiny INSS） | 全家族单边 trade-off（λ=0.5 反使 robustness 变差）→ STOP | `results/experiment_14/README.md` |
| 组合 | 15 X1 / X2b / X3 | X1 数学重复、X2b 无增益、X3 前提不满足 | `results/experiment_15/README.md` |
| 组合 | 15 X4a / X4b | R1 0/3、1/3，被 B2 双向支配 → 未晋级 | `results/experiment_15/README.md` |
| 组合 | 15 X5（category-adaptive） | 未超过固定权重（honest negative） | `results/experiment_15/README.md` |
| 候选 | 15 C1（Exp14 规则 winner） | 在 unseen seeds 上退化（12/15→6/10）→ **seed 选择效应** | `results/experiment_15/README.md` |
| 冠军修正 | Adaptive B2 | catastrophic 率 **1/15 → 3/25 → 8/50（16%）** → downside 风险此前被低估 | `results/experiment_16_final_validation/README.md` |

---

## 5. P5 — Artifact Traceability（6/6 抽查 PASS）

| # | 结论 | README 声明 | Artifact 路径 | 核对结果 |
|---|---|---|---|---|
| 1 | Exp16 X6c 50 单元 | 38/50、0 catastrophic、worst −0.1416 | `results/experiment_16_final_validation/analysis/final_50unit_table.csv` | **一致（PASS）** |
| 2 | Exp15 seed-block 敏感性 | X6c 11/15·8/10·8/10；B2 12/15·7/10·**4/10** | `results/experiment_15/analysis/seed_block_stability.csv` | **一致（PASS）** |
| 3 | Exp16 bootstrap | ΔPASS +0.160 [0.020, 0.300]、Δmean ΔR +0.0318 [0.0201, 0.0439] | `results/experiment_16_final_validation/analysis/statistics.json` | **一致（PASS）** |
| 4 | Exp1D size confound | CASE_A、ΔR²=0.188 | `results/experiment_1d/summary/verdict.json`（`delta_R2_type_given_area`=0.18836）＋ `regression_results.json` | **一致（PASS）** |
| 5 | Exp1H 层级选择性 | 家族→偏好层 3/3+3/3+3/3 | `results/experiment_1h/analysis/layer_selectivity.csv`（25 行，含 `preferred_sensitive_layer`） | **一致（PASS）** |
| 6 | Exp1E 跨类别现象 | 25/25 defect types 至少一个维度 3/3 符号一致 | `results/experiment_1e/<cat>/seed_*/group_level.csv`（per defect_type × alpha 统计） | **一致（PASS）** |

> **TRACEABILITY: PASS**

---

## 6. P4 — 根 README 作为「实验总账」的审计

| 检查项 | 结果 |
|---|---|
| 是否能看到 `Experiment → purpose → key finding → verdict → paper progress`？ | **是**：§0 状态快照（TL;DR）＋ §1 论文路线 ＋ §2 证据链（逐格结论+数字+入口）＋ §3 实验总索引（33 条，含编号/问题/关键结果/判定/状态）＋ §4–§22 每实验详节 |
| 是否有实验缺失？ | **否**（§3 覆盖 Exp1–16 全部；仅 Q2/Q3/Q4 三个子队列只在 §10.1 出现，见 G2） |
| 是否过长/难读？ | **部分是**：§0 TL;DR 为**单行 4032 字符**（G4）；§4–§22 详节虽长但属必要记录 |
| 本次动作 | ① 在根 README **顶部**加入 **恢复入口指针块**（不改写任何历史行）；② 显式说明 **README 双目录约定**；③ 其余保持原样（详细信息仍留在 Experiment README） |

---

## 7. 恢复能力总结（新会话最小阅读集）

```
PROJECT_STATE.md                     ← 唯一快速恢复入口（1–3 页）
README.md                            ← 实验总账（§0 快照 / §1 路线 / §2 证据链 / §3 索引 / §4+ 详节）
PAPER_EVIDENCE_MAP.md                ← 论文叙事 × 实验/图表证据对照（含负结果与协议可信度）
results/paper_assets/                ← 论文主表草稿 / ablation inventory / Fig1–5（PNG+PDF）
HELD_OUT_VALIDATION_PLAN.md          ← E4 计划（未执行）
results/experiment_16_final_validation/  ← 冻结方法定义 + 最终验证 + 债务审计
results/experiment_15|14/README.md   ← 赢家来源与已关闭方向
experiments/<exp>/README.md          ← Exp1C–8B 的正式文档（注意：不在 results/ 下）
```

**结论**：**RECOVERY CAPABILITY: SUFFICIENT**（无 MISSING；6 处 RECOVERY GAP 中 5 处已由本次审计补齐或显式记录，
剩余 G6 属可由后继实验恢复的 PARTIAL）。
