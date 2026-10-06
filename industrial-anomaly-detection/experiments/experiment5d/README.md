# Experiment 5D — Matched-Control Gate Validation

> 状态：**已完成**（CPU-only 历史重组，sanity S1–S18 = 18/18 PASS；5C↔5A-H raw 等价性 `max|Δscore| = 0.0`）。
> 协议冻结于运行前（`results/experiment_5d/reference/policy_freeze.json`,
> SHA256 `30150bc49484990524e88a2d06bfcee85262db8ca3f4ae3c0663c0ad32a5b263`，`target_results_read=false`）。
> **Final verdict：预注册判定链 literal = CASE_D（仅由 LOCO 单类别驱动条件触发）；实质性结论 = CASE_B
> （Geometry Useful but Identity Weak）→ 见 §15，CASE 归属请人工裁决。**
> 前序：5C ✅ **CASE_B — Harm Reduction Only**（commit `fdd6fa9`）→ 本实验 = Stage ⑧
> **Method Identity / Matched-Control Validation**。

---

## 0. Paper Navigation

📍 **Stage ⑧ — Method Identity / Matched-Control Validation**。
① 现实问题确认 ✅ → ② NDIL / defect-specific response ✅ → ③ layer-specific / transmission ✅
→ ④ normal-only geometry predictor ✅ → ⑤ predictor → executable policy ✅
→ ⑥ Geometry Conservative harm reduction（5C ✅） → ⑦ **geometry 是否真的是有效 gate 的原因？← 5D**

### ① 论文要证明的话

> Normal-feature geometry contains actionable information for deciding when normalization should be
> suppressed to preserve defect-sensitive representations.

5D **不是**证明「Geometry Conservative 全面优于 fixed normalization」（该主张在 5C 已被否定）。

### ② 本实验必须做，是因为存在替代解释

5C 的 GC 关闭 bottle/grid 的 normalization。替代解释是：

> 也许随便挑两个 category 给 α=0，也能获得类似收益；5C 的收益来自 **conservative gating 本身**，
> 而不是 **geometry 提供了有效的类别选择信息**。

### ③ 出口条件

- 支持假设并继续：GC 在同 budget 下优于 matched sensitivity control，且优于 exhaustive 分布主体，
  且不是单类别驱动 → 保留方案①（Geometry-Guided Safe Normalization）。
- 假设不成立：GC ≈ 多数 matched gates / GC ≈ sensitivity / 优势由单一 category 驱动 → 停止 geometry 故事。
- 暂停或修改：direction 或 sanity 关键项失败 → 保留现场，不得为跑完而改协议。

---

## 1. Research Question

在**完全相同** normalization budget（gate 2/5、mean α = 0.3、α ∈ {0, 0.5}）、相同模型 / 数据 /
illumination protocol / evaluation 下，**geometry predictor 是否比 matched sensitivity predictor
以及 exhaustive matched-gate 分布更可靠地选出「应当关闭 normalization」的 2 个 category？**

唯一允许变化的量：**谁决定哪两个 category α=0**。

## 2. Paper Stage

Stage ⑧（见 §0）。5D 是**身份验证（证伪）实验**，不是调参实验。

## 3. Hypothesis

H1（primary）：`radius_ratio_L3L2` 的 conservative gate 优于 matched sensitivity conservative gate
（SC-A）的 preservation，且 robustness 不明显更差。

H0（必须被排除的替代解释）：matched sensitivity gate（SC-A）与 GC 无实质差异，或 GC 的优势可被
exhaustive matched-gate 分布中大量 gate 复制 → 5C 的成功只支持 generic conservative gating。

## 4. Frozen Predictors（不得更换、不得重新拟合）

| 角色 | predictor | 来源 | 备注 |
|---|---|---|---|
| primary | `radius_ratio_L3L2` | 5B `normal_only_predictors.csv` | 5B ρ(C2)=−0.90、LOCO 5/5 |
| matched control | `sens_radius_L2` | 同上（5B Group C primary） | 5B ρ(C2)=−0.30、LOCO 2/5 |
| secondary（**仅 assignment identity check，不做 policy 评估**） | `eff_dim_L3` | 同上 | 5B ρ(C2)=−0.90、LOCO 4/5 |

全部 normal-only（只读 `<cat>/train/good`）；5B freeze 记录 `targets_read = false`。
5D **禁止**：重选 predictor、调 threshold、新造 geometry metric、改 gate size、改 α ceiling。

### 4.1 Direction 审计与 protocol ambiguity（重要）

冻结前审计 `sens_radius_L2` 的 conservative direction，发现历史记录**不唯一**：

| 来源 | 原始记录 | 推导 gate（α=0 的 2 类） |
|---|---|---|
| 5B registry `predictor_registry.csv` / `group_c_freeze.json`（`direction_hypothesis`） | `high->fragile` | {grid, screw} |
| 5B README §17.4（同次冻结的实测解释） | `ρ(C2)=−0.30`，与 geometry 族**同号（负）** | {bottle, cable} |
| 5C `geometry_policy_freeze.json` `direction` + 5C README §4（GF/GC/SG 共用同一 rank→α 函数） | 升序 rank → 升序 α（低值 = 最脆弱） | {bottle, cable} |

- 5C freeze 的 `direction` 字段名为 **geometry**；5B/5C 中**没有**任何一句显式声明过
  conservative sensitivity 的方向；registry 的 `high->fragile` 是**预注册假设**，
  与 5B 自己实测的符号相反。
- 按 5D 协议 §5：「如果确实不存在唯一方向 → 标记 protocol ambiguity → STOP 并报告，
  不得偷看结果后决定」。已如实 STOP 并上报。
- **人类裁决（2026-10-06）**：两者都预注册，**SC-A 为 primary comparison**，
  SC-B 作为对称的 ambiguity-resolving control 全量报告；不依据 target 结果选择方向。
- 附带事实：**两个候选 gate 都已在预注册的 10 个 exhaustive gate 之内**，
  因此该歧义只影响「谁是 primary comparison」，不影响数据本身。

## 5. Frozen Policies

| policy | predictor | orientation | α 分配 (bottle, cable, grid, hazelnut, screw) | gate | mean α | 角色 |
|---|---|---|---|---|---|---|
| A. Original | — | — | 0/0/0/0/0 | — | 0.0 | reference（= 5A-H B0） |
| B. Best Fixed | — | — | 0.5/0.5/0.5/0.5/0.5 | — | 0.5 | 冻结 baseline（= 5A-H B2，不重选） |
| **GC** | `radius_ratio_L3L2` | low→fragile | 0.0 / 0.5 / 0.0 / 0.5 / 0.5 | bottle+grid | 0.3 | **PRIMARY METHOD** |
| **SC-A** | `sens_radius_L2` | low→fragile（5C mechanical） | 0.0 / 0.0 / 0.5 / 0.5 / 0.5 | bottle+cable | 0.3 | **PRIMARY MATCHED CONTROL** |
| SC-B | `sens_radius_L2` | high→fragile（5B registry） | 0.5 / 0.5 / 0.0 / 0.5 / 0.0 | grid+screw | 0.3 | secondary ambiguity control |

- GC 与 5C 冻结 assignment **逐类一致**（脚本内 hard assertion，`True`）。
- `eff_dim_L3` conservative assignment 与 GC 相同（bottle+grid）→ 仅记录为 identity remark，
  **不做 policy 评估**（与 5C 对 SEC 的处理一致）。

### 5.1 Exhaustive Matched Gates（EVALUATION-ONLY）

```text
{bottle,cable}(=SC-A) {bottle,grid}(=GC) {bottle,hazelnut} {bottle,screw} {cable,grid}
{cable,hazelnut} {cable,screw} {grid,hazelnut} {grid,screw}(=SC-B) {hazelnut,screw}
```

- 被选中的 2 类 α=0，其余 3 类 α=0.5；**每个 gate mean α = 0.3、gate size 2/5**。
- **EXHAUSTIVE CONTROLS ARE EVALUATION-ONLY.**
  **NO BEST RANDOM GATE MAY REPLACE THE FROZEN GEOMETRY POLICY.**
- 不做随机抽样（C(5,2)=10 很小，直接枚举更干净）。

## 6. Gate Size

`GATE_SIZE = 2/5`（对所有 adaptive / matched policy 相同）。禁止改成 1/3 类。

## 7. α Budget

`α ∈ {0, 0.5}`（5D 唯一允许的 α 集合）；`mean α = (2×0.0 + 3×0.5)/5 = 0.3`。
不新增 α（无 0.25 / 0.6 / 0.8），不改 ceiling。

## 8. Metrics（严格沿用 5A-H / 5C 冻结口径，禁新增对 geometry 有利的主指标）

| 类型 | 名称 | 方向 |
|---|---|---|
| primary preservation | `mean defect d′`（5 cat × 3 seed 平均） | ↑ higher better |
| primary robustness | `mean |ΔNormalScore_z|`（4 photometric shifts） | ↓ lower better |
| 派生 | `Δd′` / `Δ|Δz|` vs best fixed（α=0.5） | — |
| harm | `worst_category_gain` = min over 5 categories of gain vs best fixed | ↑ better |
| harm | `negative_transfer_count` = #{category : gain < −ε} | ↓ better |
| harm | `high_damage_recovery` = mean gain over 预指定 subset {bottle, grid} | ↑ better |
| harm | `pair_win_units` = 满足 5A-H 冻结 PAIR-WIN（ΔRobust<0 AND ΔPres≥−ε）的 unit 数 /15 | ↑ better |
| 辅助 | per-category / per-seed mean ± std、image/pixel AUROC、AUPRO、FPR | 仅解释 |

`ε = 0.10`（5A-H §2 冻结的 d′ 非劣带，**未新增**）。
high-damage subset `{bottle, grid}` 由 5A-H/5B 历史 damage 排序**运行前预指定**（不看 5D 结果）。

## 9. Sanity（S1–S18；任何关键项 FAIL → STOP）

```text
S1  P0：branch=main、HEAD 含 fdd6fa9、working tree clean、local==origin/main
S2  primary predictor = radius_ratio_L3L2（与 5C 冻结一致，未改变）
S3  sensitivity predictor = sens_radius_L2（与 5C 冻结一致，未改变）
S4  所有 predictor 来自 normal train only（5B normal-only 资产；无 test 通路）
S5  freeze JSON target_results_read = false（freeze 早于任何 target 分析）
S6  geometry assignment 与 5C GC 完全一致（逐类）
S7  所有 matched policy gate size = 2/5
S8  所有 matched policy mean α = 0.3
S9  所有 assigned α ∈ {0, 0.5}
S10 fixed baseline = α0.5（5A-H B2 复用，未重新选择）
S11 ε = 0.10 未改变（与 5A-H / 5C 一致）
S12 categories = 5（bottle/cable/grid/hazelnut/screw，未变）
S13 seeds = 3（0/1/2，未变）
S14 illumination protocol 未变（brightness 0.7/1.3 + gamma 0.7/1.3 共 4 shifts）
S15 历史 raw score 等价性 PASS（5C vs 5A-H 同 α：max|Δscore| ≤ 1e-9、max|Δtau| ≤ 1e-9）
S16 10/10 exhaustive gates 完整（每个 2 类 α=0、mean α 0.3）
S17 所有输出指标 finite（无 NaN/Inf）
S18 analysis 前后冻结资产 hash 不变（reference/asset_hashes.json 复核）
```

## 10. Verdict Rules（PRE-REGISTERED；禁止看完结果后临时发明 CASE）

记号：`d′(P)` / `R(P)` = policy P 的 5cat×3seed 平均 defect d′ / mean |Δz|；
`B` = best fixed（α=0.5）；`Δd′(P)=d′(P)−d′(B)`、`ΔR(P)=R(P)−R(B)`；
`worst_gain(P)`、`neg_transfer(P)`、`pareto(X)` = 在 (R 越小越好, d′ 越大越好) 平面上不被支配（容差 1e-9）；
`rank_dprime(P)` = 在 10 个 exhaustive gates 内 1 + #{g : d′(g) > d′(P)}；
`pair_win(X,Y)` 逐 unit = `R_unit(X) − R_unit(Y) < 0` 且 `d′_unit(X) − d′_unit(Y) ≥ −ε`。

**判定顺序（首个命中即判定；primary control = SC-A）**

1. **CASE_D — Geometry Fails Matched Control**，若任一成立：
   - `Δd′(GC) − Δd′(SC-A) ≤ −ε`（GC preservation 明显更差）；
   - SC-A 在 (R, d′) 平面支配 GC；
   - `rank_dprime(GC) ≥ 8`（处于 10 个 gate 的较差区域）；
   - **单类别驱动**：存在某 category，去掉它后在剩余 4 类上 `Δd′(GC) − Δd′(SC-A) ≤ 0`。
   → Geometry predictor 不提供足够的可用策略信息；方案①停止，进入方案②。

2. **CASE_A — Geometry Identity Supported**，若**全部**成立：
   - `Δd′(GC) − Δd′(SC-A) ≥ +ε`；
   - `R(GC) − R(SC-A) ≤ +0.02`（robustness 不明显更差）；
   - `worst_gain(GC) ≥ worst_gain(SC-A)`；
   - `neg_transfer(GC) ≤ neg_transfer(SC-A)`；
   - seed-level：per-seed 5-cat 聚合上 GC 对 SC-A 满足 pair-win 的 seed 数 ≥ 2/3；
   - `rank_dprime(GC) ≤ 3` **且** GC 在 {10 gates, A, B, SC-A, SC-B} 中 Pareto-efficient；
   - LOCO（不重新拟合，只做 drop-one-category 聚合）：5 个 held-out 变体上
     `Δd′(GC) − Δd′(SC-A) > 0` 全部成立。
   → 只允许表述为 `supported`，**不得**写 `proved` / `causal` / `universally optimal`。

3. **CASE_C — Generic Conservative Gating**，若**全部**成立：
   - `|d′(GC) − d′(SC-A)| < ε`（与 sensitivity control 不可区分）；
   - `#{gates g : d′(g) ≥ d′(GC)} ≥ 4`（至少 4 个 matched gate 不劣于 GC）。
   → 5C 的收益主要来自 conservative gating 本身；**停止 Geometry-Guided 方法叙事**，不再调 predictor。

4. **CASE_B — Geometry Useful but Identity Weak**：以上均不成立时。
   → geometry-guided gating 仍可用，但**不能**强调 geometry 独特性（identity 未确立）。

同时报告以 SC-B 为 control 的同一判定链（`verdict_with_SC_B`，informational，不替代 primary）。

## 11. Stop Rules

任何关键 sanity（S1–S18）FAIL、出现 NaN/Inf、asset hash 变化、或 gate/budget 与冻结不符 →
**STOP**，保留现场，不得为了跑完而修改协议或科研定义。5D 结束后
**STOP，不启动 5E，不实现 v2/v3，不开始方案②**，等待人工审阅。

## 12. Expected Outputs

```text
results/experiment_5d/reference/  policy_freeze.json  policy_freeze.sha256
                                  geometry_assignment.csv  sensitivity_assignment.csv
                                  exhaustive_gate_assignments.csv  asset_hashes.json
results/experiment_5d/summary/    policy_summary.csv  per_category.csv  per_seed.csv
                                  gate_identity.csv  exhaustive_gates.csv
                                  pareto_summary.csv  loco_summary.csv  verdict.json
results/experiment_5d/sanity/     sanity_checks.csv
results/experiment_5d/figures/    gate_pareto.png  gate_exhaustive_rank.png
                                  geometry_vs_sensitivity.png  per_category_harm.png
                                  seed_stability.png
results/experiment_5d/logs/       policy_freeze.log  analysis_run.log
```

## 13. Execution Plan（CPU-only）

5C 已证明 5C 与 5A-H 在同一 (category, seed, α) 上 raw per-image 分数逐位一致
（33 keys / 9273 行 / `max|Δscore| = 0.0`）。5D 的所有 policy 只用 α ∈ {0, 0.5}，
而 5A-H 的 B0/B2 已完整覆盖 5 cat × 3 seed × {0, 0.5}（30/30 key）。

因此 5D **不跑 GPU**，只做：`freeze → recombine（历史 raw） → aggregate → sanity → analysis → figures → verdict`。
仅在发现 raw score 缺失 / hash 不一致 / key 不完整时才允许补跑缺失部分。

---

## 14. Results（运行后）

### 14.1 执行与等价性

- **零 GPU**：5D 不跑 inference，只重组 5A-H 已冻结的 raw per-image 分数（B0=α0 / B2=α0.5，
  30/30 condition key 齐备）。git HEAD = `1a0245e`，branch = `main`。
- **历史 raw score 等价性（S15）**：5C vs 5A-H 同 α（33 keys / 9273 行）
  `max|Δscore| = 0.000e+00`、`max|Δτ_val| = 0.000e+00`。
- **sanity S1–S18 = 18/18 PASS**（含 S18：9 个历史冻结资产 md5 分析前后不变）。
- freeze SHA256 = `30150bc49484990524e88a2d06bfcee85262db8ca3f4ae3c0663c0ad32a5b263`。

### 14.2 Policy summary（5 categories × 3 seeds 平均）

| policy | gate | mean α | defect d′ ↑ | \|Δz\| ↓ | Δd′ vs fixed | Δ\|Δz\| vs fixed | worst-cat gain | neg transfer | HD recovery |
|---|---|---|---|---|---|---|---|---|---|
| A. Original | — | 0.00 | 4.9852 | 0.3105 | +0.3183 | +0.1023 | −0.2107 | 1 | +0.8243 |
| B. Best Fixed | — | 0.50 | 4.6670 | 0.2082 | 0（参照） | 0（参照） | 0.0000 | 0 | 0 |
| **GC（primary method）** | bottle+grid | 0.30 | **4.9967** | 0.2643 | **+0.3297** | +0.0560 | **0.0000** | **0** | **+0.8243** |
| **SC-A（primary control）** | bottle+cable | 0.30 | 4.8517 | 0.2514 | +0.1847 | +0.0432 | 0.0000 | 0 | +0.4384 |
| SC-B（secondary control） | grid+screw | 0.30 | 4.8427 | 0.2547 | +0.1757 | +0.0465 | 0.0000 | 0 | +0.3859 |

逐类别 gain vs best fixed：bottle **+0.8768**、grid **+0.7718**、screw +0.1067、cable +0.0467、
hazelnut **−0.2107**（即 α=0 对 hazelnut 有害）。逐类别 robustness 代价（α=0 vs 0.5）：
bottle +0.1562、grid +0.1240、screw +0.1086、hazelnut +0.0628、cable +0.0598。

per-seed（policy × seed 的 5-cat 平均 d′）：GC = 5.0193 / 5.0179 / 4.9530；
SC-A = 4.9104 / 4.8433 / 4.8014；SC-B = 4.8449 / 4.8489 / 4.8343 →
**GC 在 3/3 seeds 上 preservation 均高于两个 control**（Δd′ GC−SC-A = +0.109 / +0.175 / +0.152）。

### 14.3 Exhaustive matched gates（10 个同 budget gate，EVALUATION-ONLY）

| gate | Δd′ vs fixed | \|Δz\| | worst-cat gain | neg transfer |
|---|---|---|---|---|
| **bottle+grid（= GC）** | **+0.3297** | 0.2643 | 0.0000 | 0 |
| bottle+screw | +0.1967 | 0.2612 | 0.0000 | 0 |
| **bottle+cable（= SC-A）** | +0.1847 | 0.2514 | 0.0000 | 0 |
| **grid+screw（= SC-B）** | +0.1757 | 0.2547 | 0.0000 | 0 |
| cable+grid | +0.1637 | 0.2450 | 0.0000 | 0 |
| bottle+hazelnut | +0.1332 | 0.2520 | −0.2107 | 1 |
| grid+hazelnut | +0.1122 | 0.2456 | −0.2107 | 1 |
| cable+screw | +0.0307 | 0.2419 | 0.0000 | 0 |
| hazelnut+screw | −0.0208 | 0.2425 | −0.2107 | 1 |
| cable+hazelnut | −0.0328 | 0.2327 | −0.2107 | 1 |

- Δd′ 跨度 **−0.0328 … +0.3297**；GC 排 **1/10**，次优 bottle+screw +0.1967（gap **+0.133 > ε**）。
- **只有 GC 自己**落在 GC − ε 内（`n_gates_not_worse_than_GC = 1`）。
- **6/10** gate 做到 worst-category gain = 0；**4/10**（全部含 hazelnut）产生 −0.2107 的 harm。
  → 「无 harm」并非 geometry 独有；「无 harm + preservation 最大」只有 GC。
- 结构性事实：每个 gate 的 gain 只来自其 2 个被 gate 的类别（其余 3 类 α=0.5 = best fixed，逐位不变），
  因此 gate 排序完全由「被 gate 类别的 α=0 收益」决定：
  bottle +0.877 > grid +0.772 > screw +0.107 > cable +0.047 > hazelnut −0.211。
  - geometry（升序 rank）= bottle, grid, cable, hazelnut, screw → top-2 = **最优 gate**
  - sensitivity SC-A（升序 rank）= bottle, cable, hazelnut, screw, grid → top-2 = {bottle, cable}（grid 排**最后**）
  - sensitivity SC-B（降序 rank）= grid, screw, hazelnut, cable, bottle → top-2 = {grid, screw}（bottle 排**最后**）

### 14.4 GC 在 10 个 gate 中的位置（§12）

| 维度 | GC 排名 | 说明 |
|---|---|---|
| defect preservation（d′ 越高越好） | **1 / 10** | 4.9967（最高） |
| worst-category harm（gain 越高越好） | **1 / 10** | 0.0000（并列最优） |
| negative transfer（越少越好） | **1 / 10** | 0 |
| robustness（\|Δz\| 越低越好） | **10 / 10** | 0.2643（10 个 gate 中**最差**） |
| Pareto status | **efficient** | 不被任何 gate / A / B / SC-A / SC-B 支配 |

### 14.5 GC vs matched controls（§11 primary comparison）

| 对照 | Δd′ (GC−ctrl) | Δ\|Δz\| (GC−ctrl) | worst-gain GC/ctrl | negTr GC/ctrl | seed 方向 | LOCO（去掉后 GC−ctrl Δd′） |
|---|---|---|---|---|---|---|
| **SC-A**（primary） | **+0.1450** | +0.0128（更差） | 0.0000 / 0.0000 | 0 / 0 | 3/3 GC 更好 | 去掉 **grid** → **−0.0117**；其余 4 个 → +0.18…+0.19 |
| SC-B（secondary） | +0.1540 | +0.0095（更差） | 0.0000 / 0.0000 | 0 / 0 | 3/3 GC 更好 | 去掉 **bottle** → **−0.0267**；其余 4 个 → +0.19…+0.22 |

**事实**：GC 在 preservation 上以 ≥ ε 的优势胜过两个 matched control，且 worst-category harm 与
negative transfer 不劣、3/3 seeds 方向一致、rank 1/10、Pareto-efficient；但 robustness 更差
（且是 10 个 gate 中最差），并且优势**在结构上只来自 GC 与对照的差异类别**（vs SC-A 是 grid/cable，
vs SC-B 是 bottle/screw）——LOCO 去掉对应类别后优势落入 ε 带内。

**解释**：在本 5 类、gate size = 2 的设定下，「选哪 2 个类别」的信息量很大（10 个 gate 的 Δd′ 相差 0.36），
而 geometry 的 top-2 恰好是唯一最优组合；两个 sensitivity orientation 都排错了最优组合中的一半。
因此 5C 的收益**不能**归因于「随便 gate 都差不多」，但 geometry 相对 sensitivity 的优势**集中在一个类别**。

**假设（未验证）**：geometry 是因为与 C2 damage 的 ρ=−0.90 而"知道"grid 比 cable/screw 更脆弱；
在更多类别（gate size 与类别数不成比例）或不同 backbone 下，这条优势是否保持未知。

## 15. Verdict（运行后）

**预注册判定链（literal）**：`results/experiment_5d/summary/verdict.json`

| 控制 | literal CASE | 触发的条件 |
|---|---|---|
| SC-A（primary） | **CASE_D** | **仅** `single_category_driver_of_identity_gain = true`；其余三条 CASE_D 条件显式为 false |
| SC-B（secondary） | **CASE_D** | 同上 |

CASE_A 条件逐条（primary = SC-A）：1 `Δd′ ≥ ε` **true**；2 `Δrobust ≤ +0.02` **true**；
3 worst-gain 不劣 **true**；4 neg-transfer 不劣 **true**；5 seed-level pair-win ≥ 2/3 **false**；
6 `rank ≤ 3` 且 Pareto-efficient **true**；7 LOCO 全为正 **false**。
CASE_C 条件：`|Δd′| < ε` **false**（0.1450 ≥ 0.10）；`≥4 个 gate 不劣于 GC` **false**（n_ge = 1）。

### 15.1 预注册缺陷披露（必须记录；**未修改协议、未重跑**）

按代码规则 literal 结果为 CASE_D，但该结果由我预注册的规则中**两处缺陷**决定，必须如实说明：

- **D1 — CASE_D driver 判据阈值过严**：driver 判据写成严格 `Δd′(GC − control) > 0`，而非冻结的 ε 带。
  由于 GC 与对照的差异按构造只落在 2 个类别上，去掉其中一个后差值天然落到 ±0.03 量级，
  实测在 **−0.0117**（= 0.12 ε，远在 ε 带内）触发 → 该条件对任意一对 gate **近乎必然触发**。
- **D2 — CASE_A 条件 5 与条件 2 互相矛盾**：条件 5 要求 seed-level **pair-win**，其定义要求
  `Δ|Δz| < 0`；而条件 2 明确容许 `Δ|Δz| ≤ +0.02`（即容许 GC robustness 略差）。
  两者不可能同时成立 → **CASE_A 结构性不可达**（实测 0/3）。
  按 §18 原文口径（"seed-level direction 至少 2/3 一致"）应为逐 seed 的 d′ 方向一致，
  实测为 **3/3 一致 → 该条应判 true**。

### 15.2 描述性复核（§18 原文口径；不是重新判定）

- CASE_A 条件 1–7 全部成立，**仅条件 8（"结果不是单一 category 完全驱动"）不成立**
  → **CASE_A 不可声称**。
- CASE_C 两条判据均显式排除 → **不是 generic conservative gating**。
- CASE_D 的实质含义（"geometry 不提供足够的可用策略信息"）与
  「GC 是 10 个同 budget gate 中唯一同时做到无 harm 且 preservation 最大、且 rank 1/10」不符。

**→ 实质性结论 = CASE_B（Geometry Useful but Identity Weak）**：
geometry-guided gating 仍然有可用价值（harm reduction 稳定、优于两个 matched sensitivity 对照、
在 matched-gate 空间中处于 preservation/harm 极端角），但 **predictor identity 未确立**
（优势由单一类别驱动、robustness 为 10 个 gate 中最差、n=5、gate size=2）。

**最终 CASE 归属请人工裁决**（literal = CASE_D / 实质 = CASE_B）。本实验**不得**据此声称 CASE_A。

## 16. Limitations

- **n_category = 5、gate_size = 2**：GC 与两个 control 的 gate 按构造共享 1–2 个类别，差异被限制在
  1–2 个类别上；identity 证据因此天然脆弱，LOCO 的「单类别驱动」结论部分是该设计的几何后果。
- **GC 的 gate = {bottle, grid} 恰好等于 5C 预先指定的 target-derived high-damage 子集**
  （`high_damage_subset_frozen`，来自 5A-H/5B 历史 damage）→ 需要警惕「目标子集被正常几何事后复现」
  这一解释；5D 能证明 geometry 复现了它，**不能**证明这是 geometry 独有的机制。
- **GC 的 robustness 是 10 个 gate 中最差**（0.2643）；本实验没有「两轴同时更优」的 policy，
  与 5C 的 trade-off 结论一致。
- **direction 依赖**：sensitivity control 的结论依赖 orientation 选择（两种 orientation 给出不同 gate，
  且都在 exhaustive 集内）；SC-A / SC-B 的结论应并读。
- 5D 是**历史 α 条件的重组**（CPU-only、零新 inference），证据强度受 5A-H 已有的 α × category 覆盖限制；
  所有指标为 5A-H 冻结口径（mean defect d′ / mean |ΔNormalScore_z|），无新主指标。
- 全部为 descriptive / exploratory，n=5，不做显著性主张，不做 causal claim。

## 17. Next（建议，未启动；需人工批准）

1. **CASE 归属裁决**：在本实验的 literal（CASE_D）与实质（CASE_B）之间由人工判定。
2. 若继续方案①：把结论口径固定为「gate 选择携带信息 + geometry 复现最优 gate，但 identity 未确立」，
   并需要**扩大类别数**（使 gate size 与类别数不成比例）来解除「单类别驱动」的设计性限制。
3. **不做**：根据 exhaustive 结果挑选新的 gate、调 threshold、改 gate size、改 α、新造 predictor、
   启动 5E / v2 / v3 / 方案②。
4. **STOP，等待人工 review**。

---

> **EXHAUSTIVE CONTROLS ARE EVALUATION-ONLY.**
> **NO BEST RANDOM GATE MAY REPLACE THE FROZEN GEOMETRY POLICY.**
