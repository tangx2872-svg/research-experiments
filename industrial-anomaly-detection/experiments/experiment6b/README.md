# Experiment 6B — Confirmatory Validation of Soft Geometry Gating

> 状态：**已完成**（GPU 27 units + 1 smoke；freeze
> `1b4d1e5db4f0e39a18dce46916aeb49749b474f5451d3ab9e972eb51b285ca06`；sanity **S1–S20 = 20/20 PASS**）。
> **FINAL VERDICT：CASE_C — 6A 的 knee 不复现为区域（α_F=0.25 是孤立点）；其收益主要来自整体
> normalization strength（60.1%）而非 selective allocation（39.9%）。**
> 前序：6A ✅ **CASE_A**（`1b3edd2`，soft-gating pilot，α_F=0.25 knee，保留 64.1% gain / 回收 55.9% penalty）
> · 5D ✅（`31150bd`，identity 未确立）

---

## 0. Paper Navigation

📍 **论文阶段 9 — Robustness–Preservation Trade-off 方法改进（confirmatory validation）**

① 现实问题 ✅ → ② 文献 ✅ → ③ defect-specific response ✅ → ④ 多类别/seed 稳定性 ✅ →
⑤ 层级机制 + geometry evidence ✅ → ⑥ Geometry-Gated 雏形 ✅ → ⑦ 5C 方法有效性 ✅ →
⑧ 5D matched-control（identity 未确立）✅ → ⑨ **6A soft-gating pilot（CASE_A）✅ → 6B confirmatory（当前）** →
⑩ Final baseline + ablation ⏸ → ⑪ Writing ⏸

## 1. Research Question（两个，分别回答）

**G1 — knee 稳定性**：6A 的 α_F=0.25 是一个**稳定 Pareto knee region**，还是一个**孤立点**？
在 α_F=0.25 周围做**最小** confirmatory sampling。

**G2 — 因果控制（本实验最重要）**：Soft-GC 的优势来自 **selective allocation**，还是仅仅来自
**整体 normalization strength**？
构造与 Soft-GC(α_F=0.25) **mean α matched** 的 **Uniform** baseline（所有类别同一 layer-uniform α）。

## 2. Frozen（全部继承，**不得重选/不得改**）

| 项 | 值 | 来源 |
|---|---|---|
| fragile | `bottle, grid` | 5D/6A 冻结 |
| tolerant @ α_T | `cable, hazelnut, screw` @ **0.5** | 5D/6A 冻结 |
| ε | **0.10** | 5A-H §2 冻结 |
| primary metrics | mean defect d′ / mean \|ΔNormalScore_z\| | 6A 未改动 |
| success criteria A–D | 阈值 **原样继承**（A: Δd′ vs fixed ≥ +0.20；B: \|Δz\| ≤ HardGC−0.02 = **0.244255**；C: worst gain ≥ −ε 且 neg transfer ≤ 0；D: ≥2/3 seeds 同时 A&B） | `results/experiment_6a/reference/policy_freeze.json` |
| 指标实现 | `experiment5a_h_analysis.load_all/unit_metrics`（经 6A 未改动） | 代码路径 |
| illumination 协议 | brightness 0.7/1.3 + gamma 0.7/1.3（4 shifts） | 5A-H 冻结 |
| seed aggregation | 每 seed 对 5 类求均值；policy 级 = 15 unit 均值 | 6A 冻结 |

## 3. Alpha Grid

| α_F | 状态 | provenance |
|---|---|---|
| 0.0 | 已有（= Hard GC） | 5A-H B0 |
| **0.125** | **新增（6B confirm）** | 人类指定的最小 confirm 采样 |
| **0.20** | **新增（6B confirm）** | 同上 |
| 0.25 | 已有（= 6A knee） | 6A（bottle+grid × 3 seeds） |
| **0.30** | **新增（6B confirm）** | 人类指定的最小 confirm 采样 |
| 0.40091275 | 已有 | 5A C1（bottle s0）+ 6A（bottle s1/s2, grid s0–s2） |
| 0.5 | 已有（= Best Fixed） | 5A-H B2 |
| **Uniform baseline α = 0.40091275** | 5 类统一 | 历史 5A C1 的 layer-uniform 点；\|α − mean α(Soft 0.25)=0.40\| = **0.000913**（最小） |

评估点共 **7 个 α_F + 1 个 Uniform + Original**（**非 dense sweep**）。

## 4. `[6B GPU REQUEST]`（审计后输出，实际执行）

```text
GOAL 1  confirm sampling（fragile 端 bottle+grid × 3 seeds × 3 个 α_F）  : 18 units
GOAL 2  mean-α matched Uniform baseline（cable/hazelnut/screw × 3 seeds × α=0.40091275）: 9 units
total 27 units | 72.5 min @1 worker | 24.2 min @3 workers | ~2.9 GB/unit（3 workers ~9.6 GB）
```

**为什么必须补**：G1 所需的 0.125/0.20/0.30 在历史中不存在（0.3018255 是 layer-wise G1 的 α_L2，非 uniform）；
G2 所需的 cable/hazelnut/screw @0.40091275 从未跑过（5A C1 只覆盖 bottle/seed0，6A 只覆盖 bottle+grid）。
所有 α 都是**人类指定**或**历史既有协议点**，无任何 α 由 6B target 结果决定（此前未读取）。

## 5. Sanity（沿用 6A 的 S1–S18 + 6B 专属项）

6A 的 S1–S18（资产存在 / identity 未改 / baseline 未改 / 指标实现未改 / grid 先于 target 冻结 /
key 完整 / 协议未改 / 等价性 / 无 target 派生重分配 / 无预选 α / mean α 公式 / per-cat / per-seed 聚合 /
negative-transfer 实现 / ε / 资产 hash / 无 GPU 误跑 / 输出完整）**全部重跑**，另加：

```text
S19  Uniform baseline 的 mean α 与 Soft-GC(0.25) 的 gap ≤ 0.01（matching 质量）
S20  knee region 判定只使用预注册的 A–D 与预定义 consecutive 规则
```

**任一项 FAIL → 不给 verdict。**

## 6. Verdict Rules（PRE-REGISTERED）

- **region** = 在排序后的 α_F 列表中**连续**满足 A∧B∧C∧D 的点的集合（0.25 必须在其中才算候选）。
- **allocation effect** = `d′(Soft 0.25) − d′(Uniform)`；**strength effect** = `d′(Uniform) − d′(Best Fixed)`。

| CASE | 条件 |
|---|---|
| **CASE_A** | region ≥ **3** 连续点（含 0.25）**且** `Δd′(Soft−Uniform) ≥ +ε` **且** `Δ\|Δz\|(Soft−Uniform) ≤ +0.02` |
| **CASE_B** | region ≥ **2** 连续点（含 0.25）**且** `Δd′(Soft−Uniform) > −ε`（region 成立，但优势不是明确的 allocation 优势） |
| **CASE_C** | 只有 α_F=0.25（或更少）满足 A∧B∧C∧D → 6A 的 knee **不复现**，是单点结果 |
| **CASE_D** | Uniform(matched) **Pareto-dominates** Soft-GC(0.25)（d′ ≥ 且 \|Δz\| ≤，至少一个严格） |

判定顺序：**D → A → B → C**（首个命中即判定）。

## 7. Stop Rules

关键 sanity FAIL、NaN/Inf、asset hash 变化、等价性非 0 → **STOP**，保留现场，不改协议。
6B 结束后 **STOP，不启动 6C / Final experiments**，等待人工 review。失败结果必须保留。

---

# 运行后

## 8. Results

- **GPU**：27 个协议必需 unit + 1 个等价性 smoke；screen detached 3 workers（每 worker 9 units），
  单 unit ~90–150 s、峰值 ~2.9 GB/worker。**smoke**：重跑 `bottle:seed0 @0.40091275`（= 5A C1 既有条件）
  → **183 keys、`max|Δscore| = 0.000e+00`**。
- **Reconstruct 等价性**：重算 Hard GC 与 6A/5D 冻结值差 `|Δd′| = 2.94e-07`、`|Δ|Δz|| = 3.08e-07`（6 位小数舍入）。
- **sanity S1–S20 = 20/20 PASS**（含 S8 等价性、S19 uniform mean-α match 0.000913、S20 区域规则预注册）。

### 8.1 全部 policy（5 cat × 3 seeds 平均）

| policy | α_F | mean α | defect d′ ↑ | \|Δz\| ↓ | Δd′ vs fixed | Δ\|Δz\| vs fixed | worst-cat gain | neg transfer | A | B | C | D |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Hard GC | 0.00000 | 0.3000 | 4.9967 | 0.2643 | +0.3297 | +0.0560 | 0.0000 | 0 | ✅ | ❌ | ✅ | ❌ |
| Soft-GC | 0.12500 | 0.3500 | 4.9308 | 0.2488 | +0.2638 | +0.0406 | 0.0000 | 0 | ✅ | ❌ | ✅ | ❌ (1/3) |
| Soft-GC | **0.20000** | 0.3800 | 4.9309 | **0.2423** | +0.2639 | +0.0341 | 0.0000 | 0 | ✅ | ✅ | ✅ | ❌ (**1/3**) |
| **Soft-GC** | **0.25000** | **0.4000** | 4.8782 | 0.2329 | **+0.2112** | +0.0247 | 0.0000 | 0 | ✅ | ✅ | ✅ | ✅ (**3/3**) |
| Soft-GC | 0.30000 | 0.4200 | 4.8220 | 0.2230 | +0.1550 | +0.0148 | 0.0000 | 0 | ❌ | ✅ | ✅ | ❌ (0/3) |
| Soft-GC | 0.40091 | 0.4604 | 4.7522 | 0.2133 | +0.0852 | +0.0051 | 0.0000 | 0 | ❌ | ✅ | ✅ | ❌ |
| Best Fixed | 0.50000 | 0.5000 | 4.6670 | 0.2082 | 0（参照） | 0（参照） | 0.0000 | 0 | ❌ | ✅ | ✅ | ❌ |
| **Uniform(matched)** | — | **0.4009** | 4.7940 | **0.2134** | **+0.1270** | +0.0052 | **+0.0159** | **0** | ❌ | ✅ | ✅ | ❌ |
| （参照）Original | — | 0.0000 | 4.9852 | 0.3105 | +0.3183 | +0.1023 | — | — | ✅ | ❌ | ❌ | ❌ |

### 8.2 knee region（G1）

```text
knee regions (consecutive A&B&C&D) : [(0.25, 0.25, 1)]   -> region length = 1
```

- **A∧B∧C∧D 只在 α_F = 0.25 成立 → 区域长度为 1（孤立点）**，不是稳定区域。
- 放宽到 A∧B∧C（不含 seed-level D）时，带为 **α_F ∈ [0.20, 0.25]（长度 2）**：
  α_F=0.125 因 **B 失败**（|Δz| 0.2488 > 0.2443）落在带外；α_F=0.30 因 **A 失败**（Δd′ +0.1550 < 0.20）落在带外。
- seed-level D 通过数：**0.125 → 1/3，0.20 → 1/3，0.25 → 3/3，0.30 → 0/3**。逐 seed 明细：

```text
α_F=0.20 : s0 +0.2581 / −0.0361 PASS ; s1 +0.2957 / −0.0156 FAIL ; s2 +0.2379 / −0.0141 FAIL  -> 1/3
α_F=0.25 : s0 +0.2033 / −0.0411 PASS ; s1 +0.2065 / −0.0327 PASS ; s2 +0.2237 / −0.0202 PASS  -> 3/3
（每格 = Δd′ vs fixed / Δ|Δz| vs Hard GC）
```

### 8.3 Uniform matched control（G2，因果控制）

| 量 | Soft-GC(0.25) mean α=0.400000 | Uniform(0.40091275) mean α=0.400913 | 差（Soft − Uniform） |
|---|---|---|---|
| mean defect d′ | 4.8782 | 4.7940 | **+0.0842**（< ε=0.10） |
| mean \|Δz\| | 0.2329 | 0.2134 | **+0.0195（Soft 更差）** |
| worst-category gain | 0.0000 | +0.0159 | −0.0159 |
| negative transfer | 0 | 0 | 0 |

**分解**（vs Best Fixed 的总 Δd′ = +0.2112）：

```text
strength effect   (Uniform − Best Fixed) : Δd′ = +0.1270  Δ|Δz| = +0.0052  -> 24.6 d′ per unit |Δz|
allocation effect (Soft    − Uniform)    : Δd′ = +0.0842  Δ|Δz| = +0.0195  ->  4.3 d′ per unit |Δz|
-> strength 占 60.1%，allocation 占 39.9%；allocation 的兑换率比 strength 差 5.7 倍
```

**附带事实**：`Uniform(0.4009)` 相对 Best Fixed 的 `Δd′ = +0.1270` 只付出 `Δ|Δz| = +0.0052`，且
worst-category gain 为正（+0.0159）、negative transfer 0 —— 即**单纯把全局 α 从 0.5 降到 0.4009，
比任何 geometry gating 都更"划算"**。

## 9. Verdict（预注册判定链，§6）

| 检查 | 结果 |
|---|---|
| CASE_D：Uniform Pareto-dominates Soft(0.25) | **❌**（Soft d′ 高 +0.0842，Uniform \|Δz\| 低 0.0195 → 互不支配） |
| CASE_A：region ≥ 3 且 allocation Δd′ ≥ ε 且 Δ\|Δz\| ≤ +0.02 | **❌**（region length = 1；allocation Δd′ +0.0842 < ε） |
| CASE_B：region ≥ 2 且 allocation Δd′ > −ε | **❌**（region length = 1） |
| CASE_C：只有 α_F=0.25（或更少）满足 A∧B∧C∧D | **✅** |

**FINAL VERDICT：CASE_C — 6A 的 knee 不复现为区域；α_F=0.25 是孤立点。**

**必须同时披露的预注册敏感性（不得省略）**

1. **region 口径决定 CASE_B/CASE_C**：本实验预注册的 region 定义是 A∧B∧C∧D。若只用**聚合级** A∧B∧C
   （不含 seed-level D），则 region = {0.20, 0.25}（长度 2）→ 判 **CASE_B**。本实验按预注册给出 **CASE_C**，
   并如实记录该敏感性（**未**回改判定）。
2. **无论哪种口径，CASE_A 都不可达**：A 在 α_F=0.30 失效（+0.1550 < 0.20）、B 在 α_F=0.125 失效
   （0.2488 > 0.2443）→ 任何连续 3 点都不可能同时满足 A∧B。**6A 的 CASE_A 不可能被 confirm 采样确认。**
3. **6A 的数值本身完全复现**（α_F=0.25：d′ 4.8782、\|Δz\| 0.2329、Δd′ +0.2112，与 6A 一致）。
   本实验否定的是「0.25 是 knee **region**」，不是否认该点满足判据。
4. **pass window 很窄**：A 要求 α_F ≲ 0.26（在 0.25 处余量 0.0112），B 要求 α_F ≳ 0.195
   （在 0.20 处余量 0.0020）；seed-level D（每 seed \|Δz\| 改善 ≥ 0.02）把带进一步压成单点。

## 10. Limitations

- n_category = 5、gate_size = 2、soft 点 7 个（协议禁止 dense sweep）；knee 定位本质受阈值分辨率限制。
- Uniform baseline 是历史最接近的 layer-uniform α（0.40091275），mean-α gap 0.000913，不是精确 0.40。
- 6B 的裁定**部分取决于预注册阈值**（§9.1/§9.4），因此结论应表述为
  「**在 6A 冻结判据下**，0.25 不构成区域」，而不是「0.25 无意义」。
- 单 backbone、synthetic photometric illumination；无显著性检验；除 matched-strength 对照外不做 causal claim。

## 11. Paper Progress

- **Q1 0.20–0.30 是否形成稳定 knee region？** ❌ 否（长度 1）。放宽到聚合级 A∧B∧C 也只有 [0.20, 0.25]（长度 2）。
- **Q2 α_F=0.25 是否为孤立点？** ✅ 在预注册判据下是（A∧B∧C∧D 仅此一点；seed-level 3/3 vs 邻点 0–1/3）。
- **Q3 Soft-GC 是否优于 mean-α matched Uniform？** 方向上是，但未达预注册 ε：d′ **+0.0842**（< 0.10），
  而 \|Δz\| 反而差 **+0.0195**。
- **Q4 优势来自 allocation 还是 strength？** **主要来自 strength**：strength 占 Δd′ 的 **60.1%**（+0.1270），
  allocation 占 39.9%（+0.0842），且 allocation 的 preservation/robustness 兑换率比 strength **差 5.7 倍**。
  即「把 fragile 端 α 提到 0.25」的大部分收益，只是「整体少施加一点 normalization」。
- **对证据链的贡献**：第 9 格由「6A：存在 CASE_A 的 soft-gating knee」修正为
  「**6A 的 CASE_A 是预注册阈值下的窄交叉点，不构成区域；其收益主要来自整体 strength，而非 selective
  allocation**」。这进一步**削弱** "geometry-guided soft gating" 的方法叙事（与 5D 的 identity 未确立一致）。

## 12. Next（建议，**未启动，需人工批准**）

1. 若继续 trade-off 方向，必须**换判据或换目标**：例如改用**前沿效用率**（preservation per robustness）
   或 budget-matched 对照下的 Pareto 支配判定，并把「global α 0.5 → ~0.40」作为**必须并列的最强 baseline**
   （本实验显示它比 geometry gating 更划算）。
2. 若要谈 identity，需要 5D 之后的新证据（如扩大类别数），不能由 6A/6B 推出。
3. **不做**：per-category 调 α、dense sweep、据 6B 结果改 threshold/ε、改 primary metric、
   新 backbone/dataset、启动 6C / Final experiments。

---

> **结论仅适用于**：fragile = bottle+grid、tolerant @ 0.5、5 类别、3 seeds、6A 冻结的 A–D 判据与 4 个 photometric shifts。
