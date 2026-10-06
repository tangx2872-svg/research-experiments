# Experiment 6A — Geometry-Guided Soft Gating / Adaptive α Pilot

> 状态：**已完成**（GPU 10 units，人类批准；tmux 不可用 → screen detached 3 workers）。
> freeze：`results/experiment_6a/reference/policy_freeze.json`，
> SHA256 `d4364457b00bb886ada747722f1d4cb82e31a44df631e0920e2d1bfd4c4b50ce`；sanity **S1–S18 = 18/18 PASS**。
> **FINAL VERDICT：CASE_A — Soft Geometry Gating Works**（α_F = 0.25 满足 A+B+C+D；
> 保留 Hard-GC preservation gain 的 64.1%，回收 robustness penalty 的 55.9%）→ 见 §11–§13。
> 前序：5D ✅（`e0bda22`）Hard GC = bottle+grid，preservation 1/10、robustness 10/10（identity 未确立）。

---

## 0. Paper Navigation

📍 **论文阶段 9 — Robustness–Preservation Trade-off 方法改进（METHOD PILOT）**

① 现实问题 ✅ → ② 文献 ✅ → ③ defect-specific response ✅ → ④ 多类别/多 seed 稳定性 ✅
→ ⑤ 层级机制 + geometry evidence ✅ → ⑥ Geometry-Gated 方法雏形 ✅ → ⑦ 5C 方法有效性 ✅
→ ⑧ 5D matched-control / predictor identity ✅ → **⑨ Trade-off 方法改进 ← 6A** → ⑩ Final baseline + ablation → ⑪ Writing

## 1. Research Question

> Can geometry-guided **soft** normalization recover illumination robustness while retaining most of the
> defect-preservation gain of Hard GC?

**目标论断（HYPOTHESIS，不得提前写成事实）**：

> Geometry-guided adaptive normalization can provide a better preservation–robustness trade-off than either
> no normalization, globally fixed normalization, or hard geometry gating.

6A **不再问**「geometry 有没有信息」（5D 已回答）；也不做机制深挖。

## 2. Frozen Protocol

### 2.1 冻结的类别身份（来自 5D，禁止更改）

| 角色 | categories | α |
|---|---|---|
| **fragile** | `bottle`, `grid` | **α_F**（候选 grid） |
| tolerant | `cable`, `hazelnut`, `screw` | **0.5**（固定） |

```text
α_F = 0.0  ==  Hard Geometry Gating (5D GC)
α_F = 0.5  ==  Best Fixed α=0.5
mean α = (2·α_F + 3·0.5) / 5
```

### 2.2 候选 grid（运行前冻结，来源＝历史可用性 + 协议最小值）

| α_F | 历史来源（provenance） | 5D/5A 中的地位 |
|---|---|---|
| **0.00** | 5A B0 / 5A-H B0 | Hard GC 端点 |
| **0.25** | **5A B1**（历史 uniform 点，非新造 α） | 唯一可用的中间点 |
| **0.50** | 5A B2 / 5A-H B2 | Best Fixed 端点 |

**预注册的条件扩展（必须先获人工批准）**：`α_F = 0.40091275`（历史 5A C1 uniform 点）。
按协议 §5，仅当 Pareto knee 落在两个已有点之间时才允许补 1–2 个 α；该值**现在就已预先指定**，
不是看结果后选的。

grid 规则：`α_F ∈ {0, 最近的可用户历史 uniform 中间点, 0.5}`，**由历史可用性与协议最小值决定，
绝不由 6A target 结果决定**。

### 2.3 Budget matching 披露（必须写进论文）

Hard GC mean α = **0.30**；Soft GC mean α = `(2α_F+1.5)/5` > 0.30（α_F>0）。
因此 Soft GC 拿到的是**更强的 normalization**，α_F 越大 budget 越不 matched。
分析中必须区分 **policy identity effect** 与 **normalization-strength effect**；
6A 是 trade-off pilot，**不做强 causal claim**。

## 3. Historical α Asset Audit（P1，`reference/asset_audit.csv`）

| α | keys | full 5cat×3seed | layer-uniform | usable as α_F |
|---|---|---|---|---|
| 0.000000000 | 15 | ✅ | ✅ | **✅** |
| 0.250000000 | 1 (bottle) | ✗ | ✅ | ✗（grid 缺） |
| 0.301825500 | 1 (bottle) | ✗ | ✗（layer-wise） | ✗ |
| 0.400912750 | 1 (bottle) | ✗ | ✅ | ✗（grid 缺） |
| 0.452738250 | 15 | ✅ | ✗（G2: α_L2=0.4527, **α_L3=0.75**） | ✗ |
| 0.500000000 | 15 | ✅ | ✅ | **✅** |
| 0.601369125 | 15 | ✅ | ✅ | ✗（>0.5） |
| 0.801825500 | 15 | ✅ | ✅ | ✗（>0.5） |

- layer-uniform 且满覆盖的 uniform α = **{0.0, 0.5, 0.6014, 0.8018}**；
- **严格位于 (0, 0.5) 的 uniform α：不存在** → 6A 的 interpolation 无法纯 CPU 构建。
- 注意 `α=0.45273825` 虽然 15/15 覆盖，但来自 **layer-wise** config G2（α_L3=0.75），
  不属于 6A 的 uniform α 族，**不可用作 α_F**。

## 4. PRE-REGISTERED Success Criteria（运行前冻结，阈值唯一来源＝协议 + 5D reference）

| 判据 | 规则 | 阈值 |
|---|---|---|
| **A — Preservation retention** | `Δd′ vs fixed α=0.5 ≥ +0.20` | 0.20（≈ Hard GC gain 4.9967−4.6670 = +0.3297 的 60%） |
| **B — Robustness recovery** | `|Δz|_soft ≤ |Δz|_hardGC − 0.02` | ceiling = **0.244255**（由脚本从 5D 冻结 summary 算出，**非手敲**） |
| **C — Safety** | `worst_category_gain ≥ −ε` **且** `negative_transfer ≤ Hard GC` | ε = 0.10；Hard GC neg-transfer = 0 |
| **D — Stability** | ≥ **2/3 seeds** 在该 seed 上同时满足 A 与 B | — |

Hard GC 参考值（读自 `results/experiment_5d/summary/policy_summary.csv`）：
d′ = 4.996715、|Δz| = 0.264255、Δd′ vs fixed = +0.329722、Δ|Δz| vs fixed = +0.056039、
worst gain = 0.0、neg-transfer = 0。

## 5. Verdict Rules（运行前冻结）

- **CASE_A — Soft Geometry Gating Works**：**存在至少一个 α_F** 同时满足 A+B+C+D
  → 允许进入 6B refined validation。
- **CASE_B — Trade-off Exists but No Clear Knee**：α_F 形成连续 Pareto 曲线
  （robustness 持续改善、preservation 持续下降），但**没有任何点**同时满足 A+B+C+D
  → STOP Soft-GC；下一方法考虑 feature/layer adaptive normalization。
- **CASE_C — Fixed α Dominates**：所有 soft policy 都不能形成相对 fixed α=0.5 有意义的
  preservation 优势，或产生明显 negative transfer → **geometry-guided soft gating 淘汰**，不再调参。
- **CASE_D — Hard GC Remains Best**：soft 只损失 preservation、几乎不恢复 robustness
  → 保留 Hard GC；后续转向 illumination branch / layer-specific intervention。

**禁止**：dense α sweep、per-category 单独调最优 α、根据结果改阈值、删除失败结果。

## 6. Metrics（沿用 5A-H / 5C / 5D 冻结定义，不换指标）

- **Preservation ↑**：`mean defect d′`（primary）；另报 per-category / per-seed / Δd′ vs fixed / Δd′ vs Hard GC。
- **Robustness ↓**：`mean |ΔNormalScore_z|`（primary）；另报 per-category / per-seed / Δ vs fixed / Δ vs Hard GC。
- **Safety**：`worst_category_gain`、`negative_transfer_count`（ε=0.10）、
  high-damage subset `{bottle, grid}`、seed consistency。
- **Trade-off**：二维 Pareto 平面（x = mean |Δz| ↓，y = mean d′ ↑），**不引入加权总分**。
- 实现：复用 `experiment5a_h_analysis.load_all/unit_metrics`（未改动）。

## 7. Sanity S1–S18（当前状态：16 PASS / 2 PENDING / 0 FAIL）

| # | check | 状态 |
|---|---|---|
| S1 | historical frozen assets exist | PASS（6/6） |
| S2 | 5D GC assignment unchanged | PASS（fragile = bottle+grid） |
| S3 | fixed baseline unchanged | PASS（α=0.5，5C/5D 冻结） |
| S4 | metric implementation unchanged | PASS（重算 α_F=0 与 5D GC 差 2.9e-07 = 6 位小数舍入） |
| S5 | α candidate grid frozen before target analysis | PASS（freeze 早于任何 6A target） |
| S6 | all required cat×seed×α keys exist | **PENDING（GPU）**：缺 5 个 unit |
| S7 | illumination protocol unchanged | PASS（4 shifts） |
| S8 | historical score equivalence | PASS（5A vs 5A-H, 732 行, max\|Δscore\|=0） |
| S9 | no target-derived category reassignment | PASS |
| S10 | no α selected before full candidate evaluation | PASS |
| S11 | mean α correctly calculated | PASS（(2α_F+1.5)/5） |
| S12 | per-category aggregation correct | PASS |
| S13 | per-seed aggregation correct | PASS |
| S14 | negative-transfer implementation unchanged | PASS |
| S15 | ε = 0.10 unchanged | PASS |
| S16 | frozen assets hash unchanged after analysis | PASS（6/6 md5 一致） |
| S17 | no accidental GPU run | PASS（本脚本 0 GPU） |
| S18 | output completeness | **PENDING**（raw/policy_results.csv 待 GPU 后生成） |

按 §13：**18/18 PASS 之前不得给出 final verdict**。

## 8. [6A GPU REQUEST]（阻塞点）

```text
missing alpha values   : 0.25            (uniform, historical 5A B1)
missing units (5)      : bottle:s1, bottle:s2, grid:s0, grid:s1, grid:s2   @ alpha=0.25
per-unit runtime        : 5C 实测 mean 102.4s / max 161.1s (n=18 units, bottle+grid)
estimated total runtime : 13.4 min @1 worker  |  4.5 min @3 workers
estimated VRAM          : mean 2935 MB / max 3265 MB per unit
  -> 3 workers          : ~9.6 GB << 24 GB (RTX 3090)
```

**为什么必须补**：6A 的 policy family 需要 `bottle/grid @ α_F`；历史中**不存在**严格位于 (0, 0.5) 的
layer-uniform 且 5cat×3seed 满覆盖的 α，因此 Hard GC(α_F=0) 与 Best Fixed(α_F=0.5) 之间的插值点
**无法纯 CPU 构建**。`α_F=0.25` 是 **5A 既有协议点**（B1），不是为 6A 新造的 α。

**可选（需另行批准）**：`α_F = 0.40091275`（历史 5A C1 uniform 点）再补 **5 units**（bottle s1/s2、grid s0/s1/s2），
总 ~27 min @1 worker / ~9 min @3 workers。按协议 §5，严格来说应等第一次结果出现未解 knee 时才补。

**任何 6A target 结果目前均未被读取**；grid 不依赖任何 6A 结果。

## 9. Planned Outputs

```text
results/experiment_6a/reference/   policy_freeze.json  policy_freeze.sha256  asset_audit.csv  (gpu_request.json)
results/experiment_6a/raw/         policy_results.csv  per_category.csv  per_seed.csv
                                   hard_gc_equivalence.json
results/experiment_6a/summary/     policy_summary.csv  pareto_summary.csv  verdict.json   [GPU 后生成]
results/experiment_6a/sanity/      sanity_checks.csv
results/experiment_6a/figures/     soft_gc_pareto.png  preservation_vs_alpha.png
                                   robustness_vs_alpha.png  bottle_response.png  grid_response.png  [GPU 后生成]
scripts/experiment6a_soft_geometry.py   (audit / freeze / sanity / reconstruct)
scripts/experiment6a_analysis.py        [GPU 后生成]
```

## 10. Stop Rules

- 关键 sanity FAIL、NaN/Inf、asset hash 变化、等价性非 0 → **STOP**，保留现场，不改协议。
- 6A 结束后 **STOP，不启动 6B / Method 2**，等待人工 review。
- 失败结果必须保留。

---

# 运行后（GPU 已批准并完成）

## 11. Results

- **GPU 范围（人类批准）**：`α_F = 0.25`（5A B1）+ `α_F = 0.40091275`（5A C1，预注册条件扩展）→
  10 units（bottle s1/s2 + grid s0/s1/s2，各 2 个 α_F），tmux 不可用 → 改用 **screen detached**
  （3 workers，等价机制）；每 unit 落盘 + `run.lock` PID + `unit_done` resume。
- **Smoke / 等价性**：先重跑 `bottle:seed0 @ α=0.25`（5A B1 既有条件）→
  **n_keys=183, `max|Δscore| = 0.000e+00`**；τ_val 19.0138 与 5A B1 一致。
  运行时等价性 `max|Δscore|` 亦为 0（同一 `run_config` 路径）。
- **Reconstruct 等价性**：用 6A 管线重算 `α_F=0` 与 5D 冻结 GC 差 `|Δd′|=2.94e-07`、`|Δ|Δz||=3.08e-07`
  （= 5D summary CSV 的 6 位小数舍入）。
- **sanity S1–S18 = 18/18 PASS**（含 S6 key 完整性、S18 输出完整性）。
- 运行时：单 unit 46–111 s（mean ~105 s），峰值显存 ~2.9 GB / worker。

### 11.1 Policy summary（4 个 α_F；5 cat × 3 seeds 平均）

| policy | α_F | mean α | defect d′ ↑ | \|Δz\| ↓ | Δd′ vs fixed | Δd′ vs HardGC | Δ\|Δz\| vs fixed | Δ\|Δz\| vs HardGC | worst cat gain | neg transfer | A/B/C/D |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **Hard GC**（= α_F=0） | 0.00000 | 0.3000 | **4.9967** | 0.2643 | +0.3297 | 0 | +0.0560 | 0 | 0.0000 | 0 | A✅ B❌ C✅ D❌ |
| **Soft-GC（α_F=0.25）** | 0.25000 | 0.4000 | 4.8782 | **0.2329** | **+0.2112** | −0.1185 | +0.0247 | **−0.0313** | **0.0000** | **0** | **A✅ B✅ C✅ D✅** |
| Soft-GC（α_F=0.40091275） | 0.40091 | 0.4604 | 4.7522 | 0.2133 | +0.0852 | −0.2446 | +0.0051 | −0.0509 | 0.0000 | 0 | A❌ B✅ C✅ D❌ |
| **Best Fixed**（= α_F=0.5） | 0.50000 | 0.5000 | 4.6670 | **0.2082** | 0（参照） | −0.3297 | 0（参照） | −0.0560 | 0.0000 | 0 | A❌ B✅ C✅ D❌ |
| （参照）Original α=0 全体 | — | 0.0000 | 4.9852 | 0.3105 | +0.3183 | −0.0115 | +0.1023 | +0.0462 | — | — | — |

### 11.2 Pareto（§7 二维，无加权总分）

```text
Original               d′=4.9852  |Δz|=0.3105   -> dominated by Hard GC
Best Fixed             d′=4.6670  |Δz|=0.2082   -> Pareto-efficient
Soft-GC α_F=0.4009     d′=4.7522  |Δz|=0.2133   -> Pareto-efficient
Soft-GC α_F=0.25       d′=4.8782  |Δz|=0.2329   -> Pareto-efficient   ← knee
Hard GC                d′=4.9967  |Δz|=0.2643   -> Pareto-efficient
```

前沿在 α_F∈[0, 0.25] 明显向外凸出：**α_F=0.25 同时比 Best Fixed 多保留 +0.2112 的 d′，
又比 Hard GC 少 0.0313 的 |Δz|** → 二维判据下**优于两个端点各自的弱轴**，且不被任何点支配。

## 12. Verdict（预注册判定链，§8/§9）

| α_F | A（Δd′ ≥ +0.20） | B（\|Δz\| ≤ HardGC−0.02 = 0.2443） | C（worst ≥ −ε 且 negTr ≤ 0） | D（≥2/3 seeds 同时满足 A&B） |
|---|---|---|---|---|
| 0.25 | ✅ +0.2112 | ✅ 0.2329 ≤ 0.2443 | ✅ 0.0000 / 0 | ✅ **3/3**（+0.203/−0.0411、+0.206/−0.0327、+0.224/−0.0202） |
| 0.40091275 | ❌ +0.0852 | ✅ 0.2133 | ✅ | ❌ 0/3（A 未达） |

**FINAL VERDICT：CASE_A — Soft Geometry Gating Works**
（存在 α_F 使 A+B+C+D 全部满足 → 允许进入 6B refined validation。）

**关键量化**：α_F=0.25 **保留 Hard GC preservation gain 的 64.1%**（0.2112 / 0.3297），
**回收 Hard GC robustness penalty 的 55.9%**（0.0313 / 0.0560），worst-category harm = 0、negative transfer = 0。

**必须声明的限定（不得省略）**

1. **A 的余量很薄**：+0.2112 对阈值 +0.20，余量仅 **+0.0112（5.6%）**；B 的余量 +0.0114。
   若阈值改为 0.22，α_F=0.25 即不成立 → **knee 的存在依赖于预注册阈值，结论稳健性有限**。
2. **budget 不 matched**：Soft GC(α_F=0.25) mean α = **0.40** > Hard GC 0.30 → 它使用了**更多** normalization。
   本实验**只改变 fragile 端的强度，不改变 category identity**（identity 冻结），
   因此这是 **normalization-strength 效应**，不是 policy-identity 效应（identity 由 5D 检验，结论为未确立）。
   历史中也**没有 budget-matched 的 uniform α≈0.40 对照**（0.40091275 只有 bottle/seed0）→ 无法排除
   "任何在该 budget 上的分配都差不多"。
3. n_category = 5、gate_size = 2、α_F 只有 2 个中间点；曲线是插值曲线，不是连续 tolerance 的测量。
4. 全部 descriptive；不做显著性主张、不做 causal claim；τ_val / bank / coreset 与 5A-H 同一路径。

## 13. Paper Progress

**Q1  Soft gating 是否恢复 robustness？** ✅ 是。α_F=0.25 把 fragile 端的 |Δz| 从 Hard GC 的 0.2643 降到 0.2329
（改善 0.0313，> 0.02）；bottle 0.4422→0.3352、grid 0.2244→0.1747（相对各自的 α_F=0）。
tolerant 端（cable/hazelnut/screw）按构造完全不变。

**Q2  保留了多少 Hard-GC preservation gain？** **64.1%**（+0.2112 / +0.3297），
即从 α=0 起每类别 gain 的一半以上被保住；代价是相对 Hard GC 损失 0.1185 的 5-cat 平均 d′。

**Q3  是否出现明确 Pareto knee？** ✅ 有 knee，落在 **α_F ∈ (0, 0.25]**（唯一满足 A+B+C+D 的点）。
α_F=0.40091275 已经开始牺牲 preservation 而 A 不达标；α_F=0 则 robustness 不达标。
但 knee 的定位依赖预注册阈值（见限定 1）。

**Q4  bottle/grid 是否表现出不同的 continuous tolerance？** **弱/否**。
两者的 marginal efficiency 都集中在 α_F∈[0, 0.25]（knee 位置一致）；
但幅度不同 —— 该区间内每损失 1 单位 d′ 换回的 robustness，**grid 0.326 vs bottle 0.243（差 ~34%）**。
因此记录为 descriptive note：**"category-specific continuous tolerance may be required"** 只作为
**未验证假设**，本实验**不**为每个类别单独调 α。

**Q5  这个结果让论文哪一格完成？** 完成 **第 9 格 "Robustness–Preservation Trade-off 方法改进"的 pilot**：
证据链新增一条 —— **hard gate 在 fragile 端过于极端；把 fragile 端 α 从 0 提到 0.25，
可在同一 category identity 下同时改善 robustness 并保留大部分 preservation**。
它**不**推进 identity 那一格（5D 的 CASE_B 结论不变），也**不**证明 geometry 在连续 α 上有特殊信息。

## 14. Limitations

- knee 余量薄（A 余量 5.6%）、budget 不 matched（mean α 0.30→0.40）、无 budget-matched uniform 对照。
- 中间点只有 2 个（0.25、0.40091275），曲线由历史协议点拼接，非密集扫描（按 §5 刻意如此）。
- 只有 bottle/grid 的 fragile 端被改动；tolerant 端固定 0.5 是 5D 冻结设定的继承，未重新优化。
- 单 backbone（wide_resnet50_2）、5 类别、synthetic photometric illumination 协议，不外推。
- 未做显著性检验；n=5；不做 causal claim。

## 15. Next（建议，**未启动，需人工批准**）

1. **6B refined validation**：在**同一 category identity** 下用**更多 α_F 点 + budget-matched 对照**
   复核 knee 是否稳定（尤其 A 的薄余量），并检验 tolerant 端是否也应软化（本实验禁止，需新协议）。
2. 若要谈 identity，需要 5D 之后的**新证据**（5D 已判 identity 未确立），不能由 6A 推出。
3. **不做**：为每个类别单独调 α、dense sweep、改 success threshold、改 ε、引入新 backbone/dataset、
   启动 Method 2。

---

> **EXHAUSTIVE / RANDOM CONTROLS 之类的外推禁止**：本实验的结论只适用于
> fragile={bottle,grid}、tolerant α=0.5、5 类别、3 seeds 的冻结设定。
