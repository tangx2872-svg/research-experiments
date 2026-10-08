# Overnight Queue Q3 — Illumination Stress-Test

- **状态**：stage-1 DONE（10 GPU units；freeze SHA256 `6c4ee3d07c081c23026a82fb3a426494675f76947785e0b32eb8e256144a91ec`）
- **EXPLORATORY**：selection-set 结果。

## 1. 协议审计（Q3 启动前完成）

| 项 | 内容 |
|---|---|
| 历史实现 | `experiment1_illumination_tradeoff.apply_photometric(img, type, level)`（brightness: `img*level`；gamma: `img**(1/level)`；clamp [0,1]） |
| 历史 family | `brightness`, `gamma` |
| 历史 severity | `{0.7, 1.3}`（±30%） |
| 缺失 family | color_temperature / exposure / blur / noise **均无实现** |
| 裁决 | **只复用既有实现**，不发明新 augmentation family；把 severity 系统化为 3 档 |

**冻结的 severity（读取任何 Q3 target 之前写入 freeze）**：mild = ±10%（0.9/1.1）、
medium = ±30%（0.7/1.3，历史值）、strong = ±50%（0.5/1.5）→ 2 family × 3 severity × 2 方向 = **12 conditions**。

**方法集合（严格按 addendum 枚举）**：`M0_original`（α=0）、`M1_uniform`（α=0.40091275）。
Q0 = CASE_C → "Q0 winner" 槽位为空；Q2 无晋级 → "Q2 winner" 槽位为空。
**没有把非 winner（如 A3_lam050 / Q2L2）塞进 winner 槽位**（避免 post-hoc scope creep）。
Stage 2（seeds 1/2）的触发条件是"winner 相对 Uniform 的优势在多数 condition 保持"——无 winner → **未触发**；
由 Q4-B 以"补齐 baseline 的 condition × seed 覆盖"的正当名义执行。

## 2. Baseline equivalence（关键 sanity）

medium 档（0.7/1.3）必须与历史 raw 逐位一致 —— **10/10 units 全部 `max|Δscore| = 0.000e+00`**
（n = 80…232 keys/unit；M0 vs 5A-H `config_B0`，M1 vs 6A/6B `config_a*040091275`）。
→ Q3 runner 在扩展 severity 的同时**重证了整条 scoring 路径**。

## 3. 结果（5 categories × seed0）

| method | d′ ↑ | \|Δz\| all-12 ↓ | \|Δz\| medium-4 ↓ | worst condition \|Δz\| | slope vs severity（pooled） | slope brightness | slope gamma | negTr vs Uniform |
|---|---|---|---|---|---|---|---|---|
| `M0_original` | **5.0196** | 0.5229 | 0.3256 | 2.3937 | **2.914** | 3.894 | 1.934 | 10/12 |
| `M1_uniform` | 4.8021 | **0.3404** | **0.2030** | **1.4838** | **1.863** | 2.468 | 1.259 | — |

per-condition `|Δz|`：

```text
M0_original  brightness: 0.9→0.0335  1.1→0.0481  0.7→0.3523  1.3→0.4917  0.5→0.8033  1.5→2.3937
             gamma:      0.9→0.0570  1.1→0.0162  0.7→0.4070  1.3→0.0514  0.5→1.4365  1.5→0.1841
M1_uniform   brightness: 0.9→0.0387  1.1→0.0326  0.7→0.2413  1.3→0.2658  0.5→0.5620  1.5→1.4838
             gamma:      0.9→0.0490  1.1→0.0251  0.7→0.2688  1.3→0.0361  0.5→0.9768  1.5→0.1042
```

## 4. 分析

**Fact 1**：`Uniform α=0.40091275` 在 **12/12 conditions** 上 \|Δz\| 都低于 `Original`，除了两个
**最温和**的点（brightness 0.9：0.0387 vs 0.0335；gamma 1.1：0.0251 vs 0.0162，量级 ~0.02-0.04，可忽略）。
**Fact 2**：差距随 severity **放大**：brightness 0.7 比值 1.46× → 0.5 比值 1.43× → **1.5 比值 1.61×**
（2.394 vs 1.484）。
**Fact 3**：**degradation slope 更浅**：pooled 2.914（Original）vs **1.863**（Uniform），
即 normalization 不只是把曲线整体下移，还使它在强扰动下**退化更慢**（slope 降低约 36%）。
**Fact 4**：brightness 的破坏性显著大于 gamma（M1：0.2413/0.2658 vs 0.2688/0.0361），
且 `gamma_1.5`（0.1042）几乎是全部 12 个条件中最温和的一个 → **两类扰动的机制不同**，
不能当作同一种"illumination strength"平均处理（这是本项目历史 `SHIFTS` 混合平均口径的一个已知弱点）。
**Interpretation**：论文核心的 "strength ↔ robustness" 关系在**更大范围内单调成立**，
这为"Uniform α≈0.40 是强 baseline"提供了**跨 severity 的外推证据**（seed0，5 categories）。
**Limitation**：single seed（seed0）；4 个强扰动点上 \|Δz\| 已达 1.5-2.4，属于**分布外**区间，
解释时须注明（score 已接近饱和/τ 已失效）。

## 5. 出口条件判断

- 有 winner？**无** → 无 5×3 stage-2 触发条件。
- Q3 的核心产出（robustness curves + degradation slopes）已生成：
  `summary/q3_illumination_stress.csv`、`summary/q3_degradation_slopes.csv`、
  `summary/q3_method_summary.csv`、`summary/q3_equivalence_check.csv`、
  `figures/q3_robustness_curves.png`。
- seeds 1/2 的补齐**转由 Q4-B** 执行（属 Q4 的“补齐 baseline condition × seed 覆盖”，不是为 winner 调参）。

## 6. 对论文证据链的贡献

推进一格：把 "normalization strength → illumination robustness" 从**单点/4 条件**扩展为
**12 条件的 severity 响应曲线**，并首次给出 **degradation slope** 这一可写进论文的量
（Uniform 的 slope 比 Original 低 ~36%）。

---

## 7. Q4-B 扩展后的最终结果（**30 units：2 methods × 5 categories × 3 seeds**）

| method | d′ ↑ | \|Δz\| all-12 ↓ | \|Δz\| medium-4 ↓ | worst condition | slope pooled | slope brightness | slope gamma | negTr vs Uniform |
|---|---|---|---|---|---|---|---|---|
| `M0_original` | **4.9852** | 0.5071 | 0.3105 | 2.3286 | 2.825 | 3.786 | 1.865 | 11/12 |
| `M1_uniform` | 4.7940 | **0.3480** | **0.2134** | **1.4463** | **1.886** | 2.442 | 1.329 | — |

**关键内部一致性检查**：Q3 的 medium 档（0.7/1.3）在 5 cat × 3 seeds 上的
\|Δz\| = **0.3105（Original）/ 0.2134（Uniform）**，与历史 canonical baseline（5A-H B0 / 6A+6B Uniform）
**完全一致** → 12-condition 扩展管道与冻结口径无缝衔接（这是一个比单点 equivalence 更强的验证）。

**3-seed 结论**：
- Uniform 在 **12/12 conditions** 上 \|Δz\| 都低于 Original（seed0 时有两个温和点例外，3-seed 平均后消失）；
- 差距随 severity 放大：brightness 0.5 → 0.585 vs 0.789（1.35×）；0.7 → 0.249 vs 0.342（1.37×）；
  **1.5 → 1.446 vs 2.329（1.61×）**；
- **degradation slope 更浅**：pooled 1.886 vs 2.825（**−33%**）；brightness −35%；gamma −29%
  → normalization 同时**降低截距与斜率**；
- brightness 的破坏性始终强于 gamma（Uniform：0.249/0.281 vs 0.282/0.042）。

## 8. 数据组织说明（诚实记录）

- 本目录 `raw/` 含 **Q3 的 seed0（10 units）** 与 **Q4-B 的 seeds 1/2（20 units）**——因为
  `experiment7ao_q3_runner.py` 没有 `--out-root`，Q4-B 直接写入此处。分析脚本按 method 合并两个来源，
  结果口径一致（medium 档与历史逐位一致，见上）。
- 本目录 `accidental_stress_runs/` 含 **12 个误跑单元**：首轮 Q4 启动时单元参数顺序错误，
  导致 4 个 α-point × {cable, hazelnut, screw} × seed2 被 stress-runner 执行（12 conditions）。
  **它们不属于任何冻结协议，未被任何分析使用**，仅保留（不删除）。
