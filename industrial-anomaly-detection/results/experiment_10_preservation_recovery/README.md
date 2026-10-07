# Experiment 10 — Preservation Recovery Tournament（Overnight, 2026-10-07/08）

**环境**：AutoDL RTX 3090 24GB / conda `base` / Python 3.12.3 / PyTorch 2.8.0+cu128 / Anomalib 2.6.2
**起始 HEAD**：`569f6d8` (main) ｜ **冻结时刻**：读取任何 target result 之前（`config/config.json` + `scripts/experiment10_candidates.py`）

## 0. 今晚唯一问题

9B-R 已确认：T1 的 robustness gain 是**真实 method effect**（corrected 之前 Δ\|Δz\| = −0.1705），
但 preservation 损失 −0.3544 远超 `EPS_DP = 0.10`。

> **能否在保留 robustness gain 的同时，把 defect preservation 损失补回来？**

方法设计逻辑（论文主线）：
> Robust branch 抑制 illumination nuisance **+** Defect-preserving branch 补偿被 normalization 丢失的信息 **→** 试着突破 robustness–preservation trade-off。

**不研究**「能不能提高 robustness」（已证明），**不做参数 fishing**：每个 family 的参数在读取结果前冻结。

---

## 1. Stage 0 — P0 审计

| 项 | 结果 |
|---|---|
| branch / HEAD | `main` / `569f6d8`（9B-R 清理 commit） |
| git status | 仅用户既有文件：`monitor/exp_monitor.py`(M)、`scripts/_lockprobe.txt`(?)、`scripts/_probe2.py`(?) —— **未删除、未提交** |
| origin/main | 落后本地 2 commit（仓库约定「push 需授权」，未 push） |
| GPU | 空闲（1 MiB / 0%，无 compute app） |
| CPU / RAM / disk | 14 cores / 755 GB（可用 726）/ `/root/autodl-tmp` 45 GB free |
| 其他实验进程 | 无（仅只读 `monitor/exp_monitor.py` watcher，未 kill） |
| detached 机制 | `screen` 可用（所有阶段用 detached screen 会话 + 每-worker 独立进程） |
| 可复用资产 | 9C-v2b（V2 协议）、9B-R（T1/Original-209 raw）、7A-O（concat/residual/energy 算子）、9B（per-layer α、channel gate + train-only s_c 缓存） |

**协议复用**（零重写）：`experiment9c_rng`（R1 `Engine.fit` 入口 + R2 `KCenterGreedy.select_coreset_idxs` 入口 replay）、
`experiment9b_runner` / `experiment7ao_runner`（两个冻结 backend）。**未修改 anomalib / 9B / 7A-O / 5A-H 任何源码。**

## 2. Stage 0.5 — Protocol Audit（209 → 189）＝ **CASE P-A**

9B-R §11 发现：`Engine.fit` 内部再次 `dm.setup("fit")` 会**重置** runner 的 train 过滤 →
memory bank 实际含 **209** 张（含 20 张 val），与注释「validation 不进 memory bank」矛盾。

**修法**（`scripts/experiment10_protocol.py`，最小侵入）：包住 `Engine.fit`，在其内部再次 setup 之后**重新施加**过滤；
并用 `Patchcore.training_step` 计数**证明**实际嵌入图像数。

**证据（bottle seed0）**：`images_embedded_at_fit = 189`（全部 unit）；`filter_log = [(189,189), (209,189)]`（先过滤、被 setup 重置为 209、再重新过滤回 189）。

| method | \|Δz\| 209 | \|Δz\| 189 | Δ | d′ 209 | d′ 189 | Δ | tau 189 | coreset sha 189 |
|---|---|---|---|---|---|---|---|---|
| Original α=0 | 0.5027 | **0.4769** | −0.0258 | 8.5328 | **8.0440** | −0.4888 | 30.0140 | 与 209 不同（协议不同，属预期） |
| T1 (L2 0, L3 0.25) | 0.3322 | **0.3789** | +0.0467 | 8.1784 | **7.9036** | −0.2748 | 30.5179 | 同上 |

**T1 − Original**：209 时 `Δ|Δz| = −0.1705 / Δd′ = −0.3544`；189 时 `Δ|Δz| = −0.0980 / Δd′ = −0.1404`
→ **方向完全一致**（robustness 改善、preservation 下降），**trade-off 判断不变**（robustness 改善 ≫ `EPS_RZ`；preservation 仍越界）
→ **CASE P-A：从 Experiment 10 起正式冻结 corrected-189。**

> **重要含义**：corrected 协议下需要补回的 preservation 缺口只有 **0.040 d′**（−0.1404 → −0.10），
> 而 T1 仍有 0.078 的 robustness 余量可让渡 —— 今晚目标比 209 协议下**更可达**。

## 3. Stage 1 — Corrected V2 Reference Frontier（bottle seed0）

`Strict Original-189`（α=0）：**\|Δz\| = 0.4769 / d′ = 8.0440 / tau = 30.0140**。
uniform-α 工作点严格复用**历史冻结** grid（**未新增任何临时 α 值**）：

| α | \|Δz\| | d′ | Δ\|Δz\| | Δd′ |
|---|---|---|---|---|
| 0 (Original) | 0.4769 | 8.0440 | 0.0000 | 0.0000 |
| 0.125 | 0.3455 | 7.7687 | −0.1314 | −0.2753 |
| 0.25 | 0.3257 | 7.7337 | −0.1511 | −0.3103 |
| 0.50 | 0.2522 | 7.0319 | −0.2247 | −1.0121 |
| 0.601369125 | 0.2429 | 6.9123 | −0.2339 | −1.1317 |
| 0.8018255 | 0.2314 | 6.5084 | −0.2455 | −1.5355 |
| 0.2 | 0.3472 | 7.6006 | −0.1297 | −0.4434 |
| 0.3 | 0.3210 | 7.3342 | −0.1559 | −0.7098 |
| 0.40091275 | 0.2598 | 7.2823 | −0.2171 | −0.7617 |

（完整 9 点见 `reference_frontier/frontier.csv`；frontier envelope 见 `frontier.json`）

**uniform-α frontier 的特征：每换取 0.1 的 \|Δz\|，要付出 0.5–1.5 的 d′。**
例外与今晚的关键参照：**T1-189 落在 corrected uniform frontier 之上 `beyond = +0.0649`**
（per-layer α 本身已经比 uniform 好一点）。

## 4. Stage 2 — 候选设计（去重后 **9 unique / 12 raw ideas**）

**去重（CPU 代数证明，未消耗 GPU）**：
- **Family A（Residual Feature Recovery）`F = F_R + λ(F_O−F_R)` 全部 3 个成员 → `EQUIVALENT`**：
  `F_R = (1−α)F_O + αIN` ⇒ `F_R + λ(F_O−F_R) = (1−α(1−λ))F_O + α(1−λ)IN`，即 **α′ = α(1−λ) 的 per-layer α**，
  属已充分探索的 M7 α-family（λ=0.25/0.50/0.75 ↔ L3 α′ = 0.1875/0.125/0.0625）。记录于 `candidates/equivalence_dedup.json`。
- Family B1/B2 与 7A-O Family B / M10 **不**等价（M10 拼标准化副本；这里拼 robust 分支，且保留 Original block）。

**统一结构约束**：所有候选 **L2 保持 Original**（9B-R 证据：动 L2 掉 preservation），只在 L3 或 L2 的 *补偿支路* 上做文章。
全部 normal-only / deployable / training-free；无 oracle、无 test/defect 数据、参数冻结。

| family | id | 定义 |
|---|---|---|
| B | `P10_B1_L3concat_robust_a025` | L3 = `cat([F, 0.75F+0.25·IN(F)])` |
| B | `P10_B2_L3concat_rscale_a025` | L3 = `cat([F, 0.75F+0.25·IN(F)·rms(F)])` |
| C | `P10_C2_L2resid025_L3a025` | L2 = `F+0.25·IN(F)·rms(F)`（scale-matched residual）；L3 = `0.75F+0.25·IN(F)` |
| C | `P10_C3_L2dual_L3a025` | L2 = `cat([F, IN(F)·rms(F)])`（dual representation）；L3 = `0.75F+0.25·IN(F)` |
| D | `P10_D1_L3m9pow_g1` | L2 identity；L3 per-channel gate `w_c=(s_c/p90)^1`（train-only s_c） |
| D | `P10_D2_L3m8z_ab050_b100` | L2 identity；L3 gate `a_c=clip(0.5+1.0·z_c)`（train-only z-map） |
| D | `P10_D3_L3m8z_ab025_b050` | L2 identity；L3 gate `a_c=clip(0.25+0.5·z_c)` |
| E | `P10_E1_L3energy025` | L2 identity；L3 = `(F+0.25·IN(F)·rms)·rms(F)/rms(mix)`（能量保持） |
| E | `P10_E2_L3energy050` | 同上，λ=0.50 |

## 5. Stage 2A — Bottle × seed0 快筛（corrected-189 + strict V2，9/9 完成，0 failed）

参考：`Original-189` \|Δz\|=0.4769 / d′=8.0440；`T1-189` \|Δz\|=0.3789 / d′=7.9036。

| Rank | Candidate | Family | \|Δz\| | d′ | Δ\|Δz\| vs Orig | **Δd′ vs Orig** | Δd′ vs T1 | beyond frontier | Verdict |
|---|---|---|---|---|---|---|---|---|---|
| 1 | `P10_C3_L2dual_L3a025` | C | 0.2580 | 7.9905 | **−0.2189** | **−0.0535** | **+0.0869** | **+0.7674** | **ADVANCE** |
| 2 | `P10_C2_L2resid025_L3a025` | C | 0.2656 | 7.9787 | **−0.2113** | **−0.0653** | **+0.0751** | **+0.6916** | **ADVANCE** |
| 3 | `P10_D3_L3m8z_ab025_b050` | D | 0.2643 | 7.8383 | −0.2126 | −0.2057 | −0.0653 | +0.5522 | HOLD |
| 4 | `P10_D1_L3m9pow_g1` | D | 0.3037 | 7.7378 | −0.1731 | −0.3062 | −0.1658 | +0.4182 | HOLD |
| 5 | `P10_D2_L3m8z_ab050_b100` | D | 0.3019 | 7.7079 | −0.1749 | −0.3361 | −0.1957 | +0.3898 | HOLD |
| 6 | `P10_E1_L3energy025` | E | 0.4184 | 7.7602 | −0.0585 | −0.2838 | −0.1434 | −0.1612 | HOLD |
| 7 | `P10_B2_L3concat_rscale_a025` | B | 0.4548 | 7.5552 | −0.0221 | −0.4887 | −0.3483 | −0.4425 | HOLD |
| 8 | `P10_E2_L3energy050` | E | 0.5070 | 7.8820 | +0.0301 | −0.1620 | −0.0215 | −0.1620 | STOP |
| 9 | `P10_B1_L3concat_robust_a025` | B | 0.4627 | 7.5571 | −0.0141 | −0.4868 | −0.3464 | −0.4572 | STOP |

**核心发现（Family C 双赢）**
- **C2/C3 同时做到两件事**：robustness gain **比 T1 更大**（−0.219/−0.211 vs T1 −0.098），
  且 **preservation 反而比 Original 只差 0.054/0.065（在 `EPS_DP=0.10` 带内）**，比 T1 **好 0.075–0.087 d′**。
- 两者都**远高于** corrected uniform-α frontier（+0.69 / +0.77 d′）→ 标记 **FRONTIER-BREAK CANDIDATE**。
- 机制解读（可解释、非调参）：**在 L2 上加一个 preservation 补偿支路**（dual 或 scale-matched residual）把
  normalization 抑制掉的 defect-sensitive 信息在**另一条通道**里补回来，而 L3 继续提供 robustness —— 
  即「Robust Representation + Defect-Preserving Information → Fusion」在**层级维度**上成立。
- **失败结构**：Family B（在 L3 拼接 robust 分支）**几乎完全失效**（robustness gain 消失、preservation 剧烈下降）——
  在**已被 normalization 污染的层**上加分支等于把污染复制一遍；
  Family E（能量保持）在 L3 恢复能量**不解决方向丢失**，λ=0.5 甚至使 robustness 变差（+0.030）→ STOP。
  Family D（channel gate）介于两者之间（+0.39~+0.55 beyond frontier，但 preservation 仍越界）。

## 6. Stage 2B — Bottle × seeds {0,1,2}（Top3，0 failed）

（聚合规则在读取 seeds 1/2 结果前写定于 analysis 代码；判据带沿用 Stage 2A 冻结值）

| Candidate | Verdict | mean Δ\|Δz\| | std | **mean Δd′** | std | **worst Δd′** | beyond(s0) |
|---|---|---|---|---|---|---|---|
| `P10_C3_L2dual_L3a025` | **ADVANCE** | −0.1911 | 0.0299 | **+0.1351** | 0.2381 | **−0.0535** | +0.7674 |
| `P10_C2_L2resid025_L3a025` | **ADVANCE** | −0.1902 | 0.0150 | **+0.0324** | 0.1245 | **−0.0653** | +0.6916 |
| `P10_D3_L3m8z_ab025_b050` | HOLD | −0.2073 | 0.0039 | −0.2436 | 0.1056 | −0.3877 | +0.5522 |

**逐 seed 明细（Δ vs 该 seed 的 corrected Original）**

| Candidate | seed0 Δd′ | seed1 Δd′ | seed2 Δd′ | 3-seed 同方向 |
|---|---|---|---|---|
| C3 | −0.0535 | **+0.4709** | −0.0122 | ✓ |
| C2 | −0.0653 | **+0.2081** | −0.0458 | ✓ |

- **C2/C3 的 preservation 损失在 3 个 seed 上全部 ≥ −0.0653**（worst seed 也远好于 `−0.10` 门槛），
  同时 robustness 改善 **3/3 seed 一致**（−0.150 ~ −0.219）→ 满足论文级成功条件。
- D3 在 seed2 出现 −0.3877 的 collapse → HOLD（不救）。

## 7. Stage 3 — 跨类别验证（5 categories × seed0，0 failed）

规则（冻结）：每类 pass = `Δ|Δz| ≤ −EPS_RZ 且 Δd′ ≥ −EPS_DP`；`ADVANCE` = wins ≥ 3/5 且**无 catastrophic**（`Δd′ ≤ −0.25`）。

| Method | bottle | cable | hazelnut | screw | grid | Wins | Verdict |
|---|---|---|---|---|---|---|---|
| **`C2_L2resid025_L3a025`** | −0.2113 / −0.0653 | +0.0247 / −0.0696 | −0.0210 / **+0.0211** | −0.1073 / **+0.0962** | −0.1068 / **+0.4978** | **4/5** | **ADVANCE** |
| `C3_L2dual_L3a025` | −0.2189 / −0.0535 | −0.0160 / **−0.4298**✗ | −0.0094 / **−0.4640**✗ | −0.0524 / +0.0702 | −0.0684 / **+0.9965** | 3/5 | HOLD |

（每格 = `Δ|Δz| / Δd′` vs 该类别 corrected Original；✗ = catastrophic collapse）

**关键结论**
- **C2（L2 scale-matched residual + L3 α=0.25）跨类别成立**：4/5 类满足预注册条件，
  无 catastrophic；mean Δ|Δz| = −0.084、**mean Δd′ = +0.096**（五项平均为**正**）。
  弱项是 cable（robustness 未改善 +0.025，preservation −0.070）。
- **C3（L2 dual representation）跨类别不成立**：cable/hazelnut 出现 −0.43/−0.46 的 preservation 崩溃
  （虽在 screw/grid 上极强，grid Δd′ = +0.9975）→ HOLD。
  → 说明「L2 复制成双通道」把 L2 的 defect 信息**稀释**了，在 defect 结构不同的类别上代价很大；
  而「L2 加 scale-matched residual」保留了单通道语义、只补回被抑制的分量，因而更稳健。
- **Stage 4 触发条件满足**（Stage 3 有 1 个明确 ADVANCE = C2）→ 对 **Top 1 = C2** 补 `4 categories × seeds {1,2}`（+ 对应参考点）。

## 8. Stage 4 — Top1 (C2) × 5 categories × seeds {0,1,2}（15 units，0 failed）

| category | seed0 ΔR / ΔP | seed1 ΔR / ΔP | seed2 ΔR / ΔP | mean ΔR | mean ΔP | worst ΔP | pass |
|---|---|---|---|---|---|---|---|
| bottle | −0.2113 / −0.0653 | −0.1820 / **+0.2081** | −0.1774 / −0.0458 | −0.1902 | +0.0324 | −0.0653 | **3/3** |
| screw | −0.1073 / +0.0962 | −0.0801 / +0.2081 | −0.1020 / +0.1656 | −0.0965 | **+0.1567** | +0.0962 | **3/3** |
| grid | −0.1068 / **+0.4978** | −0.0200 / +0.6771 | −0.0779 / +0.4482 | −0.0682 | **+0.5410** | +0.4482 | 2/3 |
| hazelnut | −0.0210 / +0.0211 | +0.0085 / +0.2444 | −0.0853 / −0.2208 | −0.0326 | +0.0149 | −0.2208 | 1/3 |
| cable | +0.0247 / −0.0696 | −0.0188 / +0.0614 | −0.1249 / **−0.3286** | −0.0397 | −0.1123 | **−0.3286** | **0/3** |

**15/15 units 完成；9/15 (cat,seed) 组合满足预注册条件；4/5 类别（按均值）满足；cable 是明确弱侧（含 1 次 catastrophic）。**

- **bottle × 3 seeds：全部满足**（worst ΔP = −0.0653，robustness 改善 −0.177 ~ −0.211）→ frontier-break 在 bottle 上稳定。
- **screw × 3 seeds：全部满足**；**grid：2/3**（seed1 的 ΔR = −0.0200 恰好卡在门槛上）。
- **hazelnut：1/3**（seed1 robustness 未改善、seed2 preservation −0.22）。
- **cable：0/3** 且 seed2 出现 ΔP = −0.3286 的 collapse。
  → 与项目既有结论一致：**cable 是 channel/层级级方法的系统性弱侧**（5A-H / 7A-O / 9B / 9B-R 均报告过），
    不是本次新引入的问题。

## 9. 最终排名（`final_ranking.csv`）

| Rank | Method | Family | \|Δz\| (bottle s0) | d′ (bottle s0) | Δ\|Δz\| | Δd′ | 3-seed mean Δd′ | Seeds | Cats | Verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| **1** | **`C2_L2resid025_L3a025`** | C | 0.2656 | 7.9787 | **−0.2113** | **−0.0653** | **+0.0324** | 3 | **4/5** | **ADVANCE** |
| 2 | `D2_L3m8z_ab050_b100` | D | 0.3019 | 7.7079 | −0.1749 | −0.3361 | – | 1 | bottle | HOLD |
| 3 | `D1_L3m9pow_g1` | D | 0.3037 | 7.7378 | −0.1731 | −0.3062 | – | 1 | bottle | HOLD |
| 4 | `D3_L3m8z_ab025_b050` | D | 0.2643 | 7.8383 | −0.2126 | −0.2057 | −0.2436 | 3 | bottle | HOLD |
| 5 | `C3_L2dual_L3a025` | C | 0.2580 | 7.9905 | −0.2189 | −0.0535 | **+0.1351** | 3 | 3/5 | HOLD |
| 6 | `B2_L3concat_rscale_a025` | B | 0.4548 | 7.5552 | −0.0221 | −0.4887 | – | 1 | bottle | HOLD |
| 7 | `E1_L3energy025` | E | 0.4184 | 7.7602 | −0.0585 | −0.2838 | – | 1 | bottle | HOLD |
| 8 | `B1_L3concat_robust_a025` | B | 0.4627 | 7.5571 | −0.0141 | −0.4868 | – | 1 | bottle | STOP |
| 9 | `E2_L3energy050` | E | 0.5070 | 7.8820 | +0.0301 | −0.1620 | – | 1 | bottle | STOP |
| — | `A1/A2/A3`（residual recovery） | A | – | – | – | – | – | 0 | 0 | **EQUIVALENT（未跑 GPU）** |

## 10. 淘汰 / 负面结果（negative results）

| 被淘汰 | 原因 |
|---|---|
| **Family A（全部）** | **代数等价**于 per-layer α（`α′=α(1−λ)`）→ 已在 M7 探索过，无新结构 |
| **Family B（B1/B2）** | 在**已被 normalization 污染的 L3** 上加 robust 分支：robustness gain 消失（−0.014/−0.022）、preservation 剧烈下降（−0.49）→ B1 判 STOP |
| **Family E（E1/E2）** | L3 的**能量**恢复不解决**方向**丢失；E2 使 robustness 反而变差（+0.030）→ STOP |
| **Family D（D1–D3）** | 有真实越界（+0.39~+0.55）但 preservation 仍越界（−0.21~−0.34）；D3 在 seed2 出现 −0.388 collapse → HOLD，不救 |
| **C3** | bottle 上最好（3 seeds 全部 ≥−0.065、beyond +0.767），但 cable/hazelnut preservation **崩溃**（−0.43/−0.46）→ HOLD |
| `T1`（9B-R 最优） | 保留为参照：corrected-189 下 Δ\|Δz\|=−0.098 / Δd′=−0.140（越界），beyond frontier 仅 +0.065 |

## 11. 关键机制解读（可解释、非调参）

- **有效的位置**：把 preservation 补偿放在 **L2（未被 normalization 破坏的层）**。
  `C2 = L2 上加 scale-matched residual（单通道语义不变）+ L3 继续 robust` → 双赢。
- **无效的位置**：在同一层（L3）上**事后**加分支（B）或恢复能量（E）——污染已经发生，补偿无效甚至有害。
- **C2 vs C3 的差别**：C3 把 L2 **复制成双通道**（`cat([F, IN(F)·rms])`），在 bottle/grid 上极强（grid Δd′ = +0.997）
  但在 cable/hazelnut 上**稀释了 L2 的 defect 信息** → 崩溃。C2 只**加**一个被抑制的分量，保持单通道语义 → 更稳健。
- **cable 弱侧的既有模式**：本项目中所有 channel/层级级操作在 cable 上都偏弱 → 指向 L3 通道选择在 cable 上的结构性问题。

## 12. 下一步（仅建议，未启动）

1. **C2 进入正式验证的第一优先级**，但**必须先在 cable 上解决**（当前 0/3，含一次 −0.33 collapse）：
   需要机制性理由，不允许调参 fishing。
2. 最小后续实验（建议）：`C2 × cable × seeds {3,4}` 判定 cable 的 −0.33 是偶发还是结构性；
   并对 C2 的 L2 residual 强度给出**训练集统计驱动**的（非 target）自适应选择规则。
3. **不要**直接进入 5×3 的论文级表格（cable 未过）。
