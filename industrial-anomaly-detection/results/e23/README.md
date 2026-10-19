# E23 — Doll Failure-Case Descriptive Audit

**日期**：2026-10-16（只读分析，0 GPU；单次运行 ≈1 min，远低于 30–90 min 预算）
**论文阶段**：`Paper Validation → Limitation Analysis`（方法搜索 CLOSED、R1-ILA 冻结不变）
**上游**：E22 归档 commit `3d09b02`（`results/e22/`，verification 36/36）
**执行**：`scripts/e23_doll_audit.py`（分析 A–F）→ `scripts/e23_figures.py`（Fig1–Fig5）
**Verdict**：**EXPLAINED_PATTERN**（预指定规则机械应用，3/3 seeds；含量级限定，见 §8）

---

## 0. 四句话定位

| 项 | 内容 |
|---|---|
| 📍 论文阶段 | Paper Validation / Limitation Analysis —— 为已冻结方法划定适用边界 |
| 🎯 服务论断 | *R1-ILA shows broadly positive category-level behavior on M²AD, but its benefit is not universal; Doll is a reproducible failure case that helps define the method's applicability boundary.* |
| 🧪 为什么非做不可 | Doll 是 3/3 seeds 一致、CI 明确低于 0（[−0.0661, −0.0140]，bootstrap 正差比例 P(ΔAUROC>0) = 0.001）的失败类，需要判断它属于"光照偏移弱 ⇒ 方法不适用"、"R1 压低了异常可分性"、"正常/异常分数一起移动"还是"现有指标无法解释" |
| 🚦 出口条件 | 完成冻结的描述性分析即 **STOP**：**无论能否解释 Doll，都不进入任何方法修改**（无重训、无调参、无 Doll-specific 处理） |

**硬性禁止（本实验全程遵守）**：改 BETA / BETA sweep / 改 LR / 改 augmentation / 改 preserve loss / 改 adapter / 改 backbone / 改 scoring / 改 bank / 换 seed / seed3+ / Doll-specific 训练或超参 / 新增模块 / 为 Doll 设计新 loss。**只读已有 scores / features / metadata / logs，0 GPU。**

## 1. Motivation

E22 在 M²AD 10 类 × 3 seeds × view120 上给出 9/10 类 mean ΔAUROC > 0 的总体正结果，唯一稳定负类为 Doll（0/3 seeds、mean −0.03850、bootstrap 显著）。这不是随机波动：三 seed 同向、单一 bank 变化不足以解释。E23 的唯一科研问题是：

> **Why does the frozen R1-ILA show a stable degradation on Doll while 9/10 M²AD categories show positive mean ΔAUROC?**

E23 **不负责修复 Doll**，只负责描述失败发生在哪里，并检验**已有冻结指标**能否解释。

## 2. Frozen scope

- **无训练、无调参、无方法改动**；不产生任何新模型、新分数、新 GPU 计算。
- 只读：`results/e22/cells/*.json`（逐图 F0/R1 分数与冻结 diagnostics）、`main_table_10cat_3seed.csv`、`illumination_breakdown.csv` / `illumination_spread.csv` / `illumination_summary.csv`、`shift_gain_association.json`、M²AD 官方元数据（`image_anomaly` + `detectable != False`）。
- **复用冻结指标定义，不新造指标**：`d′`（cell 内存值）、`D_illum_shift_F0`（E16/E18/E19 冻结定义）、illumination spread = max(good_mean) − min(good_mean)（E22 定义）、AUROC/AP（official 协议）。
- 交叉核对：`doll_illumination_breakdown.csv` 的 AUROC/AP 与冻结 `results/e22/illumination_breakdown.csv` 的 Doll 行在 **1e-12 内逐格一致**（脚本内 assert，33 行全部通过）。
- **预指定 verdict 规则**（在查看本次审计数字之前写入脚本）：`COND_A(seed) = (d′_R1 < d′_F0) AND (anomaly 均值下降量 > good 均值下降量)`；3/3 → `EXPLAINED_PATTERN`，2/3 → `PARTIALLY_EXPLAINED`，否则 `UNEXPLAINED_FAILURE`。
- 分析维度在运行前固定为 A–F 六项，**不做指标挖矿**（不相关就换下一个）；若现有指标无法解释，允许写 `UNEXPLAINED_FAILURE`。

## 3. Seed-level evidence（Analysis A，`doll_seed_summary.csv`）

| seed | F0 AUROC | R1 AUROC | ΔAUROC | F0 AP | R1 AP | ΔAP | F0 d′ | R1 d′ | Δd′ | bank hash (前 10) |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 0.71941 | 0.67556 | **−0.04385** | 0.80755 | 0.79808 | −0.00947 | 0.80832 | 0.71010 | −0.09822 | `46a03f7ed3` |
| 1 | 0.71306 | 0.67719 | **−0.03588** | 0.80318 | 0.79562 | −0.00756 | 0.78796 | 0.71585 | −0.07210 | `b60e2ca632` |
| 2 | 0.70504 | 0.66928 | **−0.03576** | 0.80981 | 0.78778 | −0.02203 | 0.78505 | 0.65098 | −0.13407 | `d4efeaf9f3` |
| mean | 0.71250 | 0.67401 | **−0.03850** | 0.80685 | 0.79383 | −0.01302 | 0.79377 | 0.69231 | −0.10146 | — |
| std | 0.00720 | 0.00418 | 0.00464 | 0.00337 | 0.00538 | 0.00786 | 0.01268 | 0.03591 | 0.03111 | — |
| min / max | 0.70504 / 0.71941 | 0.66928 / 0.67719 | −0.04385 / −0.03576 | 0.80318 / 0.80981 | 0.78778 / 0.79808 | −0.02203 / −0.00756 | — | — | — | — |

> **勘误（2026-10-19）**：此前本表的 std 行（0.00588 / 0.00341 / 0.00379 / …）与 min 行首格（0.00588）由 `analysis_a` 的汇总污染造成——汇总行被追加进 `rows` 后又被下一次统计读取，std 实际是 4 个值（3 seed + mean 行）的 std，min 又读到了该 std。现改为只从三条原始 seed 记录计算、`ddof=1`；`doll_seed_summary.csv` 已重生成。正确值：std(ΔAUROC) = **0.004638**（原误 0.0037868）、min(F0 AUROC) = **0.705037**（原误 0.00588）。原始三行未改动。

- **3/3 seeds 同方向**：AUROC、AP、d′ 三个指标全部下降；n_good = 270、n_anomaly = 400（detectable 过滤 30 条）三 seed 完全相同，标签来源 = `meta_unsupervised.json:image_anomaly`（其中 70 条位于 `NG/` 目录但 `image_anomaly = 0`，已按元数据判类）。
- 三 seed 使用**不同 bank**（hash 不同）、不同 adapter，得到同一结论 ⇒ 不是 bank/seed 偶然。

## 4. Score decomposition（Analysis B/C，`doll_score_distribution.csv`、`doll_separation_summary.csv`、`doll_per_image_delta.csv`）

### 4.1 分布统计（kept 图像，逐 seed）

| seed | good mean F0→R1 | anomaly mean F0→R1 | raw separation F0→R1 | pooled SD F0→R1 | d′ F0→R1 |
|---|---|---|---|---|---|
| 0 | 0.26405 → 0.18751 | 0.30679 → 0.22032 | 0.04274 → 0.03281 | 0.05124 → 0.04539 | 0.80832 → 0.71010 |
| 1 | 0.26638 → 0.18132 | 0.31062 → 0.21627 | 0.04424 → 0.03495 | 0.05432 → 0.04819 | 0.78796 → 0.71585 |
| 2 | 0.26844 → 0.18462 | 0.31144 → 0.21445 | 0.04300 → 0.02984 | 0.05286 → 0.04435 | 0.78505 → 0.65098 |

### 4.2 逐图配对变化（`Δscore = R1 − F0`，kept）

| seed | good 分数下降 | anomaly 分数下降 | good Δ 中位数 | anomaly Δ 中位数 | good 秩改善比例 | anomaly 秩改善比例 |
|---|---|---|---|---|---|---|
| 0 | **270/270 (100%)** | **400/400 (100%)** | −0.0728 | −0.0809 | 0.615 | 0.563 |
| 1 | **269/270 (99.6%)** | **400/400 (100%)** | −0.0833 | −0.0925 | 0.574 | 0.523 |
| 2 | **270/270 (100%)** | **400/400 (100%)** | −0.0829 | −0.0931 | 0.570 | 0.528 |

**核心要回答的问题（逐条回答）**：

1. *R1 是不是把 good scores 降低了，但 anomaly scores 降得更多？* —— **是，但幅度几乎对称**。绝对下降量：anomaly 比 good 多降 0.0093–0.0132；**相对下降比例**（anomaly/good）= 0.972 / 0.951 / 0.997（三 seed 全部 ≥ 0.95）→ **不是"anomaly 大规模 compression"**，而是近似等比例的整体下移 + 轻微不对称。
2. *good score drift 改善，但 anomaly separation 同时受损？* —— 部分成立：spread 层面 good 侧确实被压（R1/F0 ≈ 0.78/0.77/0.82），但 **anomaly 侧同样被压**（0.91/0.92/0.84），因此不是"只压正常侧"的鲁棒性改善。
3. *两边都发生变化？* —— **是**（100% 的图像分数都下降），且标准化可分性 d′ 在 3/3 seeds 下降（−0.072 … −0.134）。

> **尺度口径提示（必须随结论一起写）**：R1-ILA 改变特征尺度，F0 与 R1 的**绝对分数水平不可直接比较**；上表 raw separation / pooled SD 的比例是"同一指标内部的相对变化"，而**真正尺度无关的证据**是 AUROC / AP / d′ 三者在 3/3 seeds 一致下降。raw separation 比例（0.69–0.79）小于 pooled SD 比例（0.84–0.89），因此 d′ 比例为 0.83–0.91 —— 这是失败在分数层面的**定量分解**，不是新的指标。

## 5. Illumination breakdown（Analysis D，`doll_illumination_breakdown.csv`）

严格使用冻结的 illumination group 定义（`01`–`10`，10 组）：

| seed | ΔAUROC < 0 的组数 | 最差组 | 最差 ΔAUROC | 最好组 | 最好 ΔAUROC |
|---|---|---|---|---|---|
| 0 | **9/10** | 02 | −0.09485 | 05 | +0.00452 |
| 1 | **8/10** | 02 | −0.10117 | 08 | +0.04815 |
| 2 | **8/10** | 09 | −0.07769 | 01 | +0.02803 |

- 失败**不是集中在少数 illumination group**：每 seed 有 8–9 个组为负（≥7/10 的"broad"描述子成立），组间 ΔAUROC 跨度约 −0.10 … +0.05。
- 每个组内 good/anomaly 的 mean/std 明细（含 ALL 行）见 CSV；与 E22 冻结 `illumination_breakdown.csv` 的 Doll 行逐格 1e-12 一致。
- **本分析的目的不是找"最差 illumination 去调参"**，只是判断失败是否局限。

## 6. Cross-category comparison（Analysis E，`category_comparison.csv`、`doll_vs_others.json`）

Doll 与另外 9 类在冻结维度上的位置（rank 1 = 10 类中最低值；z 为 Doll vs 其余 9 类的描述性标准化差）：

| 维度 | Doll | 其余 9 类 mean ± sd | z | rank |
|---|---|---|---|---|
| mean ΔAUROC | **−0.03850** | +0.01272 ± 0.00772 | **−6.64** | **1 (min)** |
| mean ΔAP | **−0.01302** | +0.01008 ± 0.00638 | −3.62 | 1 (min) |
| F0 AUROC | **0.71250** | 0.85285 ± 0.05076 | −2.76 | **1 (min)** |
| R1 AUROC | **0.67401** | 0.86556 ± 0.04787 | −4.00 | 1 (min) |
| F0 AP | 0.80685 | 0.90363 ± 0.05023 | −1.93 | 2 |
| R1 AP | **0.79383** | 0.91372 ± 0.04573 | −2.62 | 1 (min) |
| F0 d′ | **0.79377** | 1.43168 ± 0.28429 | −2.24 | **1 (min)** |
| R1 d′ | **0.69231** | 1.48869 ± 0.28701 | −2.77 | 1 (min) |
| Δd′ | **−0.10146** | +0.05702 ± 0.05770 | −2.75 | 1 (min) |
| **D_shift_F0（光照偏移）** | 0.18235 | 0.18640 ± 0.03031 | **−0.13** | 4 |
| B_centroid_F0 | 0.04851 | 0.04222 ± 0.01258 | +0.50 | 8 |
| spread_F0（正常分数跨光照 spread） | 0.01678 | 0.03591 ± 0.01488 | −1.29 | 2 |
| spread_R1 | 0.01204 | 0.02403 ± 0.01116 | −1.07 | 1 (min) |
| spread ratio R1/F0 | 0.71760 | 0.73277 ± 0.32625 | −0.05 | 5 |
| C_good_ng_centroid_dist F0 / R1 | 0.00204 / 0.00032 | 0.00148 / 0.00033 | +0.73 / −0.04 | 7 / 7 |
| E_adapt_centroid_drift | 0.17534 | 0.16479 ± 0.04570 | +0.23 | 7 |
| A_pairwise_cos F0 / R1 | 0.89367 / 0.97692 | 0.91431 / 0.98062 | −0.92 / −0.44 | 3 / 4 |
| ΔAUROC 跨 seed std | 0.00464 | 0.00709 ± 0.00441 | −0.56 | 4 |

**Doll 在什么维度上与 9 个正类不同（事实）**：
- 它在**任务难度与可分性**维度上极端：F0 AUROC / AP 最低（0.7125 / 0.8069），**标准化可分性 d′ 最低且是唯一 < 1 的类别**（0.794；其次低的 Bird 为 0.926，其余均 ≥ 1.08），Δd′ 也是唯一明显负值（−0.101，其余 9 类 mean +0.057）。
- 它在**光照/适配机理**维度上**并不特殊**：D_shift 排第 4（z = −0.13，几乎正好位于 10 类中位数 0.1835）、B_centroid 排第 8、spread ratio 排第 5、适用性类诊断（C/E/A）排 3–8，跨 seed 波动排第 4。**没有 collapse 迹象**（A_pairwise_cos R1 = 0.977 与其余类同量级）。
- 因此"低 baseline 有 headroom 所以应该涨"的 ceiling-effect 解释（E13-B2 在 Cube 上的结论）**在低端不成立**：Doll 是 10 类中 baseline 最低、headroom 最大者，却得到最差结果。

## 7. Shift–gain relationship（Analysis F，复用 E22 冻结 shift 指标）

| 口径 | Pearson r (p) | Spearman ρ (p) |
|---|---|---|
| **full n = 10（主口径，与 E22 冻结值逐位一致）** | **+0.371 (0.291)** | **+0.576 (0.082)** |
| 仅作 sensitivity：leave-one-out（去掉 Doll，n = 9） | +0.807 (**0.008**) | +0.600 (0.088) |

- **Doll 是弱正 shift–gain 关联中的明显 outlier**：gain rank 1/10（最低），shift rank 4/10（中位附近），相对 full-n=10 线性拟合的残差为 **−0.0453**。
- 报告规则：**主口径永远是 full n = 10**；LOO 只作 sensitivity 一并报告，不据其改写任何结论（也不做"去掉 Doll 后更好看"的表述）。
- n = 10 → 只能写 exploratory / supportive association，**禁止因果表述**。

## 8. What explains Doll?

### FACT（数据直接观察）
1. 3/3 seeds：AUROC ↓（−0.0358 … −0.0439）、AP ↓（−0.0076 … −0.0220）、d′ ↓（−0.0721 … −0.1341）。
2. 3/3 seeds：**几乎全部图像的分数都下降**（good 269–270/270、anomaly 400/400）；anomaly 侧绝对下降量略大于 good 侧，但**相对下降比例几乎对称**（0.951–0.997）。
3. raw separation 缩到 F0 的 0.69–0.79，同时 pooled SD 缩到 0.84–0.89 → d′ 比例为 0.83–0.91；秩层面 anomaly 的秩改善比例（0.52–0.56）低于 good（0.57–0.62），与 AUROC 下降一致。
4. 失败在 illumination 上**广泛存在**（每 seed 8–9/10 组 ΔAUROC < 0），非局部现象。
5. Doll 的**光照偏移不特殊**（rank 4/10，≈ 中位数），spread 缩减比例也与其余类相当（rank 5/10）；适配类诊断无 collapse。
6. Doll 是 10 类中 **F0 baseline 最低、d′ 最低（唯一 < 1）** 的类别，Δd′ 唯一明显为负。
7. Doll 是 shift–gain 弱正关联中的 outlier（残差 −0.0453）。

### INTERPRETATION（基于事实的推断，非因果）
- 该失败是一个**可重复、广泛分布、幅度温和的可分性损失**，而不是异常信息的崩塌：两侧分数近似等比例下移，只有轻微不对称，但足以在秩层面把 AUROC 压低 ~0.036–0.044。
- **光照偏移不是解释**：Doll 的 D_shift 位于中位数附近，且全域方向（shift 越大增益略大）与"高 shift 导致失败"相反 —— 用 §7 的关联无法解释 Doll。
- **ceiling effect 也不是解释**：Doll 是 headroom 最大者却最差，说明 E13-B2 的"低/中 baseline 才有收益"经验外推到极低 baseline 时失效。
- 更符合数据的描述是：R1-ILA 的 normal-only illumination-consistency 目标在**本来 contrast 就很弱**的类别（d′ < 1）上，把正常流形收得更紧的同时也让弱对比的异常样本的相对位置略微回退。

### HYPOTHESIS（未经实验验证，**不得写成结论**）
- H1：Doll 的缺陷在 view120 下本身低对比（d′ 0.79），adaptation 对 normal 流形的重塑在这种"弱对比"区域的相对影响更大 → 需要新的机制实验（例如按缺陷对比度分层的诊断）才能检验，**E23 不做，也不批准**。
- H2：Doll 的异常外观与该类别被光照增强轴扰动最多的外观因素重叠，导致 consistency 目标把部分异常方向也纳入"应保持不变"的子空间。
- H3：以上都不成立，失败可能由 10 类中 Doll 特有的数据/标注结构（70 条 `NG/` 目录但标 0 的记录）间接引起 —— 目前**没有任何证据**支持或否定。

> **允许的最终结论（含量级限定）**：现有冻结诊断给出了一个**一致但幅度温和**的分数侧模式（anomaly 侧下降略多于 normal 侧、d′ 三 seed 下降、失败广泛分布于各 illumination），因此判定为 **EXPLAINED_PATTERN**；同时必须写明 **相对下降几乎对称（0.95–1.00）、绝对不对称量级仅 0.009–0.013**。脚本内另附透明度声明：若事后加一条 effect-size 阈值（相对下降比 ≤ 0.90），verdict 会变为 PARTIALLY_EXPLAINED —— 该阈值**不属于**预指定规则，未采用，仅记录（见 `e23_verdict.json: verdict_sensitivity_disclosure`）。

## 9. What E23 cannot establish

- **No causal proof**：任何"因为 X 所以 Doll 下降"的因果陈述都不成立；§7 的关联为 exploratory，§8 的 H1–H3 均为假设。
- **No Doll repair**：E23 未训练、未调参、未改方法，也未说明如何修复 —— 修复不在本实验范围内。
- **No universal explanation**：结论仅针对 M²AD Doll / view120 / DINOv2-B / 冻结 R1-ILA；不能外推到其他类别、视角、backbone 或数据集。
- **No guarantee of mechanism**：EXPLAINED_PATTERN 指"冻结诊断下存在一致的模式"，不等于"机制已被识别"；量级限定与 PARTIALLY_EXPLAINED 敏感性已如实披露。
- 未新增任何指标；未使用 test-time 之外的任何信息；未触碰 E22 及之前任何冻结产物。

## 10. Paper implication

Doll 应作为 **limitation（适用边界）** 写入论文，模板如下（数字自动核对过）：

> Across the complete 10-category M²AD evaluation, R1-ILA yields positive mean ΔAUROC on 9 of 10 categories (macro mean +0.00760, median +0.01080), but it degrades one category reproducibly: **Doll** (mean ΔAUROC −0.03850, 95% paired-image bootstrap CI [−0.0661, −0.0140], bootstrap positive-difference proportion P(ΔAUROC>0) = 0.001, i.e. a one-sided resampling proportion rather than a two-sided p-value; 0/3 seeds positive). A descriptive audit of the frozen per-image scores shows that the degradation is (i) present in all three bank seeds, (ii) broad across illumination groups (8–9 of 10 groups), (iii) accompanied by a mild, near-symmetric downward shift of both normal and anomaly scores with a slightly larger anomaly-side drop, and (iv) **not** explained by illumination-shift strength (Doll's shift metric is close to the category median) nor by the low-baseline/high-headroom regime (Doll has the lowest baseline yet the worst outcome). We therefore report Doll as a reproducible boundary case of the method rather than as a uniform improvement.

**写法限制（与 §9 一致）**：
- 可写：上述事实 + CI + "boundary case"；
- 不可写：因果解释、普适鲁棒性、"anomaly information collapse"、"Doll 是随机噪声"、以及任何"修复"承诺。

---

## 11. 产物与复现

| 文件 | 说明 |
|---|---|
| `doll_seed_summary.csv` | Analysis A：3 seeds + mean/std/min/max |
| `doll_score_distribution.csv` | Analysis B：seed × (Good/Anomaly) × (F0/R1) × 11 统计量（long 格式） |
| `doll_separation_summary.csv` | Analysis B 派生：drift / compression / separation / d′ 及比例 |
| `doll_per_image_delta.csv` | Analysis C：2100 行（700 图 × 3 seeds），含 `rel_path`、逐图 Δscore、秩与秩变化 |
| `doll_illumination_breakdown.csv` | Analysis D：seed × {ALL, 01–10} 的 AUROC/AP/均值/标准差（33 行） |
| `category_comparison.csv` | Analysis E：10 类 × 冻结维度（全精度） |
| `doll_vs_others.json` | Analysis E：Doll vs 其余 9 类的 z / rank（20 维） |
| `e23_verdict.json` | Verdict + 预指定规则 + COND_A 逐 seed 分解 + 量级描述子 + 敏感性声明 + §7 关联 |
| `figures/` | Fig1 逐图 F0–R1 散点 · Fig2 Δscore 分布 · Fig3 光照级 ΔAUROC/ΔAP · Fig4 10 类 shift–gain（突出 Doll）· Fig5 10 类 ΔAUROC（突出 Doll）；各 PNG + PDF |
| `logs/e23_audit_run2.log` · `logs/run1_superseded/` | run2 日志；run1（分析 E/F 曾使用 6 位小数的主表导致关联值差 2.4e-6）已修复并**保留**为 superseded 证据 |
| `verification.csv` | E23 + paper_ready 独立检查（见 `scripts/e23_verify.py`）。此为 2026-10-16 的**历史 38/38 记录**；V30/V31 的语义修订与失败迁移说明见 `VERIFICATION_SEMANTICS.md` |

```bash
cd /root/autodl-tmp/research-experiments/industrial-anomaly-detection
/root/miniconda3/bin/python -u scripts/e23_doll_audit.py    # 分析 A–F（只读，0 GPU，~1 min）
/root/miniconda3/bin/python -u scripts/e23_figures.py       # Fig1–Fig5（PNG+PDF）
/root/miniconda3/bin/python -u scripts/e23_verify.py --out <独立路径>.csv   # 独立验证（勿覆盖历史 verification.csv）
```

**术语**：`d′` = 标准化可分性（cell 内存值，尺度无关）；`D_shift_F0` = E16/E18/E19 冻结的 patch 级光照偏移度量；`spread` = good 图像分数在 10 个 illumination group 间的极差；`COND_A` = 本实验预指定的失败模式判据（见 §2）。
