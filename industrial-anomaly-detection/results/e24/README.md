# E24 — 光度增广敏感性审计（强度 / 扰动类型 / 多 seed 稳定性）

**日期**：2026-10-19（夜间批次，00:12–01:58）
**实验编号**：E24
**方法（冻结，不修改）**：R1-ILA，配置来源 `results/e17/final_method.json`
**数据**：M²AD Bird/Car/Cube，view120，300-bank，official `image_anomaly` + `detectable` 口径
**性质**：**解释性审计（post-hoc explanatory audit）**，不是方法选择实验

---

## 1. 实验题目

在冻结的 R1-ILA 上，系统扫描 **photometric augmentation 的强度** 与 **扰动类型组合**，回答三个问题：

1. E17 冻结的 `{0.85, 1.15}` 是否位于稳定区间（而不是偶然的工作点）？
2. 哪一类扰动（brightness / contrast / gamma）在承载效果？
3. 是否存在**明显更优**的强度配置（如有，只记录为候选，不修改冻结定义）？

---

## 2. 论文阶段

**Stage ⑬-b / 方法合理性（method-justification）**。E17 只用 BETA 单轴搜索后冻结了 `{0.85,1.15}`，冻结时**没有**做过强度敏感性。E24 补齐这一格：它是「冻结配置的选择是有依据的」这一论断的证据，而不是新方法或新结果。

---

## 3. 要验证的核心论断

> 「R1-ILA 的增益来自 **mild 光度增强诱导的光照一致性训练**，且冻结强度 `{0.85, 1.15}` 位于**稳定平台**；更强或更弱的增强不会显著更好，因此冻结值不是任意选择。」

**否定形式**：若某一强度显著优于 `{0.85,1.15}` 且跨类跨 seed 一致，则必须如实记录「冻结值并非最优」，但**不得**据此回改 E17/E22/E25。

---

## 4. 实验目的

- 量化 `ΔAUROC/ΔAP` 随强度的变化曲线（6 档 × 3 类，seed 0）。
- 对排名前 2 的强度做 3-seed 稳定性验证，区分「真实差异」与「adapter 训练随机性」。
- 用 7 种扰动子集做类型消融，定位 active ingredient。
- **不做**：不重选参数、不改 E17 冻结、不引入新模块、不看 RobustAD 结果。

---

## 5. 数据与参数

| 项 | 值 |
|---|---|
| 数据 | M²AD `l0_448` 缓存（Bird/Car/Cube，view120 test，300-bank 由 `E8.bank_for(pool, seed)` 生成） |
| 标签 | official `image_anomaly` + `detectable != False`（与 E18/E22 同口径） |
| 冻结超参 | BETA=0.1，lr=1e-3，AdamW，≤300 step，patience 30（每 5 步验证），batch=8，bank=300，train/val=240/60，DINOv2-B frozen，1-NN-max |
| 唯一变量 | **augmentation strength**（6 档）或 **augmentation family 子集**（7 种） |
| 注入方式 | `e13a_screen.AUG_LEVELS / AUG_FAMILIES / AUG_LIST` 模块全局（与 E17 的 `BETA` 注入同一先例，**未改任何冻结文件**），运行前对 BETA/lr/steps/patience/batch/TRAIN_N/VAL_N 做硬断言 |
| 强度档位 | S05 0.95/1.05 · S10 0.90/1.10 · **S15 0.85/1.15（冻结值）** · S20 0.80/1.20 · S25 0.75/1.25 · S30 0.70/1.30 |
| 类型子集 | A0 brightness · A1 contrast · A2 gamma · A3 B+C · A4 B+G · A5 C+G · **A6 B+C+G（冻结值）** |

---

## 6. 实验过程

```
# 阶段 1（18 cells，40 min）
python scripts/e24_runner.py --stage strength \
  --strengths S05,S10,S15,S20,S25,S30 --categories Bird,Car,Cube --seeds 0
# 阶段 2（12 cells，26 min；seed 0 复用阶段 1）
python scripts/e24_runner.py --stage stability --strengths S15,S10 \
  --categories Bird,Car,Cube --seeds 1,2
# 阶段 3（21 cells，39 min）
python scripts/e24_runner.py --stage type --single-strength S15 \
  --types A0,A1,A2,A3,A4,A5,A6 --categories Bird,Car,Cube --seeds 0
python scripts/e24_analysis.py
```

`setsid nohup` 脱离 SSH；cell 级 resume；每 cell 完成立即写 JSON + 刷新 CSV。**51 个 cell 全部一次通过**（无 OOM、无 NaN、无重试）。GPU 峰值 ≈5.4 GB，单 cell ≈130 s。

---

## 7. 结果（**事实**）

### 7.1 冻结配置等价性（关键校验）

| 检查 | 结果 |
|---|---|
| S15（=0.85/1.15）seed 0 的 AUROC/AP vs `results/e18/main_table.csv` 的 `Ours` 行（Bird/Car/Cube 各 2 个指标） | **9/9 项 `max|Δ| = 0.0`**（容差 1e-6） |
| S15 三 seed 逐类 ΔAUROC 均值 | Bird **+0.02586** / Car **+0.01123** / Cube **+0.00265** = E18-P2 已发表的 seed-mean（**完全一致**） |
| F0（同一 bank）vs E18 `F0` 行 | **逐位一致** |

→ E24 harness 与 E17/E18 冻结管线**数值等价**，因此强度/类型比较是在同一方法上做的，而不是另一个实现。

### 7.2 强度扫描（seed 0，3 类；ranked by 稳定性规则）

| rank | strength | levels | mean ΔAUROC | median | mean ΔAP | 正向 | worst ΔAUROC | mean steps |
|---|---|---|---|---|---|---|---|---|
| **1** | **S15（冻结）** | **0.85/1.15** | **+0.015401** | +0.016001 | +0.011712 | **3/3** | **+0.010614** | 54 |
| 2 | S10 | 0.90/1.10 | +0.015323 | +0.018070 | +0.012581 | 3/3 | +0.009790 | 84 |
| 3 | S05 | 0.95/1.05 | +0.014355 | +0.012612 | +0.012343 | 3/3 | +0.008530 | 81 |
| 4 | S20 | 0.80/1.20 | +0.011765 | +0.014658 | +0.008320 | 3/3 | +0.003960 | 89 |
| 5 | S25 | 0.75/1.25 | +0.011001 | +0.015716 | +0.006818 | 2/3 | −0.003144 | 74 |
| 6 | S30 | 0.70/1.30 | +0.011276 | +0.018237 | +0.007712 | 2/3 | −0.004279 | 91 |

**逐类结构（seed 0）**

| strength | Bird ΔAUROC | Car ΔAUROC | Cube ΔAUROC |
|---|---|---|---|
| S05 | +0.021922 | +0.012612 | +0.008530 |
| S10 | +0.018070 | +0.018109 | +0.009790 |
| **S15** | +0.010614 | +0.019588 | +0.016001 |
| S20 | +0.003960 | +0.016676 | +0.014658 |
| S25 | **−0.003144** | +0.020431 | +0.015716 |
| S30 | **−0.004279** | +0.019869 | +0.018237 |

→ **类别偏好分裂且单调**：Bird 随增强变强而单调变差（S05 最好 → S30 转负）；Car 在 S20–S30 最好；Cube 在 S25–S30 最好。**没有单一强度能同时取悦三类**。

### 7.3 Top-2 三 seed 稳定性（S15 vs S10）

| strength | seeds | cells | mean ΔAUROC | std | mean ΔAP | std | 正向 | 类别一致性 |
|---|---|---|---|---|---|---|---|
| S15 | 0,1,2 | 9 | **+0.013246** | 0.014589 | +0.012998 | 0.011043 | 8/9 | 3/3 |
| S10 | 0,1,2 | 9 | **+0.015984** | 0.014317 | +0.015260 | 0.012727 | 8/9 | 3/3 |

S15 逐类（3 seed mean）：Bird +0.025857（std 0.0135）、Car +0.011226（0.0073）、Cube +0.002645（0.0142）。

### 7.4 扰动类型消融（S15 strength，seed 0，3 类）

| rank | type | families | mean ΔAUROC | mean ΔAP | 正向 | worst ΔAUROC |
|---|---|---|---|---|---|---|
| 1 | A2 | **gamma** | **+0.019067** | +0.014409 | 3/3 | **+0.012100** |
| 2 | A4 | brightness+gamma | +0.015539 | +0.011918 | 3/3 | +0.011051 |
| 3 | A6 | **B+C+G（冻结）** | +0.015401 | +0.011712 | 3/3 | +0.010614 |
| 4 | A3 | brightness+contrast | +0.011223 | +0.008334 | 2/3 | −0.000188 |
| 5 | A5 | contrast+gamma | +0.009778 | +0.006037 | 3/3 | +0.002521 |
| 6 | A1 | contrast | +0.008920 | +0.004499 | 3/3 | +0.001297 |
| 7 | A0 | brightness | +0.004142 | +0.003421 | 3/3 | +0.000066 |

---

## 8. 异常情况

1. **无工程失败**：51/51 cell 一次通过（对比：E22 每 cell 265 s，E24 每 cell ≈130 s，因为省去 d′/robustness 诊断分支）。无 OOM、无 NaN、无 loss 非有限。
2. **早期停止步数差异大**（44–214 步）：不同强度/类型的验证损失轨迹不同，属方法行为，不是 bug；`steps_used` 已逐 cell 记录在 `strength_round1.csv` / `type_ablation.csv`。
3. **A3（B+C）在 Cube 上 worst = −0.000188**（2/3 正向）：单点负值，未删除。
4. 未运行：EfficientAD/SoftPatch 等其它 baseline 的强度敏感性（不在本实验范围）。

---

## 9. 分析

### 事实

- 冻结强度 S15 在 **seed 0 排名第 1**（mean +0.0154、median +0.0160、3/3、worst +0.0106 全为 6 档中最高/并列最高）。
- 强度曲线非单调：`S05 ≈ S10 ≈ S15 > S20 > S25 ≈ S30`，且 S25/S30 在 Bird 上**转负**（−0.0031 / −0.0043）。
- 3-seed 下 S10 的 mean 比 S15 高 **+0.00274**，但两者 std 均 ≈0.0143–0.0146（**gap 仅为 std 的 0.19 倍**），正向 cell 同为 8/9，Cube 上 S10 反而更低（+0.00046 vs +0.00265）。
- 类型消融：**gamma 单独（A2）最高**（+0.0191），brightness 单独最低（+0.0041）；full A6 与 A4 几乎重合（+0.0154 vs +0.0155）。
- 所有 7 种类型子集**均为正**（3/3 或 2/3），说明增益不依赖某一特定色彩变换的存在。

### 解释（推断，不作因果声明）

- **冻结值处于平台中心而非尖峰**：S05–S15 三档的 mean 差异 ≤0.001，且都被 3/3 支持；因此「0.85/1.15 是任意取的」这一质疑被削弱，但「0.85/1.15 是唯一最优」**不成立**（S10 在 2 个新 seed 上略高）。
- **Bird 对强增强敏感**：Bird 是 3 类中 F0 最低（0.747）、headroom 最大的类，却在强增强下**退化**（S25/S30 负），说明强光度扰动带来的表示形变超过了一致性收益 → 与 E17「BETA↑ → illumination shift↑、robustness 变差」同向。
- **Car/Cube 偏好更强增强**：Car F0 0.786、Cube 0.854（headroom 更小）却从更强增强获益，与「headroom 越大越受益」的简单叙事**冲突**，提示类别间的差异来自纹理/光照响应而非可用 headroom。
- **gamma 是主要载体**：单独 gamma（指数变换，改变全局色调映射）即可达到 full 的 124% 平均增益；brightness 单独仅 27%。contrast 与其它项更多是**冗余**（A3 甚至略差）。
- **为什么不改冻结**：S15 与 S10 的 3-seed 差异远小于 seed 噪声；A2（gamma only）虽然均值更高，但只是单 seed、且 E17 冻结的是「3 family 覆盖」这一更保守的设定。按预登记规则（见 §11）**保持 S15 / A6 不变**。

### 假设（未验证，不得写成结论）

- H1：gamma 的作用机制是**压缩高光/暗部动态范围**，使 patch 特征对曝光更稳定；未做逐通道/逐光照组的特征分析，无法证实。
- H2：S10 相对 S15 的 +0.0027 可能是真实的小效应，也可能只是 3 个 seed 的抽样波动（df=2，std 0.0146）。要区分需要 ≥8 seed（成本 ≈8×3 类×130 s ≈ 52 min/强度）。
- H3：A2 优于 A6 可能来自「contrast 增强放大了正常样本的纹理方差」；未做特征层面的方差分解。

---

## 10. 结论

1. **冻结强度 `{0.85,1.15}` 位于稳定平台，且在 seed 0 上是 6 档中的最优点**（mean/median/worst 三项均为 1 或并列 1）。
2. **3-seed 证据不支持「存在明显更优的强度」**：S10 领先 +0.0027，远小于 seed 噪声（0.0146），且 Cube 上更差。按预登记规则 **保留 S15**。
3. **强增强有害**：S25/S30 在 Bird 上转负（−0.003 / −0.004），说明 E17 选择 mild 强度是必要的。
4. **扰动类型的 active ingredient 是 gamma**；brightness 贡献最小；full B+C+G 的额外 family 在平均上**没有**带来增益（A4/A6 ≈ +0.0154–0.0155 vs A2 +0.0191），但 full 的 worst-case 与 A2 接近（+0.0106 vs +0.0121）。
5. **E24 不修改任何冻结定义**：E17 方法（S15 + A6 + BETA 0.1）、E22 与 E25 的结论全部保持不变。

---

## 11. 出口条件判断

预登记规则（**看数据前写定**，见脚本 docstring）：

- Top-2 选择：`(#正向 cells desc, worst cell desc, mean ΔAUROC desc)` —— **稳定性优先，不单看最高点**。
- S15 去留：若挑战者相对 S15 的增益 **≤ 0.002**，或跨类一致性更差 → **保留 S15**；若明显更好且跨类跨 seed 一致 → **只记录候选**，不改冻结。

实际判定：

| 判据 | 结果 |
|---|---|
| S15 是否在 6 档中稳居前列 | **是**（seed0 排名 1；worst-case 最高） |
| 是否存在 >0.002 且跨类一致的更优强度 | **否**（S10 +0.0027，但 Cube 更差、std 0.0146） |
| 类型消融是否显示冻结 family 组合明显次优 | **部分**（gamma-only +0.0191 vs full +0.0154，单 seed，未达"明显"） |
| 结论 | **KEEP_FROZEN（S15 / A6 / BETA 0.1）**，S10 与 A2 记为「候选配置」，需独立预注册实验才可采用 |

---

## 12. 对论文证据链的贡献

- **推进的证据格**：方法合理性（Stage ⑬-b）。E24 把「冻结强度的选择」从**未验证的默认**变成**有 6 档扫描 + 3-seed 稳定性 + 7 种类型消融支撑的审计结论**。
- **可写结论**：(i) 冻结强度位于平台且非任意；(ii) 强增强（≤0.75/1.25）会使 Bird 转负，说明 "mild" 是必要的；(iii) gamma 是主要载体，brightness 贡献最小；(iv) 完整 3-family 组合在均值上不是最优，但差距在噪声内。
- **不可写**：(i) 「0.85/1.15 是唯一最优」；(ii) 「gamma-only 应该替换 full」（单 seed）；(iii) 任何因果机制表述。
- **下一格**：若要把 A2/S10 变成主张，需要独立预注册的多 seed（≥8）实验；否则停在「审计结论」。

---

## 13. 产物清单

| 文件 | 说明 |
|---|---|
| `results/e24/cells/*.json` | **51 个 cell**（strength 18 / stability 12 / type 21），含逐图分数、标签、steps、bank hash、runtime |
| `results/e24/strength_round1.csv` | 强度扫描逐 cell（含 F0/R1 AUROC、AP、Δ、steps、runtime） |
| `results/e24/strength_round2.csv` | Top-2 三 seed 逐 cell |
| `results/e24/type_ablation.csv` | 7 种类型 × 3 类逐 cell |
| `results/e24/tables/strength_ranked.csv` | 强度排名（稳定性规则） |
| `results/e24/tables/type_ranked.csv` | 类型排名 |
| `results/e24/tables/stability_multiseed.csv` | 三 seed mean/std/正向/类别一致性 |
| `results/e24/final_strength_analysis.json` | 完整机器可读结论 + 冻结复现校验 + S15 去留判定 |
| `results/e24/figures/e24_strength_vs_dauroc.{png,pdf}` | 强度 vs ΔAUROC（bar=mean，点=逐类） |
| `results/e24/figures/e24_strength_vs_dap.{png,pdf}` | 强度 vs ΔAP |
| `results/e24/figures/e24_type_ablation.{png,pdf}` | 类型消融（ΔAUROC bar + ΔAP 线） |
| `scripts/e24_runner.py` / `e24_analysis.py` | 运行 / 分析脚本（复用 `e13a_screen`、`e13b_screen.train_r1`、`e11_p2_runner`，**未改冻结文件**） |

---

## 14. 下一步

1. **（需批准）** 若要把 「gamma-only」或 「S10」 变成论文主张：独立预注册、≥8 seed、跨 3 类 + 交叉验证；本轮证据不足以支持。
2. **（需批准）** 与 E25 联动的解释性分析：用同一 harness 在 RobustAD 上复算「强度敏感性」，检验「headroom 依赖」是否同样出现在强度轴上——**属于新实验，不得改动 E24/E25 已冻结结论**。
3. 论文写作中，E24 只能以「审计/合理性」身份出现，不得表述为新的方法贡献。
