# R1-ILA 投稿前收尾核查报告（2026-10-23）

**目标期刊**：《计算机系统应用》（备选《微型电脑应用》）
**性质**：只读核查 + CPU 统计 + 文件整理。**0 GPU、0 训练、0 下载、0 方法扩展。**
**结论（一句话）**：**未发现 P0/P1**；受控比较公平、消融可解释、关键结果可逐图复算。**建议停止新增实验，进入论文定稿**；只需按 §6 的措辞约束写作。

---

## 0. 本轮实际完成 / 未完成

| 项 | 状态 | 说明 |
|---|---|---|
| 一、受控比较与消融核查 | **已完成** | 7 项同一性检查 30/30 cell 通过，见 `fairness_checklist.md` |
| 二、修正版工件级配对 bootstrap | **已完成** | 3000 次，10 类 + 3 个宏平均，见 §2 与 `specimen_bootstrap_v2.json` |
| 三、已有光照证据整理 | **已完成** | 合成/真实/外部分开，见 `illumination_definitions.md` |
| 四、原图案例 | **未完成（NOT_RUN）** | Doll 原图不在服务器（只有 l0_448 缓存），按"缺原图就跳过、不下载"执行 |
| 五、ZIP 结果包 | **已完成** | 路径/大小/SHA-256 见 `MANIFEST_SHA256.txt` 与随附说明 |

**输入**：全部为冻结产物（`results/e22/cells` 30 个、`results/e25/cells` 39 个、E17/E18/E21/E22/E23 汇总），
**唯一数据**是逐图分数 + 身份字段，未使用任何原图、未重新提取特征、未重新训练。

---

## 1. 受控比较与基本消融（对应任务一）

### 1.1 结论

**F0 与 R1 是受控比较：唯一变化是适配器本身。** 主干权重、预处理（448 BICUBIC + ImageNet 归一化）、
正常参考图像（同一 bank，300 张，同一 seed 规则）、测试样本、标签（元数据 `image_anomaly`）、
`detectable` 过滤、逐 patch L2、1-NN-max 评分定义**全部相同**；R1 只是把同一批特征经 `z' = z + Wz` 后再送进
同一评分函数。逐项证据（含代码行号）见 `fairness_checklist.md` §2。

经验核验（`fairness_checks.json`）：30/30 cell 长度自洽、无 NaN/Inf；10/10 类的 3 个 seed 使用**同一测试图像身份集合**
（身份键 `(specimen, illumination, kind)`，逐 cell 唯一）；10/10 类 `detectable` 掩码跨 seed 完全一致；
`bank_hash` 长度 64、`bank_size=300`；`trainable_params=589824`（768×768）在所有 cell 一致。

### 1.2 三组结果分别支持什么

| 组 | Bird / Car / Cube AUROC（seed0） | 支持 | 不支持 |
|---|---|---|---|
| **F0**（不适配） | 0.74725 / 0.78552 / 0.85410 | 冻结基线 | — |
| **完整 R1**（β=0.1） | 0.75786 / 0.80511 / 0.87010 | **联合适配**（consistency + preserve）相对 F0 有正增益（+0.0106/+0.0196/+0.0160） | ❌ 不能把增益**单独**归因于 consistency |
| **β=0**（只留 consistency） | 0.73309 / 0.78587 / 0.90904 | 去掉 preserve 后 **Cube collapse**（pairwise cos 0.988），Bird/Car 相对 C0 下降（−0.0248/−0.0192） | ❌ Cube 的 0.909 是 collapse 下的名义值，不是更好的检测器 |
| **preserve-only**（R2） | 0.74724 / 0.78547 / 0.85416 | preserve-only 等价于恒等映射 | ❌ 不能当独立 baseline 的"增益" |

**β=0 的 Cube 数值必须在论文中带 `collapsed=True` 与 pairwise cos 一起出现**，否则会被读成"只留 consistency 更强"。

**collapse 判据**（`scripts/e13a_screen.py:250-256`）：`std` 降到 0.5 倍以下、或 `pairwise_cos > 0.98`、或范数爆炸，三者任一即 `collapsed=True`。
β=0 组三个类的 pairwise cos 分别是 Bird 0.979 / Car 0.976 / **Cube 0.988**——三者都贴近阈值，只有 Cube 触发；因此正确表述是
"去掉 preserve 项后表征高度退化（0.976–0.988），Cube 越过 collapse 阈值"，而不是"只有 Cube 出问题"。

### 1.3 preserve-only 的恒等性（代码 + 数学，不需要训练）

`z' = z + Wz`，`W` 零初始化（`scripts/e13a_screen.py:126-132`）；`l_preserve = 1 − cos(z', z_frozen)`（`:223-232`）。
在 `W=0` 处 `z'=z=z_frozen`，`l_preserve=0` 且是余弦最大值点，梯度为 0 → AdamW 不更新 → **恒等映射被精确保留**。
实测 R2 与 F0 的差 ≤ 6e-5（三分类），与该分析一致。**因此不需要为"preserve-only"单独训练一个 baseline。**

---

## 2. 修正版工件级配对 bootstrap（对应任务二）

### 2.1 旧版的问题（`results/paper_supplement_20261023/audit_a3_specimen_cluster_bootstrap.json`）

旧脚本对每个 seed **单独**做 3000 次工件级重采样，然后取「三个 seed 的 2.5% 分位数平均」与「97.5% 分位数平均」，
并用 bootstrap 均值当点估计。**这不是"三种子平均增益"的 95% 区间**（三个分位数的平均 ≠ 平均统计量的分位数），
且三个 seed 各用不同随机流，未做同一工件抽样。

### 2.2 修正版做法（`scripts/paper_final_pack_20261023/audit_specimen_bootstrap_v2.py`）

- 三个 seed 的测试图像按**身份**（`specimen, illumination, kind`）对齐，**不假定数组顺序**（已报告：30/30 cell 顺序恰好一致，但脚本按身份对齐）；
- 只取 `detectable != False` 的 653/656/678… 张，与主表口径一致；标签跨 seed 逐图核对一致；
- 每次重采样：以 **specimen 为 cluster** 有放回抽样，同一工件的全部图像一起进入；**同一类别内 3 个 seed 与 F0/R1 共用同一次抽样**；
- 每个 replicate **先算每个 seed 的 ΔAUROC（和 ΔAP），再对 3 个 seed 取平均**；
- 最后算类别等权宏平均（10 类 / 开发 3 类 / 扩展 7 类），分位数在全部分 replicate 上取；
- 点估计来自**原始样本**（未重采样），不来自 bootstrap 均值；
- `rng_seed=20261023`，`n_boot=3000`，退化重采样**0 次**（未出现单类别重采样）；
- 实现上用整数 `sample_weight` 传给 sklearn，并在启动时与「显式复制」对比验证：**最大绝对差 = 0.0**。

**自洽性校验**：修正版点估计与冻结全精度主表 `results/e22/main_table_10cat_3seed.csv` 的最大差 **4.3e-7**（主表 6 位小数存储）。

### 2.3 结果（AUROC）

| 类别 | 工件数 | 点估计 ΔAUROC | 95% CI（工件级） | P(Δ>0) |
|---|---|---|---|---|
| Bird | 69 | +0.02586 | [−0.00882, +0.06428] | 0.917 |
| Car | 70 | +0.01123 | [−0.01359, +0.03624] | 0.810 |
| Cube | 70 | +0.00265 | [−0.02059, +0.02627] | 0.586 |
| Dice | 70 | +0.00336 | [−0.01403, +0.02277] | 0.651 |
| Doll | 68 | **−0.03850** | [−0.09526, +0.01349] | 0.074 |
| Holder | 69 | +0.01036 | [−0.00584, +0.02892] | 0.888 |
| Motor | 70 | +0.00807 | [−0.01587, +0.03018] | 0.762 |
| **Ring** | 69 | +0.02031 | **[+0.00209, +0.04034]** | **0.986** |
| Teapot | 70 | +0.01730 | [−0.00501, +0.04048] | 0.939 |
| Tube | 67 | +0.01532 | [−0.00679, +0.03942] | 0.916 |

| 宏平均 | ΔAUROC [95% CI] | P(Δ>0) | ΔAP [95% CI] |
|---|---|---|---|
| 10 类等权 | **+0.00760 [−0.00143, +0.01635]** | 0.948 | **+0.00777 [+0.00131, +0.01469]** |
| 开发 3 类 | +0.01325 [−0.00351, +0.03031] | 0.944 | +0.01300 [+0.00001, +0.02748] |
| 扩展 7 类 | +0.00518 [−0.00588, +0.01541] | 0.831 | +0.00553 [−0.00256, +0.01349] |

**事实**：只有 Ring 的类别级 CI 排除 0；10 类宏平均的 **AUROC CI 跨 0**、**AP CI 不跨 0**。
**解释边界**：这个区间刻画的是**固定这 3 个 seed 下、测试工件的抽样不确定性**，既不是训练随机性，也不是跨类别总体泛化；
`P(Δ>0)` 是 bootstrap 正差比例，**不是 p 值**。**不得**因为区间跨 0 而增加 seed/调参/加数据集。

### 2.4 与旧版对照（必须成对引用）

见 `specimen_bootstrap_old_vs_new.md`：差异来自方法而非数据（旧版 Doll P=0.101 / CI [−0.0914,+0.0169]；新版 P=0.074 / CI [−0.0953,+0.0135]）。
**论文只引用修正版**；旧版保留作过程记录，标题需注明"方法有缺陷，勿引用"。

### 2.5 顺带发现（P2，非本轮结论）

`results/e22/bootstrap_e22.json` 的 `d_auroc_mean`（Bird 0.02602）是**bootstrap 均值**，与主表点估计（0.02586）相差 1.7e-4。
论文引用**点估计**时应使用主表值，bootstrap 只提供区间。这不影响任何区间或判定。

---

## 3. 已有光照证据（对应任务三）

完整定义、单位、覆盖与来源见 `illumination_definitions.md`；机器可读表 `illumination_evidence.csv`（151 行）。
三类证据**必须分开写**：

| 类别 | 关键结果 | 允许的表述 |
|---|---|---|
| **合成光度扰动**（brightness/contrast/gamma ×{0.85,1.15}） | `robustness_feat` **10/10 类**下降（Bird 0.0763→0.0121；Doll 0.0513→0.0103）；`robustness_score` 10/10 下降（Bird 1.350→0.319） | "训练目标在训练同款光度变换上生效"；**部分由构造保证**，不能单独当真实光照证据 |
| **真实光照分组（M²AD 10 组）** | good 分数跨组 spread **9/10 类下降**（Bird 0.0381→0.0146，Motor 0.0550→0.0179）；**Dice 反向上升** 0.0154→0.0216（ratio 1.41） | "分数层面的跨光照 spread 收窄"（9/10，Dice 例外）；**不得**写 normal dispersion reduced |
| **外部真实域偏移（RobustAD）** | illumination 相关组：PCB illumination **+0.0266**、shadow **+0.0097**、white_balance **+0.0173**；MetalParts illumination +0.0014；但 background **−0.0394**、scale **−0.0265** | "对光照相关域偏移表现为正/持平，对背景与尺度偏移为负 → **条件性**" |

**两条硬约束**：
1. 表征离散度 `A_disp_pairwise_cos` / `A_disp_patch_std` 是**上升**的（Bird 0.889→0.985；patch std 1.354→1.920），
   而**分数 spread** 是下降的——两者不是同一件事，不能写成 "normal dispersion reduced"。
2. 光照 shift 与增益只是**弱正关联**（Spearman ≈ +0.58/+0.52，n=10，p≈0.08/0.13），**不得写成因果**。

---

## 4. 原图案例（对应任务四）—— NOT_RUN

- 服务器上只有 **Bird** 的原图（`data/m2ad/extracted/Bird`，15 GB）；**Doll 没有原图**（只有 `l0_448` 缓存，448 px 重采样，不是原图）。
- 任务要求"至少包含一个正收益类别和 Doll"。Doll 缺原图 → **整体跳过，不下载**（按指令）。
- 若后续需要，最小补充方案：仅取 Doll 的 view120/good+anomaly 各若干张原始分辨率图（约 10–20 张，几十 MB），
  与 `doll_per_image_delta.csv` 的 `delta` 对齐即可；**不需要重新推理**。

---

## 5. 问题分级

| 级别 | 内容 | 处理 |
|---|---|---|
| **P0** | 无 | — |
| **P1** | 无 | — |
| **P2-1** | 主表 seed 覆盖不同（3/3 vs 1/3），需表注 | 写作侧加表注 |
| **P2-2** | β=0 组 Cube 名义 AUROC 若出现，必须带 `collapsed=True` 与 pairwise cos | 写作侧约束 |
| **P2-3** | `bootstrap_e22.json` 的 mean 列是 bootstrap 均值（差 1.7e-4），点估计应引主表 | 写作侧约束 |
| **P2-4** | Dice 的真实光照 spread 反升，需在正文如实并列 | 写作侧约束 |
| **P2-5** | `results/e17/README.md` 旧措辞"BETA=0 三类最差"按 AUROC 对 Cube 不成立 | 引用时勿复述；文档本身已记录 |

---

## 6. 停止条件判断

- **未发现 P0/P1**：无数据泄漏、无评估错误、无标签错误、受控比较公平、关键结果可逐图复算。
- 现有负结果（Doll −0.0385、Macro AUROC CI 跨 0、MetalParts background/scale 为负、Dice spread 反升）**均为真实结果，不构成增加实验的理由**。
- **建议：停止新增 GPU 实验、不增加 seed/数据集/方法，进入论文定稿。**
  写作时按 §2.3、§3 的措辞约束调整论证强度即可。

---

## 7. 交付物

| 文件 | 内容 |
|---|---|
| `REPORT.md` | 本文件 |
| `fairness_checklist.md` | 受控比较与消融核查表（含代码行号） |
| `illumination_definitions.md` | 光照证据定义、覆盖、禁用措辞 |
| `ablation_summary.csv` | F0 / C0 / C1 / R2 逐类对照 |
| `illumination_evidence.csv` | 151 行：合成 / 真实 / 外部 / 多视角，含定义键 |
| `illumination_synthetic_summary.csv` | 合成扰动 3-seed 均值 |
| `specimen_bootstrap_v2.json` / `_per_category.csv` | 修正版统计（全部 CI、P(Δ>0)、退化计数） |
| `specimen_bootstrap_old_vs_new.md` | 新旧差异与引用规则 |
| `fairness_checks.json` | 30 cell 输入同一性核验 |
| `per_image/e22_cells/*.json`、`per_image/e25_cells/*.json` | 逐图分数 + 身份字段（**仅 ZIP 内**；`results/e22/cells`、`results/e25/cells` 已在版本库中，ZIP 内为其字节副本） |
| `code/*.py` | 本轮 4 个脚本（**仅 ZIP 内**；仓库内以 `scripts/paper_final_pack_20261023/` 为准） |
| `provenance.json`、`MANIFEST_SHA256.txt` | 版本、来源；`MANIFEST_SHA256.txt` 覆盖**仓库内**本目录文件 |
| `MANIFEST_ZIP.txt` | **仅 ZIP 内**，覆盖 ZIP 全部条目（含 `per_image/` 与 `code/`） |

> 说明：`provenance.json` 的 `git_head_at_pack_time` 是打包时刻的 HEAD；本目录所属提交是它的**子提交**（用 `git log -- results/paper_final_pack_20261023` 查）。
> 原始图像与模型权重不在包内；`per_image` 只是已入库逐图分数的副本，不涉及数据集再分发。
