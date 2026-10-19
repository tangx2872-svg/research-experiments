# EVIDENCE_MAP — 论文证据地图（唯一权威版）

**生成日期**：2026-10-19（paper evidence reconciliation，0 GPU、0 新训练）
**覆盖范围**：E17（冻结）→ E18/E19 → E22（10 类 × 3 seeds）→ E21 → E23（Doll 审计）→ **E24（敏感性审计）** → **E25（RobustAD 外部验证）**
**性质**：口径统一 + 证据索引。**不含任何新结果**；所有数字均来自已入库的冻结产物，未重算、未挑选。

> 与 `paper_evidence_map.md` 的关系：后者由 `scripts/paper_ready_build.py` 生成，覆盖 Claims A–H 的简表；**本文件是手写、不可被重新生成覆盖的权威版本**，两者冲突时以本文件为准，并应修正生成脚本。

---

## 0. 术语裁决（先读这一节）

| 对象 | 准确角色 | 禁止表述 |
|---|---|---|
| **M²AD**（10 类，view120） | **主数据集 / main evaluation dataset** | 不得称为 "external dataset"、"第二个外部数据集" |
| **M²AD Bird / Car / Cube** | 主数据集 10 类中的 **3 类**，且是方法开发所用类别（development categories） | 不得称为 "held-out external"、"independent external" |
| **M²AD Dice…Tube（7 类）** | 主数据集内的 **held-out categories**（同数据集、不同类别） | 不得称为 "external dataset" |
| **RobustAD** | **唯一独立外部数据集**（independent external dataset），官方由 3 个 **sub-dataset** 组成：PCB / MetalParts / PiledBags | — |
| **PCB / MetalParts** | RobustAD 的 **sub-dataset（子数据集）**；每个 sub-dataset 内含 1 个 **source domain**（train）+ 若干 **target domain**（test0…testN） | 不得称为 "category"、"category group"、"M²AD category" |
| **E24** | 主数据集 3 类子集上的 **敏感性审计（sensitivity audit / post-hoc audit）** | 不得称为 "外部验证"、"新方法"、"方法选择实验" |
| **E25** | **external dataset validation with re-training（重训式外部数据集验证）** | 不得称为 "direct cross-dataset transfer" |

**历史口径说明（不修改历史文档）**：`docs/E4_*`、`PROJECT_STATE.md`、根 `README.md` 中 "PRIMARY EXTERNAL DATASET = M²AD" 属于 Phase II（相对 MVTec AD 而言）的历史表述。M²AD 在 E17 之后已成为**主数据集**；当前论文链条中**只有 RobustAD 是外部数据集**。读旧文档时须按此换算。

**代码命名与官方命名的差异（需在论文中统一）**：`scripts/e25_robustad_eval.py` 用 `DOMAINS = {"PCB", "MetalParts"}` 表示两个 sub-dataset（代码内简称 domain）。论文正文应写 **sub-dataset**，`test0…testN` 写 **source / target domain**。

---

## 1. Frozen Method

- **方法**：R1-ILA（Illumination-consistent Linear Residual Adaptation）
- **定义**：`z' = z + W z`，`W ∈ R^{768×768}`，**零初始化**，normal-only 训练，可训练参数 **589,824**
- **backbone**：DINOv2-B，**冻结**，最后一层 patch token，**1024 patch × 768 维**，L2 归一化
- **打分**：1-NN-max（patch 级最近邻取最大），与全部历史实验一致
- **损失**：`L = L_cons + BETA · L_preserve`，`BETA = 0.1`
- **优化**：AdamW，lr = 1e-3，≤300 step，patience 30（每 5 step 验证），batch = 8 张
- **训练/验证**：M²AD 上 TRAIN_N = 240 / VAL_N = 60；300-image memory bank
- **增广**：brightness / contrast / gamma × {0.85, 1.15}，共 **6 个确定性变体**
- **唯一合法超参来源**：`results/e17/final_method.json`（E25 运行前用 `assert_frozen_config()` 硬断言；不一致即报错终止）
- **冻结判据**：E17 Round-1 BETA sweep {0.0, 0.05, 0.1, 0.2, 0.5} 无挑战者通过 +0.003 门 → `FREEZE_CURRENT_R1`

**事实**：E24 与 E25 均未修改上述任何一项。
**边界**：不得把 `final_method.json` 之外的值写进论文。

---

## 2. Main Dataset — M²AD

| 项 | 内容 |
|---|---|
| 数据集 | M²AD（本地 `data/m2ad`），论文链条中的**主数据集** |
| 类别 | **10 类**：Bird, Car, Cube, Dice, Doll, Holder, Motor, Ring, Teapot, Tube |
| 其中开发类 | Bird, Car, Cube（E13–E17 方法开发与 BETA 选择） |
| 其中 held-out 类 | Dice, Doll, Holder, Motor, Ring, Teapot, Tube（E22 新增 7 类） |
| 缓存 / 视角 | `l0_448` 缓存；主表用 **view 120**；E21 另测 090/120/240/330 |
| bank | 每类 300 张（`E8.bank_for(pool, seed)`，seed 决定选图） |
| 标签口径 | 官方 `image_anomaly` + `detectable != False`（E18/E22/E24 一致） |
| 光照扰动 | 方法侧的 6 个确定性光度变体（训练时）；评测量为 view120 原始图 |

**事实**：E18 = 3 类（Bird/Car/Cube）× 单 seed；E22 = 10 类 × 3 seeds；E24 = 3 类（Bird/Car/Cube）子集审计。
**因此**：E24 是**同一主数据集内部的子集扩展**，不构成任何形式的外部验证。

---

## 3. Main Results

**E22，10 类 × 3 seeds × view120 × 300-bank（`results/e22/main_table_10cat_3seed.csv`）**

| 指标 | 值 |
|---|---|
| 正向类别（mean ΔAUROC） | **9 / 10** |
| 正向类别（mean ΔAP） | **9 / 10** |
| 三个 seed 全正（ΔAUROC） | **7 / 10** |
| macro mean ΔAUROC | **+0.00760** |
| median ΔAUROC | **+0.01080** |
| macro mean ΔAP | **+0.00777** |
| 唯一稳定负类 | **Doll −0.03850**（0/3 seed 正；bootstrap CI [−0.0661, −0.0140]，bootstrap 正差比例 **P(ΔAUROC>0) = 0.001**，单侧重采样比例而非双侧 p 值；跨 seed 样本 std **0.004638**，ddof=1） |
| 协议判定 | **HOLD**（Doll 触发 severe-regression 排除项，非 STRONG GO） |

**E18（3 类，seed0，`results/e18/main_table.csv`）**：F0 → Ours(R1) 的 AUROC = Bird **0.74725→0.75786**、Car **0.78552→0.80511**、Cube **0.85409→0.87010**；E24 的 S15 seed 0 在这 9 项上与 `Ours` 行 `max|Δ| = 0.0`、`F0` 行逐位一致，故以此作为等价性锚点。**2026-10-19 修正**：旧文本写作「0.75786→0.77717、…、Cube 0.87010→0.86645」，把 Ours **seed0** 值与 **3-seed 均值**混用，并因此把 Cube 的方向写反（实际 seed0 为上升）；`main_table.csv` 中单 seed 列即 seed0，3-seed 列另列，二者不得互串。

**边界**：HOLD 是协议判定，不得改写为 STRONG GO，也不得写成"方法失败"。

---

## 4. Ablation

**E17 BETA sweep（5 配置 × Bird/Car/Cube，`results/e17/summary.csv` → `paper_ablation_table.csv`）**

- BETA = 0.0：Bird/Car 显著变差（Bird+Car mean ΔAUROC vs C0 = −0.02200），Cube 名义 AUROC 最高（0.90904）但**表示崩塌**（pairwise cos 0.988）→ 不得只引 AUROC
- BETA = 0.2：最佳挑战者，Bird+Car +0.00173，未过 +0.003 门，且 Cube −0.0087 触发 safety gate
- BETA = 0.5：Cube −0.0098
- **BETA = 0.1（冻结）**：内部最优点，五项 gate 中未通过 +0.003 提升门，但无崩塌、无安全门失败

**E24 增广类型消融（7 子集 × 3 类，seed0）**见 §5。

---

## 5. Sensitivity Audit — E24

**性质**：post-hoc explanatory audit（**不是**方法选择、**不是**新方法贡献）。针对**已冻结**配置做强度扫描 + 类型消融 + Top-2 三 seed 稳定性。

**5.1 等价性（前置门，已通过）**
- S15 seed0 与 `results/e18/main_table.csv` 的 `Ours`/`F0` 行：**9/9 项 max|Δ| = 0.0**（容差 1e-6）
- 结论：E24 与 E17/E18 冻结管线**数值等价**，比较对象是同一方法而非另一实现

**5.2 强度扫描（seed0，3 类，按预登记规则 `(#正向 desc, worst desc, mean desc)` 排序）**

| rank | 强度 | mean ΔAUROC | median | worst | 正向 |
|---|---|---|---|---|---|
| **1** | **S15 = 0.85/1.15（冻结）** | **+0.015401** | +0.016001 | **+0.010614** | **3/3** |
| 2 | S10 = 0.90/1.10 | +0.015323 | +0.018070 | +0.009790 | 3/3 |
| 3 | S05 = 0.95/1.05 | +0.014355 | +0.012612 | +0.008530 | 3/3 |
| 4 | S20 = 0.80/1.20 | +0.011765 | +0.014658 | +0.003960 | 3/3 |
| 5 | S25 = 0.75/1.25 | +0.011001 | +0.015716 | −0.003144 | 2/3 |
| 6 | S30 = 0.70/1.30 | +0.011276 | +0.018237 | −0.004279 | 2/3 |

- S15 排名第 1 的依据是 **worst-case + 均值**；**median 仅第 3**（S30/S10 更高）→ 不得写"三项全优"
- 类别结构：Bird 随增强单调变差（S05 +0.021922 → S30 −0.004279）；Car 偏好 S20–S30；Cube 偏好 S25–S30 → **无单一强度同时最优**

**5.3 Top-2 三 seed 稳定性**

| 强度 | mean ΔAUROC | std | 正向 cell | 类内一致 |
|---|---|---|---|---|
| S15（冻结） | +0.013246 | 0.014589 | 8/9 | 3/3 |
| S10 | +0.015984 | 0.014317 | 8/9 | 3/3 |

- S10 领先 **+0.00274 ≈ 0.19 × std**；Cube 上 S10 更差（+0.00046 vs +0.00265）→ **不可区分**

**5.4 类型消融（seed0）**

| rank | 类型 | mean ΔAUROC | AP |
|---|---|---|---|
| 1 | A2 gamma | +0.019067 | +0.014409 |
| 2 | A4 B+G | +0.015539 | +0.011918 |
| **3** | **A6 B+C+G（冻结）** | **+0.015401** | +0.011712 |
| 4 | A3 B+C | +0.011223 | +0.008334 |
| 5 | A5 C+G | +0.009778 | +0.006037 |
| 6 | A1 contrast | +0.008920 | +0.004499 |
| 7 | A0 brightness | +0.004142 | +0.003421 |

**5.5 判定**：`KEEP_FROZEN（S15 / A6 / BETA 0.1）`。S10 与 A2 仅为**候选配置**，不进入方法定义。
**标准表述**：*E24 does not redefine the frozen method; it audits the robustness of the existing frozen configuration.*

---

## 6. Multi-seed Evidence

| 实验 | seeds | 覆盖 | 目的 | 结论 |
|---|---|---|---|---|
| E22 | 0/1/2（bank seed） | 10 类 | 主结果稳健性 | 9/10 正、7/10 三 seed 全正；Doll 稳定负 |
| E21 | **仅 seed0** | 3 类 × 4 视角 | 视角泛化 | 8/12 正，macro +0.00142，**HOLD**（seed1/2 未跑） |
| E24 | 0 + (S15/S10 的 1/2) | 3 类 | 强度稳定性 | S15 与 S10 在 seed 噪声内不可区分 |
| E25 | 0/1/2（**adapter 训练种子**） | 2 sub-datasets × 13 splits | 外部验证稳健性 | 域结构跨 3 seed 不翻转 |

**关键区分（不得混用）**：
- E22 的 seed = **bank 采样种子**（选哪 300 张 normal）
- E25 的 seed = **adapter 训练种子**（batch 采样 + 增广变体抽样）；bank 与测试图固定不变
- 两者的 std 语义不同，**不可跨实验比较**

---

## 7. External Dataset Validation — RobustAD

**7.1 数据集独立性**：RobustAD（HF `AmazonScience/RobustAD`，CC-BY-4.0）为独立采集的真实工业数据集，**与 M²AD 无重叠**。官方定义：3 个 sub-dataset（PCB / MetalParts / PiledBags），每个含 1 个 source domain（train）+ 多个 target domain（test0…testN）。

**7.2 覆盖**：PCB 6 splits（test0=source + 5 target）、MetalParts 7 splits（test0 作废 + 6 target）；**PiledBags 未下载（磁盘不足），不推测、不补值** → 覆盖 **2 / 3** sub-datasets。

**7.3 训练方式（逐代码确认，`scripts/e25_robustad_eval.py`）**

| 问题 | 事实 | 证据 |
|---|---|---|
| adapter 是否在 RobustAD normal 上重训 | **是**，`run_domain()` 内 `LinearAdapter()` 新建 + `train_r1()` | L364–374 |
| 是否每个 domain / seed 重训 | **是**，每个 (sub-dataset, seed) 独立训练一次；同一 sub-dataset 的 6–7 个 target split 共用该 adapter（**不是**每 split 一个） | `partial.json` 6 条记录；`adapter_hash` 6 个互不相同 |
| 是否用过 M²AD 的 adapter | **从未**（无任何 load/checkpoint 路径） | 全文件无 `load_state_dict` / 权重读取 |
| baseline 与 R1 是否同划分 | **是**，B0 与 B1 用同一 180-normal bank、同一测试图、同一标签 | L357–409 |
| bank 与训练/验证划分 | bank = 该 sub-dataset 的 **180 张 source-train normal**；adapter 在这 180 张上重训，**前 120 训练 / 后 60 验证**（E17 协议 `TRAIN_N=240` 因 bank 规模不可满足降为 120，属 bank 约束，**非调参**） | `run_domain()`；`README` §4/§5 |
| 几何预处理 | **方形 448×448**：PCB 2048² 与原短边路径**逐位等价**（tensor 与 DINOv2 特征 `max|Δ| = 0.0`）；MetalParts 2681×1500 为 16:9 → 1:1 **各向异性压缩**，绝对水平**不可**与 PCB 横向比较（配对差仍公平） | `geometry_PCB.json` / `geometry_MetalParts.json`；`ROBUSTAD_PROTOCOL_AUDIT.md` §10 |
| seed 改变什么 | 仅 (i) 训练 batch 的 normal 采样顺序、(ii) 每样本抽到 6 个增广变体中的哪一个；`W` 为零初始化，**初始化本身与 seed 无关**；bank / 测试图 / 增广集合 / 超参均不变 | `LinearAdapter.__init__` 零初始化；`train_r1` 内 `rng.integers` ×2 |

**结论（唯一）**：**A 分支 —— external dataset validation**。**禁止**写 "direct cross-dataset transfer"。

**7.4 前置守卫**：`assert_frozen_config()` 对 BETA / lr / steps / patience / eval_every / batch / AUG_LIST / D,N_PATCH 逐项硬断言，不一致即抛异常终止。

**7.5 数据完整性与 cell 账目**：**39/39 cell 均完成且 `status=OK`**，逐图复算 AUROC/AP 的 `max|Δ| = 0.0`（容差 1e-12）。**39 与 33 的关系**：

| 类别 | cell 数 | 是否进入 target 汇总 |
|---|---|---|
| 11 个有效 target split × 3 seed | **33** | **是** |
| PCB test0（source reference，`valid=1`） | 3 | 否（仅用于 source→target 描述） |
| MetalParts test0（`valid=0`） | 3 | 否（**同名重叠、非字节重复；近重复未检验**，见 §7.6 / §9） |

即 **39 = 33 + 3 + 3**。因此**不得笼统写"无排除"**：样本没有被删除，指标也没有被填零，被排除的是**聚合口径**；同时**无缺失 cell、无重复重跑 cell**。推荐表述：**"39/39 cell 均完成；MP test0 依照所述有效性规则不纳入汇总。"**
**日志中的 6 次 Traceback 均为写 cell 之前的工程错误**（MetalParts 非方形导致 1824 token、`AUG_BS`/`geom` 未定义），修复后重跑，**未影响任何已落盘结果**。

**7.6 MP test0 有效性规则：依据与时点（含 2026-10-19 勘误）**：规则 = "145 个文件名与 source-train 重合（104 normal / 41 anomaly）→ `valid=0`，不纳入 held-out"。

- **勘误（事实层）**：字节级 SHA-256 审计（`results/e25/overlap_audit.json`）比对全部 145 对同名文件：**字节相同 0 对、不同 145 对、缺失 0**；抽样 20 对 64×64 灰度平均绝对差 **10.94–32.46/255**。⇒ **只能排除字节完全重复**；**近重复（perceptual / near-duplicate）未检验**，缩略图差异也不证明采集对象独立。原协议「test0 图像与 train 是同一批物理文件」**事实错误，已被取代**（`ROBUSTAD_PROTOCOL_AUDIT.md` §11）。
- **决定不变**：`valid=0`、cell 账目 **39 = 33 + 3 + 3**、以及全部聚合数字**均不变**；理由从「已证明污染」降级为「**保守排除、理由被削弱**」。若要改为纳入，必须走单独批准的新分析。
- **时点（诚实标注）**：该规则是**文件名级、结果无关**的判定，不依赖任何指标。文档位置 `results/e25/ROBUSTAD_PROTOCOL_AUDIT.md` §6；该文件最后写入 2026-10-18 21:41，早于 MetalParts 首个 cell（21:52），但**协议与结果同处一个提交 `8ea1d73`**，Git 无法证明它先于任何结果写定。

→ 论文中只能写 **"145 file names are shared with source-train; a byte-level audit found 0/145 byte-identical (near-duplication untested); the split is excluded conservatively under the original protocol, documented before this sub-dataset was run but not provably pre-registered"**；**不得**写「同一批图像同时出现在 train 和 test0」，**不得**写「已证明无污染 / 无近重复」，也**不得**声称预注册。

---

## 8. Positive Domain: PCB

| 项 | 值 |
|---|---|
| 有效 target splits | **5**（test1 illumination / test2 white balance / test3 pose / test4 pose / test5 shadow） |
| 观测数 | 5 splits × 3 seeds = **15** |
| mean ΔAUROC | **+0.018352**（≈ +0.01835） |
| mean ΔAP | **+0.012358** |
| 逐 seed 域均值 | 0: +0.021713 / 1: +0.018500 / 2: +0.014843（seed 间 std **0.00344**） |
| 正向 split | **5 / 5**（均值口径）；**5 / 5 在 3 个 seed 上全正** |
| 基线水平 | target 平均 AUROC **0.4447**（test3 rotation 仅 0.349，**低于随机**） |
| **绝对水平** | **F0 0.444722 → R1 0.463074**（5 split 等权、3 seed 均值；`multiseed_analysis.json:per_domain.PCB`） |
| source split | test0 **−0.00807 ± 0.00192**（全 3 seed 负，不计入域均值） |

**边界**：增益发生在**接近失效的基线**上（headroom 极大），**不得**称为"外部验证成功"，也不得把绝对值与 MetalParts 横向比较。

---

## 9. Negative Domain: MetalParts

| 项 | 值 |
|---|---|
| 有效 target splits | **6**（test1 illumination / test2 pose / test3 pose / test4 scale / test5 background / test6 background） |
| 观测数 | 6 splits × 3 seeds = **18** |
| mean ΔAUROC | **−0.017251**（≈ −0.01725） |
| mean ΔAP | **−0.020277** |
| 逐 seed 域均值 | 0: −0.020816 / 1: −0.018524 / 2: −0.012413（seed 间 std **0.00434**） |
| 正向 split | **2 / 6**（test1 illumination +0.00139、test3 pose +0.00358） |
| 全 seed 负 | **3 / 6**：test4 scale（−0.02649）、test5 background（−0.05556）、test6 background（−0.02319） |
| 基线水平 | target 平均 AUROC **0.9261**（接近饱和） |
| 作废 split | test0：145 个文件名与 source-train 同名重叠，但字节级审计 **0/145 内容相同** ⇒ **仅文件名层面**；仍按原协议 `valid=0` **保守排除**（理由被削弱，见 §7.6） |

**几何事实**：MetalParts 原图 **2681×1500（16:9）**，按协议**方形 448×448** resize（`T.Resize` 短边在 1024-token 模型下不可用）；PCB 为 2048×2048 方图，方形 resize 与原路径**逐位等价（max|Δ|=0）**。
**边界**：MP 的负向**不得隐藏**，也不得在未验证前归因于 any 单一机制（见 §12）。

---

## 10. Claims We CAN Make

1. **R1-ILA 在既定 M²AD 光照变化协议下取得稳定的小幅改善。**
   *Across the frozen M²AD evaluation (10 categories, view120, 3 seeds, 300-bank), R1-ILA improves mean ΔAUROC on 9/10 categories (macro +0.00760, median +0.01080), with one stable regression (Doll, −0.03850).*
2. **当前冻结增广配置位于稳定有效区域。**
   *The frozen augmentation setting (strength 0.85/1.15, brightness+contrast+gamma, BETA = 0.1) is not an arbitrary choice: it ranks first of six strengths under a pre-registered stability rule on the three development categories, and strengths 0.95/1.05–0.85/1.15 form a plateau; re-running the top two strengths with three adapter seeds does not separate them.*
3. **多 seed 支持主要趋势不是单一随机种子偶然。**
   *The category-level sign pattern is reproduced across three bank seeds (7/10 categories positive in all three) and, externally, across three adapter-training seeds (PCB positive and MetalParts negative in every seed).*
4. **RobustAD 外部验证显示显著的 sub-dataset 依赖性。**
   *On RobustAD, with the adapter re-trained per sub-dataset under identical frozen hyperparameters, the effect is sub-dataset dependent: PCB improves (mean ΔAUROC +0.01835, 5/5 target splits positive in all seeds) while MetalParts degrades (mean ΔAUROC −0.01725, 2/6 positive). The overall macro is weighting-dependent (−0.001068 split-weighted, +0.000550 sub-dataset-weighted), so no overall direction is claimed.*
5. **PCB 域稳定为正，MetalParts 域稳定为负。**
   *The sign of the effect does not flip across three independent adapter-training seeds in either sub-dataset.*
6. **Doll 的退化是可复现的、可描述的（非随机噪声）。**
   *Doll is a reproducible regression (−0.03850, 0/3 seeds) consistent with a mild near-symmetric score shift; no causal mechanism is claimed.*
7. **E24 只审计、不重定义方法。**
   *E24 does not redefine the frozen method; it audits the robustness of the existing frozen configuration.*

---

## 11. Claims We CANNOT Make

1. **不得声称 RobustAD 整体显著提升。**（macro ΔAUROC **−0.00107**，t-interval [−0.0033, +0.0012] 跨 0；7/11 split 正向）
2. **不得声称所有工业域普遍泛化。**（2/3 sub-datasets；PiledBags 未覆盖；单 backbone）
3. **不得把 PCB / MetalParts 的差异归因于尺度、背景、材质、几何等具体机制**（未做机制验证；MP 同时存在方形 resize 混杂）。该差异**只支持"效果具有 sub-dataset 依赖性"**，不构成失败原因的因果解释。
4. **不得声称 direct cross-dataset transfer**（adapter 在每个 sub-dataset 上重训；见 §7.3）。
5. 不得声称 universal robustness / viewpoint invariance / improves all categories。
6. 不得把 E21 的 HOLD 写成通过（seed0 only、8/12 正、Car 240/330 与 Cube 090/330 为负）。
7. 不得把 image bootstrap CI 当作 cross-seed 不确定性。
8. 不得把 E24 的 S10 / A2 写成"应采用的配置"（单 seed 类型消融 + 3 seed 差距在噪声内）。
9. 不得忽略 Doll（主表按构造包含它）。
10. 不得把 M²AD Bird/Car/Cube 称为外部或独立数据集。
11. **不得声称表征保持项"保证异常信息保留"**：`L_preserve` 只约束**正常特征**与冻结特征的一致性，异常侧行为是间接后果，未经独立验证（E16 已否定 normal-dispersion 叙事）。
12. 不得把 E25 的 PCB/MetalParts 差异当作子数据集之外的"第二个外部数据集"或"跨数据集证据量"叠加。

---

## 12. Open Hypotheses / Discussion Only

| 编号 | 假设 | 状态 |
|---|---|---|
| H1 | PCB 的增益来自"基线接近失效、headroom 大"；MetalParts 近饱和故无空间甚至有害 | **未验证**；seed0 探索性 Spearman(B0, ΔAUROC) = −0.627 (p = 0.039, n = 11)，仅作 association |
| H2 | MetalParts 的负向由 16:9 → 448×448 各向异性压缩导致 | **未验证**；需 letterbox / 保长宽比对照（`results/e25/README.md` §9-H3） |
| H3 | background / scale 退化源于纹理统计改变而非缺陷可分性下降 | **未验证** |
| H4 | gamma 是增广的 active ingredient（压缩高光/暗部动态范围） | **未验证**；仅单 seed 类型消融，无特征层分析 |
| H5 | S10 相对 S15 的 +0.0027 是真实小效应 | **未验证**；差距 0.19×std，需 ≥8 seeds |
| H6 | A2（gamma-only）优于 A6（full）源于 contrast 放大正常纹理方差 | **未验证**；无方差分解 |

**统一措辞**：以上只能用 "consistent with" / "may be explained by" / "remains untested"，**不得**用 "because" / "proves" / "demonstrates that … is caused by"。

---

## 13. Exact Source Files（每个数字的唯一出处）

### 13.1 表 / 图

| 论文产物 | 文件 | 上游来源 |
|---|---|---|
| Table 主结果（10 类） | `results/paper_ready/paper_main_table.csv` | `results/e22/main_table_10cat_3seed.csv` |
| Table BETA 消融 | `results/paper_ready/paper_ablation_table.csv` | `results/e17/summary.csv` |
| Table 多视角 | `results/paper_ready/paper_multiview_table.csv` | `results/e21/analysis_cells.csv` |
| Fig A ΔAUROC 10 类 | `figures/main_A_delta_auroc_10cat.{png,pdf}` | `results/e22/figures/` |
| Fig B F0 vs R1 | `figures/main_B_f0_vs_r1_auroc_10cat.{png,pdf}` | `results/e22/figures/` |
| Fig C shift–gain | `figures/main_C_illumination_shift_vs_gain.{png,pdf}` | `results/e22/shift_gain_association.json` |
| Fig D 多视角热图 | `figures/supp_D_multiview_heatmap.{png,pdf}` | `results/e21/figures/` |
| Fig E Doll 失败 | `figures/supp_E_doll_failure_scatter.{png,pdf}` | `results/e23/figures/` |
| Fig E2 Doll 分布 | `figures/supp_E2_doll_delta_distributions.{png,pdf}` | `results/e23/figures/` |
| Fig F forest plot | `figures/supp_F_forest_plot_10cat.{png,pdf}` | `results/e22/figures/` |
| Fig E24 强度 | `results/e24/figures/e24_strength_vs_{dauroc,dap}.{png,pdf}` | `results/e24/tables/strength_ranked.csv` |
| Fig E24 类型 | `results/e24/figures/e24_type_ablation.{png,pdf}` | `results/e24/tables/type_ranked.csv` |
| Fig E25 域 | `results/e25/figures/e25_multiseed_domain_delta.{png,pdf}` | `results/e25/multiseed_summary.csv` |
| Fig E25 逐 target | `results/e25/figures/e25_multiseed_per_target.{png,pdf}` | 同上 |

### 13.2 关键数字 → 文件 → 聚合层级

| 数字 | 值 | 文件 | 聚合层级 |
|---|---|---|---|
| 主表 macro ΔAUROC | +0.00760 | `paper_main_table.csv`（Macro Avg 行） | 10 类**等权**均值（每类先做 3-seed 均值） |
| 主表 median | +0.01080 | 同上 | 10 类中位数 |
| Doll | −0.03850 | 同上 + `e22/bootstrap_e22.json` | 3-seed 均值；CI 来自逐图 Bootstrap(3000) |
| E21 macro pooled | +0.00142 | `e21/analysis_overall.csv` | 12 cell 池化（seed0） |
| E24 S15 seed0 | +0.015401 | `e24/strength_round1.csv` → `tables/strength_ranked.csv` | 3 类等权（单 seed） |
| E24 S15 3-seed | +0.013246 ± 0.014589 | `e24/tables/stability_multiseed.csv` | 3 类 × 3 seed 的 9 cell 均值 / std |
| E24 A2 | +0.019067 | `e24/tables/type_ranked.csv` | 3 类等权（单 seed） |
| **PCB +0.01835** | +0.018352 | `e25/multiseed_summary.csv` → `multiseed_analysis.json:per_domain.PCB` | **5 split 等权**均值；每 split 先做 3-seed 均值（15 obs） |
| **MP −0.01725** | −0.017251 | 同上 `per_domain.MetalParts` | **6 split 等权**均值（18 obs） |
| **macro（split 等权，主值）** | −0.001068 | `multiseed_analysis.json:overall.macro_d_auroc_mean_over_splits` | **11 个 split 等权**（5 PCB + 6 MP）；**不是子数据集等权** |
| **macro（子数据集等权）** | **+0.000550** | 本文件由两个 per-domain 全精度均值复算：(PCB + MP)/2 | **2 个 sub-dataset 等权**（"sub-dataset-weighted"，**不写 domain-balanced**） |
| **t-interval** | [−0.003329, +0.001193] | 同上 `macro_d_auroc_t_interval95` | 统计样本 = **3 个 per-seed macro**（每个 = 该 seed 下 11 split 等权均值），n = 3，df = 2，t = 4.303 |
| macro ΔAP | −0.005443 | 同上 `macro_d_ap_mean_over_splits` | 11 split 等权 |

**必须同时声明的三个口径事实**：
1. **两种总体均值并存，且符号相反**：split 等权 **−0.001068** vs 子数据集等权 **+0.000550**。差异来自 5 vs 6 个 split 的数量不平衡。**总体结论对聚合权重敏感**，因此不选择性突出正值：主值用 split 等权（与逐 split 表一致），并列给出子数据集等权，二者合起来只能支持「整体无一致方向」。
2. **两者均由冻结全精度结果计算**（cell JSON 的 `d_auroc` float / 逐图 score 复算），**不是**拿 5 位舍入的均值二次平均。代码路径 `multiseed_analysis.py` 在 split 级先求均值，再求总体均值。
3. **t-interval 的 n = 3（seed 级），不是 11 或 39**；它描述 **adapter 训练种子随机性**，不得把 split×seed 当独立重复，也不是跨 sub-dataset 泛化保证。它是描述性区间，**不是**显著性检验；不得写 "statistically significant"。

### 13.3 运行与验证记录

| 内容 | 文件 |
|---|---|
| E17 冻结参数 | `results/e17/final_method.json` |
| E18 等价性锚点 | `results/e18/main_table.csv` |
| E22 主表 / 验证 | `results/e22/main_table_10cat_3seed.csv`、`results/e22/verification.csv`（36/36） |
| E23 证据包校验 | `results/e23/verification.csv`（历史 38/38）；当前 **45 项**套件见 `verification_reconciliation_20261019_e22_erratum.csv`；脚本 `scripts/e23_verify.py`＋`scripts/verification_reverse_tests.py` |
| E24 判定 | `results/e24/final_strength_analysis.json`（`s15_decision.keep_s15 = true`） |
| E24 复现校验 | 同上 `frozen_config_reproduction`（`pass: true`, `max_abs_diff: 0.0`） |
| E25 复现校验 | `results/e25/multiseed_analysis.json:verification`（`max_abs_diff: 0.0`） |
| E25 运行台账 | `results/e25/run_manifest.json`、`partial.json`（6 条 (sub-dataset, seed) 记录） |
| E25 数据校验 | `results/e25/download_verification.json`（missing = 0, bad_size = 0） |
| E25 几何审计 | `results/e25/geometry_PCB.json`（方形，逐位等价）、`geometry_MetalParts.json`（16:9，非等价，属协议规定） |
| 冻结文件 md5 基线 | `scripts/e23_verify.py:LEGACY_MD5`（V29/V30 检查 6 个历史遗留文件） |

### 13.4 叙述性文档（不作为数字来源）

`results/e17/README.md`、`results/e18/README.md`、`results/e19/README.md`、`results/e21/README.md`、`results/e22/README.md`、`results/e23/README.md`、`results/e24/README.md`、`results/e25/README.md`、`results/e25/ROBUSTAD_PROTOCOL_AUDIT.md`、`results/paper_ready/paper_claims.md`、`paper_limitations.md`、`paper_evidence_map.md`。

---

## 附：本文件与验证脚本的关系

- `scripts/e23_verify.py` 的 V21–V24 检查 `paper_claims.md` 的字符串与 `paper_limitations.md` 的 5 条否定项（其中 "untested" 一词保留在 L3 的**时间限定**表述中，语义已由 L6 更新为"条件性已验证"）。**V24 另独立复算** Doll 三种子 ΔAUROC 的 `ddof=1` 样本标准差（**0.004638**），并要求同一数值同时出现在 L5、`results/e23/doll_seed_summary.csv` 与 `paper_main_table.csv`；**V17** 要求 Macro Avg 行六个 std 单元格**留空**（不得填 0，`mean(category std)` 不是 std）。
- **2026-10-19 勘误轮验证**：Doll 样本 std（0.00379→0.004638）+ Macro Avg std 留空 + E18 锚点修正 + E25 MetalParts `test0` 措辞勘误 → `results/paper_ready/verification_reconciliation_20261019_post_errata.csv`（**38 项 / 0 FAIL**，检查总数不变、容差未放宽）。
- **V30/V31 已于 2026-10-19 修订**（见 `results/e23/VERIFICATION_SEMANTICS.md`）：
  - V29 = 六个 legacy 路径**字节未变**（文件缺失 → FAIL，不再抛异常中断）；
  - V30 = **当前**处置与对账台账一致（3 项有意未提交 + 2 项经授权入库 + 1 项被忽略）。范围**仅限这六个 legacy 路径**（不宣称自动审计全仓）：缺失文件、状态不符、已入库文件被再次修改 → FAIL；其他未跟踪文件**单独报告**，不计为 legacy 违规。
  - V31 = **历史（逐提交核验）**：E21→E23 区间（`8957237..5b24193`）内无 legacy 路径变动；该检查点之后，只有授权提交 `9781e9d` 的两个授权路径可以出现，变动类型仅限 **A/M**，且**该历史提交对象内的 blob** 必须等于 Stage-0 md5。未授权提交、**改后还原**、删除（D）、重命名（R）、以及合并提交中的同类变动（按第一父核验）均判 FAIL；所有 Git 子进程检查退出码，命令失败**不得**等价于“空历史”。
  - **反向测试**：`scripts/verification_reverse_tests.py` 在隔离临时仓库中覆盖 11 个场景（缺失、台账错、未授权修改、**改后还原**、删除、重命名、Git 失败、合并、多条目解析）；记录见 `verification_reverse_tests_20261019.csv`；V44 每次运行都会重新执行它们并要求按预期 FAIL/PASS。
  - 历史 38/38 记录保留在 `results/e23/verification.csv`；当前运行写入 `results/paper_ready/verification_reconciliation_<date>.csv`（`--out`），当前套件为 **45 项**。
- 本文件**不被任何自动生成脚本覆盖**；若与生成产物冲突，以本文件为准并修正 `scripts/paper_ready_build.py`。
