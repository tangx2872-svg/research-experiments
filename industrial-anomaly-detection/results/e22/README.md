# E22 — M²AD 全类别多 seed 验证（10 categories × 3 seeds × view120，冻结 R1-ILA）

**本目录包含两个阶段，请分开阅读**：

- **Part I（主结果）**：E22-P2 GPU 验证 —— 30/30 cells 完成、独立验证 **36 checks / 0 FAIL**、Verdict = **HOLD**（待用户裁决）
- **Part II（数据前置）**：E22-P0/P1 —— 7 个新类别 `l0_448` 缓存获取（2026-10-15 白天；**原始记录保留未改动**，含已修复的 status 写冲突与标签陷阱）

---

# Part I — E22-P2：10 类 × 3 seeds 冻结方法验证（主结果）

**日期**：2026-10-15 21:46 → 23:42（矩阵）｜2026-10-15 23:54 → 2026-10-16 00:04（分析/图）｜2026-10-16 00:11 → 00:13（独立验证）
**执行**：`scripts/e22_overnight_run.sh`（无人值守驱动）→ `scripts/e22_analysis.py` → `scripts/e22_figures.py` → `scripts/e22_verification.py`
**协议**：[`protocol.md`](protocol.md)（2026-10-15 21:40 CST 冻结，先于任何新类别模型结果）
**方法**：F0（frozen DINOv2-B，1-NN-max）/ R1-ILA（`z' = z + Wz`，BETA = 0.1，`results/e17/final_method.json` 冻结；本实验未调任何参数）
**上游 commit**：`88883d0`（P0/P1 取数）；E22 协议冻结 commit `8273525`
**Verdict**：**HOLD**（预注册 STRONG GO 4 条中 3 条满足，c4 失败：Doll 3-seed mean ΔAUROC = −0.0385 < −0.03）

## 1. 论文阶段

Phase II — Paper Validation 的**跨类别外部有效性扩展**阶段（E20-A 三条扩展路径中的 **M1**）。方法搜索已 CLOSED（E16/E17 冻结 R1-ILA）；本实验是**验证性**的，不是搜索性的：不引入新模块、不调参、不挑 seed、不挑类别。

## 2. 本实验服务论文的哪句话

> R1-ILA（DINOv2-B bank 归一化精修）的增益不是 Bird / Car / Cube 三类的偶然现象：在 M²AD 的 10 个类别 × 3 个独立 bank seed × view120 评测集上，其整体方向仍为正，且正类占多数。

**边界（先说清它不检验什么）**：
- 不检验视角不变性（E21 负责，结果 HOLD）；
- 不检验跨数据集泛化（E20-B，STOP）；
- 不做因果归因（§5.6 的 illumination shift ↔ gain 只是 n = 10 的 exploratory association）；
- 不声称"所有类别都提升"（Doll 稳定为负）。

## 3. 预注册判据与判定（协议 §6，写于任何新类别结果产生之前）

| 条件（STRONG GO 需全部满足） | 实测 | 满足 |
|---|---|---|
| ① ≥ 7/10 类别的 3-seed mean ΔAUROC > 0 | **9/10** | ✅ |
| ② 10 类 macro mean ΔAUROC > 0 | **+0.00760** | ✅ |
| ③ 10 类 median ΔAUROC > 0 | **+0.01080** | ✅ |
| ④ 无任何类别 3-seed mean ΔAUROC < −0.03 | **1 个（Doll −0.03850）** | ❌ |
| STOP 条件（≤4/10 为正；macro ≤ 0 且无正趋势；多个严重负类别） | 未触发 | — |

→ **HOLD**（`verdict.json`）。

> **事实/口径说明**：协议 §6 对 HOLD 的散文描述写的是"5–6/10 类为正且总体为正趋势；或结论明显依赖 illumination shift"这一分支。本次实测落在这个描述**之外**：它是"STRONG GO 因 c4 失败、STOP 未触发"因而被判为 HOLD（判定实现即 `STRONG GO if all 4 else STOP if stop-condition else HOLD`）。**该差异属预注册文本的边界情形，必须如实记录，不得把本次结果写成 STRONG GO，也不得写成 STOP。**

## 4. 数据与协议（要点；细节见 `protocol.md`）

- **矩阵**：10 categories × 3 seeds = **30 cells**，每 cell = F0 + R1-ILA；其中 **9 cells 复用** E16 历史结果（Bird/Car/Cube × seed0/1/2），**21 cells 新算**（Dice, Doll, Holder, Motor, Ring, Teapot, Tube × seed0/1/2）。
- **评测集**：M²AD **view120** 官方 test 全部原始记录（700/类，Holder 690），缓存 `l0_448`。
- **标签**：元数据 `image_anomaly`（**不是**目录名）+ `detectable != False` 过滤；runner 内硬断言禁止目录名判类。
- **冻结超参数**：BETA = 0.1｜LR = 1e-3（AdamW）｜MAX_STEPS = 300 / PATIENCE = 30 / EVAL_EVERY = 5｜batch 8｜train/val = 240/60｜bank = 300 张，`E8.bank_for(pool, seed)`（`BANK_SUBSET_SEED = 20261009`）｜augmentation = brightness/contrast/gamma × {0.85, 1.15}（6 变体）｜可训练参数 589,824。
- **禁用**：调 BETA/LR/aug/bank、换 seed、加模块、改 scoring、为失败类别专门重训、删除负结果、挑最好 seed 汇报。E22 只新增 `scripts/e22_full_category_validation.py`（复制 `run_cell` 路径并注明来源、移除历史断言改为条件比较），冻结文件未改动（验证 V31 独立确认）。
- **复现 gate（矩阵开跑前的硬门槛，失败即 STOP）**：
  - **Gate A（跨代码路径，bit 级）**：`Bird|0` 经冻结 `e16_qualify.run_cell` 与 E22 复制路径在同一进程执行 → `max|Δscore| F0 = 0.0，R1 = 0.0`，`bit_identical = True`；
  - **Gate B（对历史，tol 1e-6）**：`Bird|0 / Car|0 / Cube|0 / Car|1` 4 cells 重算 vs `e16_partial.json` → 逐图分数差 **0.0**，AUROC/AP 差 **0.0**，标签一致；
  - `repro_gate.json: all_pass = true`。**Gate 通过后矩阵才开始**（`repro_gate.json` mtime 22:09:00 → 矩阵 22:09:06 起跑）。
- **复用校验**：9 个历史 cell 的 `reuse_verification` 全部为 true（labels / illum / specimen / n_test / F0·R1 指标重算 / bank_hash vs E21 / n_ok·n_ng），且逐图分数与历史一致（Gate B）。
- **标签陷阱（实测命中）**：Dice view120 test 按目录统计为 `NG/` 500 + `Good/` 200，而元数据 `image_anomaly` 为 0→300 / 1→400（**100 条位于 `NG/` 却标 0**）；detectable 过滤剔除 12 条（全部为 `image_anomaly=1` 且 `detectable=False`），保留 300 good / 388 anomaly。独立验证 V06/V11 确认 30/30 cell 的标签与元数据逐一相同、且该陷阱确实存在（未触发错误判类）。

## 5. 结果

### 5.1 逐类主表（`main_table_10cat_3seed.csv`；3-seed mean ± std）

| category | source | F0 AUROC | R1 AUROC | **ΔAUROC mean ± std** | pos seeds | ΔAP mean ± std | pos seeds |
|---|---|---|---|---:|---|---:|---|
| Bird | HISTORICAL_REUSED | 0.7513 | 0.7772 | **+0.02586 ± 0.01354** | 3/3 | +0.02371 ± 0.01155 | 3/3 |
| Car | HISTORICAL_REUSED | 0.7998 | 0.8110 | +0.01123 ± 0.00728 | 3/3 | +0.01115 ± 0.00187 | 3/3 |
| Cube | HISTORICAL_REUSED | 0.8638 | 0.8664 | +0.00265 ± 0.01421 | 2/3 | +0.00413 ± 0.00746 | 2/3 |
| Dice | NEW | 0.9046 | 0.9079 | +0.00336 ± 0.00078 | 3/3 | +0.00212 ± 0.00150 | 3/3 |
| **Doll** | NEW | 0.7125 | 0.6740 | **−0.03850 ± 0.00464** | **0/3** | −0.01302 ± 0.00786 | 0/3 |
| Holder | NEW | 0.8932 | 0.9036 | +0.01036 ± 0.00445 | 3/3 | +0.01122 ± 0.00384 | 3/3 |
| Motor | NEW | 0.8262 | 0.8342 | +0.00807 ± 0.00765 | 2/3 | +0.00547 ± 0.01206 | 1/3 |
| Ring | NEW | 0.8933 | 0.9136 | +0.02031 ± 0.00363 | 3/3 | +0.01379 ± 0.00305 | 3/3 |
| Teapot | NEW | 0.8723 | 0.8896 | +0.01730 ± 0.00729 | 3/3 | +0.01070 ± 0.00312 | 3/3 |
| Tube | NEW | 0.8712 | 0.8865 | +0.01532 ± 0.00499 | 3/3 | +0.00846 ± 0.00264 | 3/3 |

### 5.2 胜率与总体（`win_rates.json`）

- **ΔAUROC 胜率 9/10**；**ΔAP 胜率 9/10**；两项同时为正 9/10。
- macro mean ΔAUROC **+0.00760**，median **+0.01080**；macro mean ΔAP **+0.00777**，median **+0.00958**。
- 最大增益 **Bird +0.02586**；最大回退 **Doll −0.03850**。
- 正 seed 数分布：**3/3 → 7 类**、2/3 → 2 类（Cube, Motor）、1/3 → 0、0/3 → 1（Doll）。
- 30 个 cell 中 ΔAUROC < 0 者 **5 个**（`failure_cases.md` 全列，未删）。

### 5.3 不确定性（两种口径，分开报告，**不得合并**）

**(a) image-level paired bootstrap**（测试集抽样不确定性；`n_boot = 3000`，rng = 20261015，paired indices，detectable-filtered；`bootstrap_e22.json`）：

| category | ΔAUROC CI（seed-mean scores） | p(ΔAUROC>0) | ΔAP CI | p(ΔAP>0) |
|---|---|---|---|---|
| Bird | [+0.0085, +0.0436] | 0.998 | [+0.0038, +0.0423] | 0.991 |
| Car | [−0.0052, +0.0280] | 0.917 | [−0.0028, +0.0218] | 0.942 |
| Cube | [−0.0144, +0.0150] | 0.530 | [−0.0054, +0.0114] | 0.757 |
| Dice | [−0.0076, +0.0155] | 0.746 | [−0.0047, +0.0096] | 0.760 |
| **Doll** | **[−0.0661, −0.0140]** | **0.001** | [−0.0335, +0.0077] | 0.097 |
| Holder | [−0.0004, +0.0196] | 0.971 | [+0.0034, +0.0190] | 1.000 |
| Motor | [−0.0095, +0.0207] | 0.753 | [−0.0107, +0.0094] | 0.429 |
| Ring | [+0.0053, +0.0275] | 0.999 | [+0.0045, +0.0195] | 0.999 |
| Teapot | [+0.0015, +0.0277] | 0.986 | [+0.0028, +0.0163] | 0.998 |
| Tube | [−0.0005, +0.0246] | 0.970 | [−0.0000, +0.0143] | 0.974 |

**(b) 跨 seed 变异**：即 §5.1 的 ΔAUROC std（3 个独立 bank）。这是**与 (a) 完全不同的量**，本 README 与论文中不得把 (a) 当作"跨 seed 不确定性"。

> 注：Doll 的两种口径**符号不一致**（ΔAUROC 显著负 p = 0.001；ΔAP 的 CI 跨 0，p = 0.097）。事实按两种口径分别记录；不得只引用有利的一边。

### 5.4 seed 稳定性（描述性标签；不删任何类别）

`stable_positive`（3/3 且 std ≤ 0.01）：Car, Dice, Holder, Ring, Teapot, Tube；`stable_negative`：Doll（0/3）；`seed_sensitive`（std > 0.01）：Bird（3/3 但 std 0.0135）、Cube（2/3, std 0.0142）；`mixed`：Motor（2/3, ΔAP 仅 1/3）。逐 seed 显著性见 `failure_cases.md` 与 `bootstrap_e22.json`（例：Doll 3 个 seed 全部显著负 p = 0.000/0.004/0.004；Bird seed0 不显著 p = 0.865 而 seed1/2 显著）。

### 5.5 illumination（描述性，`illumination_breakdown.csv` / `illumination_spread.csv`）

- **normal 图像分数的跨光照 spread** 在 **9/10 类** 上被 R1-ILA 压低（R1/F0 比值 0.32–0.94）；唯一例外 **Dice 1.407**。
- 只有 spread 可比；F0 与 R1 的绝对分数水平不可比（R1-ILA 改变特征尺度）。
- 逐 illumination 分组的 AUROC/AP 见 `illumination_breakdown.csv`（10 类 × 3 seeds × 2 方法 × (ALL + 10 组) = 660 行，独立验证 V36 PASS）。
- **Fig8** 的 best/median/worst 类别（Bird / Holder / Doll）为 **post-hoc descriptive visualization**，不构成选择依据。

### 5.6 偏移 ↔ 增益关联（exploratory，n = 10 categories，**禁止因果表述**）

| association | Pearson r (p) | Spearman ρ (p) |
|---|---|---|
| D_shift（F0 patch 相对位移，seed-mean）vs ΔAUROC | +0.371 (0.291) | +0.576 (0.082) |
| D_shift（seed0）vs ΔAUROC | +0.363 (0.302) | +0.515 (0.128) |
| B_centroid（seed-mean）vs ΔAUROC | +0.154 (0.671) | +0.527 (0.117) |
| D_shift（seed-mean）vs ΔAP | +0.523 (0.121) | +0.515 (0.128) |

方向为"光照偏移越大 → 增益略大"（弱正关联，全部未达 p < 0.05）。**Doll 的 D_shift = 0.182 处于 10 类中位数附近（median 0.183），因此"负向来自高光照偏移"这一解释与数据方向相反、不被支持**（该句为解释，不是因果结论）。

### 5.7 图（`figures/`，8 张 × PNG + PDF）

`fig1_f0_vs_r1_auroc_10cat`（10 类 F0 vs R1）· `fig2_delta_auroc_10cat` · `fig3_delta_ap_10cat` · `fig4_heatmap_delta_auroc`（10×3）· `fig5_positive_seed_ratio` · `fig6_illumination_shift_vs_gain`（exploratory, n = 10）· `fig7_forest_plot`（seed-mean bootstrap CI）· `fig8_illumination_best_median_worst`（post-hoc）。

## 6. 负证据（全部保留，无一条被移除或改写）

1. **Doll（严重负，0/3 seeds）**：ΔAUROC −0.03850（CI [−0.0661, −0.0140]，p = 0.001）；3 个 seed 全部显著负；ΔAP −0.01302（CI 跨 0，p = 0.097）。
2. **最差单 cell**：`Doll|seed0` ΔAUROC **−0.04385**、ΔAP −0.00947（F0 0.71941 → R1 0.67556）。
3. **单 seed 为负的 cell**：`Cube|1` ΔAUROC −0.01229 / ΔAP −0.00321；`Motor|1` ΔAUROC −0.00025 / ΔAP −0.00258。
4. **AUROC/AP 口径分歧**：Doll（AUROC 显著负、AP 不显著）、Motor（ΔAUROC 2/3 正、ΔAP 1/3 正）。
5. 明细：`failure_cases.md`（含 5 个负 cell 全表、最大跨 seed 波动类别 Cube）。

> 独立验证 V30 确认 `failure_cases.md` 列出了全部 5 个负 cell（5/5）。

## 7. 异常情况与实现缺陷（如实记录）

### 7.1 【执行异常，无科研影响】watchdog 启动竞态 → 重复驱动实例

- **事实**：21:46:13 启动驱动 `overnight1`；21:46:26 watchdog（第一修订版）把"矩阵进程尚未出现（其实还在 repro gate 内）"误判为 runner 已死，**又启动了一个驱动 `recovery1`**（PID 358066，`driver.pid` 记录的就是它）。重复实例的最后一条日志时间 21:46:48（停在 Gate A 内部、`e16_qualify.run_cell` 的 xFormers 警告之后，无任何 gate 结果行）；`watchdog_state.json` 记录 "Detected at 21:47, duplicate driver + old watchdog were terminated with SIGTERM"。`overnight1` 独立完成 gate 与全部 21 cells（其日志含 114 条心跳与 `matrix rc=0`，23:42:30 结束）。
- **处置**：修复 watchdog 的存活判定（`pgrep` runner+driver、`driver.pid`、status pid、日志 mtime）并加 90 s startup grace；21:48:56 以修复版重启 watchdog，recovery 计数重置为 0（`watchdog_state.json` 事件 `bootstrap_race_duplicate_driver`）。
- **影响（已核验）**：重复实例**未写入任何 cell、未覆盖任何结果**（21 个 NEW cell 全部由 `overnight1` 单次产生；`repro_gate.json` 写于 22:09:00，晚于重复实例终止时间）；同一时段唯一被两个实例共同写入的文件是 `data_preflight.csv`（只读检查的产物，两实例的逐类 preflight 结果行相同，当前文件 ALL=PASS）。**无需重跑，历史结果不受影响**。

### 7.2 【实现缺陷，已修复，无结果影响】独立验证 V29 自引用

- **bug 位置**：`scripts/e22_verification.py` 的 V29 把 `verification.csv`（该脚本自身的输出）列入"必须存在的产物"清单，而该文件在本函数末尾才写出 → 首次运行必然 FAIL。
- **为什么是 bug**：检查依赖自己的输出，构成自引用；它检测不到真正的产物缺失问题之外，任何单次运行都必然报告 1 个假 FAIL。
- **影响范围**：仅验证报告本身；不涉及 cell、指标、判据。第一次运行（`36 checks, 1 FAIL`，FAIL 仅 V29 且 `observed=['verification.csv']`）已保留为 `verification_pre_fix_run1.csv` 与日志 `logs/verification_20261016.log`。
- **修复**：V29 的清单改为"上游产物"，并在代码内注明原因（2026-10-16）。**无需重跑任何历史实验**；重跑验证脚本后为 **36 checks / 0 FAIL**（`logs/verification_20261016_run2.log`）。

### 7.3 无其他异常

30/30 cells `status = OK`；无 NaN、无 OOM、无重试、无中断（矩阵 21/21 一次跑完）；steps 早停生效（见 §8）；期间磁盘自由空间恒定 12.42 GB。

## 8. 成本核算（事实，供后续扩展预算）

| 项 | 实测 |
|---|---|
| NEW cells | **21 个**，GPU 累计 **92.9 min**，均值 **265.6 s/cell**（min 260.3 / max 271.0） |
| 历史复用 cells | 9 个（不新增 GPU；历史记录合计 40.3 min） |
| 复现 gate | 约 23 min（21:46:13 → 22:09:00，含 Gate A 双路径 bit 对拍 + Gate B 4 cells 重算） |
| 矩阵墙钟 | 22:09:06 → 23:42:29（**1h33m23s**）；驱动总时长 21:46:13 → 23:42:30（1h56m17s） |
| 单类别成本 | 约 **13.3 min GPU / 类别 × 3 seeds**（4.43 min per cell） |
| 显存 | `peak_gpu_mb = 5371.4`（30 cells 完全相同，即固定占用）；训练期观测 GPU util 94–100%，cell 之间 0% |
| steps | `steps_used ∈ {44, 49, 54, 59, 104, 124}`，远低于 MAX_STEPS = 300（early stop 生效） |
| 磁盘 | `results/e22/` 共 **3.9 MB**（图 + CSV + JSON）；运行期间 `/root/autodl-tmp` 自由空间 12.42 GB 不变（本阶段不取数） |

## 9. 独立验证（`verification.csv`：36 checks，**0 FAIL**）

验证脚本独立于 runner：从元数据与逐图分数**重新推导** scope / bank hash / 标签 / 指标 / 判据，并与产物比对。覆盖：完整性（10 类 30 cells、无重复、view120 only）· 标签与 detectable 过滤逐条一致（30/30）· bank hash 依冻结 `bank_for` 复算（30/30）· AUROC/AP 从逐图分数重算（tol 1e-12）· 冻结超参数与增强表一致 · Gate A/B 与 `all_pass` · 复用校验 9/9 · 主表/胜率/verdict 独立重算一致 · bootstrap 确定性且与存储 CI 完全相同 · 8 图齐备 · 产物齐备（V29）· 负结果全列（V30）· 冻结文件与 E21 未被触碰 · 6 个 legacy 文件状态保持 · 缓存聚合 sha256 未变（`1569cc57…e47812`）· illumination 明细 660 行完整。

## 10. 结论（区分事实 / 解释 / 假设）

**事实**：
- 10 类中 9 类的 3-seed mean ΔAUROC 与 ΔAP 同时为正；macro mean ΔAUROC +0.00760、median +0.01080（ΔAP +0.00777 / +0.00958）。
- 7/10 类 3 个 seed **全部**为正；2/3 → Cube、Motor；0/3 → Doll。
- Doll 三 seed 一致为负且 bootstrap 显著（ΔAUROC CI [−0.0661, −0.0140]，p = 0.001）；30 cells 中 5 个负 cell 全部保留。
- 逐类 bootstrap CI：Bird / Ring / Teapot（并 Holder 的 ΔAP）**不跨 0**；Car / Cube / Dice / Motor / Tube 跨 0。
- 预注册 4 条 STRONG GO 判据 3 条满足、c4 失败 → HOLD。

**解释**：R1-ILA 的方向性收益在 10 类上大体可重复，但**不是普适的**：存在 1 个稳定负类别（Doll）与 2 个 seed 敏感类别（Cube、Motor），且类别间幅度差异大（+0.003 → +0.026）。因此论文中可写"跨类别整体方向为正、正类占多数"，不可写"对所有类别有效"。

**假设（尚未验证，不得写成结论）**：Doll 的负向可能与缺陷在 view120 下的可见性/标签结构或该类别特征分布有关——本实验未做机制验证；光照偏移解释与数据方向相反（§5.6）。

## 11. 出口条件判断

- STRONG GO（4 条全满足）**未达成**；STOP 条件（≤4/10 为正、macro ≤ 0、多个严重负类别）**未触发** → 判定 **HOLD**、`verdict.json` 记录，**不自动补任何实验**（不加类别、不加 seed、不调参）。
- HOLD 的含义：可以写 §10"事实"部分的**无选择性**结果与负证据；**不得**写 broad generalization / 普适提升 / 因果归因；也不得把 HOLD 写成 "paper failed"。
- 边界情形提示（已写入 §3）：本次落在预注册 HOLD 散文描述的边界之外（9/10 正但含 1 个 severe negative），**该差异需用户裁决如何在论文中表述**。

## 12. 对论文证据链的贡献

- **完成的格子**：跨类别外部有效性的**无选择性正式矩阵**（10 类 × 3 seeds × view120，含 21 个新算 cells、9 个经复现 gate 复用、逐图分数与独立验证）。
- **被排除的替代解释**：(i) "增益只是 Bird/Car/Cube 的偶然"——9/10 类为正，不成立；(ii) "R1-ILA 对所有类别都有效"——Doll 稳定负，同样不成立（负证据必须写进论文）。
- **下一格（待裁决，不自动执行）**：(a) 按 HOLD 收口，把 10 类正/负结果写成主表 + limitation；(b) 若要解释 Doll，需要新的机制实验（未批准）；(c) E20-A 的 M2（4-view，零获取成本）扩展。

## 13. 复现命令

```bash
cd /root/autodl-tmp/research-experiments/industrial-anomaly-detection
# 0) 只读数据前置 + 复现 gate（矩阵开跑前必须通过）
/root/miniconda3/bin/python -u scripts/e22_full_category_validation.py --preflight
/root/miniconda3/bin/python -u scripts/e22_full_category_validation.py --repro
# 1) 无人值守矩阵（resume-safe：已 OK/NEW 的 cell 自动跳过）
setsid nohup bash scripts/e22_overnight_run.sh <tag> > results/e22/logs/driver_stdout.log 2>&1 &
#    （watchdog 可选；注意：grace 不得设为 0 —— 见 §7.1）
/root/miniconda3/bin/python -u scripts/e22_watchdog.py
# 2) 分析 → 3) 图 → 4) 独立验证
/root/miniconda3/bin/python -u scripts/e22_analysis.py
/root/miniconda3/bin/python -u scripts/e22_figures.py
/root/miniconda3/bin/python -u scripts/e22_verification.py
```

## 14. 产物清单（Part I）

| 文件 | 说明 |
|---|---|
| `protocol.md` | 预注册协议（含 amendment 声明与判据） |
| `cells/*.json` | 30 个 cell（逐图分数、bank/adapter hash、runtime、steps、标签溯源） |
| `all10_3seed_cells.csv` / `live_results.csv` | 逐步/逐 cell 汇总（含 source 与 status） |
| `main_table_10cat_3seed.csv` | 逐类 mean ± std 主表 |
| `category_summary.csv` / `win_rates.json` / `verdict.json` | 胜率、稳定性标签与判定 |
| `bootstrap_e22.json` | paired-image bootstrap（seed-mean 与 per-seed） |
| `illumination_breakdown.csv` / `illumination_spread.csv` / `illumination_summary.csv` | 光照分组明细与 spread |
| `shift_gain_association.json` | 偏移 ↔ 增益关联（exploratory） |
| `statistical_summary.md` / `failure_cases.md` | 12 问答与全部负案例 |
| `figures/` | Fig1–Fig8（PNG + PDF） |
| `verification.csv`（36/36）+ `verification_pre_fix_run1.csv`（证据） | 独立验证与修复前快照 |
| `data_preflight.csv` / `repro_gate.json` | 数据门与复现门记录 |
| `status.json` / `watchdog_state.json` / `logs/` | 运行状态、watchdog 事件与全部原始日志 |
| `scripts/e22_full_category_validation.py` · `e22_analysis.py` · `e22_figures.py` · `e22_verification.py` · `e22_watchdog.py` · `e22_overnight_run.sh` | E22 专用脚本（冻结文件未改） |

**术语**：F0 = frozen DINOv2-B 特征 1-NN-max 基线；R1-ILA = BETA = 0.1 的 illumination adaptation 冻结方法；cell = (category, seed) 单元（每 cell 含 F0 与 R1）；seed-mean scores = 先按 seed 平均逐图分数再算指标的口径（bootstrap 表所用）；image bootstrap ≠ cross-seed 不确定性。

---

# Part II — E22-P0/P1：M2AD 7 个新类别 `l0_448` 缓存获取（数据前置阶段）

> **后续状态（2026-10-16 补记，本 Part 正文未改动）**：本 Part 记录的取数完成后，用户批准了 overnight 全类别验证；GPU 阶段（7 新类 × 3 seeds + 3 历史类复用）已于 2026-10-15 夜执行完毕，结果见 **Part I**。因此本 Part §12 第 1 条的"是否执行 GPU 单元格"已由 Part I 回答（已执行，Verdict = HOLD）；第 2、3 条（E18 风格 baseline / M2·M3 路径）仍待裁决。

**日期**：2026-10-15（18:33:06 → 21:13:28，含 13 s dry-run）
**执行**：`scripts/e22_manifest.py`（P0，元数据盘点）、`scripts/e22_fetch.py`（P1，流式选择性取数）
**上游 commit**：`8957237`（E21: validate R1-ILA across M2AD views）
**输出目录**：`results/e22/`、缓存写入 `data/m2ad/e7_cache/l0_448/<Category>/`（gitignored）

---

## 1. 论文阶段

证据链中的**跨类别外部有效性**扩展阶段。E20-A 盘点给出的三条扩展路径（M1 新类别 / M2 4-view / M3 Bird 12-view）中，本实验执行 **M1 的第一步：数据获取**。尚未进入 GPU 评估阶段。

## 2. 本实验要服务的那句话（先说清楚它不检验什么）

**本实验不检验任何性能断言，也不提供任何性能证据。**

它只把下面这句话从「不可检验」变为「可检验」：

> R1-ILA（DINOv2-B bank 归一化精修）在 M2AD 上的增益不限于 Bird / Car / Cube 三类。

该断言的真假，必须由后续 GPU 单元格（新类别 × 冻结 seeds 的 F0 与 R1-ILA）判定。E22 只负责消除"数据不可得"这一替代解释。

## 3. 实验目的

在不整包解压（单类 zip 15.0–17.5 GB 十进制，磁盘仅剩 16 GB）的前提下，只获取**冻结 R1-ILA 协议实际读取的像素**，并逐类施加硬 gate，使后续 GPU 阶段可执行：

- bank 池：`sorted(train_specs)[::3][:10]` × 12 views × 10 illum = **1200 图/类**（冻结规则，`e4d1_m2ad_loader.bank_and_val_specimens`）
- 评测集：view120 test 官方划分全部原始记录（**700 图/类**，Holder 690）
- **不获取**：`l0_256`、`clahe_mild/medium`、GT mask、原图（R1 增强在 448 图上施加，见 `e13a_screen.prep_image`）

## 4. 数据与参数

| 项 | 值 |
|---|---|
| 数据集 | M2AD（HF 镜像 `hf-mirror.com/datasets/ChengYuQi99/M2AD`），元数据 `data/m2ad/jsons_extract/meta_unsupervised.json` |
| 新类别（7） | Dice, Doll, Holder, Motor, Ring, Teapot, Tube |
| 缓存种类 | 仅 `l0_448` |
| 派生路径 | `TF.resize(rgb, [448,448], antialias=True, BICUBIC)` → WebP lossless q100 method 0（与 `e8_fetch.write_image_caches` 逐字一致，`.png` 扩展名、WebP 容器，与既有 Car/Cube 缓存签名相同） |
| 随机性 | 取数本身无随机性（路径全部由官方元数据决定）；下游 bank 抽样使用冻结 seed `BANK_SUBSET_SEED=20261009` |
| 环境 | conda `base`，Python 3.12.3，PyTorch 2.8.0+cu128，torchvision 0.23.0+cu128，Anomalib 2.6.2，OpenCV 5.0.0，RTX 3090 24 GB（本实验 CPU/网络密集，未使用 GPU） |

## 5. 实验过程

```
# P0 元数据盘点（只读）
python scripts/e22_manifest.py
# P1 dry-run（范围确认，无下载）
python scripts/e22_fetch.py --categories Dice,Doll,Holder,Motor,Ring,Teapot,Tube --dry-run
# P1 正式取数（后台，脱离会话）
setsid nohup /root/miniconda3/bin/python -u scripts/e22_fetch.py \
  --categories Dice,Doll,Holder,Motor,Ring,Teapot,Tube \
  > results/e22/logs/fetch_20261015_183306.log 2>&1 &
```

- dry-run 确认：pool=1200（7/7 类）、test120=700（Holder 690）、缺失=1900/1890，磁盘 16.2 GB 可用。
- 后台 PID 335905；`results/e22/status.json` 在每个类别完成后原子写入；逐成员进度每 250 成员打印（members / kept / streamed / 速率 / 已用时间 / 剩余磁盘）。
- resume：重新运行前先扫描既有 `l0_448` 文件，已存在者从待取集合中剔除；单成员解码/写盘失败即中止（不静默跳过）。
- gate：取数后逐文件校验存在性 + 448×448 + 色彩模式。

## 6. 结果

### 6.1 逐类（事实）

| 类别 | pool | test120 | 归档(GiB) | 流式读取(GiB) | 耗时 | 缓存体积 | gate missing | bad_shape |
|---|---|---|---|---|---|---|---|---|
| Dice | 1200 | 700 | 15.54 | 15.53 | 16m44s | 401 MB | 0 | 0 |
| Doll | 1200 | 700 | 15.68 | 15.68 | 17m04s | 390 MB | 0 | 0 |
| Holder | 1200 | 690 | 16.27 | 16.27 | 18m56s | 391 MB | 0 | 0 |
| Motor | 1200 | 700 | 13.93 | 13.93 | 16m25s | 413 MB | 0 | 0 |
| Ring | 1200 | 700 | 15.43 | 15.43 | 25m57s | 385 MB | 0 | 0 |
| Teapot | 1200 | 700 | 14.68 | 14.68 | 42m59s | 359 MB | 0 | 0 |
| Tube | 1200 | 700 | 15.56 | 15.56 | 20m28s | 381 MB | 0 | 0 |
| **合计** | 8400 | 4890 | 107.08 | 107.08 | **2h38m33s** | **2.626 GiB / 13,290 文件** | 0 | 0 |

- 磁盘：16.2 GB → 13.3 GB 可用（净增 ≈2.9 GB，含 1.0 MB 清单与 40 KB 日志）。
- 平均吞吐 ≈11.5 MiB/s（波动 6–16 MB/s；Teapot 段带宽下降导致 43 min）。
- **早停收益≈0**：想要的成员分散在归档中，最后一个成员几乎都在归档尾部，实际流式读取量≈整包（≥99.9%）。

### 6.2 独立复核（不依赖脚本自身 gate）

- 文件数逐类精确匹配预期：1900/1900/1890/1900/1900/1900/1900；零字节文件 0 个；每类抽样 3 张解码均为 448×448 RGB。
- 容器签名与既有缓存一致：`RIFF….WEBP`（Car 与 Dice 头部字节相同）。
- **pool 规则等价**：磁盘上 10 个 Good specimen 目录与 `estimate_scope()["bank_specimens"]` 完全一致（Dice 例：`000,009,012,019,022,025,029,032,039,048`），每 specimen 恰好 12 views × 10 illum = 120 图。
- **test 视图纯度**：非 pool 文件全部为 `A120_*`（无其他 views 混入）。
- **resume 复核**：直接调用 `fetch_category(..., dry_run=True)`，7/7 类报告 `present=1900/1890, missing=0`（不写状态文件）。
- 完整性清单：`results/e22/cache_l0_448_manifest.csv`（13,290 行 path/size/md5）+ 聚合
  `sha256 = 1569cc57f8c316056b3f445c13779e36aee6e861433c020c3d0a683a16e47812`
  （见 `cache_l0_448_aggregate_sha256.txt`；该 sha256 覆盖每个文件的 path+size+md5，可在缓存损坏或迁移后复验）。

### 6.3 冻结记录

`results/e22/status_fetch_20261015.json`（取数结束时的 `status.json` 快照，含逐类 streamed_gb / seconds / gate 明细 / free_gb_after）；原始 stdout：`results/e22/logs/fetch_20261015_183306.log`（390 行）。

## 7. 异常情况

### 7.1 【实现缺陷，已修复，无数据影响】status 文件写冲突

- **bug 位置**：`e22_fetch.stream_e22` 把流式进度写入 `results/e22/status.json` —— 与 `main` 的 master 记录同一路径。
- **为什么是 bug**：流式进行中，master 记录（已完成类别清单）被逐成员状态临时覆盖；若进程在类别中途被杀，崩溃时刻的 master 记录丢失。
- **影响范围**：仅崩溃期 bookkeeping。缓存像素与 gate 完全不受影响；resume 依据是既有缓存文件扫描（已单独验证 present=1900/missing=0），因此**无需重跑，历史结果无效力损失**。
- **修复**：`write_status(state, path=None)`，流式进度改写入 `results/e22/stream_progress.json`（2026-10-15）；已验证修复后写流式状态时 master `status.json` 的 md5 不变。
- **修订说明**：本次 7 类数据由**修复前**的脚本修订产生（数据路径与缓存生成代码路径在修复中未被触碰）；修复后的文件已纳入版本控制，后续 resume 运行将使用新修订。

### 7.2 网络吞吐波动

Ring 25m57s、Teapot 42m60s（其他类 16–20 min），吞吐 6–16 MB/s，无重试、无数据错误，仅影响耗时。

### 7.3 【需下游注意】NG 目录名与 `image_anomaly` 标签存在 100 条不一致（Dice 例）

- **事实**：Dice view120 test 的 700 图按目录名统计为 Good 200 / NG 500（50 个缺陷 specimen × 10 illum），按元数据 `image_anomaly` 统计为 0→300 / 1→400；`detectable != False` 过滤后官方口径为 Good 300 / NG 388（= 6.1 的 `category_manifest.csv` 记法）。
- **核对**：冻结评测代码的标签取自元数据 —— `e16_qualify.py:305` `labels = [int(m["image_anomaly"]) …]`，目录名不参与判类 → **对协议无影响**。
- **解释（假设，未验证）**：这 100 图大概率是"缺陷实体存在但从 view120 不可见"的图像级正常样本（M2AD 的 image-level 标签随视角变化）；本实验未做验证，仅记录该结构。
- **风险提示**：任何按目录名（`Good/` vs `NG/`）判类、绕开元数据的下游代码，在 M2AD 上会得到 300/500 的错误混淆矩阵；若后续新增统计脚本，必须从元数据取标签。

## 8. 分析

- **成本模型（事实）**：新类别取数成本 = **整包流式 ≈15.5 GiB 归档 / 类**，20–25 min（网络好）/ 43 min（网络差）；磁盘 ≈380–410 MB/类。早停不可依赖（成员分散）。→ 未来新增类别按 ~25 min/类 + 0.4 GB/类 预算，仍受磁盘 16 GB 上限约束（当前余 13.3 GB）。
- **GPU 预算（估算，待确认）**：按 E16 实测 4.5 min/cell（F0+R1-ILA+diagnostics，view120），7 类 × 3 seeds = 21 cells ≈ **1.6 h GPU**；这是估算，不含新脚本变体的调试开销。
- **口径一致性（事实）**：pool 抽样规则、view120 test 范围、448 缓存派生路径均与冻结协议一致（6.2 逐条验证），未改动任何 R1-ILA 参数。
- **未产生任何性能结论**：本实验没有运行任何模型，不应被引用为"跨类别泛化"的证据。

## 9. 结论

7 个新类别的 R1-ILA 输入缓存已在 AutoDL 服务器上完整落盘并通过逐类硬 gate 与独立复核；「R1-ILA 增益仅限 3 类」这一替代解释中的**数据不可得**分支被消除。协议、参数、指标口径均未改动。

## 10. 出口条件判断

- 预设 GO 条件：每类 `gate.n_missing == 0` 且 `bad_shape == 0`。
- 实际：7/7 全绿 → **数据层 GO**。
- 若任一类出现缺失/坏图：停在数据层修复，不得进入 GPU 阶段（未触发）。
- 注意：该 GO **仅限数据层**，不代表任何科学结论成立。

## 11. 对论文证据链的贡献

- **完成的格子**：跨类别外部有效性扩展的**输入可获取性**（3 类 → 10 类的数据前置条件成立）。
- **被排除的替代解释**：外部读者质疑"R1-ILA 结论无法在其他 M2AD 类别上检验，可能因为数据拿不到"—— 已排除。
- **下一格**：新类别上的 F0 与 R1-ILA 单元格（含多 seed 一致性）：只有它才能回答第 2 节的断言。

## 12. 下一步（需用户裁决，不自动执行）

1. **是否执行新类别 GPU 单元格**：需要用户指定类别子集与 seeds（多 seed 不得只报最好一次）。⚠️ 实现约束：`e16_qualify.run_cell` 含历史复现断言 `hist[(cat,seed)]`（`load_history()` 无新类别条目）→ 必须新建变体脚本（例如 `e22_eval.py`），**禁止修改 `e16_qualify.py` 本体**；R1-ILA/ILA 参数沿用 E17 冻结值（`BETA=0.1`）。
2. **是否同时补 E18 风格 baseline**：PatchCore/ResNet 线读 `l0_256`（`e18p1_patchcore_blocker.py` 已记录 `l0_256` 对 MVTec 基线不等价），因此新类别的 baseline 单元格需要另行审计缓存依赖，本实验的 `l0_448`-only 资产不覆盖它。
3. 备选路径 M2（4-view 扩展，零获取成本）与 M3（Bird 12-view）仍待裁决，见 `results/e20/README.md`。

## 13. 复现命令与文件清单

| 文件 | 说明 |
|---|---|
| `results/e22/README.md` | 本文件 |
| `results/e22/category_manifest.csv` | P0：10 类的元数据/本地覆盖盘点（7 类像素零） |
| `results/e22/status_fetch_20261015.json` | P1 冻结记录（逐类 streamed_gb / seconds / gate / free_gb_after） |
| `results/e22/cache_l0_448_manifest.csv` | 13,290 个缓存文件的 path/size/md5 清单 |
| `results/e22/cache_l0_448_aggregate_sha256.txt` | 聚合 sha256（`1569cc57…e47812`） |
| `results/e22/logs/fetch_20261015_183306.log` | 原始 stdout（含逐类早停/ gate 行） |
| `scripts/e22_manifest.py`、`scripts/e22_fetch.py` | P0/P1 脚本 |

```bash
# 完整性复验（迁移/备份后）
python - <<'PY'
import hashlib, csv
rows = list(csv.DictReader(open("results/e22/cache_l0_448_manifest.csv")))
agg = hashlib.sha256("\n".join(f"{r['rel_path']}:{r['bytes']}:{r['md5']}" for r in rows).encode()).hexdigest()
print(agg)
PY
```

**术语**：`l0_448` = 448×448 BICUBIC 无损 WebP 缓存；`pool` = bank 池 1200 图；`test120` = view120 评测集原始记录；`gate` = 取数后存在性+尺寸+模式校验。
