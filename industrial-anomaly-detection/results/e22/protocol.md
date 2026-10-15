# E22 — M²AD Full-Category Multi-Seed Validation (PROTOCOL)

**Phase**：Phase II — Paper Validation ｜ **Method search**：CLOSED ｜ **Frozen method**：R1-ILA ｜ **Frozen BETA**：0.1
**Protocol frozen at**：2026-10-15 21:40 CST（在任何新增 7 类的 F0 / R1-ILA 结果产生之前）
**执行者**：Research Assistant（无人值守 overnight run，用户批准）

---

## 0. Protocol amendment notice（必须如实记录，不得伪装成最初设计）

- 本协议于 **E22-P0/P1 数据获取完成之后**（commit `88883d0`，2026-10-15 21:13 CST）、且**在任何新增类别的模型结果被计算之前**写定。
- **原始计划**（用户初稿）：`10 categories × seed0 × view120` 的 Round 1 普查。
- **本 amendment 后的预注册**：`10 categories × 3 seeds × view120`，共 **30 cells**，每 cell 含 F0 与 frozen R1-ILA。
- **原因**：用户批准无人值守长时间验证 → 一次性预注册完整 3-seed 矩阵，使 **seed 深度不可能依赖 seed0 结果**，从而排除 seed cherry-picking 与 category cherry-picking。
- **事实陈述**：截至本协议写定时刻，Dice / Doll / Holder / Motor / Ring / Teapot / Tube 上 **不存在**任何 F0 或 R1-ILA 数值结果（数据 gate 是纯文件完整性检查，不含模型）。
- **禁止**：不得把本 amendment 描述为"最初即如此设计"。

## 1. 冻结方法（与 E17/E19 完全一致，本实验不得改动）

| 项 | 冻结值 |
|---|---|
| F0 | DINOv2-B frozen final-layer patch tokens（1024×768）+ L2 + 1-NN-max |
| R1-ILA | `z' = z + W z`，W: 768×768，zero-init 线性残差，可训练参数 **589,824** |
| loss | `L = L_cons + BETA * L_preserve`，**BETA = 0.1**（`e13a_screen.BETA` 全局，默认值即 0.1，不注入） |
| optimizer | AdamW，**LR = 1e-3** |
| steps | MAX_STEPS 300，PATIENCE 30，EVAL_EVERY 5 |
| batch | 8（`BATCH_IMGS`） |
| train/val | 240 / 60（bank 顺序前 240 / 次 60） |
| augmentation | brightness / contrast / gamma × {0.85, 1.15}（6 变体，`e13a_screen.AUG_LIST`） |
| bank | 300 张，`E8.bank_for(pool, seed)` = `sorted(random.Random(20261009+seed).sample(range(len(pool)), 300))` |
| 参数来源 | `results/e17/final_method.json`（唯一参数来源；禁止手工复制参数或调参） |
| 代码来源 | `e16_qualify.run_cell`（冻结 cell 路径）＋ `e13b_screen.train_r1`（seed-aware trainer）＋ `e11_p2_runner`（scoring/metrics） |

**禁止修改**：`e16_qualify.py`、`e13a_screen.py`、`e13b_screen.py`、`e11_p2_runner.py`、`e8_common.py`。
E22 只新增 `scripts/e22_full_category_validation.py`（复制 `run_cell` 逻辑并注明来源，移除历史断言、改为条件比较）。

## 2. 数据与标签协议

- 数据集：M²AD，**10 categories**（Bird, Car, Cube, Dice, Doll, Holder, Motor, Ring, Teapot, Tube）。
- 评测集：**view 120 only**，官方 `test` split 全部原始记录（700/类，Holder 690）。
- 缓存：`l0_448`（E22-P1 已 gate：`missing=0 / bad_shape=0`，聚合 sha256 `1569cc57…e47812`）。
- **标签来源（最高优先级）**：官方元数据 `data/m2ad/jsons_extract/meta_unsupervised.json` 的 **`image_anomaly`**，评测过滤 `detectable != False`（严格继承 E18/E19/E21）。
- **标签陷阱（E22-P0 发现，必须写入论文与 README）**：
  > 文件位于 `NG/` 目录 ≠ `image_anomaly = 1`。
  Dice view120 test 中 500 个文件位于 `NG/` 目录，但元数据 NG 仅 400 条（100 条 `NG/` 路径记录的 `image_anomaly=0`）。
- **硬断言（runner 内）**：标签只能来自 `raw_test_map(category)`（元数据）；若任一 test 记录在元数据中不存在 → `raise RuntimeError`；禁止由 `Good/` `NG/` 目录名、路径字符串生成标签；禁止 legacy `object_anomaly`。

## 3. 矩阵与复用策略

| 来源 | cells | 处理 |
|---|---|---|
| Bird / Car / Cube（历史） | 3 × 3 = 9 | **优先复用** E16 全精度结果（`results/e16/e16_partial.json`，含逐图分数），前提是 §4/§4b 的等价性验证通过 |
| Dice … Tube（新增 7 类） | 7 × 3 = 21 | **必须新算**（每 cell = F0 + R1-ILA） |
| 合计 | **30 cells** | |

复用条件（全部满足才复用，且必须在 `verification.csv` 中留证）：
1. 协议一致（同一 runner 代码路径与冻结超参数）；
2. 标签一致（同一元数据 `image_anomaly` + 同一 detectable 过滤）；
3. **bank hash 一致**（E22 侧用同一 `bank_for` 重算并与 E21 已存 hash 比对；seed0 三类）；
4. 分数一致（E16 存有 9/9 cell 的逐图分数；seed0 三类另由 E21 独立进程验证 `max|Δscore| = 0`）。

## 4. Reproduction gate（矩阵开跑前必须通过；失败 = STOP，不进入矩阵）

- **Gate A（跨代码路径、bit 级）**：`Bird | seed0` 同时经
  (i) 冻结的 `e16_qualify.run_cell`（历史本体）与
  (ii) E22 runner 的复制路径
  在同一进程中执行 → 两者逐图分数必须 **`max|Δ| = 0`**（严格 bit 相同）。
- **Gate B（对历史、1e-6）**：`Bird|0`、`Car|0`、`Cube|0`、`Car|1` 经 E22 路径重算 →
  与 `e16_partial.json` 的逐图分数 `max|Δ| ≤ 1e-6`，且 AUROC/AP 差 ≤ 1e-6。
- 任一失败：写 `results/e22/STOP_repro_fail.txt`，停止矩阵，保留现场。

## 5. 运行顺序与无人值守策略

- **Round A**：7 新类 × seed0（Dice, Doll, Holder, Motor, Ring, Teapot, Tube）
- **Round B**：7 新类 × seed1
- **Round C**：7 新类 × seed2
- Round A 结果无论正负 **自动进入 Round B**；Round B 同理进入 Round C。
- **不存在** "seed0 不好 → 停止后续 seed" 规则。唯一停止理由：数据损坏 / 协议不等价 / GPU 或程序真实故障 / 磁盘安全风险（< 8 GB free）/ 元数据映射错误。
- 每个 cell 完成即持久化：`results/e22/cells/<category>_seed<seed>.json`（tmp → atomic rename）＋ 追加/刷新 `results/e22/live_results.csv` ＋ 更新 `results/e22/status.json`（heartbeat / ETA / GPU / disk）。
- 运行于 `screen` 会话 `e22_full3seed`，脱离前端会话；日志 `results/e22/logs/overnight_<timestamp>.log`。
- Watchdog 允许：检测 runner 退出、日志 mtime 停滞、status heartbeat 停滞 → 从最近完整 cell 断点恢复；**同一 cell 连续失败 2 次 → 标记 `ENGINEERING BLOCKED` 并停止自动重试**（不得无限重试烧 GPU）。

## 6. 预注册科研判据（评估 broad category-generalization claim）

**STRONG GO**（全部满足）：
1. ≥ **7/10** 类别的 3-seed mean ΔAUROC > 0；
2. 10 类 macro mean ΔAUROC > 0；
3. 10 类 median ΔAUROC > 0；
4. 无任何类别 3-seed mean ΔAUROC < −0.03。

**HOLD**：5–6/10 类为 3-seed mean ΔAUROC > 0 且总体 mean/median 至少为正趋势；或结论明显依赖 illumination shift。

**STOP BROAD-GENERALIZATION CLAIM**：≤ 4/10 类为正；或 macro mean ΔAUROC ≤ 0 且整体无正趋势；或存在多个严重负类别。
> STOP 只能写成"broad-generalization claim 不成立"，**禁止**写成 "paper failed"。

## 7. 分析计划（预注册输出，全部自动执行）

| 输出 | 内容 |
|---|---|
| `all10_3seed_cells.csv` | 30 cells 逐 cell（F0/R1 AUROC、AP、Δ、n_good、n_anomaly、bank hash、adapter hash、runtime、timestamp、status、source） |
| `main_table_10cat_3seed.csv` | 每类 mean ± std（F0/R1 AUROC/AP、ΔAUROC/ΔAP）+ positive seeds / 3 |
| `category_summary.csv` | 类别级胜率、AP 方向一致性、seed 稳定性（mean/std/min/max/positive ratio） |
| `illumination_breakdown.csv` | 类别 × illumination 的 F0/R1 AUROC、AP、score spread（仅用 E18/E19 已冻结定义） |
| `statistical_summary.md` | 12 问逐条回答（含 verdict、claim boundary） |
| `failure_cases.md` | 最差类别/seed、最大 AUROC/AP regression、seed variance 最大类别（负结果全保留） |
| `figures/` | Fig1–Fig8（PNG + PDF） |
| `verification.csv` | ≥ 20 项独立检查（0 FAIL 目标） |
| `README.md` | Motivation / Protocol / Data / Label trap / Reuse / New computation / Results / Negative evidence / Seed stability / Illumination / Verdict / Claim boundary |

- Bootstrap：**paired-image**，`n_boot = 3000`，固定 RNG；逐类给 ΔAUROC / ΔAP 95% CI。
  **image bootstrap 不确定性 ≠ cross-seed 不确定性，两者必须分开报告。**
- 机制关系：illumination shift strength（E16 `D_illum_shift_F0`，patch 级相对位移，冻结定义）× 类别级 3-seed mean ΔAUROC → Pearson（若合理）与 Spearman，**n = 10 categories → 只作 exploratory / supportive association，禁止写 causal proof**。
- Fig8 的 best/median/worst 类别选择必须标注为 **post-hoc descriptive visualization**。

## 8. 禁止事项（无人值守期间同样禁止）

调 BETA / 调 LR / 改 augmentation / 改 bank / 换 seed / 加模块 / 改 scoring / 为失败类别专门重训 / 删除负结果 / 挑最好 seed 汇报 / 因结果不好看而省略类别 / 修改 E21 及其之前任何冻结产物 / `git add .` / 在前端会话中跑正式实验（必须 screen）。

## 9. Claim boundary（本实验能写什么、不能写什么）

- **可以写**：R1-ILA 在 M²AD 10 类 × 3 seeds × view120 上的无选择性正式结果（均值、方差、胜率、CI、逐类负结果）。
- **不能写**：因果证明（illumination shift ↔ gain 只是 association）；对其他数据集/backbone/view 的推广；image bootstrap CI 当作跨 seed 不确定性；把 STOP 写成 "paper failed"。
