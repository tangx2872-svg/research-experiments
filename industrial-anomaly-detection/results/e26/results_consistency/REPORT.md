# E26-D · 已发表数字的结果一致性核验（只读复算，1249 项）

| 项 | 值 |
|---|---|
| 日期 | 2026-11-01 |
| 实验编号 | E26-D（投稿前 P0 审计，与 E26-A / E26-B / E26-C 同组） |
| 性质 | **只读复算**。未训练、未推理、未重跑任何 cell、未修改任何历史结果文件 |
| GPU | 0（纯 CPU，端到端 **1.7 s**） |
| Git | branch `research/industrial-anomaly`，HEAD `417d883`（新增脚本 + 本目录未提交） |
| 环境 | Python 3.12.3 / numpy 2.3.2 / scikit-learn 1.9.1（conda base） |
| 脚本 | `scripts/e26_d_results_consistency.py` |
| 机器可读产物 | `consistency_checks.csv`（1249 行）、`consistency_report.json`（含全部复算中间值） |

---

## 1. 论文导航（四问）

| 问 | 答 |
|---|---|
| 📍 论文阶段 | 投稿前 P0 审计（证据链的「可核验性」格），写作开始前的最后一格机器检查 |
| 🎯 要证明的那句话 | 「正文与附表引用的每个数值，都能由仓库内归档的**逐图分数**重新算出，且各处口径一致」 |
| 🧪 为什么必须做 | 审稿人拿不到 AutoDL 实例，只能读 CSV/JSON。若表值不能由归档分数复算，证据等级即下降；本实验同时排除两个替代解释：①表值是手工抄写/跨文件搬运产生的偏差；②同一位移在不同表里用了不同 seed 口径 |
| 🚦 出口条件 | **0 硬失败** 且所有「1 ulp 级」差异能被发布脚本的**机械规则**（截断 / 双舍入）解释 → PASS；任一硬失败 → 回查该表并冻结投稿；发现口径（seed 语义）冲突 → 记为口径缺陷，要求论文措辞修正（不重跑） |

---

## 2. 审计设计

**复算，而非重跑。** 所有指标由归档的逐图分数用 `sklearn.roc_auc_score` / `average_precision_score` 重新计算，再与每个发布位置比对。因此本审计验证的是「归档分数 → 发表数字」这一段链条；「原始图像 → 归档分数」由 E26-C（reproducibility）与各实验自身的 recompute gate 负责。

**单来源单归属。** 每个发布文件只有**一个** owner 函数负责核验，避免同一数字被重复检查而漏掉另一个。

**容差分级（每项检查都带类别标签）**

| 类别 | 适用 | 断言 |
|---|---|---|
| `full_precision` | cell JSON / 分析 JSON（~16 位有效数字） | `abs(diff) ≤ 1e-6` |
| `published_6dp` | 6 位小数打印表 | `abs(diff) ≤ 1e-6`（末位 1 ulp） |
| `published_5dp` | 5 位小数打印表（paper_ready） | `abs(diff) ≤ 1e-5`（末位 1 ulp） |
| `categorical_exact` | 标志位 / 计数 / 类别标签 | 严格相等（tol = 0） |
| `cross_source_1e-5` | 跨工件一致性（不验证发布值） | `abs(diff) ≤ 1e-5` |

**1 ulp 项必须归因，不得吞掉。** 对每项用到整个 ulp 的检查，脚本机械判定其是否由以下两种发布规则复现：①**截断**（向零取整）；②**双舍入**（先保留 n+1 位，再打印 n 位）。两者都不是分歧，而是打印管线的性质；无法归因者会被单独列出。

**框架灵敏度自查（防止「恒真审计」）。** 本脚本在开发中确实报过 FAIL，并且都指向真实问题（§6），说明 PASS 不是构造出来的。

---

## 3. 覆盖矩阵

| 来源文件 | 粒度 | 检查数 | 精度类 |
|---|---|---:|---|
| `results/e22/cells/*.json` | 30 cell ×（f0/r1 × AUROC/AP） | 120 | full |
| `results/e22/main_table_10cat_3seed.csv` | 10 类 × 8 列 | 80 | 6dp |
| `results/paper_ready/paper_main_table.csv` | 10 类 × 8 列 | 80 | 5dp |
| `results/e25/cells/*.json` | 39 cell ×（f0/r1 AUROC、ΔAUROC） | 117 | full |
| `results/e25/multiseed_analysis.json` | 13 split × 4 + 2 域 × 2 + 2 总体 | 58 | full |
| `results/e25/multiseed_summary.csv` | 13 ×（4 mean + 2 std）+ 13 × 3 flags | 78 + 39 | 6dp / exact |
| `results/e25/e25_summary.csv` | 13 × 6 + 13 × 3 flags | 78 + 39 | 6dp / exact |
| `results/e21/e21_partial.json` | 12 view ×（f0/r1 × AUROC/AP） | 48 | full |
| `results/paper_ready/paper_multiview_table.csv` | 12 × 6 + 12 × 2 | 72 + 24 | 5dp / exact |
| `results/e26_anomalydino/scores/*.json` → `baseline_30cells.csv` | 30 × 6 列 | 180 | 6dp |
| `results/e26_anomalydino/baseline_category_summary.csv` | 10 类 ×（10 mean + 10 sd） | 200 | 6dp |
| `results/e26_anomalydino/baseline_macro.csv` | 3 层 ×（10 列 + n_categories） | 30 + 3 | 6dp / exact |
| 跨工件：E21 view120 ↔ E22 seed 0 | 3 类 × ΔAUROC | 3 | cross 1e-5 |
| **合计** | | **1249** | |

---

## 4. 结果

### 4.1 总判定

| 指标 | 值 |
|---|---|
| 检查总数 | **1249** |
| 通过 | **1249** |
| 硬失败 | **0** |
| 最大数值差（非类别项） | **5.449e-06** |
| 需要整个 ulp 的项（1 ulp notes） | **7**（全部被机械规则解释） |
| 判定 | **PASS** |

按类别：`full_precision` 343 / `published_6dp` 646 / `published_5dp` 152 / `categorical_exact` 105 / `cross_source_1e-5` 3，**全部 0 失败**。

### 4.2 七项 1 ulp 差异（全部可归因）

| id | 打印值 | 复算值 | 差异 | 归因 |
|---|---|---|---:|---|
| `paper_ready-Dice-dAUROC_std` | 0.00077 | 0.000775404 | 5.40e-06 | 截断（同时也满足双舍入） |
| `paper_ready-Motor-R1_AP_mean` | 0.918 | 0.918005303 | 5.30e-06 | 截断（同时也是双舍入） |
| `paper_ready-Ring-F0_AUROC_std` | 0.011 | 0.011005184 | 5.18e-06 | 截断（同时也是双舍入） |
| `paper_ready-Teapot-dAP_mean` | 0.01069 | 0.010695311 | 5.31e-06 | 截断（同时也是双舍入） |
| `E21-table-Car-330-R1_AP` | 0.78368 | 0.783674795 | 5.21e-06 | 双舍入（6dp→5dp），非截断 |
| `E21-table-Cube-120-F0_AUROC` | 0.8541 | 0.854094590 | 5.41e-06 | 双舍入，非截断 |
| `E21-table-Cube-330-R1_AP` | 0.94076 | 0.940754551 | 5.45e-06 | 双舍入，非截断 |

即：4 项为「打印值 = 复算值向零截断」，3 项为「先取 6 位再打印 5 位」。两类差异均 ≤ 5.5e-6，**不构成数值分歧**；但复现脚本若用「直接四舍五入到 5 位」去比对，会在这 7 处误报（本文档记录该规则，避免后续再踩）。

### 4.3 复算出的关键数字与引用位置

| 量 | 复算值 | 发布位置（本次已核验一致） | 是否已出现在 `results/paper_ready/` |
|---|---:|---|---|
| M²AD 10 类 F0 AUROC / AP（3 seed） | 0.838813 / 0.893955 | `e22/main_table_10cat_3seed.csv`、`e26_anomalydino/baseline_macro.csv` | 逐类值在 `paper_main_table.csv`；macro 未在 md 中引用 |
| M²AD 10 类 R1 AUROC / AP（3 seed） | 0.846409 / 0.901728 | 同上 | 同上 |
| M²AD 10 类 ΔAUROC / ΔAP | **+0.007596 / +0.007773** | 同上 | 是（`EVIDENCE_MAP.md` / `paper_claims.md` / `paper_evidence_map.md` 均含 0.0076 量级表述） |
| 负类（ΔAUROC < 0） | **Doll −0.0385**（唯一负类） | `paper_main_table.csv` | 是（多份材料一致引用） |
| RobustAD PCB 域 ΔAUROC（5 split） | +0.018352 | `e25/multiseed_analysis.json`、`multiseed_summary.csv` | 是（+0.01835） |
| RobustAD MetalParts 域 ΔAUROC（6 split） | −0.017251 | 同上 | 是（−0.01725） |
| RobustAD overall macro ΔAUROC（11 split） | −0.001068 | 同上 | 是（−0.001068） |
| 子数据集等权 ΔAUROC | +0.000550 | 同上 | 是（+0.000550） |
| AnomalyDINO macro_10cat AUROC | 0.803565 | `e26_anomalydino/baseline_macro.csv` | **否**（见 §7 说明） |
| R1 − AnomalyDINO（macro） | +0.042844 | 同上 | **否** |
| AnomalyDINO − F0（macro） | −0.035247 | 同上 | **否** |
| AnomalyDINO dev_3cat / ext_7cat AUROC | 0.743570 / 0.829278 | 同上 | **否** |
| E21 多视角 ΔAUROC 区间（12 view） | −0.0260（Car-330）… +0.0214（Bird-090） | `paper_multiview_table.csv` | 是（该表即发布位置） |

---

## 5. 口径事实（事实与解释分离）

**F1（事实）`results/e25/e25_summary.csv` 是 seed-0 表，不是 3-seed 表。**
对 13 个 split：`auroc_r1 / ap_r1 / d_auroc / d_ap` 与 seed 0 复算值 **13/13 匹配**，与 3-seed 均值 **0/13 匹配**；其 `d_auroc_ci_*` 同为 seed-0 配对 bootstrap CI。`multiseed_summary.csv` 的 `*_mean` 列则与 3-seed 均值 13/13 匹配。
（解释）两表并存但**默认读者不会区分**；论文面向材料 `paper_evidence_map.md` 引用的是 3-seed 值（已核对一致），因此当前正文引用**未**受影响。风险在复现者：直接引用 `e25_summary.csv` 的 ΔAUROC 会得到 seed-0 数字（例如 PCB +0.0217 而非 +0.01835）。

**F2（事实）E25 的 f0 跨 seed **逐位相同**，E22 的 f0 跨 seed **不同**；两者不可混用同一解释。**
- E25：13 个 split 上 seed 0/1/2 的 f0 AUROC/AP 差值 = **0.0**，且三个 seed 的 `bank_hash` 完全一致（PCB `08821b36…`、MetalParts `b4b51729…`）→ bank 固定，只有适配器随 seed 变。
- E22：同样比较下 f0 的跨 seed 最大差 = **0.028464 AUROC / 0.028881 AP**（即每 seed 重新抽 300-bank）。
（解释）E25 的 seed 方差是**纯适配器训练方差**（`multiseed_summary.csv` 的 `d_auroc_std` 范围 0.000217–0.012405 应按此解释）；E22 的 seed 方差**同时含 bank 抽样方差与适配器方差**，两者不可直接比较量级。若论文把 E22 的 seed std 与 E25 的 seed std 并列，必须注明这一差异。

**F3（事实）`is_source` 的发布口径是 `split == "test0"`（前缀语义），不是"有效源域"。** `MetalParts:test0` 的 `condition_group = "source_reference_INVALID"`，其 `is_source = 1` 且 `valid = 0`。
（解释）源域标记与有效性是两个正交字段，聚合时必须用 `valid == 1 and is_source == 0`（11 个 target split），只用 `is_source == 0` 会把无效的 MP test0 算进目标集。本审计第一版正是踩了这个坑（§6）。

**F4（事实）5 位小数表不是「四舍五入到 5 位」。** 7 项 1 ulp 中 3 项只能由「先取 6 位、再打印 5 位」复现。
（解释）属打印管线特性，量级 1e-5；**建议**未来表格统一用一次舍入，或在校验脚本中显式声明该规则（本轮不修改任何历史表）。

**F5（事实）E21 的 view120 与 E22 的 seed 0 是**逐位相同**的同一计算。** 三类 ΔAUROC 复算差 = **0.0**（Bird 0.01061373873873861、Car 0.019588014981273494、Cube 0.01600110415899869）；两者的 `bank_hash` 一致（`1cac68f7…` / `02884fc0…` / `d4132e59…`）、`steps_used` 一致（59 / 44 / 59）。E22 cell 的 `adapter_hash` 字段为 null，故适配器权重无法逐字节比对。
（解释）同一固定 seed、同一 bank 下，两条链路得到**完全相同**的逐图分数，说明 f0/r1 打分路径是确定性的（不含未固定的随机源）。因此 `paper_multiview_table.csv` 的 Bird-120 ΔAUROC = 0.01061 与 `paper_main_table.csv` 的 Bird ΔAUROC = 0.02586 **不是矛盾**：前者是 **seed 0 单视角**，后者是 **view120 的 3-seed 均值**。这是读者最容易误读的一处，建议在表格脚注写明 seed 口径。

---

## 6. 审计框架的灵敏度（反例证据，证明 PASS 非恒真）

| 版本 | 结果 | 暴露的问题 |
|---|---|---|
| run1（`is_source` 用 `condition_group == "source_reference"` 严格等值） | 460 项 / **6 失败** | PCB 域均值算成 +0.013948（把无效的 MP test0 与源域 PCB test0 混入 6 个 split）；published +0.018352 才对 |
| run2（`e25_summary` 误按 3-seed 均值比对） | 675 项 / **26 失败** | 全部落在 `auroc_r1/ap_r1` 列 → 由此反推出 F1（该表为 seed-0） |
| 最终版 | 1249 项 / **0 失败** | — |

三版的差异都指向**口径**而非数值；这也是本实验的第二个产出：把「表格语义」从口头约定变成可执行断言（`table_specs` 里逐列声明 `mean / std / seed0`）。

---

## 7. 未覆盖项（诚实清单）

| 未覆盖 | 原因 |
|---|---|
| bootstrap CI（E21 / E25 / AnomalyDINO 的配对 bootstrap） | 需复现原重采样实现与 RNG 状态；本轮不重建随机过程（不做「看起来一样」的伪核验） |
| `paper_ablation_table.csv`（E24，15 行）的 AUROC/AP 列 | 需 config（C0–C4）→ cell 的映射；其中 C0 行（3 行）等于 E22 seed-0 值，已被本审计间接覆盖 |
| `paper_ablation_table.csv` 的派生列（`d_prime`、`robustness_feat/score`、`illum_shift`、`collapsed`） | 由 E24 专用分析脚本的中间量导出，重新实现会引入新解释风险；如需核验应复用原脚本并单独编号 |
| E23 Doll 审计、E26-A 重复审计、E26-B checkpoint 审计 | 属独立审计，产物自带 `REPORT.md` |
| 图（figures）与 `.tex`/正文草稿 | 仓库内无正文草稿；图为渲染结果，非数值来源 |

**附注（文档传播缺口，事实陈述）**：`results/e26_anomalydino/` 的外部基线数值（`0.803565` / `0.042844` / `0.035247` / `0.743570` / `0.829278`）在 `results/paper_ready/*.md`、`results/paper_ready/*.csv` 与根目录 `PROJECT_STATE.md` 中 **grep 命中 0 次**（`PROJECT_STATE.md:19` 的 "AnomalyDINO" 指的是 E14 早期筛选，不是 E26 完整基线）。
数值本身已由本审计确认与归档的逐图分数一致 → 这是**论文面向材料尚未吸收 2026-10-30 那一轮补充**的问题，不是数值问题。是否属于有意分层（留到写作阶段再并入）需用户裁决，本轮不擅自修改 `results/paper_ready/`。

---

## 8. 结论与出口条件判定

- **结论：PASS。** 1249 项检查全部通过，最大数值差 5.449e-6（出现在 5 位打印表，且 7 项 1 ulp 差异全部由「截断 / 双舍入」机械复现）。
- **出口条件（预设）逐条满足**：
  1. 0 硬失败 ✓
  2. 1 ulp 项全部可归因 ✓（4 截断 / 3 双舍入）
  3. 未发现「同一数字在不同表里用不同 seed 口径」的**引用冲突** ✓（F1 为表格语义问题，非引用冲突）
- **必须随论文记录的 3 条口径事实**：F1（`e25_summary.csv` = seed-0）、F2（E25 的 f0 跨 seed 恒等 = bank 固定 → 其 seed std 是纯适配器方差；E22 的 seed std 含 bank 抽样方差，二者不可并列比较）、F3（`is_source` 与 `valid` 正交）。

---

## 9. 对论文证据链的贡献

- **推进的证据格**：「可核验性 / 数值一致性」（投稿前 P0 审计组）。
- **闭合的替代解释**：「附表中的数字是手工搬运或跨文件抄写，与归档分数脱钩」——用 1249 项复算断言排除（覆盖 E21 / E22 / E25 / E26-AnomalyDINO 四组结果的全部主表与两张 `paper_ready` 主表）。
- **产生的约束（写作层）**：①`e25_summary.csv` 若被单独引用，必须标注 seed-0；②5 位小数表的第 5 位可能与「直接四舍五入」差 1 ulp；③`is_source` 不能单独当作「有效目标 split」筛选用。
- **下一格（需批准）**：把 E24 消融表的 AUROC/AP 列纳入同一框架（需先固化 config → cell 映射），以及用原脚本复算 bootstrap CI；二者都不改变本结论，只扩大覆盖面。

---

## 10. 产物与复现

```bash
cd /root/autodl-tmp/research-experiments/industrial-anomaly-detection
python scripts/e26_d_results_consistency.py     # 1.7 s, CPU, 退出码 0 = PASS
```

| 文件 | 内容 |
|---|---|
| `results/e26/results_consistency/REPORT.md` | 本报告 |
| `results/e26/results_consistency/consistency_checks.csv` | 1249 行明细（`check_type`、`expected`、`actual`、`diff`、`tol`、三个舍入归因列、`pass`） |
| `results/e26/results_consistency/consistency_report.json` | 汇总 + 每类复算中间值（`m2ad` / `robustad` / `e21` / `anomalydino`）+ 全部检查 |
| `results/e26/results_consistency/run1_prefix_bug/` | 开发期第一版（`is_source` 语义错误）产物，**保留作反例证据**，不参与结论 |
| `scripts/e26_d_results_consistency.py` | 审计脚本（只读） |

**写入声明**：本实验只向 `results/e26/results_consistency/` 写入，未改动任何历史 CSV / JSON / checkpoint / figure / README。

---

## 11. 限制

1. 本审计证明「归档分数 → 发表数字」一致；**不**证明「原始图像 → 归档分数」可复现（后者依赖 E26-C 与各实验的 recompute gate）。
2. 复算使用当前 sklearn 1.9.1；若换 sklearn 版本，AP 的插值细节可能在 1e-12 量级变化，不影响 6 位/5 位打印断言。
3. 未覆盖 bootstrap CI 与 E24 派生列（§7），因此不能宣称「仓库内**每一个**数字都被复算」——只能宣称「**全部主表数值列**已被复算」。
