# E18 — Final Baseline Benchmark · FROZEN PROTOCOL

冻结时间：2026-10-14（**在本轮任何新指标产生之前**）
HEAD：`28d02b3` · 阶段：`METHOD SEARCH CLOSED → FINAL BENCHMARK AND PAPER CONVERGENCE`
性质：**公平比较 + 统计 + 可视化；不是搜索，不是调参**

---

## 0. 冻结约束（E18–E20）

**禁止**：
- 修改 R1-ILA 结构 / BETA / 任何最终超参数
- 为单个 category 设置不同配置
- 重新开展 detector / scoring / fusion 搜索
- 为追求更高分数反复试验后挑选最好结果

**允许**：基线公平比较、预先定义的消融、统计分析与可视化。
最终方法参数**只能**从 `results/e17/final_method.json` 读取，禁止手工复制。

---

## 1. 统一评价协议（主表口径）

```
dataset      : M²AD
categories   : Bird / Car / Cube
view         : 120
seed         : 0
test set     : view120 全部 700 图
labels       : image_anomaly
filter       : detectable != False   (official)
kept         : Bird 653 (320 ok / 333 ng, -47) | Car 656 (300/356, -44) | Cube 678 (260/418, -22)
bank         : 300 normal, E8.bank_for(pool, 0), random.Random(20261009)
               sha256(seed0) = 33381cd3004bfe68
metrics      : image-level AUROC + AP (sklearn)
implementation: scripts/e11_p2_runner.py::official_metrics
```

**同一主表内的所有 AUROC/AP 必须由上述定义产生。** 历史结果若使用其他标签口径，
必须先用 §2 的合法重算规则转换，否则不得进入主表。

---

## 2. 合法重算规则（metric-only, 0 GPU）

若一个历史运行满足**全部**下列条件，则允许用其**保存的 per-image 分数**重算 official 指标：

1. bank、test set、view、seed 与 §1 完全一致（bank 需 sha256 校验一致）；
2. 存在 **1:1** 的 `(specimen, illumination) → 元数据` 映射（无未匹配、无重复键）；
3. **分数与标签无关**（标签只影响指标，不影响分数）；
4. **身份校验**：用其原始标签重算必须**逐位复现**历史报告值（容差 ≤5e-6）。

→ 任一条件不满足即不得重算，该单元标记为需 GPU 补跑或不可比。

---

## 3. 主表构成（6 行）

| # | 行 | native scoring | 来源 |
|---|---|---|---|
| 1 | **PatchCore** | coreset 0.1 + k=9 + max-pool（官方） | E7-1 P00（official 重算，Bird）；Car/Cube 待补 |
| 1b | *PatchCore-emb + unified scoring*（对照行，representation-only） | 1-NN-max（统一） | E11-P2（REUSE） |
| 2 | **AnomalyDINO** | mean_top1p（官方） | E14 F1（Cube 待补） |
| 3 | **SoftPatch** | soft-weighted kNN（官方） | E14 F2（Cube 待补） |
| 4 | **EfficientAD** | student-teacher discrepancy（官方） | E15（Bird only，EARLY_STOP） |
| 5 | **F0** = DINOv2-B + 1-NN-max | 1-NN-max | E16（REUSE，3 seed） |
| 6 | **Ours** = R1-ILA (BETA=0.1) | 1-NN-max | E16/E17（REUSE，3 seed） |

**禁止**为了表面统一而改写任何方法的 native scoring。
主表须并列标注每条行的 native scoring；缺单元必须留空并标注原因，**不得插值或推测**。

---

## 4. 允许的预先定义消融（E18 内部，不新开搜索）

1. **Representation vs scoring 分解**：PatchCore(official scoring) vs PatchCore-emb+1-NN-max。
2. **Illumination-conditioned breakdown**：按 illumination 01–10 分组报告 AUROC/AP（仅描述，不做选择性挑选）。
3. **Ours 的 3-seed 稳定性**（复用 E16）。
4. **配对 bootstrap CI**（仅在 per-image 分数可得的方法上；F0/Ours/PatchCore-Bird 可用）。

## 5. 统计规则

- 主表报告**点估计（全精度）**；有 per-image 分数时附 **paired bootstrap 95% CI**（3000 draws，detectable-filtered）。
- 单 seed 的方法必须标注 `single-seed`，**不得**声称 seed 稳健。
- 任何 "Ours 更好" 的表述必须指明是**同 seed / 同 bank / 同测试集 / official 口径**下的比较。

## 6. 论文解释约束

- 允许：*R1 reduces illumination-induced feature shift / illumination-group discrepancy while preserving anomaly discrimination.*
- 禁止：`normal dispersion reduction`（E16 已证伪）；universal robustness 或跨数据集外推。

## 7. 输出

```
results/e18/
  audit.md                      <- 已完成
  compatibility_matrix.csv      <- 已完成
  protocol.md                   <- 本文件（冻结）
  run_plan.md                   <- 执行计划 + 成本
  main_table.csv                <- 最终主表（全精度）
  main_table.md                 <- 论文用表
  illum_breakdown.csv           <- §4.2
  bootstrap.csv                 <- §5
  decision_memo.md / README.md
  figures/ logs/ recomputed/
```

## 8. 出口

- 主表完整（或显式标注每个空单元）→ 允许进入 E19（消融/统计）/ E20（论文收敛）。
- 若补跑发现某基线显著超过 Ours，**不得**回到调参；按事实更新论文定位。
