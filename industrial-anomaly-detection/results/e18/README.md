# E18 · Final Baseline Benchmark — README

**日期**：2026-10-14 · **阶段**：`METHOD SEARCH CLOSED → FINAL BENCHMARK AND PAPER CONVERGENCE`
**状态**：主表定稿完成（P0 审计 → P1 补齐 → P2 主表 + 统计）
**零 GPU**（P1 仅用 1.8 min；P0/P2 全部 0 GPU）

## 一句话结论
在 M²AD view120 / seed0 / 300-bank / official 图像级协议下，**Ours（R1-ILA）在三个类别上均高于所有可比基线**；
主表 6 行中 **5 行 3/3 完备**，2 行因**数据不可得**或**预注册早停**而显式留空，未做任何插值。

## 1. 主表（seed0；AUROC / AP 全精度见 `main_table.csv`）

| method | Bird | Car | Cube | 完备 | seeds | native scoring |
|---|---|---|---|---|---|---|
| PatchCore-emb + unified | 0.68454 / 0.72797 | 0.62247 / 0.59298 | 0.60857 / 0.66489 | 3/3 | 1 | 1-NN-max (统一) |
| AnomalyDINO | 0.70214 / 0.72234 | 0.75684 / 0.78277 | **0.84889 / 0.91134** | 3/3 | 1 | mean_top1p |
| SoftPatch | 0.64396 / 0.57510 | 0.50219 / 0.55541 | **0.69343 / 0.75908** | 3/3 | 1 | soft-weighted kNN |
| EfficientAD | 0.61206 / 0.62196 | — | — | 1/3 | 1 | teacher-student |
| **F0**（DINOv2-B 1-NN-max） | 0.74725 / 0.78863 | 0.78552 / 0.84207 | 0.85409 / 0.91686 | 3/3 | 3 | 1-NN-max |
| **Ours（R1-ILA）** | **0.75786 / 0.79911** | **0.80511 / 0.85502** | **0.87010 / 0.92856** | 3/3 | 3 | 1-NN-max |

（粗体新增单元由 E18-P1 补齐；其余为 E11/E14/E15/E16 复用）

## 2. 原生评分附表（`native_scoring_appendix.csv`）

| method | Bird | Car | Cube | 状态 |
|---|---|---|---|---|
| PatchCore（official scoring） | **0.6931775525525525 / 0.7230433828349869** | **BLOCKED** | **BLOCKED** | 1/3 |
| EfficientAD（official scoring） | 0.612058933933934 / 0.6219598680138538 | EARLY_STOP | EARLY_STOP | 1/3 |

- **PatchCore Bird 的 official 值 = 0.69318 / 0.72304**。旧值 **0.78089 / 0.91723 使用 legacy `object_anomaly` 标签（200 Good / 500 NG）**，
  **不可作为官方协议结果**；已单独放入 `LEGACY - DO NOT USE IN THE MAIN TABLE` 桶，历史文件未改动。
- **PatchCore-emb + unified scoring 与 PatchCore official 是两种不同配置，禁止混为一行**。
- **PatchCore official Car/Cube = BLOCKED**：`run_patchcore` 需要原图 1024px，而 `data/m2ad/extracted/` 只有 Bird；
  `l0_256` 缓存经实测为不同输入（max|Δ| = 0.0568），已拒绝替代。人工裁决（2026-10-14）= **选项 C：保持 BLOCKED**。

## 3. 统计证据（`statistical_analysis.md`）

**F0 vs Ours 配对 bootstrap**（同一重采样索引、同一 official 标签、detectable-filtered；n_boot=3000，rng seed 20261014）：

| cell | ΔAUROC | 95% CI | P(Δ>0) | ΔAP | 95% CI | P(Δ>0) |
|---|---|---|---|---|---|---|
| Bird seed0 | +0.01061 | [−0.00920, +0.02977] | 0.854 | +0.01048 | [−0.01154, +0.03222] | 0.829 |
| Bird seed1 | +0.03049 | **[+0.01092, +0.04953]** | 0.999 | +0.02890 | **[+0.00786, +0.05093]** | 0.997 |
| Bird seed2 | +0.03647 | **[+0.01733, +0.05591]** | 0.999 | +0.03176 | **[+0.01106, +0.05419]** | 0.998 |
| **Bird seed-mean** | **+0.02586** | **[+0.00833, +0.04310]** | **0.998** | **+0.02371** | **[+0.00498, +0.04364]** | **0.993** |
| Car seed0 | +0.01959 | **[+0.00345, +0.03606]** | 0.991 | +0.01295 | [−0.00085, +0.02704] | 0.967 |
| Car seed1 | +0.00785 | [−0.01251, +0.02696] | 0.791 | +0.00922 | [−0.00480, +0.02345] | 0.902 |
| Car seed2 | +0.00626 | [−0.01135, +0.02467] | 0.750 | +0.01128 | [−0.00161, +0.02530] | 0.954 |
| **Car seed-mean** | **+0.01123** | [−0.00412, +0.02615] | 0.933 | **+0.01115** | [−0.00034, +0.02320] | 0.970 |
| Cube seed0 | +0.01600 | **[+0.00082, +0.03065]** | 0.980 | +0.01170 | **[+0.00275, +0.02091]** | 0.995 |
| Cube seed1 | −0.01229 | [−0.02875, +0.00405] | 0.073 | −0.00321 | [−0.01241, +0.00648] | 0.260 |
| Cube seed2 | +0.00423 | [−0.01216, +0.02035] | 0.690 | +0.00389 | [−0.00545, +0.01346] | 0.780 |
| **Cube seed-mean** | **+0.00265** | [−0.01171, +0.01633] | 0.637 | **+0.00413** | [−0.00429, +0.01249] | 0.827 |

**⚠️ 区间语义边界**：该 bootstrap 只量化**测试集抽样不确定性**，**不是跨 seed 不确定性**。
跨 seed 方差由 3 个独立 bank 的标准差单独报告（Bird AUROC F0 0.75131±0.00427 vs Ours 0.77717±0.01757）。

**多 seed 汇总**：Bird +0.02586、Car +0.01123、Cube +0.00265（Ours − F0 的 3-seed 均值）。

## 4. Illumination 01–10 分析（`illum_breakdown.csv`）

**可用**：F0/Ours（3 cats × 3 seeds）、PatchCore-emb（3 cats）、PatchCore official（Bird）、AnomalyDINO/SoftPatch（Cube）。
**不可用**（仅有聚合指标）：**AnomalyDINO / SoftPatch 的 Bird/Car（E14）**、**EfficientAD（E15）**
→ 按协议**报告为缺失，未重跑任何模型**。

**关键观察（seed0，good 图像的分数跨光照波动）**：

| category | F0 spread | Ours spread | 变化 |
|---|---|---|---|
| Bird | 0.036 | **0.018** | −50% |
| Car | 0.089 | **0.069** | −22% |
| Cube | 0.028 | **0.019** | −32% |

- 这是**分数层面（下游）**量，**不是**训练目标（`L_cons` 作用于特征）；三类同向下降。
- 只能写 **"consistent with"** E16 的 illumination-shift 说明，**不得**写因果证明。
- ⚠️ **只可解读 spread，不可解读绝对值**：R1-ILA 改变特征尺度，F0 与 Ours 的 good_mean 绝对水平不可比。

## 5. PatchCore official vs unified（Bird only）

ΔAUROC（official − unified）= **+0.00612** [−0.01126, +0.02340]，P(Δ>0)=0.754；ΔAP = **−0.00644** [−0.02493, +0.01206]。
**两者在表示管线与打分方式上同时不同**，属配置对比而非受控 scoring 消融；
且**官方 Car/Cube 缺失 → 不主张完成三类别 official-vs-unified 比较**。

## 6. 质量门（§4 强制项）

| 检查 | 结果 |
|---|---|
| 每个结果可由逐图分数独立复算 | ✅ **40/40 PASS**（`verification.csv`，1e-12 容差） |
| 统一标签协议 `image_anomaly` + `detectable` | ✅ 全部来源逐一校验 |
| 各方法保留 native scoring | ✅ 未改任何配置 |
| 缺失单元留空并标注 | ✅ `BLOCKED` / `EARLY_STOP`，**无 0 填充、无估计值** |
| 测试样本数 / 过滤数 / 标签分布 | ✅ Bird 653(47)·Car 656(44)·Cube 678(22)，逐来源断言 |
| 表值 == 权威 JSON 全精度 | ✅ 1e-12 |
| 旧 `object_anomaly` 与 official 隔离 | ✅ 保留历史文件，附录单列 LEGACY 桶 |

## 7. 结论边界（必须随表引用）

1. 主表**统一口径为 official `image_anomaly` + `detectable`**；E7 线（`af56dd3`）的 image 指标为 `object_anomaly` 口径，**不得混入**。
2. 主表**基线均为单 seed**；仅 F0/Ours 有 3-seed 统计。**不得**声称基线 seed 稳健。
3. 各方法使用**各自的原生打分**，这是方法定义的一部分；**不得**为表面统一而改写。
4. 允许的方法表述：*R1 reduces illumination-induced feature shift / illumination-group discrepancy while preserving anomaly discrimination.*
   **禁止** `normal dispersion reduction`（E16 已证伪）、universal robustness、跨数据集外推。
5. 空单元（PatchCore official Car/Cube、EfficientAD Car/Cube）**必须随表标注原因**。

## 8. 文件索引

| 文件 | 内容 |
|---|---|
| `audit.md` | P0 审计 + 两套标签协议发现 + P1 更新 + PatchCore 0.78089 说明 |
| `compatibility_matrix.csv` | 7 行兼容性矩阵（REUSE / RECOMPUTE / EARLY_STOP / BLOCKED） |
| `protocol.md` | E18 冻结协议（统一评价协议、合法重算规则、主表构成） |
| `run_plan.md` | 执行计划 + P1 结果 + 阻断根因 + 三选项 |
| `main_table.csv` / `main_table.md` | **最终主表**（全精度） |
| `native_scoring_appendix.csv` | 原生评分附表 + LEGACY 隔离桶 |
| `statistical_analysis.md` | 配对 bootstrap / 多 seed / illumination / PatchCore 对照 / 质量门 |
| `bootstrap_f0_vs_ours.json` · `patchcore_official_vs_unified.json` | 统计原始输出 |
| `illum_breakdown.csv` | 24 个来源 × 11 组的光照分解 |
| `verification.csv` | 40 项独立复算校验记录 |
| `filled/` · `recomputed/` | P1 补齐单元与 PatchCore official 重算产物 |
| `logs/` | 运行日志 |

## 9. 下一步（唯一推荐，不自动启动）
```
NEXT = E19 Paper convergence: ablation + figures
```
建议内容：① R1-ILA 受控消融（BETA 扫描已由 E17 完成，可直接出图）；② illumination breakdown 图
（Bird/Car/Cube × F0/Ours）；③ 主表图表化（含 BLOCKED / EARLY_STOP 标注）。
**不新增 GPU 实验，不改动任何冻结方法参数。**
