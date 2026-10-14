# E19 · Paper Convergence — Figures / Tables / Verification — README

**日期**：2026-10-15
**论文阶段**：`METHOD SEARCH CLOSED → FINAL BENCHMARK AND PAPER CONVERGENCE`（论文资产化）
**状态**：**完成（零 GPU；未新增任何实验、未修改任何冻结参数）**
**核验**：**77/77 PASS**（[`verification.csv`](verification.csv)），图表/表格**重跑 byte-identical**

---

## 1. 论文阶段与本轮目标

本仓库 DINOv2 线（E11→E18）已关闭方法搜索并冻结最终方法；E18 已定稿最终基线主表与统计
（`results/e18/README.md`）。**E19 的唯一目标是把已有结论资产化为论文可直接使用的图表与表格，
并建立机器可核验的证据追溯链**——不新增 GPU 运行、不调参、不重开任何方向。

本轮回答三个执行问题：
1. 论文主结果、BETA 受控消融、光照分组三类图是否能**从既有冻结 artifact 确定性重建**？→ 是（见 §4、§5）。
2. 每个图表数值能否追溯到**单一权威来源文件**？→ 是（77 项逐值核验，1e-12 容差）。
3. 论文主张是否**不超过证据边界**？→ 是（§3、§6 逐条标注边界；机制表述按 E16/E17 冻结口径）。

## 2. 最终冻结方法与其论文主张

**最终方法 = R1-ILA（Illumination-consistent Linear Residual Adaptation），BETA = 0.1。**
唯一权威配置：`results/e17/final_method.json`（E18+ 禁止手工复制参数）。

- 结构：`z' = z + Wz`（768×768 zero-init 线性残差，589,824 参数），作用于 frozen DINOv2-B
  final-layer patch tokens（+L2），scoring 为 1-NN-max；normal-only 训练；
  损失 `L = L_cons + BETA·L_preserve`（BETA=0.1），AdamW lr=1e-3，≤300 step。
- 资格与冻结：E16（3-seed 资格赛 GO：Bird +0.02586 / Car +0.01123 / Cube +0.00265）→
  E17（BETA 扫描，无挑战者过 +0.003 晋级门 → `FREEZE_CURRENT_R1`）。

对应论文主张（本轮图表服务的对象）：

| # | 主张 | 证据级别 |
|---|---|---|
| **A1** | 在 M²AD view120 / seed0 / 300-bank / official image 协议下，R1-ILA 在 Bird/Car/Cube 三类上均高于所有可比基线（含最强 frozen baseline F0 = DINOv2-B 1-NN-max） | seed0 主表 + F0/Ours 3-seed + 配对 bootstrap |
| **A2** | BETA 是检测 ↔ 光照一致性的单调旋钮；BETA=0.1 为聚合内点最优；BETA=0（纯一致性、无冻结锚定）造成表示退化（Cube collapse） | E17 受控消融（seed0 × 3 类，5 点 BETA） |
| **A3** | R1-ILA 降低照明条件变化所诱发的特征不一致性 —— 与照明诱导的检测分数漂移降低**相一致**（descriptive；**非因果证明**） | E18 光照 01–10 分组（score 级 spread，3 类同向） |

**机制表述边界（沿用 E16/E17 冻结口径）**：允许写
*"R1 reduces illumination-induced feature shift / illumination-group discrepancy while preserving
anomaly discrimination"*；**禁止**写 `normal dispersion reduction`、因果机制、universal robustness、
跨数据集外推。

## 3. 图表 → 主张映射

### 图（`figures/`，均 PNG + PDF）

| 图 | 文件 | 主张 | 内容要点 |
|---|---|---|---|
| BETA 主曲线 | `fig_beta_auroc` | A2 | AUROC vs BETA（3 类，symlog x）；虚线 = frozen F0；星 = 冻结 BETA=0.1；红环 = collapse 点 |
| BETA 决策视图 | `fig_beta_decision` | A2 | 挑战者 ΔAUROC vs BETA=0.1（含 +0.003 晋级门与 −0.003 Cube 安全线）+ 检测↔鲁棒性 trade-off |
| BETA 塌缩诊断 | `fig_beta_collapse` | A2 | pairwise cosine（0.98 阈值）/ std ratio / steps used vs BETA |
| 光照 AUROC | `fig_illum_auroc` | A3 | 每 illumination 01–10 的 AUROC（F0 vs Ours，3 类；虚线 = ALL；描述性） |
| 光照漂移 | `fig_illum_drift` | A3 | good 图像分数相对漂移 + 跨光照 spread 汇总（F0 vs Ours） |
| 光照 context | `fig_illum_context` | A3（附录） | Cube/Bird 上全部可得来源的每光照 AUROC（上下文，非排名） |
| 主结果柱状 | `fig_main_auroc` | A1 | 6 方法 × 3 类 AUROC；缺失单元为中性虚线框（**不插值**） |
| 配对 forest | `fig_main_forest` | A1 | ΔAUROC / ΔAP（Ours−F0）配对 bootstrap 95% CI（per-seed + seed-mean） |
| 3-seed 扩展 | `fig_main_3seed` | A1 | F0 / Ours 的 3-seed mean ± std（AUROC、AP） |

### 表（`tables/`）

| 表 | 文件 | 主张 | 生成来源 |
|---|---|---|---|
| BETA 消融表 | `beta_ablation.csv`（15 行，全精度） | A2 | `results/e17/e17_partial.json` + `results/e16/e16_partial.json` + 冻结 gates |
| 光照分组表 | `illum_groups.csv`（264 行，全精度） | A3 | `results/e18/illum_breakdown.csv`（+ 派生列 `rel_good_mean`、`auroc_minus_all`） |
| 主结果表 | `main_results.csv` / `main_results.md` | A1 | `results/e18/main_table.csv`（全精度，逐值 1e-12 核验） |
| 3-seed 扩展表 | `main_results_3seed.csv` | A1 | `results/e18/main_table.csv` + `results/e18/bootstrap_f0_vs_ours.json` |

### notes（`notes/`，含来源/命令/边界声明）

`e19_beta_ablation_notes.md` · `e19_illum_groups_notes.md` · `e19_main_results_notes.md`

## 4. 每个数值的来源文件（追溯链）

| 数值 | 权威来源（只读） | E19 产物 | 核验 |
|---|---|---|---|
| BETA 消融全部指标（auroc/ap/d′/robustness/collapse 等） | `results/e17/e17_partial.json`（全精度） | `tables/beta_ablation.csv` | 逐列 max\|Δ\| ≤ 1e-12 |
| BETA 冻结门（Δ 聚合值） | `results/e17/final_method.json` | `notes/e19_beta_ablation_notes.md` 表 | 全精度复算 ≤ 9.2e-6（<5e-5 容差；门线不受影响） |
| F0 参考指标 | `results/e16/e16_partial.json` | `beta_ablation.csv` Δ 列 | ≤ 1e-12 |
| 主表 6 行（seed0，全精度） | `results/e18/main_table.csv` | `tables/main_results.csv/.md` | 32/32 非空单元 ≤ 1e-12；4 个空单元保持为空 |
| 3-seed mean/std、per-seed Δ、bootstrap CI | `results/e18/main_table.csv` + `bootstrap_f0_vs_ours.json` | `tables/main_results_3seed.csv`、`fig_main_forest` | ≤ 1e-12 |
| 光照 01–10 分解（24 来源 × 11 组） | `results/e18/illum_breakdown.csv` | `tables/illum_groups.csv`、`fig_illum_*` | 逐列 ≤ 1e-12；spread 与 E18 文档 4 位显示一致 |
| BLOCKED / EARLY_STOP 状态 | `results/e18/native_scoring_appendix.csv` + `statistical_analysis.md` | 主表 notes / md 脚注 | 人工核对（§6） |

`verification.csv` 同时记录 8 个源文件的 **sha256（前 16 位）**，锁定本轮核验时的来源版本。

## 5. 复现命令（CPU-only，确定性）

```bash
cd /root/autodl-tmp/research-experiments/industrial-anomaly-detection

# 1) 从冻结 artifact 重建全部表格 / 图 / notes（只写 results/e19/）
python scripts/e19_figures.py

# 2) 核验：逐值复算 + 重跑幂等性 + Git 边界（写 verification.csv，exit 0 = 全过）
python scripts/e19_verify.py
```

已验证：`e19_figures.py` 重跑后 **8 个 CSV/MD/notes 产物 byte-identical**（verification.csv，IDEM 组）；
核验脚本自身除重跑外不写任何文件。环境：conda `base`，Python 3.12.3，matplotlib 3.10.5（**全程未用 GPU**）。

## 6. 已知限制 / 缺失实验（随图引用）

1. **基线均为单 seed**（PatchCore-emb / AnomalyDINO / SoftPatch / EfficientAD）；仅 F0/Ours 有 3 seeds。
   不得声称基线 seed 稳健。
2. **EfficientAD Car/Cube 未运行**（E15 预注册 Stage-A STOP；`EARLY_STOP`，未插值）。
3. **PatchCore official Car/Cube = BLOCKED**（原图不可得；l0_256 实测不等价被拒；人工裁决选项 C）。
4. **AnomalyDINO / SoftPatch 的 Bird/Car 无逐图分数**（E14 只存聚合）→ 光照分组图不含这两来源。
5. 单数据集（M²AD Bird/Car/Cube）× 单 view（view120）× 300-bank；无跨 view / 跨数据集外推主张。
6. bootstrap 区间只量化**测试集抽样**不确定性，**不是**跨 seed 不确定性。
7. 光照分组为**描述性**分析（每组 n 小、ok/ng 不平衡）；A3 只能写 "consistent with"，**非因果**。
   注意：spread 下降是分布层面性质，**不代表每个单组都改善** —— Car 组 07 的最大偏差在 Ours 上
   反而更大（+40.8% vs F0 +34.0%，见 `notes/e19_illum_groups_notes.md`）。
8. 事实注记（E19 记录，不修改历史文档）：`results/e17/README.md` 的 "BETA=0 三类最差检测器" 按
   **AUROC** 对 Cube 不成立（Cube 上名义最高 0.9090，但伴随 collapse）；按 **d′** 三类均最低。
   E17 裁决（拒绝 C1）不变。

## 7. 本轮执行结果与 Git

- **执行结果**：9 图（PNG+PDF）+ 4 表 + 3 notes + `verification.csv` 全部生成；**77/77 PASS**
  （SRC 8 / BETA 28 / MAIN 7 / ILLUM 10 / FIG 9 / NOTES 3 / IDEM 9 / GIT 3）。
- **GPU**：未使用（0%），未下载任何数据，磁盘仅新增本目录产物。
- **Git**：分支 `research/industrial-anomaly`；本轮提交仅包含 E19 新增与文档修改
  （`results/e19/**`、`scripts/e19_figures.py`、`scripts/e19_verify.py`、`README.md`、`PROJECT_STATE.md`、
  `.gitignore` 中新增的 `!results/e19/` 白名单行）；
  commit message：`docs(e19): paper-convergence figures, tables, verification and report`。
  **6 个历史遗留工作区文件保持不动**（`monitor/exp_monitor.py`、`results/e8/status.json`、
  `monitor/events.jsonl`、`scripts/_lockprobe.txt`、`scripts/_probe2.py`、`scripts/e11_p4_runner.py`）。
  未使用任何破坏性 git 命令。

## 8. 后续论文大纲工作建议（不自动执行）

1. **默认路径 = 直接进入论文大纲与初稿**：A1/A2/A3 三类证据已齐，图表为论文级候选；
   建议结构：Intro → Related → Method (R1-ILA) → Protocol（M²AD official 口径 + 两套标签协议审计）
   → Results A1（主表 + forest + 3-seed）→ Ablation A2（BETA）→ Illumination A3 → Limitations → Conclusion。
2. **写作期需人工裁决的两点**：① 负面结果（detector-family search = CLOSED、E11-P4/P5、E12-A 等
   全灭）是否作为独立 subsection / artifact 贡献；② 主表脚注如何同时表达 "native scoring 不做统一"
   与 "Ours 全面领先" 的可比性边界。
3. **仅在完整稿暴露实质证据缺口时才讨论补实验**。已知候选缺口（全部需人工批准，默认不做）：
   基线多 seed（成本高）、跨 view 稳健性、PatchCore official Car/Cube（需下载原始数据，E18 已裁决 C）。
4. E19 不产生新的科学结论；本目录所有图表**不得**被引用为超出 E16/E17/E18 证据边界的主张。
