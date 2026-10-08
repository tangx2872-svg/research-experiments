# E4-X — M²AD Bird Real-Illumination Candidate Kill Test — PROTOCOL (FROZEN)

**冻结时间**：2026-10-08（**先于读取任何 B2 / X6c target result**）
**起点 HEAD**：`b0cc74c`（branch `research/industrial-anomaly`）
**性质**：**candidate kill test（KEEP OR KILL）**，**不是**方法优化实验，**不是**完整 external validation
**本实验只回答一个问题**：
> 在真实物理 illumination variation 下，冻结的 B2 / X6c 是否比 Original 显示出**足够明显**的
> robustness–detection 优势，值得保留进入下一轮候选池？

---

## 0. 科研阶段

```text
CURRENT STAGE = Phase II — External Validation / Candidate Screening
METHOD = X6c frozen
PRIMARY EXTERNAL DATASET = M²AD
CURRENT TASK = E4-X Candidate Kill Test
```

---

## 1. Frozen view（机械选择，**不可更换**）

`docs/E4_D0_INTEGRITY` / 既有 frozen protocol 中**不存在**「用一个全局固定 seed 抽一个全局 view」的规则
（E4-D0 的 `pairing_audit` 是 **per-specimen** 抽 view，非全局单 view）⇒ 采用 **Rule B**。

```text
VIEW_SELECTION_SEED = 20261008
合法 view 集合 = 12 个真实存在 view
VIEWS = [000,030,060,090,120,150,180,210,240,270,300,330]

FROZEN VIEW = 120        (random.Random(20261008).choice(VIEWS)；确定性，三次抽签一致)
```

**冻结后不允许更换 view。** 无论结果多难看都不换。

## 2. Original reference（**直接复用 E4-D1，GPU = 0**）

来源：`results/e4_d1_smoke/per_image.csv`（8,400 行，E4-D1 原始产物，未重跑）。
筛选 `view == "120"` → **700 行**（20 Good specimen × 10 illumination + 50 NG specimen × 10 = 200 + 500）。

**Original GPU units = 0。** 除非资产损坏，不得为「统一流程」重跑 Original。

## 3. 数据范围（严格）

```text
Category = Bird   |   Seed = 0   |   View = 120 (frozen)   |   Illumination = 01–10
Good specimen = 20 ; NG specimen = 50 ; 共 70 个 (specimen × view) 单元
Scoring images = 700
```

## 4. Bank / 训练口径（复用 E4-D1 已验证配置，**不得为提升表现修改**）

| 项 | 值 |
|---|---|
| Bank | 官方 `train`，30 normal specimen 字典序 `[::3][:10]` → 10 specimen × 12 view × 10 illum = **1,200 图** |
| Val | `[1::3][:2]` → 240 图（仅报 τ，不参与判定） |
| backbone / layers | `wide_resnet50_2` / `["layer2","layer3"]` |
| coreset_sampling_ratio / num_neighbors | `0.1` / `9` |
| IMAGE_SIZE / 归一化 | `256` / ImageNet（`experiment1b.preprocess_for_model`） |
| datamodule | anomalib `Folder` + symlink farm（E4-D1 工程适配 E1/E3/E4） |

## 5. 候选方法（冻结定义，**不得改 α / weight / layer / fusion**）

### 5.1 Candidate 1 — Adaptive B2

```text
L2 : kind = "residual", lambda = beta_Bird,  norm=instance, scale_match=per_sample_per_channel_rms
L3 : kind = "alpha_in", alpha = 0.25
```

实现：`experiment7ao_model.ModulePatchcoreModel._step`（`residual` / `alpha_in`）→
`experiment10_model.RecoveryPatchcoreModel` → `experiment11_model.AdaptivePatchcoreModel` →
Lightning 包装 `experiment11_model.AdaptivePatchcore(spec=...)`。

**`beta_Bird` 的机械确定（冻结定义，不是参数搜索）**：冻结定义为
`beta_c = clip(BETA0 * (q_c / q_ref)^3, 0.05, 0.50)`，`BETA0=0.25`，`q_ref = q_bottle = 1.5726841688159168`
（来自 `results/experiment_11_c2_adaptive/stats/normal_statistics.csv`）；
`q_c = inmag_L2.mean() / inmag_L3.mean()`，其中 `inmag = ||IN(pooled F) − pooled F||` 的逐通道空间 RMS 之和，
**只用 train/good**（`scripts/experiment11_normal_stats.py` 的冻结实现）。

⇒ 在本轮**运行前**用同一实现计算 `q_Bird`（M²AD Bird 的 3,600 张 train/good），
再代入冻结公式得 `beta_Bird`。**不读任何 test/defect 数据；不做任何候选比较。**

> 已知参照：MVTec 5 类的 β = bottle 0.2500 / cable 0.2948 / hazelnut 0.2279 / screw 0.2164 / grid 0.4601。

### 5.2 Candidate 2 — X6c（**0 GPU**）

```text
z_o  = (score_Original − mu_o) / sigma_o
z_b  = (score_B2       − mu_b) / sigma_b
      mu/sigma 取自该方法自己的 clean_good（= 该方法的 test Good 分数，normal-only）
z_C2 = 0.35 * z_o + 0.65 * z_b
z_C6 = max(z_o, z_b)
X6c  = mean(z_C2, z_C6)
```

权重 `0.35/0.65`、`0.5/0.5` 与 Exp15/16 冻结值**逐字一致**。X6c 为 score-level 融合，**不需要任何额外 GPU**。

## 6. 指标

### 6.1 Detection（Primary: Image AUROC；Secondary: d′）

```text
Image AUROC ↑   (Good = 负类, NG = 正类)
d′ ↑            (pooled: (mean_def − mean_good) / sqrt((sd_def² + sd_good²)/2))
Pixel AUROC     可计算则记录（可选）
AUPRO           SKIP — known incompatibility（M²AD 无 mask NG；本轮不修）
```

### 6.2 Illumination robustness —— Paired score dispersion

对每个 `specimen × frozen view`，取 10 个 illumination 的分数 `s_01 … s_10`：

```text
R_x = mean over (specimen × frozen view) of std_illum( z )      # x ∈ {Good, NG}
R_all = mean over ALL 70 (specimen × view) units
PRIMARY = R_all  ↓ (越低越鲁棒)
同时报告 R_good / R_ng 与 median / p25 / p75
```

**⚠️ 刻度归一化（预注册决定，先于结果）**：
B2 的 L2 residual 会改变 embedding ⇒ **原始 score 刻度在不同方法间不可比**。
因此 **primary R 在标准化分数上计算**：

```text
z = (s − mu_g) / sigma_g
mu_g, sigma_g = 该方法自己在 frozen view 下全部 Good 分（200 张）的 mean / std
```

该 `(mu_g, sigma_g)` 是**方法自己的 normal-only 标定**，不使用任何 defect 信息。
**`R_raw`（按任务书字面定义、直接在原始 score 上算）一并报告**，但**不作为 primary**。
两套定义都在此冻结，**不得看到结果后改 primary**。

### 6.3 为什么不能只看 dispersion（必须写入 README）

```text
低 dispersion ≠ 自动代表好模型。
```

恒定输出模型 dispersion 也可以为 0。因此必须二维判断：

```text
Detection ↑  +  Illumination Dispersion ↓      →  robustness–detection plane
```

## 7. Pre-registered GO / HOLD / STOP

> ## `SCREENING THRESHOLD — NOT FINAL STATISTICAL CLAIM`
> **不是最终统计结论**，只是 30–60 min 级别的候选快筛带宽。

### 7.1 阈值来源（机械，先于结果）

| 符号 | 值 | 机械来源 |
|---|---|---|
| `eps_det` | **0.03** AUROC | E4-D1 实测 per-illumination AUROC sd = **0.0262**（同数据集同 view 族内的条件间波动），以及 frozen view 下 AUROC 的 Hanley–McNeil SE ≈ 0.02 → 取较大者并向上取整到 0.03 |
| `delta_robust` = `eps_robust` | **0.05**（5% 相对） | E4-D1 Original 在 frozen view 的 `R_all` 均值的标准误：`sd(R_all)/mean(R_all)/sqrt(70) = 0.2747/0.675427/8.3666 = 0.0486` → 取 0.05 |
| `meaningful_gain` | **0.03** AUROC | 与 `eps_det` 对称 |

### 7.2 🟢 GO（满足其一）

```text
Pattern A:  ImageAUROC >= AUROC_orig − 0.03   AND   R_all <= R_all_orig × 0.95
Pattern B:  ImageAUROC >  AUROC_orig + 0.03   AND   R_all <= R_all_orig × 1.05
```

### 7.3 🟡 HOLD

存在**有意义 trade-off**（detection 改善但 robustness 小幅恶化；或反之），且不满足 GO、也不触发下述 STOP 任一条。
**HOLD 只代表保留候选；不允许立即调参。**

### 7.4 🔴 STOP（任一成立 ⇒ `KILLED`）

1. detection 与 robustness 均不改善（`ΔAUROC <= +eps_det` 且 `R_all > orig×0.95`）
2. 一轴小幅改善、另一轴明显恶化
3. 与 Original 几乎相同（两轴均落在带宽内）
4. improvement 小于运行/seed 波动量级（即不超过本协议带宽）
5. 出现明显 negative transfer
6. 只有单个 illumination condition 拉高平均
7. 需要「换 view / 改参数」才能解释失败

**归档 negative result。禁止 rescue。**

## 8. Sanity（Primary FAIL ⇒ `ABORT verdict`）

| ID | 检查 |
|---|---|
| S1 | dataset / view / illumination count（Bird=12,000 记录；frozen view=120；illum 01–10） |
| S2 | frozen view 正确（`random.Random(20261008).choice(VIEWS) == "120"`） |
| S3 | Original 完全来自 E4-D1 reuse（GPU Original units = 0；行数 700；文件哈希记录） |
| S4 | B2 冻结 config 定义核验（kind 白名单；`beta_Bird` 由冻结公式得到；`L3 alpha = 0.25`） |
| S5 | X6c 冻结定义核验（`0.35/0.65`、`0.5/0.5` 常量逐字比对） |
| S6 | score finite |
| S7 | specimen × illumination 完整性（70 单元 × 10 = 700） |
| S8 | 无方法参数修改（无 α / λ / γ / weight 搜索；无 view search） |
| S9 | 未修 AUPRO（`compute_aupro` 源码哈希不变） |
| S10 | 无额外 view / category / seed |
| S11 | 输出隔离（只写 `results/e4_x/`） |
| S12 | 无 target 驱动的阈值修改（阈值常量写死在 protocol 与本脚本中） |

## 9. GPU 计划与 ETA（**不使用 micro-benchmark 线性外推 coreset**）

| 项 | 计划 |
|---|---|
| GPU units | **1（B2 一次 fit + 一次 scoring）**；X6c = 0 GPU；Original = 0 GPU |
| fit | 参考 E4-D1 同 bank 规模实测 **≈774 s**（coreset 超线性 ⇒ 不外推、按实测） |
| scoring | 700 图 @ ≈15–27 img/s ⇒ **≈0.5–1 min** |
| 峰值 VRAM | ≈14.5 GB alloc / 18.3 GB reserved（沿用 E4-D1 实测，1 worker） |
| 峰值 RAM | ≈35 GB（E4-D1 实测；本轮 700 图远小于该量级） |
| **总 wall-clock 预估** | **≈15–20 min**（含 q_Bird 特征提取 ≈1–2 min） |
| 并发 | **强制单 worker / 单 GPU unit**；**不并发 B2 + X6c**；顺序 **B2 → X6c** |

进度显示必须包含（动态 ETA，按实际图片速率）：

```text
E4-X | completed/total | XX.X% | elapsed=XX:XX | ETA=XX:XX | stage=FIT/SCORE/ANALYZE
method=B2/X6c | view=120 | images=completed/total | img/s=XX.X | GPU VRAM=XX GB
```

**若 wall-clock > 90 min 且未进入第二个 candidate ⇒ 立即检查工程异常，不无限等待。**

## 10. 明确不做（违反即 STOP + REPORT）

修 AUPRO / 改 `compute_aupro` / 研究无 mask NG / 改 τ_val / 新设 threshold / 改 X6c / 改 B2 /
调 α / 调 λ / 调 γ / 改 fusion weight / layer search / normalization search / view search /
metric search / seed search / category 扩展 / 下载其余 M²AD category /
因结果不好增加额外实验 / 自动设计 X6c-v2 / 自动进入下一方法实验。

**AUPRO = KNOWN ISSUE，本轮不处理。**

## 11. 出口条件与本轮之后的阶段

- 无论结果如何，`E4-X = candidate screening complete`。
- 研究阶段切换为 **Phase III — Module Composition Screening**。
- **E4-X 之后禁止继续围绕 X6c 自动实验。**
- 若 X6c = STOP ⇒ README 必须写明 `X6c STOPPED. No rescue experiment is authorized.`，且不得提出修复建议。
- 剩余时间只允许 CPU / 文献 / 代码资产整理（`docs/MODULE_CANDIDATE_REGISTRY.md`），**不得启动新 GPU 实验**。
