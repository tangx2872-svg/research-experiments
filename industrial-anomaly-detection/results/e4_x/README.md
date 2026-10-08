# E4-X — M²AD Bird Real-Illumination Candidate Kill Test

**日期**：2026-10-08 ｜ **性质**：**candidate kill test (KEEP OR KILL)**，不是方法优化、不是完整 external validation
**论文阶段**：Phase II — External Validation / Candidate Screening
**起点 HEAD**：`b0cc74c` ｜ **协议**：`docs/E4_X_PROTOCOL.md`（**先于读取任何 B2/X6c 结果**冻结）
**数据**：M²AD **Bird** ｜ **seed 0** ｜ **frozen view = 120** ｜ illumination **01–10** ｜ 700 张评分图

## 0. 为什么做 E4-X

E4-D0（integrity PASS）与 E4-D1（pipeline PASS）之后，必须先回答一个廉价的生死问题，而不是直接进入昂贵的 E4-A：

> 在**真实物理 illumination variation** 下，冻结的 B2 / X6c 是否比 Original 显示出**足够明显**的
> robustness–detection 优势，值得保留进入下一轮候选池？

**本实验不证明论文、不优化 X6c。**

## 1. Frozen view（机械选择，不可更换）

既有 protocol **不存在**「用全局固定 seed 抽一个全局 view」的规则（E4-D0 是 per-specimen 抽 view）⇒ 采用 **Rule B**：

```text
VIEW_SELECTION_SEED = 20261008
FROZEN_VIEW = random.Random(20261008).choice(12 个合法 view) = "120"
```

**冻结后未更换。** 结果难看也没有换。

## 2. Original reference（**直接复用 E4-D1，GPU = 0**）

来源 `results/e4_d1_smoke/per_image.csv`（sha256 `5dc8db3a93371210…`），筛 `view=="120"` → **700 行**。
**Original GPU units = 0**，未重跑。

## 3. 本轮实际执行的 GPU 工作

| 项 | 值 |
|---|---|
| GPU units | **1**（B2 一次 fit + 一次 scoring）；X6c = 0 GPU；Original = 0 GPU |
| 顺序 | **B2 → X6c**，**未并发**（单 worker） |
| `q_Bird`（normal-only，冻结 11A 实现） | 3,600 张 train/good，**95.1 s**，`S_IN_L2=1.5210 / S_IN_L3=1.0452` → **q=1.455177** |
| **`beta_Bird`** | `clip(0.25·(q/q_ref)³, 0.05, 0.50)`，`q_ref=q_bottle=1.5726841688159168` → **0.198045** |
| B2 fit | **778.4 s**（coreset 122,879 候选项 @≈167 it/s） |
| B2 scoring（700 图） | **43.9 s** |
| **总 wall-clock** | **≈822.9 s = 13.7 min**（含 q_Bird 共 ≈15.3 min） |
| **peak VRAM** | **14,497 MB allocated / 18,258 MB reserved**（24,126 MB） |
| **peak RAM** | 2,567 MB |

> `beta_Bird = 0.198` 落在 MVTec 冻结 β 区间下沿附近（bottle 0.2500 / cable 0.2948 / hazelnut 0.2279 / screw 0.2164 / grid 0.4601）。
> **β 由冻结公式机械得出，无任何候选比较、未读任何 test/defect 数据。**

## 4. Sanity — **12 / 12 PASS**

S1 dataset/scope ｜ S2 frozen view（seed 20261008 → 120）｜ **S3 Original 完全来自 E4-D1 reuse（GPU=0）** ｜
**S4 B2 冻结定义**（`residual(λ=0.198045)` + `alpha_in(0.25)`，kind 白名单）｜ **S5 X6c 冻结定义**（`0.35/0.65`、`0.5/0.5` 逐字）｜
S6 score finite ｜ S7 specimen×illumination 完整（70×10=700）｜ S8 无参数搜索 ｜
**S9 AUPRO 未修**（`experiment5a_h_runner.py` sha256 `43921a763135e837` 与 HEAD 逐字一致）｜
S10 无额外 view/category/seed ｜ S11 输出隔离 ｜ S12 阈值硬编码未被改动。

## 5. 结果（Primary / Secondary）

| Method | **Image AUROC ↑** | d′ ↑ | R_good ↓ | R_ng ↓ | **R_all ↓** | Rraw_all（非 primary） | Decision |
|---|---:|---:|---:|---:|---:|---:|---|
| **Original** | 0.7677 | 1.2068 | 0.7789 | 0.6341 | **0.6754** | 1.9635 | **REFERENCE** |
| **B2** | 0.7680 | 1.1833 | 0.8418 | 0.6293 | **0.6900** | 2.4993 | **STOP** |
| **X6c** | 0.7689 | 1.2059 | 0.8142 | 0.6243 | **0.6786** | 0.6689 | **STOP** |

- **B2**：ΔAUROC **+0.0003**、`R_ratio = 1.0215`（dispersion 反而**差 2.2%**）
- **X6c**：ΔAUROC **+0.0012**、`R_ratio = 1.0046`（dispersion 反而**差 0.5%**）

**R 的刻度说明（预注册）**：primary `R_all` 在**标准化分数** `z=(s−μ_g)/σ_g` 上计算（μ_g/σ_g 取该方法**自己**的 Good 分数，normal-only）。
原因：B2 的 L2 residual 改变 embedding ⇒ **原始 score 刻度跨方法不可比**（X6c 的 `Rraw_all=0.6689` 与 Original 的 `1.9635` 显然不是同一把尺）。
`Rraw_all` 按任务书字面定义一并报告，但**不是 primary**。

## 6. 判定（机械，未经人工改写）

阈值（`SCREENING THRESHOLD — NOT FINAL STATISTICAL CLAIM`）：`eps_det=0.03`、`delta_robust=eps_robust=0.05`、`meaningful_gain=0.03`。
来源：E4-D1 实测 per-illumination AUROC sd = 0.0262 与 SE ≈0.02（取大者向上取整）；`R_all` 均值的相对标准误 `0.2747/0.675427/√70 = 0.0486` → 0.05。

| Candidate | GO Pattern A | GO Pattern B | STOP 触发 | 判定 |
|---|---|---|---|---|
| B2 | 否 | 否 | **S1（两轴均平）＋ S3（与 Original 几乎相同）** | **STOP / KILLED** |
| X6c | 否 | 否 | **S1 ＋ S3** | **STOP / KILLED** |

**附加检查（单档 illumination 是否拉高平均）**：剔除 AUROC 最高那一档 illumination 后，
`ΔAUROC` 反而转负（B2 **−0.0041**、X6c **−0.0004**）⇒ 这点微弱差异**连"单档驱动"都算不上，是噪声**。

### 结论

```text
B2 STOPPED.
X6c STOPPED.
No rescue experiment is authorized.
```

**在 M²AD Bird（seed 0, view 120, 10 illuminations）上，冻结的 B2 与 X6c 相对 Original：
检测没有实质提升，跨光照 dispersion 没有改善（甚至略差），两者都落在预注册筛选带内 —— 与 Original 不可区分。**

## 7. 负结果与必须保留的限定

1. **这是 negative result，必须归档，不得软化。** B2 与 X6c 在真实光照变化下**没有可观测优势**。
2. **单一 (category, seed, view)**：1 类别 × 1 seed × 1 view（frozen `120`）× 10 光照。**不外推到其他类别 / seed / view。**
3. **bank 受显存所迫为 1,200 图**（E4-D1 工程约束 E2），非 M²AD 全量 train；**本实验不构成对 bank 规模的验证**。
4. **阈值是 screening band，不是统计结论**；无置信区间、无多重比较校正。
5. **AUPRO 仍为 KNOWN ISSUE，本轮按协议未处理**。
6. **不能只读 dispersion**：低 dispersion ≠ 好模型（恒定输出 dispersion 也为 0）。故必须二维判断（detection ↑ 且 dispersion ↓）—— 两候选**两轴都没达到**。
7. **运行记录（显示缺陷）**：进度行 ETA 因分母混用「unit 数(2)」与「图片数(700)」而在前段偏大（显示 ~50 min，实际 13.7 min）；
   `completed/total`、`%`、`elapsed`、`img/s`、`GPU VRAM`、`stage`、`method`、`view` 均正确。**不影响任何数值结果。**

## 8. 对论文证据链的贡献

- **完成**：E4 的**候选可用性门** —— 冻结 X6c（及其上游 B2）在**真实物理光照变化**下**未被证实**比 Original 更有价值。
- **推进**：论文阶段从 *Phase II — 单方法外部验证* 切换到 **Phase III — Module Composition Screening**。
- **不改变** Exp16 的任何内部数字；只把方法结论的外部适用范围进一步收窄。
- **下一格**：Phase III 候选模块组合筛选（`docs/MODULE_CANDIDATE_REGISTRY.md`）。

## 9. 产物

| 文件 | 说明 |
|---|---|
| `summary.csv` | 三方法 primary/secondary 指标 |
| `verdict.json` / `decision_detail.json` | 机械判定与逐条 STOP 触发 |
| `illumination_metrics.csv` | 3 方法 × 10 光照的 AUROC / d′ / good_mean / ng_mean |
| `per_image_{original,b2,x6c}.csv` | 700 行逐图分数 |
| `sanity.json` / `info.json` | 12 项 sanity 与运行元数据 |
| `figures/fig1_detection_vs_dispersion.png` | robustness–detection plane（理想方向 = 左上） |
| `figures/fig2_illumination_auroc.png` | 10 档 illumination AUROC 曲线 |
| `figures/fig3_paired_dispersion.png` | Good / NG paired dispersion 分布 |

## 10. 本轮未做

未修 AUPRO / 未改 `compute_aupro` / 未研究无 mask NG / 未改 τ_val / 未新设 threshold /
未改 X6c / 未改 B2 / 未调 α / λ / γ / weight / layer / normalization / view / metric / seed /
未扩展 category / 未下载其余 M²AD category / 未设计 X6c-v2 / **未自动进入下一方法实验**。
