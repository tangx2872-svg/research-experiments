# E16 — R1 Signal Qualification · FROZEN PROTOCOL

冻结时间：2026-10-14（**在任何 E16 新结果产生之前**）
HEAD：`55542e9`（push 后 0/0）
性质：**资格赛（qualification），不是搜索，不是调参，不是救火**

---

## 0. Research Question（唯一问题）

> R1 illumination-consistent adaptation 在 Bird / Car 上观察到的正收益，是否足够**稳定、简单、可解释**，从而具备进入论文方法阶段的资格？

具体化为 4 条：

1. 在 Bird / Car 两个 illumination-sensitive 类别上产生**稳定正收益**；
2. 在 Cube **control** 类别上**不造成明显退化**；
3. 同时**改善或保持** illumination robustness；
4. 保持方法结构**简单**，具备应用型论文价值。

**本轮不回答**：R1 是否 universal、是否能外推到其他数据集/backbone。

---

## 1. 冻结实验矩阵

```
Categories : Bird, Car, Cube          （3）
Seeds      : seed0, seed1, seed2      （3）
Methods    : F0 = frozen DINOv2-B final-layer + LayerNorm + 1-NN-max
             R1 = z' = z + Wz          （768->768, zero-init）
------------------------------------------------------------
逻辑矩阵   : 3 x 3 x 2 = 18 metric-cells
需 GPU 训练的新 cell = 0（AUROC/AP 全部历史可复用；见 §3 复用规则）
```

**禁止**：改 bank size / 改 view120 / 改 backbone / 改输入分辨率 / 改 kNN / 改 score aggregation / 改 R1 参数（结构、lr、step、early-stop、α、augmentation 强度）/ seed-specific tuning。

### R1 冻结配置（不得修改，来自 E13-A/B）
| 项 | 值 |
|---|---|
| 结构 | `z' = z + W z`，W: 768→768，**zero-init** |
| trainable params | 589,824 |
| 损失 | `L = L_cons + 0.1 · L_preserve`（β=0.1 冻结） |
| optimizer / lr | AdamW / 1e-3 |
| batch / steps | 8 / ≤300，early-stop 30（每 5 步评估 val consistency） |
| train/val | bank 240 / 60（按 image_path 排序切分，与 E13 一致） |
| augmentation | mild illumination：{brightness, contrast, gamma} × {0.85, 1.15} = 6 变体，**无几何变换** |
| DINO | 全 frozen（grad 不流经 backbone；特征预先缓存） |

---

## 2. 本轮评价指标（定义冻结）

每个 (category, seed, method) 至少输出：

| 字段 | 定义 |
|---|---|
| `AUROC` | 官方 `image_anomaly` + raw `detectable` 口径，1-NN-max 图像分数 |
| `AP` | 同上标签口径 |
| `d_prime` | `(mean_ng - mean_good) / sqrt(0.5*(var_ng + var_good))`，分数来自 detectable 过滤后的 test |
| `robustness_feat` | **特征级光照鲁棒性** = `mean_{i,k}[1 - cos(F(x_i), F(T_k(x_i)))]`，**越低越鲁棒**；F = 用于打分的表示（F0 = L2-norm final；R1 = L2-norm adapted）；x_i = bank 300 normal；T_k = 6 个 mild 光照变换 |
| `robustness_score` | **分数级光照鲁棒性** = `mean_{i,k}[ norm_s |s(F(T_k(x_i))) - s(F(x_i))| ]`，其中 s = 1-NN-max 图像分数（bank 自身作为 memory），`norm_s` = 用该 cell 的 test 分数标准差归一化（`std(test_scores_F0)` 固定作分母，保证 F0/R1 可比）；**越低越鲁棒** |
| `runtime_s` | 该 cell 的 extract+train+score 墙钟秒 |
| `peak_gpu_mb` | `torch.cuda.max_memory_allocated()` |

派生：`ΔAUROC = R1 - F0`、`ΔAP = R1 - F0`、`Δd_prime`、`Δrobustness_feat`、`Δrobustness_score`
（**必须同时保留 raw metrics**，不得只留 delta）。

---

## 3. 复用规则（强制，禁止重跑）

1. **F0 的 9 个 cell：全部直接复用**历史结果（seed0 来自 `results/e13a/e13a_screening.csv`，seed1/2 来自 `results/e13b/e13b_results.csv` 的 `r0_*` 列）。**禁止重跑**。
2. **R1 的 9 个 cell 的 AUROC/AP：直接复用**同一批 CSV 的 `R1` / `r1_*` 列。`source = REUSED`。
3. `d_prime` / `robustness_*` / `runtime` / `peak_gpu_mb` **历史上未落盘**（E13 未保存 per-image scores 与 adapter 权重）→ 必须做一次**确定性复现运行**获得。
4. 复现运行**必须**证明其身份：其 AUROC/AP 与 §3.1/3.2 的历史值必须 **一致到 1e-6**；任一处不匹配即视为复现失败 → 该 cell 标记 `REPRO_FAIL` 并停机报告（**不得**用复现值覆盖历史值）。
5. 复现运行**不得**改变任何超参；seed 控制包括 `torch.manual_seed(seed)` + `np.random.default_rng(seed)` + `E8.bank_for(pool, seed)`。
6. `summary.csv` 的 `source` 列：AUROC/AP 行标 `REUSED`（附 `repro_ok=True`）；d′/robustness 行标 `NEW`。

---

## 4. 冻结判据：GO / HOLD / STOP

### GO（满足全部 5 条）
```
1. Bird + Car 的 6 个 category-seed cells 中：至少 4/6 的 ΔAUROC > 0
2. Bird + Car：mean ΔAUROC >= +0.010
3. Bird + Car：mean ΔAP >= 0
4. illumination robustness：总体方向改善
   （判据：6 个 cell 中 robustness_feat 与 robustness_score 的 mean Δ 均 < 0，
     即两个鲁棒性指标都朝"更鲁棒"方向移动；且不得出现单个 cell 的
     Δrobustness 恶化 > +10% 相对量）
5. Cube：mean ΔAUROC >= -0.005
```
Cube 不要求提升；Cube 是 **negative/control category**，用于证明方法不是无条件强行改变特征。

### HOLD（满足其一）
```
- Bird + Car mean ΔAUROC 落在 [+0.005, +0.010)
- 或只有 Bird / Car 其中一个类别表现稳定（另一类 mean ΔAUROC < +0.005）
- 或 AUROC 有弱正收益但 robustness 信号不够清晰（§4 GO-4 不成立，但确实未恶化）
```
HOLD 后**不允许**立即大规模调参；只能进入一个非常小的 E17 lightweight refinement。

### STOP（任一）
```
- Bird + Car mean ΔAUROC < +0.005
- 或不同 seed 符号频繁翻转（Bird 或 Car 的 3 个 seed 中正负混合且 mean < +0.005）
- 或明显依赖单个 seed（去掉最好 seed 后 mean ΔAUROC < +0.005）
- 或 R1 显著损害 Cube（mean ΔAUROC < -0.005）
- 或 robustness 与 detection 同时无稳定改善
```
STOP → **R1 主线终止**，不允许继续调参救火。

### 止损（early STOP）
- Bird 3 个 seed 全部 `ΔAUROC <= 0` → 立即 early STOP，不跑 Car/Cube 补全。
- Bird 明显正、Car 前两个 seed 明显负 → 仍完成 Car seed2 后再判断。

---

## 5. Diagnostic（低成本，不训练新模型）

完成主矩阵后，对每个 category 计算 adaptation 前后（F0 vs R1）：

| 代号 | 指标 | 定义 |
|---|---|---|
| A | normal feature dispersion | bank normal patch 特征的 mean pairwise cosine（越高越聚集）与 per-image std 均值 |
| B | illumination-group centroid distance | test 按 illumination 分组（I01..I10），组质心两两 cosine 距离均值 |
| C | good-vs-defect separation | `d_prime` + good/NG 质心 cosine 距离 |
| D | illumination shift magnitude | `mean_i ||F(T(x_i)) - F(x_i)|| / ||F(x_i)||`（bank normal） |
| E | adaptation induced centroid drift | `||centroid(F_R1) - centroid(F_F0)|| / ||centroid(F_F0)||` |

只回答：**Bird / Car 是否因 illumination-induced feature shift 更明显 → R1 有收益；Cube 是否 shift 小/baseline ceiling → 收益有限。**
方法限：PCA / cosine / L2 简单统计 + bootstrap CI + group-wise mean/std。**禁止新建复杂机制模型。**

---

## 6. 输出目录

```
results/e16/
  protocol.md          <- 本文件（冻结）
  audit.md             <- P0 只读审计
  metrics_raw.csv      <- 每 cell × 每指标 raw
  summary.csv          <- §2 字段 + source(REUSED/NEW) + repro_ok
  diagnostics.csv      <- §5 的 A–E
  verdict.json         <- GO/HOLD/STOP + 触发条件
  decision_memo.md
  README.md            <- 必须回答 Q1–Q6
  figures/             <- fig1..fig5（数据不足则不强行生成）
  logs/
```

---

## 7. 禁止事项（本轮）

- 新 detector / 新 backbone / scoring family / fusion family / teacher-student 搜索
- 大规模超参搜索；看结果后改 lr / hidden dim / β / augmentation / steps
- seed-specific tuning；为某个 category 单独优化
- 重跑已有 F0/R1 cell（除 §3.4 的确定性复现）
- 自动继续 E17

## 8. 最终科研解释约束

- 若 GO，只允许主张：「R1 在本次 M²AD Bird/Car 光照敏感场景上显示**稳定收益**，而在 Cube control 类别中**没有明显负面影响**。」
- **不得**主张 universal robustness / 所有工业异常检测都有效 / 所有 illumination 场景都有效。
- diagnostic 支持时只能写 **"consistent with"** illumination-induced feature dispersion reduction，**不得**写 causal proof。

---

## 附注 1 — 身份校验实现修正（2026-10-14，在任何 E16 科学指标产生之前记录）

首次身份校验失败（`r0_ok=False r1_ok=False`，失配 ~4e-6 > 1e-6 容差）。诊断与处置：

1. **决定性实验**（`/tmp/e16_nondet.py`，只读）：同一进程内对同一批图像做两次完全相同的 DINOv2 前向，
   `TEST feature max|Δ| = 0.0`、`BANK feature max|Δ| = 0.0`；同样的特征做两次 `nn_image_scores`，
   `max|Δ| = 0.0`。→ **前向与打分在同进程内是确定性的，失配不是浮点非确定性。**
2. **真因**：`e13a_screening.csv` / `e13b_results.csv` 是 `f"{x:.5f}"` 的**5 位小数展示格式**（有损 ~4e-6）；
   权威全精度记录在 `e13a_partial.json`（`sanity` = E13-A 自己的 R0 复现值；`cells["cat|R1"]`）与
   `e13b_partial.json`（`cells["cat|seed"]["r0"/"r1"]`）。
3. **处置**：history loader 改为读取全精度 partial JSON（修正的是"读哪个文件"，不是容差）。

**不变更**：1e-6 容差、§4 的 GO/HOLD/STOP 判据、§1 的 R1 超参、§2 的指标定义、§3 的复用规则。
本修正仅涉及历史值的**引用来源**，且在任何 d′ / robustness 数值被读出之前完成记录。
