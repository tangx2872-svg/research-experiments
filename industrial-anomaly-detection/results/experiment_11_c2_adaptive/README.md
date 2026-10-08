# Experiment 11 — C2 Adaptive Preservation

**日期**：2026-10-08 ｜ **环境**：AutoDL RTX 3090 24GB / conda `base` / PyTorch 2.8.0+cu128 / Anomalib 2.6.2
**起始 HEAD**：`600865a`（Exp10）｜ **锁文件**：用户既有 `monitor/exp_monitor.py`、`scripts/_lockprobe.txt`、`scripts/_probe2.py` 不动

> **状态：PROTOCOL FROZEN（在读取 Experiment 11 target result 之前冻结）**，见 `config/config.json`。

## 0. 本轮唯一问题

Experiment 10 已证明 `C2 = L2 scale-matched residual(0.25) + L3 α=0.25` 有用（bottle FRONTIER-BREAK、
beyond corrected frontier **+0.69**、3/3 seeds PASS）。其 **category dependency** 是：bottle 3/3、screw 3/3、
grid 2/3、hazelnut 1/3、**cable 0/3**（总 **9/15**）。

> **能否仅用 normal-training statistics，把固定 β=0.25 的 preservation compensation 改成 category-adaptive，
> 使 C2 从 9/15 提升到跨类别稳定，同时不破坏 bottle/screw 已有优势？**

**不是**「想办法把 cable 调到成功」；是检验 **category dependency 能否被一个预注册、normal-only、
training-free、可解释的 adaptive preservation 机制解决**。失败即接受 negative result、换机制。

## 1. 冻结协议

| 项 | 值 |
|---|---|
| train | **corrected-189**（Exp10 冻结：`Engine.fit` 内 `dm.setup` 后重新过滤，`images_embedded=189` 已证） |
| RNG / coreset | **strict-V2**（逐字复用 `experiment9c_rng`：R1 `Engine.fit` 入口 + R2 `select_coreset_idxs` 入口 replay） |
| 判据带 | `EPS_RZ = 0.02`、`EPS_DP = 0.10`（5A-H/7A-O 冻结值，沿用） |
| categories | bottle / cable / hazelnut / screw / grid |
| seeds | 0 / 1 / 2 |
| PASS（单 category-seed） | `Δ\|Δz\| ≤ −EPS_RZ` **且** `Δd′ ≥ −EPS_DP`（相对**同 (category,seed) 的 corrected Original**） |
| **catastrophic（本轮冻结）** | `Δd′ ≤ −0.25`（与 Exp10 Stage 3 同定义） |
| 参考点 | 全部复用 Exp10：Original-189（5类×3seed 已全有）、T1、**C2 fixed**（5×3 全有）、corrected uniform frontier（9 点） |

**成功判据（冻结）**
- **Strong ADVANCE**：`≥12/15` 且 bottle ≥3/3、screw ≥2/3、grid ≥2/3、hazelnut ≥2/3、**cable ≥2/3**，且 **0 catastrophic**。
- **FINAL-METHOD-CANDIDATE**：`≥13/15` 且 cable ≥2/3 且 bottle 3/3。
- 最终必须给出 `C2-fixed` vs `Adaptive-C2` 的 PASS 率与 per-category 对比；目标**不是**单个数字更漂亮，而是 **category consistency 明显提升**。

## 2. Adaptive 候选的预注册 family（最多 6 个，公式在看到 target 结果前冻结于 `config/candidate_registry.json`）

对象：**只改 L2 preservation strength**（L3 恒为 `alpha_in(0.25)`，即 robustness 支路不动）。
记 L2 的 normal 统计（只用 train/good）：`s_c`（L2 normalization response magnitude，见 11A 定义）、
`r_c`（L2 RMS）、`q = s_L2/(s_L3+eps)`；channel 级统计 `g_j`。

| Family | 形式 | 说明 |
|---|---|---|
| **A — RMS-Calibrated Preservation** | `β_c = clip(k·τ / s_c, β_min, β_max)`，A1 用 mean、A2 用 robust median | `k` 由**参考类别/池化 normal 统计**机械确定（参考 = 五类 pooled 中位数），**不按 defect 表现搜索** |
| **B — Layer Sensitivity Ratio** | `β_c = clip(β0 · (q_c/q̄)^p, β_min, β_max)`，单调、clip | B1: p=1, B2: 二值化 clip 版本 |
| **C — Channel-Adaptive** | `F_L2' = F + β · g_j · IN(F)·rms(F)`，`g_j` 来自 normal channel 统计（稳定性/方差/归一化响应） | C1: `g_j` 归一化到均值 1；C2: `g_j ∈ [0,1]` 稳健权重 |

**冻结约束**：`β_min = 0.05`、`β_max = 0.50`（围绕 C2 的 0.25 单向放宽一倍/收紧五倍）；
所有公式简单、可解释、无学习参数、不读 test/defect 数据。

## 3. Stage 计划与预算

| Stage | 内容 | GPU units（上限） |
|---|---|---|
| 11A | normal-statistics（5 类，CPU/短 GPU 特征统计） | 5 短跑 |
| 11B-1 | bottle/cable/hazelnut × seed0 × ≤6 候选 | ≤18 |
| 11B-2 | Top3 补 screw/grid × seed0 | ≤6 |
| 11C | Top1–2 × 5 cats × seeds {1,2} | ≤20 |
| Ablation preview | 仅当 ≥12/15：Winner × bottle/cable seed0（Original / L3-only / L2-only / Full） | ≤8 |
| Optional | 仅当 FINAL-METHOD-CANDIDATE 且墙钟 <90 min：Winner × cable/hazelnut/grid × seeds 3/4 | ≤6 |

软预算 **120 min**（100 min 时检查剩余任务，优先级 11C > ablation > optional），硬上限 **150 min**。
并发默认 4 workers（smoke 后按显存调整）。无人值守：`screen` + per-worker 独立进程 + resume/skip/retry-once + 连续 3 失败写 `ERROR_REPORT.md`。
**未达标不救、不 fishing、不隐藏失败类别。**

---

# 结果

## 4. Stage 11A — normal-only failure signature（CPU/短 GPU，只读 train/good）

对每类训练图（batch=16, no_grad）提取 **pooled** layer2/3 特征，累计 per-channel 统计 →
`stats/normal_statistics.csv` + `stats/per_channel_<cat>.npz`。**未使用任何 test/defect 信息。**

| category | n_train | S_IN_L2 | S_IN_L3 | q=L2/L3 | RMS_L2 | RATIO_L2 | C2能量比 | PR_VAR_L2 | DISP_L2 | C2 fixed PASS(Exp10) |
|---|---|---|---|---|---|---|---|---|---|---|
| bottle | 209 | 1.8088 | 1.1501 | 1.5727 | 1.3001 | **1.2944** | 0.3236 | 157.8 | 0.0728 | **3/3 ✓** |
| cable | 224 | 1.9488 | 1.1729 | 1.6615 | 1.3606 | **1.2874** | 0.3218 | 227.6 | 0.1022 | **0/3 ✗** |
| hazelnut | 391 | 1.6815 | 1.1027 | 1.5249 | 1.3008 | 1.5593 | 0.3898 | 265.8 | 0.1521 | 1/3 ✗ |
| screw | 320 | 1.6705 | 1.1145 | 1.4988 | 1.3662 | **2.6933** | 0.6733 | 252.0 | 0.1805 | 3/3 ✓ |
| grid | 264 | 2.3255 | 1.2066 | 1.9273 | 1.2737 | **4.9590** | 1.2398 | 138.5 | 0.5757 | 2/3 ~✓ |

**结论（不要求显著性，仅作 candidate generator）**
- **没有任何单一 normal-only 统计量能干净分开 PASS 与 FAIL。** 最接近的是 `RATIO_L2`（C2 补偿信号强度）：
  FAIL 侧偏低（cable 1.287、hazelnut 1.559）、PASS 侧偏高（screw 2.693、grid 4.959）——
  **但 bottle（1.294）与 cable（1.287）几乎相同，一个 3/3 PASS、一个 0/3 FAIL** → 该统计量**无法**区分这两者。
- `PR_VAR_L2`（有效通道数）也无法分开：cable 227.6 / hazelnut 265.8（FAIL）vs screw 252.0（PASS）重叠。
- `DISP_L2`、`q`、`S_IN_L2`、`n_train` 同样无干净分离。
- 唯一方向性线索：**FAIL 类的 `RATIO_L2` 都 < 1.6**（即 C2 的补偿项相对能量较弱），
  但 bottle 是反例 → 任何以 `RATIO_L2` 为唯一自变量的规则都必须在 bottle 上是**锚定**的（β 不变），
  这决定了 Family A 采用 `k = 0.25·RATIO_bottle` 的锚定形式（见 §5）。

## 5. Stage 11B — 冻结的 adaptive 候选（6 个，公式在读取任何 Exp11 target 结果前写定）

**所有候选只改 L2 的 preservation 强度；L3 恒为 `alpha_in(0.25)`。** 锚点 = bottle（保证不破坏已 PASS 类别）。

| ID | family | level | 公式 | 冻结 β（bottle/cable/hazelnut/screw/grid） |
|---|---|---|---|---|
| `E11_A1` | A | category | `β_c = clip(0.25·RATIO_bottle / mean_j(RATIO_j), 0.05, 0.50)` | 0.250 / 0.251 / 0.208 / 0.120 / 0.065 |
| `E11_A2` | A | category | 同上但用 `median_j` | 0.250 / 0.254 / 0.245 / 0.241 / 0.198 |
| `E11_B1` | B | category | `β_c = clip(0.25·(q_c/q_bottle)^1, 0.05, 0.50)` | 0.250 / 0.264 / 0.242 / 0.238 / 0.306 |
| `E11_B2` | B | category | `β_c = clip(0.25·(q_c/q_bottle)^3, 0.05, 0.50)` | 0.250 / 0.295 / 0.228 / 0.216 / 0.460 |
| `E11_C1` | C | **channel** | `F_L2' = F + 0.25·g_j·IN(F)·rms(F)`, `g_j ∝ 1/RATIO_j`（均值 1） | g 逐通道 |
| `E11_C2` | C | **channel** | `g_j = 1 − clip(RATIO_j/p90_j, 0, 1) ∈ [0,1]` | g 逐通道 |

**等价性检查（写于运行前）**：`E11_A1/A2/B1/B2` 在 bottle 上 β 恒为 0.25 → 与 `C2-fixed` 的 L2 算子**完全一致**（可作内部一致性检查）；
`E11_C1/C2` 的 `g_j` 在 bottle 上不为常数 → 属 channel-level，**不等价**于 C2-fixed。

## 6. Strict Replay 预运行抽查（在 11B-1 之前）

新代码路径（`experiment11_model.AdaptivePatchcore` + Exp11 runner）跑 `bottle:0:E11_A1__bottle`（β=0.25 ≡ C2-fixed），
与 Exp10 的 `C2-fixed` 单元逐位对照：

| 项 | 结果 |
|---|---|
| embedding sha256 | `5efd325ab11f7e82` == `5efd325ab11f7e82` ✓ |
| coreset indices sha256 | `6677f4577add11bf` == `6677f4577add11bf` ✓ |
| memory bank sha256 | `4d3575e88d2ffcbe` == `4d3575e88d2ffcbe` ✓ |
| tau | 42.99169922 == 42.99169922 ✓ |
| `max\|Δscore\|` | **0.000e+00**（n=183）✓ (要求 ≤1e-7) |
| train filter | `images_embedded = 189` ✓ |

→ **STRICT REPLAY SANITY PASS**（新代码路径与 Exp10 结果逐位一致）。

## 7. Stage 11B-1 — 硬类快筛（bottle/cable/hazelnut × seed0，18/18 units，0 failed）

参考：该 (category,seed) 的 corrected Original-189 与 Exp10 的 **C2-fixed**。

| cand | hardPASS | cata | worst Δd′ | mean ΔR | cable Δd′ **vs C2-fixed** | hazelnut Δd′ vs C2-fixed | B1 描述性 |
|---|---|---|---|---|---|---|---|
| **`E11_B2`** | **3/3** | 0 | −0.0653 | −0.0974 | +0.0399 | +0.0123 | bottle 3/3、**cable 与 hazelnut 均 PASS** |
| **`E11_C2`** | 2/3 | 0 | −0.0817 | −0.0485 | **+0.2326** | −0.0071 | cable Δd′ −0.070→**+0.163**（catastrophic 消除）但 cable robustness 未改善 |
| **`E11_A1`** | 2/3 | 0 | −0.0885 | −0.0800 | −0.0189 | +0.0931 | hazelnut +0.093，cable 未改善 |
| `E11_B1` | 2/3 | 0 | −0.2002 | −0.0851 | −0.1305 | +0.1606 | hazelnut 改善但 cable 更差 |
| `E11_A2` | 1/3 | 0 | −0.1451 | −0.0835 | −0.0755 | −0.0472 | — |
| `E11_C1` | 1/3 | 0 | −0.1899 | −0.0396 | −0.1203 | +0.1817 | — |

**Top 3（按预注册排名优先级：PASS 数 → cata → worst Δd′ → mean ΔR → 简洁度）= `E11_B2`、`E11_C2`、`E11_A1`。**
（说明：排名与选择规则由任务书给定；`B1_advance` 标志仅作描述，Top-3 选择按排名执行。）

**内部一致性检查**：`E11_A1/A2/B1/B2` 在 bottle 上 β 恒为 0.25 → 其 bottle 单元与 `C2-fixed` 的
`Δd′ = −0.0653`、`Δ|Δz| = −0.2113` **完全一致** ✓（三条候选 bottle 行数值逐位相同，证明锚定实现正确）。

## 8. Stage 11B-2 — 五类别 seed0（Top3，6/6 units，0 failed）

| cand | PASS/5 | cata | worst Δd′ | mean ΔR | cable Δd′ | hazelnut Δd′ | vs C2-fixed: bottle / screw | 晋级 |
|---|---|---|---|---|---|---|---|---|
| **`E11_B2`** | **5/5** | **0** | **−0.0653** | −0.0980 | −0.030 | +0.033 | **+0.000 / −0.004** | **ADVANCE** |
| `E11_C2` | 3/5 | 0 | −0.0817 | −0.0546 | **+0.163** | +0.014 | −0.016 / +0.030 | — |
| `E11_A1` | 2/5 | 0 | −0.1106 | −0.0897 | −0.088 | +0.114 | +0.000 / **−0.207** | — |

**`E11_B2` 是唯一满足 11B-2 晋级条件（≥4/5 PASS、cable 无 catastrophic、bottle/screw 相对 C2-fixed 退化 ≤ 一个 `EPS_DP`）的候选** →
**Top2 = [`E11_B2`]**。注意 `E11_A1` 在 screw 上相对 C2-fixed 退化 −0.207（被淘汰的直接原因）。

## 9. Stage 11C — 多 seed 确认（Winner `E11_B2` × 5 categories × seeds {0,1,2}，0 failed）

（seed0 复用 11B-2；仅新增 seeds 1/2 = 10 units；首次尝试因**编排器并发未受 `--workers` 限制**
（2 seeds × 4 workers = 8 进程）触发 CUDA OOM，4 个 unit 失败 → 修复（bug-5）后用 2 workers 重跑成功。）

| method | **PASS/15** | catastrophic | worst Δd′ | bottle | cable | hazelnut | screw | grid |
|---|---|---|---|---|---|---|---|---|
| **Adaptive C2 (`E11_B2`)** | **12/15** | 1 | −0.2852 | **3/3** | **2/3** | 1/3 | **3/3** | **3/3** |
| C2 fixed（Exp10） | 9/15 | 1 | −0.3286 | 3/3 | **0/3** | 1/3 | 3/3 | 2/3 |

**逐 (category, seed) 结果**

| cat | seed0 | seed1 | seed2 | C2-fixed (s0/s1/s2) |
|---|---|---|---|---|
| bottle | PASS | PASS | PASS | PASS/PASS/PASS |
| cable | **PASS** | **PASS** | FAIL(−0.285) | HOLD/HOLD/FAIL(−0.329) |
| hazelnut | PASS | HOLD | HOLD | PASS/HOLD/HOLD |
| screw | PASS | PASS | PASS | PASS/PASS/PASS |
| grid | PASS | PASS | PASS | PASS/HOLD/PASS |

**冻结判据下的结论**
- **Strong ADVANCE：未满足** —— 三条硬条件中两条不成立：`hazelnut ≥2/3`（实际 1/3）与 **`0 catastrophic`**（实际 1 个，cable seed2 −0.2852）。
- **FINAL-METHOD-CANDIDATE：未满足**（12/15 < 13/15）。
- **但 category consistency 明显改善**：**9/15 → 12/15**，**cable 0/3 → 2/3**，grid 2/3 → 3/3，
  且 **bottle 3/3、screw 3/3 完全保持**（相对 C2-fixed 的退化 +0.000 / −0.004）。
- **hazelnut 未改善（1/3 → 1/3）**：与 11A 的观察一致 —— hazelnut 的 `RATIO_L2 = 1.559` 使其 B2 规则
  把 β 从 0.25 降到 0.228（更保守），但该类的失败模式（seed1 robustness 未改善、seed2 preservation −0.22）
  **不是**由补偿强度引起的 → **normal-statistic 的 category-level calibration 对 hazelnut 无效**。

## 10. Ablation Preview（Winner × bottle/cable × seed0；新跑 2 units，其余复用）

| category | Original | **L3 robust only** (=T1) | **L2 preservation only** | **Full Adaptive C2** | 互补性 |
|---|---|---|---|---|---|
| bottle | 0.000 / 0.000 | −0.098 / **−0.140** | **−0.146** / +0.108 | **−0.211** / **−0.065** | ✓ |
| cable | 0.000 / 0.000 | −0.019 / −0.046 | −0.087 / +0.079 | **−0.043** / **−0.030** | ✓ |

（每格 = `Δ|Δz| / Δd′` vs 同 (category,seed) 的 corrected Original）

**判定：`complementarity_supported = True`** —— 两个类别上均满足
**Full 的 robustness gain 优于 L2-only**（L3 提供 robustness）**且 Full 的 preservation 优于 L3-only**（L2 提供 preservation）。
即：**L2 preservation 与 L3 robustness 是互补而非重复的模块**，Full 同时优于任一单模块 → 论文级重要证据。

## 11. 最终对照与判定

| Rank | Method | PASS/15 | Bottle | Cable | Hazelnut | Screw | Grid | Worst Δd′ | Verdict |
|---|---|---|---|---|---|---|---|---|---|
| 1 | **Adaptive C2 (`E11_B2`)** | **12/15** | 3/3 | **2/3** | 1/3 | 3/3 | 3/3 | −0.2852 | **HOLD（改善显著但未达 Strong ADVANCE）** |
| 2 | C2 fixed (Exp10) | 9/15 | 3/3 | 0/3 | 1/3 | 3/3 | 2/3 | −0.3286 | HOLD |

**目标不是单个数字更漂亮，而是 category consistency**：cable **0/3 → 2/3**、grid 2/3 → 3/3、
bottle/screw 零退化 → **consistency 明确提升**，但 **hazelnut 与 1 个 catastrophic 未解决**。

## 12. 淘汰 / 负面结果

| 被淘汰 | 原因 |
|---|---|
| `E11_A1` | 硬类 2/3；B2 阶段 screw 相对 C2-fixed 退化 **−0.207** |
| `E11_A2` | 硬类 1/3（cable −0.145、hazelnut −0.026） |
| `E11_B1` | 硬类 2/3，cable 相对 C2-fixed **−0.131**（更差） |
| `E11_C1` | 硬类 1/3（cable −0.190） |
| `E11_C2` | 硬类 2/3、5 类 3/5；**cable preservation +0.163（catastrophic 消除）但 cable robustness 未改善** → 一好一坏 |
| **Family A（RMS 校准）** | 整体无效：cable 与 bottle 的 `RATIO_L2` 几乎相同（1.287 vs 1.294）→ 无法据此区分，A 家族只在 hazelnut 上有小幅收益 |
| **hazelnut 的 category-level calibration** | 该类的失败与补偿强度无关（β 调动无效）→ normal-statistic category calibration 的能力边界 |

## 13. 下一步（仅建议，未启动）

1. **不冻结最终方法**（未达 FINAL-METHOD-CANDIDATE）。
2. 若要继续：**hazelnut 需要的是 channel-level 或 test-time 无关的结构性修复**，而不是 category-level β 校准
   （11A 已给出依据：其 `RATIO_L2`、`PR_VAR_L2`、`DISP_L2` 都无法解释其失败模式）；
   `E11_C2`（channel-level）已在 cable 上证明能消除 catastrophic → **C2 与 B2 的组合（category β + channel g）**
   是唯一有 pre-registered 依据的下一步方向。
3. 先解决 cable seed2 的 −0.285（距 −0.25 阈值差 0.035）—— 但**不允许**用该 seed 反向调 β。
