# Experiment 9A — M1–M6 Method Screening（Round-1 Mini Screening）

**日期**：2026-10-07 ｜ **环境**：AutoDL RTX 3090 24GB / conda `base` / Python 3.12.3 / PyTorch 2.8.0+cu128 / Anomalib 2.6.2
**Git HEAD（运行前）**：`5384f3a` (main) ｜ **运行脚本**：`scripts/experiment9a_runner.py`、`scripts/experiment9a_model.py`、`scripts/experiment9a_analysis.py`

---

## 1. 论文导航（实验前四问）

| 项 | 内容 |
|---|---|
| 📍 论文阶段 | 方法改善阶段 → **多候选快速海选**（不是最终方法证明，不做机制深挖） |
| 🎯 要证明的话 | 「在 NDIL/α-IN 的 robustness–preservation trade-off 上，是否存在**值得继续投入**的新方法候选」 |
| 🧪 当前实验 | 用最低 GPU 成本，在 bottle/cable × seed0 上统一比较 M1–M6，排除「已有手段（uniform α frontier）已经覆盖」的候选 |
| 🚦 出口条件 | ① 有候选同时在两类别上改善 trade-off 且跑出 frontier → 晋级 Round-2；② 只有单类别改善 → HOLD，不扩展；③ 不改善/被 frontier 支配/数学冗余 → STOP；④ 出现 NaN / baseline 等价性失败 → 停止并保留现场 |

## 2. 实验目的

唯一问题：**M1–M6 中哪些方法有希望比现有 baseline 更好地改善 robustness–preservation trade-off？**
本轮目标是「筛选」，不是「证明」。失败方法直接淘汰，不做救援。

## 3. Frozen protocol

- **评价管线**：完全复用 `experiment5a_h_runner.run_config`（bank / coreset 0.1 / kNN 9 / illumination ±30% brightness·gamma / scoring / `per_image.csv`），只在 `fit_dual_model` 处注入新模型 —— 与 7A-O / 8B 同一注入方式。
- **主指标（冻结，未因结果更换）**：
  - robustness = `mean |ΔNormalScore_z|`（低好）
  - preservation = `mean defect d′`（高好）
  - 均由 `experiment5a_h_analysis.unit_metrics` 从 `per_image.csv` 重算。
- **判据带**：`EPS_DP = 0.10`（5A-H 冻结 preservation band）、`EPS_RZ = 0.02`（7A-O Tier-A 冻结 robustness band）。
- **零调参**：`k = 16`、`λ = 0.5`、`α' = 0.40091275` 全部冻结；无 grid/k/λ/α search；无多 seed；无第 3 类别。
- **数据**：MVTec AD `bottle`、`cable`；`seed = 0`；split 由 `experiment1b_defect_sensitivity.make_validation_split` 决定（val=20，train=189/204）。

## 4. M1–M6 状态与冻结定义（未擅自改变）

| 方法 | 状态 | 冻结定义要点 |
|---|---|---|
| **M1** illumination-robust representation | **NEW / RUN** | `x' = (x − μ_c(x)) · (σ_ref_c / σ_c(x)) + μ_ref_c`（per-image per-channel，输入空间）；`(μ_ref, σ_ref)` 只由 **train normal** 图像估计（n=189 bottle / 204 cable，resize 256，[0,1] 空间）；同一变换一致作用于 train / val / clean test / defect test / 光照偏移 test；α=0 |
| **M2** dual concat | **REUSE** | 严格复用 7A-O Family B `B1_g100`（15/15 panel）；0 GPU |
| **M3** nuisance subspace suppression | **NEW / RUN** | 逐层 `F' = F − U_k U_kᵀ F`，`k = 16`；`U_k` 由 **train normal** clean vs 4 个冻结光照偏移（brightness/gamma ±30%）的 feature-difference 协方差 top-k 特征向量得到；pooled pre-concat 空间；layer2/layer3 各自独立估计；α=0 |
| **M4** residual compensation | **ALGEBRAICALLY REDUNDANT / STOP** | `F_r=(1−α')F+α'IN(F)`，`F_out=F_r+λ(F−F_r)` ≡ `(1−α)F+αIN(F)`，`α=(1−λ)α'`；只跑等价性 sanity |
| **M5** selective bypass | **IMPLEMENTATION_UNRESOLVED / HOLD** | 只有 oracle channel selection，无公平 lightweight deterministic rule → 本轮不实现、不发明 heuristic |
| **M6** channel-selective | **REUSE / ORACLE UPPER BOUND** | 复用 8B `PROPOSED25`/`PROPOSED50`；**selection 使用 test defect mask → oracle，不是 deployable 方法**；0 GPU |

## 5. 哪些是真跑（GPU），哪些是历史复用（0 GPU）

**真跑 8 个 GPU unit（全部 status=OK，0 failed）**

| round | unit | 秒 |
|---|---|---|
| round0 | bottle:0:B0_original（路径完整性对照） | 93.6 |
| round0 | bottle:0:M1_illum_standard | 105.5 |
| round0 | bottle:0:M3_nuis_k16（含 basis 估计） | 121.2 |
| round0 | bottle:0:M4_equiv_lam050_a040091275（direct α_eff 对照） | 94.1 |
| round1 | cable:0:M1_illum_standard | 157.8 |
| round1 | cable:0:M3_nuis_k16（含 basis 估计） | 167.1 |
| round1 | bottle:0:M4_resid_lam050_a040091275（组合形式） | 97.3 |
| m4diag | bottle:0:M4_resid_lam050_a040091275（**同配置重复**，确定性检验） | 86.0 |

**历史复用（0 GPU）**：M2（7A-O `B1_g100`）、M6（8B `PROPOSED25|50`）、B0/B2 与全部 uniform-α frontier 工作点（5A-H、6A、6B、7A-O Q4）。

## 6. Runtime

- Round-0 wallclock **215 s**（2 workers，4 units）；Round-1 wallclock **211 s**（3 workers，3 units）；m4diag **86 s**（1 worker）。
- GPU 合计 **922.6 GPU·s（约 15.4 分钟）**，落在预设 10–18 分钟预算内。
- 峰值显存 ~2.6 GB/worker（无 concat 类双倍显存问题，无 OOM）。
- 进度/ETA：每个 unit 结束立即刷新 `progress/<round>__<worker>.json`，并打印动态
  `[progress] done/total | % | elapsed | avg/unit | ETA | finish`（`scripts/experiment_progress.py`，多 worker 感知，ETA 依据真实已完成 unit 速率动态更新）。

## 7. 结果

### 7.1 判据基准值（bottle / cable，seed 0）

| 工作点 | bottle d′ | bottle \|Δz\| | cable d′ | cable \|Δz\| |
|---|---|---|---|---|
| B0 (α=0) | 8.2650 | 0.4782 | 5.1463 | 0.3455 |
| B1 fixed (α=0.5) | 7.2638 | 0.2846 | 4.9153 | 0.2581 |
| **B2 Uniform (α=0.40091275)** | 7.4638 | 0.2703 | 5.0488 | 0.2641 |
| frontier 最优点（per-category） | α=0.8018 → 6.5025 / **0.2360** | | α=0.25 → **5.2723** / 0.3017 | |

完整 uniform-α frontier（α = 0 / 0.125 / 0.2 / 0.25 / 0.3 / 0.4009 / 0.5 / 0.6014 / 0.8018，两类别 9 点全覆盖）见 `summary/uniform_alpha_frontier.csv`。

### 7.2 各候选（seed 0，2 类别）

| 方法 | type | bottle d′ | bottle \|Δz\| | cable d′ | cable \|Δz\| | Δd′ vs B0 (2cat) | Δ\|Δz\| vs B0 (2cat) | 跑出 frontier | 被 frontier 支配 | Verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| M6 oracle 50% mask | REUSE | 8.8594 | 0.1622 | 4.7743 | 0.1750 | +0.1112 | −0.2433 | bottle | — | HOLD (ORACLE) |
| M3 nuisance k=16 | NEW | 8.3909 | 0.4695 | 5.2094 | 0.3808 | +0.0945 | +0.0133 | bottle | cable | HOLD |
| M2 dual concat (`B1_g100`) | REUSE | 8.1708 | 0.3379 | 5.1008 | 0.3275 | −0.0698 | −0.0792 | bottle | cable | HOLD |
| M1 illum standardization | NEW | 8.2904 | 0.2850 | 4.8428 | 0.6687 | −0.1391 | +0.0650 | bottle | cable | HOLD |
| M6 oracle 25% mask | REUSE | 7.7321 | 0.1276 | 4.1392 | 0.2140 | −0.7700 | −0.2411 | bottle | cable | HOLD (ORACLE) |
| **M4 residual compensation** | SANITY | 7.8499 (≡α_eff) | 0.3520 | — | — | — | — | — | — | **STOP — algebraically redundant** |
| **M5 selective bypass** | HOLD | — | — | — | — | — | — | — | — | **HOLD — implementation unresolved** |

### 7.3 排名（Round-1，2 类别 seed0 均值；**无 PROMOTE**）

| Rank | Method | Type | Bottle | Cable | Robustness (\|Δz\| 2cat) | Preservation (d′ 2cat) | Trade-off | Runtime | Verdict |
|---|---|---|---|---|---:|---:|---|---:|---|
| 1 | M6 oracle 50% mask | REUSE | 8.8594 / 0.1622 | 4.7743 / 0.1750 | 0.1686 | 6.8169 | bottle 跑出 frontier；cable 被支配 → 单侧 | 0 (复用) | HOLD — **ORACLE UPPER BOUND** |
| 2 | M3 nuisance k=16 | NEW | 8.3909 / 0.4695 | 5.2094 / 0.3808 | 0.4251 | 6.8001 | bottle 跑出 frontier，但 robustness 几乎未改善；cable 被支配 | 144 s/unit | HOLD |
| 3 | M2 dual concat | REUSE | 8.1708 / 0.3379 | 5.1008 / 0.3275 | 0.3327 | 6.6358 | bottle 跑出 frontier；cable 被支配（与 7A-O 结论一致） | 0 (复用) | HOLD |
| 4 | M1 illum standardization | NEW | 8.2904 / **0.2850** | 4.8428 / **0.6687** | 0.4768 | 6.5666 | **bottle 大幅跑出 frontier（+0.83 d′ @同 robustness）**；cable robustness 反而恶化 +0.323 → 方向相反 | 132 s/unit | HOLD |
| 5 | M6 oracle 25% mask | REUSE | 7.7321 / 0.1276 | 4.1392 / 0.2140 | 0.1708 | 5.9357 | 最激进工作点，preservation 损失最大 | 0 (复用) | HOLD — **ORACLE UPPER BOUND** |
| — | M4 residual compensation | SANITY | 7.8499 | — | — | — | 与 α_eff 同一方法 | 97 s/unit | **STOP — ALGEBRAICALLY REDUNDANT** |
| — | M5 selective bypass | HOLD | — | — | — | — | 无公平 selection rule | — | **HOLD — IMPLEMENTATION_UNRESOLVED** |

判定规则（由冻结判据带机械产生，未事后调整）：STOP 需「两类别同时恶化」或「两类别都被 uniform-α frontier 支配」；PROMOTE 需「两类别同时满足预注册 trade-off 条件且至少一侧跑出 frontier」；其余为 HOLD。

## 8. 关键观察（事实 → 解释 → 假设）

**事实**
- F1：bottle 上 M1 得到 d′ = 8.2904 / |Δz| = 0.2850；所有 uniform-α 工作点在该 robustness 水平的最好 d′ ≈ 7.46（α=0.4009）/ 7.26（α=0.5）→ M1 在 bottle 上**同时**超过 B0 的 d′（8.2650）并把 robustness 改善 40%，明显跑出既有 frontier。
- F2：cable 上 M1 的 |Δz| = **0.6687**，比 B0 的 0.3455 **恶化 +0.323（+94%）**，d′ 也降到 4.8428（−0.303）。
- F3：M3 在两类上 d′ 都不低于 B0（+0.126 / +0.063），但 robustness 基本不变（−0.009 / +0.035）；cable 上被既有 frontier 支配。M3 nuisance 子空间占 trace 比例：layer2 0.768、layer3 0.665（bottle）。
- F4：M2（concat）与 M6（oracle channel mask）都是「bottle 跑出 frontier、cable 被支配」的单侧现象，cable 是共同弱侧（与 8B 记录的 cable 负迁移一致）。
- F5：M4 组合形式与直接 α_eff 实现的 **embedding 完全等价**（max|ΔF| = 1.91e-06，相对 2.09e-06，float32 舍入级）；两实现的 **score** 并不逐位相同（max|Δscore| = 1.773，r = 0.99949，mean Δ = −0.033），但同一配置**重复运行**的 score 逐位相同（max|Δscore| = **0.0**）。
- F6：M1 不数值退化 —— per-image σ 与 train 参考 σ 之比 bottle ∈ [0.98, 1.02]、cable ∈ [0.83, 1.20]，无 σ→0 放大（`M1_EPS = 1e-6` clamp 未触发）。

**解释（基于以上事实的推断）**
- E1：F5 表明 M4 与 α_eff 是同一方法（表示层严格等价）；score 级差异来自 PatchCore **fit 路径对 1e-6 级表示扰动的放大**（KCenterGreedy 在 ~2.1×10⁵ patch 上做 21401 步顺序 argmax，bank 组成对微小扰动敏感），而非实现错误 —— 依据是「同配置重复运行 bit-identical」+「表示层等价」两条独立事实。因此「score 逐位相等」对该管线不是合适的等价性判据（与 8B 预注册修正 A1 同类结论）。
- E2：F2 说明 M1 的收益不是通用 illumination-invariance 收益，而是**类别相关的**；F6 排除「数值爆炸」这一解释。
- E3：F1+F4 合看，本轮唯一「跑出 frontier 且幅度很大」的信号是 M1，但它在 cable 上方向相反 → 按预注册判据只能 HOLD。

**假设（尚未被本轮证据验证）**
- H1：M1 的 cable 失效与 cable 图像「背景主导、目标细长」的统计结构有关 —— 输入空间 per-image 统计被背景主导，标准化后目标对比度被重标定，导致缺陷相关特征在光照下的漂移被放大。
- H2：M1 在 bottle 上的收益可能部分来自「bottle 图像统计高度同质」（σ 变异仅 ±2%）→ 变换近似为全局 (μ_ref, σ_ref) 对齐，因此有效；这预测 M1 在统计同质类别上更可能有效、在背景复杂类别上失败。

## 9. 异常情况

1. **进度条显示瑕疵（不影响任何结果）**：round0 最后一次刷新打印 `[progress] 4/2 | 200.0%`（completed 正确，total 在该瞬间被低估为 2）。最终值正确（4/4，见第 6 节与 `progress.log`），无法复现该瞬时状态；判定为纯显示层问题，未影响判定，未回改任何口径。
2. **M4 score 级等价性初次 FAIL → 重新取证后修正判据（完整披露）**：初次以「两实现 score 逐位相等」为判据得到 FAIL（max|Δscore| = 1.773）。经三组证据（表示层等价 1.91e-06；同配置重复运行 bit-identical；score 相关 0.99949、均值差 −0.033）改为「表示层等价 + 管线确定性」判据 → PASS。**此处改变了 sanity 判据，属方法学修正而非结果驱动调参**：它把 M4 的结论从「不确定」变为「明确冗余」，不改变任何候选的晋级/淘汰，未修改候选定义或主指标。
3. `monitor_progress.py` 自动发现 glob 本轮修复（基路径中的 `*` 不被 pathlib 展开 → 永远找不到分片）；修复前由 `monitor/exp_monitor.py` 事件监控作为后备。监控脚本不参与任何实验计算。

## 10. Negative results（全部保留，不救援）

- **M1 在 cable 上显著恶化 robustness**（|Δz| +0.323）与 preservation（d′ −0.303）→ 不得声称 M1 是通用 illumination-robust 方法。
- **M3 的 robustness 收益近似为零**，其 d′ 提升在 cable 上仍被既有 frontier 支配 → 「去掉 top-16 光照差分方向」不足以改变 trade-off 形状。
- **M2（dual concat）与 M6（oracle channel mask）只能沿既有 frontier 移动**，两类别不同时成立；cable 是共同弱侧。
- **M6 是 oracle**（selection 用 test defect mask），其数字**不能**当作方法性能；25% mask 在 cable 上甚至显著低于 B0。
- **M4 不构成新方法**（代数冗余）；M5 本轮不可实现（无公平 selection rule）。
- 本轮**无候选达到 PROMOTE**；bottle/cable 不一致是最强共同特征。

## 11. 出口条件判断

- ① 两类别同时改善且跑出 frontier：**未满足**（最强候选 M1 在 cable 方向相反）→ 不晋级方法结论。
- ② 仅单类别改善：**满足**（M1/M2/M3 的 bottle 侧）→ 记 HOLD，不扩展。
- ③ 被 frontier 支配 / 数学冗余：M4 = STOP（冗余）；M1/M2/M3/M6 在 cable 侧被支配。
- ④ NaN / baseline 等价性失败：未触发 —— 10/10 sanity PASS，且 `bottle:0 B0_original` 与历史 5A-H B0 **逐位一致**（max|Δscore| = 0.0），证明新 runner 未破坏 baseline 管线。

**结论：Round-1 无 PROMOTE；M4 永久 STOP（代数冗余）；M5 保持 HOLD；M1/M2/M3/M6 全部 HOLD（单侧信号，bottle/cable 不一致）。**

## 12. Sanity 汇总（10/10 PASS，明细见 `summary/sanity_checks.csv`）

S1 数据完整；S2 28 个所需来源全部就位；S3 8556 个分数有限（无 NaN/Inf）；S4 行数一致（bottle 183 / cable 402）；**S5 9A B0 vs 5A-H B0 逐位一致（max|Δscore| = 0.0）**；S6 M4 表示层等价 + 管线确定性；S7 M6 复用值与 8B 冻结指标逐位一致；S8 M2 来源为 7A-O 冻结 `B1_g100`；S9 8/8 unit status=OK；S10 新方法无 target leakage（M1 参考统计、M3 basis 均只用 train normal，`n_train` 已记录）。

## 13. 对论文证据链的贡献

本实验**没有推进方法贡献格**，它推进的是「候选淘汰」格：
- 完成第 ⑩-c 格：在冻结口径下 M1–M6 的统一海选结论 —— 除 M4（代数冗余）、M5（不可实现）外，其余 4 个候选都只能单侧跑出 frontier，共同弱侧是 cable。
- 强化已有结论：**「改 representation 只能选择工作点」在 M1/M2/M3 三种不同机制上再次成立**；唯一例外（M1 bottle 大幅越界）被 cable 反向结果抵消。
- 为方法章节留下一个可复用的**统一筛选协议**（同一 runner 注入、同一指标重算、同一 frontier 参照、机械判定）。

## 14. 下一步（仅建议，不自行启动）

**仅推荐 M1 进入 Round-2 候选验证（带条件）**：在 **5 类别 × ≥2 seed** 上检验 M1 的 bottle 收益能否复现、cable 恶化是否系统性；并把「类别统计同质性」作为预注册调节变量检验 H1/H2。M2/M3 已被本轮与历史（7A-O）双重判定为沿 frontier 移动，建议不再投入；M6 仅作 oracle 参照；M4 永久 STOP；M5 仅在 Round-2 仍无更好候选时再考虑实现公平 selection rule。

Round-2 若要做，必须先由用户批准新协议（类别 / seed / 判据 / 是否纳入统计同质性调节分析）。

## 附录 A：P0 审计（只读，执行前）

| 项 | 结果 |
|---|---|
| branch / HEAD | `main` / `5384f3a` |
| worktree | 非 clean：6 个 untracked（上一轮会话留下的 `experiment9a_*`、`monitor_progress.py`、`experiment_progress.py`、`monitor/`）；无 tracked 文件被改动 |
| GPU | RTX 3090 24GB；执行前有 2 个 python 进程 —— 经核对为**本任务 round0 runner（PID 408978/409047）**，非未知任务 |
| CPU / RAM | 14 cores / 755 GB（available 711 GB） |
| 磁盘 | `/root/autodl-tmp` 50G（已用 5.5G，可用 45G）；overlay 可用 28G |
| MVTec AD | 5 类齐全（bottle/cable/grid/hazelnut/screw） |
| 历史资产 | 7A-O（M2 `B1_g100` 15/15）、8B（M6 `PROPOSED25/50`）、5A-H + 6A/6B + 7A-O Q4（baseline 与 uniform-α frontier 全覆盖）**均在位** |
| runner 可复用 | 是（`experiment5a_h_runner.run_config` 注入式复用，与 7A-O/8B 一致） |

## 附录 B：目录与产物

```
results/experiment_9a_screening/
  README.md   config.json   progress.log
  raw/<round>/<cat>/seed_0/config_<name>/{per_image.csv, info.json, module_meta.json}   # 主证据，入库
  summary/{raw_results.csv, screening_summary.csv, screening_summary.json,
           uniform_alpha_frontier.csv, runtime_summary.json, m4_equivalence.json, sanity_checks.csv}
  figures/fig1_robustness_preservation_plane.png
  cache/  logs/  progress/                                                              # git-ignored
```

复现：`python -u scripts/experiment9a_analysis.py`（CPU-only，读上述 raw + 历史 frozen raw，0 GPU）。
