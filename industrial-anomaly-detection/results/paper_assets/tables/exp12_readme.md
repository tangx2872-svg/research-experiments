# Experiment 12 — Category × Channel Adaptive Compensation

**日期**：2026-10-08 ｜ **环境**：AutoDL RTX 3090 24GB / conda `base` / PyTorch 2.8.0+cu128 / Anomalib 2.6.2
**起始 HEAD**：`ba9653d`（Exp11 编排器修复）｜ **冻结物**：`config/candidate_registry.json`（在看任何 Exp12 结果前写定）

> **状态：PROTOCOL + CANDIDATES FROZEN。** 用户既有文件（`monitor/exp_monitor.py`、`monitor/events.jsonl`、
> `scripts/_lockprobe.txt`、`scripts/_probe2.py`）不动。

## 1. Motivation（为什么做这一轮）

Exp11 得到 **Adaptive C2 / `E11_B2`**（category-level β 校准）：PASS **12/15**、cable **0/3 → 2/3**、
grid 2/3 → 3/3，但 **hazelnut 1/3 未改善**、**仍有 1 个 catastrophic**（cable seed2 Δd′ ≈ −0.285）→ HOLD。
Exp11 同时发现：**`E11_C2`（channel-level gate）在 cable 上把 catastrophic 消除**（preservation +0.163），
但牺牲了 cable 的 robustness；而 **B2 在 cable/hazelnut 上改善 preservation**。两者机制尺度不同、**正交**。

## 2. Hypothesis

> **Category-level 决定"补多少"（β_c），Channel-level 决定"补哪些通道"（g_k）；**
> 组合（Category × Channel）应比单用 category-level（B2）**更稳定地保护 defect information**，
> 同时保持 illumination robustness。

**不是**围绕 cable seed2 / hazelnut 调参，**不是**参数搜索。

## 3. Frozen protocol

| 项 | 值 |
|---|---|
| train | **corrected-189**（Exp10 冻结实现，`images_embedded=189` 已证） |
| RNG/coreset | **strict-V2**（`experiment9c_rng`：R1 `Engine.fit` 入口 + R2 `select_coreset_idxs` 入口 replay） |
| 判据带 | `EPS_RZ=0.02`、`EPS_DP=0.10`；catastrophic = `Δd′ ≤ −0.25`（**沿用 Exp11 冻结定义，不修改**） |
| categories / seeds | bottle/cable/hazelnut/screw/grid；seeds 0/1/2 |
| 参考点 | 全部复用：corrected Original-189、Fixed C2、Adaptive B2（Exp10/Exp11 已有，**0 GPU**） |
| 并发 | `--workers=N` = **全局 GPU 并发上限**（Exp10 编排器已修：`per_seed = max(1, N // n_seeds)`；Exp11 曾出现 8 进程 OOM，本轮沿用修复版并验证） |

## 4. Candidate registry（冻结；M4 按规则跳过）

统一结构：`L2 compensation = β_c × g_k × IN(F_k) × rms(F)`，**L3 恒为 `alpha_in(0.25)`**。
**零新模型代码**：复用 Exp11 的 `adaptivePatchcoreModel.residual_gated`。

| ID | 定义 | gate 均值（bottle/cable/hazelnut/screw/grid） | β_c（B2 冻结值） |
|---|---|---|---|
| **M0** | `g ≡ 1`（category-only，= Exp11 B2） | 1.000 ×5 | 0.250 / 0.2948 / 0.2279 / 0.2164 / 0.4601 |
| **M1** | `g = 0.75 + 0.25·g_C2` | 0.857 / 0.795 / 0.848 / 0.866 / 0.921 | 同上 |
| **M2** | `g = 0.50 + 0.50·g_C2` | 0.714 / 0.590 / 0.695 / 0.732 / 0.842 | 同上 |
| **M3** | `g = g_C2` | 0.427 / 0.179 / 0.390 / 0.464 / 0.683 | 同上 |
| ~~M4~~ | **按规则跳过** | — | — |
| G1 | `g ≡ 1`（sanity 专用，须逐位 == M0） | 1.000 | 同上 |

其中 `g_C2 = 1 − clip(RATIO_j / p90_j, 0, 1)`（Exp11 E11_C2 原始定义，`RATIO_j = ‖IN(F)_j − F_j‖ / ‖F_j‖`，**train/good only**）。

**为什么跳过 M4**：任务书规定「如果 Exp11 C2 本身已经是 soft continuous gate，则不要为凑 M4 发明新公式」。
实测 `g_C2` 是**连续**映射（90% 通道 g>0，仅 **10.2%** 恰好为 0，std 0.13–0.26）→ **只测 M0–M3**，
不为 M4 发明新公式（记录于 `config/candidate_registry.json` 的 `m4_skipped_by_rule`）。

## 5. Strict Replay Sanity（正式运行前）

- **Sanity A**：M0 与 Exp11 B2 **严格等价** —— M0 直接复用 Exp11 的 15 个 B2 units（同一 spec 语义），等价性由既有证据保证。
- **Sanity B**：新代码路径在 `g ≡ 1` 时必须**逐位退化为 B2** → 跑 `E12_G1__bottle`（1 unit）与 Exp11 `E11_B2__bottle` 比对：
  embedding / coreset / bank sha256、tau、score、`max|Δscore|`。**要求 `max|Δscore| = 0`。**
- **Sanity 不通过 → 禁止进入正式实验。**

## 6. 计划与预算

| Stage | 内容 | units（上限） |
|---|---|---|
| Sanity | `E12_G1__bottle` | 1 |
| Round 1 | M1–M3 × bottle/cable/hazelnut × seed0 | 9 |
| Round 2 | Top1（若有）补 screw/grid × seed0 | ≤6（每候选 2） |
| Round 3 | Top1 × 5 cats × seeds {1,2} | ≤10 |
| Ablation | **仅当 ≥13/15 且 0 catastrophic** 才追加 | ≤8（优先复用） |

**Round-1 排名规则（写死于 analysis 代码，见结果节）**：PASS 类数 → catastrophic 数 → cable Δd′ 改善 →
hazelnut Δd′ 改善 → worst Δd′ → ΔR → 简洁度；硬约束：bottle 不得 PASS→FAIL、不得新增 catastrophic、
robustness 不得明显恶化、cable/hazelnut 不得都无改善（否则立即淘汰）。

**晋级**：最多 Top2，需满足 ADVANCE-A（cable 相对 B2 明显改善且 robustness 不退化）或
ADVANCE-B（hazelnut 向 PASS 移动且 cable 不退化）或 ADVANCE-C（cable+hazelnut 均小幅改善且 bottle 保持）。

**时间预算**：预计总墙钟 **45–75 min**，软预算 **90 min**（超出则停 optional，先完成分析与 README）。

---

# 结果

## 7. Strict Replay Sanity（正式运行前，全部 PASS）

**Sanity B**（`g ≡ 1` 时新代码路径必须逐位退化为 B2）：跑 `bottle:0:E12_G1__bottle` 与 Exp11 `E11_B2__bottle` 比对：

| 项 | 结果 |
|---|---|
| embedding sha256 | `5efd325ab11f7e82` == `5efd325ab11f7e82` ✓ |
| coreset indices sha256 | `6677f4577add11bf` == `6677f4577add11bf` ✓ |
| memory bank sha256 | `4d3575e88d2ffcbe` == `4d3575e88d2ffcbe` ✓ |
| tau | 42.99169922 == 42.99169922 ✓ |
| **`max\|Δscore\|`** | **0.000e+00**（n=183）✓ |
| train filter | `images_embedded = 189` ✓ |

**Sanity A**：M0 = Exp11 B2（同 spec 语义，直接复用 15 units）✓

**额外跨实验等价性检查**：`E12_M3__bottle`（β=0.25, g=g_C2）与 Exp11 `E11_C2__bottle`（同 spec）
**逐位一致**（embedding/bank sha 相同、`max|Δscore| = 0.0`）✓ —— 同时验证了 spec 构造与跨实验一致性。

## 8. Round 1 — 硬类快筛（bottle/cable/hazelnut × seed0，9 units，0 failed）

`M0 (B2)` 参考：bottle Δd′ = −0.0653、cable = −0.0298、hazelnut = +0.0333。

| cand | PASS | cata | cable Δd′ vs B2 | hazelnut Δd′ vs B2 | worst Δd′ | mean ΔR | ADVANCE | **淘汰原因** |
|---|---|---|---|---|---|---|---|---|
| `E12_M1`（g=0.75+0.25g_C2） | 1 | 0 | **+0.0663** | **+0.1261** | +0.0365 | −0.0599 | ✗ | **robustness 明显退化**（cable ΔR = **+0.029** → 失去 robustness gain；hazelnut ΔR = −0.005 临界） |
| `E12_M2`（g=0.50+0.50g_C2） | 0 | 0 | +0.0018 | −0.0810 | −0.1871 | −0.0590 | ✗ | **bottle PASS→FAIL**（Δd′ = −0.187）+ 两项无改善 |
| `E12_M3`（g=g_C2） | 1 | **1** | **−0.3140** | +0.1487 | −0.3438 | −0.0590 | ✗ | **新增 catastrophic**（cable Δd′ = **−0.344**） |

**逐类明细（ΔR/ΔP）**

| cand | bottle | cable | hazelnut |
|---|---|---|---|
| B2 (M0) | −0.211 / −0.065 | −0.043 / −0.030 | −0.038 / +0.033 |
| M1 | −0.204 / **+0.086** | +0.029 / +0.036 | −0.005 / **+0.160** |
| M2 | −0.187 / −0.187 | +0.018 / −0.028 | −0.009 / −0.048 |
| M3 | −0.154 / −0.082 | −0.009 / **−0.344** | −0.015 / +0.182 |

**判定：三个候选全部触发硬性淘汰 → ADVANCE-A/B/C 均不成立 → 0 ADVANCE → 按 §11 规则 STOP。**

> `Category × Channel combination did not outperform Adaptive B2 under frozen protocol.`

## 9. Round 2 / Round 3 / Ablation — **未触发**

- Round 2/3 需要 Round 1 有 ≥1 ADVANCE → **未触发**（0 ADVANCE）。
- Ablation Preview 需要最终 ≥13/15 且 0 catastrophic → **未触发**。
- **未新增任何参数、未围绕 cable seed2 / hazelnut 调参**（这是本轮最重要的纪律）。

## 10. C2 家族横截面（seed0，全部复用已有 raw，0 GPU）

| config | bottle ΔR/ΔP | cable ΔR/ΔP | hazelnut ΔR/ΔP |
|---|---|---|---|
| Original | 0.000 / 0.000 | 0.000 / 0.000 | 0.000 / 0.000 |
| T1（L3-only） | −0.098 / −0.140 | n/a（未跑） | n/a |
| Fixed C2（β=0.25, g=1） | −0.211 / −0.065 | +0.025 / −0.070 | −0.021 / +0.021 |
| **Adaptive B2（M0）** | −0.211 / −0.065 | −0.043 / −0.030 | −0.038 / +0.033 |
| E11_C2（β=0.25, g_C2） | −0.154 / −0.082 | +0.063 / **+0.163** | −0.054 / +0.014 |
| Exp12 M1 | −0.204 / +0.086 | +0.029 / +0.036 | −0.005 / +0.160 |
| Exp12 M2 | −0.187 / −0.187 | +0.018 / −0.028 | −0.009 / −0.048 |
| Exp12 M3 | −0.154 / −0.082 | −0.009 / **−0.344** | −0.015 / +0.182 |

**机制读数（可解释）**
- **gate 越"选择性"（g 均值越小）→ preservation 在本轮并不单调改善**：M1（ḡ≈0.86）在三类上 preservation 都最好，
  M2（ḡ≈0.66）在 bottle/hazelnut 变差，M3（ḡ≈0.43）在 cable 崩溃。
  → **"补哪些通道"这一机制在本协议下没有产生稳定的增益**，其效果被**补偿强度**（β_c × g 的有效值）主导。
- **M1 的方向与预期相反**：更 mild 的 gate 让 preservation 上升、robustness 下降 ——
  说明中间层补偿与 robustness 之间仍是**同一 frontier 上的兑换**，而不是正交增益。

## 11. 稳定性诊断（重要 caution）

`cable seed0` 的 Δd′ 在 C2 家族**很窄的参数区间内跨度达 0.507**
（Fixed C2 −0.070 / B2 −0.030 / M2 −0.028 / M1 +0.036 / E11_C2 **+0.163** / M3 **−0.344**）。
→ **cable 的 preservation 对该 family 的 (β, g) 高度敏感**；因此
**Exp11 中 cable 0/3→2/3 的改善不宜被单独当作机制性结论**，只能作为观察事实（已如实记录）。
这也是本轮**不允许**用 cable seed2 反向调参的直接理由之一。

## 12. Bugs / OOM / reruns

- 本次运行**无 OOM、无 failed unit**（含 sanity 共 10 units 全部 OK）。
- 编排器沿用 Exp10 修复版（`per_seed = max(1, workers // n_seeds)`）→ **实测全局 GPU 进程数 = 4 = `--workers` 上限** ✓
  （smoke 阶段 VRAM 20640 MiB < 22GB → 保持 4 workers，未升 5）。

## 13. Verdict 与论文含义

| 判据 | 结果 |
|---|---|
| FINAL-METHOD-CANDIDATE（≥13/15 + 0 catastrophic + cable≥2/3 + hazelnut≥2/3 + bottle 3/3 + screw 3/3） | **未达到**（Round 1 即 0 ADVANCE，未进入正式 15-unit 对照） |
| HOLD / STOP | **STOP 该组合路线**（按 §11 与 §14 的 STOP 定义） |

**Verdict：C — Category × Channel combination 在冻结协议下未带来净收益 → 停止该组合。**

**论文含义**
1. **Category-level calibration（B2）仍是目前最好的单一机制**：它把 PASS 从 9/15 提到 12/15，且 bottle/screw 零退化。
2. **Channel-level gating 与 category-level 不是正交增益**：两者叠加在本轮全部触发硬性淘汰
   （robustness 退化 / bottle 失守 / 新增 catastrophic），因此**不支持"二者互补"的假设**。
3. 由此得到一条**方法学结论**：在 PatchCore + 本文协议下，**中间层补偿的"强度"与"位置"无法被独立优化**——
   改变位置（channel 选择）等价于改变有效强度，而有效强度的任何移动都会在同一 robustness–preservation
   兑换曲线上滑动（并伴随 cable 的高敏感性）。

**下一步（仅建议，未启动）**
- **不再围绕本 family 调参**（含 cable seed2 / hazelnut）。
- 值得做的只有两件事：①把 **Adaptive B2（12/15）** 作为当前最佳可部署候选**提交人工决策**是否接受；
  ②若继续方法侧，应换**另一条有文献依据的成熟模块路线**（而非继续 C2 家族的变体）。
