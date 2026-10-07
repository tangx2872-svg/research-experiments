# Experiment 9C — Matched-RNG Calibration

**日期**：2026-10-07 ｜ **环境**：AutoDL RTX 3090 24GB / conda `base` / Python 3.12.3 / PyTorch 2.8.0+cu128 / Anomalib 2.6.2
**Git HEAD（冻结时）**：`3595c8d` (main)

> **状态：PROTOCOL FROZEN（在读取任何 9C target result 之前冻结）** —— 见 `config.json`。
> 本实验**不是新方法实验**：不做 M12、不做 5-category validation。唯一目标是建立 **Matched-RNG Protocol V2**。

---

## 1. 为什么需要 9C

9B 在方法海选过程中发现一个**评测方法学问题**：两个生成**完全相同 embedding** 的方法，会得到不同的 score / robustness / d′。
决定性已有证据（全部来自 9B，均为实测）：

| # | 证据 | 数值 |
|---|---|---|
| 1 | equivalent gate 与 uniform α 的 embedding 差异 | `max\|ΔF\| = 0.0`（GPU 实测） |
| 2 | 同一配置重复运行 | score **bit-identical**（max\|Δscore\| = 0.0） |
| 3 | equivalent gate vs uniform α（同 runner） | fit 后 `max\|Δscore\| ≈ 1.330` |
| 4 | 两条代码路径进入 fit 前的 torch RNG state | `torch.rand` 0.9509 vs 0.5424、`randint` 78637 vs 3666 → **不一致** |
| 5 | 归因 | RNG state divergence → KCenterGreedy 初始点/轨迹分歧 → **不同 coreset** → 下游指标噪声 |

已测 **V1 噪声地板**：`NF(\|Δz\|) = 0.020224`、`NF(d′) = 0.054211`（来源 `../experiment_9b_screening/summary/smoke_equivalence.json`）。
9B 因此必须把「frontier 越界」类结论的判读阈值放在该地板上，这是不可接受的长期状态 → 9C 负责**消除**它。

## 2. 假设（冻结）

- **H1**：representation 完全一致时，不同代码路径的 score 差异主要来自 fit 前 RNG state 不一致。
- **H2**：若在进入 coreset selection 前显式恢复完全相同的 RNG state，则 equivalent representations 应产生**相同 coreset**，score 达到 bit-exact / machine-precision 等价。
- **H3**：Matched-RNG 不改变 representation 本身。

**Primary success criteria（冻结，不允许事后放宽）**：A `max\|ΔF\| ≤ 1e-7`（理想 0.0）；B coreset indices **exact equality**；C `max\|Δscore\| ≤ 1e-7`（理想 0.0）；D `|Δ mean\|Δz\|| ≤ 1e-6`；E `|Δ mean d′| ≤ 1e-6`。

## 3. RNG 调用链（P0/P2 审计结果，实测）

```
run_config → fit_dual_model（入口 seed 一次）→ Engine.fit
  → Patchcore(Lightning).fit → MemoryBankMixin.fit → PatchcoreModel.subsample_embedding
    → KCenterGreedy(embedding=memory_bank).sample_coreset() → select_coreset_idxs()
        ├─ SparseRandomProjection.fit → _sparse_random_matrix
        │     ├─ torch.distributions.Binomial(...).sample()        → torch CPU RNG
        │     └─ sklearn sample_without_replacement(random_state=None) → **NumPy global RNG**
        └─ torch.randint(high=n, device=self.features.device)      → **CUDA RNG**（embedding 在 GPU 上）
```

**实际消耗的 RNG 流（CPU 实测）**：`torch(CPU) = YES`、`numpy = YES`、`python random = no`；GPU 上 `torch.randint(device=cuda)` 额外消耗 **CUDA RNG**。
（审计脚本证据：`summary/rng_audit.json`；`select_coreset_idxs` 前后 torch/numpy state hash 变化，python random 不变。）

## 4. Replay 点（冻结）

**位置**：`KCenterGreedy.select_coreset_idxs` 入口（wrapper 注入，**不改 anomalib 源码**）。

**为什么选这里**：
1. 它是「**最晚但仍能保证 coreset selection 一致**」的点 —— 所有 coreset RNG 消耗（投影矩阵生成 + 初始点采样）都发生在它内部；
2. 早于它的所有上游差异（方法类构造次数、timm 权重加载等）被完全屏蔽；
3. 晚于它的阶段（distance update、`get_new_idx` 的 `torch.max` 归约、`sample_coreset` 的索引取值）**不再消耗 RNG**，因此不影响结论。

**控制哪些 RNG**：torch CPU、torch CUDA、NumPy global，外加 python `random`（审计确认 coreset 未消耗，仍纳入快照以求完备）。
**不触碰**：模型数学定义、feature 值、train split、coreset ratio(0.1)、kNN(9)、illumination shifts、test label / defect mask。
**为什么无 test information 泄漏**：canonical state 只由 `(category, seed)` 常量决定，不读取任何 test/defect 数据。

## 5. V1 与 V2 的区别

| | V1（历史 / 冻结） | V2（本实验建立） |
|---|---|---|
| seed 位置 | `fit_dual_model` 入口 seed 一次，**不重播** | 同上 **+** coreset 入口恢复 canonical state |
| coreset 依赖 | 方法构造路径消耗 RNG 的数量 | 只依赖 `(category, seed)` |
| equivalent-pair score | 可差 O(1)（max 1.330） | 目标：**bit-exact / ≤1e-7** |
| 历史可比性 | 与 5A-H/6A/7A-O/9A/9B **逐位可比** | 与历史 raw **不可逐位比较**（coreset 不同，属预期） |

## 6. 校准单元（冻结）

`C0 = Original α=0`（runner sanity）、`C1 = Uniform α=0.25`、`C2 = Constant gate 0.25`（与 C1 representation 等价）；核心比较 **C1 vs C2**。
V1 现象**直接复用 9B raw**（`SMOKE_uniform_a025` / `SMOKE_gate_const025`），**不重跑 V1**。

## 7. 预算（冻结）

GPU 主单元 3（+诊断上限 2）；预计 GPU 墙钟 5–10 min、总时长 10–20 min。
**HARD STOP**：总墙钟 > 30 min 或 GPU units > 5 → 停止扩展并报告卡点。

---

# 结果（P0–P7 完成，2026-10-07）

**GPU：5 unit-runs / 483 GPU·s（`v2`=3 units 97s 均值；`v2b`=2 units 58s 均值）、failed = 0、峰值显存 2605 MB。**
**P4 CPU sanity：7/7 PASS。** V1 对照复用 9B frozen raw（0 GPU）。

## 7. 三方对照（bottle seed0；uniform α=0.25 vs constant-gate 0.25）

| 协议 | RNG state 一致 | coreset 一致 | score bit-exact | NF(\|Δz\|) | NF(d′) | CASE |
|---|---|---|---|---|---|---|
| **V1**（无 replay；9B raw 复用） | ✗（`torch.rand` 0.9509 vs 0.5424） | ✗（tau 19.01377 vs 19.00691） | ✗（max\|Δscore\| 1.330） | **0.020224** | **0.054211** | **B** |
| **V2.0**（仅 coreset 入口 replay） | ✓（哈希全同） | ✗（首个索引相同 51413，其后分叉） | ✗ | 0.027988 | 0.005269 | **B**（诊断出第二个 replay 点） |
| **V2.1**（coreset + Engine.fit 双 replay） | ✓ | **✓（sha `a894c7b0c22302b1`）** | **✓（max\|Δscore\| = 0.0）** | **0.000000** | **0.000000** | **A** |

**V2.1 的逐位一致证据链（全部实测，`summary/*.json`）**

| 环节 | 证据 |
|---|---|
| representation | embedding `sum_abs` = 290696768.0（两侧完全相等）；GPU 端 `max\|ΔF\| = 0.0`（同进程 16 图 batch 验证） |
| pre-coreset RNG state | `torch_cpu 1ccf17250133dec5 / numpy 012a3189dd9a9348 / python 46f264538534643a / cuda 374708fff7719dd5` **两侧相同** |
| coreset | indices sha `a894c7b0c22302b1` **完全相同**（n=21401，first20 相同） |
| memory bank | sha `cfe4ecff39d6a652` **完全相同** |
| tau | 18.88035774 **完全相同** |
| per-image score | n=183，**max\|Δscore\| = 0.000e+00**（bit-exact） |
| 指标 | \|Δ mean\|Δz\|\| = 0.0、\|Δ mean d′\| = 0.0、image/pixel AUROC、AUPRO 全部 \|Δ\| = 0 |

## 8. CASE 判定（按冻结标准）

**CASE A —— Matched-RNG 成功。**

> **“9B observed noise floor was caused by uncontrolled coreset RNG trajectory divergence.”**

更精确地说，噪声地板由**两个**未被控制的 RNG 消费者共同造成（9C 首次把两者分离：
V2.0 只固定了第二个，V2.1 才固定第一个）：

1. **R2 — KCenterGreedy 采样 RNG**（`select_coreset_idxs`：`SparseRandomProjection` 的 torch Binomial + sklearn `sample_without_replacement`(numpy) + `torch.randint`）
   → 决定 greedy **初始点**。V1 中两条路径初始点不同；V2.0 固定后，两侧初始索引都是 51413（已证）。
2. **R1 — train DataLoader 的 shuffle RNG**（`DataLoader(shuffle=True)`，无显式 generator，行序由全局 RNG 决定）
   → 决定 **embedding 行序**。V1/V2.0 中两条路径的行序不同 → 行置换（aggregate `sum_abs` 差 32/290M ≈ 1.1e-7，k-center-greedy 对行序高度敏感）→ coreset 轨迹不同。
   V2.1 在 `Engine.fit` 入口 replay 后，行序被固定，coreset/memory bank/score **逐位一致** → 证明这是主因（**CASE B 的诊断结论**）。

**H1**：支持（V1 中 RNG state 确实不同；固定后差异消失）。
**H2**：支持，但需要**两个** replay 点（不是 coreset 前一个点就够）。
**H3**：支持（Matched-RNG 只改变 RNG 时机，不改 representation 数学；V2.1 中两侧 embedding 逐位一致）。

## 9. Noise floor Re-estimation（P7）

| | V1 | V2.1 |
|---|---|---|
| NF_V2(\|Δz\|) | 0.020224 | **0.000000** |
| NF_V2(d′) | 0.054211 | **0.000000** |

达到 numerical-zero（bit-exact）。**未美化、未放宽任何阈值**：A–E 五条冻结标准全部满足（C 为 bit-exact 0.0，优于 ≤1e-7 的要求）。

## 10. Protocol V2 的冻结定义

- **V2 = 在两个点恢复 canonical RNG state**：
  - **R1**：`anomalib.engine.Engine.fit` 入口（wrapper 注入）→ 固定 train DataLoader shuffle 行序
  - **R2**：`KCenterGreedy.select_coreset_idxs` 入口（wrapper 注入）→ 固定投影矩阵与 greedy 初始点
- canonical state 由 `(category, seed)` 唯一确定（seed 四条流后捕获），**不含任何 test/defect 信息**。
- 不改 anomalib 源码、不改 9B/5A-H 代码（9C 用薄驱动 + 运行时注入）。
- **V2 下的结果与历史 V1 raw 不可逐位比较**（coreset 不同）—— 这是采纳 V2 的既定代价。

## 11. 对历史结论的影响（P13）

- **不需要因为 V2 的建立而宣布 9A/9B 无效**。
- **远大于 V1 噪声地板的结果 = 仍然有效的强信号**（例如 9B 的 M8 bottle 越界 +0.690 / +0.937，远超 NF(d′)=0.0542；M11_b100 +0.804）。判读时还应乘以 V1 噪声的经验放大（同一模型族内多次重选 coreset 的离散度 ≈ NF 量级）。
- **与噪声地板同量级的边际结果必须标记为 `requires V2 re-evaluation before promotion`**：
  - 9B `M7_L2a000_L3a025`：bottle 越界 +0.047、cable +0.019（均 < 0.0542）
  - 9B M9_sw_gamma1：cable 越界 +0.013
- **本轮不重跑它们**（按 P13 要求）。
- **对 9A M4 归因的最终确认**：9A 曾把 M4「组合形式 vs 直接 α_eff」的 score 差异归因于 fit 路径放大 1e-6 级表示扰动；9B 提出真因是 RNG 状态分歧；**9C 给出最终结论**：score 差异来自**未被控制的 coreset 轨迹分歧**（R2 + R1 两个消费者），与「表示扰动放大」无关；9A 的 M4 结论（代数冗余）由 embedding 级等价确立，**不受影响、无需重跑**。

## 12. 后续实验必须采用的 protocol

> **From Experiment 9C onward, new method comparisons use Matched-RNG Protocol V2 for fair coreset selection. Historical Protocol V1 assets remain frozen and are not overwritten.**

- 启用方式（不改历史代码）：
  ```python
  import experiment9c_rng as r9c
  r9c.install_matched_rng(seed=<unit seed>, out_dir=<unit dir>, protocol="v2")
  r9c.install_fit_replay()
  ```
- V2 参考工作点（本轮已建立，可直接复用，无需重跑）：`bottle seed0 / α=0 (C0)`、`bottle seed0 / α=0.25 (C1)`。完整 V2 frontier 留到下一轮真正方法筛选时按需最小补充。
- **V1 的历史 raw / CSV / JSON 全部冻结**，不被 V2 覆盖，仍作为历史证据（与 V1 baseline 的比较必须同时报告 V1 噪声地板）。
