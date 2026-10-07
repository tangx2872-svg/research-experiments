# Experiment 9B-R — Strict Replay Re-evaluation

**日期**：2026-10-07 ｜ **环境**：AutoDL RTX 3090 24GB / conda `base` / PyTorch 2.8.0+cu128 / Anomalib 2.6.2
**Git HEAD（冻结时）**：`608a81b` (main) —— 即 Experiment 9C 的协议建立 commit

> **状态：PROTOCOL FROZEN（在读取本轮 target result 之前冻结）。** 冻结内容见 `config/config.json`。

---

## 1. Scientific Question（唯一问题）

> 在严格控制 **(1) representation、(2) RNG state、(3) coreset selection** 之后，9B 观察到的
> robustness–preservation improvement **是否仍然存在**？

不是继续研究噪声：9C 已结束。本实验是**用已经校准好的尺子重新量 9B**。
只回答：候选优势是**真实 method effect**，还是**measurement artifact**。

## 2. 为什么需要重审 9B

9B（M7–M11）**无候选晋级**，但其判读受一个当时无法消除的限制：所有「frontier 越界」幅度都与
**V1 coreset 噪声地板**（`NF(|Δz|)=0.0202`、`NF(d′)=0.0542`）同量级或接近，因此 9B 的 HOLD 里
混杂了**不可判读**的成分。9C 随后证明该噪声来自**未受控的 coreset RNG 轨迹分歧**（两个消费者：
R2 = KCenterGreedy 初始点；R1 = train DataLoader shuffle 行序），并建立 **Matched-RNG Protocol V2**：
equivalent pair 的 embedding / RNG state / coreset indices / memory bank / tau / per-image score
**全部逐位一致**（`max|Δscore| = 0.0`，NF 0.020224/0.054211 → **0.000000/0.000000**）。

9B 的 HOLD 结论因此**必须**用 V2 重审：噪声地板归零后，越界幅度才真正可判读。

## 3. 本条协议（冻结，复用而非重写）

**Protocol = Matched-RNG Protocol V2，逐字复用 9C-v2b 已验证实现**：

| 项 | 内容 |
|---|---|
| R1 | `anomalib.engine.Engine.fit` 入口恢复 canonical RNG state（固定 train shuffle 行序） |
| R2 | `KCenterGreedy.select_coreset_idxs` 入口恢复 canonical RNG state（固定投影矩阵 + greedy 初始点） |
| canonical state | 仅由 `(category, seed)` 确定，**不含 test/defect 信息** |
| 实现来源 | `scripts/experiment9c_rng.py`（`install_matched_rng` / `install_fit_replay`）+ 9C 的薄驱动模式 |
| 禁则 | **禁止另写一套「看起来差不多」的 strict replay**；不修改 anomalib / 9B / 7A-O / 5A-H 源码 |
| 不可比性 | V2 结果与 V1 frozen raw **不可逐位比较**（coreset 不同，属既定代价） |

## 4. 指标与判据（冻结，沿用 9B）

- **Primary robustness** = `mean |ΔNormalScore_z|`（越低越好）
- **Primary preservation** = `mean defect d′`（越高越好）
- 指标由 `experiment5a_h_analysis.unit_metrics` 重算（与 5A-H/7A-O/9A/9B/9C 同一实现）
- 判据带：`EPS_DP = 0.10`、`EPS_RZ = 0.02`（5A-H / 7A-O Tier-A 冻结）
- **Strict replay 下的参考点** = 同协议下的 `Original (α=0)`（V2），而不是 V1 frontier
  （理由：V1 frontier 的 coreset 由 V1 协议产生，跨协议比较会重新引入不可比性；V2 frontier 留待下一轮）
- **ADVANCE / HOLD / STOP 规则**：见 `config/config.json` 的 `verdict_rules_frozen`；
  **不允许在看到结果后修改阈值**。

## 5. 候选选择（P2，机械来自 9B 冻结 ranking）

读取 `../experiment_9b_screening/summary/ranking.csv`（9B 冻结 ranking），机械取 **rank #1–#3**，
外加必须的 **Original baseline** 与 **一个理论等价 control**（Round 1 共 5 个模型，其中 control 复用 9C，0 GPU）。

| ID | Method | 9B evidence | Why retained | Duplicate? |
|---|---|---|---|---|
| `REF_original` | Original PatchCore (α=0) | 9B ranking BASELINE `B0` | 必需 reference（兼作 Control 1） | – |
| `T1_M7_L2a000_L3a025` | M7 per-layer independent α-IN (L2=0.00, L3=0.25) | **rank #1**, HOLD；bottle 8.0216/0.3355、cable 5.2912/0.3154；beyond bottle +0.0467、cable +0.0189（**≤ NF 0.0542**） | rank #1（preservation 损失最小） | no |
| `T2_M10_concat_g100` | M10 fixed concat `[F, γ·ÎN(F)]`, γ=1.00 | **rank #2**, HOLD(*REUSE*)；bottle 8.1708/0.3379、cable 5.1008/0.3275；beyond bottle **+0.1911**、cable −0.1716 | rank #2（bottle 越界幅度最大） | **YES ≡ 7A-O Family B `B1_g100`** |
| `T3_M7_L2a025_L3a000` | M7 per-layer independent α-IN (L2=0.25, L3=0.00) | **rank #3**, HOLD；bottle 7.8876/0.3756、cable 5.3635/0.3594 | rank #3 | no |
| `EQ_control_a025` | uniform α=0.25 vs constant-gate 0.25 | 9B `smoke_equivalence`（V1 下 max\|Δscore\|=1.330，RNG 分歧） | 理论等价 control | – |

**Equivalence mapping（去重，禁止重复运行）**
- `M10_concat_g100` **≡ 7A-O Family B `B1_g100`**（9B 已标注为 REUSE）。本轮以 7A-O 的冻结实现
  （`experiment7ao_model.py` / `kind=concat_dual, gamma=1.0`）**只跑一次**，不重写 concat，不重复运行。
- `EQ_control` 的两个成员（uniform α=0.25 / const-gate 0.25）**已在 9C-v2b 以 V2 跑过** →
  本轮**复用其 raw（0 GPU）**，只补 9C 当时缺失的**逐元素 `max|ΔF|`** 检查。
- 9B 其余 rank #4–#9（M7 其他配置、M8/M9/M11）本轮**不跑**（Top-3 规则）。

## 6. Round 0 Sanity（必须先 PASS）

- **Control 1**：`Original α=0` 在 V2 下**两次独立进程**运行 → 必须
  representation bit-identical、coreset indices identical、memory bank SHA256 identical、
  `max|Δscore| ≤ 1e-7`（理想全 0）。该 residual 即本轮 strict replay 的**尺子**（NF_strict）。
- **Control 2**：理论等价 pair（uniform α=0.25 vs const-gate 0.25）→ 必须
  **逐元素 `max|ΔF|`**（不能用 sum/mean 代替）、coreset indices、bank SHA256、`max|Δscore|`、
  robustness、preservation 全部一致。
- **Sanity FAIL ⇒ 立即 STOP，不做候选比较。**

## 7. 运行计划与时间估算（基于历史实测秒/unit）

历史 median：bottle 88–99s、cable 162s、hazelnut 220s、screw 178s、grid 106s（5A-H/9B/9C）；
M10 concat 约 1.5–2×（embedding dim 翻倍）。

| 阶段 | 单元数 | 预计 |
|---|---|---|
| Round 0 Sanity（Control 1 ×2 + 逐元素等价检查） | 2 GPU units + 1 检查 | **3–5 min** |
| Round 1（bottle seed0：Original(复用 Control1) + T1 + T2 + T3） | 3 GPU units（并行 3 workers） | **3–5 min** |
| Round 2（仅当 ≥1 ADVANCE；Top 1–2 × 5 cats × seed0） | ≤ 8 GPU units | **6–9 min** |
| **Maximum total（含分析/README/git）** | ≤ 13 GPU units | **18–25 min** |

---

# 结果（Round 0 + Round 1 完成，2026-10-07）

**GPU：5 unit-runs / 538.9 GPU·s（bottle seed0），failed = 0，峰值显存 ~13.9 GB（5 workers 并行）。**
**Wall-clock：P0 审计 + 实现 ≈ 25 min；GPU 运行 4:49；分析 + README ≈ 10 min。**
**Round 2 未触发**（Round 1 无 ADVANCE，按 P7 规则不进入）。

## 8. Round 0 — Sanity（10/10 PASS）

**Control 1：Original α=0 在 V2 下两次独立进程（w0 / w4，并行 GPU 负载下）**

| 检查 | 结果 |
|---|---|
| representation bit-identical | ✓ `embedding sha256 = 6594060903d165aa`（两侧相同） |
| coreset indices identical | ✓ `coreset sha256 = 0c7270fb9cccfd68` |
| memory bank SHA256 identical | ✓ `4f3b8962df8dfbed` |
| `max|Δscore|` | **0.000e+00**（n=183，bit-exact） |
| tau | 20.3010158539（两侧相同） |
| 反证：运行时长 | 52.4s vs 133.3s（**不同**）→ 一致性不来自相同条件 |

**NF_strict = 0.000e+00（\|Δz\|）/ 0.000e+00（d′）** —— 本轮「尺子」的零点。

**Control 2：理论等价 pair（uniform α=0.25 vs constant-gate 0.25）**

| 检查 | 结果 |
|---|---|
| **逐元素 `max|ΔF|`** | **0.000e+00**；`n_diff = 0 / 328,728,576`（**全 209 张训练图**，逐 batch 记录在 `analysis/equivalence_evidence.json`） |
| coreset indices | ✓ `a894c7b0c22302b1`（9C-v2b，跨进程） |
| memory bank SHA256 | ✓ `cfe4ecff39d6a652` |
| `max|Δscore|` | **0.000e+00** |
| robustness / preservation | \|Δ\| = 0.000e+00 / 0.000e+00 |

> 9C 当时只报告聚合量（`sum_abs` / `mean`）；本轮按要求补了**真正的逐元素**检查（全训练集、逐 batch），
> 结论与 9C 一致：等价实现的 representation **逐位相同**。

## 9. Round 1 — 候选结果（bottle seed0，V2）

**参考点（同协议）**：`Strict Original (α=0)` = **\|Δz\| 0.5027 / d′ 8.5328 / tau 20.3010**。
（Old 9B V1 的 B0 = \|Δz\| 0.4782 / d′ 8.2650 —— **跨协议，coreset 不同，不可逐位比较**）

| Candidate | Old 9B (\|Δz\| / d′) | Strict 9B-R (\|Δz\| / d′) | Δ\|Δz\| vs Original | Δd′ vs Original | Old 9B Δ vs B0 | 方向一致 | Verdict |
|---|---|---|---|---|---|---|---|
| `T1_M7_L2a000_L3a025` (rank#1) | 0.3355 / 8.0216 | **0.3322 / 8.1784** | **−0.1705** | **−0.3544** | −0.1427 / −0.2434 | yes | **HOLD** |
| `T2_M10_concat_g100` (rank#2) | 0.3379 / 8.1708 | **0.3834 / 8.2881** | **−0.1193** | **−0.2448** | −0.1403 / −0.0942 | yes | **HOLD** |
| `T3_M7_L2a025_L3a000` (rank#3) | 0.3756 / 7.8876 | **0.4378 / 7.9432** | **−0.0649** | **−0.5897** | −0.1026 / −0.3774 | yes | **HOLD** |

**判读（按冻结规则，未改阈值）**
- **robustness 改善真实存在**：三个候选的 Δ\|Δz\| 为 −0.065 ~ −0.171，**远超** `EPS_RZ = 0.02`，
  更是远大于本轮实测的 `NF_strict = 0`。且与 9B(V1) 的改善**同号、同量级**（见 `d|dz|V1` 列）。
- **但 preservation 同时明显下降**：Δd′ = −0.245 ~ −0.590，**超过** `EPS_DP = 0.10` 的带宽
  → 不满足「robustness 改善且 preservation 不明显恶化」的预注册强条件；
  也不满足对称条件（preservation 改善）。
- 因此三者都是**纯 trade-off 兑换点**，而不是「免费」改善 → **HOLD**（对应 P6 的「一好一坏」）。
- **不判 STOP**：improvement 并没有消失（这正是本次重审要回答的问题），
  也没有「退回 Original」，更不是 coreset 伪差异造成。

## 10. 回答本轮唯一问题

> **9B 的候选优势在 strict replay 下：保留（robustness 改善是真实 method effect），但全部仍是纯 trade-off 兑换 → 无候选晋级。**

| 问题 | 答案 |
|---|---|
| 9C-v2b 的 strict replay 是否成功复现？ | **是**。本轮用**完全相同的实现**（`experiment9c_rng` 的 R1+R2）重跑，Control 1/2 共 10 项全 PASS，`max|Δscore| = 0.0`。 |
| 9B 哪些 improvement 是 measurement artifact？ | **本轮未发现任何被证伪的 improvement**。三候选的 robustness 改善在 V2 下不仅保留，**幅度还略大**（V1 −0.14/−0.14/−0.10 → V2 −0.17/−0.12/−0.06，同为负号）。此前 V1 噪声地板（0.020/0.054）**没有**制造出这些改善，只是限制了它们的**可判读性**。 |
| 哪些候选仍有真实 improvement？ | 三者都在 robustness 维度有真实改善（T1 > T2 > T3）；preservation 维度三者都劣于 Original。 |
| 是否存在至少一个值得进入完整验证的方案？ | **当前证据下没有**。三者都是 trade-off 兑换点，且 **bottle 之外的跨类别稳定性尚未检验**（9B 记录 T2 在 cable 上越界为负、T1 的越界幅度 ≈ V1 噪声地板）。按冻结规则 0 ADVANCE → 不触发 Round 2。 |
| 下一步最小实验是什么？ | 见 §12。 |

## 11. 发现：一个**既有**的协议属性（跨实验，必须记录）

**在审查「尺子」时发现（非 9B-R 引入、非 9C 引入）**：

`experiment5a_h_runner.run_config` 会把 `train_ids`（189）过滤进 datamodule，并写下注释
「validation 不进 memory bank」；但 `Engine.fit` 内部会再次调用 `dm.setup(stage="fit")`，
**重建 `train_data` 并丢弃该过滤**，因此实际进入 memory bank 的是**全部 209 张 train/good**
（含 `make_validation_split` 划出的那 20 张 val 图）。

**决定性证据**
- `dm.setup(stage=<TrainerFn.FITTING>) -> len(train_data)=209`（过滤后曾是 189）；
  `training_step images = 209`，`memory_bank = (21401, 1536) → 209 张 × 1024 patch`。
- 经验签名：`val` 子集分数（mean **18.27**）系统性低于**不在 bank 内**的 `clean_good`（mean **24.19**）。

**影响评估（本轮结论不受影响）**
- 该属性对**所有方法一视同仁**（每个 arm 的 bank 都含同样 20 张 val 图，且 shift/robustness 目标
  `shift_good` 是 test/good，**不在** bank 内）→ **跨方法比较仍然公平**，本轮的 verdict 不变。
- 它影响的是**绝对量级**（tau 与 NormalScore 的标定基于在 bank 内的图像）。
- **9C 的结论也完全不受影响**（等价性 bit-exact 与 bank 内容无关）。
- **不修**：修正它（189 而非 209）会破坏与 5A-H→9C 全部历史 raw 的可比性，
  必须作为一个**单独的、经批准的协议变更**（并同步重算 baseline + frontier）。本轮仅记录并标记。

## 12. 下一步（仅建议，未启动）

1. **本轮 0 ADVANCE，因此不进入 Round 2，更不进入 5×3 完整实验**（按 P7/P8）。
2. 若要继续推进 T1/T2/T3，**必须解决的核心问题不是稳健性而是 preservation**：
   现在它们相对 Original 的损失（−0.25 ~ −0.59 d′）**大于** robustness 收益的可判读价值。
   任何后续工作必须给出**机制性理由**（不能靠调参），并先在 bottle 上把 d′ 拉回到
   Original − `EPS_DP` 之内。
3. **必须先建立 V2 参考 frontier**（V2 下的 uniform-α 工作点）才能回答「候选是否跑出 frontier」——
   本轮判据只能相对 Original；9B 的「beyond frontier」需要在 V2 下重算（留待下一轮批准）。
4. 若要检验跨类别，最小实验是 `Top1 × 5 categories × seed0`（不需要 3 seeds）。
5. **协议层**：把 §11 的 val⊂bank 属性列入「待批准修正」清单，任何修正都需重算历史 baseline。
