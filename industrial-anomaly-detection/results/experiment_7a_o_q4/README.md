# Overnight Queue Q4 — Strong-Baseline Extended Validation

- **状态**：DONE（**56 GPU units**：Q4-A 36 + Q4-B 20；freeze SHA256 `36c4115262fa70fb112a189a0363d317a7605d5e05c1f8cec7fa89db59c2bab2` 见 `reference/q4_protocol_freeze.json`）
- **EXPLORATORY / asset completion**：不引入新方法、不引入新 benchmark，只把 baseline 响应图补完整。

## 1. 目的与允许范围

6B 的核心发现是 `Uniform α≈0.40091275` 本身就是很强的 baseline。Q4 **不继续找新方法**，只补：
1. **Q4-A**：把"uniform normalization strength → preservation / robustness"响应图补完整
   （只用**历史已定义的 α points**）；
2. **Q4-B**：把 Q3 的 illumination stress-test 从 seed0 扩展到 seeds 1/2（同一协议/条件/方法），
   使 `method × category × seed × condition` 完整。

**禁止**（addendum）：new dataset / new backbone / per-category α tuning / dense α sweep / new illumination family。
→ 遵守：只运行**历史缺失单元**，bottle/grid 的 α 单元直接复用 6A/6B raw（0 GPU）。

## 2. 资产审计（Q4 启动前）

uniform-α 覆盖（5 cat × 3 seeds = 15 units/点）：

| α | 0.0 | 0.125 | 0.20 | 0.25 | 0.30 | 0.40091275 | 0.5 | 0.601369 | 0.801825 |
|---|---|---|---|---|---|---|---|---|---|
| 补前 | 15/15 | 6/15 | 6/15 | 6/15 | 6/15 | 15/15 | 15/15 | 15/15 | 15/15 |
| 补后 | 15/15 | **15/15** | **15/15** | **15/15** | **15/15** | 15/15 | 15/15 | 15/15 | 15/15 |

## 3. Q4-A 结果 — uniform normalization strength 响应图（5 cat × 3 seeds，全部 15/15）

| α | d′ ↑ | \|Δz\| ↓ | image AUROC | fpr_clean |
|---|---|---|---|---|
| 0.0 | 4.9852 | 0.3105 | 0.9866 | 0.9333 |
| 0.125 | 4.9649 | 0.2867 | 0.9876 | 1.0000 |
| **0.20** | **5.0094** | 0.2777 | **0.9877** | 1.0000 |
| 0.25 | 4.9320 | 0.2503 | 0.9869 | 0.9333 |
| 0.30 | 4.8831 | 0.2394 | 0.9864 | 1.0000 |
| **0.40091275** | 4.7940 | 0.2134 | 0.9827 | 0.9333 |
| 0.50 | 4.6670 | 0.2082 | 0.9757 | 0.9333 |
| 0.601369 | 4.5078 | **0.1895** | 0.9710 | 0.9333 |
| 0.801825 | 4.2272 | 0.1902 | 0.9597 | 0.9333 |

**读法（Fact）**
1. `|Δz|` 随 α **单调下降**直到 α≈0.60（0.3105 → 0.1895），之后**饱和/回升**（α=0.8018 为 0.1902）。
2. `d′` 随 α 总体下降，但**低 α 端非单调**：**α=0.20 的 d′ (5.0094) 高于 α=0 (4.9852)**，同时 `|Δz|` 更低
   → **α=0.20 Pareto 支配 α=0**。
3. **支配关系（本 9 点内）**：α=0.20 ⊃ α=0 与 α=0.125；**α=0.6014 ⊃ α=0.8018**。
   → "strength 越大越好"与"越小越好"**都错**；前沿是 α ≈ 0.20–0.60 这一段。
4. `image AUROC` 在 α ∈ [0, 0.25] 基本持平（0.9866–0.9877），α ≥ 0.40 后开始下降（0.9757 @0.5，
   0.9597 @0.8018）→ **过强 normalization 同时损害 image-level 判别**。
5. `fpr_clean` 分辨率很差（τ_val = 20 张 val good 的最大分数；clean good 几乎总超过 τ），
   在 α ∈ {0.125, 0.20, 0.30} 上恒为 1.0 → **该指标不具判别力，不应作为主要依据**（已知口径缺点）。

**跨来源一致性 sanity**：4 个新 α 上，"历史 bottle/grid"与"新增 cable/hazelnut/screw"两组的
`|Δz|` 量级一致（例：α=0.125 → 0.2946 vs 0.2814；α=0.20 → 0.2785 vs 0.2772）→ **无来源伪影**；
prov=6B:6 + Q4:9（α=0.25 为 5A:1 + 6A:5 + Q4:9）。

## 4. Q4-B 结果

Q4-B 的 20 个单元完成并写入 `results/experiment_7a_o_q3/raw/*/seed_{1,2}/method_*`（q3 runner 无 `--out-root`），
由 Q3 分析统一合并 → 见 `results/experiment_7a_o_q3/README.md §7`（3-seed 曲线与 slopes）。

## 5. 异常情况（诚实记录）

1. **首轮 Q4 启动的参数顺序 bug**：`experiment7ao_runner` 的 `--units` 约定是 `category:seed:CONFIG`，
   而 Q4-A 单元首轮按 `CONFIG:category:seed` 传入 → w1/w2 立即 `ValueError` 退出，w3 把 4 个 α-point
   当成 "method" 交给 stress-runner 跑了 12 个 condition（**12 个误跑单元**，已隔离到
   `results/experiment_7a_o_q3/accidental_stress_runs/`，未用于任何分析，未删除）。
   修正后重跑，Q4-A 的 36 units 全部按冻结协议完成。
2. Q4-B 首轮只完成 6/20（其余配额被上述误跑占用）→ 事后补齐 14 个缺失单元，最终 **20/20**。
3. **无数据损坏、无 baseline mismatch**（Q0/Q3 的 equivalence 检查均为 `max|Δscore| = 0`）。

## 6. 对论文证据链的贡献

把 baseline 响应图从"6 个点、覆盖不均"变成 **9 个 α 点 × 5 cat × 3 seeds 全满**，
并得到两个可写进论文的可核验事实：
**① `|Δz|` 对 α 单调下降并在 α≈0.6 饱和；② α=0.20 Pareto 支配 α=0（而 α≥0.8 被 α≈0.60 支配）**。
→ 支撑"normalization strength 是一维旋钮，但有**甜点区间**而不是单调最优"的表述。

## 7. 下一步

Q4 已完整；不继续 GPU 工作。结论汇总进 `OVERNIGHT_FINAL_REPORT.md` 与根 README。
