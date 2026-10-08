# Experiment 16 — PRE-RUN PROTOCOL（FROZEN before any Exp16 target result）

**冻结时间**：2026-10-08 16:40+08:00 ｜ **起点 HEAD**：`3dba831`（Exp15）｜ **LOCAL AHEAD OF ORIGIN: 10 commits**
**性质**：Exp14–Exp15 方法筛选阶段的**最终冻结验证**（不是方法搜索）。上一轮 verdict = `HOLD — one final validation needed`。

## 0. 唯一核心问题

> **X6c 的 defect-preservation tail-safety 优势能否在扩大 seed（新增 block D = seeds 7–9）后继续成立，
> 并据此决定最终方法是否冻结。**

## 1. 冻结内容

| 项 | 值 |
|---|---|
| **Categories** | bottle / cable / hazelnut / screw / grid（本机全部 5 类；**禁止新增或删除**） |
| **Seeds** | **0,1,2,3,4,5,6,7,8,9** → **5 类 × 10 seeds = 50 evaluation units** |
| **Methods** | 仅 5 个：Original、Adaptive B2、C2、C6、X6c（定义逐字取自 Exp14/15 冻结实现） |
| **Primary metrics** | PASS、catastrophic、worst Δd′、mean/median Δd′、ΔR、robustness–preservation trade-off |
| **阈值** | `EPS_RZ = 0.02`、`EPS_DP = 0.10`、catastrophic `Δd′ ≤ −0.25`（**与 5A-H/7A-O/Exp10/11/14/15 完全一致，禁止修改**） |
| 参考 | 每个 (cat,seed) 的 **同单元 corrected Original** |
| 实现来源 | `scripts/experiment16_fusion.py`（融合规则）、Exp14/15 的 runner（Original/B2 unit） |

## 2. 方法定义（冻结，不得重新解释）

| method | 定义 |
|---|---|
| Original | PatchCore α=0（Exp14 `E14_A0`：l2/l3 `alpha_in(0)` 严格 short-circuit → 与 plain PatchCore 逐位一致） |
| Adaptive B2 | L2 `residual(β_c)` + L3 `alpha_in(0.25)`（Exp14 `E14_A1`；与 Exp11 `E11_B2` **bit-exact 已核验**） |
| C2 | `z = 0.35·z_orig + 0.65·z_B2` |
| C6 | `z = max(z_orig, z_B2)` |
| X6c | `z = mean(z_C2, z_C6)` |
| z 校准 | `z = (score − μ_g)/σ_g`，μ_g/σ_g 取自**该路自己的 clean_good/none**（normal-only，禁 defect 校准） |

## 3. 绝对禁止（违反即 STOP + REPORT）

禁止新 feature module / 新 score fusion / 新候选 X7+ / 修改任何方法定义 / 重新 fine-tune 权重 /
新 grid search / 修改 PASS 阈值 / 因 seed 失败而删 seed / 为跑满时间造实验 / 临时下载数据集 /
覆盖 Exp14/15 原始结果 / 重跑已有 protocol-compatible 历史结果 / 自动 push Git。

## 4. 阶段计划

| 阶段 | 内容 | GPU |
|---|---|---|
| P0.5 | 债务审计（`EXPERIMENT_DEBT_AUDIT.md` + csv）→ **VERDICT: CLEAN** | 0 |
| P0 | 启动审计（git/GPU/数据/覆盖） | 0 |
| P1 | 本冻结文件 | 0 |
| P2 | `reuse_manifest.csv`（尽可能复用 seeds 0–6） | 0 |
| P3 | Sanity S1–S4（历史重建一致性 / X6c 定义 / 阈值 / 输出隔离） | 0 |
| P4 | GPU 补跑：**seeds 7,8,9 × 5 类 × {Original, B2} = 30 units** | 30 |
| P5 | 50-unit 最终表 | 0 |
| P6 | seed-block 稳定性（A:0–2 / B:3–4 / C:5–6 / **D:7–9（新）**） | 0 |
| P7 | 统计：bootstrap 10000（X6c vs B2/C2/C6）+ tail analysis（worst 1/3/5） | 0 |
| P8 | robustness cost 决策表 | 0 |
| P9 | Figures 1–5（PNG + PDF） | 0 |
| P10 | README（Exp16 + 根 README） | 0 |
| O1–O4 | paper assets / evidence map / main table draft / ablation inventory | 0 |

## 5. 并发与 OOM 策略

Exp15 曾在 4 workers 下 OOM（hazelnut/screw/grid 单进程 ~7.4 GiB）。本轮：**默认 2 workers**，
smoke 后若显存充裕可升至 3，**必须保留余量**（目标 <20 GB）。OOM 时：写 ERROR_REPORT → 降并发 → **只补跑失败 unit** → 协议不变。

## 6. Verdict（三选一，**不允许 HOLD**）

- **A — FREEZE X6c**：50 单元稳定 + block D 不崩 + catastrophic 明显低于 B2 + tail 优势仍明显 + robustness cost 未恶化到不可接受 + 相对 C2/C6 的复杂度有足够收益
- **B — FREEZE SIMPLE WINNER: `<method>`**：X6c/C2/C6 统计不可区分且 X6c 的额外复杂度没有足够收益 → Occam
- **C — STOP METHOD SEARCH**：扩展后 X6c 优势消失且 C2/C6 也无法稳定优于 B2（论文转向 phenomenon + analysis + simple baseline）

**方法搜索阶段在本轮结束。**

## 7. 时间预算

P0–P3 ≈ 5–15 min；GPU 补跑 ≈ 20–45 min（30 units）；重建 + 统计 ≈ 5–10 min；图 + README ≈ 10–20 min
→ **预计 45–90 min**；提前完成则执行 O1–O4，不为凑时间造实验。
