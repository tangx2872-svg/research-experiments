# E17 — Lightweight Refinement (论文方法收敛) · FROZEN PROTOCOL

冻结时间：2026-10-14（**在任何 E17 新结果产生之前**）
HEAD：`52b2b74` · 性质：**收敛/冻结轮，不是海选，不是调参救火**

---

## 0. 唯一目标

> 在不增加明显复杂度的前提下，判断当前 R1 是否已经足够好，或是否存在一个更简单 / 更稳定 / 略优的小变体值得冻结为最终 **Ours**。
>
> **若没有明显更好的配置 → 直接冻结当前 R1 为最终 Ours（合法且优先的结果）。**

当前论文机制表述（允许）：
> R1 reduces illumination-induced feature shift / illumination-group discrepancy while preserving anomaly discrimination.

**禁止表述**（E16 已证伪）：normal dispersion reduction。

---

## 1. 唯一搜索轴：`BETA`

`L = l_cons + BETA · l_preserve`（`l_cons` 权重硬编码 1.0；`BETA` 是唯一显式损失级自由参数）。
`BETA` ↑ = frozen 锚定更强 = **适配更弱**；`BETA` ↓ = **适配更强**；`BETA=0` = 纯光照一致性。

运行时通过 `e13a_screen.BETA = <value>` 注入（`compute_loss` 调用时读模块全局），**不修改任何冻结代码**。

### 冻结候选（5 个，含当前 R1）

| ID | BETA | vs 当前 | 语义 |
|---|---|---|---|
| **C0** | **0.1** | 1.0× | 当前 R1（**全部复用，禁止重跑**，仅作复现门） |
| C1 | 0.0 | 0× | 纯光照一致性（最强适配） |
| C2 | 0.05 | 0.5× | 锚定减半 |
| C3 | 0.2 | 2× | 锚定加倍 |
| C4 | 0.5 | 5× | 强锚定（温和适配） |

### 冻结不变量（禁止修改）
结构（`z + Wz`，768×768，zero-init）· `LR_ADAPTER=1e-3` · `MAX_STEPS=300` · `PATIENCE_STEPS=30` ·
`EVAL_EVERY=5` · `BATCH_IMGS=8` · `TRAIN_N/VAL_N=240/60` · `AUG_LIST`（brightness/contrast/gamma × {0.85,1.15}）·
backbone DINOv2-B frozen · scoring 1-NN-max · bank 300 · view120 · seed 规则 `torch.manual_seed(seed)` + `np.random.default_rng(seed)` + `E8.bank_for(pool, seed)`。

**禁止**：大网格 / 随机搜索 / Bayesian optimization / 多参数组合 / 结构搜索 / 新 loss / 新 backbone / 新 scoring / category-specific 参数。

---

## 2. Round 1（seed0 only）

```
Bird seed0 / Car seed0 / Cube seed0  ×  C0(复用) + C1/C2/C3/C4
新增 GPU = 3 category-run × 4 新配置 = 12 新 unit
```
**工程优化**：每 category-run 只做 **1 次特征提取**（bank/test/aug），随后在缓存特征上循环 5 个 BETA。
C0 作为**机器正确性复现门**：其 AUROC/AP 必须与 `e16_partial.json` 一致到 **1e-6**，否则停机。

### Round 1 指标（每配置）
`AUROC` · `AP` · `d_prime` · `robustness_score` · `illumination shift magnitude (D)` · `runtime` · `steps_used`
并分别计算 **Δ vs F0** 与 **Δ vs current R1 (C0)**。

### Round 1 淘汰规则（挑战者必须证明自己值得替换 C0）
进入 Round 2 需**同时**满足：
```
1. Bird + Car mean ΔAUROC（相对 C0） >= +0.003
2. Cube 相对 C0 不差超过 -0.003
3. mean ΔAP 相对 C0 不恶化超过 -0.002
4. robustness_score 相对 C0 不明显变差
5. 无 collapse（std 不降 >50%；mean pairwise cos 不 >0.98；norm 不爆炸）
```
若**无候选满足** → 直接结束 E17，`FREEZE_CURRENT_R1`。

### SIMPLIFY_CANDIDATE 特殊规则
若 `|Bird+Car mean AUROC − C0| <= 0.002` 且候选**结构更简单 / 参数更少 / 更稳定 / 更快 / 解释更自然** → 标记 `SIMPLIFY_CANDIDATE`，可进入 Round 2。

### Early Stop
- Round 1 后若所有挑战者都**没有达到 `C0 − 0.002` 范围** → 直接停止，不跑 Round 2。
- 若某候选 Bird 明显提高但 Car/Cube 明显恶化 → 直接淘汰（**不做 category-specific 参数**）。

---

## 3. Round 2（最多 2 个挑战者）

```
current R1 (9/9 cell 全部复用)
+ 最多 2 个挑战者 × (Bird/Car/Cube × seed0,1,2)
```
**禁止超过 2 个挑战者**。Round 2 中 F0 与 C0 的 9/9 cell 全指标**复用 E16**。

---

## 4. 最终选择规则（严格按优先级）

```
1. 多 seed 稳定性
2. Bird / Car detection gain
3. Cube safety
4. robustness_score
5. 方法简洁性
6. runtime
7. 单个最高 AUROC        <- 排在最后
```

## 5. FREEZE 判据

### 情况 A（默认，完全成功）
没有挑战者明显胜出 →
```
E17 VERDICT = FREEZE_CURRENT_R1
FINAL METHOD = current R1
```

### 情况 B
多 seed 后**同时**满足：
```
Bird+Car mean ΔAUROC >= C0 + 0.003
且 Bird+Car 至少 4/6 cells 不低于 C0
且 Cube mean 差值 >= -0.003
且 AP / robustness 不恶化
```
→ `E17 VERDICT = FREEZE_REFINED_R1`

### 情况 C
`|ΔAUROC vs C0| <= 0.002` 且挑战者明显更简单 →
```
E17 VERDICT = FREEZE_SIMPLIFIED_R1
```
decision memo **必须**写明：选择依据是 complexity-performance tradeoff，**不是**性能更高。

---

## 6. 禁止救火

若所有候选都比 current R1 差：**不得**新增第 6 个参数 / 细分网格 / 改 loss / 改 epoch / 换 scoring / 补模块。
立即 `FREEZE_CURRENT_R1`。

## 7. Diagnostic 约束

E16 已有完整机制链。E17 **不重复机制研究**，只为最终候选记录
`illumination shift magnitude` 与 `robustness_score`，确认没有破坏 E16 的机制链。

## 8. 时间预算

目标 1–2 h。单 unit >10 min 需 heartbeat。若超 2.5 h 硬顶则停止扩展、只完成当前 unit。

## 9. 输出

```
results/e17/{audit.md, protocol.md, round1.csv, round2.csv, summary.csv,
             decision_memo.md, final_method.json, README.md, figures/, logs/}
```
`final_method.json` 为 E18+ 的**唯一参数来源**，后续实验不得手工复制参数。

## 10. README 必须回答
Q1 当前 R1 是否已接近局部最优？ Q2 是否找到稳定更优配置？ Q3 是否存在性能基本不变但更简单配置？
Q4 最终冻结哪个方法？ Q5 为什么选它？ Q6 后续论文实验是否还允许调参？（原则答案：**NO**）
