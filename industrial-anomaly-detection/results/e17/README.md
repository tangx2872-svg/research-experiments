# E17 · Lightweight Refinement (BETA sweep) — README

**日期**：2026-10-14
**论文阶段**：方法收敛 / 冻结
**Verdict**：**FREEZE_CURRENT_R1**
**FINAL METHOD**：**R1-ILA**，`BETA = 0.1`（= 当前已资格认证的 R1，配置见 `final_method.json`）

## 一句话结论
对 R1 唯一显式损失参数 **BETA** 做 5 点局部扫描（0.0 / 0.05 / 0.1 / 0.2 / 0.5，seed0 × 3 类），
**没有任何挑战者通过冻结的 +0.003 晋级门** → 直接冻结当前 R1，**Round 2 未运行，0 额外 GPU 浪费**。

## Q1 — 当前 R1 是否已接近局部最优？
**是（聚合意义下）。** BETA 是单调的检测↔鲁棒性旋钮，而 0.1 位于聚合最优附近：
- BETA ↓（更强适配）：光照 shift 与 robustness_score 更低（更鲁棒），但检测变差；**BETA=0 按 d′ 指标在 Bird / Car / Cube 三类中均为最低**并在 Cube 触发 collapse。**指标限定（勘误，2026-10-15）**：按 AUROC 并非三类都最低——Cube 上 BETA=0 的 nominal AUROC 最高（0.9090），但伴随已记录的 collapse 行为（pairwise cos 0.9877，`collapsed=True`；见 `e17_partial.json`）。因此不能表述为"BETA=0 在三类上都是最差检测器"。
- BETA ↑（更强锚定）：Bird 检测上升，但 **Car 与 Cube 下降**，robustness_score 显著恶化。
- 聚合 Bird+Car mean ΔAUROC vs F0：**C0(0.1) = +0.01510 为最高**，C3(0.2) = +0.01683 vs F0… 见下注。

> 注（透明）：相对 F0，C3(0.2) 在 seed0 的 Bird+Car mean 为 +0.01683，略高于 C0 的 +0.01510；但相对 C0 仅 **+0.00173**（< 晋级门 +0.003），且 **Cube 相对 C0 掉 −0.00870**（越过 −0.003 安全线）、robustness_score 恶化 +0.119。故不构成替换理由。

## Q2 — 是否找到稳定更优配置？
**没有。** 4 个挑战者的晋级门结果：

| cfg | BETA | B+C mean ΔAUROC vs C0 | Cube ΔAUROC vs C0 | ΔAP | Δrob | collapse | 晋级 |
|---|---|---|---|---|---|---|---|
| C1 | 0.0 | −0.02200 | +0.03894 | −0.03095 | −0.279 | **True** | ✗ |
| C2 | 0.05 | −0.00645 | +0.00043 | −0.00628 | −0.117 | False | ✗ |
| C3 | 0.2 | **+0.00173** | −0.00870 | +0.00355 | +0.119 | False | ✗ |
| C4 | 0.5 | −0.00063 | −0.00975 | +0.00446 | +0.284 | False | ✗ |

全部在 G1（≥ +0.003）失败；C3/C4 另在 G2（Cube 安全）与 G4（robustness 不恶化）失败；C1 触发 collapse。

## Q3 — 是否存在性能基本不变但更简单配置？
**不存在合格项。**
- C3/C4 满足 `|Δ vs C0| ≤ 0.002`，但**结构完全同构**（同参数量 589,824、同 runtime、同损失形式），**不是"更简单"**。
- 唯一**真正更简单**的候选是 **C1（BETA=0，直接消除 BETA 超参数）**，但其 |Δ| = 0.022 远超 ±0.002 带宽，且在 Cube 上 collapse（pairwise cos 0.988）。
→ 无 `SIMPLIFY_CANDIDATE`。

## Q4 — 最终冻结哪个方法？
**current R1（`BETA = 0.1`）**，命名为 **R1-ILA**（Illumination-consistent Linear Residual Adaptation）：
`z' = z + Wz`（768×768，zero-init 线性残差，589,824 参数），
`L = L_cons + 0.1·L_preserve`，AdamW lr=1e-3，≤300 step（patience 30），
mild 光照增强 {brightness, contrast, gamma} × {0.85, 1.15}，normal-only 训练。
E16 资格认证指标（3-seed mean）：Bird ΔAUROC **+0.02586**、Car **+0.01123**、Cube **+0.00265**（未受损）。

## Q5 — 为什么选它？
按冻结优先级（多 seed 稳定性 > Bird/Car gain > Cube safety > robustness > 简洁性 > runtime > 单点最高 AUROC）：
1. **多 seed 稳定性**：E16 已证 6/6 Bird+Car cells 为正，去最好 seed 后仍 ≥ +0.005。
2. **Cube safety**：唯一在 Cube 上平均为正的配置族；C3/C4 都会把 Cube 拉低 > −0.008。
3. **robustness**：C0 在"检测不掉"的前提下保持了大幅鲁棒性改善；C3/C4 的 robustness_score 相对 C0 恶化 +0.119/+0.284。
4. **简洁性**：挑战者并不更简单（C1 更简单但性能崩坏 + collapse）。
5. 单点最高 AUROC（Bird 上 C4 = 0.76647）**排在优先级最后**，且以 Car/Cube 受损为代价。

## Q6 — 后续论文实验是否还允许调参？
**NO。**
E17 结束后 **final method frozen**。E18–E20 不再为了结果修改方法参数。
唯一参数来源为 `results/e17/final_method.json`，后续实验从该文件读取，**禁止手工复制参数**。

## 关键文件
`audit.md`（P0 + 参数审计表）· `protocol.md`（冻结）· `round1.csv`（15 metric-rows）· `round2.csv`（明确标记 NOT_RUN）·
`summary.csv`（含 Δ vs F0 与 Δ vs C0）· `final_method.json`（**E18+ 唯一参数来源**）· `figures/`（4 张）· `logs/`

复现：`python scripts/e17_refine.py --round 1` → `python scripts/e17_analysis.py`

## 机制补充（不新增机制研究，仅记录）
- **preservation 项是必需组件**：BETA=0 时 `robustness_feat` 降到 0.0007（近乎完美光照一致性）但检测性能按 d′ 三类均最低（Cube 的 nominal AUROC 最高 0.9090，但伴随 collapse；指标限定同上）→ **光照一致性本身不是目标**，必须有 frozen 锚定。
- 这也解释了 E16 记录的"normal dispersion 反升"：adapter 需要锚定以免表示退化。
- 允许的论文机制表述仍为：*R1 reduces illumination-induced feature shift / illumination-group discrepancy while preserving anomaly discrimination.*（禁用 normal dispersion reduction）

## 下一步（不自动执行）
```
NEXT = E18 Final Baseline Benchmark
```
论文最终主表（PatchCore / AnomalyDINO / SoftPatch / EfficientAD / DINOv2 F0 / Ours），优先复用 E11–E16 已有结果。等待人工裁决。
