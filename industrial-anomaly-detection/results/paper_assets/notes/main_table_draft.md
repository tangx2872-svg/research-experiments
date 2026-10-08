# Main Table Draft（论文主表草稿）

**来源**：Exp16 final 50-unit table（5 MVTec 类 × seeds 0–9）。判据与 5A-H/Exp10–16 完全一致：
`PASS = ΔR ≤ −0.02 且 Δd′ ≥ −0.10`（相对同 (cat,seed) Original）；`catastrophic = Δd′ ≤ −0.25`。

| Method | PASS/50 | PASS rate | catastrophic | worst Δd′ | mean Δd′ | median Δd′ | mean ΔR | Complexity |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| Original | 0/50 | 0.0% | 0 (0.0%) | 0.0000 | 0.0000 | 0.0000 | 0.0 | low (1 unit) |
| Adaptive B2 | 30/50 | 60.0% | 8 (16.0%) | -0.4944 | 0.1438 | 0.0763 | -0.0811 | low (1 unit, category-adaptive beta) |
| C2 | 35/50 | 70.0% | 0 (0.0%) | -0.1817 | 0.1857 | 0.1129 | -0.0545 | low (1 fixed weight) |
| C6 | 38/50 | 76.0% | 0 (0.0%) | -0.2142 | 0.1468 | 0.0511 | -0.0455 | low (max, 1 rule) |
| X6c | 38/50 | 76.0% | 0 (0.0%) | -0.1416 | 0.1831 | 0.0991 | -0.0493 | medium (mean of 2 rules) |

## Historical baselines（方法筛选阶段的冻结结果，供对照）

| Method | PASS | Evidence | Source |
|---|---:|---|---|
| Fixed C2 | 9/15 | E2 | Exp10（`results/experiment_10_preservation_recovery/README.md`） |
| Adaptive B2 | 12/15 | E2 | Exp11（本表已含 50 单元版本） |
| C1（Exp14 规则 winner） | 25/35 | E3+ | Exp15（**在 unseen seeds 上退化，已归档**） |
| Category × Channel (M1/M2/M3) | STOP | E1 | Exp12 |
| Dual-Path（Family A） | 结构性否定 | E1 | Exp14 |
| Tiny INSS（Family B） | STOP | E1 | Exp14 |

**TODO — requires experiment**：E4 held-out dataset（MPDD）行，见 `held_out_validation_plan.md`。
