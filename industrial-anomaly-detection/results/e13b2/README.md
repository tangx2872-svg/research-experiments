# E13-B2 · Bank Sensitivity + Extended Stability Sweep

**日期**：2026-10-14
**论文阶段**：稳定性验证 / 混杂诊断
**Verdict**：**STOP**（Stage A = UNSTABLE；R1 增益是 ceiling-limited，非普遍适用）

## 一句话结论
R1 的跨 seed 波动**不是随机 bank noise，而是系统性 ceiling effect**：ΔAUROC 与 frozen baseline 强负相关（r ≈ −0.78~−0.84）。R1 可靠提升低/中 baseline 类别（Bird +0.026、Car +0.011，各 3/3 正），但对近饱和的 Cube 无益（8 seed 仅 4/8 正，mean −0.0015）。

## Cube 8-seed 结果（Stage A 焦点）

| seed | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---|---|---|---|---|---|---|---|
| ΔAUROC | +0.0160 | −0.0123 | +0.0042 | −0.0192 | +0.0036 | −0.0048 | −0.0037 | +0.0045 |
| R0 | 0.8541 | 0.8720 | 0.8653 | 0.8681 | 0.8631 | 0.8620 | 0.8637 | 0.8595 |

- Cube：4/8 正，mean ΔAUROC **−0.00146**，mean ΔAP **−0.00097** → **UNSTABLE**。

## 三类对比

| category | mean ΔAUROC | std | positive | R0 范围 |
|---|---|---|---|---|
| Bird | +0.02586 | 0.0135 | 3/3 | 0.747–0.756 |
| Car | +0.01123 | 0.0073 | 3/3 | 0.786–0.814 |
| Cube | −0.00146 | 0.0110 | 4/8 | 0.854–0.872 |

## Ceiling effect
Pearson(R0, ΔAUROC) = −0.839（Cube 内）/ −0.783（整体）；Spearman = −0.762。frozen baseline 越高，R1 增益越低。

## decision memo 十问十答（摘要）
1. Cube seed1 仍是异常点？否——seed3（−0.0192）更负；4 个负 seed 是 Cube 高 baseline 的普遍现象。
2. Cube 新 5 seeds 几正？2/5（seed4 +0.0036、seed7 +0.0045）。
3. R0 漂移范围？Cube 0.8541–0.8720、Bird 0.7473–0.7558、Car 0.7855–0.8140。
4. ceiling effect？是（r≈−0.8）。
5. Bird/Car 仍稳定？是（各 3/3 正）。
6. 方差最大？数值上 Bird（std 0.0135），但 Cube 的方差跨零（符号翻转），是真正的问题点。
7. R1 真的 bank-sensitive？是，但表现为 ceiling effect，非随机噪声。
8. 方法问题还是 baseline 采样问题？baseline 采样/饱和问题（R0 越高增益越低）。
9. 值得扩类别？谨慎——R1 仅在低/中 baseline 有效。
10. PASS-LIKE/HOLD/STOP？**STOP**（Cube 是核心类别，R1 无法稳定提升）。

## 关键文件
- `e13b2_protocol.json`、`e13b2_results.csv`、`e13b2_summary.json`、`e13b2_decision_memo.md`、`README.md`、`figures/`（5 张图）、`logs/`
- 复现：`python scripts/e13b2_screen.py --categories Cube --seeds 3,4,5,6,7` + `scripts/e13b2_analysis.py`

## 对论文证据链的贡献
把「R1 跨 bank-seed 稳定性」填上关键诊断：R1 的波动来自 ceiling effect，不是方法失效。论文可据实表述为「R1 在 baseline 尚有 headroom 的类别上稳定增益 +0.01~+0.026，在近饱和类别上无增益」。
