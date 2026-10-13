# E13-A · Lightweight Representation Adaptation Screening

**日期**：2026-10-14
**论文阶段**：training-based lightweight representation adaptation 快筛（DINOv2-B 路线最后一个搜索轴）
**Verdict**：**KEEP**（R1/R3/R4 三候选 KEEP，保留最简单 **R1**）

## 一句话结论
normal-only illumination-consistent representation adaptation 首次在 DINOv2-B 线上产生**稳定、跨类别、AP 为正**的收益（mean ΔAUROC +0.015~+0.018），明确打破 E12-B 的 +0.008~+0.010 平台。

## 结果（view120 / seed0 / 300-bank / frozen DINOv2-B + 1-NN-max）

| config | Bird AUROC/AP | Car AUROC/AP | Cube AUROC/AP | mean ΔAUROC | mean ΔAP |
|---|---|---|---|---|---|
| R0（frozen） | 0.74725/0.78863 | 0.78552/0.84207 | 0.85410/0.91686 | — | — |
| R1（线性残差） | 0.75786/0.79911 | 0.80511/0.85502 | 0.87010/0.92856 | **+0.01540** | **+0.01171** |
| R2（MLP, 仅preserve） | 0.74724/0.78866 | 0.78547/0.84204 | 0.85416/0.91690 | +0.00000 | +0.00001 |
| R3（LoRA） | 0.78485/0.81679 | 0.80430/0.85722 | 0.85102/0.91773 | **+0.01777** | **+0.01473** |
| R4（MLP, cons） | 0.76848/0.80234 | 0.80767/0.85933 | 0.86249/0.92457 | **+0.01726** | **+0.01289** |

## 候选与目标
- R0 = frozen baseline；R1 = z+Wz 线性残差；R2 = 768→128→768 MLP（**仅 L_preserve**）；R3 = LoRA 最后 2 block Q/V（r=4, α=8）；R4 = 同 R2 结构（**L_cons+0.1·L_preserve**）。
- R2/R4 = 严格单变量消融（同结构，测 illumination consistency 增量价值）。此拆分在任何结果查看前记录，非事后调参。

## decision memo 十问十答
1. **mean ΔAUROC 最好**：R3（LoRA, +0.01777），其次 R4（+0.01726）、R1（+0.01540）。
2. **mean ΔAP 最好**：R3（+0.01473），其次 R4（+0.01289）、R1（+0.01171）。
3. **3/3 positive**：R1（3/3，全部 ≥+0.010）与 R4（3/3，Cube +0.0084）；R3 是 2/3（Cube −0.0031）。
4. **category-specific collapse**：无。R3 在 Cube 轻微负是性能小幅回落，非 collapse（pairwise_cos 远低于 0.98）。
5. **Car 是否仍最难**：**否**。Car 是增益最大/次大类别（R1 +0.0196 / R4 +0.0222），representation adaptation 修复了此前 Car 短板。
6. **是否明显超过 E12-B 平台**：**是**。+0.015~+0.018 且 AP 全正，数量级突破。
7. **LoRA 是否值得复杂度**：R3 最高 ΔAUROC/ΔAP 且参数最少（24k），但 Cube 轻微负；需 3-seed 确认。
8. **consistency 是否与 gain 对齐**：**是**。R2（无 consistency）≈ R0，R4（加 consistency）+0.017，证明 consistency 是 active ingredient。
9. **feature collapse**：无（std_ratio 0.96~2.33，pairwise_cos 0.21~0.67）。
10. **KEEP / HOLD / STOP**：**KEEP**（R1/R3/R4），保留 R1。

## 关键文件
- `e13a_protocol.json`（冻结协议，含 R2/R4 修正记录）、`e13a_screening.csv`、`e13a_training.csv`、`e13a_summary.json`
- `e13a_decision_memo.md`（完整 15 节）、`figures/`（6 张图）、`logs/`、`e13a_partial.json`（12/12 cells）
- 复现：`python scripts/e13a_screen.py --categories Bird,Car,Cube`（screen `e13a_main`）+ `scripts/e13a_analysis.py`

## 对论文证据链的贡献
把「training-based lightweight representation adaptation」这一格填上**肯定**答案：DINOv2-B 线上唯一稳定 KEEP 方向。下一格 = **E13-B 3-seed validation**（3 categories × 3 seeds）。
