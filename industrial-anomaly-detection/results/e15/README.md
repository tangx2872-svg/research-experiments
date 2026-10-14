# E15 · EfficientAD Teacher-Student Feasibility + Mini-Screen

**日期**：2026-10-14
**论文阶段**：detection paradigm change feasibility test
**Verdict**：**STOP**（EfficientAD Bird ΔAUROC −0.135，teacher-student 未提供足够优势）

## 一句话结论
EfficientAD（官方 teacher-student 蒸馏，small 模型 70000 steps）在 M2AD Bird 上 AUROC 0.612，远低于 frozen DINOv2-B + 1-NN-max 的 0.747（−0.135）。teacher-student paradigm 在真实光照（view120）下不敌 DINOv2-B。

## 结果

| category | F0 AUROC/AP | EfficientAD AUROC/AP | ΔAUROC / ΔAP |
|---|---|---|---|
| Bird | 0.74725 / 0.78863 | 0.61206 / 0.62196 | −0.13519 / −0.16667 |
| Car | 0.78552 / 0.84207 | —（未跑，STOP） | — |
| Cube | 0.85410 / 0.91686 | —（未跑，STOP） | — |

- Bird ΔAUROC −0.135 < −0.020 → 按协议直接 STOP，不跑 Car/Cube。
- train time = 29.2 min；n=653（good 320 / ng 333）。

## decision memo 十问十答
1. 工程是否适合服务器：是（torch 2.8 实测可跑、teacher 权重仓库自带、训练正常，仅 2 处纯工程修复）。
2. Bird 是否接近 F0：否（−0.135，远低于 −0.010 门槛）。
3. Car 是否接近 F0：未跑（Bird 即 STOP）。
4. Cube 是否抗 ceiling：未跑。
5. 3-cat mean ΔAUROC/ΔAP：未达（仅 Bird，−0.135/−0.167）。
6. error pattern 是否明显不同：未深查（未晋级）。
7. teacher-student 是否值得训练成本：否（29 min 换来 −0.135）。
8. 训练不稳定：无（loss 正常收敛）。
9. 是否值得第二轮：否。
10. **STOP**。

## 关键文件
- `e15_audit.md`、`e15_protocol.json`、`e15_metrics.csv`、`e15_summary.json`、`e15_decision_memo.md`、`README.md`、`figures/`（6 张图）、`logs/`
- 复现：`python scripts/e15_screen.py --categories Bird --train-steps 70000` + `scripts/e15_analysis.py`

## 对论文证据链的贡献
补齐「换 detection paradigm（teacher-student）」的否定答案。**major detector-family search = CLOSED**：DINOv2-B + 1-NN-max 是最强 baseline，frozen scoring / fusion / adaptation / anomaly-oriented family / teacher-student paradigm 均无法超越。下一步 = 人工裁决（重定义问题 / 收敛有限应用方案 / 用 R1 正信号做小论文包装）。
