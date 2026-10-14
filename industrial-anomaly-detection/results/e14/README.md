# E14 · Anomaly-Oriented Representation Family Screening

**日期**：2026-10-14
**论文阶段**：换 representation family（不继续救 DINOv2-B）
**Verdict**：**STOP**（F1/F2 均远低于 F0，无 family 晋级）

## 一句话结论
DINOv2-B + 1-NN-max（LayerNorm + max）仍是 M²AD Bird/Car/Cube 上最强的 frozen baseline；AnomalyDINO 与 SoftPatch 两个 anomaly-oriented family 都无法超越它。

## 结果

| family | Bird AUROC | Car AUROC | mean(Bird,Car) ΔAUROC |
|---|---|---|---|
| F0（DINOv2-B 1-NN-max） | **0.74725** | **0.78552** | — |
| F1（AnomalyDINO） | 0.70214 (−0.045) | 0.75684 (−0.029) | −0.0369 |
| F2（SoftPatch） | 0.64396 (−0.103) | 0.50219 (−0.283) | −0.1933 |

- F1/F2 在 Bird 就跌破 Stage A 的 −0.015 门槛 → 均 STOP，未进 Stage B（Cube）。
- F3（EfficientAD，唯一真正新 paradigm）本轮 SKIP（下载+训练成本、第三优先）。

## decision memo 十问十答
1. Bird 最好：F0（0.747）。
2. Car 最好：F0（0.786）。
3. Cube 最稳：F0（0.854，F1/F2 未跑 Cube）。
4. 是否打破 DINOv2-B ceiling：否（两候选 family 在 Bird/Car 远低于 F0）。
5. AnomalyDINO 值得继续：否（raw 无 LayerNorm + mean_top1p 劣于 F0 的 LayerNorm + max）。
6. SoftPatch robust memory benefit：否（Car 崩 −0.283；印证 E7「干净 bank 上软加权是扰动」）。
7. EfficientAD 训练成本：未测试，不评估。
8. 性价比最高：F0（0 GPU，最强）。
9. 存在 KEEP：否。
10. 下一步 family：无，E14 CLOSED。

## 关键文件
- `e14_family_audit.md`、`e14_protocol.json`、`e14_screening.csv`、`e14_summary.json`、`e14_decision_memo.md`、`README.md`、`figures/`（5 张图）、`logs/`
- 复现：`python scripts/e14_screen.py --categories Bird,Car --families F1,F2` + `scripts/e14_analysis.py`

## 对论文证据链的贡献
补齐「换 representation family 是否值得」的否定答案：DINOv2-B + 1-NN-max 仍是最强 baseline，AnomalyDINO / SoftPatch 无法超越。结合 E11–E13，证据链完整：表示（DINOv2-B）是主导杠杆，scoring/fusion/adaptation/family 换轴均无法在该表示之上稳定超越 1-NN-max。下一步 = 人工裁决（换数据问题定义 / 换 detection paradigm / 收敛成应用型有限结论）。
