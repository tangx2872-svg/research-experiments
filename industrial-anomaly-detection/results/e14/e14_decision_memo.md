# E14 · Anomaly-Oriented Representation Family Screening — Decision Memo

## Verdict: **STOP**（F1/F2 均远低于 F0，无 family 晋级，family screening CLOSED）

## 1. 实验题目
换 anomaly-oriented representation family 是否比当前 DINOv2-B + 1-NN-max baseline 提供更稳定的 headroom？

## 2. 日期
2026-10-14

## 3. 论文阶段
方法搜索重新换 family（不继续救 DINOv2-B）。

## 4. 要验证的核心论断
「是否存在一个 anomaly-oriented representation/detector family，比当前 baseline 更适合工业异常检测，并能为 illumination-consistency adaptation 提供更好底座。」

## 5. 实验目的
E13 证明方法收益受 DINOv2-B ceiling 限制；先判断换 family 是否值得。

## 6. 数据与参数
Bird/Car/Cube，view120，seed0。F0=DINOv2-B 1-NN-max（复用 E13 R0）；F1=AnomalyDINO（DINOv2 448 raw → L2-kNN k=1 → mean_top1p）；F2=SoftPatch（WideResNet50 PatchCore + 官方 WeightedGreedyCoresetSampler soft weighting）；F3=EfficientAD（本轮 SKIP）。

## 7. 实验过程
P0 审计（结论：F1/F2 不改变 representation，仅 scoring/memory 差异）→ smoke（Bird）→ Stage A（Bird/Car，4 cells）。两 family 均在 Bird 就跌破 −0.015 → 均 STOP，未进 Stage B（Cube）。

## 8. 结果（Fact）

| family | Bird AUROC | Car AUROC | mean(Bird,Car) ΔAUROC | mean ΔAP |
|---|---|---|---|---|
| F0（DINOv2-B 1-NN-max） | **0.74725** | **0.78552** | — | — |
| F1（AnomalyDINO） | 0.70214 (−0.045) | 0.75684 (−0.029) | **−0.0369** | −0.0628 |
| F2（SoftPatch） | 0.64396 (−0.103) | 0.50219 (−0.283) | **−0.1933** | −0.2501 |

## 9. 异常情况
无 OOM / NaN / 崩溃。F2 Car 近乎随机（AUROC 0.502）。

## 10. 分析（Fact → Interpretation → Hypothesis）

**Fact**
1. F1/F2 在 Bird/Car 上全部低于 F0，且远低于 Stage A 的 −0.015 门槛。
2. F2 在 Car 上灾难性（−0.283）。

**Interpretation**
1. **DINOv2-B + 1-NN-max（LayerNorm + max）仍是 Bird/Car/Cube 上最强的 frozen baseline**，AnomalyDINO 与 SoftPatch 两个 anomaly-oriented family 都不如它。
2. **AnomalyDINO 的 raw（无 LayerNorm）特征 + mean_top1p 明显劣于 F0 的 LayerNorm + max**（−0.037），说明 LayerNorm 对 DINOv2 patch 特征的可分性很关键，mean_top1p 也不如 max。
3. **SoftPatch 的 soft weighting 在干净 bank 上是扰动**（Car 崩 −0.283），印证 E7 结论「soft weighting 为噪声 bank 设计，干净 bank 上只是扰动距离场」。

**Hypothesis（未验证）**
- 本轮未测试 F3（EfficientAD，唯一真正新 paradigm）；不排除 teacher-student 蒸馏仍有价值，但需后续专门评估。

## 11. 结论
无 family 突破 F0。E14 family screening **CLOSED**。DINOv2-B + 1-NN-max 是当前该协议下的最强 baseline。

## 12. 出口条件判断
F1/F2 均未达 Stage A 晋级门槛（any-cat ΔAUROC ≥ −0.015 不满足），全部 **STOP**，未进入 Stage B。

## 13. 对论文证据链的贡献
本实验补齐了「换 representation family 是否值得」这一格的**否定答案**：在 M²AD Bird/Car/Cube 上，DINOv2-B + 1-NN-max 仍是最强 baseline，AnomalyDINO / SoftPatch 两个 anomaly-oriented family 无法超越。结合 E11-E13，形成完整证据链：**表示（DINOv2-B）是主导杠杆，scoring/fusion/adaptation/family 换轴均无法在该表示之上稳定超越 1-NN-max**。

## 14. 下一步（人工裁决，不自动执行）
按协议 E14 STOP → 人工裁决：
- A. 换数据问题定义；
- B. 换 detection paradigm（如 F3 EfficientAD teacher-student，唯一未测的新 paradigm）；
- C. 收敛成应用型有限结论。

## 15. 复现信息
- 环境：RTX 3090 24GB / PyTorch 2.8.0+cu128 / Python 3.12.3 / conda base
- 命令：`python scripts/e14_screen.py --categories Bird,Car --families F1,F2` + `scripts/e14_analysis.py`
- 输出：`results/e14/{e14_family_audit.md, e14_protocol.json, e14_screening.csv, e14_summary.json, e14_decision_memo.md, README.md, figures/}`
