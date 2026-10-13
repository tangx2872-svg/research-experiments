# E12-B · Lightweight Illumination-Adaptive Multi-Level Fusion Screening

**日期**：2026-10-13
**论文阶段**：representation-side lightweight adaptation（frozen-feature scoring 搜索 CLOSED 后的新轴）
**Verdict**：**HOLD**（保留 1 个最简单候选 **S2**）

## 协议摘要
frozen DINOv2-B（block9 + final patch tokens，1024×768）之上，只训练一个**可学习融合 gate**，normal-only 训练，scoring 固定为 1-NN-max。候选：S0(final-only，复用) / S1(concat，复用) / S2(标量 gate) / S3(通道 gate) / S4(逐样本 SE gate)。协议冻结于 `e12b_protocol.json`（HEAD 495a6c8b）。

## 结果（view120 / seed0 / 300-bank / official image_anomaly+detectable）

| config | Bird AUROC/AP | Car AUROC/AP | Cube AUROC/AP | mean ΔAUROC | mean ΔAP |
|---|---|---|---|---|---|
| S0（复用） | 0.74725/0.78863 | 0.78552/0.84207 | 0.85410/0.91686 | — | — |
| S1（复用） | 0.75629/0.78730 | 0.79442/0.82817 | 0.85963/0.91915 | +0.00782 | −0.00431 |
| S2 | 0.75739/0.79161 | 0.79999/0.83385 | 0.85887/0.91936 | +0.00979 | −0.00091 |
| S3 | 0.75434/0.78090 | 0.80128/0.83761 | 0.86190/0.92072 | +0.01022 | −0.00278 |
| S4 | 0.75470/0.78755 | 0.80034/0.83242 | 0.85750/0.91828 | +0.00856 | −0.00310 |

## README 九问九答

1. **adaptive fusion 是否超过 final-only baseline（S0）？** 是，但幅度小。3/3 候选 AUROC 3/3 positive，mean ΔAUROC +0.0086~+0.0102，全部超过 S0。但 AP 几乎打平（mean ΔAP −0.0009~−0.0031）。

2. **是否超过 fixed block9+final（S1）？** 否（未达可发表门槛）。S2 mean ΔAUROC +0.00979 vs S1 +0.00782，仅 +0.0020，未过 complexity gate 的 +0.003；S3 的 +0.01022 略高但以 AP 为代价。learnable 化本质只是**复现**了固定 concat 的收益。

3. **AP 是否被救回来？** 否。3/3 候选 mean ΔAP 仍为负（与 P4-A1 同现象：AUROC 微升、AP 微降）。S2 最接近 0（−0.0009）。

4. **Bird/Car/Cube 是否仍发生 category split？** 是。增益分布不均（S2：Car +0.0145 最大、Cube +0.0048 最小；S3 在 Bird 最弱），延续 E5–E12-A 反复出现的类别特异模式，无跨类别稳健且 AP 为正的信号。

5. **gate 实际偏向 block9 还是 final？** 一致偏向 **block9**（init 0.5 → 学到 0.53–0.70）。S2 λ=0.57–0.61，S3 λ_mean=0.53–0.58，S4 w_block9=0.59–0.70。

6. **学习后的 illumination consistency 是否真的改善？** 改善但很小，且与 AUROC 不对齐。learned gate 相对固定等权（before）一致性再降 0.002–0.008；但一致性改善最多的 S3 并非 AUROC/AP 最好的候选。

7. **是否存在 collapse？** 否。std_fused 训练前后均 ≈ 0.023–0.025（非 0），loss 全 finite，9/9 cell 无 collapse。

8. **KEEP / HOLD / STOP？** **HOLD**（S2/S3/S4 全部 HOLD）。无候选满足 KEEP 的「mean ΔAP ≥ 0」；无候选 STOP（均满足 HOLD 下限）。保留 1 个最简单候选 **S2**（1 参数）。

9. **下一步是否值得 3 seeds？** 按预注册规则本轮**不自动跑**。S2 是唯一 HOLD 候选，是否值得 1 次 3-seed 小验证交由人工裁决；若无裁决价值，则关闭 adaptive fusion 轴，转向 training-based representation/scoring 方向（新预注册协议）。

## 关键文件
- `e12b_protocol.json`（冻结协议）、`e12b_screening.csv`、`e12b_training.csv`、`e12b_summary.json`
- `e12b_consistency.json`（before/after 一致性）、`e12b_partial.json`（9/9 cells）
- `e12b_decision_memo.md`（完整 15 节决策 memo）、`figures/`（4 张图）
- 复现：`python scripts/e12b_screen.py --categories Bird,Car,Cube` + `scripts/e12b_consistency.py` + `scripts/e12b_analysis.py`

## 对论文证据链的贡献
把「frozen DINOv2-B 表示 + lightweight learnable fusion」这一格填上否定性答案：learnable 融合**不构成**相对固定 concat 融合（S1）的可发表改进。frozen-feature 侧的 scoring 微调（P4/P5/E12-A）与 representation 侧轻量 adaptation（E12-B）两条子轴均已关闭。下一格 = 人工裁决（负面叙事收尾 vs 训练式方向，需新协议）。
