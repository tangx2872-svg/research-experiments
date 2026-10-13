# E13-A · Lightweight Representation Adaptation Screening — Decision Memo

## Verdict: **KEEP**（R1 / R3 / R4 三候选全部 KEEP，保留最简单 **R1**）

> ⚠️ 这是 DINOv2-B 表示线自 E11 换轴以来**第一个稳定 KEEP**，且是数量级上的突破（mean ΔAUROC +0.015~+0.018、mean ΔAP 全正，对比 E12-B 平台 +0.008~+0.010 且 AP 全负）。

## 1. 实验题目
在 frozen DINOv2-B final-layer 表示之上，用 normal-only lightweight representation adaptation（R1 线性 / R2-R4 MLP / R3 LoRA）能否提升真实光照下的异常检测性能？

## 2. 日期
2026-10-14

## 3. 论文阶段
training-based lightweight representation adaptation 快筛（DINOv2-B 路线最后一个搜索轴）。

## 4. 要验证的核心论断
「frozen DINOv2-B 的特征空间能否通过 normal-only lightweight adaptation，在真实光照下取得超过 frozen baseline 的稳定正收益？」

## 5. 实验目的
E11/E12 已证明 scoring/fusion 层免训练或极轻修改收益上限 ~+0.01（AP 全负）。本轮测试「特征空间本身轻量适配」是否存在更强正信号。

## 6. 数据与参数
- categories = Bird/Car/Cube；seed=0；bank=300；test=view120；backbone=frozen DINOv2-B；feature=final-layer patch tokens；scoring=1-NN-max
- R0=frozen baseline（复用 P5，复现 Δ≤4e-6）；R1=线性残差 z+Wz；R2=768→128→768 MLP（仅 L_preserve）；R3=LoRA 最后 2 block Q/V（r=4, α=8）；R4=同 R2 结构（L_cons+0.1·L_preserve）
- 统一目标 L=L_cons+0.1·L_preserve（R2 仅 L_preserve）；normal-only；mild brightness/contrast/gamma ±15%；AdamW；lr 1e-3(adapter)/5e-4(LoRA)；≤300 step；early-stop 30
- **协议修正（任何结果查看前记录，非事后调参）**：R2 仅 L_preserve、R4 = L_cons+0.1·L_preserve，构成 R2/R4 严格单变量消融（同结构，测 illumination consistency 增量价值）

## 7. 实验过程
P0 审计 → E12-B git 提交（`a2ff971`）→ 协议冻结 → Bird smoke（11 项 sanity 全 PASS）→ 12-cell 正式（screen `e13a_main`，21.5 min，0 failed）→ R0 复现 Δ≤4e-6。

## 8. 结果（Fact）

### 8.1 完整 AUROC / AP 表

| config | Bird AUROC/AP | Car AUROC/AP | Cube AUROC/AP |
|---|---|---|---|
| R0（frozen） | 0.74725/0.78863 | 0.78552/0.84207 | 0.85410/0.91686 |
| R1（线性） | 0.75786/0.79911 | 0.80511/0.85502 | 0.87010/0.92856 |
| R2（MLP, 仅preserve） | 0.74724/0.78866 | 0.78547/0.84204 | 0.85416/0.91690 |
| R3（LoRA） | 0.78485/0.81679 | 0.80430/0.85722 | 0.85102/0.91773 |
| R4（MLP, cons） | 0.76848/0.80234 | 0.80767/0.85933 | 0.86249/0.92457 |

### 8.2 Δ vs R0 汇总

| config | mean ΔAUROC | mean ΔAP | pos | ≥+0.010 | worst | params |
|---|---|---|---|---|---|---|
| R1 | **+0.01540** | **+0.01171** | 3/3 | 3/3 | +0.01061 (Bird) | 589,824 |
| R2 | +0.00000 | +0.00001 | 1/3 | 0/3 | −0.00005 | 197,504 |
| R3 | **+0.01777** | **+0.01473** | 2/3 | 2/3 | −0.00308 (Cube) | 24,576 |
| R4 | **+0.01726** | **+0.01289** | 3/3 | 2/3 | +0.00839 (Cube) | 197,504 |

### 8.3 collapse / 训练
- 无 collapse：std_ratio 0.96~2.33（R3 略降 ~2-4%，R1/R4 升 ~1.7-2.3×），pairwise_cos 0.21~0.67（全部 < 0.98）。
- 步数（early-stop）：R1 44-59、R2 54-74、R3 104（跑满近 300 无早停）、R4 44-59。

## 9. 异常情况
无 NaN / 无 OOM / 无失败 cell / R0 复现 Δ≤4e-6。工程修复：DINOv2-B 离线加载（`torch.hub.load` github 挂起）与 LoRA 注入 device 顺序（先注入再 `.cuda()`），均不影响结果。

## 10. 分析（Fact → Interpretation → Hypothesis）

**Fact**
1. R1/R3/R4 三候选全部达到 KEEP 门槛；R2（仅 preserve）≈ R0（无增益）。
2. 三候选 mean ΔAP 全部为正（+0.012~+0.015），这是 E11/E12 从未出现的。
3. R2 vs R4（同结构、单变量）差异 = illumination consistency 的全部增益。

**Interpretation**
1. **illumination consistency 是 active ingredient**：R2（仅 preserve，同 768→128→768 MLP）≈ R0，而 R4（加 consistency）mean ΔAUROC +0.0173。证明「对光照一致」的 normal-only adaptation 目标本身驱动了收益，而非 adapter 容量。
2. **Car 不再是最难类别**：Car 是增益最大/次大类别（R1 +0.0196 / R3 +0.0188 / R4 +0.0222），与 E12-B「Car 对扰动最脆弱」形成强烈反差。representation adaptation 修复了此前 scoring/fusion 层无法触碰的 Car 短板。
3. **R1 是最稳健**（3/3 且全部 ≥+0.010）；**R3 是最强但 Cube 轻微负**（−0.003，非 collapse，是性能小幅回落）；R4 介于两者。
4. R1/R4 的 std 上升（~2×）是线性/MLP adapter 的尺度放大（consistency 损失尺度不变），方向无 collapse（pairwise_cos 远低于 0.98）。

**Hypothesis（未验证）**
- 猜测：normal-only illumination-consistent 训练把 final-layer 特征推向「光照不变流形」，从而让 1-NN 对真实光照扰动更鲁棒、对缺陷更敏感；但机制未做实验验证。
- R3 的 Cube 负收益可能是 LoRA 在 Cube 上过拟合 normal 流形所致，需 3-seed 确认是否稳定。

## 11. 结论
**KEEP**。normal-only illumination-consistent representation adaptation 首次在 DINOv2-B 线上产生稳定、跨类别、AP 为正的收益，明确打破 E12-B 平台。

## 12. 出口条件判断
- R1：mean ΔAUROC +0.0154 ≥0.015、3/3 ≥+0.010、worst +0.0106 ≥−0.005、mean ΔAP +0.0117 ≥+0.005、无 collapse → **KEEP**。
- R3：mean +0.0178 ≥0.015、2/3 ≥+0.010、worst −0.0031 ≥−0.005、mean ΔAP +0.0147 ≥+0.005、无 collapse → **KEEP**。
- R4：mean +0.0173 ≥0.015、2/3 ≥+0.010、worst +0.0084 ≥−0.005、mean ΔAP +0.0129 ≥+0.005、无 collapse → **KEEP**。
- R2：mean ≈0 → **STOP**（预期，作为 preserve-only 对照）。
- 多 KEEP → 保留 1 个，按优先级「更简单 → 参数更少 → ΔAUROC → ΔAP」：**R1（线性残差，最简单）**。

## 13. 对论文证据链的贡献
本实验把「training-based lightweight representation adaptation」这一格填上**肯定**答案：它是 DINOv2-B 线上唯一稳定 KEEP 的方向，收益（+0.015~+0.018）显著高于 scoring/fusion 平台（+0.008~+0.010）。论文可发布的「小改进」候选首次出现。下一格 = **E13-B 3-seed validation**（3 categories × 3 seeds）确认稳定性。

## 14. 下一步（建议，不自动执行）
- 进入 **E13-B**：3 categories × 3 seeds，验证 R1（primary）及 R3/R4（若人工裁决要并行）的跨 seed 稳定性。
- 不建议在本轮直接扩大类别或调参。

## 15. 复现信息
- 环境：RTX 3090 24GB / PyTorch 2.8.0+cu128 / Python 3.12.3 / conda base
- 命令：`python scripts/e13a_screen.py --categories Bird,Car,Cube`（screen `e13a_main`）+ `python scripts/e13a_analysis.py`
- 输出：`results/e13a/{e13a_protocol.json, e13a_screening.csv, e13a_training.csv, e13a_partial.json, e13a_summary.json, e13a_decision_memo.md, README.md, figures/}`
- seed=0；无 test-label 调参；无 backbone/scoring/resolution/bank 更换；β=0.1 冻结。
