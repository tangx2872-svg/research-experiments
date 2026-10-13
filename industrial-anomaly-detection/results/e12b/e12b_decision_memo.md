# E12-B · Lightweight Illumination-Adaptive Multi-Level Fusion Screening — Decision Memo

## Verdict: **HOLD**（3/3 候选均为 HOLD，保留 1 个最简单候选 **S2**）

## 1. 实验题目
在 Bird / Car / Cube 的真实光照测试协议（view120 / seed0 / 300-bank / frozen DINOv2-B / 1-NN-max）下，normal-only 训练的 lightweight learnable fusion gate（S2/S3/S4）能否把 P4-A1 block9+final 的弱正收益转化成可发表的小改进？

## 2. 日期
2026-10-13

## 3. 论文阶段
representation-side lightweight adaptation（frozen-feature scoring 搜索 CLOSED 后的新轴）。

## 4. 要验证的核心论断
「在 frozen DINOv2-B 表示上，用 normal-only illumination-adaptive 融合 gate 学习 block9 与 final 的权重，能否稳定超过 final-only baseline，并超过固定 concat fusion（S1）？」

## 5. 实验目的
P4-A1 block9+final 是 frozen-feature 搜索中最有价值的残余正信号（3/3 AUROC positive，mean +0.0078，但 AP 不理想）。本轮测试唯一新轴：**可学习融合 gate**（禁止改 backbone/scoring/bank/test protocol）。

## 6. 数据与参数
- categories = Bird / Car / Cube；seed = 0；bank = 300（E7-1 冻结 bank）；test = view120（700 图，official `image_anomaly` + raw `detectable`）
- backbone = frozen DINOv2-B（无梯度）；features = block9(l9) + final(l11) patch tokens（1024×768，LayerNorm）
- scoring = 固定 1-NN-max（与 E11-P4/P5 一致）
- candidates：S0(final-only, 复用) / S1(concat, 复用) / S2(scalar gate, 1 param) / S3(channel gate, 768) / S4(SE gate, 73874)
- normal-only 训练：240 train + 60 val bank normals，mild 光照增强 {brightness, contrast, gamma}×{0.9, 1.1} = 6 变体；L = L_consistency + 0.1·L_identity；Adam；≤500 step，patience 20
- 协议冻结见 `e12b_protocol.json`（HEAD 495a6c8b）

## 7. 实验过程
- P0 只读审计（git/GPU/disk/资产/等价性）→ 冻结协议 → smoke（Bird，7 项 sanity 全 PASS）→ 正式 9 cell，16.1 min，0 failed。
- DINO forward 离线加载（绕过 `torch.hub.load` 的 github 网络挂起），特征缓存于 RAM，gate 反向传播不经过 DINO。

## 8. 结果（Fact）

### 8.1 完整 AUROC / AP 表（view120 / seed0）

| config | Bird AUROC / AP | Car AUROC / AP | Cube AUROC / AP |
|---|---|---|---|
| **S0（final-only，复用）** | 0.74725 / 0.78863 | 0.78552 / 0.84207 | 0.85410 / 0.91686 |
| **S1（concat，复用）** | 0.75629 / 0.78730 | 0.79442 / 0.82817 | 0.85963 / 0.91915 |
| S2（scalar gate） | 0.75739 / 0.79161 | 0.79999 / 0.83385 | 0.85887 / 0.91936 |
| S3（channel gate） | 0.75434 / 0.78090 | 0.80128 / 0.83761 | 0.86190 / 0.92072 |
| S4（SE gate） | 0.75470 / 0.78755 | 0.80034 / 0.83242 | 0.85750 / 0.91828 |

### 8.2 Δ vs S0 汇总

| config | Bird ΔAUROC | Car ΔAUROC | Cube ΔAUROC | **mean ΔAUROC** | mean ΔAP | pos | worst |
|---|---|---|---|---|---|---|---|
| S1 | +0.00904 | +0.00890 | +0.00553 | +0.00782 | -0.00431 | 3/3 | Cube |
| S2 | +0.01014 | +0.01447 | +0.00477 | **+0.00979** | **-0.00091** | 3/3 | Cube |
| S3 | +0.00709 | +0.01576 | +0.00780 | **+0.01022** | -0.00278 | 3/3 | Bird |
| S4 | +0.00745 | +0.01482 | +0.00340 | **+0.00856** | -0.00310 | 3/3 | Cube |

### 8.3 learned gate 权重（init = 0.5，偏 block9）
- S2 λ(w_block9)：Bird 0.613 / Car 0.592 / Cube 0.566
- S3 λ_mean：Bird 0.578 / Car 0.548 / Cube 0.533（λ 分布 0.11–0.90）
- S4 w_block9_mean：Bird 0.700 / Car 0.637 / Cube 0.589

### 8.4 illumination consistency（1−cos，越低越好）
| category | final-only(参考) | 固定等权(before) | S2 after | S3 after | S4 after |
|---|---|---|---|---|---|
| Bird | 0.0647 | 0.0466 | 0.0443 | 0.0383 | 0.0418 |
| Car | 0.0427 | 0.0310 | 0.0291 | 0.0255 | 0.0282 |
| Cube | 0.0274 | 0.0199 | 0.0197 | 0.0179 | 0.0193 |

### 8.5 无 collapse
std_fused before/after 均 ≈ 0.023–0.025（非 0）；loss 全 finite；9/9 cell 训练 loss_finite=True。

## 9. 异常情况
- 无 NaN / 无 OOM / 无失败 cell / S0 复现 Δ=0、S1 复现 Δ≤4e-6（feature source 正确）。
- 工程坑（已修复，纯工程）：`torch.hub.load` 在 github 不可达时挂起 → 改为离线加载本地缓存 repo+checkpoint，不影响模型/权重/结果。

## 10. 分析（Fact → Interpretation → Hypothesis 分离）

**Fact**
1. 3 个 learned gate 对 S0 均 3/3 AUROC positive（mean +0.0086 ~ +0.0102）。
2. 3 个 learned gate 的 mean ΔAP 全为负（−0.0009 ~ −0.0031）；S2 最接近 0。
3. 所有 gate 学到的 block9 权重 > 0.5（偏 block9），且 S4 偏得最多（0.59–0.70）。
4. 融合（等权/learned）的 illumination consistency 显著低于 final-only（0.020–0.065 → 0.018–0.047），learned 相对等权再小幅下降（0.002–0.008）。

**Interpretation**
1. learnable fusion 的 AUROC 收益 ≈ 复现了 S1（固定 concat）的水平：S2 mean +0.00979 vs S1 +0.00782（仅 +0.0020，未达 complexity gate 的 +0.003）；S3 虽 mean ΔAUROC 略超 +0.010，但 mean ΔAP −0.0028 < 0，无法通过 KEEP 的「mean ΔAP ≥ 0」。
2. AUROC 收益**不传导到 AP**：与 P4-A1 的现象一致（AUROC 微升、AP 微降），这是该类 block9+final 融合的固有特征，learnable 化没有改变。
3. illumination consistency 目标与 AUROC **不对齐**：S3 达到最低 consistency 但 AUROC 最高、AP 却最差；S2 一致性改善最少但 AP 最不劣化。说明「让融合对光照更一致」不能机械地转化为「更好的异常检测」。
4. 类别翻转模式仍在（Bird/Car/Cube 增益分布不均），未出现跨类别稳健、AP 为正的信号。

**Hypothesis（未验证，不得当结论）**
- 猜测 block9 携带对光照更稳定、对缺陷仍敏感的局部信息，gate 学到的轻微 block9 倾斜反映了这一点；但这一倾斜幅度太小，不足以在 image 级 AUROC/AP 上产生可发表增益。
- 本轮负结果**不能外推**为「illumination-adaptive 训练式方向整体无效」，只证明「在 frozen DINOv2-B + 1-NN-max + 只训练 1 个轻量融合 gate 的约束下，不构成相对 fixed concat 的可发表改进」。

## 11. 结论
learnable fusion gate 能复现 S1（固定 concat）的 AUROC 收益，并略微超过 S1（+0.0020），但未通过 KEEP（mean ΔAP < 0），也未通过相对 S1 的 complexity gate（需 +0.003 AUROC 或 +0.010 AP）。**全部候选 HOLD。**

## 12. 出口条件判断
- KEEP 条件：mean ΔAUROC ≥ +0.010 且 pos ≥ 2/3 且 worst ≥ −0.010 且 mean ΔAP ≥ 0。→ **无候选满足**（S2 mean ΔAUROC +0.0098<0.010 且 ΔAP<0；S3 ΔAP<0；S4 ΔAUROC+0.0086<0.010）。
- HOLD 条件：mean ΔAUROC ≥ +0.005 且 3/3 ≥ 0 且 mean ΔAP ≥ −0.005。→ **S2/S3/S4 全部满足**。
- 按「有 HOLD 最多保留 1 个」，保留**最简单 = S2（1 参数）**。

## 13. 对论文证据链的贡献
本实验把「frozen DINOv2-B 表示 + lightweight learnable fusion」这一格填上否定性答案：**learnable 融合没有超越固定 concat 融合（S1）的可发表收益**（AUROC 略升但 AP 仍负、幅度在噪声内）。至此 frozen-feature 侧的 scoring 微调（P4/P5/E12-A）与 representation 侧轻量 adaptation（E12-B）两条子轴都被关闭。下一格是**人工裁决**：接受负面叙事收尾，或转向**训练式** representation/scoring 方向（需新预注册协议、重新冻结 baseline）。

## 14. 下一步（建议，不自动执行）
- **不放大验证**（HOLD，无 KEEP 候选）。
- 保留 S2（1 参数 scalar gate）作为唯一 HOLD 候选，供人工裁决是否值得 1 次 3-seed 小验证；按协议本轮不直接跑。
- 若裁决「不值得」，则关闭 adaptive fusion 轴，转向 training-based representation axis（新协议）。

## 15. 复现信息
- 环境：RTX 3090 24GB / PyTorch 2.8.0+cu128 / Python 3.12.3 / conda base
- 命令：`python scripts/e12b_screen.py --categories Bird,Car,Cube`（runner）+ `python scripts/e12b_consistency.py` + `python scripts/e12b_analysis.py`
- 输出：`results/e12b/{e12b_protocol.json, e12b_screening.csv, e12b_training.csv, e12b_partial.json, e12b_consistency.json, e12b_summary.json, e12b_decision_memo.md, figures/}`
- seed=0；无 test-label 调参；无 backbone/layer/scoring 更换；α=0.1 冻结。
