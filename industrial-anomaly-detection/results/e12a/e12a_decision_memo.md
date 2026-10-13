# E12-A · Subspace Scoring Screening — Decision Memo

## Verdict: **NO SURVIVOR**（6/6 配置全部 STOP）

## 1. 实验题目
在 Bird / Car / Cube 的真实光照测试协议（view120 / seed0 / 300-bank / frozen DINOv2-B）下，PCA/subspace residual scoring 能否稳定优于当前 frozen 1-NN-max baseline？

## 2. 日期
2026-10-13

## 3. 论文阶段
Frozen-feature module search 已 CLOSED；本轮换轴测试 **scoring paradigm**（PCA/subspace residual vs 1-NN-max）。

## 4. 要验证的核心论断
「保持 frozen DINOv2-B representation 不变，仅把 1-NN-max 换成 subspace-based anomaly scoring，能否取得跨类别稳定正收益？」

## 5. 实验目的
P4（融合/平滑/多尺度）与 P5（prototype/kNN/CLS 融合）共 6 方向 13 配置全部无稳定 survivor，NN 家族小修小补已穷尽。本轮是 scoring 轴的最后一次低成本探测：PCA 重构残差是 NN 之外最主要的免训练 scoring 范式，必须一次测清。

## 6. 数据与参数
- categories = Bird / Car / Cube；seed = 0；bank = 300（E7-1 冻结 bank）；test = view120（700 图，official `image_anomaly` + raw `detectable` 过滤后 653/656/678）
- backbone = frozen DINOv2-B；feature = final-layer patch tokens（`get_intermediate_layers(n=1, norm=True)`，1024×768）
- S0 = 复用 E11-P5 baseline（0 GPU 重跑）；S1/S2/S3 = PCA residual τ∈{0.95, 0.99, 0.995}；AGG-1 = max；AGG-2 = top-1% mean（SubspaceAD TVaR）
- PCA 定义：raw（LayerNorm'd）均值中心化特征 + 协方差 PCA，residual = ‖x−μ−CCᵀ(x−μ)‖₂²（详见 `e12a_protocol_audit.md`）

## 7. 实验过程
- P0 只读审计：确认 baseline（Bird 0.74725 / Car 0.78552 / Cube 0.85410）与数据协议；overlap 审计无重复。
- 文献审计：SubspaceAD（CVPR 2026，arXiv 2602.23013）官方定义 → `e12a_protocol_audit.md`（A 原定义 / B 公平适配）。
- 冻结协议 → `e12a_config.json`。
- smoke sanity check 通过（数据/模型/GPU/PCA/指标全通）。
- 正式运行：3 类 × 6 PCA cell = 18 cells，runtime **4.8 min**，0 failed，baseline 复现 Δ≤4e-6。

## 8. 结果（Fact）

### 8.1 完整 AUROC / AP 表

| config | Bird AUROC / AP | Car AUROC / AP | Cube AUROC / AP |
|---|---|---|---|
| **S0（1-NN-max，复用）** | **0.74725 / 0.78863** | **0.78552 / 0.84207** | **0.85410 / 0.91686** |
| S1-max（τ=0.95） | 0.72393 / 0.75106 | 0.62060 / 0.66919 | 0.85935 / 0.90736 |
| S1-subspace | 0.67443 / 0.68194 | 0.60381 / 0.65544 | 0.87736 / 0.91479 |
| S2-max（τ=0.99） | 0.72840 / 0.75977 | 0.68277 / 0.72707 | 0.87027 / 0.91116 |
| S2-subspace | 0.67778 / 0.68887 | 0.64411 / 0.70019 | 0.87944 / 0.91439 |
| S3-max（τ=0.995） | 0.73819 / 0.76272 | 0.68525 / 0.72899 | 0.87253 / 0.91599 |
| S3-subspace | 0.68299 / 0.68991 | 0.65177 / 0.69934 | **0.88658 / 0.92445** |

### 8.2 Δ vs S0 汇总

| config | Bird ΔAUROC | Car ΔAUROC | Cube ΔAUROC | **mean ΔAUROC** | mean ΔAP | pos | worst |
|---|---|---|---|---|---|---|---|
| S1-max | −0.02332 | −0.16492 | +0.00525 | **−0.06100** | −0.07332 | 1/3 | Car |
| S1-subspace | −0.07282 | −0.18171 | +0.02326 | **−0.07709** | −0.09846 | 1/3 | Car |
| S2-max | −0.01885 | −0.10275 | +0.01617 | **−0.03514** | −0.04985 | 1/3 | Car |
| S2-subspace | −0.06947 | −0.14141 | +0.02534 | **−0.06185** | −0.08137 | 1/3 | Car |
| S3-max | −0.00906 | −0.10027 | +0.01843 | **−0.03030** | −0.04662 | 1/3 | Car |
| S3-subspace | −0.06426 | −0.13375 | +0.03248 | **−0.05518** | −0.07795 | 1/3 | Car |

PCA 主成分数 r（τ=0.95/0.99/0.995）：Bird 233/535/623，Car 204/499/595，Cube 167/460/565。

## 9. 异常情况
- **Baseline 复现 gate 全 PASS**（Bird Δ=+0.000000、Car/Cube Δ=+0.000004）→ PCA 确实拟合在正确的 feature source 上，结果可信。
- 无 NaN / 无 OOM / 无失败 cell / 无重复计算（S0 复用 P5）。

## 10. 分析（Fact → Interpretation → Hypothesis 分离）

**Fact**
1. 6/6 配置 mean ΔAUROC < 0；6/6 mean ΔAP < 0。
2. 所有配置只有 **Cube（1/3）为正**；Bird 全负；**Car 灾难性下降（ΔAUROC −0.10 ~ −0.18，ΔAP −0.11 ~ −0.19）**。
3. Cube 上 subspace（top-1% mean）优于 max（如 S3：+0.0325 vs +0.0184）；Bird/Car 上 max 优于 subspace。
4. 更高 τ（更多主成分、更小残差）在所有类别上单调更优（S3 > S2 > S1）。

**Interpretation**
1. PCA 重构残差在本 frozen DINOv2-B final-layer 特征上，**不构成比 1-NN-max 更好的异常信号**；其收益是 Cube 特异性的。
2. Cube 正 / Bird 负 / Car 崩，与 E5–E11 反复出现的「类别特异翻转」模式一致（不是跨类别稳健信号）。
3. Car 崩溃表明其缺陷信息大量落在 **PCA 主子空间内**（重构残差小），而非残差尾；即 Car 的异常方向与 normal 高方差方向高度重叠，subspace residual 天然失效。

**Hypothesis（未验证，不得当结论）**
- 猜测「residual 是否有效取决于该类缺陷是否落在 normal 低方差方向」，但这只是事后解释，未做任何机制实验验证。
- 本轮未复现 SubspaceAD 完整配方（DINOv2-**G** + Middle-7 层平均 + 30 旋转增强）；因此本负结果**不能外推为「SubspaceAD 无效」**，只证明「在冻结 DINOv2-B final-layer + 300-bank + 无增强 的约束下，PCA residual 劣于 1-NN-max」。

## 11. 结论
在 frozen DINOv2-B 表示不变的公平约束下，PCA/subspace residual scoring 未能稳定优于 1-NN-max：6/6 配置全部 STOP，无 HOLD 候选。

## 12. 出口条件判断
按 §9 冻结规则：所有配置触发 STOP（mean ΔAUROC ≤ 0；仅 1/3 类为正；Car 灾难性下降）。→ **NO SURVIVOR**。按协议禁止自行添加 Mahalanobis / whitening / adapter / MLP。

## 13. 对论文证据链的贡献
本实验使「scoring paradigm 是否值得换轴」这一格完成并给出否定答案：**frozen-feature 下 NN 与 subspace 两大免训练 scoring 范式都无法跨类别稳定超越 1-NN-max**。至此「frozen DINOv2-B 表示 + 免训练 scoring 微调」这一整条轴被关闭（P4 模块 / P5 NN 变体 / E12-A subspace 全灭）。下一格是**人工裁决**：要么接受负面叙事收尾，要么转向**训练式**方向（如 DINOv2 + 可训练 adaptation head，属于新协议、需重新冻结）。

## 14. 下一步（建议，不自动执行）
1. 不放大验证（无 survivor，无候选值得放大）。
2. 不做第二轮 scoring 小海选（协议 §9 已冻结禁止，且 P4/P5/E12-A 已覆盖两大免训练 scoring 家族）。
3. **彻底 STOP subspace 路线**；回到人工裁决：负面叙事 vs 训练式 representation/scoring 方向（需新预注册协议）。

## 15. 复现信息
- 环境：RTX 3090 24GB / PyTorch 2.8.0+cu128 / Python 3.12.3 / conda base
- 命令：`python scripts/e12a_screen.py`（runner）+ `python scripts/e12a_analysis.py`（分析/图）
- 输出：`results/e12a/{e12a_config.json, e12a_protocol_audit.md, e12a_screening.csv, e12a_partial.json, e12a_summary.json, e12a_decision_memo.md, figures/}`
- seed=0；无 test-label 调参；无 backbone/layer 更换。
