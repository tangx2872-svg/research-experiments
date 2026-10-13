# E12-A · SubspaceAD 文献实现审计（read-only）

来源：
- 论文：*SubspaceAD: Training-Free Few-Shot Anomaly Detection via Subspace Modeling*（arXiv:2602.23013，CVPR 2026）
- 全文：https://arxiv.org/html/2602.23013（v3，2026-05-13）
- 官方代码：https://github.com/CLendering/SubspaceAD（许可证 CC BY 4.0；本服务器 `github.com` 不可达，未逐行核对源码，以下来自论文全文）

---

## A. 原论文定义

### A1. 提取哪一层 / 哪类 token
- 骨干：**冻结 DINOv2-G**（约 1.1B 参数，特征维 D=1536）。
- Token：**patch tokens**（非 CLS），每个 patch 对应输入 14×14 区域。
- 层选择：**中间第 22–28 层**（论文称 “Middle-7”），对所选层的 patch token 做**平均池化**：

  $$x_p = \frac{1}{|\mathcal L|}\sum_{l \in \mathcal L} f_l(p),\quad \mathcal L = \{22,\dots,28\}$$

### A2. PCA / subspace 如何拟合 normal features
- 建模对象：从 k 张正常图像及其**旋转增强视图**提取的全部 patch 特征集合 $X_{\text{normal}}$。
  - 每张正常图像额外生成 $N_a=30$ 个旋转视图（0°–345°）；对方向敏感类别（transistor）不用旋转。
  - PCA 拟合图像总数 = $k \times (1+30)$。
- 中心化：计算经验均值 μ 与协方差 Σ；**确定性 PCA，均值中心化**；**未对特征做额外标准化/L2 归一化**（基于协方差的 PCA）。
- 子空间模型：$x = \mu + C z + \epsilon$，$C \in \mathbb R^{D \times r}$ 为 Σ 的前 r 个特征向量正交基。
- 主成分数 r：**不手工指定**，按**解释方差阈值 τ** 决定——取最小 r 使 $\sum_{i=1}^{r}\lambda_i \ge \tau \sum_{i=1}^{D}\lambda_i$；论文设定 **τ = 0.99**。
  - 论文指出 τ=1.00 时性能大幅下降 ⇒ 异常主要位于残差子空间。

### A3. anomaly residual 如何计算
- 子空间投影：$x_{\text{proj}} = \mu + C C^\top (x_p - \mu)$。
- patch 级异常分数：$S(x_p) = \|x_p - x_{\text{proj}}\|_2^2$（**平方重构残差**）。
- 该分数构成低分辨率异常图 $M \in \mathbb R^{h \times w}$。

### A4. image-level 分数如何从 patch 分数聚合
- **TVaR（尾部风险价值）**：取异常图 M 中分数最高的**前 ρ% patch** 的平均分：

  $$s_{\text{img}} = \operatorname{mean}(H_\rho(M)),\quad \rho = 1\%$$

- 即：图像分数 = **得分最高的 1% patch 的平均值**（top-region 聚合）。
- 像素级（本轮不做）：patch 图双线性上采样 + σ=4 高斯平滑；AUROC/PRO 用未归一化原始分数。

### A5. few-shot 设置
- k ∈ {1, 2, 4}（1/2/4-shot），每配置 5 个随机种子，报告均值±标准差；仅用正常无缺陷图像拟合 PCA。
- 另有 batched 0-shot / full-shot 变体（本轮无关）。

---

## B. 本轮为了公平比较而做的适配（必须与 §4 冻结协议一致）

> 目标不是复现 SubspaceAD，而是：**在 E11 冻结的 DINOv2-B + 1-NN-max baseline 上，只把 scoring 换成 PCA/subspace residual**，其余不变。以下每一项都来自「禁止换 backbone/layer/resolution/token」的硬约束。

| # | 原论文 | 本轮（E12-A） | 原因 |
|---|---|---|---|
| B1 | DINOv2-G（1536-d） | **DINOv2-B（768-d，frozen）** | §4 冻结「backbone = frozen DINOv2-B」，禁止换 backbone |
| B2 | Middle-7（22–28 层）平均 | **final layer（n=1）patch tokens** | §4「feature source 与 E11 baseline 完全一致」；baseline 用 `get_intermediate_layers(n=1, norm=True)` |
| B3 | 旋转增强 ×30（few-shot 扩充） | **不加任何增强** | §4「与 baseline 相同的 normal bank」；bank 就是冻结的 300 张正常图 |
| B4 | k∈{1,2,4} few-shot | **bank = 300 张正常图**（E7-1 seed0 冻结 bank） | §4 冻结 `bank_size=300` |
| B5 | τ=0.99 | **τ ∈ {0.95(S1), 0.99(S2), 0.995(S3)}** | §5 候选矩阵冻结 3 个 variance |
| B6 | PCA 作用于 raw（均值中心化）特征 | **同左：raw（LayerNorm'd final-layer）均值中心化特征，协方差 PCA，无 L2 归一化** | 遵循 SubspaceAD 官方定义；baseline 的 L2 归一化是 cosine 1-NN 特有的，PCA 前不做（否则改变子空间几何） |
| B7 | residual = 平方 L2 | **同左：$S(x)=\|x-x_{\text{proj}}\|_2^2$** | 遵循官方；平方/开方对 AUROC/AP（rank-based）等价，取官方更规范的平方形式 |
| B8 | 聚合 = TVaR top-1% | **AGG-1 = MAX（与 1-NN-max 公平对照）；AGG-2 = top-1% mean（官方 TVaR）** | §6 冻结：每配置只比 MAX 与 SUBSPACE 两种聚合 |
| B9 | 像素级 AUROC/PRO | **本轮不做**（只做 image-level AUROC/AP） | §8 指标范围 |

### B 中两个关键口径的明确记录

1. **PCA 特征 = raw（LayerNorm'd）final-layer patch tokens，不 L2 归一化。**
   - baseline 的 cosine 1-NN 需要 L2 归一化；PCA 子空间建模按官方定义在 raw 均值中心化特征上进行。
   - 两者共享同一 feature source（DINOv2-B final layer / 448 输入 / patch token / `norm=True`），**没有**换 backbone/layer/resolution/token。
2. **AGG-2 的 top-1% 定义**：$k = \max(1, \lceil 0.01 \times N_{\text{patch}}\rceil)$，$N_{\text{patch}}=1024$ → **k = 11**。论文原文是「最高分 1% patch 的平均」，未给出 1024 网格下精确取整；此处固定 ceil 取整并冻结（10 与 11 对 image-level AUROC/AP 的影响属噪声级，不改变 GO/STOP 结论）。

### 禁止改动（§15 复述）
不改 test view、不改 bank=300、不用 test label 调参、不换 backbone、不换 layer、不加类别、不加第 4 种 aggregation、不调参救火、不开 trainable adapter、不重跑 baseline。

---

## 重叠审计（overlap audit）

- **在 E11 及所有历史脚本中检索 `PCA residual / reconstruction error / subspace anomaly score / top-region aggregation`：**
  - E11（P1–P5）脚本均无 PCA/subspace scoring；`results/e11/` 无任何 PCA 产物。
  - 历史「subspace」相关仅 **Exp13-P INSS**（`scripts/experiment13p_*.py`）：那是 **ResNet/PatchCore 上投影掉 illumination-nuisance 子空间**（目的=去光照方向），作用于 MVTec 5 类，已判「非 defect 特异」→ STOP。**与本轮「DINOv2-B 上 PCA 重构残差作为异常分数」目的、表示、数据集均不同，非重复。**
  - **Exp8B**（channel 级 D/I 可分性）用的是通道统计量，非 PCA 残差。
  - 结论：**无实质重复**；本轮无需重跑历史，直接复用 P5 baseline 数字。
