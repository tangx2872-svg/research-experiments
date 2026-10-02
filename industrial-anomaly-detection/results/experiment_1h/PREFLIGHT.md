# Experiment 1H — Preflight Report

**日期**：2026-10-02
**状态**：Preflight 完成，待实现 + smoke test

---

## 1. 当前 feature pipeline（已确认，来自 anomalib 2.6.2 源码 + 实测）

PatchCore（`wide_resnet50_2`, `layers=["layer2","layer3"]`, image 256²）：

```text
input (B,3,256,256)
  → TimmFeatureExtractor(layers=["layer2","layer3"])
      → layer2: (B, 512, 32, 32)
      → layer3: (B, 1024, 16, 16)
  → feature_pooler = AvgPool2d(3, 1, 1)  [逐 layer 独立池化，尺寸不变]
      → layer2: (B, 512, 32, 32)
      → layer3: (B, 1024, 16, 16)
  → generate_embedding(features):
      embeddings = features["layer2"]                      # (B,512,32,32)
      layer3 → bilinear upsample 到 32×32 → (B,1024,32,32)
      concat → F = (B, 1536, 32, 32)
  → [1E α-IN 插在这里：F_alpha = (1-α)F + α·IN(F)]  ← 当前 post_concat 位置
  → reshape_embedding → (B, 1024, 1536)
  → nearest_neighbors vs memory_bank → patch_scores
  → compute_anomaly_score → pred_score
  → anomaly_map
```

**关键事实**（决定了 1H 的实现方式）：
- `TimmFeatureExtractor` 只返回 `self.layers` 指定的层（layer2 + layer3），`forward` 里 `features` dict 只含这两层。
- pooling 是逐 layer 独立的 `AvgPool2d`，在 `generate_embedding` **之前**完成。
- concat 在 `generate_embedding` **内部**完成（layer3 先 upsample 再 concat）。

## 2. 可安全插入 intervention 的位置

| location | 插入点 | tensor shape | 说明 |
|---|---|---|---|
| `layer2` | `generate_embedding` 内部，`features["layer2"]` 取出后、concat 前 | (B,512,32,32) | 只对 layer2 做 α-IN |
| `layer3` | `generate_embedding` 内部，`features["layer3"]` 取出后、upsample 前 | (B,1024,16,16) | 只对 layer3 做 α-IN（在 upsample 前做，与 1E 对 concat 后做 IN 的语义一致：IN 作用于该层的原始统计） |
| `post_concat` | concat 后、reshape 前 | (B,1536,32,32) | **与 1E 完全一致**（reference condition） |

**IN 定义**（与 1E 完全一致）：`F.alpha = (1-α)·F + α·IN(F)`，`IN = F.instance_norm(affine=False)`（逐样本、逐通道、在 H×W 空间维归一化）。α=0 直接返回原始 feature，跳过 IN。

**设计选择说明**：layer2/layer3 的 IN 都在 `generate_embedding` 内、**concat 之前**做。这样：
1. 不影响 `feature_pooler`、`reshape_embedding`、memory bank、NN 等任何 PatchCore 其他逻辑；
2. train(fit) 与 test 走同一个 `forward`，memory bank 与 query 在同一特征空间；
3. IN 只作用于被干预的那一层，另一层完全保持原始统计。

## 3. 是否影响 PatchCore 其他逻辑

**否**。intervention 只改变 `generate_embedding` 返回的 embedding 内容，不改变 shape、不改变 memory bank 构建流程、不改变 NN 距离计算。唯一差异：不同 location 下 IN 作用的空间分辨率/通道数不同（layer2 在 512ch@32²，layer3 在 1024ch@16²，concat 在 1536ch@32²），这正是 1H 要研究的机制差异来源。

## 4. α=0 等价性

三个 location 在 α=0 时都直接返回原始 feature（不调用 IN），因此理论上 bit-wise 等价。smoke test 会做 numerical check：`max|Δ| < 1e-6`。

## 5. 预计显存

沿用 1E 实测：coreset size 22k–40k，peak GPU ~2.8–4.8 GB（`torch.cuda.max_memory_allocated`）。layer2/layer3 intervention 不增加任何持久 tensor，IN 是逐样本确定性变换，显存峰值与 1E 相当，8GB 内安全。**仍沿用 1E 的 OOM 处理策略**：先清模型 + gc + empty_cache，再降 batch。

## 6. 预计运行时间

1E 实测每 (category, seed) 单 α 5 档约 10–20 min。1H 每 (category, seed) 需跑 3 location × 5 α = 15 condition（但 α=0 三个 location 可共享缓存，实际 13 个唯一 condition：3×4 + 1 baseline）。

总规模（5 category × 3 seed × 13 唯一 condition）≈ 195 个 fit。按 1E 平均 15 min/(cat,seed,5α) 估算，约 **8–12 小时**（过夜量级）。α=0 baseline 缓存可省 ~15% 计算。

## 7. 与 1G 的关系（互补不重复）

- **1G**：问「变化怎么传下去」——固定 post_concat α-IN，拆 NN-distance → feature geometry → norm/channel variance 传导链。
- **1H**：问「变化从哪里产生」——固定 concat 前各层，看不同 defect 对 layer2/layer3/concat 的 IN 是否有层级选择性。

## 8. 待确认的风险点

1. `F.instance_norm` 在 layer3 的 (B,1024,16,16) 上做（16×16=256 空间像素），IN 统计量基于 256 个像素估计，比 layer2（1024 像素）噪声更大——这本身就是 1H 可能观察到的「层级敏感性」的一个候选机制，记录在案但不预设结论。
2. post_concat 必须能复现 1E（方向一致），否则 STOP 排查实现差异——这是 sanity check 的硬门槛。

## 9. 协议冻结清单（对齐 1E）

- backbone=wide_resnet50_2, layers=[layer2,layer3], coreset_ratio=0.1, num_neighbors=9, image 256², batch 16
- α grid = [0, 0.25, 0.5, 0.75, 1.0]
- seeds = [0, 1, 2]
- categories = bottle, cable, hazelnut, screw, grid（5 个，与 1E 一致）
- validation split: 每 category 从 train/good 固定 seed 划 20 张
- z 标准化: contemporaneous category×seed×α×location test/good 分布, ddof=1
- Primary 指标: image AUROC, pixel AUROC, defect score distribution (score_mean/score_std)
- Δ response: alpha=1 − alpha=0（per defect × location）
- LSI = |Δ_layer2 − Δ_layer3|（descriptive only）
