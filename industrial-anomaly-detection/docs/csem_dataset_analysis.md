# CSEM-MISD Dataset Analysis

> **⚠️ 重要声明：本地尚未下载数据。**
>
> 本文档的数据结构描述基于以下两个权威来源，**非本地实测**：
> 1. 官方 Zenodo record 7410513 说明 + 论文（Honzátko et al., 2021, Machine Vision and Applications）；
> 2. 官方仓库 `DawyD/illumination-preserving-rotations` 的 `data/dataloader.py` 源码
>    （已归档于 `experiments/2026-09-30_illumination_sensitivity_exploration/csem_misd_illumination_sensitivity/reference/dataloader.py`）。
>
> 图片数量、缺陷类别、normal/anomaly 划分等**实际值必须在数据下载并解压后重新核验**，
> 本文件届时需更新为实测值。当前 `data/csem_misd/` 目录为空。

---

## Dataset Path

- 目标本地路径：`data/csem_misd/`（当前为空）
- 官方来源：Zenodo record 7410513（`gear.tar.gz` 约 6.8 GB，md5 `9a896591ff69fc0d64ee93458e6ad924`）
- 论文：Honzátko, Türetken, Bigdeli, Dunbar, Fua. *Defect segmentation for multi-illumination quality control systems*. Machine Vision and Applications, 2021.
- 官方代码：https://github.com/DawyD/illumination-preserving-rotations
- 许可：CC BY-NC-ND 4.0（非商业、禁止演绎）

## Directory Structure

```text
<base_path>/                        # 例如 gear 目录
├── Train/                          # 官方划分的训练样本
│   ├── gear_001/                   # 每个 specimen 一个目录
│   │   ├── gear_001_101.png        # 108 张灰度图，每张一个光照方向
│   │   ├── gear_001_102.png
│   │   ├── ...                     # 共 108 张
│   │   ├── gear_001_segmentation.png   # 前景 mask（自动提取，每 specimen 一个）
│   │   └── gear_001_mask.png            # 缺陷 mask（手工标注，有 = 缺陷件）
│   └── ...
├── Test/                           # 官方划分的测试样本
│   └── ...（同上结构）
├── Unannotated/                    # 数百个未标注样本
│   └── ...
├── light_vectors.csv               # 108×3 光照方向向量（按 illuminationNr 字典序）
└── light_intensities.csv           # 108×1 光强（0–127）
    （官方代码用名 `light_currents.csv`，Zenodo 说明用 `light_intensities.csv`，以实际为准）
```

## Categories

| 对象 | 缺陷件 | 完好件 | 未标注 | 缺陷像素占比 | 可用状态 |
|---|---:|---:|---:|---:|---|
| washer | 70 | 0 | 无 | ~1.4% | 最早发布；**无完好件** |
| gear | 35 | 35 | 数百 | ~0.2% | 已发布 |
| screw | 35 | 35 | 数百 | ~0.8% | 已发布 |

## Defect Types

- 论文描述了三类缺陷：**scratch（划痕）、notch（缺口）、hole（孔洞）**。
- 三类缺陷的"光照可见性"差异很大：notch/hole 在多数光照下可见（强度/纹理变化），
  scratch 只在少数光照方向可见。
- **关键限制：数据集不提供 per-specimen 的 defect type 标签。**
  官方 `dataloader.py` 中没有任何缺陷类型字段，只有二值的缺陷 mask（`_mask.png` 有无）。
  因此**无法按 scratch/notch/hole 分组做 defect-type 级分析**，只能做 specimen 级分析。

## Illumination Information

- 半球形 light-dome，过滤环境光，**108 个真实光照方向**（非合成）。
- 每 12 个光照共享同一仰角；同一仰角内相邻方位角差 30°。
- 光照编号 illuminationNr = 3 位：**首位 = 仰角索引（1 最高 … 9 最低），后两位 = 方位角索引（01–12）** → 9×12 = 108。
- 同一 illuminationNr 在不同 specimen 之间严格对应同一物理光照位置。
- `light_vectors.csv`（108×3 方向向量）与 `light_intensities.csv`（0–127 光强）按 illuminationNr 字典序索引。
- 已知局限：方位角未经严格标定，可能偏差几度；金属镜面反射可能饱和相机传感器。

## Ground Truth / Mask

- **前景 mask**：`{prefix}_{sample_dir}_segmentation.png`，每 specimen 一个，自动提取。
- **缺陷 mask**：`{prefix}_{sample_dir}_mask.png`，每 specimen 一个，手工标注；
  存在即表示该 specimen 有缺陷（二值）。
- 缺陷 mask 是**静态的（每 specimen 一个）**，与光照无关，因此可用于任意 illumination 下的 pixel 评估。
- 提供 **pixel-level mask**，但不提供 defect type / 实例级标签。

## Suitable Experiment Subset

（详见 `docs/csem_exp2_plan.md` 的 Task 3 部分）

**推荐对象：gear（或 screw）**，而非 washer，理由：
1. gear/screw 同时有 35 缺陷 + 35 完好，满足 anomaly detection 的 normal/anomaly 双类需求；
   washer 只有 70 缺陷件、无完好件，无法构建 normal memory。
2. gear 缺陷最稀疏（0.2%），最能体现"缺陷可见性"对光照的敏感性（研究目标聚焦点）。

## Problems and Notes

1. **数据未下载**：gear.tar.gz 6.8GB，Zenodo 当前网络 ~10–27 KB/s（约 45–67 小时），曾因连接中断失败（`curl:18`）。需换网络/代理。
2. **无 defect type 标签**：这是最大的研究适配障碍——无法直接做"不同 defect 对 illumination invariant representation 的差异化敏感性"（该问题要求按 defect type 分组）。**当前只能做 specimen 级**。
3. **normal 样本量偏少**：若"每个 illumination 单独建 normal memory"，则每光照仅 ~35 张完好图（train/test 再分后更少），PatchCore memory 偏小。
4. **镜面反射混淆**：金属镜面反射随光照大幅变化，anomaly score 可能混入"反射差异"而非"缺陷可见性"。
5. **类严重不平衡**：缺陷像素占比 0.2%–1.4%。
6. **标注主观**：缺陷边界标注不精确。
7. **许可限制**：CC BY-NC-ND（非商业、禁止演绎），仅限研究。
