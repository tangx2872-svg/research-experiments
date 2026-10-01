# CSEM-MISD Experiment 2 数据准备规划

> 本文件回答任务指令的 Task 2 / Task 3：评估 CSEM-MISD 是否适合 PatchCore，并设计数据子集。
> **当前状态：数据未下载，本文为设计规划，非已执行结果。**

---

## Task 2：CSEM-MISD 是否适合 PatchCore（anomaly detection 适配性评估）

### 目标格式（MVTec AD 风格）

```text
dataset/
├── train/
│   └── good/          # 正常样本（完好件）
├── test/
│   ├── good/          # 正常样本
│   ├── defect_A/      # 缺陷样本
│   └── ...
└── ground_truth/      # 像素级缺陷 mask
    └── defect_A/
```

### 2.1 原始数据是否满足 anomaly detection 设置？

**部分满足，需转换。**

- ✅ **满足**：gear/screw 同时有完好件（35）与缺陷件（35），可以构建"normal 训练 + normal/anomaly 测试"的经典 AD 设置。
- ⚠️ **需转换**：原始数据是"每个 specimen = 108 张不同光照的图"的**多光照栈**结构，不是"每张图独立 + 类别目录"的 MVTec 格式。需要把每个 specimen 的 108 张图**拆开**，并决定"以光照为单位"还是"以 specimen 为单位"来组织。
- ❌ **不满足（致命）**：**无 per-specimen defect type 标签**，无法产生 `test/defect_A/`、`test/defect_B/` 这样的缺陷类别目录。只能得到 `test/defective/`（无类型细分）。

### 2.2 train/test 如何划分？

两个可选粒度：

| 方案 | train（normal memory） | test | 优点 | 缺点 |
|---|---|---|---|---|
| **A. 官方 Train/Test 划分** | 官方 Train 目录中的完好件 | 官方 Test 目录中的完好+缺陷件 | 尊重官方划分，可与其他工作对比 | 需核验官方 Train/Test 内完好/缺陷的分布 |
| **B. specimen 级自划分** | 完好件按 specimen 划分（如 25 train / 10 test） | 剩余完好 + 全部缺陷 | 完全可控 | 需自己确定划分，且 intact 总数仅 35，偏少 |

> ⚠️ 官方对 washer 的描述是 "train 32 / test 38"（washer 是 70 缺陷件），
> 但对 gear/screw 的 "35 defective + 35 intact" 如何落入 Train/Test 目录**尚未核验**。
> **数据下载后第一件事：统计 Train/Test/Unannotated 各目录内的完好件/缺陷件数量。**

### 2.3 normal image 是否足够？

- **按 specimen 计**：gear/screw 各 35 个完好件，若每个完好件取 108 张图，则 normal 图最多 35×108 = 3780 张。
- **按 illumination 计**：若"每个光照方向单独建 normal memory"（candidate 设计），则每个光照只有 35 张 normal 图（再 train/test 拆分后更少），**对 PatchCore 的 coreset 采样偏少**。
- 结论：normal 样本量**按 specimen 建全局 memory 足够，按 illumination 单独建 memory 偏少**。这直接决定了候选实验设计（见 plan 文档）。

### 2.4 defect 是否可以作为 test anomaly？

**可以**，但有两个限制：
1. 缺陷图数量 = 35 个 specimen × 108 光照 = 3780 张（若不按光照缩减）。
2. 缺陷 mask 是每 specimen 静态的，可用于 pixel 评估；但**无 defect type 细分**。

### 2.5 illumination 是否可以作为额外实验变量？

**是，这是该数据集的核心价值。** 108 个真实光照方向可作为实验变量，研究：
- 同一 specimen 的缺陷在不同光照下的可检测性变化（defect visibility）；
- normal 样本在不同光照下的分数稳定性（illumination robustness）。

---

## Task 3：Experiment 2 数据子集推荐

### 推荐方案

```text
Object:
  gear（首选）或 screw（备选）

Defects:
  无 defect type 标签 → 无法按 scratch/notch/hole 分组
  只能按 specimen 分组（specimen-level），或按"缺陷可见性"事后分组（需人工观察）

Illumination:
  从 108 个光照中选一个子集，而非全用（见下方理由）
  例如：全部 9 个仰角 × 每个仰角取 1–2 个方位角，或按研究问题聚焦
```

### 选择理由

1. **选 gear 而非 washer**：gear/screw 才有完好件（35），washer 只有缺陷件，无法做 AD。gear 缺陷最稀疏（0.2%），最能暴露"缺陷可见性 vs 光照"的关系。
2. **illumination 子集而非全 108**：108 个光照 × 35 specimen × 5 α 的计算量与存储很大（108×35=3780 张/类 × 多 α 重复 fit）。第一轮建议先选有代表性的光照子集（如高/中/低仰角 × 固定方位角），验证现象后再扩展。
3. **specimen 级而非 defect-type 级**：受"无类型标签"硬约束，当前研究粒度只能是 specimen 级，除非后续人工标注 defect type。

### ⚠️ 与原始研究目标的冲突（必须正视）

用户研究目标原为：
> "不同 defect 是否对 illumination/style invariant representation 有不同敏感性？"

但 CSEM-MISD **没有 defect type 标签**，无法直接回答"不同 defect type"层面的问题。
因此存在两个可能路径（**需用户决策**，见 plan 文档 Task 4 的 decision 部分）：

- **路径 1（specimen 级）**：把问题降级为"不同 **specimen** 对 illumination invariant representation 的敏感性是否不同"，或"缺陷**可见性**（而非类型）如何影响敏感性"。可直接用现有数据。
- **路径 2（defect 级）**：需要额外人工标注 defect type（把 35 个缺陷件按 scratch/notch/hole 分类），或改用有 defect type 标签的其它多光照数据集。

**建议先走路径 1（specimen 级）验证现象是否存在，同时评估路径 2 的人工标注成本。**
