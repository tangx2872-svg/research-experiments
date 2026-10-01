# MSC-AD → Experiment 2 下一步开发计划（设计稿，暂不实现）

> 数据确认 + 你拍板研究粒度后，才进入下面 Phase 1 的代码编写。当前不写任何训练/适配代码。

## Experiment Question（一句话）

不同缺陷类型（5 类铸件缺陷）在真实亮度变化（low/mid/high）下，是否对 illumination/style-invariant representation（α-IN）表现出不同敏感性？

## Variables

固定：
- Model：PatchCore（复用已归档的 `FAlphaPatchcore`，一行不改）
- Backbone：wide_resnet50_2（与 Experiment 1 完全一致，保证可比）
- Dataset：MSC-AD（选定 1 种铸件表面，固定 1 档分辨率）

变化：
- alpha ∈ {0, 0.5, 1.0}（第一轮 3 档，不必 5 档；α 的意义见 Experiment 1）
- illumination ∈ {low, mid, high}（真实采集亮度，替代 Experiment 1 合成 brightness）

## Evaluation（不直接比 anomaly score）

沿用 Experiment 1 的教训：跨 α 绝对分数有 IN 尺度混淆，不可直接比较。改用：

### 1. defect–normal separation（同 α、同光照内）
- image AUROC / AUPR
- pixel AUROC / AUPR
- d'（同 α 内 normal 与 defect 的标准化分离度）

### 2. degradation ratio（跨 α）
- 对每个 defect type，比较 α=0 → α=1 的 separation 相对降幅
- 目标：看 5 类缺陷的降幅是否分化（对应 Case B 的 defect-specific 现象）

### 3. illumination robustness
- normal 样本在 low/mid/high 三档下的 score 分布 / 方差
- 观察 α 增大是否压缩"跨光照 normal score 方差"（即 robustness 提升）

## 分阶段

### Phase 1：数据整理
- 输入：MSC-AD 原始包（1.07GB）
- 动作：确认目录结构、缺陷类型名、光照档、分辨率档、各类别数量；转 RGB；重排为 MVTec 目录结构
- 输出：`data/msc_ad/<surface>/{train/good, test/good, test/defect_x, ground_truth/defect_x}`
- 文件：`scripts/prepare_msc_dataset.py`
- 预计难点：官方目录结构未知，需先解包实测；低分辨率档的 resize 伪影

### Phase 2：PatchCore baseline
- 输入：整理好的子集
- 动作：α=0 跑通，校验与 MVTec 上 baseline 行为一致
- 输出：baseline 指标 + 阈值
- 文件：`scripts/msc_baseline.py`（复用 `falpha_patchcore.py` 的 fit）
- 预计难点：灰度→RGB 是否正确；mask 对齐

### Phase 3：α-IN experiment
- 输入：baseline 权重 + 5 类缺陷 + 3 光照档
- 动作：每 α 顺序 fit + 推理，逐图逐 α 逐光照记录 score
- 输出：`results/experiment2_msc/raw_results.csv`（字段对齐 Experiment 1，加 defect_type、illumination、resolution）
- 文件：`scripts/experiment2_msc_illumination.py`
- 预计难点：每 α 独立 default_root_dir（规避 Experiment 1 已踩的 Windows SHFileOperationW 坑）；fit 后模型 CPU→device + memory_bank 显式移动

### Phase 4：Analysis
- 输入：raw_results.csv
- 动作：生成 fig（α vs robustness、α vs 各 defect separation、trade-off、逐样本轨迹）
- 输出：`results/experiment2_msc/figures/` + 结论
- 文件：`scripts/analyze_experiment2_msc.py`
- 预计难点：跨 α 尺度混淆处理（复用 Experiment 1 的同 α 内 d' + degradation ratio 方案）

## 关键前置决策（写代码前必须先定，由你拍板）

1. **研究粒度**：确认走 defect-type 级（MSC-AD 有标签，推荐）——这决定了 Experiment 2 完全替换 CSEM 方向。
2. **光照变量语义**：确认问题聚焦"亮度强度（MSC-AD 覆盖）"而非"光照方向（仅 CSEM 覆盖）"。
3. **train 光照策略**：train normal 是否跨光照混合（测泛化）还是单光照（测同光照检测）。
4. **对象+分辨率**：选定 1 种表面、1 档分辨率（下载后按数据量定）。
5. **下载路径**：优先试 nbsdc 国内镜像（免大学邮箱），失败再走官方邮箱申请。

## 重要限制（本阶段不做的）

- ❌ 不下载后自动训练
- ❌ 不改 PatchCore / 不写新模型 / 不加新 normalization
- ❌ 不做 5 档 α 或 defect×illumination 二维全量（先确认可行性）
- 本阶段唯一目标：确认 MSC-AD 是否值得替换 CSEM 作为 Experiment 2 数据。
