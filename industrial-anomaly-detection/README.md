# Industrial Anomaly Detection

工业视觉异常检测学习与实验项目。

## 实验归档

- [2026-10-02｜Experiment 1G: Feature-Space Representation Reshaping](experiments/exp1g_feature_space/README.md)：双轨 memory-bank 分解（matched Mα vs frozen M0）追踪 α-IN 在 PatchCore 计算链中的传导，9 个代表 defect × 5 α × seed0。结论：**CASE_A — REPRESENTATION_CHAIN_SUPPORTED**——1E Δdefect_std 与 defect patch NN-distance dispersion 强对应（matched ρ=+0.833 p=0.005；frozen ρ=+0.950 p<0.001），且 frozen track 更强 → 机制主要在 defect 特征表示自身漂移；L2 经 feature norm-std 通过（ρ=+0.700 p=0.036）；L3 channel-var 仅弱 hint。诚实记录：bottle/contamination 的 matched/frozen 方向分歧（bank 端反向推回）；background 存在非镜像的全局分量（expand +4.4 / shrink -1.0）。下一步：seed 1/2 复核 + 1H layer/channel 机制。

- [2026-10-02｜Experiment 1F: Offline Defect-Attribute Mechanism Screening](experiments/exp1f_mechanism_screening/README.md)：完全复用 1E，从原图+GT mask 提取 4 组预注册视觉属性（size/contrast/frequency/morphology），检验它们能否解释 25 个 defect type 的 response heterogeneity。结论：**CASE_D — NO_EXPLAINABLE_STRUCTURE**——全部单属性 |rho|≤0.24 且 CI 跨零，shrink/expand 组属性画像不可区分，双变量组合最好 adj R²=0.141（对比 identity 饱和 1.0）；频率族存在方向一致的弱 hint（laplacian partial rho=-0.361, p=0.076，LOCO/dilation 稳定）。机制不在图像空间简单属性里，若继续应转向 representation-space 分析。附带技术记录：FFT HF-ratio 对自然图像无区分度（1/f 谱数学必然），改用梯度能量比。

- [2026-10-02｜Experiment 1E: Formal Cross-Category Pilot](experiments/exp1e_cross_category/README.md)：5 类别（bottle/grid/cable/screw/hazelnut）× 3 seeds × 5 α = 75 conditions 跨类别验证（bottle 复用 1C 重建，equivalence check 9/9 PASS）。结论：**CASE_A — CROSS_CATEGORY_HETEROGENEITY_SUPPORTED**。25/25 defect types 有 3/3 seed 稳定响应；5/5 类别内部存在方向分化；1D 的"方差收缩→Δd' 反升"模式在 cable/bent_wire 与 hazelnut/print 跨类别复现；area-only 解释力接近零（R²≤0.079），heterogeneity 载体在 defect identity 而非 size。
- [2026-10-02｜Experiment 1D: Size-Controlled Defect Sensitivity Analysis](experiments/exp1d_size_control/README.md)：控制缺陷面积后的敏感性分析（完全离线，复用 1B/1C 逐样本结果）。结论：**CASE_A**——控制 area 后 type 仍提供显著额外解释力（ΔR²=0.188，area-only R² 仅 0.115）；matched-area 下 broken_large vs contamination 仍差 Δz -2.23；contamination 残差 +1.11 与 broken_large -1.10 方向相反。附加 metric decomposition 发现：α-IN 对三类缺陷方差的作用方向不同（large 增方差/small 微缩/contamination 强缩 -2.3），这是 Δz（sample-level）与 d'（group-level）方向分歧的根源。
- [2026-10-02｜Experiment 1C: Multi-Seed Stability Validation](experiments/exp1c_multiseed/README.md)：多 seed（0/1/2）稳定性验证，确认 1B 的 defect-specific α-IN response 非 coreset 随机性产物。broken_large Δd'=-5.87±0.39（三 seed 全部大幅下降）、broken_small -0.30±0.24、contamination +0.50±0.09（全部反向上升），判定 **CONTINUE** → 进入 1D size confound analysis。
- [2026-10-01｜Experiment 1B: Defect-Specific α Sensitivity Screening](results/experiment_1b/README.md)：在 MVTec AD bottle + PatchCore 上，仅研究内部 α-IN representation probe 对不同缺陷类型的差异化影响（不引入合成光照）。结论：三类缺陷的 α-response 曲线明显分化——broken_large 分离度 d' 13.63→7.32（-46%）、broken_small 稳定、contamination 反向上升（+10%）；方向与初始假设相反。size confound 部分存在（type-only R²=0.292 > area-only R²=0.130），属 A+C 混合结论。
- [2026-10-01｜Experiment 1: Synthetic Illumination × α-IN 机制筛查](results/experiment1_illumination_tradeoff/README.md)：PatchCore 特征混合 InstanceNorm（α-IN）在简单光度扰动下的机制筛查。结论：robustness 提升与 pixel-level sensitivity cost 的 trade-off 苗头存在，无 defect-type 分化；仅 synthetic 证据，待真实光照数据验证。
- [2026-09-30｜工业异常检测光照敏感性探索](experiments/2026-09-30_illumination_sensitivity_exploration/README.md)：包含 F_alpha/PatchCore、MVTec AD 2 与 CSEM-MISD 调查；未找到明确可推进方向，暂时搁置。代码、说明和已有结果已集中归档。


本项目用于研究生阶段探索 **工业视觉与视觉异常检测（Industrial Anomaly Detection）** 方向。

通过公开工业数据集、经典异常检测算法和小型对比实验，逐步完成：

- 基础知识学习
- Baseline 模型复现
- 实验结果分析
- 模型对比与消融实验
- 小型改进实验
- 论文方向探索

当前以 **MVTec AD + PatchCore** 作为第一个完整 Baseline，优先建立一套能够重复运行、记录和分析的工业异常检测实验流程。

---

## Environment

当前实验环境：

- Windows 11
- Python 3.11
- PyTorch 2.11.0 + CUDA 12.8
- NVIDIA GeForce RTX 5060 Laptop GPU
- Anomalib 2.6.2
- OpenCV
- Jupyter Notebook
- VS Code

Conda 环境：

```bash
conda activate industrial-ad
```

当前 GPU 已能够被 PyTorch 正常识别并用于 Anomalib 实验。

---

## Project Structure

```text
industrial-anomaly-detection/
├── data/                         # 数据集（不上传 Git）
│   └── mvtec_ad/
│
├── experiments/                  # 正式模型实验
│
├── notebooks/                    # 学习、分析与 Baseline Notebook
│   └── 01_patchcore_bottle.ipynb
│
├── results/                      # 实验结果、可视化图片、模型权重等
│   └── patchcore_bottle_broken_large.png
│
└── scripts/                      # 通用训练、测试和数据处理脚本
```

其中：

- `data/`：存放 MVTec AD 等实验数据，不提交至 Git；
- `notebooks/`：保存模型学习、Baseline 复现和结果分析过程；
- `results/`：保存正式实验结果和可视化图片；
- `experiments/`：后续用于更加规范的模型对比实验；
- `scripts/`：后续逐步抽离可重复使用的训练、测试和数据处理代码。

---

## Dataset

当前使用：

### MVTec AD

MVTec AD 是工业视觉异常检测常用公开数据集，包含多种工业物体和纹理类别。

当前首先使用：

```text
MVTec AD
└── Bottle
```

Bottle 实验数据：

- 训练集：209 张正常图片
- 测试集：83 张图片
- 训练阶段仅使用正常样本
- 测试集包含正常样本和异常样本
- 异常样本提供像素级 Ground Truth Mask

当前策略是：

> **先跑通单个类别 → 理解完整实验流程 → 再扩展到其他类别和其他模型。**

---

## Models

计划逐步实验以下工业异常检测方法：

- [x] PatchCore
- [ ] PaDiM
- [ ] EfficientAD
- [ ] 其他近期工业异常检测方法

当前第一个重点 Baseline：

### PatchCore

当前配置：

```text
Backbone: Wide ResNet-50-2
Feature Layers: layer2 + layer3
Coreset Sampling Ratio: 0.1
```

PatchCore 的基本思路是：

```text
正常训练图片
        ↓
预训练 CNN 提取局部特征
        ↓
layer2 + layer3 多尺度特征
        ↓
Coreset Sampling
        ↓
建立正常特征 Memory Bank
        ↓
测试图片特征
        ↓
与正常特征进行最近邻比较
        ↓
异常分数
        ↓
Anomaly Map
        ↓
Predicted Mask
```

与传统监督目标检测不同，PatchCore 不需要提前收集大量不同类型的缺陷样本。

它主要学习：

> **正常产品的局部特征应该是什么样子。**

测试时，如果某个区域的特征与正常 Memory Bank 中的特征差异较大，则该区域会获得更高的异常分数。

---

## Research Roadmap

当前研究路线：

- [x] 1. 搭建工业异常检测实验环境
- [x] 2. 熟悉 MVTec AD 数据集
- [x] 3. 跑通 Anomalib Baseline
- [x] 4. 初步理解 PatchCore 原理
- [ ] 5. 系统分析 Image-level / Pixel-level 评价指标
- [ ] 6. 在更多 MVTec AD 类别上进行实验
- [ ] 7. 对比不同异常检测 Baseline
- [ ] 8. 进行参数对比与消融实验
- [ ] 9. 尝试小型模型改进
- [ ] 10. 探索可形成论文的具体研究问题

当前原则：

> **先复现，再理解；先做小实验，再决定具体改进方向。**

---

# Experiment Log

## 2026-09-24｜环境搭建与异常检测入门

### 已完成

- [x] 创建 `industrial-ad` Conda 环境
- [x] 配置 PyTorch + CUDA
- [x] RTX 5060 Laptop GPU 测试成功
- [x] 安装 OpenCV / Jupyter
- [x] 安装 Anomalib 2.6.2
- [x] 创建工业异常检测实验目录
- [x] 明确使用 MVTec AD 作为第一阶段实验数据集
- [x] 初步理解工业异常检测与普通监督分类 / YOLO 检测的区别
- [x] 理解重建式异常检测的基本思想
- [x] 明确当前主线优先放在工业视觉异常检测

### 当日里程碑

> **完成工业异常检测实验环境搭建，并建立对工业异常检测任务的基本认识。**

---

## 2026-09-25｜PatchCore Bottle Baseline

### 已完成

#### 1. 数据集

- [x] 加载 MVTec AD Bottle 数据集
- [x] 确认训练集包含 209 张正常图片
- [x] 确认测试集包含 83 张图片
- [x] 理解训练集、测试集和 Ground Truth Mask 的作用

#### 2. PatchCore 模型

- [x] 确定 PatchCore 作为第一个重点 Baseline
- [x] 使用 `wide_resnet50_2` 作为 Backbone
- [x] 使用 `layer2 + layer3` 提取局部特征
- [x] 设置 `coreset_sampling_ratio=0.1`
- [x] 初步理解 PatchCore 的特征式异常检测思路

#### 3. Memory Bank

- [x] 使用正常 Bottle 图片执行 PatchCore Fit
- [x] 完成正常样本局部特征提取
- [x] 完成 Coreset Sampling
- [x] 建立正常特征 Memory Bank
- [x] 理解 PatchCore Fit 与普通神经网络训练的区别

#### 4. Test / Predict

- [x] 完成 Bottle 测试集 Test
- [x] 完成测试图片 Predict
- [x] 获取 Anomalib `ImageBatch` 预测结果
- [x] 学习读取以下模型输出：
  - `image`
  - `gt_mask`
  - `anomaly_map`
  - `pred_mask`
  - `pred_score`
  - `pred_label`

#### 5. 可视化

- [x] 选择 `broken_large` 异常样本
- [x] 完成 Original Image 可视化
- [x] 完成 Ground Truth Mask 可视化
- [x] 完成 PatchCore Anomaly Map 可视化
- [x] 完成 Predicted Mask 可视化
- [x] 完成图像反归一化，恢复正常 RGB 显示
- [x] 将四联图保存至 `results/`

实验结果文件：

```text
results/patchcore_bottle_broken_large.png
```

#### 6. Notebook 整理

- [x] 清理重复的 Fit / Predict / 可视化代码
- [x] 按完整实验流程重新组织 Notebook
- [x] 添加实验目的和方法说明
- [x] 添加各阶段 Markdown 实验记录
- [x] 添加实验结果与总结

当前 Notebook：

```text
notebooks/01_patchcore_bottle.ipynb
```

Notebook 当前结构：

```text
实验说明
   ↓
1. 实验环境与依赖
   ↓
2. MVTec AD Bottle 数据集
   ↓
3. PatchCore 模型
   ↓
4. Engine
   ↓
5. Fit：建立 Memory Bank
   ↓
6. Test：模型性能评估
   ↓
7. Predict：生成预测结果
   ↓
选择 broken_large 样本
   ↓
8. 异常检测结果可视化
   ↓
9. 实验结果与总结
```

### 当日里程碑

> **完成第一个 PatchCore + MVTec AD Bottle Baseline 的完整实验闭环。**

已经从：

```text
了解工业异常检测
```

推进到：

```text
数据集
→ 模型
→ Fit
→ Memory Bank
→ Test
→ Predict
→ Anomaly Map
→ Predicted Mask
→ 实验结果保存
```

---

## 2026-10-01｜Experiment 1: Synthetic Illumination × α-IN 机制筛查

### 实验性质（边界声明）

**Synthetic Illumination Mechanism Screening / Sanity Check**。

brightness / gamma 等简单数字变换**不能**模拟真实工业光照（specular reflection / highlight / shadow / local contrast / material response / defect visibility / illumination direction 均未覆盖）。因此本实验只能回答：

> 在简单 photometric perturbation 下，α-IN 是否表现出值得进一步验证的现象？

**不能**回答"在真实工厂光照变化下该方法是否有效"。任何结论只能表述为 "Synthetic illumination screening suggests..."。

### 启动前记录（四句话，预注册）

**① 我怀疑什么？**
增强特征归一化（α-IN）可能减少 PatchCore 对简单 photometric variation 的敏感性，使正常产品在亮度/曝光/gamma 改变后不易被误判为异常；但过强归一化也可能削弱真实缺陷相关的外观/纹理/局部对比度信息，且这种损失可能具有 defect-type dependence（structural defects 如 broken_* 相对稳定，appearance-related 如 contamination 更敏感）。**仅为待验证假设。**

**② 准备干什么？**
MVTec AD bottle + PatchCore，固定除 α 外全部条件，`F_alpha = (1-α)F + α·IN(F)`（InstanceNorm affine=False，复用 2026-09-30 归档实现），α ∈ {0, 0.25, 0.5, 0.75, 1.0}。A 组：正常 test 图 + 简单 synthetic photometric perturbation；B 组：原始真实 defect 图（不做人为光照修改）。

**③ 看什么结果？**
Robustness side：perturbation 后正常图 anomaly score 升高多少，随 α 是否减弱；Sensitivity side：各 defect type 的 score / 检测性能随 α 的变化。保留 defect-type level 结果，不只看 overall AUROC。

**④ 什么结果意味着继续？**
若 α 增大同时出现"perturbation 导致的正常样本响应下降"与"真实 defect 响应/检测能力稳定下降"，则存在值得用真实光照数据验证的 trade-off 苗头；若 α 对 photometric perturbation 无帮助或 defect sensitivity 完全不变或趋势混乱，则停止围绕 α 调参。

### 实验设置

- **Dataset**: MVTec AD Bottle（train 209 / test good 20, broken_large 20, broken_small 22, contamination 21）
- **Model**: PatchCore，backbone `wide_resnet50_2`，layers layer2+layer3，coreset_sampling_ratio=0.1，num_neighbors=9，seed=0（与归档 baseline 相同）
- **α**: 0, 0.25, 0.5, 0.75, 1.0（每 α 独立 fit 一次）
- **Synthetic perturbations**（只施加于正常 test 图）：
  - `brightness 0.7 / 1.3`：线性缩放后 clamp 到 [0,1]（brightness 1.3 时约 37% 像素被裁剪饱和）
  - `gamma 0.7 / 1.3`：output = input^(1/γ)，即 **gamma 0.7 变暗、gamma 1.3 变亮**（已实测确认：mean 0.54→0.47 / 0.54→0.60，无裁剪）
  - `original`：原图
- **评分**：原始 PatchCore pred_score（max-NN 距离）。跨 α 的绝对分数**不可直接比较**（IN 改变特征/分数尺度），跨 α 只比较：同 α 内配对差值、秩相关指标（AUROC/AUPR）、同 α 内 defect-normal 分离度。
- **阈值**：α=0 正常图 original 的 max score = 28.173（recall 全部 100%，饱和，不具区分力）。

### 结果

**α=0 baseline 校验：通过。** image AUROC=1.0000、pixel AUROC=0.98557、pixel F1=0.72704、pixel AUPR=0.77112，与 2026-09-30 归档 alpha=0 结果完全一致。

**Robustness side（正常图，配对 delta = perturbed − original，跨 α 可比）：**

| condition | α=0 | α=0.25 | α=0.5 | α=0.75 | α=1.0 |
|---|---:|---:|---:|---:|---:|
| brightness 0.7（变暗） | +1.28 | +0.81 | +0.80 | +0.86 | +1.04（U 型回升） |
| brightness 1.3（变亮，37% 裁剪） | +1.18 | +0.95 | +0.73 | +0.63 | +0.65（单调下降） |
| gamma 0.7（变暗） | +1.28 | +0.81 | +0.78 | +0.70 | +0.70（单调下降） |
| gamma 1.3（变亮） | +0.31 | +0.22 | +0.17 | +0.11 | +0.01（趋近 0，α=1 时 Wilcoxon p=0.31 不显著） |

α=0 时所有扰动都显著推高正常图分数（Wilcoxon p<1e-4，18-19/20 样本为正），即 baseline 确实受光度扰动影响；变暗类扰动（约 +1.28）远强于平滑变亮（+0.31）。

**Sensitivity side（真实 defect 图，无扰动）：**

| 指标 | α=0 | α=0.25 | α=0.5 | α=0.75 | α=1.0 |
|---|---:|---:|---:|---:|---:|
| image AUROC | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| pixel AUROC | 0.98557 | 0.98541 | 0.98481 | 0.98376 | 0.98218 |
| pixel AUPR | 0.77112 | 0.77044 | 0.76287 | 0.74929 | 0.73129 |
| 分离度 d'（broken_large） | 17.6 | 15.1 | 14.0 | 13.2 | 11.5 |
| 分离度 d'（broken_small） | 16.3 | 14.6 | 14.0 | 13.7 | 12.4 |
| 分离度 d'（contamination） | 14.8 | 13.2 | 12.7 | 12.6 | 11.5 |

原始 defect 绝对分数呈 U 型（先降后升），但这是分数尺度混淆：α=1 时正常图 original 分数本身从 24.7 涨到 29.3（整体分数膨胀）。同 α 内的分离度 d' 与秩相关 pixel 指标才是跨 α 可比的量。

**六问回答：**

- **Q1** synthetic photometric shift 是否让 baseline normal score 上升？——**是**，α=0 下四种扰动均显著推高（+0.31 ~ +1.28）。
- **Q2** α 增大后 photometric sensitivity 是否下降？——**4 个条件中 3 个单调下降**（brightness 1.3、gamma 0.7、gamma 1.3；gamma 1.3 在 α=1 几乎被完全吸收）；**brightness 0.7 例外**，呈 U 型（α=1 回升到 +1.04）。
- **Q3** α 增大后真实 defect sensitivity 是否下降？——**image-level 无变化**（AUROC 饱和于 1.0）；**pixel-level 单调轻度下降**（pixel AUROC 0.9856→0.9822，pixel AUPR 0.7711→0.7313）；同 α 内 defect-normal 分离度 d' 单调下降（broken_large 17.6→11.5 最明显）。
- **Q4** defect type 趋势是否不同？——**没有明显分化**：三类 defect 的 d' 下降幅度接近（-3.3 ~ -6.1），未出现"contamination 独降、broken 稳定"的 defect-specific 行为。
- **Q5** 是否出现 robustness↑ + defect sensitivity↓ 的稳定趋势？——**苗头存在但温和**：robustness 改善（3/4 条件）伴随 pixel-level 指标单调轻度下降；image-level 检测在该单类别饱和设置下无可见代价。
- **Q6** 是否由极少数样本驱动？——**不是**：各条件下 13-19/20 样本 delta 为正，top-3 样本只贡献总正增量的 25-46%，且逐样本轨迹（fig4）显示 α=1 的分数上移是普遍模式。

### Observations

- α-IN 对平滑光度变换（gamma 类）的吸收效果最好；对带裁剪饱和的 brightness 1.3 和变暗类 brightness 0.7 仍有残余敏感性，后者在 α=1 回升，说明 IN 并未消除所有光度敏感成分。
- 变暗类扰动（+1.28）远强于平滑变亮（+0.31）：扰动强度本身不对称。
- α=1 出现整体分数膨胀（normal original 24.7→29.3），三类 defect 同步 U 型回升——这是跨 α 比较绝对分数时的主要混淆，已在分析中用同 α 内配对差值和秩指标规避。
- 本次未出现 OOM；RTX 5060 Laptop 8GB 顺序执行 5 个 α 正常完成（每 α 约 1.5 分钟）。
- 实现问题：Anomalib 在 Windows 下复用同一 default_root_dir 时第二次 fit 会因版本目录清理失败报 `SHFileOperationW 0x2`，已改为每 α 独立目录规避。

### Conclusion（Case 判定）

**属于 Case A（trade-off 苗头存在，需真实光照验证），附带两点保留**：(1) defect-side cost 主要体现在 pixel-level 指标与同 α 分离度上，image-level 在当前饱和设置下无可见变化；(2) 未观察到 defect-specific 分化，Case B 不成立。依据当前结果**不支持**继续围绕 α 精细调参（如 0.1/0.2/0.3 扫描）。

### Next Step

- 本探针的第一轮低成本筛查已完成，**停止扩展 synthetic 实验**（不做 defect × illumination 二维实验、不加扰动种类）。
- 若继续该方向：需要真实 multi-illumination 数据验证 trade-off 是否在真实光照下存在。此前调查的两个来源受阻（CSEM-MISD 下载失败；MVTec AD 2 光照划分不满足需求），需先解决数据问题再决定是否重启。
- 若不解决数据问题：当前探针暂停，与 2026-09-30 归档状态一致。

### 文件

- 脚本：`scripts/experiment1_illumination_tradeoff.py`（主实验）、`scripts/analyze_experiment1.py`（汇总+图）、`scripts/analyze_experiment1_supplementary.py`（Q6+分离度）
- 结果：`results/experiment1_illumination_tradeoff/`（raw_results.csv 815 条、summary_results.csv、metrics.json、analysis_summary.json、supplementary_analysis.json、figures/fig1-fig4）
- 复用实现：`experiments/2026-09-30_illumination_sensitivity_exploration/falpha_patchcore/falpha_patchcore.py`（未修改）

---

## 2026-10-01｜Experiment 1B: Defect-Specific α Sensitivity Screening

### 实验性质（边界声明）

**正式 Mini Experiment**，目的不是证明想法正确，而是快速、诚实地判断「不同 defect type 对 α-IN representation probe 是否存在稳定差异」。

- 本实验**不做任何 synthetic illumination perturbation**（无 brightness/gamma/exposure/shadow）。
- 只研究**内部 representation probe**：α-IN 对不同 defect type 的影响。
- α 只是「原始特征 F 与 IN 特征 F_IN 的线性混合权重」，**不是**「去除了多少光照信息」，也**不是**提出的创新方法。

### 预注册四句话（实验前记录）

- **① 怀疑什么？** 当 α 增大时，不同 defect type 的异常表示可能不同步变化：结构明显的大缺陷（broken_large）可能相对稳定，而较小、弱对比度、外观/纹理相关缺陷（contamination）可能更敏感。
- **② 干什么？** 在 MVTec AD bottle + 现有 PatchCore + α-IN 上，只改变 α ∈ {0, 0.25, 0.5, 0.75, 1.0}，其余条件不变，测试原始 good / broken_large / broken_small / contamination。
- **③ 看什么？** 三类 defect 随 α 的 image-level AUROC、Recall/TPR、raw score distribution、individual trajectory、d' 分离度，以及 pixel-level AUROC、anomaly map、defect area 分析。
- **④ 什么结果意味着继续？** 若不同 defect type 在多个 α 下出现稳定、明显、可重复、非少数样本导致的不同响应曲线，则继续；否则如实报告、暂停。

### 实验设置

- **Dataset**：MVTec AD bottle（train/good 209 → 划 20 作 validation、189 进 memory bank；test/good 20、broken_large 20、broken_small 22、contamination 21）
- **Model**：PatchCore，backbone `wide_resnet50_2`，layer2+layer3，coreset_sampling_ratio=0.1，num_neighbors=9，input_size 256×256
- **IN 位置**：`generate_embedding` concat 后、reshape 前；affine=False；α=0 直接返回原始 feature（bit-wise 一致）
- **Seed**：0（固定 python/numpy/torch CPU+CUDA 随机种子）
- **阈值规则**：`tau_alpha = max(validation normal scores)`（最保守，验证集 FPR=0），三类 defect 共享，不用 test 数据/defect 标签调阈值

### 结果（六问回答）

- **Q1 α=0 是否复现 baseline？** —— **是**。α=0 直接返回原始 feature（代码层 bit-wise 一致），test/good 分数（mean 24.7，range 21.5-28.2）与 Experiment 1 原始 PatchCore 分数量级完全一致。
- **Q2 是否出现 defect-specific 分化？** —— **是，且方向与假设相反**。同 α 内标准化分离度 d'：

| alpha | broken_large | broken_small | contamination |
|---|---:|---:|---:|
| 0 | **13.63** | 8.31 | 3.88 |
| 0.25 | 11.46 | 8.01 | 3.91 |
| 0.5 | 10.31 | 8.16 | 4.10 |
| 0.75 | 8.55 | 7.97 | 4.18 |
| 1.0 | **7.32** | 7.73 | **4.27** |

  - **broken_large**：d' 单调大幅下降 13.63→7.32（-46%，几乎腰斩）
  - **broken_small**：基本稳定 8.31→7.73（-7%）
  - **contamination**：反而缓慢上升 3.88→4.27（+10%）

  初始假设（contamination 最敏感）被推翻，实际最敏感的是 broken_large。

- **Q3 是否由少数样本驱动？** —— **否**。broken_large 的下降在 16/20 样本出现、contamination 的上升在 16/21 样本出现；类别均值曲线与个体曲线形态一致（普遍 U 型）。
- **Q4 是否 size confound？** —— **部分，但不完全**。broken_large 类内 corr(area, delta)=-0.747、broken_small=-0.725（面积越大下降越多，size 效应真实存在）；但 contamination 面积中等（0.085 > small 的 0.031）却反向上升，无法用 size 解释。回归 type-only R²=0.292 > area-only R²=0.130，defect type 是更强解释变量。
- **Q5 是否只是 scale 改变？** —— **否**。good mean 24.7→29.3（+19%）而 broken_large 63.9→58.5（-8%），方向相反，d' 变化无法用单一 scale 因子解释。
- **Q6 heatmap 是否一致？** —— **一致**。三类缺陷定位在所有 α 下保持准确（与 GT 重合，无漂移），变化在响应强度/范围。

### 如实记录的问题

- image-level AUROC 在所有 α 下全部饱和于 1.0，无区分度；分化只能靠 score-level 分离度 d' 观察。
- 预注册阈值规则导致 test/good FPR=1.0：validation（train 分布）分数系统性低于独立 test/good（val max ~20.4 < test min ~21.5），是「train-derived normal 与 test normal 存在分布 gap」的真实统计发现，非 bug，但使 Recall/FPR 失去区分度。
- 单类别、单 seed、每类 n=20-22，样本量小。

### Conclusion（Case 判定）

**A 与 C 之间，偏向 A（defect-specific difference 明确存在），附 C 限定**：三类 d' 曲线稳定分化（单调降 / 平稳 / 缓慢升）、跨样本普遍、阈值无关、可复现；但 broken_large vs broken_small 的差异部分可由 defect area 解释，contamination 的上升则是真正 type-specific 现象。

### Next Step（供决策，不自动执行）

1. **多 seed 验证**（seeds={0,1,2}）确认 d' 分化非 coreset 随机性——成本低，优先。
2. **跨类别验证**（cable/screw 等 defect 更多样的类别）检验分化是否普遍。
3. **size-controlled / matched-area 分析**彻底剥离 size confound。
4. 若多 seed + 跨类别稳定，则「defect-specific representation sensitivity」可作为正式研究问题推进；若 seed 敏感，按判据 D 暂停。

### 文件

- 脚本：`scripts/experiment1b_defect_sensitivity.py`（主实验）、`scripts/analyze_experiment1b.py`（汇总+图）
- 结果：`results/experiment_1b/`（README、config/、summary/、figures/、heatmaps/、raw/all_sample_scores.csv；`raw/anomaly_maps.npz` 约 89MB 不上传 Git）
- 数据集侦察：`docs/msc_dataset_analysis.md`、`docs/csem_dataset_analysis.md`、`docs/msc_download_guide.md`、`datasets/MSC-AD/access.md`（MSC-AD / CSEM-MISD / BGA 多光照数据集调查）

---

## 2026-10-02｜Experiment 1C: Multi-Seed Stability Validation

### 实验性质（边界声明）

**稳定性验证实验**。目的不是发现新结果，而是给 Experiment 1B 的 defect-specific α-IN response「办身份证」：排除 PatchCore coreset sampling / feature randomness 导致的偶然现象。

- 保持 1B **所有实验条件不变**（dataset/model/α-IN/阈值规则/split 规则），唯一允许变化的是 random seed。
- 复用 1B 的 `run_screening`（import 复用，实验逻辑零改动），仅按 seed 分结果目录。

### 预注册四句话（实验前记录）

- **① 怀疑什么？** 1B 观察到的 broken_large 大幅下降 / broken_small 稳定 / contamination 反向，可能只是 seed=0 的随机波动，需要排除 coreset sampling 随机性。
- **② 干什么？** 固定除 seed 外全部条件，seed ∈ {0, 1, 2}，重复 1B 全流程（每 seed 独立 validation split + 每 α 独立 fit）。
- **③ 看什么？** 每 seed × alpha × defect 的 d'；Δd' = d'(α=1) − d'(α=0) 的跨 seed mean/std；三类 defect 曲线是否跨 seed 保持形态。
- **④ 什么结果意味着继续？** broken_large 三个 seed 全部明显下降、broken_small 变化小、contamination 方向不同且 std 小 → 稳定存在，进入 1D；若 seed 间方向混乱 → 1B 可能是随机产物，重新设计。

### 实验设置

- **Dataset**：MVTec AD bottle（每 seed：209 train/good → 20 validation + 189 bank；test 20 good + 20/22/21 defect）
- **Model**：PatchCore，wide_resnet50_2，layer2+layer3，coreset 0.1，num_neighbors 9
- **Seeds**：0, 1, 2（固定 python/numpy/torch CPU+CUDA）
- **α**：0, 0.25, 0.5, 0.75, 1.0（每 α 独立 fit）
- **阈值规则**：tau_alpha = max(validation normal scores)，与 1B 一致

### Smoke 复现校验

seed=0 smoke（每类 5 张）的 α=0 阈值 = 20.433，与 1B 全量 validation max ~20.4 一致；seed=0 的 validation split 与 1B 完全相同。三个 seed 的 validation split 各不相同，确认多 seed 真实覆盖 split + coreset 两层随机性。

### 结果

**d' 曲线（三 seed 形态几乎重合，Figure 1）：**

| alpha | broken_large (s0/s1/s2) | broken_small (s0/s1/s2) | contamination (s0/s1/s2) |
|---:|---|---|---|
| 0 | 13.63 / 13.16 / 12.87 | 8.31 / 8.09 / 8.07 | 3.88 / 3.71 / 3.78 |
| 1.0 | 7.32 / 7.45 / 7.30 | 7.73 / 7.89 / 7.95 | 4.27 / 4.27 / 4.32 |

**Δd' = d'(α=1) − d'(α=0) 跨 seed 统计（sample std, ddof=1）：**

| defect | seed0 | seed1 | seed2 | mean | std |
|---|---:|---:|---:|---:|---:|
| broken_large | -6.31 | -5.71 | -5.57 | **-5.87** | **0.39** |
| broken_small | -0.58 | -0.20 | -0.12 | -0.30 | 0.24 |
| contamination | +0.40 | +0.56 | +0.54 | **+0.50** | **0.09** |

### 分析要点

- **现象跨 seed 高度稳定**：broken_large 三个 seed Δd' 全部 < -5.5，std 0.39（约效应量 7%）；contamination 三个 seed 全部为正，std 0.09。
- **方向完全一致**：不存在「seed0 降 / seed1 升 / seed2 平」的混乱模式；三类相对排序（large 降 >> small 平 >> cont 升）在所有 seed 中不变。
- **整条曲线逐 α 对齐**：不仅端点稳定，α-response 曲线在三 seed 间几乎重合，说明 α 的作用是确定性系统效应，随机性只带来 ±0.4 内的水平抖动。
- **效应量层级**：|Δd'(large)| ≈ 12 × |Δd'(cont)|，两组置信区间完全不重叠。

### Conclusion（Case 判定）

**CONTINUE —— defect-specific α sensitivity 稳定存在，不是 coreset/feature 随机性的偶然产物。** 预注册判断标准四条全部通过：

1. broken_large 三 seed 全部明显下降 ✅（-6.31 / -5.71 / -5.57）
2. broken_small 变化较小 ✅（|Δd'| ≤ 0.58）
3. contamination 方向不同或不下降 ✅（全部为正）
4. broken_large std 较小 ✅（0.39 << 1.0）

### Next Step

按协议进入 **Experiment 1D: Size Confound Analysis**：matched-area 比较彻底剥离 size confound，回答「控制面积后 type 效应是否仍存在」（1B 已知类内 corr(area, delta) ≈ -0.73~-0.75，size 效应真实存在）。

### 文件

- 驱动脚本：`scripts/experiment1c_multiseed.py`（seed 循环 + 目录隔离）
- 分析脚本：`scripts/analyze_experiment1c.py`（d' 表、Δd' 统计、Figure 1/2、verdict）
- 目录：`experiments/exp1c_multiseed/`（configs/seed_{0,1,2}.yaml + README）
- 结果：`results/experiment_1c/seed_{0,1,2}/` + `summary/`（verdict.json 等）+ `figures/`（dprime_curves_per_seed.png、delta_dprime_stability.png）；`raw/`（含大体积 npz）与 `logs/` 不上传 Git

---

## 2026-10-02｜Experiment 1D: Size-Controlled Defect Sensitivity Analysis

**解释实验**（非模型优化）。完整复用 1B/1C 逐样本 score 与 GT area，零 fit/predict，零模型/口径修改。核心问题：控制 defect area 后，defect type 是否仍能解释 α-IN sensitivity 的差异？

### 预注册口径（冻结，执行中未更改）

- normal reference = test/good 20 张，每 (seed, alpha) 独立 μ/σ
- PRIMARY（sample-level）：`Δz = z(α=1) - z(α=0)`，`z = (score - μ_good) / σ_good`
- SECONDARY：`slope_z`；GROUP-LEVEL CONTEXT：1B/1C 的 d'（不作单样本 sensitivity）
- 主分析 per-image 三 seed mean；禁止把 63×3 当 189 独立样本
- area = GT defect pixels / total pixels；matched-area caliper = 全样本 area 的 MAD（0.03744，看结果前锁定）

### 数据审计

63 样本（20/22/21），缺失 mask=0、area=0 defect=0、重复=0、三 seed test samples 完全一致、三对 type 面积 common support 充足（overlap 0.553/0.780/1.000）。无需重新运行任何模型部分。

### Metric decomposition sanity check（执行中追加的诊断）

Δz 与 1C Δd' 方向分歧（如 contamination Δz<0 而 Δd'>0）不是 bug：二者回答不同问题。分解（3-seed mean，α=0→1）显示 **defect variance 响应是类型分化的真正载体**：

| type | Δmean_gap | Δstd_good | Δstd_defect | Δmean_z | Δd' |
|---|---:|---:|---:|---:|---:|
| broken_large | -8.44 | +0.17 | **+1.73** | -4.56 | -5.72 |
| broken_small | -3.40 | +0.17 | **-0.50** | -2.44 | -0.30 |
| contamination | -2.44 | +0.17 | **-2.30** | -1.95 | **+0.48** |

α-IN 同时作用于 mean separation / within-defect variance / normal variance 三个分量，且对不同 type 的作用结构不同。d' 混合三者；Δz 只反映相对 contemporaneous good 分布的标准化距离。"敏感性"不是单一统计量可完全描述的——记录为实验发现。

### 六问回答

- **Q1 area 分布**：large(mean .117) > cont(.085) > small(.031)，但三对均有 common support。
- **Q2 area-Δz 关系**：type-dependent——两类 broken 类内强负相关（-0.76/-0.75），contamination 类内**弱正相关**（+0.21）。
- **Q3 area-only**：R²=0.115，解释有限。
- **Q4 +type**：R²→0.304，**ΔR²(type|area)=0.188**；反向 ΔR²(area|type)=0.025。type 不可被 area 替代，area 大部分被 type 吸收。
- **Q5 matched-area**：large vs contamination 15 对，同面积 Δz 仍差 **-2.23**；残差 large **-1.10** / small -0.06 / cont **+1.11**，控制面积后 type 分化清晰。
- **Q6 contamination**：面积匹配后仍整体高于 matched broken（-2.24 vs -3.18，28 对）；类内正相关与 broken 类反向，无法由 area 单独解释。注意 Δz 口径下其 Δz mean=-1.95（也下降，只是显著慢于 broken 类）；1C 中 d' 上升是 group-level 方差收缩驱动的现象。

### Multi-seed robustness

各 seed 独立重复主要分析：overall corr -0.33/-0.34/-0.35，type 均值排序三 seed 完全一致（large 最负、cont 最不负）。方向不依赖单一 seed。

### 判定：CASE_A — TYPE EFFECT REMAINS

控制 area 后 type 仍提供显著额外解释力；contamination 的响应模式无法由 size 单独解释。限定：d' 的 group-level 上升部分来自 within-defect variance 收缩，准确表述为 **defect type 影响 α-IN 对缺陷分布的完整作用结构（mean + variance）**。

### 局限

n=63 单类别；large-small 匹配仅 5 对（不可靠）；Δz 依赖 n=20 good 估计 μ/σ；matched-area 是观察性控制；单 backbone/detector 外推性未知。

### Next Step

**Experiment 1E — Cross-Category Validation**（cable/screw 等），检验 type effect 与 decomposition 结构是否跨类别成立。未自动开始，等待确认。

### 文件

- 脚本：`scripts/analyze_experiment1d_size_control.py`、`scripts/metric_decomposition_check.py`
- 目录：`experiments/exp1d_size_control/`（README + 全部口径/规则/结果记录）
- 结果：`results/experiment_1d/summary/`（audit/statistics/correlation/regression/matched_pairs/metric_decomposition/verdict）、`figures/`（6 张）、`tables/sample_level_response.csv`

---

## 2026-10-02｜Experiment 1E: Formal Cross-Category Pilot

**解释实验**（非模型优化）。验证 1B–1D 在 Bottle 上发现的 defect-dependent α-IN response heterogeneity 是否跨 MVTec AD 类别成立。预注册协议驱动，配置冻结后零修改。

### 设计（冻结于 results/experiment_1e/config.json）

- 5 categories（bottle/grid/cable/screw/hazelnut）× 3 seeds × 5 α = **75 conditions**；bottle 复用 1C raw + 1B area 重建（不重训），新增 60 fits 串行运行
- α-IN / backbone / coreset / z / d' 口径与 1B/1C/1D 完全一致；z 用 contemporaneous category×seed×α test/good（ddof=1）
- 每 category 完成后自动 sanity checkpoint（completeness/count/finite/good normalization/discovery/area/重复错位）——4 类全部 PASS，无 OOM 无中断
- 资源实测：coreset 22.9k–40.0k patches，peak GPU 2.8–4.8 GB（`max_memory_allocated()` 实测）

### Bottle 重建 equivalence check

9/9 PASS（3 defect types × 3 seeds 的 Δz/Δdefect_std/Δd' 与 1D 确认值完全一致，阈值 1e-6）。

### 核心结果

**Seed stability**：25/25 defect types 至少一个 response 维度 3/3 seeds 符号一致——现象普遍稳定，非随机产物。方向分化是关键：

- **Δdefect_std**：13 POS vs 11 NEG vs 1 MIXED——方差响应强烈类型分化
- 5/5 类别内部出现方向分化（cable/hazelnut 在 3 个维度分化；grid 全类型同向但幅度差 10 倍）

**方差收缩模式跨类别复现**（Δz<0 & Δdefect_std<0 & Δd'>0，全部 3/3 稳定）：

| pattern | Δz | Δstd | Δd' |
|---|---:|---:|---:|
| bottle/contamination | -1.95 | -2.30 | +0.48 |
| cable/bent_wire | -1.07 | -3.51 | +2.19 |
| hazelnut/print | -4.63 | -2.77 | +0.70 |

1D 发现的"α-IN 收缩 defect 方差 → sample z 降但 group d' 反升"模式有两个新增类别的正式确认实例（hazelnut print 即 smoke test 预测的正式验证）。

**Area control**（Phase 15，25 types）：

| response | A: log(area) | B: +category |
|---|---:|---:|
| Δz | R²=0.015 | R²=0.692 |
| Δdefect_std | R²=0.079 | R²=0.796 |
| Δd' | R²=0.052 | R²=0.246 |

area-only 解释力接近零 → **area contributes but is NOT sufficient**；异质性载体在 defect identity（Model C 饱和 R²=1.0 为 one-hot 饱和拟合，仅作方向参考）。

### 判定：CASE_A — CROSS_CATEGORY_HETEROGENEITY_SUPPORTED

A1（≥2 新增类别方向分化：cable/screw/hazelnut）+ A2（稳定覆盖 25/25）+ A3（area 不充分）+ A4（方差收缩模式复现 3 例）全部满足。

### 边界与局限

不做机制归因（texture/structure/frequency/clustering 属 1F）；单 backbone/detector/size；bottle 为重建数据（equivalence check 缓解）；grid texture 类方差普遍放大（Δstd 至 +11.4）现象记录待 1F 解释。

### Next Step

**停止，不自动进入 1F。** 人工判断：方差收缩模式与 grid texture 方差放大是否值得机制级研究。

### 文件

- 脚本：`scripts/experiment1e_{runner,orchestrator,bottle_reconstruct,analysis,figures_area,verdict}.py`
- 实验记录：`experiments/exp1e_cross_category/README.md`
- 结果：`results/experiment_1e/`（config.json 冻结配置、`<category>/seed_*/` 原始 schema、`analysis/`（含 verdict.json）、`figures/figure1~4`）

---

## PatchCore Bottle Result

当前选择 `broken_large` 异常样本进行可视化。

结果包括：

1. **Original Image**
   - 输入模型的 Bottle 图片；

2. **Ground Truth**
   - 数据集提供的真实缺陷区域；

3. **PatchCore Anomaly Map**
   - PatchCore 输出的连续像素级异常分数；
   - 高响应区域表示该位置与正常特征差异较大；

4. **Predicted Mask**
   - 根据异常阈值得到的最终异常区域。

实验中可以观察到：

> PatchCore Anomaly Map 的主要高响应区域与 Ground Truth 缺陷位置基本一致。

同时，Predicted Mask 能够定位主要缺陷区域，但预测边界与 Ground Truth 仍存在一定差异。

### Result Image

![PatchCore Bottle Result](results/patchcore_bottle_broken_large.png)

---

# Current Progress

当前已经完成：

## 阶段 1：工业异常检测入门 ✅

理解：

- 工业异常检测任务
- 正常 / 异常样本
- Ground Truth
- 重建式异常检测
- 特征式异常检测

## 阶段 2：实验环境搭建 ✅

完成：

- Conda
- PyTorch
- CUDA
- Anomalib
- Jupyter
- MVTec AD

## 阶段 3：PatchCore Baseline ✅

完成：

- Dataset
- Model
- Fit
- Memory Bank
- Test
- Predict
- Anomaly Map
- Predicted Mask
- 结果保存
- Notebook 整理

## 阶段 4：实验分析与 Baseline 扩展 🚧

接下来开始从：

> **“把模型跑起来”**

逐渐进入：

> **“分析为什么得到这样的结果，并设计自己的实验。”**

---

# Next Step

下一阶段暂时不急着直接修改模型。

优先完成以下几个小任务：

1. **记录并理解 PatchCore 的评价指标**
   - Image AUROC
   - Image F1
   - Pixel AUROC
   - Pixel F1

2. **分析图像级检测与像素级定位的区别**
   - 为什么 Image-level 指标很高；
   - 为什么 Pixel-level F1 相对较低；
   - 结合 Ground Truth 和 Predicted Mask 分析误差。

3. **完成第一个 Mini Experiment**
   - 修改 `coreset_sampling_ratio`
   - 例如比较：
     - 0.05
     - 0.10
     - 0.20
   - 观察检测性能、运行时间和 Memory Bank 规模变化。

4. **逐步扩展 Baseline**
   - PatchCore
   - PaDiM
   - EfficientAD

最终逐渐形成：

```text
Baseline 复现
      ↓
指标分析
      ↓
参数实验
      ↓
模型对比
      ↓
发现问题
      ↓
提出小型改进
      ↓
论文方向探索
```

---

## Current Goal

当前短期目标：

> **从“能够跑通 PatchCore”推进到“能够独立分析 PatchCore 实验结果，并完成第一个小型对比实验”。**

暂时不追求复杂模型改进。

优先保证：

**每学一个方法，都留下一个能够运行、能够解释、能够复现的实验成果。**