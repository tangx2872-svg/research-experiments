# Industrial Anomaly Detection

工业视觉异常检测学习与实验项目。

本项目用于研究生阶段探索工业视觉与视觉异常检测方向，
通过公开工业数据集和经典异常检测算法完成：

- 基础学习
- 模型复现
- 对比实验
- 小型改进实验
- 论文方向探索

## Environment

- Windows 11
- Python 3.11
- PyTorch 2.11.0 + CUDA 12.8
- NVIDIA GeForce RTX 5060 Laptop GPU
- Anomalib 2.6.2
- OpenCV
- Jupyter

Conda 环境：

```bash
conda activate industrial-ad
Project Structure
industrial-anomaly-detection/
├── data/           # 数据集（不上传 Git）
├── experiments/    # 正式模型实验
├── notebooks/      # 数据分析与学习 Notebook
├── results/        # 实验结果、图片、模型权重
└── scripts/        # 通用训练、测试和数据处理脚本
Dataset
当前计划使用：
- MVTec AD
首先从单个类别开始实验，后续逐渐扩展到完整数据集。
Models
计划逐步实验：
- PatchCore
- PaDiM
- EfficientAD
后续根据论文阅读和实验结果增加其他方法。
Research Roadmap
当前路线：
1. 搭建工业异常检测实验环境
2. 熟悉 MVTec AD 数据集
3. 跑通 Anomalib Baseline
4. 理解 PatchCore 原理
5. 对比不同异常检测方法
6. 尝试小型改进与消融实验
7. 探索可形成论文的研究问题
Experiment Log
2026-09-24
- 创建 industrial-ad Conda 环境
- 配置 PyTorch + CUDA
- RTX 5060 Laptop GPU 测试成功
- 安装 OpenCV / Jupyter
- 安装 Anomalib 2.6.2
- 创建工业异常检测实验目录
```



## Experiment Log

### 2026-09-24｜环境搭建与异常检测入门

#### 已完成

- 创建 `industrial-ad` Conda 环境
- 配置 PyTorch + CUDA
- RTX 5060 Laptop GPU 测试成功
- 安装 OpenCV / Jupyter
- 安装 Anomalib 2.6.2
- 创建工业异常检测实验目录
- 明确使用 MVTec AD 作为第一阶段实验数据集
- 初步理解工业异常检测与普通监督分类/YOLO 检测的区别
- 理解重建式异常检测的基本思想
- 明确当前主线优先放在工业视觉异常检测，图学习与推荐系统暂时作为长期事项

#### 当日里程碑

完成工业异常检测实验环境搭建，并建立对异常检测任务的基本认识。

### 2026-09-25｜PatchCore Baseline 实验

#### 已完成

- 确定 PatchCore 作为第一个重点 Baseline
- 使用 Anomalib 开始 PatchCore 实验
- 跑通基本推理/测试流程
- 获取模型输出结果
- 接触并区分以下四类实验输出：
  - Original Image
  - Ground Truth Mask
  - Anomaly Map
  - Predicted Mask
- 学习如何从 Anomalib 输出对象中读取：
  - `image`
  - `gt_mask`
  - `anomaly_map`
  - `pred_mask`
- 学习 PyTorch 图像 `[C, H, W]` 与 Matplotlib `[H, W, C]` 的维度转换
- 使用 Matplotlib 编写 PatchCore 异常检测结果可视化代码

#### 正在进行

- 完整生成一组 PatchCore 四联图结果
- 能够自己解释异常热力图与预测 Mask 的含义
- 整理并保存第一组正式实验结果

#### 当日里程碑

PatchCore 已从“只知道名字”推进到“能够实际运行并读取、可视化模型输出”。

## Current Progress

当前阶段：

**阶段 3：跑通 Anomalib Baseline → 阶段 4：理解 PatchCore 原理**

目前不是单纯学习概念，而是已经进入：

**运行模型 → 查看结果 → 理解输出 → 保存实验结果**

下一阶段再逐渐进入：

**理解 PatchCore 原理 → 指标评价 → 模型对比 → 小型改进实验**

## Next Step

下一次实验只推进一个小目标：

> **完成 PatchCore 第一组标准实验结果并保存四联图。**

完成标准：

- 成功显示 Original Image
- 成功显示 Ground Truth
- 成功显示 Anomaly Map
- 成功显示 Predicted Mask
- 保存结果图片到 `results/`
- 用自己的话解释这四张图分别代表什么

完成以上内容后，再将“PatchCore Baseline 跑通”正式标记为完成。