# MSC-AD → Experiment 2 适配评估

> 只做分析，不修改模型、不写训练代码。目标：判断 MSC-AD 是否适合直接替换/补充 Experiment 2 的数据。

## 一、原始数据是否满足 anomaly detection 设置？

**满足。** MSC-AD 是为 unsupervised anomaly detection 专门构建的数据集：

- 有 normal（无缺陷铸件表面）样本可用于训练；
- 有 anomaly（5 类缺陷）样本用于测试；
- 提供 sample-level 与 pixel-level GT。

对照 MVTec AD 式 `train/good + test/good + test/defect_x + ground_truth/defect_x` 结构，**理论上可无损转换**。

## 二、train/test 如何划分？

- 官方未在公开文本给出固定划分规则，需下载后按目录结构确认。
- 若官方已按 scene 分目录，转换脚本需要把"同一表面类型下、固定 scene（光照+分辨率）"的 normal 归入 train，defect 归入 test。
- **决策点**：train 是否只放"某一种光照"的 normal，还是混合多种光照的 normal——这直接决定实验是在测"跨光照泛化"还是"同光照检测"，必须在 plan 阶段明确（见下）。

## 三、normal 图像是否足够？

- 待下载实测（官方未给各类别数量）。6 表面 × 12 scene，normal 总量大概率充足。
- 需确认：**每种光照档位下的 normal 是否成组**（同一块表面在 low/mid/high 三档下是否都有对应 normal），这决定能否做"跨光照 robustness"实验。

## 四、defect 是否可作为 test anomaly？

**可以，且比 CSEM-MISD 更合适**：

- CSEM-MISD **无 defect type 标签**（只能 specimen 级分析，无法回答"不同 defect 的差异敏感性"）；
- MSC-AD **有 5 种 defect type 标签**，可直接做 defect-type 级分组，**正中你的研究目标**："不同缺陷类型是否对 illumination invariant representation 有不同敏感性"。

## 五、illumination 是否可作为额外实验变量？

**可以，但维度要注意**：

- MSC-AD 的 illumination 是 **3 档亮度强度（low/mid/high）**；
- 它适合作为"真实亮度变化下的 robustness 变量"，直接替代 Experiment 1 的合成 brightness 扰动，且是**真实采集、非合成**，结论更强；
- 但它**不是 CSEM 的"光照方向"变量**，无法覆盖 specular/shadow/局部对比度方向变化。

## 六、能否直接接 Anomalib / PatchCore？

**能，但需要一个小适配脚本**（复用已有 prepare 思路，非改模型）：

1. MSC-AD 是 bmp/png 灰度图 → 转 RGB 三通道（PatchCore 的 wide_resnet50_2 需 3 通道）或灰度复制为 3 通道。
2. 重排为 MVTec AD 目录结构（train/good、test/good、test/defect_x、ground_truth/defect_x）。
3. 分辨率跨 150~600，Anomalib 预处理会 resize 到 256，可直接接，但低分辨率档（150）resize 上采样会引入伪影，**第一轮建议固定高/中分辨率档**。
4. 写自定义 datamodule 或在 `Folder` 数据格式上套 MVTec 结构即可，无需动 PatchCore 模型本体。

## 七、结论：MSC-AD vs CSEM-MISD（对 Experiment 2 的适配度）

| 维度 | MSC-AD | CSEM-MISD | 对研究目标的权重 |
|---|---|---|---|
| defect type 标签 | ✅ 5 类 | ❌ 无 | **决定性**（你的核心问题是 defect-specific） |
| pixel mask | ✅ | ✅ | 均可 |
| 光照维度 | 3 档亮度 | 108 方向 | 不同子问题 |
| 数据规模 | 1.07GB | 6.8GB | MSC 更友好 |
| 下载可行性 | 国内镜像/邮箱申请 | 需翻墙+极慢 | MSC 明显更优 |

**建议：MSC-AD 更适合作为 Experiment 2 的首选数据**，因为它补上了 CSEM 缺失的 defect type 标签，且数据量小、有国内镜像。前提是你确认研究问题是"亮度/曝光维的 defect-specific 敏感性"；若你更在意"光照方向维"，则 MSC-AD 不覆盖，仍需 CSEM（或二者组合）。
