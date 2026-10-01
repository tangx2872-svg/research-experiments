# MSC-AD Dataset Analysis

> 本文档为**文档级侦察**（基于官方页面、论文、nbsdc 国内镜像三处权威来源），
> **本地尚未下载任何 MSC-AD 图像数据**。下文凡是无法从公开材料确认的项，均显式标注"待实测"。

## Dataset Path

- 本地：**无**（`data/` 下尚未建立 `msc_ad/` 目录，全盘无任何 MSC-AD 文件）。
- 官方主页：<https://msc-ad.github.io/MSC-AD2023>
- 论文：Zhao et al., *MSC-AD: A Multiscene Unsupervised Anomaly Detection Dataset for Small Defect Detection of Casting Surface*, IEEE Trans. Industrial Informatics, 20(4), 2024. DOI: 10.1109/TII.2023.3341259
- 国内镜像：国家基础学科公共科学数据中心（nbsdc.cn），CSTR:16666.11.nbsdc.zT8h9kUc，创建机构复旦大学（张文强团队，即论文作者团队）。

## 基本信息（权威来源可确认）

| 项目 | 值 |
|---|---|
| 对象 | 大型**铸件表面**（6 种不同结构表面） |
| 缺陷 | **5 种缺陷类型**（有类型标签） |
| 场景 | **12 种成像场景** = 3 档光照 × 4 档分辨率 |
| 光照 | low / mid / high 三档**亮度强度**（非方向） |
| 分辨率 | 150×150 → 600×600，共 4 档（端点值 150/600 已确认，中间两档待实测） |
| 标注 | **sample-level + pixel-level 精确 GT**（即提供 mask） |
| 数据量 | **1.07 GB，10888 个文件**（nbsdc 镜像） |
| 格式 | bmp / png（含 DS_Store / docx 说明文件） |

## Categories（6 种铸件表面）

官方文本仅表述为 "six types of large casting surfaces with different structures"，**具体名称未在文本中列出**（在页面图片 Fig1 / MSCAD-30 中，待下载后确认）。

## Defect Types（5 种缺陷）

- 官方确认有 **5 种缺陷**，且提供缺陷类型标签（这一点与 CSEM-MISD 截然不同）。
- **具体 5 个类型名未在官方文本公开**，需下载后从目录结构/标注文件确认。
- 从铸件缺陷领域常识推测可能包含气孔 / 缩孔 / 砂眼 / 划痕 / 裂纹等，但**此为推测，不得当作事实**。

## Illumination Information

**关键差异点（务必注意）：**

- MSC-AD 的"光照变化"是 **3 档亮度强度（low/mid/high）**，是标准成像系统里通过光学照明模块调节的整体亮度。
- 这与 CSEM-MISD 的 **108 个光照方向（9 仰角 × 12 方位角）** 是**不同维度**：
  - MSC-AD → **intensity / exposure 维**（更接近 Experiment 1 里用的 brightness 扰动，但为真实采集而非合成）。
  - CSEM-MISD → **direction 维**（改变 specular / shadow / 局部对比度，更"物理"）。
- 对研究目标"illumination invariant representation"而言，两者回答的是不同子问题。

## Ground Truth / Mask

- ✅ 提供 pixel-level mask。
- ✅ 提供 defect type 标签（5 类）。
- sample-level 标签（anomaly / normal）齐备。

## 目录结构（待实测）

官方页面的目录结构图（MSC-AD.png）未以文本形式呈现。**推断**（待下载确认）可能接近：

```
msc_ad/
  表面类型1/
    scene_illum1_res1/
      train/good/...
      test/good/...
      test/defect_类型A/...
      ground_truth/defect_类型A/...
    ...
```

## Suitable Experiment Subset（初步建议）

- 对象：**1 种铸件表面**（具体哪种待下载后按"缺陷类型覆盖 + normal 数量"选定）。
- 缺陷：按 5 种类型分组，优先选 defect-type 分布均衡、样本量够的表面。
- 光照：3 档全用（第一轮直接做 low vs mid vs high 的 robustness screening，比 Experiment 1 的合成 brightness 更真实）。
- 分辨率：第一轮**固定一档**（建议最高 600×600 或中间档），避免 resolution 与 illumination 混淆。

## Problems and Notes

1. **下载门槛（国外官方）**：需用**大学邮箱**发邮件至 mscad2023@gmail.com 申请，填姓名/单位/导师等 8 项，附研究用途声明，换取密码。对无大学邮箱/国外邮箱的用户有门槛。
2. **国内镜像 nbsdc.cn 是更优路径**：复旦大学官方发布，1.07GB，可能免邮箱申请、直接下载——**这是相较 CSEM-MISD 下载失败的最大优势**。但具体是否需要登录/审批待确认。
3. **光照维度 ≠ CSEM 的光照方向**：MSC-AD 是 intensity 维，如果研究问题核心是"光照方向引起的 specular/shadow 变化"，MSC-AD 无法覆盖；如果是"亮度/曝光变化"，MSC-AD 是真实数据、优于 Experiment 1 的合成扰动。
4. **具体类别名/缺陷名/目录结构/中间分辨率/图片数量均待下载实测**，本文档是文档级结论，不是实测结论。
5. **数据规模小（1.07GB）**，RTX 5060 8GB 与 PatchCore 完全可承受，无 CSEM 6.8GB 的显存/磁盘压力。
