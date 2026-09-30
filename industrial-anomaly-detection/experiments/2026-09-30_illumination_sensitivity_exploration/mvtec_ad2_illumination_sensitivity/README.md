# MVTec AD 2 Illumination Distribution Shift — Defect Sensitivity

> 归档状态（2026-09-30）：本路线暂时搁置，未形成明确可推进的研究方向。下文的后续计划为当时记录，当前状态以[总览](../README.md)为准。

> 实验名称：MVTec AD 2 光照分布变化下的缺陷敏感性分析
> Defect-Level Sensitivity under Illumination Distribution Shift on MVTec AD 2

---

## 1. Research Question

在**真实工业 illumination distribution shift** 下，不同 defect type 的异常检测性能
是否表现出不同程度的变化？

例如：明显结构缺陷可能对光照变化不敏感；细微表面缺陷可能受较大影响。

> 注意：这是**研究问题 / 待验证假设**，不是已证明结论。

---

## 2. Motivation

之前的 F_alpha（Instance Normalization 混合）实验存在混淆因素：Instance Normalization
不仅可能改变 style / illumination-sensitive information，还会改变 feature distribution /
scale，因此**不能**直接解释为「光照变化」。本实验改走更直接、更可信的路线：

> 使用 MVTec AD 2 官方**真实采集**的 illumination distribution shift 数据，
> 不人为修改图片、不人为模拟亮度、不使用 Instance Normalization。

---

## 3. Hypothesis

> 假设 ≠ 已证明结论。

- H1（待验证）：不同 defect type 对 illumination distribution shift 的敏感性不同。
- H0（零假设）：不同 defect type 对 illumination shift 的响应没有系统性差异。

本实验的目的就是判断该现象**是否存在**，不预设方向。

---

## 4. Dataset

- 名称：MVTec AD 2（MVTec AD 第二代基准）
- 论文：Heckler-Kram et al., "The MVTec AD 2 Dataset: Advanced Scenarios for
  Unsupervised Anomaly Detection", IJCV 134(4), 2026. DOI: 10.1007/s11263-026-02743-0；
  arXiv:2503.21622
- 官方主页：https://www.mvtec.com/company/research/datasets/mvtec-ad-2
- 下载页：https://www.mvtec.com/research-teaching/datasets/mvtec-ad-2/downloads
- 许可证：CC BY-NC-SA 4.0（**非商业**）
- 总量：约 **30.4 GB**（完整）；分类别 0.77–10 GB
- 目标本地目录：`data/mvtec_ad_2/`（**尚未下载**，Phase 0 只做结构验证）

### 8 个类别

`can, fabric, fruit_jelly, rice, sheet_metal, vial, wallplugs, walnuts`

### 官方目录结构（由 MVTec 官方 code utils 确认，见 `reference/mvtec_ad_2_public_offline.py`）

```text
mvtec_ad_2/{object}/
├── train/good/*.png                    # 正常，仅常规光照
├── validation/good/*.png               # 正常，仅常规光照
├── test_public/good/*.png              # 正常，混合光照，有 GT（空 mask）
├── test_public/bad/*.png               # 所有异常混在一起，混合光照
├── test_public/ground_truth/bad/*_mask.png   # 像素级 GT（仅 bad）
├── test_private/*.png                  # 常规光照，无 GT，扁平、无标注
└── test_private_mixed/*.png            # 变化光照，无 GT，扁平、与 private 同场景
```

### 各 category 样本量（来自论文 Table；private 集 GT 为官方私有）

| Object | Train | Val | TESTpub (normal/anom) | TESTpriv (normal/anom) | TESTpriv,mix (normal/anom) |
|---|---|---|---|---|---|
| Can | 412 | 46 | 162 (72/90) | 321 (145/176) | 321 (145/176) |
| Fabric | 387 | 43 | 156 (66/90) | 314 (133/181) | 314 (133/181) |
| Fruit Jelly | 263 | 37 | 80 (20/60) | 255 (71/184) | 255 (71/184) |
| Rice | 313 | 35 | 132 (42/90) | 277 (96/181) | 277 (96/181) |
| Sheet Metal | 137 | 19 | 114 (24/90) | 142 (36/106) | 142 (36/106) |
| Vial | 291 | 41 | 140 (35/105) | 276 (78/198) | 276 (78/198) |
| Wall Plugs | 293 | 33 | 150 (60/90) | 232 (96/136) | 232 (96/136) |
| Walnuts | 432 | 48 | 150 (60/90) | 228 (93/135) | 228 (93/135) |

---

## 5. Experimental Assumptions（实验前提，已核验）

| 前提 | 是否成立 | 说明 |
|---|---|---|
| 光照分离且可本地评估 | ❌ | 光照分离在 `test_private` vs `test_private_mixed`，但两者均**无公开 GT** |
| 公开 GT 数据中可区分光照条件 | ❌ | `test_public` 混合了所有光照条件，图片**不标注**光照条件 |
| 公开 GT 数据中可区分 defect type | ❌ | `test_public` 仅 `good/` 与 `bad/`，所有异常混在 `bad/`，**无缺陷类型标注** |
| 场景级配对（同一物理样本不同光照） | ✅ | `test_private` ↔ `test_private_mixed` 是**同一批场景**，但无 GT/无标签 |
| 可做 defect-type-level 比较 | ❌ | 无缺陷类型标签（公开数据），故无法本地做 |
| 可做 image-level paired 比较（带 GT） | ❌ | 光照分离的配对数据无 GT |

> **核心结论**：MVTec AD 2 的公开数据**不支持**「defect-type-level illumination
> sensitivity」的本地评估。详见 §11 / §12。

---

## 6. Experimental Design

（Phase 0 阶段：仅数据与前提验证，无模型实验。）

原设想（需修改）：训练 PatchCore 于 `train/good`，分别在常规光照与变化光照下
按 defect type 比较性能。

因 §5 的前提不成立，待修改后确定（见 §12 Decisions）。

---

## 7. Environment

- OS: Windows 11
- Python: 3.11.16（conda env `industrial-ad`，解释器 `D:/miniconda/envs/industrial-ad/python.exe`）
- PyTorch: 2.11.0+cu128；torchvision: 0.26.0+cu128
- CUDA: 12.8
- GPU: NVIDIA GeForce RTX 5060 Laptop GPU（8 GB）
- Anomalib: 2.6.2（内置 `MVTecAD2` datamodule）
- OpenCV: 5.0.0；scipy 1.17.1；numpy 2.4.6；matplotlib 3.11.2

---

## 8. Procedure / Research Log

### 2026-09-30（Phase 0）

- 检查 `data/`：已有旧版 `data/mvtec_ad/`（bottle），**无** MVTec AD 2。
- 检索官方资料：官方主页、官方论文（arXiv 2503.21622 / Springer IJCV）、官方下载页。
- 下载并阅读**官方 code utils**（`MVTecAD2_public_code_utils.tar.gz`，约 15 KB，
  官方 mydrive.ch 直链），其中 `mvtec_ad_2_public_offline.py` 的 glob 模式
  `{split}/[gb][oa][od]*/[0-9][0-9][0-9]*.png` 与 `ground_truth/bad/*_mask.png`
  **权威确认**了目录结构：public 集只有 `good`/`bad`，无缺陷类型、无光照标注。
- 交叉核对 anomalib 2.6.2 的 `MVTecAD2` 源码（`data/datasets/image/mvtecad2.py`），
  结构与官方 code utils 一致。
- 关键发现：光照分离数据（private/private_mixed）无公开 GT；公开 GT 数据（public）
  无缺陷类型与光照标注 → 原始研究问题无法本地直接验证。
- 结论：**MODIFY**（详见 §12）。

> 本阶段未下载图像数据（约 30.4 GB），未训练任何模型。

---

## 9. Results

**尚未进行模型实验。**（Phase 0 无模型结果。）

---

## 10. Observations

- MVTec AD 2 的光照 shift 是**真实采集**的：改变曝光时间（更亮/更暗）+ 类别特定的
  额外光源（反射、光照梯度、色温变化等），每类至少 4 种光照条件。
- 不同类别的光照变化方式**不同**（非统一连续光强）。
- 数据集把「本地评估」与「官方排行榜」明确分开：只有 `test_public` 带 GT 供本地评估；
  `test_private`/`test_private_mixed` 的 GT 仅在官方评估服务器（benchmark.mvtec.com）。

---

## 11. Limitations

1. **MVTec AD 2 的 illumination shift 不是统一连续物理光强**，而是曝光 + 类别特定光源
   的离散组合。
2. **公开数据无缺陷类型标注**：`test_public/bad/` 把所有异常混在一起，无法本地区分
   scratch / hole / contamination 等。
3. **光照分离的数据无公开 GT**：`test_private`（常规）与 `test_private_mixed`（变化）
   虽然场景配对，但 GT 私有，本地无法计算 AUROC / per-defect 指标。
4. 因此「defect-level illumination sensitivity」无法用公开数据本地验证。

---

## 12. Decisions

- **当前（Phase 0）：MODIFY。**
- 原因：原始研究问题（defect-type-level illumination sensitivity）依赖「公开数据里
  既有光照分离、又有缺陷类型标注」这一前提，而该前提**不成立**。
- 候选修改方向（待用户确认，未擅自继续）：
  - (a) **降粒度**：改为「scene-paired 的 anomaly score 在光照 shift 下的变化」
    （用 `test_private` vs `test_private_mixed` 的同一场景做配对，但无 GT、无缺陷标签，
    只能看 score 分布变化，不能算 accuracy、不能区分缺陷类型）。
  - (b) **走官方评估服务器**：提交 `test_private`/`test_private_mixed` 得到官方
    robustness 指标（聚合、按 category，非 per-defect）。
  - (c) **换数据集/换路线**：如对旧版 MVTec AD（有缺陷类型标签）施加**受控合成光照**
    （注：这违背「不人为模拟亮度」的初衷，需重新权衡）。
  - (d) **放弃 defect-level**，仅做 category-level illumination robustness baseline。

- 旧 F_alpha 实验状态：**Preliminary / Deprecated for illumination conclusion**，
  保留不删除（见 `experiments/2026-09-30_illumination_sensitivity_exploration/falpha_patchcore/falpha_patchcore.py` 及其 README）。

---

## 13. Next Step

等待用户对 §12 修改方向的确认。若选 (a) 或 (b)，则进入 Phase 1：下载 1–2 个较小
类别（如 `vial` 0.77 GB / `sheet_metal` 1.53 GB / `fruit_jelly` 1.2 GB）做最小
PatchCore baseline 与结构二次核验；若选 (c)/(d) 需重新设计。

---
