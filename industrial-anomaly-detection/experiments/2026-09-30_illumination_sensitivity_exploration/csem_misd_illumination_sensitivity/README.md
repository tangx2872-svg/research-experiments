# Industrial Defect Detectability under Real Multi-Illumination Conditions (CSEM-MISD / Gear)

> 归档状态（2026-09-30）：本路线暂时搁置，未形成明确可推进的研究方向。下文的后续计划为当时记录，当前状态以[总览](../README.md)为准。

## 1. Experiment Title
真实多光照条件下工业缺陷可见性与异常检测敏感性实验（Phase 0.5）

## 2. Date
2026-09-30

## 3. Research Question
真实光照方向变化时，同一个工业表面缺陷的「可见程度」与「异常检测效果」是否明显变化？
进一步：不同缺陷 / 不同异常样品对 illumination direction 的敏感程度是否不同？

## 4. Motivation
之前的 F_alpha / Instance Normalization 实验存在混淆（IN 同时改变 style 与 feature scale），
不能解释为光照。改用 CSEM-MISD：同一 specimen 在真实 light-dome 中以 108 个真实光照方向拍摄，
illumination 是真实采集的，非合成。

## 5. Hypothesis
> 假设 ≠ 已证明结论。
- H1：同一缺陷在不同光照方向下的「可检测性（anomaly score / 检出）」会明显变化；
- H2：不同缺陷样品对光照方向的敏感性不同。
目的：判断该现象是否存在，不预设方向。

## 6. Dataset
- 名称：CSEM-MISD — CSEM's Multi-Illumination Surface Defect Detection Dataset
- 对象：三类金属件（washer / screw / gear），本实验用 **gear**
- 来源：Zenodo record 7410513（`gear.tar.gz`，约 6.8 GB，md5 `9a896591ff69fc0d64ee93458e6ad924`）
- 论文：Honzátko et al., "Defect segmentation for multi-illumination quality control systems",
  Machine Vision and Applications (2021)
- 官方代码：https://github.com/DawyD/illumination-preserving-rotations
- 许可：CC BY-NC-ND 4.0（非商业、禁止演绎）
- 本地目录：`data/csem_misd/`（解压后）；详见 `data/README.md`

## 7. Dataset Structure
（由官方 `dataloader.py` + Zenodo 说明确认；图像数据本身因下载受阻未逐一核验）
- 顶层分 `Train / Test / Unannotated` 三个目录；
- 每个 specimen 一个目录，内含 108 张灰度图（每张一个光照方向）、前景 mask、缺陷 mask；
- 图像：8-bit 灰度 PNG，512×512；
- 文件命名（dataloader.py 第 340 行）：`{prefix}_{sample_dir}_{illum_nr}.png`，
  例如 `gear_001_101.png`；illuminationNr 为 3 位：首位 = 仰角索引（1 最高…9 最低），
  后两位 = 方位角索引（01–12）→ 共 9×12=108；
- 前景 mask：`{prefix}_{sample_dir}_segmentation.png`（每 specimen 一个，自动提取）；
- 缺陷 mask：`{prefix}_{sample_dir}_mask.png`（每 specimen 一个，手工标注；存在即表示有标注）；
- 每类根目录含 `light_vectors.csv`（108×3 光照方向向量，按 illuminationNr 字典序）与
  `light_currents.csv`（官方代码用名）/ `light_intensities.csv`（Zenodo 说明用名，光强 0–127），
  两个文件名可能不一致，需以实际下载为准。

## 8. Experimental Assumptions（前提核验状态）
| 前提 | 状态 | 依据 |
|---|---|---|
| 35 defective + 35 intact + 若干 unannotated | ✅ 官方说明确认（Zenodo/论文），具体 Train/Test 内如何分 | 官方描述 |
| 每个 specimen 完整 108 张 | ✅ 官方确认（9 仰角 × 12 方位角 = 108） | dataloader.py + 论文 |
| illumination ID 跨 specimen 严格对应 | ✅ 同一 ID = 同一真实光照位置（light_vectors 按 ID 字典序索引） | dataloader.py |
| defect mask 每 specimen 一个（静态） | ✅ 一个 `_mask.png`，可用于任意 illumination 的 pixel 评估 | dataloader.py |
| 是否有 per-specimen defect type 标签 | ❌ **无**（代码中无任何缺陷类型字段） | dataloader.py |
| intact 是否足以构建 normal memory | ⚠️ 35 intact × 108 光照 = 3780 张；若每光照单独建 memory 则仅 35 张/光照，偏少 | 官方说明 |
| Train/Test 内 intact/defective 如何区分 | ⚠️ 未核验（需实际数据；代码仅靠 `_mask.png` 有无判断是否标注） | dataloader.py |

## 9. Experimental Design
（Phase 0.5 仅验证数据/前提，未训练。）
候选设计 A（illumination-matched baseline）：对每个 illumination I_k，用 intact@I_k 建
PatchCore normal memory，再用 defective@I_k 评估 —— train/test 光照一致，尽量隔离
illumination domain shift，主要观察 defect visibility/detectability。
候选设计 B（global baseline）：用 intact 全光照建一个 memory，再逐 illumination 评估，
用于对比 domain-shift 的影响。两种设计回答不同问题，详见 §18/§19。

## 10. Environment
- OS: Windows 11
- Python 3.11.16（conda `industrial-ad`）
- PyTorch 2.11.0+cu128；torchvision 0.26.0+cu128；CUDA 12.8
- GPU: RTX 5060 Laptop 8GB
- Anomalib 2.6.2；OpenCV 5.0.0；scipy 1.17.1；numpy 2.4.6；matplotlib 3.11.2

## 11. Procedure / Research Log
### 2026-09-30（Phase 0.5）
- 检索官方 Zenodo 7410513 / 论文 / 代码仓库，确认 Gear 下载地址（`gear.tar.gz` 6.8 GB，md5 `9a896591ff69fc0d64ee93458e6ad924`）。
- 尝试下载 Gear：**失败**（Zenodo 从当前网络约 10–27 KB/s，预计 45–67 小时，且连接中断 `curl:18`）。已删除不完整文件。
- 从官方 GitHub 仓库（DawyD/illumination-preserving-rotations）获取 `data/dataloader.py`，权威核验目录结构、文件命名、mask 组织、光照编号与 CSV 索引方式。
- 更新 `.gitignore`、创建 `data/README.md`、建立本实验 README。
- 结论：数据下载受阻 + 无缺陷类型标签 → 研究问题需 MODIFY（见 §18/§19）。

## 12. Parameters
（暂无，Phase 0.5 未训练）

## 13. Results
**Phase 0.5，尚未进行正式 PatchCore 实验。**（无模型结果，不虚构。）

## 14. Observations
- CSEM-MISD 的 illumination 是**离散 108 个真实光照方向**（9 仰角 × 12 方位角，方位角间隔 30°），非连续光强。
- 缺陷 mask 是**每 specimen 一个（静态）**，因此可用于任意 illumination 下的 pixel 评估。
- **数据集不提供 per-specimen defect type 标签**（官方代码中无此字段），因此无法按 scratch/hole/notch 分组。

## 15. Problems Encountered
- **Gear 下载失败**：Zenodo 从当前网络下载速度过低（约 10–27 KB/s），6.8 GB 需 45–67 小时，且连接会被中断。需更换网络/代理后由用户自行下载，或等待网络改善后重试。

## 16. Modifications
（暂无）

## 17. Limitations
- illumination 是离散 108 方向（非连续光强）；方位角未经严格标定（官方注明可能偏差几度）。
- 金属表面强镜面反射可能饱和相机传感器。
- 缺陷占比极低（gear 约 0.2% 像素），类严重不平衡。
- 缺陷标注主观、边界不精确。
- 无 defect type 标签，无法做 defect-type 级分组。

## 18. Plain-language Conclusion（方法学批判）
- 「每个 illumination 单独建 normal memory」这个设计**能有效隔离 illumination domain shift**
  （train 与 test 光照一致），性能随光照的变化主要反映缺陷在该光照下的「可检测性」，
  而非「模型是否适应新光照」。就这一点，设计是合理的。
- 但存在三个主要 caveat：
  1) 每 illumination 的 normal memory 只看到约 35 张 intact 图（每 specimen 一张），
     PatchCore 的 normal 样本量偏少；
  2) **无 per-specimen defect type 标签** → 无法按缺陷类型分组，只能做 specimen 级分析；
  3) 金属镜面反射随光照方向大幅变化，anomaly score 可能混入「反射差异」而非纯「缺陷可见性」。

## 19. Decision
**MODIFY**（理由见 Phase 0.5 报告）

## 20. Next Step
等待用户：① 解决 Gear 下载（更换网络/代理）；② 确认是否接受「specimen 级」而非
「defect-type 级」的研究粒度。二者确认后进入 Phase 1（最小 PatchCore baseline）。
