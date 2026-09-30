# data/ — 数据集说明

本目录存放大型数据集文件，**不提交到 Git**（见项目根 `.gitignore`：`data/*`，仅保留本 README 与 `.gitkeep`）。

## CSEM-MISD（当前实验使用）

- **数据集名称**：CSEM-MISD — CSEM's Multi-Illumination Surface Defect Detection Dataset
- **当前子集**：Gear（`data/csem_misd/`）
- **官方下载来源**：Zenodo record 7410513
  - 文件：`gear.tar.gz`（约 6.8 GB，md5 `9a896591ff69fc0d64ee93458e6ad924`）
  - 下载 URL：https://zenodo.org/records/7410513/files/gear.tar.gz?download=1
- **论文**：Honzátko, D., Türetken, E., Bigdeli, S. A., Dunbar, L. A., & Fua, P. (2021).
  "Defect segmentation for multi-illumination quality control systems." *Machine Vision and Applications*.
- **官方代码仓库**：https://github.com/DawyD/illumination-preserving-rotations
- **许可**：CC BY-NC-ND 4.0（**非商业、禁止演绎**，仅限科研用途）
- **本地目录**：`data/csem_misd/`（解压后）
- **下载日期**：2026-09-30
- **内容概要**：每个 specimen 在半球形 light-dome 中以 108 个不同光照方向拍摄；
  含 train / test / unannotated 划分、前景 mask、缺陷分割 mask、
  `light_vectors.csv`（光照方向）、`light_intensities.csv`（光照强度 0–127）。

## 其他数据集

- `data/mvtec_ad/`：旧版 MVTec AD（bottle），用于早期 PatchCore baseline。
- `data/bottle.tar.xz`：MVTec AD bottle 的压缩包。

## 说明

- 原始 `*.tar.gz` 与解压后的大图片均**不加入 Git**。
- 如未来初始化 Git 仓库，请保持上述 `.gitignore` 规则。
