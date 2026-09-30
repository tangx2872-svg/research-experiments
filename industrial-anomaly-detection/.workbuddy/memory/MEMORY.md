> 2026-09-30 归档更新：当天实验已集中到 `experiments/2026-09-30_illumination_sensitivity_exploration/`，状态为暂时搁置、未找到明确可推进方向。以下内容为历史记录，旧路径以归档总 README 为准。

# 项目长期备忘

## 环境（industrial-ad conda，Windows 11，RTX 5060 Laptop 8GB）
- Python 3.11.16 / PyTorch 2.11.0+cu128 / torchvision 0.26.0 / Anomalib 2.6.2 / OpenCV 5.0
- 解释器：`D:/miniconda/envs/industrial-ad/python.exe`

## Anomalib 2.6.2 使用要点（踩坑记录）
1. `Engine()` 不要传 `device=`（Trainer 不接受），用默认自动检测 GPU。
2. 用 `Engine(enable_progress_bar=False, logger=False, barebones=True)`：`barebones=True` 才能关掉 Anomalib 自动添加的 ModelCheckpoint（否则 checkpoint 序列化在 RAM 紧张时 OOM）；`enable_checkpointing=False` 会报 MisconfigurationException。
3. `engine.test()` 返回 list[dict]，默认指标：image_AUROC / image_F1Score / pixel_AUROC / pixel_F1Score。AP 需自定义 evaluator 加 `AUPR`（anomalib.metrics.AUPR，支持 fields+prefix）。
4. PatchCore coreset（KCenterGreedy）用 `torch.randint` + SparseRandomProjection，**必须固定随机种子**才能复现。
5. GPU 被图形应用（Chrome/ChatGPT/微信）争用 ~3GB，全量 MVTec 单类 coreset 峰值 ~2.6GB 处于边缘，可能间歇 OOM。`experiments/falpha_patchcore.py` 提供 `limit_train` 回退；正式实验建议关闭图形应用。

## F_alpha 实验（PatchCore + InstanceNorm 混合）
- F = concat 后 embedding (B,1536,32,32)；F_IN = InstanceNorm(F)；F_alpha = (1-alpha)F + alpha·F_IN。
- IN 插在 `PatchcoreModel.generate_embedding`（覆写），alpha∈[0,0.2,0.4,0.6,0.8,1.0]。
- 代码：`experiments/falpha_patchcore.py`（模型+run_experiment+CLI）、`scripts/smoke_test.py`。

## 数据
- MVTec AD：`data/mvtec_ad/bottle`（train good=209；test good=20 + broken_large=20 + broken_small=22 + contamination=21）。

## CSEM-MISD（Gear）实验（Phase 0.5，未训练）
- 目录 `experiments/csem_misd_illumination_sensitivity/`；数据目标 `data/csem_misd/`。
- 每 specimen 108 光照（9 仰角×12 方位角），缺陷 mask 每 specimen 一个（静态）。
- **无 defect type 标签** → 只能 specimen 级分析。
- Zenodo 下载极慢（~10-27KB/s），需换网络/代理下载 gear.tar.gz（6.8GB，md5 9a896591ff69fc0d64ee93458e6ad924）。
- 研究路线：真实多光照下的缺陷可见性/检测敏感性；设计 candidate「每 illumination 单独建 normal memory」。
