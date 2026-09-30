# 工业异常检测光照敏感性探索｜2026-09-30

**状态：已归档，暂时搁置。此次探索未达到预期，未找到明确可继续推进的研究方向。**

这一天尝试了 PatchCore 特征归一化，以及两个真实光照数据集的可行性调查。代码跑通不等于研究假设得到支持；数据调查受阻也不等于模型实验失败。

## 尝试与结论

| 路线 | 实际完成 | 结果与停止原因 |
|---|---|---|
| [F_alpha / PatchCore](falpha_patchcore/README.md) | Bottle 上 alpha=0/1 运行及配对分析 | IN 同时改变特征分布与尺度，不能把差异解释为纯光照作用；未建立有效改进方向 |
| [MVTec AD 2](mvtec_ad2_illumination_sensitivity/README.md) | 数据结构及前提调查，未训练 | 按当日调查，公开数据不能同时提供所需光照划分、缺陷类型和 GT，原问题无法直接本地验证 |
| [CSEM-MISD Gear](csem_misd_illumination_sensitivity/README.md) | 数据与官方加载代码调查，未训练 | 数据下载失败，且缺陷类型标签不足，未进入模型验证 |

## 已有模型结果

以下数值来自保存的 metrics.json：Bottle，209 张训练图、83 张测试图，seed=0。

| 指标 | alpha=0 | alpha=1 |
|---|---:|---:|
| Image AUROC | 1.0000 | 1.0000 |
| Pixel AUROC | 0.9856 | 0.9822 |
| Pixel F1 | 0.7270 | 0.7136 |
| Pixel AUPR | 0.7711 | 0.7313 |

alpha=1 未带来上述指标改善。单类别、单种子结果不足以证明普遍规律；跨运行归一化的 anomaly score 差异也不能直接作为真实光照敏感性的证据。

- [alpha=0 原始指标](results/falpha_experiment/bottle/alpha_0/metrics.json)
- [alpha=1 原始指标](results/falpha_experiment/bottle/alpha_1/metrics.json)
- [配对统计](results/falpha_experiment/bottle/analysis/stats_summary.json)
- [逐图配对表](results/falpha_experiment/bottle/analysis/paired_scores.csv)

## 归档结构

```text
2026-09-30_illumination_sensitivity_exploration/
├── README.md                              # 统一入口与最终结论
├── falpha_patchcore/                      # 模型、运行检查、分析脚本及说明
├── mvtec_ad2_illumination_sensitivity/     # 可行性调查与官方参考代码
├── csem_misd_illumination_sensitivity/     # 可行性调查与官方参考代码
├── results/falpha_experiment/             # 原始指标、预测、统计与图表
└── notes/                                # 当日工作日志与数据说明快照
```

原始结果完整保留，参考代码及许可证保持原样。共享数据仍在项目根目录 `data/`，9 月 25 日的 Baseline Notebook 和结果图保留原位。

## 脚本使用

在项目根目录、industrial-ad 环境下运行（仅在需要复查时执行）：

```powershell
python experiments/2026-09-30_illumination_sensitivity_exploration/falpha_patchcore/analyze_falpha.py
python experiments/2026-09-30_illumination_sensitivity_exploration/falpha_patchcore/falpha_patchcore.py --category bottle --alpha 0
python experiments/2026-09-30_illumination_sensitivity_exploration/falpha_patchcore/smoke_test.py
```

分析脚本读取归档结果；新分析写入 `reruns/analysis/`，模型默认写入 `reruns/falpha_experiment/`，smoke test 写入 `reruns/smoke_test/`，避免覆盖历史结果。当前 smoke test 限制 64 张训练图，与归档的 209 张训练图实验不同。

## 后续处理

本次仅整理归档，不继续训练或下载数据。子目录说明中的候选计划和日志中的判断保留为历史材料，不表示当前已确定研究路线。若将来重启，应先重新明确可验证的问题及数据条件。
