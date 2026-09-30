# F_alpha PatchCore（Instance Normalization 混合）实验

> 归档状态（2026-09-30）：本路线暂时搁置，未形成明确可推进的研究方向。下文的后续计划为当时记录，当前状态以[总览](../README.md)为准。

## Status

- **Preliminary / Deprecated for illumination conclusion**
- 说明：该实验用 `F_alpha = (1-α)F + α·IN(F)` 控制 Instance Normalization 的混合强度，
  作为「style / illumination-sensitive feature suppression」的初步探索。
- 该实验**不能**直接解释为「illumination suppression strength」，原因：
  Instance Normalization 同时改变 style-sensitive information 与 feature
  distribution / scale，存在混淆因素。
- 该实验仍有**方法探索与工作流验证**价值（验证了 F_alpha 可接入 PatchCore、
  建立了可复现的 run_experiment / paired 分析流程），故保留不删除。

## 文件

- `experiments/2026-09-30_illumination_sensitivity_exploration/falpha_patchcore/falpha_patchcore.py`：模型（FAlphaPatchcoreModel/FAlphaPatchcore）+ `run_experiment` + CLI
- `experiments/2026-09-30_illumination_sensitivity_exploration/falpha_patchcore/smoke_test.py`：smoke test（F_alpha 数值校验 + bottle alpha=0/1）
- `experiments/2026-09-30_illumination_sensitivity_exploration/falpha_patchcore/analyze_falpha.py`：alpha=0 vs alpha=1 的 paired 分析
- `experiments/2026-09-30_illumination_sensitivity_exploration/results/falpha_experiment/`：原始结果（勿覆盖）

## 后续

本实验不再扩展。illumination 相关研究改由
`experiments/2026-09-30_illumination_sensitivity_exploration/mvtec_ad2_illumination_sensitivity/` 承担（使用 MVTec AD 2 真实光照 shift 数据）。
