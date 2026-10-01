# Experiment 1: Synthetic Illumination × α-IN Mechanism Screening

> **实验性质：Synthetic Illumination Mechanism Screening / Sanity Check**
>
> 本实验只用 brightness / gamma 等简单光度变换，**不能**模拟真实工业光照
> （specular reflection / highlight / shadow / illumination direction 等均未覆盖）。
> 所有结论只能表述为 "Synthetic illumination screening suggests..."，
> 不得表述为真实光照结论。完整实验记录见项目根目录 README 的
> 「2026-10-01」条目。

- 日期：2026-10-01
- 数据：MVTec AD Bottle（train 209 / test 83）
- 模型：PatchCore（wide_resnet50_2，layer2+layer3，coreset 0.1，seed=0）
- 探针：`F_alpha = (1-α)F + α·IN(F)`，复用 `experiments/2026-09-30_illumination_sensitivity_exploration/falpha_patchcore/`（未修改）
- α：0 / 0.25 / 0.5 / 0.75 / 1.0，每 α 独立 fit

## 结论摘要（Case A：trade-off 苗头存在，待真实光照验证）

- α=0 baseline 与归档结果完全一致（image AUROC 1.0 / pixel AUROC 0.98557）。
- 光度扰动在 α=0 显著推高正常图分数（+0.31~+1.28，Wilcoxon p<1e-4）；α 增大后 4 个扰动条件中 3 个的 delta 单调下降（gamma 1.3 在 α=1 趋近 0），brightness 0.7 呈 U 型例外。
- defect 侧：image AUROC 全 α 饱和于 1.0；pixel AUROC 0.9856→0.9822、pixel AUPR 0.7711→0.7313 单调轻度下降；同 α 内 defect-normal 分离度 d' 单调下降（broken_large 17.6→11.5，broken_small 16.3→12.4，contamination 14.8→11.5）。
- 无 defect-type 分化（Case B 不成立）；趋势非极少数样本驱动。
- 跨 α 绝对分数不可比（IN 尺度混淆，α=1 正常图分数整体膨胀 24.7→29.3），分析只用同 α 配对差值与秩指标。

## 文件

| 文件 | 内容 |
|---|---|
| `raw_results.csv` | 每图 × 每 α × 每 condition 的 pred_score（815 条） |
| `summary_results.csv` | 分 side / condition × α 的均值、中位数、分位数汇总 |
| `metrics.json` | 每 α 的 engine.test 聚合指标（含 α=0 baseline 校验） |
| `threshold.json` | α=0 正常图 original max score |
| `analysis_summary.json` | Q1-Q6 定量依据（Q1/Q2/Q3-Q4/recall） |
| `supplementary_analysis.json` | Q6 逐样本分析、top-3 贡献、Wilcoxon、d' 分离度 |
| `figures/fig1_illumination_robustness.png` | α vs 正常图光度扰动 delta |
| `figures/fig2_defect_sensitivity.png` | α vs 各 defect type 绝对分数（附尺度混淆警告，见 README） |
| `figures/fig3_tradeoff.png` | robustness–sensitivity 平面（每 α 一点） |
| `figures/fig4_sample_trajectories.png` | 逐样本 α 0→1 分数轨迹 |
| `logs/` | 每 α 独立的 Anomalib fit/test 工作目录（可安全清理） |

## 复现

```powershell
# 在项目根目录、industrial-ad 环境
python scripts/experiment1_illumination_tradeoff.py            # 全量（~9 分钟）
python scripts/analyze_experiment1.py                          # 汇总 + 图
python scripts/analyze_experiment1_supplementary.py            # Q6 + 分离度
```
