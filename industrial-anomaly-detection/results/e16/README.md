# E16 · R1 Signal Qualification

**日期**：2026-10-14
**论文阶段**：方法资格赛（qualification）
**Verdict**：**GO**（5/5 预注册判据满足）

## 一句话结论
R1 illumination-consistent adaptation 在 Bird / Car 上给出**跨 3 seed 一致的正收益**（6/6 ΔAUROC>0，mean **+0.0185**，mean ΔAP **+0.0174**），在 Cube control 上**不造成明显退化**（mean +0.0026），且两个鲁棒性指标**全部 9 cell 同向改善**——具备进入论文方法阶段的资格。

## Q1 — R1 在 Bird / Car 是否稳定？
**是。**
- Bird+Car 6/6 ΔAUROC > 0（Bird +0.0106/+0.0305/+0.0365；Car +0.0196/+0.0079/+0.0063），6/6 ΔAP > 0。
- 3-seed mean：Bird **+0.02586**、Car **+0.01123**。
- 配对 bootstrap（test-image resampling）：4/9 cell 的 95% CI 排除 0（Bird\|1、Bird\|2、Car\|0、Cube\|0，全为正）；Bird 的两格最硬（P>0 = 0.999/1.000）。
- 去掉各自最好 seed 后：Bird 仍 +0.0205、Car 仍 +0.0071 —— 非单 seed 依赖。

## Q2 — Cube 是否受到损害？
**没有。** Cube 3-seed mean ΔAUROC = **+0.00265 ≥ −0.005**（门槛）。
- 逐 seed：+0.0160 / **−0.0123** / +0.0042。seed1 为负（该 seed 的 frozen baseline 异常高 0.87203），但均值仍在门槛内，且 d′/robustness 在该 seed 上同样改善（rob−0.0266/−0.686）。
- Cube 作为 **negative/control category** 的角色成立：方法不是无条件强行改变特征。

## Q3 — 收益是否超过噪声 / seed variation？
**方向上超过，幅度上需分层看。**
- seed 级：符号 6/6 一致，无翻转；seed 间 std（Bird+Car）0.0126，mean 0.0185 → 方向一致性远大于 seed 噪声。
- test-image 级（配对 bootstrap）：9 cell 中 4 个 CI 排除 0；其余 5 个点估计为正但 CI 跨 0。**Bird 的增益统计上最硬，Car 的两个 seed 在噪声带内。**

## Q4 — robustness 是否同步改善？
**是，且 9/9 cell 同向。**
- `robustness_feat`（特征级 1−cos）：0.0346–0.0766 → 0.0076–0.0125（相对降 65–89%）。
- `robustness_score`（分数级归一化漂移）：0.885–1.387 → 0.217–0.328（相对降 67–77%）。
- ⚠️ **诚实标注**：`robustness_feat` 正是 R1 的训练目标（L_cons），其改善**部分由构造保证**，不构成独立证据。`robustness_score`（异常分数稳定性）与 diagnostic B/D 才是更有说服力的旁证。

## Q5 — 是否有可支持论文叙事的 illumination-shift 证据？
**有，但只能写 "consistent with"。**
- **D（illumination shift magnitude）**：Bird 0.238→0.117、Car 0.183→0.097、Cube 0.138→0.078。
- **B（illumination-group centroid distance）**：Bird 0.0652→0.0074、Car 0.0334→0.0056、Cube 0.0314→0.0111。
- **排序完全一致**：shift 越大（Bird > Car > Cube）→ 收益越大（+0.0259 > +0.0112 > +0.0026）。
- ⚠️ **反直觉事实（如实记录）**：A（normal dispersion）**升高** —— bank normal 的 pairwise cos 0.89→0.98、patch std 1.35→2.0。即 R1 不是"增加多样性"，而是**收紧 normal 流形并放大尺度**。因此本轮**不支持**"dispersion reduction 导致增益"的简单叙事，支持的是"**illumination-induced shift reduction**"。
- 可写：*results are consistent with an illumination-induced feature-shift reduction account.* **不得**写 causal proof。

## Q6 — 最终 verdict
**GO**（全部 5 条预注册判据：6/6>0 · mean ΔAUROC +0.0185≥+0.010 · mean ΔAP +0.0174≥0 · robustness 双向改善 · Cube +0.0026≥−0.005）

## 关键文件
`protocol.md`（冻结 + 附注1）· `audit.md` · `metrics_raw.csv` · `summary.csv` · `diagnostics.csv` · `bootstrap.json` · `verdict.json` · `e16_partial.json`（9 cell 全量 + per-image scores）· `figures/`（5 张）· `logs/`

复现：`python scripts/e16_qualify.py --categories Bird,Car,Cube --seeds 0,1,2` → `scripts/e16_analysis.py` → `scripts/e16_bootstrap.py`

## 对论文证据链的贡献
把 R1 由「seed0 KEEP 但 Cube 不稳」升级为「跨 3 seed 稳定 + AP 同向 + control 不受伤 + robustness 双向改善 + 与 illumination-shift 幅度排序一致」，并明确记录两条限制（rob_feat 部分自证；normal dispersion 反升）。**R1 获得论文方法资格。**

## 下一步（不自动执行）
```
NEXT = E17 Lightweight Refinement   （仅 3–5 个极简配置，等人工裁决）
```
