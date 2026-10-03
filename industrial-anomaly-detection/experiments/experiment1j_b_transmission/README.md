# Experiment 1J-B — Geometry → NN → Score Transmission Intervention

> 状态：已完成。**MECHANISM EXPLORATION STOPPED AFTER 1J-B.**
> 前序链：1E → 1G → 1H → 1H-S → 1I → 1J-A → **1J-B（机制探索最后一站，硬停止线）**。
> 协议运行前冻结于 `results/experiment_1j_b/protocol.md`；artifact audit 见 `results/experiment_1j_b/artifact_audit.md`。

---

## Research Question

人为压制/保留 Layer3 α-IN 引起的 geometry spread change，下游 NN dispersion 与 anomaly-score dispersion 是否随之系统变化？即 1J-A 发现的 Layer3 geometry expansion 是 NN/score amplification 的**中介环节**（mediator），还是仅伴随 signature（signature）？

## Intervention（冻结）

对每张 defect 图，用 1J-A 缓存的 Layer3 F0（α=0）/ F1（α=1）：

```
X(β) = μ1 + (r0 + β·(r1 − r0)) · D1/(r1+eps)，β ∈ {0, 0.5, 1}
```

保留 α=1 的 centroid 与方向结构，仅控制 RMS radius：β=1 等价原 α=1；β=0 把 radius 压回 α=0 水平。frozen M0（α=0 normal Layer3 bank）对全部 β 共用。

## Sanity Checks（全部通过）

- β=1 reconstruction：`max_abs_diff = 9.5e-7 < 1e-5`（462/462 图）✅
- radius 控制：`|target − actual| ≤ 1.9e-6` ✅
- 数据完整性：462 defect 图 × 3 β = 1386 per-image 行，与理论值精确一致 ✅
- defect mapping / Layer3 hook / preprocessing 与 1J-A 一致（复用同一 npz 与 bank）✅
- 完整 PatchCore post_concat score：**unavailable**（post_concat 1536 维 bank 未保存，按协议 Step 8 不伪造，score 采用 layer-specific proxy 并明确标注）

## Primary Results

### expand 剂量-反应（β=0 → 0.5 → 1，跨 seed mean）

三个 expand defect 的 radius / NN std / score proxy 随 β **全部近线性单调上升**（Figure 1）：

| defect | Δradius | ΔNN std | Δscore proxy |
|---|---|---|---|
| grid/glue | +10.84 | +3.45 | +20.60 |
| grid/metal_contamination | +11.08 | +4.08 | +25.40 |
| grid/thread | +10.73 | +3.59 | +20.73 |

### Transmission Recovery Ratio（descriptive）

`TRR_NN = ΔNN(β0→β1) / ΔNN(α0→α1)`：

| family | TRR_NN |
|---|---|
| **expand** | **+0.64 / +0.82 / +1.39** |
| neutral | +0.03 / +0.04 / +0.05 |
| shrink | +0.11 / −0.05 / −0.08 |

**含义**：expand 的 radius expansion 解释了其 NN amplification 的约 64%~139%；neutral/shrink 的 NN 变化几乎完全不由 radius 驱动。

### Family Controls

- **neutral（negative control）**：β 干预对其 NN/score 影响极弱（ΔNNstd=+0.16, Δscore=+0.82，均为 expand 的 ~4%）。**1J-A 的「neutral NN↑ 但 score≈0」之谜就此解答：neutral 的 NN 扩张不是 radius 驱动的**（来自 centroid/方向等其他 geometry 分量），且不传导到 score。
- **shrink（direction control）**：Δradius≈0（+0.03，因其 L3 本无 expansion 可压），NN/score 几乎不动。expand 的 transmission pattern 具有明确 family specificity（Figure 4 三家族在 (Δradius, ΔNN) 平面完全分离）。

## Verdict

**CASE A — Transmission Supported**

1. ✅ expand 3/3：β↑ → radius↑ → NN↑ → score↑（单调，NN 单调性 3/3）；
2. ✅ β=0 压制 geometry expansion 后，NN/score amplification 同步大幅减弱；
3. ✅ neutral/shrink 不出现同等强度 transmission。

结论（受措辞上限约束）：

> Controlled manipulation of Layer3 feature spread systematically modulates downstream NN-distance and anomaly-score dispersion in expand defects, providing intervention-based support for the geometry-mediated transmission hypothesis.

**不写** proved causal mechanism——intervention 仅控制 RMS radius 一个分量。

## Limitations

1. **score 是 layer-specific proxy（max patch NN），非完整 PatchCore post_concat score**（bank 缺失，unavailable）。score 传导的证据强度弱于 NN 传导。
2. intervention 仅控制 RMS radius；glue/metal 的 TRR_NN≈0.64~0.82 提示约 18%~36% 的 NN 变化来自 radius 以外分量；thread 的 1.39 为超调（descriptive，不过度解读）。
3. β=0 状态（α=1 方向结构 + α=0 radius）是反事实构造，不对应任何真实 α。
4. TRR 为 descriptive，不升级为 primary；n=9 不做显著性宣称。

## Mechanism Boundary

本证据链（1E→1G→1H→1H-S→1I→1J-A→1J-B）支持的最强表述：

> 对 expand defects，α-IN 在 Layer3 引起的 feature spread expansion 是其 NN-distance 与 anomaly-score dispersion amplification 的主要中介环节（intervention-based evidence, layer-specific score 口径）；该通路为 expand 家族特有，neutral 的 NN 变化走 radius 以外的非传导通路。

**不能说**：Layer3 semantics「导致」dispersion；geometry 是唯一中介；完整 PatchCore score 口径下传导成立（未测）。

## Final Mechanism Conclusion

**Mechanism evidence level: Strong**（NN 传导：intervention 级证据；score 传导：proxy 级证据 + 1H-S 完整 score 口径的自然相关性证据）。

## Transition to Method Design

**MECHANISM EXPLORATION STOPPED AFTER 1J-B.**

研究主线转入：**Defect-Preserving Illumination Robustness → method design → illumination benchmark → paper**。机制部分以本链结果写入论文（1J-A：Layer3-specific geometry signature；1J-B：intervention 级 transmission 支持）。1J-C / channel attribution / receptive-field / covariance 深挖等仅作 Future Work 候选。
