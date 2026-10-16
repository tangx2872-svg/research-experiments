# Paper limitations (factual material, ready for the limitation section)

## L1 — Doll stable regression
Doll is the only category with a stable negative ΔAUROC (3-seed mean −0.03850, 0/3 positive seeds, paired-image bootstrap CI [−0.0661, −0.0140], p = 0.001). The audit in `results/e23/` shows the failure is present in all three bank seeds, broad across illumination groups (8–9 of 10 groups negative per seed), and consistent with a mild near-symmetric score shift (the anomaly-side relative drop is 0.95–1.00 of the normal-side drop) rather than an anomaly-information collapse. No causal mechanism is established; the low-baseline/high-headroom explanation does not apply (Doll has the lowest F0 baseline and the lowest d′).

## L2 — Multi-view heterogeneity (E21)
E21 (4 views × Bird/Car/Cube × seed0) reports 8/12 positive cells with a macro pooled ΔAUROC of +0.00142 (stratified bootstrap CI [−0.0032, +0.0062]) and ΔAP of −0.00165, and it is a HOLD: the benefit is not restricted to view120 but is highly heterogeneous, with Car at views 240/330 and Cube at views 090/330 being negative (worst: Car v330 ΔAUROC −0.02600). Seed1/2 were not run for E21.

## L3 — Evaluation scope
The complete 10-category validation is on M²AD (view120, 300-bank, official `image_anomaly` + `detectable != False`) with a single backbone (DINOv2-B). MVTec AD 2 external validation was audited but is a HOLD (needs manual dataset access; single-scene ΔAUROC noise half-width 0.04–0.07). Cross-dataset generalization is therefore untested.

## L4 — Shift–gain association is exploratory
The illumination-shift ↔ gain association has n = 10 categories (Pearson r = +0.371, p = 0.291; Spearman ρ = +0.576, p = 0.082) and is reported as association only. Doll is an outlier of this weak trend (residual −0.0453 against the n = 10 fit); a leave-one-out sensitivity excluding Doll raises the Pearson r to +0.807 (p = 0.008) but is reported only alongside the full n = 10 result.

## L5 — Two different uncertainties
Per-category paired-image bootstrap CIs quantify test-set sampling uncertainty only; cross-seed variability is reported separately as the std over the 3 bank seeds (e.g. Doll ΔAUROC −0.03850 ± 0.00379). The two are never merged, and image bootstrap CIs must not be read as cross-seed uncertainty.
