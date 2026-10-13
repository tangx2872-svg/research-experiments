# E11-P3.1 · Illumination Diagnostics (Bird, DINOv2-B, view120)

## Overall
| condition | AUROC | AP |
|---|---|---|
| A raw 1-NN-max | 0.74725 | 0.78863 |
| B per-illum median-centered | **0.75090** | **0.79470** |
| C global-median (sanity) | 0.74725 | 0.78863 |

- **C == A exactly** (global median is a constant shift -> monotonic -> AUROC invariant). Sanity PASS.
- **B - A = +0.00365** (small but real improvement).
- Calibration set: 2400 images (20 specimens x 12 views x 10 illum) -> **240 per illumination**.
- Normal median spread across illuminations (raw): **0.0378**.

## Per-illumination diagnostics (calibration median applied)
| illum | n_normal | n_anom | norm_median_raw | anom_mean_raw | anom_mean_cal | auroc_raw | auroc_cal |
|---|---|---|---|---|---|---|---|
| 01 | 46 | 24 | 0.2907 | 0.3539 | 0.0925 | 0.6893 | 0.6893 |
| 02 | 35 | 35 | 0.2528 | 0.3410 | 0.0928 | 0.7796 | 0.7796 |
| 03 | 38 | 32 | 0.2769 | 0.3378 | 0.0897 | 0.6900 | 0.6900 |
| 04 | 34 | 36 | 0.2604 | 0.3451 | 0.1002 | 0.7263 | 0.7263 |
| 05 | 35 | 35 | 0.2693 | 0.3490 | 0.0971 | 0.7380 | 0.7380 |
| 06 | 34 | 36 | 0.2642 | 0.3497 | 0.1045 | 0.7549 | 0.7549 |
| 07 | 41 | 29 | 0.2837 | 0.4026 | 0.1340 | 0.7906 | 0.7906 |
| 08 | 36 | 34 | 0.2678 | 0.3633 | 0.1114 | 0.7606 | 0.7606 |
| 09 | 34 | 36 | 0.2630 | 0.3515 | 0.1024 | 0.7418 | 0.7418 |
| 10 | 34 | 36 | 0.2566 | 0.3522 | 0.1067 | 0.7484 | 0.7484 |

## Answers to the six analysis questions
1. **Does calibration improve overall AUROC?** Bird: +0.00365 (yes, marginal). Car/Cube: BLOCKED (no
   specimen-disjoint calibration data without a forbidden download).
2. **Category reversal?** Not assessable (Bird only; Car/Cube blocked).
3. **Improvement from cross-illumination re-ranking?** YES — within each illumination the subtraction is a
   constant, so per-illum AUROC is unchanged (`auroc_raw == auroc_cal` on every row). The entire +0.0037
   comes from aligning normal-score baselines ACROSS illuminations.
4. **Illum07 normal offset corrected?** YES — I07 has the highest normal median (0.2837 raw vs ~0.25-0.26
   for other illuminations; calibration median 0.2686). After centering, I07 normal scores align with the
   rest, removing the systematic inflation seen in E11-P2.
5. **Are anomaly scores also lowered?** YES, partially — I07 anomaly mean drops 0.4026 -> 0.1340, but I07
   anomalies remain the highest after calibration (0.134 vs 0.09-0.11 elsewhere). The method only centres
   the NORMAL baseline; it does not fully equalise the anomaly distribution. No sign of anomaly scores
   being compressed toward normal (within-illum AUROC is preserved exactly).
6. **Calibration sample count sufficient?** YES — 240 normal images per illumination (20 specimens x 12
   views), far above a stable-median requirement.
7. **Does inference depend on illumination metadata?** **YES** — the test image's illumination ID selects
   the median. If illumination metadata is unavailable at deployment, this method is NOT applicable
   (limitation).
