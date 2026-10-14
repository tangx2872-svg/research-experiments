# E18-P2 — Statistical Analysis (F0 vs Ours and baseline context)

All numbers are derived from artifacts already on disk. **Zero GPU.**

Official protocol: M2AD view120, seed0 (F0/Ours also seeds 1-2), 300-bank (`random.Random(20261009+seed)`), test = 700 view120 images, labels = `image_anomaly`, filter = `detectable != False`.

| category | kept | ok | ng | filtered |
|---|---|---|---|---|
| Bird | 653 | 320 | 333 | 47 |
| Car | 656 | 300 | 356 | 44 |
| Cube | 678 | 260 | 418 | 22 |

## A. F0 vs Ours (R1-ILA) — paired bootstrap

**Method**: one resample index vector per draw is applied to BOTH methods on the SAME detectable-filtered test images (true pairing: F0 and Ours are scored from the same bank, same features-up-to-adapter, same run). `sklearn.roc_auc_score` / `average_precision_score`.  `n_boot = 3000` draws, `np.random.default_rng(20261014)`, percentile CI (2.5 / 97.5). Draws without both classes are skipped; the effective count is recorded.

A test-order identity check asserts that within a category the 700 test images (specimen, illumination) are in the SAME order for seeds 0/1/2, which is what makes the seed-mean bootstrap valid. **This passed for all three categories.**

| cell | ΔAUROC | 95% CI | P(Δ>0) | ΔAP | 95% CI | P(Δ>0) | n_boot |
|---|---|---|---|---|---|---|---|
| Bird|seed0 | +0.01061 | [-0.00920, +0.02977] | 0.854 | +0.01048 | [-0.01154, +0.03222] | 0.829 | 3000 |
| Bird|seed1 | +0.03049 | [+0.01092, +0.04953] | 0.999 | +0.02890 | [+0.00786, +0.05093] | 0.997 | 3000 |
| Bird|seed2 | +0.03647 | [+0.01733, +0.05591] | 0.999 | +0.03176 | [+0.01106, +0.05419] | 0.998 | 3000 |
| Bird|seedmean | +0.02586 | [+0.00833, +0.04310] | 0.998 | +0.02371 | [+0.00498, +0.04364] | 0.993 | 3000 |
| Car|seed0 | +0.01959 | [+0.00345, +0.03606] | 0.991 | +0.01295 | [-0.00085, +0.02704] | 0.967 | 3000 |
| Car|seed1 | +0.00785 | [-0.01251, +0.02696] | 0.791 | +0.00922 | [-0.00480, +0.02345] | 0.902 | 3000 |
| Car|seed2 | +0.00626 | [-0.01135, +0.02467] | 0.750 | +0.01128 | [-0.00161, +0.02530] | 0.954 | 3000 |
| Car|seedmean | +0.01123 | [-0.00412, +0.02615] | 0.933 | +0.01115 | [-0.00034, +0.02320] | 0.970 | 3000 |
| Cube|seed0 | +0.01600 | [+0.00082, +0.03065] | 0.980 | +0.01170 | [+0.00275, +0.02091] | 0.995 | 3000 |
| Cube|seed1 | -0.01229 | [-0.02875, +0.00405] | 0.073 | -0.00321 | [-0.01241, +0.00648] | 0.260 | 3000 |
| Cube|seed2 | +0.00423 | [-0.01216, +0.02035] | 0.690 | +0.00389 | [-0.00545, +0.01346] | 0.780 | 3000 |
| Cube|seedmean | +0.00265 | [-0.01171, +0.01633] | 0.637 | +0.00413 | [-0.00429, +0.01249] | 0.827 | 3000 |

Seed-mean rows report the CI of the **3-seed mean Δ** under test-image resampling.

## Multi-seed summary (F0 / Ours)

| category | metric | F0 mean ± std | Ours mean ± std | per-seed Δ |
|---|---|---|---|---|
| Bird | AUROC | 0.75131 ± 0.00427 | 0.77717 ± 0.01757 | +0.01061, +0.03049, +0.03647 |
| Bird | AP | 0.79084 ± 0.00193 | 0.81456 ± 0.01343 | +0.01048, +0.02890, +0.03176 |
| Car | AUROC | 0.79979 ± 0.01423 | 0.81102 ± 0.00810 | +0.01959, +0.00785, +0.00626 |
| Car | AP | 0.85001 ± 0.00688 | 0.86117 ± 0.00543 | +0.01295, +0.00922, +0.01128 |
| Cube | AUROC | 0.86380 ± 0.00906 | 0.86645 ± 0.00582 | +0.01600, -0.01229, +0.00423 |
| Cube | AP | 0.92213 ± 0.00493 | 0.92625 ± 0.00261 | +0.01170, -0.00321, +0.00389 |

**Boundary (explicit)**: the per-image bootstrap interval quantifies TEST-SET sampling uncertainty only. It is **NOT** cross-seed uncertainty. Seed variance is reported separately as the std over the 3 independent banks above.

## B. Illumination groups 01–10

Per-image scores with illumination metadata are available for: **AnomalyDINO, F0, Ours, PatchCore-emb(unified), PatchCore-official, SoftPatch**.

**Not available** (metrics only, no per-image scores on disk): AnomalyDINO and SoftPatch on Bird/Car (E14 saved aggregates only), and EfficientAD (E15). Per the protocol these are **reported as missing** — no model was re-run to fill them.

Groups found per source: 10 (illuminations 01–10, plus an `ALL` row).

| source | illumination | n_ok | n_ng | AUROC | good score mean |
|---|---|---|---|---|---|
| F0|Bird|seed0 | ALL | 320 | 333 | 0.7473 | 0.2797 |
| F0|Bird|seed0 | 01 | 32 | 24 | 0.7331 | 0.2958 |
| F0|Bird|seed0 | 02 | 32 | 35 | 0.7661 | 0.2699 |
| F0|Bird|seed0 | 03 | 32 | 32 | 0.7178 | 0.2716 |
| F0|Bird|seed0 | 04 | 32 | 36 | 0.7352 | 0.2725 |
| F0|Bird|seed0 | 05 | 32 | 35 | 0.7482 | 0.2762 |
| F0|Bird|seed0 | 06 | 32 | 36 | 0.7535 | 0.2772 |
| F0|Bird|seed0 | 07 | 32 | 29 | 0.8114 | 0.3057 |
| F0|Bird|seed0 | 08 | 32 | 34 | 0.7978 | 0.2776 |
| F0|Bird|seed0 | 09 | 32 | 36 | 0.7335 | 0.2761 |
| F0|Bird|seed0 | 10 | 32 | 36 | 0.7422 | 0.2741 |
| Ours|Bird|seed0 | ALL | 320 | 333 | 0.7579 | 0.1386 |
| Ours|Bird|seed0 | 01 | 32 | 24 | 0.7734 | 0.1335 |
| Ours|Bird|seed0 | 02 | 32 | 35 | 0.7330 | 0.1361 |
| Ours|Bird|seed0 | 03 | 32 | 32 | 0.7656 | 0.1302 |
| Ours|Bird|seed0 | 04 | 32 | 36 | 0.7422 | 0.1372 |
| Ours|Bird|seed0 | 05 | 32 | 35 | 0.7259 | 0.1418 |
| Ours|Bird|seed0 | 06 | 32 | 36 | 0.7561 | 0.1429 |
| Ours|Bird|seed0 | 07 | 32 | 29 | 0.8513 | 0.1480 |
| Ours|Bird|seed0 | 08 | 32 | 34 | 0.7702 | 0.1381 |
| Ours|Bird|seed0 | 09 | 32 | 36 | 0.7439 | 0.1400 |
| Ours|Bird|seed0 | 10 | 32 | 36 | 0.7639 | 0.1385 |
| PatchCore-emb(unified)|Cube|seed0 | ALL | 260 | 418 | 0.6086 | 0.1417 |
| PatchCore-emb(unified)|Cube|seed0 | 01 | 26 | 44 | 0.6233 | 0.1398 |
| PatchCore-emb(unified)|Cube|seed0 | 02 | 26 | 44 | 0.5656 | 0.1427 |
| PatchCore-emb(unified)|Cube|seed0 | 03 | 26 | 43 | 0.5590 | 0.1378 |
| PatchCore-emb(unified)|Cube|seed0 | 04 | 26 | 44 | 0.6250 | 0.1402 |
| PatchCore-emb(unified)|Cube|seed0 | 05 | 26 | 44 | 0.6731 | 0.1401 |
| PatchCore-emb(unified)|Cube|seed0 | 06 | 26 | 44 | 0.6469 | 0.1454 |
| PatchCore-emb(unified)|Cube|seed0 | 07 | 26 | 23 | 0.5585 | 0.1437 |
| PatchCore-emb(unified)|Cube|seed0 | 08 | 26 | 44 | 0.6329 | 0.1413 |
| PatchCore-emb(unified)|Cube|seed0 | 09 | 26 | 44 | 0.5795 | 0.1423 |
| PatchCore-emb(unified)|Cube|seed0 | 10 | 26 | 44 | 0.6023 | 0.1434 |
| SoftPatch|Cube|seed0 | ALL | 260 | 418 | 0.6934 | 628.8519 |
| SoftPatch|Cube|seed0 | 01 | 26 | 44 | 0.6897 | 625.2868 |
| SoftPatch|Cube|seed0 | 02 | 26 | 44 | 0.6329 | 623.0664 |
| SoftPatch|Cube|seed0 | 03 | 26 | 43 | 0.6986 | 590.9451 |
| SoftPatch|Cube|seed0 | 04 | 26 | 44 | 0.7115 | 609.5930 |
| SoftPatch|Cube|seed0 | 05 | 26 | 44 | 0.7587 | 616.9010 |
| SoftPatch|Cube|seed0 | 06 | 26 | 44 | 0.7247 | 629.3684 |
| SoftPatch|Cube|seed0 | 07 | 26 | 23 | 0.7893 | 650.6710 |
| SoftPatch|Cube|seed0 | 08 | 26 | 44 | 0.7325 | 644.3943 |
| SoftPatch|Cube|seed0 | 09 | 26 | 44 | 0.6416 | 646.2792 |
| SoftPatch|Cube|seed0 | 10 | 26 | 44 | 0.7185 | 652.0137 |

### B.1 Normal-image score drift across illumination (seed0)

`good_mean` = mean anomaly score of the detectable-filtered GOOD images within an illumination group. **Only the SPREAD across groups is interpretable**: R1-ILA changes the feature scale, so absolute levels are not comparable between F0 and Ours.

| category | F0 good_mean min | F0 max | F0 spread | Ours min | Ours max | Ours spread |
|---|---|---|---|---|---|---|
| Bird | 0.2699 | 0.3057 | 0.0358 | 0.1302 | 0.1480 | 0.0178 |
| Car | 0.1605 | 0.2494 | 0.0889 | 0.1052 | 0.1744 | 0.0692 |
| Cube | 0.2133 | 0.2412 | 0.0279 | 0.1608 | 0.1800 | 0.0192 |

This is a **downstream** (score-level) quantity, independent of the training objective (`L_cons` acts on features, not on image scores), and it moves in the same direction in all three categories. It is reported as **consistent with** the E16 illumination-shift account and **not** as causal proof.

**Interpretation boundary**: illumination-group differences are reported **descriptively**. Small per-group `n` and unbalanced ok/ng make single-group AUROC noisy, and no causal claim is made. Full table: `results/e18/illum_breakdown.csv`.

## C. PatchCore official vs unified scoring

Only **Bird** supports this comparison (official Car/Cube are BLOCKED). **A 3-category official-vs-unified comparison is therefore NOT claimed.**

- ΔAUROC (official − unified) = **+0.00612** [-0.01126, +0.02340], P(Δ>0)=0.754
- ΔAP = **-0.00644** [-0.02493, +0.01206], P(Δ>0)=0.268

The two PatchCore rows differ in BOTH representation pipeline and scoring, so this is a configuration contrast, not a controlled scoring ablation.

## D. Quality checks performed

- Every per-image source was re-validated on load: 700 rows, 700 unique (specimen, illumination) keys, label distribution == F0 reference, filtered count == expected.
- All table values were copied from the authoritative JSON artifacts at full precision (no rounded CSV was used as the source of truth).
- Legacy `object_anomaly` numbers are isolated in the appendix and in a `LEGACY - DO NOT USE IN THE MAIN TABLE` bucket; the historical files were not modified.
- Empty cells are left empty with an explicit status (`BLOCKED` / `EARLY_STOP`); no imputation.
- **Independent verification gate** (`scripts/e18p2_verify.py`, output `results/e18/verification.csv`): every main-table value that has per-image evidence was recomputed from scratch and compared to the authoritative full-precision JSON at 1e-12. **40/40 checks PASS.**

