# E22 — Statistical summary (M²AD 10 categories × 3 seeds × view120)

Verdict: **HOLD**

## 1–2. Categories with positive mean ΔAUROC / ΔAP

- ΔAUROC > 0: **9/10** — Bird, Car, Cube, Dice, Holder, Motor, Ring, Teapot, Tube
- ΔAP > 0: **9/10** — Bird, Car, Cube, Dice, Holder, Motor, Ring, Teapot, Tube
- both > 0: **9/10** — Bird, Car, Cube, Dice, Holder, Motor, Ring, Teapot, Tube

## 3–4. Macro mean / median

- macro mean ΔAUROC = **+0.00760**; median = **+0.01080**
- macro mean ΔAP = **+0.00777**; median = **+0.00958**
- worst category mean ΔAUROC = -0.03850; best = +0.02586

## 5. Seed stability

| category | ΔAUROC mean ± std | positive seeds | label |
|---|---|---|---|
| Bird | 0.025857 ± 0.013535 | 3/3 | seed_sensitive |
| Car | 0.011233 ± 0.007279 | 3/3 | stable_positive |
| Cube | 0.002647 ± 0.014214 | 2/3 | seed_sensitive |
| Dice | 0.003362 ± 0.000775 | 3/3 | stable_positive |
| Doll | -0.038497 ± 0.004638 | 0/3 | stable_negative |
| Holder | 0.010360 ± 0.004453 | 3/3 | stable_positive |
| Motor | 0.008070 ± 0.007646 | 2/3 | mixed |
| Ring | 0.020312 ± 0.003634 | 3/3 | stable_positive |
| Teapot | 0.017298 ± 0.007286 | 3/3 | stable_positive |
| Tube | 0.015322 ± 0.004989 | 3/3 | stable_positive |

Seed-count distribution: 0/3: 1, 1/3: 0, 2/3: 2, 3/3: 7

## 6. Uncertainty (two separate forms — never merged)

**(a) Image-level paired bootstrap** (test-set sampling uncertainty; n_boot=3000, rng=20261015; paired indices; detectable-filtered):
| category | ΔAUROC CI (seed-mean scores) | p(ΔAUROC>0) | ΔAP CI (seed-mean scores) | p(ΔAP>0) |
|---|---|---|---|---|
| Bird | [+0.0085, +0.0436] | 0.998 | [+0.0038, +0.0423] | 0.991 |
| Car | [-0.0052, +0.0280] | 0.917 | [-0.0028, +0.0218] | 0.942 |
| Cube | [-0.0144, +0.0150] | 0.530 | [-0.0054, +0.0114] | 0.757 |
| Dice | [-0.0076, +0.0155] | 0.746 | [-0.0047, +0.0096] | 0.760 |
| Doll | [-0.0661, -0.0140] | 0.001 | [-0.0335, +0.0077] | 0.097 |
| Holder | [-0.0004, +0.0196] | 0.971 | [+0.0034, +0.0190] | 1.000 |
| Motor | [-0.0095, +0.0207] | 0.753 | [-0.0107, +0.0094] | 0.429 |
| Ring | [+0.0053, +0.0275] | 0.999 | [+0.0045, +0.0195] | 0.999 |
| Teapot | [+0.0015, +0.0277] | 0.986 | [+0.0028, +0.0163] | 0.998 |
| Tube | [-0.0005, +0.0246] | 0.970 | [-0.0000, +0.0143] | 0.974 |

**(b) Cross-seed variability** = std of ΔAUROC over the 3 independent banks (reported per category in §5 and in `main_table_10cat_3seed.csv`). This is a different quantity from (a) and is never combined with it.

## 7–8. Largest gain / largest regression

- largest category gain: **Bird** (+0.02586 ΔAUROC)
- largest category regression: **Doll** (-0.03850 ΔAUROC)

## 9. Illumination shift ↔ gain association (exploratory, n = 10 categories)

| association | Pearson r (p) | Spearman ρ (p) |
|---|---|---|
| D_shift_seed_mean_vs_dAUROC | +0.371 (0.291) | +0.576 (0.082) |
| D_shift_seed0_vs_dAUROC | +0.363 (0.302) | +0.515 (0.128) |
| B_centroid_seed_mean_vs_dAUROC | +0.154 (0.671) | +0.527 (0.117) |
| D_shift_seed_mean_vs_dAP | +0.523 (0.121) | +0.515 (0.128) |

**Interpretation boundary**: n = 10 categories → exploratory / supportive association only. **No causal claim may be written from this table.**

## Normal-image score spread across illumination (descriptive)

| category | F0 spread (3-seed mean) | R1 spread | ratio R1/F0 |
|---|---|---|---|
| Bird | 0.0381 | 0.0146 | 0.382 |
| Car | 0.0639 | 0.0508 | 0.796 |
| Cube | 0.0278 | 0.0161 | 0.579 |
| Dice | 0.0154 | 0.0216 | 1.407 |
| Doll | 0.0168 | 0.0120 | 0.718 |
| Holder | 0.0325 | 0.0307 | 0.945 |
| Motor | 0.0550 | 0.0179 | 0.325 |
| Ring | 0.0264 | 0.0212 | 0.802 |
| Teapot | 0.0328 | 0.0187 | 0.569 |
| Tube | 0.0313 | 0.0248 | 0.791 |

Only the SPREAD across groups is interpretable (R1-ILA changes the feature scale, so absolute score levels are not comparable between F0 and R1).

## 10. Broad-generalization claim verdict

**HOLD**  
| pre-registered condition | value | satisfied |
|---|---|---|
| ≥ 7/10 categories with 3-seed mean ΔAUROC > 0 | 9/10 | True |
| macro mean ΔAUROC > 0 | +0.00760 | True |
| median ΔAUROC > 0 | +0.01080 | True |
| no category mean ΔAUROC < −0.03 | 1 severe | False |

## Claim boundary

- **May be claimed**: the unselected result of frozen R1-ILA across 10 M²AD categories × 3 seeds × view120 (per-category means, seed variability, win rates, CIs, all negatives).
- **May not be claimed**: causal attribution (illumination shift ↔ gain is an association at n=10); generalization beyond M²AD / this backbone / view120; image bootstrap CIs as if they were cross-seed uncertainty; a STOP verdict as 'paper failed'.