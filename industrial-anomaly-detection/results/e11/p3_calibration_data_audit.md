# E11-P3.1 · Calibration Data Audit

## Verdict: PASS (specimen-level isolation clean for all 3 categories)

| category | train specimens | bank specimens | val specimens | calibration specimens | test specimens |
|---|---|---|---|---|---|
| Bird | 30 | 10 | 2 | 20 | 70 |
| Car | 30 | 10 | 2 | 20 | 70 |
| Cube | 30 | 10 | 2 | 20 | 70 |

## Answers to the six compliance questions

1. **Is the 300-bank 300 normal train images?** YES — `random.Random(20261009).sample(1200,300)`
   over the 10-bank-specimen pool (1200 = 10 specimens x 12 views x 10 illum). Confirmed 300 images /
   10 distinct specimens.
2. **Does train normal data include all illumination IDs?** YES — bank and calibration both cover
   illuminations 01..10 (240 calibration images per illumination).
3. **Per-illumination normal specimen count:** 20 calibration specimens -> 20 x 12 views = **240 images
   per illumination** (each specimen contributes 12 view images per illumination).
4. **Specimen-level memory-bank vs calibration split possible?** YES — frozen stride split
   `bank = sorted(train)[::3][:10]`, calibration = the other 20 specimens. Verified
   bank-specimens ∩ calibration-specimens = 0.
5. **Test specimen overlap?** NONE — bank ∩ test = 0, calibration ∩ test = 0 (70 test specimens are all
   held-out; M2AD splits are specimen-disjoint).
6. **Test illumination ID from legitimate metadata?** YES — read from the raw metadata `illumination`
   field (01..10), never inferred from the image.

## Leakage audit
- Calibration uses ONLY the 20 non-bank TRAIN specimens (normal images). No test image, test label, or
  test normal score is used to fit the per-illumination median.
- The memory bank (10 specimens, 300 images) is UNCHANGED from E11-P2, so the frozen P2 baseline is not
  contaminated and does NOT need to be re-derived under a reduced bank.

## Addendum (after P3.1 run)
- **Bird**: calibration set (20 specimens, 2400 imgs) built from LOCAL originals -> l0_448 cache. PASS.
- **Car/Cube**: calibration set NOT constructible without downloading their originals (forbidden this
  round). Their cached images are only the 10 bank-pool specimens + test, so no specimen-disjoint normal
  calibration data exists. => **BLOCKED** (reported, not faked).
