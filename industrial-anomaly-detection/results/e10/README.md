# E10 — Local-Manifold Calibrated Scoring for Illumination Robustness

```text
📍 PAPER STAGE          Phase III — Final Candidate Stability Validation (last cheap-modification test)
🎯 QUESTION             Does local-manifold scale-calibration of PatchCore NN distance reduce
                        illumination-induced false anomalies while preserving defect separation,
                        AND does it break the Bird↑/Car↓/Cube↓ category-specific pattern?
🚦 EXIT                 NO SURVIVOR (Round-1 ADVANCE S4/S3 not confirmed at 3 seeds x 2 views)
```

## 1. Question

E8/E9 established that preprocessing, illumination normalisation and feature normalisation all
produce category-specific (never cross-category-robust) gains. E10 tests the last cheap
hypothesis: that illumination shift inflates the absolute NN distance of normal patches through a
local change in manifold density/scale, so **calibrating the anomaly distance by the local normal
scale** should help.

## 2. Candidates (frozen, scoring-only)

Input / backbone / feature extraction / memory bank / coreset are untouched. Only patch-level
score aggregation changes.

| id | mechanism |
|---|---|
| S0 | Original (frozen softmin-reweighted max d1) — baseline |
| S1 | kNN3 mean: (d1+d2+d3)/3 |
| S2 | kNN5 trimmed: (d2+d3+d4)/3 (drop min/max of top-5) |
| S3 | local density ratio k=3: d1 / (rho3(m*) + 1e-8) |
| S4 | local density ratio k=5: d1 / (rho5(m*) + 1e-8) |
| S5 | normal ECDF calibration of S0 (monotone remap, control) |

`rho3/rho5(m)` = mean distance from memory patch `m` to its 3/5 nearest OTHER memory patches
(normal-only, computed from the coreset). `eps=1e-8`, never tuned.

## 3. Sanity

- **S0 equivalence vs frozen `predict_one` = 0.000e+00** (bit-exact) after seeding the coreset RNG
  with `np.random.seed(0)` (an earlier un-seeded run gave a different coreset; fixed pre-result).
- S1–S4 manual recompute max error = **0.0**; rho finite & > 0; S5 ECDF monotone / [0,1] / normal-only
  (reference = the 900 un-sampled bank-pool normal images, held out from the 300-bank).

## 4. Round 1 (3 cats x seed0 x view120, one fit per category -> 6 methods)

| candidate | Bird Δ | Car Δ | Cube Δ | mean | Δd' | pos | verdict |
|---|---:|---:|---:|---:|---:|---:|---|
| **S4** | +0.0039 | +0.0263 | +0.0188 | **+0.0163** | +0.0118 | 3/3 | ADVANCE |
| **S3** | -0.0004 | +0.0238 | +0.0069 | +0.0101 | -0.0067 | 2/3 | ADVANCE |
| S5 | -0.0002 | +0.0003 | +0.0046 | +0.0016 | -0.1052 | 2/3 | HOLD |
| S1 | +0.0037 | -0.0030 | +0.0005 | +0.0004 | +0.0078 | 2/3 | HOLD |
| S2 | +0.0067 | -0.0205 | -0.0035 | -0.0057 | -0.0057 | 1/3 | STOP |

Top-2 (ADVANCE) = **S4, S3** — the local-density-ratio family was, on Round-1 evidence, the first
candidate across E5/E8/E9 to look cross-category positive (S4: 3/3 categories up).

## 5. Round 2 (Top-2 = S4/S3, 3 cats x 3 seeds x 2 views)

| candidate | Bird | Car | Cube | overall | mean Δd' | pos_cat | worst | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| S4 | -0.0130 | +0.0321 | +0.0005 | +0.0065 | -0.0082 | 2/3 | **-0.0130** | STOP |
| S3 | -0.0094 | +0.0236 | -0.0051 | +0.0031 | -0.0220 | 1/3 | -0.0094 | STOP |

Round-1's S4 "3/3 positive" **flipped**: Bird went from +0.0039 (view120) to **-0.0130**
(2-view mean, 3/3 seeds negative; view 090 alone is -0.0246). The gain is entirely **Car-driven**
(+0.032, 3/3 seeds positive); Cube is ~0 and Bird is clearly negative.

## 6. Verdict — NO SURVIVOR

S4 fails the frozen `worst >= -0.010` GO criterion (Bird -0.0130); S3 is single-category (Car).
Both STOP ⇒ E10 = NO SURVIVOR.

## 7. Scientific interpretation (answers to the frozen questions)

1. **kNN aggregation (S1/S2) does not help** — S1 ≈ 0, S2 negative.
2. **Local-density calibration** improves the *hard* categories (Car +0.03, Cube ~0) but **hurts
   Bird** (-0.01). It is a genuine *scoring* effect (not a monotone remap), but still
   category-specific.
3. **S3 vs S4 direction agree** (both Car-positive, Bird-negative) — the family behaves
   consistently, so it is not a random artefact, but it is not universal.
4. **ECDF remapping (S5) is ineffective** — AUROC is invariant under monotone transforms, and its
   Δd' is strongly negative (-0.11), confirming remapping alone is insufficient.
5. **The category-specific pattern persists** — it is merely *relocated*: E8/E9 gave Bird↑/Car↓/Cube↓,
   E10 gives Car↑/Bird↓/Cube~0.
6. **Scoring modification does NOT break the category-specific pattern.**

## 8. Final cross-round conclusion

After **E5 (module insertion, 0 GO) + E8 (CLAHE, Bird-only) + E9 (normalisation, NO SURVIVOR) +
E10 (local-manifold scoring, NO SURVIVOR)**, there is **no sufficient reason to keep searching for
cheap PatchCore modifications**. Every cheap transform — input, feature, or score level — produces
category-specific gains that do not generalise across M²AD categories. This is a robust, replicated
negative result.

## 9. Cost

Round 1 ≈ 6 min (3 fits) · Round 2 ≈ 16 min (9 fits) · sanity ≈ 5 min. Peak VRAM ≈ 1.8 GB.
Scoring-only: no new caches; baseline S0 reproduced E8 Original bit-exactly.
