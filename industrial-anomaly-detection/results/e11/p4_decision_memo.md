# E11-P4 Round 1 · Lightweight Module Screening — Decision Memo

## Verdict: **NO SURVIVOR** (strict pre-registered rule: mean ΔAUROC ≥ +0.010 AND ≥2/3 categories)

## Results (view120, seed0, official image_anomaly + detectable, 300-bank, frozen DINOv2-B 1-NN-max)

| module | direction | Bird | Car | Cube | mean ΔAUROC | mean ΔAP | #pos |
|---|---|---|---|---|---|---|---|
| A1 | concat(last, block9) | +0.0090 | +0.0089 | +0.0055 | **+0.0078** | -0.0043 | 3/3 |
| A2 | concat(last, block7) | +0.0083 | +0.0206 | -0.0389 | -0.0033 | -0.0161 | 2/3 |
| B1 | 3x3 smooth | -0.0985 | -0.1662 | -0.0424 | -0.1024 | -0.1210 | 0/3 |
| B2 | 5x5 smooth | -0.1645 | -0.1985 | -0.1127 | -0.1585 | -0.1884 | 0/3 |
| C1 | 448+224 | -0.0232 | -0.0902 | +0.0179 | -0.0318 | -0.0390 | 1/3 |
| C2 | 448+336 | -0.0119 | -0.0524 | -0.0073 | -0.0239 | -0.0267 | 0/3 |

baseline (DINOv2-B last-layer) = Bird 0.74725 / Car 0.78552 / Cube 0.85410 (reproduced exactly from P2).

## Per-direction finding
- **A (intermediate+last fusion)**: the ONLY direction with any positive signal. A1 (block9) is positive on
  3/3 categories but small (+0.006..+0.009, mean +0.0078) — a consistent but sub-threshold near-miss; its AP
  is slightly negative, so the AUROC gain does not translate into precision. A2 (block7) is unstable
  (Cube -0.039) -> too shallow.
- **B (local neighbourhood smoothing)**: strongly harmful (mean -0.10..-0.16). Interpretation (inference):
  the DINOv2-B anomaly signal is LOCALISED; averaging neighbouring patch tokens blurs the fine-grained
  defect evidence.
- **C (multiscale)**: harmful (mean -0.02..-0.03). The coarser 224/336 features are less discriminative and
  the nearest-neighbour upsampling to 32x32 introduces blocky artefacts; a single exception (C1 on Cube)
  is not reproducible across categories.

## Screening outcome
- **Survivors: NONE.** No module clears mean ≥ +0.010 AND ≥2/3 categories.
- **Near-miss: A1** — 3/3 positive, mean +0.0078 (just under +0.010), but AUROC gain does not carry to AP.

## Resource
- Runtime ~20 min total (3 categories x 7 configs). Peak VRAM 10.5 GB. No new downloads, no training,
  no history modification. Disk ~16 GB free.

## Recommendation (awaiting human decision for Round 2 — NOT auto-started)
The frozen DINOv2-B last-layer + 1-NN-max is already near-optimal for this screening family; none of the
three lightweight modules (intermediate fusion, smoothing, multiscale) produces a threshold-clearing,
AP-positive gain. A1 (block9 concat) is the single candidate worth a second look IF the human accepts a
lower bar, but its negative ΔAP argues against promotion. No further tuning performed.
