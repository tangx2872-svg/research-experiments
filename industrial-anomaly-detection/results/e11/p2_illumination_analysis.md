# E11-P2 · Illumination Analysis

Unified 1-NN-max scoring, official image_anomaly + detectable filter, view120. Scores are cosine
distance (1 - max cosine similarity to nearest bank patch), max-pooled over 1024 patches.

## Normal-vs-anomaly score separation (mean)

| backbone | Bird (n/a) | Car (n/a) | Cube (n/a) |
|---|---|---|---|
| resnet  | 0.116 / 0.137 | 0.066 / 0.074 | 0.142 / 0.150 |
| dinov2s | 0.245 / 0.299 | 0.153 / 0.212 | 0.176 / 0.261 |
| dinov2b | 0.280 / 0.354 | 0.186 / 0.273 | 0.223 / 0.334 |

Gap (anomaly - normal): ResNet ~0.008-0.020; DINOv2-S ~0.054-0.084; DINOv2-B ~0.074-0.111.
**DINOv2 yields a 3-6x larger normal-anomaly gap** => the AUROC gap is driven by better separation,
not by any label/scoring artefact.

## Does normal score vary with illumination? (P2.5-C) — YES
Per-illumination normal-score mean spread (max-min across the 10 illuminations):

| backbone | Bird | Car | Cube |
|---|---|---|---|
| resnet  | 0.023 | 0.057 | 0.008 |
| dinov2s | 0.050 | 0.065 | 0.020 |
| dinov2b | 0.036 | 0.089 | 0.028 |

**Illumination 07 is consistently the highest normal-score illumination across backbones and
categories** (e.g. dinov2b/Car: I07=0.249 vs I09=0.164). This is a real illumination-specific
hard-condition signal, present in BOTH ResNet and DINOv2 features. It is a property of the data,
not of a specific backbone.

## Anomaly-normal overlap (P2.5-D)
ResNet's tiny gap (esp. Car 0.008, Cube 0.008) means heavy overlap (low AUROC 0.62/0.61). DINOv2's
larger gap reduces overlap (higher AUROC 0.75-0.85). No backbone shows a clean bimodal separation;
all remain partially overlapping (consistent with image-level AD on illumination-varying data).

## Interpretation (observation vs hypothesis, kept separate)
- **Observation**: DINOv2-S/B representation improves image-level AUROC over ResNet on ALL 3
  categories, under identical scoring and labels.
- **Observation**: illumination 07 raises normal scores for every backbone.
- **Hypothesis (not proven)**: the residual normal-score illumination drift is what an
  illumination-invariant method should target.
