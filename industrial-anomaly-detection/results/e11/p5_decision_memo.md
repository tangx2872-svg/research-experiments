# E11-P5 · Final Lightweight Screening — Decision Memo

## Verdict: **NO SURVIVOR** (mean ΔAUROC ≥ +0.010 AND ≥2/3 categories)

## Results (view120, seed0, official image_anomaly + detectable, 300-bank, frozen DINOv2-B 1-NN-max)

| module | direction | Bird | Car | Cube | mean ΔAUROC | mean ΔAP | #pos |
|---|---|---|---|---|---|---|---|
| D1a | K-means K=1024 | -0.0170 | -0.1462 | +0.0123 | -0.0503 | -0.0444 | 1/3 |
| D1b | K-means K=4096 | -0.0205 | -0.0838 | +0.0077 | -0.0322 | -0.0257 | 1/3 |
| D2a | k-NN k=3 | +0.0017 | +0.0030 | -0.0016 | +0.0010 | +0.0019 | 2/3 |
| D2b | k-NN k=5 | +0.0059 | -0.0021 | -0.0001 | +0.0012 | +0.0021 | 1/3 |
| D3a | CLS+local α=0.3 | -0.0347 | +0.0000 | +0.0032 | -0.0105 | -0.0175 | 2/3 |
| D3b | CLS+local α=0.5 | -0.0698 | -0.0001 | +0.0056 | -0.0214 | -0.0319 | 1/3 |

baseline = Bird 0.74725 / Car 0.78552 / Cube 0.85410 (reproduced exactly).

## Per-candidate finding
- **D1 (K-means prototype bank)**: harmful. Compressing the 307200-patch bank to 1024/4096 centroids loses
  discriminative detail, worst on Car (K=1024 -> -0.146). The full bank's diversity is load-bearing.
- **D2 (k-NN local consistency)**: effectively NEUTRAL (mean +0.001, within noise). k=3/k=5 averaging of
  neighbour distances does not move image-level AUROC; the 1-NN-max signal already captures the anomaly.
- **D3 (global CLS + local fusion)**: harmful on Bird (-0.035/-0.070), neutral elsewhere. The DINOv2 CLS
  token carries weak/noisy image-level anomaly signal; weighting it in dilutes the local patch evidence.

## Screening outcome
**NO SURVIVOR.** No candidate reaches the +0.010 / 2-of-3 bar; D2 is the only near-neutral one and is far
below threshold with no AP upside.

## Combined P4+P5 conclusion (6 directions, 13 configs)
Across P4 (intermediate fusion / smoothing / multiscale) and P5 (K-means / k-NN / global fusion), no
lightweight module on top of frozen DINOv2-B + 1-NN-max produces a stable, threshold-clearing, AP-positive
gain. The frozen DINOv2-B last-layer + 1-NN-max is near-optimal for this family.

## Recommendation
**STOP further module search.** Continued lightweight-module screening has low expected value. The remaining
real lever (established in E11-P2) is the representation itself (DINOv2 >> ResNet); any publishable small
improvement should come from a different axis than post-hoc module stacking (e.g. a trained adaptation head
or a different scoring formulation), not more frozen-feature modules. Awaiting human decision.
