# E11-P3.1 · Decision Memo

## Result
Per-illumination normal median centering on frozen DINOv2-B 1-NN-max (view120, official image_anomaly +
detectable): **Bird AUROC 0.74725 -> 0.75090 (+0.00365)**, AP 0.78863 -> 0.79470. Global-median sanity
control reproduces raw exactly (0.74725), confirming the implementation is correct and the gain is a real
per-illumination effect, not an artefact.

## What was learned (observation vs inference kept separate)
- **Observation**: DINOv2-B's raw normal score already has small illumination spread (0.038) — the
  illumination-conditioning problem is LARGELY absorbed by the representation itself (established in P2).
- **Observation**: centring that residual offset recovers only +0.0037. The mechanism is purely
  cross-illumination re-ranking (within-illum AUROC unchanged on all 10 rows).
- **Inference (not proven)**: illumination-conditioned calibration is a marginal refinement, NOT the main
  lever. The main lever was the representation (DINOv2 >> ResNet, P2).

## P3.2 recommendation: **INCONCLUSIVE, leaning NO-GO**
- The calibration method is technically sound and its mechanism is verified, but the gain is marginal
  (+0.0037) because DINOv2-B already suppresses most illumination variation.
- It REQUIRES illumination metadata at inference (deployment limitation).
- Cross-category verification (Car/Cube) is BLOCKED: their 20 non-bank normal specimens were never
  downloaded and downloading originals is forbidden in this round.
- Recommendation: do NOT invest further in per-illumination / learnable calibration heads. Redirect effort
  to the representation-level direction (DINOv2-family trained methods), where P2 showed the large effect.
