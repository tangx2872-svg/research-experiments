# E11-P1 · Experiment Protocol Draft (NOT executed)

## Stage A — Representation capability diagnosis
Candidates: PatchCore (frozen baseline) / Frozen DINOv2-NN (E7, Bird-only; extend Car/Cube via l0_448) /
Dinomaly / MSFlow (optional).
Precondition: ONE test set + ONE label convention + ONE metric implementation.
**Decision needed (human)**: adopt official `image_anomaly` + `detectable` filter (recommended), or keep legacy `object_anomaly`. Baselines already computed for both (p1_baseline_dual_protocol.csv).

## Stage B — Method validation (reserved)
- Base / Base + Stylization A / Base + Structure Consistency B / Base + A+B.
- A/B implementation + training objectives + params NOT frozen (await literature + novelty check).
- Prohibited: test-label-driven hyperparameter selection.

## Stage C — Robustness verification
- Bird / Car / Cube; multi-seed; multi-view; per-category AUROC + worst-category Δ; mean Δ + seed variance; illumination grouping.

## Pre-registered verdict draft (await PI confirmation)
- GO: mean ΔAUROC ≥ +0.005, ≥2/3 categories positive, worst ≥ −0.010, Δd′ ≥ −0.03, multi-seed stable.
- HOLD: +0.002~+0.005 with 2/3 positive.
- STOP: ≤ +0.002, single-category, seed collapse, worst < −0.010, or preservation degradation.
(These mirror E10; final thresholds to be confirmed.)

## BLOCKED items
- Car/Cube RAW images needed for any TRAINING (Dinomaly/MSFlow). Only caches exist (inference-capable via l0_256/l0_448, training-inadequate). Raw = ~15 GB/category, ~17 min/category stream-extract.
- mmcv/mmdet/adeval + FrEIA + DINOv2-reg weight + (RD++/CDO) DTD needed for the official methods.
