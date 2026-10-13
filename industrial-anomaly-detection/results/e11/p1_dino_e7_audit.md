# E11-P1 · E7 DINOv2 Evidence Review (read-only)

Source: `scripts/e7_pipelines.py::run_dinov2` + `results/e7_1/summary.csv`.

## What E7 actually ran
- Model: `torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14' | 'dinov2_vitb14')` — **FROZEN**, no projection/decoder, no training.
- Input: 448 BICUBIC + ImageNet norm + patch-grid crop (AnomalyDINO `prepare_image` semantics).
- Features: `get_intermediate_layers()[0]` = final-layer patch tokens (no CLS).
- Scoring: L2-normalised, cosine distance to 1-NN, image score = **mean of top-1% patch distances** (`mean_top1p`). NOT PatchCore coreset+kNN.
- Scope: **Bird only**, view120, seed0, 300-bank.
- Result (vs PatchCore P00=0.78089): P05 DINOv2-S −0.0692, P06 DINOv2-B −0.0493 → STOP.

## Reuse verdict for E11
- **NOT a PatchCore-equivalent scoring** — it is AnomalyDINO scoring (cosine 1-NN, top-1% mean).
- **Bird-only** — no Car/Cube numbers; cannot serve as the cross-category frozen-DINOv2 control as-is.
- **Reusable as** a frozen-DINOv2-NN Bird reference (already run). For Car/Cube, it would need re-run;
  the l0_448 cache (square 448 of 1024² images) is input-equivalent to `_dinov2_prepare` for square M²AD images, so re-run does NOT require raw Car/Cube images.
- **Verdict**: NOT a direct E11 Frozen DINOv2-NN control (different scoring + Bird-only), but the
  frozen-DINOv2-vs-PatchCore gap on Bird is already measured (DINOv2 worse). E11 must NOT conclude
  "DINOv2 family invalid" from E7 alone — E7 only tested frozen kNN, not a trained DINOv2 head (Dinomaly/INP-Former).
