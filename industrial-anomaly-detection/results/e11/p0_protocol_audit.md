# E11-P0 · M²AD Official Protocol Audit

Official repo: `github.com/hustCYQ/M2AD` (shallow-cloned to `third_party/M2AD`, HEAD `a606aa0`,
**no LICENSE file**). Paper: arXiv 2505.10996 (accepted Pattern Recognition 2026).

## 1. Official M²AD structure

- Dataset: 10 categories, 999 specimens, **12 views × 10 illuminations = 120 configs**, 119,880 imgs.
- Two protocols:
  - **M2AD-Synergy**: multi-configuration information fusion (uses ALL images).
  - **M2AD-Invariant**: single-image robustness (our interest; closest to E4–E10).
- Metadata: `jsons/meta_unsupervised.json` (identical file to what we already use).
- Split: `train` = Good only (unsupervised), `test` = Good + NG. Class list = 10 categories.

## 2. M2AD-Invariant train/test construction (from `data/ad_dataset.py`)

1. `meta_info = meta_unsupervised.json['train'|'test']`.
2. train = all train records (Good only); `random.shuffle` for train.
3. test = all test records **then filtered: keep only `detectable != False`** (i.e. `''` or `True`).
   - This **excludes non-detectable NG images** from evaluation.
4. Labels: `image_anomaly` (0/1). masks loaded from `mask_path` for NG, zero-mask for Good.
5. Synthetic anomaly generation (`anomaly_generator`) used during TRAIN by CDO/RD++ (DRAEM) and
   MSFlow (white_noise, disabled by default).

## 3. Metrics

`evaluator` uses `use_adeval=True` → **adeval** library computes image/pixel AUROC, AUPRO, AP
(`max_step_aupro=100`). Standard unsupervised AD metrics.

## 4. Official methods (configs/benchmark/)

| method | backbone | batch | epochs | synthetic anomaly | notes |
|---|---|---|---|---|---|
| CDO | mshrnet32 (teacher) | 12 | 100 | DRAEM (DTD) | mshrnet32 NOT in our cache |
| Dinomaly | dinov2reg_vit_base_14 | 12 | 100 | none | needs reg-weight (Baidu link) |
| INP-Former | dinov2reg_vit_base_14 | 12 | 100 | none | + 6 INP blocks |
| MSFlow | wide_resnet50_2 (internal) | 24 | 100 | white_noise (disabled) | backbone we already have |
| RD++ | wide_resnet50_2 + de_wide_resnet50_2 | 8 | 100 | DRAEM (DTD) | backbone we have; needs DTD |

All 5 require **full training** (100 epochs). No pre-trained checkpoints shipped in the repo.

## 5. Protocol differences vs our E4–E10 (audit only, NOT a change proposal)

| dimension | our E4–E10 | official M2AD-Invariant |
|---|---|---|
| test set | ALL test images @ frozen view(s) (700 = 200 Good + 500 NG) | ALL test images, **minus non-detectable NG** (Bird −365 / Car −309 / Cube −222) |
| view/illum | fixed view (120) or 4 frozen views | all 12 views × 10 illum (single-image scoring) |
| train | 300-image bank (coreset 0.1) | ALL normal train images (3600/category) |
| baseline | PatchCore (wide_resnet50_2, L2+L3, k=9) | deep methods (CDO/Dinomaly/INP/MSFlow/RD++) |
| resolution | 256 | 256 or 512 (config-dependent) |
| metrics | image AUROC (primary) + d′/AUPR/pixel | adeval image/pixel AUROC + AUPRO + AP |

**Key point**: our `detectable` handling differs — we score non-detectable NG as anomalies;
official Invariant drops them. Any E11 method benchmarked against our E8/E9/E10 numbers must either
(a) match our `detectable`-inclusive test set, or (b) explicitly switch to the official filter and
recompute baselines. **This must be decided by the human before any E11 run — do not silently change.**

## 6. Benchmark numbers

Official per-method tables are in the paper (arXiv 2505.10996, Tables T1/T2) and `Imgs/T1.png`,
`Imgs/T2.png` (figures, not machine-readable). Bird/Car/Cube image-AUROC per method would need to be
extracted from the paper. **Not reconstructed here** (avoid transcription errors).
