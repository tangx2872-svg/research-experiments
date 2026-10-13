# E11-P0 · M2AD Official Protocol Audit

Official repo: `github.com/hustCYQ/M2AD` (shallow-cloned to `third_party/M2AD`, HEAD `a606aa0`, **no LICENSE**).
Paper: arXiv 2505.10996 (Pattern Recognition 2026).

## 1. Structure
- 10 categories, 999 specimens, **12 views x 10 illum = 120 configs**, 119,880 imgs.
- Two protocols: **M2AD-Synergy** (multi-config fusion) / **M2AD-Invariant** (single-image robustness).
- Metadata: `jsons/meta_unsupervised.json` (same file we use). Split: train=Good only, test=Good+NG.

## 2. M2AD-Invariant construction (`data/ad_dataset.py`)
1. meta = meta_unsupervised.json['train'|'test'].
2. train = all train (Good); test = all test **filtered to `detectable != False`** (drops non-detectable NG).
3. label = `image_anomaly`; mask from `mask_path` (NG) / zero (Good).
4. synthetic anomaly (DRAEM for CDO/RD++, white_noise disabled for MSFlow) used during TRAIN.

## 3. Metrics
adeval (`use_adeval=True`) -> image/pixel AUROC, AUPRO, AP.

## 4. Official methods (configs/benchmark/)
| method | backbone | batch | epochs | synthetic anomaly |
|---|---|---|---|---|
| CDO | mshrnet32 (teacher) | 12 | 100 | DRAEM (DTD) |
| Dinomaly | dinov2reg_vit_base_14 | 12 | 100 | none |
| INP-Former | dinov2reg_vit_base_14 | 12 | 100 | none |
| MSFlow | wide_resnet50_2 | 24 | 100 | disabled |
| RD++ | wide_resnet50_2 + de_wide_resnet50_2 | 8 | 100 | DRAEM (DTD) |

All require full 100-epoch training; no shipped checkpoints.

## 5. Protocol differences vs our E4-E10 (audit only, NOT a change)
| dimension | ours | official Invariant |
|---|---|---|
| test set | ALL test @ frozen view(s) (700) | ALL test minus non-detectable NG (Bird -365/Car -309/Cube -222) |
| view/illum | fixed 120 or 4 frozen | all 12 x 10 |
| train | 300-image bank (coreset 0.1) | all 3600 normal |
| baseline | PatchCore | deep methods |
| resolution | 256 | 256 or 512 |

Key: we score non-detectable NG; official drops them. Any E11 benchmark vs our E8/E9/E10 numbers
must either keep our detectable-inclusive test or explicitly switch filter + recompute baselines.
**Human must decide before E11 runs.**

## 6. Benchmark numbers
In paper Tables T1/T2 (arXiv 2505.10996) + Imgs/T1.png,T2.png (figures). Not machine-readable;
Bird/Car/Cube per-method AUROC not transcribed here (avoid errors).
