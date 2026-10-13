# E11-P0 · Local Asset Matrix

## Already-tested (E5-E10) — excluded as duplicates
| method | repo | weights | train | tested | verdict |
|---|---|---|---|---|---|
| PatchCore | anomalib | wide_resnet50_2 (HF) | no | E4-E10 baseline | KEEP as baseline |
| Retinex/URetinex | piad_baseline | 4 .pth | no | E5-1 HOLD, E9 STOP | excluded |
| SimpleNet | third_party | none | yes | E5-1 STOP | excluded |
| CRAD | third_party | none | yes | E5-1 STOP | excluded |
| ReConPatch | none | none | yes | E5-1 STOP | excluded |
| SoftPatch | third_party | none | yes | E7 STOP | excluded |
| AnomalyDINO | third_party (LICENSE ok) | DINOv2-B (torch hub) | yes | E7 DINOv2-NN STOP | excluded (DINOv2 family weak) |
| DINOv2-S/B NN | anomalib | vit_small/vit_base | no | E7 STOP | excluded |
| CLAHE / GrayWorld / gamma / L2-norm / local-density | ours | n/a | no | E8/E9/E10 STOP | excluded |

## Candidate methods (E11, NOT yet tested here)
| method | repo | weights | deps ready? | needs train | backbone (cached?) | note |
|---|---|---|---|---|---|---|
| CDO | M2AD/model/cdo.py | none | partial | yes 100e | mshrnet32 (NOT cached) | needs DTD for DRAEM |
| Dinomaly | M2AD/model/dinomaly.py | none | partial | yes 100e | dinov2reg_vit_base_14 (NOT cached, Baidu) | reg weight needed |
| INP-Former | M2AD/model/inpformer.py | none | partial | yes 100e | dinov2reg_vit_base_14 | + 6 INP blocks |
| MSFlow | M2AD/model/msflow.py | none | **yes** | yes 100e | wide_resnet50_2 (cached) | no synthetic anomaly |
| RD++ | M2AD/model/rdpp.py | none | partial | yes 100e | wide_resnet50_2 (cached) | needs DTD for DRAEM |
| Dinomaly2 | github (arXiv 2510.17611, Oct 2025) | none | unknown | yes | DINOv2 | very new; unified framework |

## Cached weights (reusable)
- wide_resnet50_2 (HF timm) — for PatchCore baseline, MSFlow, RD++ teacher.
- DINOv2 vit_small/vit_base (torch hub) — for DINOv2-NN baseline, AnomalyDINO.
- Retinex ckpt (piad_baseline) — excluded.

## Cached data
- extracted/Bird (15 GB). Car/Cube only as e7_cache (l0_256/l0_448/clahe/retinex).
- metadata all 10 categories (meta_unsupervised.json).

## Recommendation
- **MSFlow** and **RD++** are the lowest-friction (wide_resnet50_2 cached). MSFlow needs no DTD.
- **Dinomaly / INP-Former / CDO** need a not-yet-cached backbone or DTD -> verify download reachability first.
- **Dinomaly2** is newest (Oct 2025) — inspect before committing.
