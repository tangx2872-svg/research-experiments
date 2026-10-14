# E18 Main Table — M²AD view120 / seed0 / 300-bank / official protocol

AUROC / AP are **full precision**. Empty cells are NOT estimated.

| method | Bird AUROC | Bird AP | Car AUROC | Car AP | Cube AUROC | Cube AP | complete | seeds | scoring |
|---|---|---|---|---|---|---|---|---|---|
| PatchCore-emb + unified scoring | 0.68454 | 0.72797 | 0.62247 | 0.59298 | 0.60857 | 0.66489 | 3/3 | 1 | 1-NN-max (unified) |
| AnomalyDINO | 0.70214 | 0.72234 | 0.75684 | 0.78277 | 0.84889 | 0.91134 | 3/3 | 1 | official mean_top1p (DINOv2-448, L2 kNN k=1) |
| SoftPatch | 0.64396 | 0.57510 | 0.50219 | 0.55541 | 0.69343 | 0.75908 | 3/3 | 1 | official soft-weighted kNN (WeightedGreedyCoresetSampler, nearest) |
| EfficientAD | 0.61206 | 0.62196 | — | — | — | — | 1/3 | 1 | official teacher-student discrepancy (PDN-small) |
| F0 | 0.74725 | 0.78863 | 0.78552 | 0.84207 | 0.85409 | 0.91686 | 3/3 | 3 | 1-NN-max |
| Ours (R1-ILA) | 0.75786 | 0.79911 | 0.80511 | 0.85502 | 0.87010 | 0.92856 | 3/3 | 3 | 1-NN-max (on adapted features) |

## 3-seed extension (F0 / Ours only)

| category | metric | F0 mean ± std (n=3 seeds) | Ours mean ± std (n=3 seeds) |
|---|---|---|---|
| Bird | AUROC | 0.75131 ± 0.00427 | 0.77717 ± 0.01757 |
| Bird | AP | 0.79084 ± 0.00193 | 0.81456 ± 0.01343 |
| Car | AUROC | 0.79979 ± 0.01423 | 0.81102 ± 0.00810 |
| Car | AP | 0.85001 ± 0.00688 | 0.86117 ± 0.00543 |
| Cube | AUROC | 0.86380 ± 0.00906 | 0.86645 ± 0.00582 |
| Cube | AP | 0.92213 ± 0.00493 | 0.92625 ± 0.00261 |

**Boundary**: the primary comparison is seed0 for every method (the only seed all methods share). The 3-seed columns quantify seed variance for F0/Ours only; they are not a claim that the single-seed baselines are seed-stable.

All main-table cells above are **seed0** except the 3-seed extension block.
