# Paper claims — SAFE / LIMITED / FORBIDDEN

All SAFE numbers below are injected programmatically from `paper_main_table.csv` / `results/e22/*` (see `scripts/paper_ready_build.py`) and are re-checked by `scripts/e23_verify.py`.

## SAFE CLAIMS

- Across the complete 10-category M²AD evaluation (view120, 3 seeds, 300-bank), R1-ILA achieves positive mean ΔAUROC on **9 of 10** categories and positive mean ΔAP on **9 of 10** categories.
- **7 of 10** categories show a positive ΔAUROC across all three evaluated seeds.
- The macro-average and median ΔAUROC are **0.00760** and **0.01080** (macro-average ΔAP 0.00777).
- The frozen per-category paired-image bootstrap (n_boot = 3000) excludes zero for Bird, Ring and Teapot (and for Holder's ΔAP); the pre-registered STRONG-GO exclusion criterion is not met because Doll drops below −0.03, so the protocol verdict is **HOLD**.
- Doll is a reproducible regression: mean ΔAUROC **-0.03850** (3-seed mean; 0/3 positive seeds), paired-image bootstrap CI [-0.0661, -0.0140], p(ΔAUROC>0) = 0.001.
- Multi-view evaluation (E21, 4 views × 3 categories × seed0): 8 of 12 cells have ΔAUROC > 0; the macro pooled ΔAUROC is 0.00142 and ΔAP is -0.00165; E21 verdict = **HOLD**.
- E22 passed **36 of 36** independent verification checks with no rewriting of any stored score, and every negative cell is retained in the committed artifacts.
- E23 (Doll audit) verdict = **EXPLAINED_PATTERN**: in 3/3 seeds the anomaly-side score drop exceeds the normal-side drop and d' decreases; the degradation is broad across illumination groups (8-9 of 10 groups per seed).
- E24 (sensitivity audit of the FROZEN configuration, Bird/Car/Cube only): the frozen strength 0.85/1.15 ranks first of six strengths under a pre-registered stability rule (3/3 positive cells, best worst-case, seed0 mean +0.015401), and strengths 0.95/1.05-0.85/1.15 form a plateau; 3-seed re-runs of S15 and S10 are not separable (+0.013246 ± 0.014589 vs +0.015984 ± 0.014317). E24 does not redefine the frozen method.

## LIMITED CLAIMS (allowed only with the stated qualifier)

- *Results suggest broadly positive cross-category behaviour on M²AD.* (Qualifier: 9/10 categories, one reproducible regression; not universal.)
- *Multi-view experiments indicate that benefits are not restricted exclusively to view120, but substantial viewpoint heterogeneity remains.* (Qualifier: 8/12 cells positive; Car 240/330 negative; HOLD verdict.)
- *Score-drift diagnostics are consistent with reduced illumination-related variation in several evaluated settings.* (Qualifier: 'consistent with' only; a normal-dispersion increase is also observed, and BETA=0 shows the opposite failure mode.)
- *The Doll failure is consistent with a mild, near-symmetric score shift in which the anomaly mass falls slightly more than the normal mass.* (Qualifier: relative drops 0.95-1.00 of each other; no causal mechanism established.)
- *External evaluation on RobustAD indicates a stable sub-dataset dependence: PCB improves and MetalParts degrades.* (Qualifier: adapter re-trained per sub-dataset, not transferred; 11 valid target splits × 3 seeds = 33 cells; overall macro ΔAUROC is −0.001068 split-weighted and +0.000550 sub-dataset-weighted, i.e. the sign depends on the weighting, and the descriptive t-interval [−0.0033, +0.0012] over 3 per-seed macros includes 0; 2 of 3 sub-datasets covered.)

## FORBIDDEN CLAIMS

- Universal robustness across datasets, backbones or views
- Improves all categories / improves every category
- Viewpoint invariant / view-invariant improvement
- Every illumination group improves
- Causal proof of the illumination-shift mechanism
- Normal dispersion reduction
- Cross-dataset generalization (one conditional external evaluation exists; a general improvement claim is still forbidden)
- Statistically significant improvement on 9/10 categories
- Doll is random noise / an artefact to be ignored
- The method universally preserves anomaly information
- Direct cross-dataset transfer (the adapter is re-trained on each RobustAD sub-dataset's own source normals)
