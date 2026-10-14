# E19-2 · Illumination groups 01-10 — notes

**Source (read-only)**: `results/e18/illum_breakdown.csv` (24 sources x 11 groups).
**No new GPU run, no model re-executed.**

**Claim served**: R1-ILA is *consistent with* reducing illumination-induced score drift:
the cross-illumination spread of good-image scores shrinks in the same direction in all three
categories. This is a descriptive, downstream (score-level) observation — **not a causal proof**.

**Figures**
- `figures/fig_illum_auroc.png/.pdf` — per-group AUROC, F0 vs Ours, three categories
  (descriptive; dotted = ALL).
- `figures/fig_illum_drift.png/.pdf` — (left) good-score drift relative to ALL; (right) spread
  max-min of good-image score across groups.
- `figures/fig_illum_context.png/.pdf` — appendix: every available source (Cube / Bird), context only.

**Facts — spread of good-image score across groups (seed 0)**

| category | F0 spread | Ours spread | change |
|---|---|---|---|
| Bird | 0.0358 | 0.0178 | -50.3% |
| Car | 0.0889 | 0.0692 | -22.1% |
| Cube | 0.0279 | 0.0192 | -31.1% |

- Ours = R1-ILA (BETA=0.1), F0 = frozen DINOv2-B 1-NN-max.
- The spread decreases in all three categories; per-group AUROC differences are noisy
  (per-group n_ok ~ 32 for Bird and n_ok ~ 26 for Car/Cube) and are reported descriptively only.
- Largest single-group drift: Car group 07, +34.0%
  (F0) vs +40.8% (Ours); that group has
  n_ok=32, n_ng=29. Note this single-group excursion is **larger under Ours**: the spread reduction
  above is a distribution-level property, **not** the removal of this peak. Recorded as a data
  fact; no mechanism is claimed for it.

**Reference (E18)**: identical values are reported in `results/e18/statistical_analysis.md` §B.1
(cross-checked in `../verification.csv`, group ILLUM).

**Interpretation boundary (mandatory wording)**
- Only the **spread / shape** across groups is interpretable. Absolute score levels are NOT
  comparable between F0 and Ours (R1-ILA changes the feature scale).
- This is reported as **"consistent with"** the E16 illumination-shift account. The correlation
  between the training objective (feature-level L_cons) and this downstream score-level quantity
  does **not** establish causality.
- Context sources (PatchCore-emb unified, PatchCore-official, AnomalyDINO, SoftPatch) are
  single-seed and have small per-group n; they are shown for context, never as a ranking.
