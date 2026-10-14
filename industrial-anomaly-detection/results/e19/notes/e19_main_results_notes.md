# E19-3 · Main results table + figures — notes

**Source (read-only)**: `results/e18/main_table.csv` (final main table, full precision),
`results/e18/bootstrap_f0_vs_ours.json` (paired bootstrap), `results/e18/native_scoring_appendix.csv`.
**No new GPU run.**

**Claim served**: under the M²AD view120 / seed 0 / 300-bank / official image protocol,
R1-ILA (BETA=0.1) is above every comparable baseline in all three categories, and the
F0-vs-Ours gap is supported by a paired per-image bootstrap on the two methods that share
3 seeds and an identical bank/feature pipeline.

**Tables**
- `tables/main_results.csv` / `main_results.md` — final 6-row main table (full precision).
- `tables/main_results_3seed.csv` — 3-seed extension (F0 / Ours only) + per-seed deltas + CIs.

**Figures**
- `figures/fig_main_auroc.png/.pdf` — per-category AUROC bars, all methods; missing cells are
  marked with a neutral dashed frame and labelled (never imputed).
- `figures/fig_main_forest.png/.pdf` — paired bootstrap deltas (Ours − F0), AUROC and AP.
- `figures/fig_main_3seed.png/.pdf` — 3-seed mean ± std for F0 / Ours.

**Facts — seed-mean delta AUROC (Ours − F0), paired bootstrap**

| category | Ours AUROC (seed 0) | F0 AUROC (seed 0) | delta seed-mean [95% CI] | P(delta>0) |
|---|---|---|---|---|
| Bird | 0.75786 | 0.74725 | +0.02586 [+0.00833, +0.04310] | 0.998 |
| Car | 0.80511 | 0.78552 | +0.01123 [-0.00412, +0.02615] | 0.933 |
| Cube | 0.87010 | 0.85409 | +0.00265 [-0.01171, +0.01633] | 0.637 |

**Mandatory boundaries**
1. Baselines are **single-seed**; only F0 / Ours have 3 seeds. No claim that baselines are
   seed-stable.
2. The per-image bootstrap quantifies **test-set sampling uncertainty only** — it is NOT
   cross-seed uncertainty (seed variance is the ± std over 3 independent banks).
3. Empty cells are explicit gaps (`BLOCKED` / `EARLY_STOP`); no imputation, and the main table
   is not a single-scoring leaderboard (each method keeps its native scoring).
4. The main-table protocol on view120 uses official labels `image_anomaly` + `detectable`;
   legacy `object_anomaly` numbers (e.g. PatchCore Bird 0.78089) are excluded and quarantined
   in `results/e18/native_scoring_appendix.csv`.
5. Results are limited to M²AD Bird/Car/Cube, view120, 300-image bank; no cross-dataset or
   cross-view generalization is claimed here.
