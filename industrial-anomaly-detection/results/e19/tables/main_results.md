# E19 Main results — M²AD view120 / seed 0 / 300-bank / official image protocol

AUROC / AP at full precision; empty cells are `not run` (NOT estimated).
F0 vs Ours additionally carry 3 seeds (E16); all baselines are single-seed.

| method | Bird AUROC | Bird AP | Car AUROC | Car AP | Cube AUROC | Cube AP | complete | seeds |
|---|---|---|---|---|---|---|---|---|
| PatchCore-emb + unified scoring | 0.68454 | 0.72797 | 0.62247 | 0.59298 | 0.60857 | 0.66489 | 3/3 | 1 |
| AnomalyDINO | 0.70214 | 0.72234 | 0.75684 | 0.78277 | 0.84889 | 0.91134 | 3/3 | 1 |
| SoftPatch | 0.64396 | 0.57510 | 0.50219 | 0.55541 | 0.69343 | 0.75908 | 3/3 | 1 |
| EfficientAD | 0.61206 | 0.62196 | — | — | — | — | 1/3 | 1 |
| F0 | 0.74725 | 0.78863 | 0.78552 | 0.84207 | 0.85409 | 0.91686 | 3/3 | 3 |
| Ours (R1-ILA) | 0.75786 | 0.79911 | 0.80511 | 0.85502 | 0.87010 | 0.92856 | 3/3 | 3 |

Notes:
- PatchCore official scoring (native) is Bird-only: Car/Cube `BLOCKED` (raw 1024px inputs unavailable); see `results/e18/native_scoring_appendix.csv`.
- EfficientAD Car/Cube: `EARLY_STOP` (E15 pre-registered Stage-A STOP).
- Blank cells are gaps, never imputed. Values verified against `results/e18/main_table.csv` at 1e-12 (`../verification.csv`, group MAIN).
- All methods keep their native scoring; the main table is not a single-scoring leaderboard.
