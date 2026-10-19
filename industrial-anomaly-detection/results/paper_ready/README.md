# results/paper_ready/ — Paper-Ready Evidence Pack (Stage 3)

**Created**: 2026-10-16. **Scope**: consolidation only — no new experiment, no GPU, no method change.

This directory contains the paper-facing tables, figures, claims and limitations, all built deterministically from FROZEN artifacts by `scripts/paper_ready_build.py` and re-checked by `scripts/e23_verify.py` (`results/e23/verification.csv`).

| File | Content | Source of truth |
|---|---|---|
| `paper_main_table.csv` | 10 categories × (F0/R1 AUROC·AP mean±std, ΔAUROC/ΔAP mean±std, positive seeds) + **Macro Avg** row (std cells intentionally blank in that row) | `results/e22/main_table_10cat_3seed.csv` |
| `paper_ablation_table.csv` | E17 BETA ablation (5 configs × 3 categories) with the corrected BETA=0 wording in the `note` column | `results/e17/summary.csv` |
| `paper_multiview_table.csv` | E21 12 cells (3 categories × 4 views), including all negative cells; `e21_verdict = HOLD` on every row | `results/e21/analysis_cells.csv` |
| `paper_claims.md` | SAFE / LIMITED / FORBIDDEN claims (numbers injected from the tables) | `results/e22/*`, `results/e21/analysis_overall.csv` |
| `paper_limitations.md` | L1–L6 limitation material | E22/E21/E20/E23/E25 |
| `paper_evidence_map.md` | Claim → evidence → files → boundary (Claims A–H) | all of the above |
| `EVIDENCE_MAP.md` | **Single authoritative 13-section evidence map** (frozen method → M²AD → main results → ablation → E24 → multi-seed → E25 → PCB/MP → CAN/CANNOT → open hypotheses → exact source files); hand-curated, not regenerated | E17/E18/E21/E22/E23/E24/E25 |
| `figures/` + `figure_manifest.csv` | Main Fig A–C, Supplement Fig D–E (+2 optional), copied (never redrawn) with provenance | `results/e22|e21|e23/figures/` |

**Hard rules for anyone writing the paper from this pack**:
1. Doll must never be removed from the main table; `paper_main_table.csv` contains it by construction (verified).
2. E22 protocol verdict stays **HOLD**; it must not be silently rewritten as STRONG GO, nor described as a method failure.
3. E21 stays **HOLD**; Car 240/330 and Cube 090/330 must remain visible.
4. Image bootstrap CIs are test-set sampling uncertainty, never cross-seed uncertainty.
5. Numbers must be quoted from these tables (or from the frozen E22/E17/E21 artifacts they mirror), not retyped from prose.
6. Dataset names: M²AD is the main evaluation dataset (10 categories); Bird/Car/Cube are 3 of those 10 and are the method-development categories. Never call them an external or second dataset. RobustAD is the only independent external dataset in this chain.
7. E25 re-trains the adapter on each RobustAD sub-dataset: write 'external dataset validation' or 'external evaluation', never 'direct cross-dataset transfer'.
8. Quote the RobustAD overall result with its weighting: −0.001068 split-weighted vs +0.000550 sub-dataset-weighted (never 'domain-balanced'). Both come from full-precision frozen values; the sign is weighting-dependent, so no overall direction may be claimed.

```bash
cd /root/autodl-tmp/research-experiments/industrial-anomaly-detection
/root/miniconda3/bin/python -u scripts/paper_ready_build.py   # rebuild all of the above
/root/miniconda3/bin/python -u scripts/e23_verify.py          # independent verification
```
