# E19-1 · BETA ablation — notes

**Source (read-only)**: `results/e17/e17_partial.json` (full-precision sweep configs, seed 0),
`results/e16/e16_partial.json` (full-precision F0 reference), `results/e17/final_method.json`
(frozen advancement gates). **No new GPU run.**

**Claim served**: R1-ILA's only explicit loss parameter BETA is a monotone detection ↔
illumination-consistency knob; the frozen BETA=0.1 is an aggregate interior optimum, and
BETA=0 (pure consistency, no frozen anchor) degrades the representation (Cube collapse flag;
lowest d' in all three categories).

**Figures**
- `figures/fig_beta_auroc.png/.pdf` — AUROC vs BETA per category (seed 0); dotted = frozen F0.
- `figures/fig_beta_decision.png/.pdf` — (left) challenger delta vs frozen BETA=0.1 with the
  pre-registered +0.003 advance gate and -0.003 Cube safety line; (right) detection vs
  robustness_score trade-off, red ring = BETA=0.
- `figures/fig_beta_collapse.png/.pdf` — collapse diagnostics (pairwise cosine, std ratio,
  steps used) vs BETA.

**Facts (full precision, seed 0)** — read the AUROC and d' statements together:
- BETA=0 (C1) is the **worst challenger by AUROC on Bird and Car**
  (0.7331 / 0.7859) and the **lowest by d' in all
  three categories** (0.8013 / 0.8625 /
  1.4489).
- On **Cube**, BETA=0 has the nominally **highest** AUROC among the sweep
  (0.9090) — but it is the only configuration flagged `collapsed=True`
  (pairwise cosine 0.9877 > 0.98; std ratio
  5.32; steps_used 300 = 300/300).
  The elevated Cube AUROC therefore comes together with a degenerate (near-constant) adapted
  representation and fails frozen gate 5 (`no_collapse`); it is **not** accepted as a gain.
  Bird/Car pairwise cosines are also at the threshold edge (0.979 / 0.976).
- Lower BETA monotonically lowers robustness_score and illum_shift (more illumination-consistent);
  higher BETA raises AUROC on Bird but not on Car/Cube.

> Documentation note (fact, recorded 2026-10-15): `results/e17/README.md` states "BETA=0 is the
> worst detector on all three categories". By AUROC this holds for Bird/Car but **not** for Cube
> (where d' is instead the lowest metric). The E17 verdict itself is unchanged. The E17 document
> was NOT modified by E19; this note only aligns the E19 wording with the E17 artifact data.

**Frozen advancement gates (authoritative values, `final_method.json`)**

| cfg | BETA | Bird+Car mean dAUROC vs C0 | Cube dAUROC vs C0 | Bird+Car mean dAP vs C0 | Bird+Car mean d_rob vs C0 | collapsed | verdict |
|---|---|---|---|---|---|---|---|
| C1 | 0 | -0.02200 | +0.03894 | -0.03095 | -0.27942 | True | FAIL |
| C2 | 0.05 | -0.00644 | +0.00043 | -0.00628 | -0.11723 | False | FAIL |
| C3 | 0.2 | +0.00173 | -0.00870 | +0.00355 | +0.11871 | False | FAIL |
| C4 | 0.5 | -0.00063 | -0.00975 | +0.00446 | +0.28414 | False | FAIL |

No challenger passes; E17 verdict = `FREEZE_CURRENT_R1` and Round 2 was NOT run.

**Precision audit note (recorded, not a defect)**: the frozen gates were computed from
`results/e17/round1.csv`, whose values are recorded to 5 decimals. An independent recompute from
`results/e17/e17_partial.json` (full precision) differs by at most **9.2e-6** (see
`../verification.csv`, group BETA). This is far below every gate margin (smallest relevant margin:
C3 Bird+Car mean dAUROC +0.00173 vs the +0.003 bar), so no gate verdict is affected.

**Boundary**: BETA sweep covers one seed (seed 0) x three categories; it is a controlled
ablation (same backbone, same scoring, same bank protocol). It does not by itself establish
cross-seed stability — that is E16's 3-seed result. `robustness_score` / `illum_shift` are
feature- and score-level diagnostics frozen in E16 (`scripts/e16_qualify.py`), not causal
measurements.
