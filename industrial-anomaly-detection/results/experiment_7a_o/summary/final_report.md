# Experiment 7A-O — overnight module screening (EXPLORATORY)

- freeze sha256: `e046a89a852da9aad55d2eafeea1c3bc05a5aff640be4338f7e515e53150f224`
- GPU units: 92 (baselines B0/B1/B2/B3 reconstructed from frozen raw, 0 GPU)
- sanity: 18/18 PASS
- **FINAL VERDICT: CASE_C** — NO USEFUL MODULE: no simple module beats the strong Uniform baseline alpha=0.40091275.

## Family summary (Round 1, 3 categories × seed0, vs Uniform alpha=0.40091275)

| family | n | best Δd′ | best Δ|Δz| | promoted |
|---|---|---|---|---|
| A | 5 | A3_lam050 (+0.8597) | A1_lam025 (+0.0637) | — |
| B | 3 | B1_g100 (+0.7905) | B1_g100 (+0.0933) | — |
| C | 4 | C3_l2_045_l3_035 (+0.0107) | C1_l2_030_l3_050 (-0.0044) | — |
| D | 2 | D1_ln_s040 (+0.2808) | D2_gn8_s040 (+0.1520) | — |

## Promoted to Round 2

— (none)

## Round-2 (full 5 categories × 3 seeds)

| config | family | d′ | |Δz| | Δd′ vs Uniform | Δ|Δz| vs Uniform | worst cat Δ | negTr | tier |
|---|---|---|---|---|---|---|---|---|
| A3_lam050 | A | 5.3552 | 0.2753 | +0.5612 | +0.0619 | -0.0493 | 0 | - |
| B1_g100 | B | 5.3044 | 0.2850 | +0.5104 | +0.0717 | -0.0001 | 0 | - |
| D1_ln_s040 | D | 4.9678 | 0.3298 | +0.1739 | +0.1164 | -0.2796 | 1 | - |
| C3_l2_045_l3_035 | C | 4.8159 | 0.2248 | +0.0219 | +0.0115 | -0.0538 | 0 | - |
| B0_original | baseline | 4.9852 | 0.3105 | +0.1913 | +0.0971 | -0.3104 | 1 |  |
| B3_soft_gc_af025 | baseline | 4.8782 | 0.2329 | +0.0842 | +0.0195 | -0.0996 | 0 |  |
| B2_uniform_040091275 | baseline | 4.7940 | 0.2134 | +0.0000 | +0.0000 | +0.0000 | 0 |  |
| B1_fixed_050 | baseline | 4.6670 | 0.2082 | -0.1270 | -0.0052 | -0.2806 | 2 |  |

## Negative results (retained)

- A3_lam050 (family A): Δd′ +0.8597, Δ|Δz| +0.0872, negTr 0 — eliminated
- A3_lam025 (family A): Δd′ +0.7962, Δ|Δz| +0.0812, negTr 0 — eliminated
- B1_g100 (family B): Δd′ +0.7905, Δ|Δz| +0.0933, negTr 0 — eliminated
- B1_g050 (family B): Δd′ +0.6891, Δ|Δz| +0.0937, negTr 0 — eliminated
- A1_lam050 (family A): Δd′ +0.6290, Δ|Δz| +0.0744, negTr 1 — eliminated
- B1_g025 (family B): Δd′ +0.5777, Δ|Δz| +0.1204, negTr 0 — eliminated
- A1_lam010 (family A): Δd′ +0.5068, Δ|Δz| +0.1013, negTr 1 — eliminated
- A1_lam025 (family A): Δd′ +0.4746, Δ|Δz| +0.0637, negTr 1 — eliminated
- D1_ln_s040 (family D): Δd′ +0.2808, Δ|Δz| +0.1581, negTr 1 — eliminated
- D2_gn8_s040 (family D): Δd′ +0.2496, Δ|Δz| +0.1520, negTr 1 — eliminated
- C3_l2_045_l3_035 (family C): Δd′ +0.0107, Δ|Δz| +0.0204, negTr 0 — eliminated
- C4_l2_050_l3_030 (family C): Δd′ -0.0547, Δ|Δz| +0.0255, negTr 2 — eliminated
- C2_l2_035_l3_045 (family C): Δd′ -0.1370, Δ|Δz| +0.0168, negTr 3 — eliminated
- C1_l2_030_l3_050 (family C): Δd′ -0.1959, Δ|Δz| -0.0044, negTr 2 — eliminated

## Leakage statement

All results are **EXPLORATORY / selection-set**. The winner must be re-validated under an independent confirmation protocol (frozen structure/params/metrics) before any paper claim.