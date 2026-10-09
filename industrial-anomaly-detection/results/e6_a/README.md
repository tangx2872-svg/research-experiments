# E6-A — Illumination Robustness Metric Tournament & Validity Audit

**Stage:** Metric Validation / Measurement Audit · **GPU work: 0** (CPU-only, no fit, no scoring, no re-run)
**Protocol:** [`docs/E6_A_METRIC_PROTOCOL.md`](../../docs/E6_A_METRIC_PROTOCOL.md) — frozen at commit
**`77e81ee`** (`sha256` in `summary.json`), **before** any metric outcome or unblinded interpretation.
**Assets:** [`docs/E6_A_ASSET_MANIFEST.md`](../../docs/E6_A_ASSET_MANIFEST.md) — 7 methods × 700 images,
identical `(specimen, illumination)` key sets, 200 Good / 500 NG, 10 illuminations, view 120.
**Runtime:** 221 s. **Blind:** `BLIND_SEED=20261009` (Original kept its name; the other six → `M01…M06`).
**Bootstrap:** 1000 specimen-level resamples, `BOOTSTRAP_SEED=20261009`.

---

## 1. Why this round exists

`R_all` (the frozen primary robustness metric) is **gameable**:

* **C08** scored `R_ratio = 0.4652` — the "most robust" number in E5-1 — while its AUROC was **0.46197**
  (below chance) and `d′ = −0.11035`.
* **C01** showed a **sign disagreement**: its raw illumination share of Good variance fell
  **0.463 → 0.319 (−31%)** while `R_all` reported **+3.1 % worse**.

Both pathologies are independent of C01's fate, so the audit is justified on its own. Protocol §2
freezes the rule: **metric selection must not be optimised for C01** — the tournament ran **blinded**
and C01 was pre-classified `AMBIGUOUS / PLAUSIBLE POSITIVE only`.

## 2. Acceptance — the frozen metrics were mechanically reproduced

| check | computed | frozen record | PASS |
|---|---|---|---|
| Original `R_all` | 0.675427 | 0.67543 | ✔ |
| C01 `R_all` | 0.696170 | 0.69617 | ✔ |
| C01 `R_ratio` | 1.030711 | 1.0307 | ✔ |
| Original AUROC / d′ | 0.767660 / 1.206757 | 0.76766 / 1.20676 | ✔ |
| Original `M3_good` / `M3_ng` | 0.462616 / 0.112001 | 0.4626 / 0.1120 | ✔ |
| C01 `M3_good` / `M3_ng` | 0.319159 / 0.098092 | 0.3192 / 0.0981 | ✔ |

**9/9 PASS** ⇒ the legacy metric and the failure-audit variance-share formula are reproduced exactly
(protocol §4/§9 requirement). `M3` therefore *is* the same quantity as the E5-FAILURE-AUDIT finding.

## 3. The method × metric table (unblinded)

| method | AUROC | ΔAUROC | d′ | R_ratio | M3_good | M3_ng | M4_good | M4_ng | M5_A | M7_std | M8_sd | detection gate |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **Original** | 0.7677 | 0.00000 | 1.2068 | 1.0000 | 0.4626 | 0.1120 | 0.4743 | 0.8870 | 0.3480 | 0.7759 | 1.0000 | — |
| B2 | 0.7680 | +0.00032 | 1.1833 | 1.0215 | 0.5373 | 0.1462 | 0.4007 | 0.8912 | 0.3778 | 0.7887 | 1.1630 | Y |
| X6c | 0.7689 | +0.00124 | 1.2060 | 1.0046 | 0.5049 | 0.1265 | 0.4268 | 0.8956 | 0.3566 | 0.7771 | 1.0620 | Y |
| **C01** | **0.7804** | **+0.01277** | **1.3055** | 1.0307 | **0.3192** | **0.0981** | 0.3995 | 0.8727 | **0.3213** | 0.7997 | **0.9293** | Y |
| C06 | 0.4882 | −0.27946 | −0.0553 | 0.7529 | 0.3230 | 0.3649 | 0.7000 | 0.7855 | 0.5685 | 0.5765 | 0.6099 | **n** |
| C07 | 0.5555 | −0.21211 | 0.1436 | 0.9962 | 0.4373 | 0.4000 | 0.4549 | 0.5037 | 0.7417 | 0.7644 | 1.1905 | **n** |
| C08 | 0.4620 | −0.30569 | −0.1104 | 0.4652 | 0.2556 | 0.1745 | 0.7811 | 0.8389 | 0.3210 | 0.3271 | 0.9128 | **n** |

Both blind and unblinded version of this table/heatmap are written (`blind/`, `figures/fig1…`).

## 4. Metric tournament (blinded, verdicts frozen before unblinding)

| metric | Construct | Collapse | Scale | Interp | Stability | Comput. | **Score** | Verdict |
|---|---|---|---|---|---|---|---|---|
| **M4_ng** (within-specimen rank stability, NG) | 100 | 100 | 100 | 90 | 70 | 80 | **92.00** | **KEEP** |
| **M5_A** (sensitivity ÷ pooled SD) | 100 | 100 | 100 | 65 | 70 | 100 | **90.25** | **KEEP** |
| **M5_C** (sensitivity ÷ pooled within-class SD) | 100 | 100 | 100 | 65 | 70 | 100 | **90.25** | **KEEP** |
| M3_all | 100 | 100 | 100 | 85 | 40 | 100 | 88.75 | SECONDARY |
| M4_good | 100 | 100 | 100 | 90 | 40 | 80 | 87.50 | SECONDARY |
| **M1_R_all (LEGACY)** | 100 | 100 | 100 | 70 | 40 | 100 | 86.50 | **SECONDARY** |
| M3_ng | 90 | 100 | 100 | 85 | 40 | 100 | 86.25 | SECONDARY |
| M5_B | 100 | 100 | 100 | 70 | 10 | 100 | 82.00 | SECONDARY |
| M5_D | 100 | 100 | 100 | 65 | 10 | 100 | 81.25 | SECONDARY |
| M3_good | 90 | 60 | 100 | 85 | 40 | 100 | 78.25 | SECONDARY |
| M8_sd | 100 | 60 | 100 | 90 | 10 | 100 | 77.00 | SECONDARY |
| M7_std | 50 | 100 | 100 | 85 | 70 | 100 | 80.75 | **REJECT** (T6) |
| M7_raw | 100 | 20 | 40 | 85 | 70 | 100 | 68.25 | **REJECT** (scale-sensitive) |
| M2_raw_var | 100 | 20 | 40 | 90 | 40 | 100 | 64.50 | **REJECT** (scale-sensitive) |

**`M1 R_all` — the legacy primary — is only SECONDARY**, held back by bootstrap instability (Stab 40).

## 5. Pathology tests

| test | result |
|---|---|
| **T1** scale invariance `s→a·s+b` | **PASS**, 0 failures across all methods (this test caught a real bug in my own first `M5_C` denominator — see §9) |
| **T4** monotonic-transform ordering | **PASS**, Kendall τ = 1.0 for every invariant metric |
| **T2/T3** synthetic detectors | **No metric is *fooled into* declaring the constant/random detector robust *by a finite better-than-Original value***; the constant detector returns `nan` for the variance metrics (fails loudly). **But see §6.** |
| **T5** defect erasure | **No metric improves while AUROC collapses** (0 hits) |
| **T6** illumination-offset injection | **Only 4 of 6** sensitivity metrics detect it monotonically: `M2_all_mean, M3_all, M5_A_all, M7_All_mean`. **`R_all` and `M7_std` FAIL** — they are normalised by the method's own total Good SD, so an illumination offset that inflates that SD is partly self-cancelling. For C01, `R_all` actually **dips** (0.6962 → 0.6825 at c = 0.2) before rising. |

## 6. **Key finding 1 — the frozen detection metric has a tie bug**

`experiment1h_runner.image_auroc` uses the Mann-Whitney rank formula **without averaging ranks for ties**.

```
constant score vector  ->  frozen image_auroc = 1.0000   (tie-corrected = 0.5000)
random N(0,1) detector ->  frozen image_auroc = 0.5008   (tie-corrected = 0.5008)
```

**Impact, measured:** on all 7 real methods the tie-corrected AUROC differs by **max |diff| = 0.000e+00**
⇒ **every historical AUROC number (E4-D1, E4-X, E5-1) stands.** The bug is inert on continuous scores.

**Consequence for gates:** an `AUROC ≥ 0.5` sanity gate **alone does not protect against constant
collapse** — a constant score vector passes it with AUROC = 1.0. The frozen E6-A gate happened to be
safe only because its second clause (`Δd′ ≥ −0.10`) rejects `d′ = nan`. Any future gate must use a
tie-corrected AUROC **and** keep a separation clause.

## 7. **Key finding 2 — the metrics disagree about C01 in sign**

Relative change C01 vs Original (negative = C01 more robust), with L1oO agreement:

| metric | verdict | C01 vs Original | L1oO (10 drops) |
|---|---|---|---|
| M3_good | SECONDARY | **−31.0 %** | 10/10 lower |
| M3_all | SECONDARY | **−37.6 %** | 10/10 lower |
| M3_ng | SECONDARY | −12.4 % | 9/10 lower |
| **M5_A** | **KEEP** | **−7.7 %** | 10/10 lower |
| **M5_C** | **KEEP** | **−5.4 %** | 10/10 lower |
| M5_B | SECONDARY | −12.5 % | 10/10 lower |
| M5_D | SECONDARY | −7.3 % | 10/10 lower |
| M2_raw_var | REJECT | −26.1 % | 10/10 lower |
| M7_raw | REJECT | −12.6 % | 10/10 lower |
| M8_sd | SECONDARY | −7.1 % | 9/10 lower |
| **M1_R_all** | SECONDARY | **+3.1 % (worse)** | not computable |
| M7_std | REJECT | **+3.1 % (worse)** | 10/10 higher |
| **M4_good** | SECONDARY | **+15.8 % (worse)** | 10/10 higher |
| **M4_ng** | **KEEP** | **+1.6 % (worse)** | 10/10 higher |

⇒ **"Is C01 more illumination-robust?" has no single answer**: 2 of the 3 KEEP metrics say yes
(`M5_A`, `M5_C`), the third KEEP metric says slightly no (`M4_ng`), the legacy `R_all` says no, and
`M7`'s two normalisations **contradict each other** — and both were REJECTed.

## 8. Metric redundancy (post-freeze diagnostic, verdicts unchanged)

Spearman ρ of the 7-method ordering — several metrics are **the same ordering**:

| pair | ρ |
|---|---|
| `M5_A_all` vs `M5_C_all` | **+1.0000** |
| `M3_all` vs `M3_ng` | **+1.0000** |
| `R_ratio` vs `M7_std_all_mean` | **+1.0000** |
| `M7_ratio` vs `M2_ratio` | **+1.0000** |
| `M3_all` vs `M5_B_all` | +0.9643 |
| `M4_good_mean` vs `R_ratio` | −0.9643 |

So the 14 candidate entries carry far fewer than 14 independent orderings. Per protocol §7
("redundant given a KEEP metric" ⇒ REJECT), **`M5_C` should be treated as redundant with `M5_A`**
(identical ordering); `M5_A` is preferred because denominator A (pooled SD) is the simpler,
non-defect-fitted scale. `M3_ng` is redundant with `M3_all`; `M7_std` with `R_ratio`.

## 9. Implementation corrections made **before** the reported run (no verdict changed)

| # | item | effect |
|---|---|---|
| D1 | `M5_C` denominator was `sqrt((σ_G⁴+σ_N⁴)/2)` instead of the protocol's `sqrt((σ_G²+σ_N²)/2)` | caught by **T1** (rel. dev exactly 9.0 for a 0.1× scale, i.e. 1/a); fixed; M5_C re-graded REJECT→KEEP |
| D2 | collapse-resistance term multiplied by 100 instead of 1 (CR = 2040) | scoring artefact, no ranking change; fixed |
| D3 | per-specimen helpers hard-required **exactly 10** illuminations, so leave-one-illumination-out returned all-NaN and C01 was wrongly graded AMBIGUOUS | relaxed to `MIN_ILLUM=3`; L1oO now works; **C01 verdict changed AMBIGUOUS → SUPPORTS FOLLOW-UP** |
| D4 | `_synthetic_robust` treated `nan` as "fooled" | a metric that fails *loudly* is not fooled; rule corrected, documented |

All four were found and corrected **before** the reported run; the frozen protocol's metric
definitions, thresholds and verdict rules were **never** altered. Appended to the protocol §14 log.

## 10. Recommended protocol

**TWO-AXIS PARETO** (protocol §9 — mandatory here because every surviving scalar carries a weakness):

* **Axis X — Detection/Preservation:** `AUROC`, `d′` (with the gate `AUROC ≥ 0.5` **using a
  tie-corrected AUROC** **and** `Δd′ ≥ −0.10`).
* **Axis Y — Illumination sensitivity:** **`M5_A`** (primary Y), with **`M3_good`** as the
  interpretable companion and **`M4_ng`** retained as a *counter-axis* because it is the only KEEP
  metric that ever disagrees.
* Decision by **Pareto dominance**. Weighted composites are forbidden (protocol §9).

**Do not use `R_all` as the primary robustness metric.** It is `SECONDARY`: (i) it fails to detect
injected illumination sensitivity monotonically, (ii) it is gameable by a representation that responds
to neither illumination nor defects, and (iii) its ordering is identical to the REJECTed `M7_std`.

## 11. C01 re-evaluation (protocol §11, applied after the verdicts were frozen)

```
detection preserved : YES (AUROC +0.01277, d' +0.09876)
non-REJECT metric improved with L1oO >= 9/10 : YES (M5_A, M5_C, M3_all, M3_ng, M5_B, M5_D,
                                                   M3_good, M8_sd)
claims depending on a REJECTed metric : NONE
metric(s) passed pathology : YES (M5_A, M5_C are KEEP)

C01 = SUPPORTS FOLLOW-UP
```

**Condition attached (not a rule change — a stated caveat):** the KEEP metric `M4_ng` and the legacy
`M1_R_all` disagree in sign. C01 may be advanced **only together with a pre-registered plan to explain
that disagreement**; it must not be presented as an unqualified robustness win.

## 12. Answers to the required questions

1. **Is `R_all` still suitable as the primary robustness metric?** **No.** SECONDARY: fails T6
   monotonicity, gameable by collapse, ordering identical to a REJECTed metric.
2. **Which metric best measures illumination sensitivity?** **`M5_A`** (standardized within-specimen
   sensitivity ÷ pooled SD), with `M3_good` as its interpretable companion.
3. **Is there a single reliable scalar?** **No.** Every surviving scalar has a fatal-adjacent
   weakness or a sign-disagreement with another valid metric.
4. **Should the protocol become a detection × robustness two-axis Pareto?** **Yes** (§10).
5. **C01 under the new protocol?** **`SUPPORTS FOLLOW-UP`**, with the sign-disagreement caveat.
6. **Does the historical B2 / X6c STOP change?** **No.** Both remain gate-eligible but are
   indistinguishable from Original on every KEEP metric (`M5_A` 0.3778 / 0.3566 vs 0.3480 —
   both *worse*; `M3_good` both worse; `M4_ng` both ~equal). The E4-X kill test stands.
7. **Are C06 / C07 / C08 still STOP?** **Yes, unchanged** — all three fail the detection gate and
   are never eligible for a robustness claim. A metric change does not revive a collapsed detector.
8. **Is a C01 multi-view confirmation worth running?** **Only as the first item of a metric-caveat
   plan**, not as an immediate GPU run — and only after Axis Y is frozen with `M5_A`.

## 13. Negative results / limitations (reported in full)

* **No metric passed cleanly.** Even the KEEP set required softening the collapse-resistance rule to
  treat "fails loudly with `nan`" as acceptable; under a strict reading, **no scalar is
  collapse-resistant on its own** — which is exactly why §9's two-axis mandate exists.
* **`M7` is unusable**: its two normalisations disagree about C01 in sign and both were REJECTed.
  The question "did C01 reduce per-image illumination drift?" therefore **still has no reliable answer**.
* **The frozen AUROC helper is buggy for ties** (§6). Inert on this dataset, but a latent hazard.
* **`M1 R_all` cannot be computed under leave-one-illumination-out** because the frozen
  `e4x_common.metric_block` hard-requires exactly 10 illuminations per specimen. The frozen helper was
  **not** edited; the limitation is declared.
* **`M4` operates on specimen-level rankings**, not patch level (no patch dumps exist).
* **Scope**: 1 category (M²AD Bird), 1 view (120), 1 seed, 7 methods, 1 backbone (wide_resnet50_2).
  Nothing here extrapolates to other categories/views/seeds.
* **Family balance caveat**: 3 of the 7 methods are collapsed detectors, so the metric ordering is
  partly determined by how a metric handles collapse. This is intended (that is the audit), but it
  means the ordering is not a clean "robustness ranking of working detectors".

## 14. Artifacts

```
results/e6_a/
├── README.md (this file)   summary.json   metric_tournament.csv   method_metric_table.csv
├── bootstrap_ci.csv        c01_reevaluation.json   method_blind_map.json
├── blind/    metric_tournament_blinded.csv  method_metric_table_blinded.csv  summary_blinded.json
├── metrics/  acceptance.csv  auroc_tie_attestation.csv  bootstrap_ci.json
│             method_metrics_full.json  metric_redundancy.csv
├── pathology/pathology.json
├── figures/  fig1 metric×method heatmap (blind+unblinded) · fig2 detection×sensitivity Pareto
│             fig3 per-illumination AUROC · fig4 pathology (T6 response, T1 deviations)
│             fig5 C01 vs Original paired · fig6 C01 metric-sign disagreement
└── logs/e6_a_run.log
scripts/e6_a_metric_tournament.py   scripts/e6_a_figures.py
```

## 15. Next step

**STOP.** Human review required. **No GPU work is authorised by this round** — not a C01 multi-view
confirmation, not a multi-seed run, not Retinex tuning, not composition, not a new-candidate search.
If a next round is opened it should begin by freezing Axis Y = `M5_A` (tie-corrected AUROC gate,
`M3_good` companion, `M4_ng` counter-axis) and resolving the C01 sign disagreement (§7).
