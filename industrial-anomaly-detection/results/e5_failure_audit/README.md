# E5-FAILURE-AUDIT (artifacts)

CPU-only diagnosis of the E5-1 `0 GO / 1 HOLD / 3 STOP` outcome.
**Entered mechanically** from E5-2A protocol §2 **CASE C**: `SURVIVORS (GO) = []` ⇒ E5-2A has no
subject ⇒ **STOP GPU METHOD WORK**, route to failure audit. **Zero GPU used.**

**Formal report:** [`docs/E5_FAILURE_AUDIT.md`](../../docs/E5_FAILURE_AUDIT.md)
**Script (reproducible):** [`scripts/e5_failure_audit.py`](../../scripts/e5_failure_audit.py)
**Dispatcher / safety gate:** [`scripts/run_e5_2a_after_e5_1.sh`](../../scripts/run_e5_2a_after_e5_1.sh)

## Files

| file | contents |
|---|---|
| `diagnostics.csv` | per candidate (incl. Original reference row): AUROC, ΔAUROC, d′, R_all, R_ratio, **score-σ ratio vs Original**, **Spearman ρ vs Original ordering**, **illumination share of Good/NG score variance**, AUROC-below-chance flag |
| `per_illumination_delta.csv` | per candidate × 10 illuminations: AUROC and d′ for candidate and Original plus deltas |
| `summary.json` | same as above in JSON + sign-consistency summary (Q2) + branch tag |
| `figures/e5_failure_audit.png` | Audit Fig 1: per-illumination ΔAUROC (all 10 negative for C06/C07/C08); Audit Fig 2: ρ vs Original and illumination share |

## Headline diagnostics

| candidate | insertion | decision | AUROC | ΔAUROC | d′ | R_ratio | σ ratio | ρ vs Orig | illum share Good | illum share NG |
|---|---|---|---|---|---|---|---|---|---|---|
| **Original** | reference | REFERENCE | 0.76766 | 0 | 1.20676 | 1.0000 | 1.000 | +1.000 | 0.4626 | **0.1120** |
| C01 | input | **HOLD** | 0.78043 | +0.01277 | 1.30552 | 1.0307 | 0.947 | +0.904 | 0.3192 | 0.0981 |
| C06 | post-concat | STOP | 0.48820 | −0.27946 | −0.05534 | 0.7529 | 0.031 | **−0.343** | 0.3230 | 0.3649 |
| C07 | post-concat | STOP | 0.55555 | −0.21211 | 0.14363 | 0.9962 | 0.322 | +0.480 | 0.4373 | 0.4000 |
| C08 | memory | STOP | 0.46197 | −0.30569 | −0.11035 | 0.4652 | 4.422 | **−0.402** | 0.2556 | 0.1745 |

* **C06 and C08 invert the anomaly ordering** (ρ = −0.343 / −0.402); C06's score spread collapses to
  0.031× Original while C08's *grows* to 4.42× — so C08's low `R_ratio` is **not** score compression,
  it is a representation that stopped co-varying with illumination (and with defects).
* **All three STOP candidates are negative at all 10 illuminations** (0/10 positive) — failure is
  global, not illumination-specific.
* **All three amplify illumination dependence in the NG score** (0.112 → 0.175/0.365/0.400).
* **Only the training-free module (C01) survived.** Every fitted module failed.
* **Metric hazard (recorded):** C01's **raw** illumination share of Good variance falls
  0.463 → 0.319 (−31%) while the frozen primary metric `R_all` reports **+3.1% worse** — raw share
  and `R_all` disagree in sign for C01.

## Scope limits

1 category × 1 view × 1 seed × 1 backbone × 4 candidates. A **negative result**, not proof that
module insertion is impossible. No rescue, tuning, C10 top-up or E5-2 composition is authorised.
