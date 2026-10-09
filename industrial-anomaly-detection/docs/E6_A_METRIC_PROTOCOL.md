# E6-A — Illumination Robustness Metric Tournament & Validity Audit (FROZEN PROTOCOL)

> **Status: FROZEN — written BEFORE any metric outcome or unblinded interpretation existed.**
> Frozen at git `c71a791` + protocol commit (recorded in `results/e6_a/summary.json` →
> `protocol_commit`). Nothing in §1–§14 may be edited after an unblinded result is read.
> Post-unblinding changes would require a new version (**E6-A2**) with a fresh freeze and a fresh blind.

```
CURRENT STAGE      = Metric Validation / Measurement Audit
GPU WORK           = 0 (hard ban: no fit, no scoring, no re-run, no new seed/view, no method tuning)
PRECEDING RESULT   = E5-1: GO = 0 | HOLD = C01 | STOP = C06, C07, C08
PRECEDING AUDIT    = E5-FAILURE-AUDIT: raw illumination variance share Original 0.463 -> C01 0.319
                     while frozen R_all reported C01 +3.1% WORSE (sign disagreement);
                     C08 got R_ratio 0.4652 with AUROC 0.462, d' -0.110 (collapse).
```

---

## 1. The research question

The primary question is **not** "which method is better" and **not** "how do we make C01 win".
It is:

> **Is the quantity we use to measure illumination robustness actually valid?**

A valid illumination-robustness metric should: (A) measure illumination nuisance sensitivity;
(B) not reward detection collapse; (C) be insensitive to arbitrary score-scale changes; (D) separate
illumination sensitivity from defect sensitivity; (E) give sensible answers on known-negative /
known-neutral / plausible-positive cases; (F) aggregate across specimens and illuminations;
(G) be freezable into a single protocol.

**Exit of E6-A:** KEEP 1–3 metrics/metric groups, **or** — if no single scalar is reliable — freeze a
**two-axis (Detection/Preservation × Illumination Sensitivity) Pareto protocol**.

## 2. HARD RULE — metric selection must not be optimised for C01

`METRIC SELECTION MUST NOT BE OPTIMIZED FOR C01.`

The re-audit is independently justified: **C08 already exposes the pathology** (`R_ratio` 0.4652 with
AUROC 0.46197 and d′ −0.11035) with no reference to C01. C01 additionally exposes a **sign
disagreement** between raw illumination share and `R_all`. Concretely:

* the primary tournament (§17) is run **blinded** (§6) — method identities are hidden as `M01…M06`;
* C01 is pre-classified as **AMBIGUOUS / PLAUSIBLE POSITIVE only** (§15) and is **never**
  pre-labelled known-good;
* no denominator, weight, or aggregation is selected because it flatters C01;
* if a metric is only defensible by appealing to C01's outcome, it is REJECT.

## 3. Scope, assets, and the GPU ban

GPU WORK = 0. No model is fitted, no dataset is scored, PatchCore / C01 / C06 / C07 / C08 are never
re-run, no seed or view is added, no method parameter is touched. Only existing per-image scores and
metadata are read (`docs/E6_A_ASSET_MANIFEST.md`).

If a metric needs a field that does not exist: rebuild it from existing assets if possible, otherwise
record `NOT_COMPUTABLE_FROM_EXISTING_ASSETS`. **Never start GPU work to obtain it.**

**Frozen analysis frame:** M²AD **Bird**, frozen **view 120**, **seed 0**; 700 images = **20 Good
specimens × 10 illuminations** + **50 NG specimens × 10 illuminations**; all 7 methods share an
identical `(specimen, illumination)` key set (manifest §3).

### Notation

* `s_m(x, i)` = raw score of method `m` for specimen `x` at illumination `i` (fixed view).
* `G` = 20 Good specimens, `D` = 50 NG specimens, `R = G ∪ D` (70).
* `μ_g(m), σ_g(m)` = mean/SD of method `m`'s **Good** scores (normal-only calibration — the frozen
  E4-X/E5-1 convention). `z_m = (s_m − μ_g)/σ_g`.
* All SDs use `ddof=1`. `X` = Original (the frozen reference).

---

## 4. Candidate metrics M1–M8 (exact definitions)

### M1 — Frozen `R_all` (LEGACY)
Reproduce `scripts/e4x_common.py::metric_block` **mechanically**, unmodified:
per specimen `r`, `a_r = std_{i∈01..10}( z_m )`; `R_kind = mean_{r∈kind} a_r`;
`R_all = mean_{r∈R} a_r`; `R_ratio = R_all(m)/R_all(X)`.

* **Acceptance (must reproduce the frozen record):** `X.R_all = 0.67543`, `C01.R_all = 0.69617`,
  `C01.R_ratio = 1.0307`, `X.image_auroc = 0.76766`, `X.d_prime = 1.20676` (tolerance 1e-4).
  Failure ⇒ STOP and localise the definition difference; do not proceed with an inconsistent formula.
* This is the **legacy** metric. It is **not** the default winner.

### M2 — Raw within-specimen illumination variance  · **SCALE-SENSITIVE**
For each specimen `x`: `v(x) = Var_{i}( s_m(x,i) )` (raw score, over the 10 illuminations).
Report `mean / median / IQR` of `v` over `G`, over `D`, and over `R`.
`M2_ratio = mean_{x∈R} v_m(x) / mean_{x∈R} v_X(x)`.
**Explicitly flagged `SCALE-SENSITIVE` ⇒ must never be used as a final metric on its own.**

### M3 — Illumination variance share
Reproduce the E5-FAILURE-AUDIT definition **exactly** (same code path, `scripts/e5_failure_audit.py::structured_variance`):
* `ill_mean_i = mean` of scores over the specimens of a given kind at illumination `i`;
* `spec_mean_x = mean` of scores over the 10 illuminations for specimen `x`;
* `illum_share(kind) = std_i(ill_mean_i) / ( std_i(ill_mean_i) + std_x(spec_mean_x) )`.
Report for `Good`, `NG`, and `All` (all 70 specimens pooled).
* **Acceptance (must reproduce the failure audit):** `X.Good = 0.4626`, `C01.Good = 0.3192`,
  `X.NG = 0.1120`, `C01.NG = 0.0981` (tolerance 5e-4). Failure ⇒ STOP (§9 of the task).

### M4 — Within-specimen rank stability across illuminations
Missing patch-level dumps (manifest §5) ⇒ **the ranking is taken over specimens, declared not faked.**
For each illumination `i` rank the specimens of a kind by `s_m(·, i)`; compute **Spearman ρ** between
the rankings at illumination `i` and `j` for all `C(10,2)=45` pairs, separately for `G` (n=20) and
`D` (n=50). Report `mean / median / worst-pair ρ`. **Higher = more stable.**
(Equivalently: does the method's own ordering of objects survive a change of light?)

### M5 — Standardized within-specimen sensitivity  · **denominator sensitivity analysis required**
Numerator `N_m(kind) = mean_{x} std_{i}( s_m(x,i) )` — identical to M2's mean.
Denominator candidates, **all four computed, none pre-selected**:
* **A** pooled SD of all scores (Good ∪ NG)
* **B** `|mean_NG − mean_Good|`  (a *defect-separation* scale)
* **C** pooled within-class SD `= sqrt((sd_G² + sd_N²)/2)`
* **D** robust scale `1.4826 · median(|s − median(s)|)` over all scores

`M5_X = N / denom_X`; `M5_X_ratio = M5_X(m)/M5_X(X)`; report for `Good`, `NG`, `All`.
**Sensitivity analysis:** Kendall τ between the 7-method orderings induced by denominators A, B, C, D.
If any pair has `τ < 0.5`, **M5 = UNSTABLE** and may not be a primary metric.

### M6 — Detection-conditioned robustness  · **TWO-AXIS, never a scalar**
* X-axis (Detection/Preservation): `AUROC`, `d′`, plus `ΔAUROC`, `Δd′` vs Original.
* Y-axis (Illumination sensitivity): one of M2/M3/M4/M5 **that passed sanity** (named explicitly in
  the tournament result; not chosen here).
* **Eligibility gate (hard):** a method may be discussed as a robustness *winner* only if
  `AUROC ≥ 0.5` **and** `Δd′ ≥ −0.10`. Otherwise its Y value is printed but marked
  `INVALID (detection collapse)` and it can never be a robustness winner.
* **The 0.5 line is a sanity gate only, not a research threshold.**

### M7 — Pairwise illumination score drift
For every specimen `x` and every illumination pair `(i,j)`, `i<j`:
`drift = |s_m(x,i) − s_m(x,j)|`. Report `mean / median / P90 / max` over all `70 × 45 = 3150` triples,
separately for `G`, `D`, `R`. Three normalisations:
`raw`, `standardized = drift/σ_g(m)`, `relative = drift/std_all(m)`.
`M7_ratio` uses the **standardized** variant. This directly answers:
*"the same object under a different light — how far did its anomaly score move?"*

### M8 — Detection stability across illumination
For each illumination `i`: `AUROC_i` (Good vs NG **within** that illumination; 20 vs 50).
Report `mean_i AUROC_i`, `sd_i AUROC_i`, `min_i`, `max_i`, `gap = max−min`, `CV = sd/mean`
(CV reported but not used for ranking when `mean` is near 0.5).
Also `Δmean = mean_m − mean_X` and `M8_ratio = sd_m/sd_X`.
**Pre-registered reading rule:** M8 must always be read **with** mean AUROC — a detector that is
uniformly near-random is *not* robust. M8 alone is never sufficient.

---

## 5. Pathology tests T1–T6 (all CPU-only, synthetic score manipulation)

Applied to every metric. Each test is run on all 7 methods **and** on 3 synthetic detectors
(`constant`, `random`, `perfect`).

| id | test | procedure | FAIL condition |
|---|---|---|---|
| **T1** | Scale invariance | `s → a·s + b` for `(a,b) ∈ {(1,0),(0.1,0),(10,0),(1,+100),(1,−50)}` | a metric marked **EXPECTED-INVARIANT** that changes (rel. tol 1e-6) |
| **T2** | Constant-score collapse | synthetic method with `s ≡ c` (c ∈ {0, 1, 100}) | metric calls it "perfectly robust" **without** the M6 detection gate ⇒ flag `NEEDS-DETECTION-GATE` |
| **T3** | Random-score detector | `s ~ N(0,1)` i.i.d., seeded | metric reports an excellent robustness **without** the gate ⇒ flag `NEEDS-DETECTION-GATE` |
| **T4** | Monotonic transform | `s → a·s+b`, `a>0`; Kendall τ between the 7-method ordering and the untransformed ordering | τ < 1.0 for an EXPECTED-INVARIANT metric |
| **T5** | Defect erasure | `s_NG ← α·s_NG + (1−α)·mean(s_Good)`, `α ∈ {1, .75, .5, .25, 0}` | metric **improves** (sensitivity falls) while AUROC/d′ collapse ⇒ FAIL **as a standalone objective** |
| **T6** | Illumination perturbation injection | `s(x,i) += c·σ_all·(i−5.5)/4.5` for `c ∈ {0, .2, .5, 1, 2}` | reported sensitivity is **not** monotonically increasing in `c` ⇒ FAIL construct validity |

**Expected-invariance declaration (frozen here):** INVARIANT — M1(`R_ratio`), M3, M4, M6-X, M8,
M5 with denominators A/C/D, M7-standardized. SENSITIVE-BY-DESIGN — M2, M7-raw, M5 with denominator B.

**Fatal pathology set:** T1 FAIL (where invariance is expected) · T4 FAIL (where invariance is
expected) · T6 FAIL · T2 or T3 without a detection gate · T5 improving under defect erasure.
**A metric with any fatal pathology cannot be PRIMARY regardless of its total score.**

---

## 6. Blind procedure

* `BLIND_SEED = 20261009`. `Original` keeps its name (it is the frozen reference and is needed for
  every ratio). The other six methods are mapped to `M01…M06` by a seeded permutation stored in
  `results/e6_a/method_blind_map.json`.
* The **primary** metric tournament, the pathology tables and the KEEP/SECONDARY/REJECT verdicts are
  produced and written **blinded** (`results/e6_a/blind/` snapshot with `"blinded": true`).
* **Only then** is the map read and the interpretation written (§19). Method names appear in the
  final tables, figures and README.
* **Forbidden:** modifying any metric formula after unblinding. If a change is unavoidable the round
  becomes **E6-A2** (new freeze, new blind).

---

## 7. Tournament scoring and verdicts

| criterion | weight |
|---|---|
| Construct validity (measures illumination sensitivity, distinguished from defect sensitivity) | 25 % |
| Collapse resistance (does not reward a detector that stopped working) | 20 % |
| Scale behaviour (T1/T4, robustness to arbitrary rescaling) | 15 % |
| Interpretability (can the number be explained in one sentence?) | 15 % |
| Statistical stability (bootstrap CI, §8) | 15 % |
| Computability (available from existing assets, GPU-free) | 10 % |

Each criterion scored 0–100; total = weighted sum. **Fatal pathology overrides the total.**

**Pre-registered KEEP criteria** (all must hold):
1. no fatal pathology;
2. Construct validity ≥ 20/25 and Collapse resistance ≥ 16/20;
3. computable from existing assets;
4. bootstrap-stable: the 95 % CI of `metric_ratio(m)` vs Original excludes a degenerate range
   (i.e. CI width < 50 % of the point estimate) OR the metric's method ordering is stable across
   ≥ 95 % of bootstrap resamples.

**SECONDARY** = useful as a companion axis / diagnostic but fails KEEP on ≥1 criterion.
**REJECT** = fatal pathology, or scale-dependent, or uninterpretable, or redundant given a KEEP metric.

## 8. Statistical analysis

Specimen-level **bootstrap, 1000 resamples, `BOOTSTRAP_SEED = 20261009`**: resample the 70 specimens
with replacement (each keeps its 10 illuminations, preserving the paired structure), recompute every
metric for every method, and report **mean and 95 % percentile CI** for (a) the metric value and
(b) `metric_ratio(m)` vs Original, plus the rank stability of the method ordering.
Small effective sample sizes are reported as such; **the statistic is never swapped after seeing a CI
that crosses 0.**

## 9. Two-axis mandate (hard requirement)

Even if one scalar survives KEEP, the recommended protocol is **TWO-AXIS PARETO** whenever the scalar
carries any collapse-resistance weakness:
* **Axis X** = Detection/Preservation (`AUROC`, `d′`)
* **Axis Y** = a KEEP-grade within-specimen illumination-sensitivity metric
* decision by **Pareto dominance**, not by a weighted sum.

**Arbitrary weighted composites such as `0.37·AUROC − 0.63·variance` are FORBIDDEN** — no
hand-tuned weights without theory.

## 10. Known-case sanity expectations (pre-registered, set BEFORE unblinding)

| case | independent fact | pre-registered expectation |
|---|---|---|
| **C06 / C07 / C08** | E5-1 detection collapse (AUROC 0.488 / 0.556 / 0.462; d′ −0.055 / 0.144 / −0.110) | any metric that calls them **strong robustness winners** must be heavily questioned |
| **B2 / X6c** | E4-X real-illumination kill test: indistinguishable from Original | a claim of **huge** robustness improvement requires explanation |
| **C01** | the object under re-evaluation | **AMBIGUOUS / PLAUSIBLE POSITIVE only** — never pre-labelled good |

## 11. C01 re-evaluation rule (applied only after §7 is frozen and written)

`C01 = SUPPORTS FOLLOW-UP` requires **all** of:
1. detection/preservation not degraded (`AUROC ≥ Original` and `d′ ≥ Original`);
2. **at least one post-freeze illumination metric improves consistently**, and the improvement
   survives leave-one-illumination-out (not driven by a single illumination);
3. no improvement claim that depends on a REJECTed metric;
4. the metric(s) used passed the pathology tests.

Otherwise `AMBIGUOUS`. `REJECT` if the improvement disappears or reverses under the KEEP metrics.
**`ΔAUROC = +0.0128` alone never promotes C01.**

## 12. Deliverables

```
docs/E6_A_METRIC_PROTOCOL.md      (this file, frozen)      docs/E6_A_ASSET_MANIFEST.md
results/e6_a/README.md  summary.json  metric_tournament.csv  method_metric_table.csv
             bootstrap_ci.csv  method_blind_map.json
             blind/  metrics/  pathology/  figures/  logs/
scripts/e6_a_metric_tournament.py
```

**Figures:** F1 metric × method heatmap (**blind and unblinded** versions) · F2 detection vs
illumination-sensitivity Pareto plot · F3 per-illumination AUROC stability · F4 pathology-test
comparison · F5 C01-vs-Original specimen-level paired comparison (if the data allows).
**Failed methods are never hidden.**

## 13. Out of scope / forbidden this round

No GPU · no new method · no C01 rescue · no Retinex parameter search · no multi-view or multi-seed
model re-run · no B2/X6c tuning · no C06/C07/C08 retraining · no automatic composition · no
hand-tuned metric weights · **no hiding a pathology to make the paper's story prettier**.

## 14. Deviations log (append-only; empty at freeze)

| # | time | deviation | reason |
|---|---|---|---|
| — | — | *(none at freeze)* | — |
