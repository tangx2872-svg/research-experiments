# E5-FAILURE-AUDIT — why E5-1 produced `0 GO / 1 HOLD / 3 STOP`

**Status:** CPU-only. **Zero GPU, zero fitting, zero new inference.** Entered mechanically from
E5-2A protocol §2 **CASE C (0 GO)** — GPU method work is stopped and no rescue is authorised.

**Inputs (all already frozen, nothing re-run):**
`results/e4_x/per_image_original.csv` · `results/e5_1/raw/<C>_per_image.csv` · `results/e5_1/raw/<C>.json`
**Outputs:** `results/e5_failure_audit/{diagnostics.csv,per_illumination_delta.csv,summary.json,figures/e5_failure_audit.png}`
**Script:** `scripts/e5_failure_audit.py` (reproducible, CPU-only)

---

## 0. Safety gate (7/7 PASS — required before any downstream step)

| # | check | result |
|---|---|---|
| 1 | E5-1 result complete | PASS |
| 2 | E5-1 `README.md` exists | PASS |
| 3 | E5-1 decisions exist (4/4 JSON) | PASS |
| 4 | no active E5-1 GPU process | PASS (GPU 1 MiB / 0 %) |
| 5 | Git protocol/result state known | PASS (`e96cd5a` → `9455de9`, local == `origin/research/industrial-anomaly`) |
| 6 | leakage flagged | **NO** |
| 7 | unresolved protocol deviation | **NONE** (D1–D5 all logged pre-result, all resolved) |

**Mechanical decision read (no assumption, no pre-declared winner):**

```
SURVIVORS (GO) = []          ->  E5-2A GPU survivor validation is NOT EXECUTABLE
HOLD           = ['C01']
STOP           = ['C06','C07','C08']
BRANCH         = CASE C (0 GO)  ->  STOP GPU METHOD WORK -> E5-FAILURE-AUDIT
```

> **E5-2A (multi-view / multi-seed survivor confirmation) does not run in this round.** It has no
> subject. `docs/E5_2A_FROZEN_PROTOCOL.md` is therefore **not** created — freezing additional views
> to validate a survivor that does not exist would be exactly the kind of empty procedure the
> project's protocol discipline forbids. The dispatcher
> `scripts/run_e5_2a_after_e5_1.sh` implements the gate and refuses GPU work on CASE C.

---

## 1. Q1 — Why did each candidate fail?

Reference row is **Original itself**, so the diagnostics below are interpretable in absolute terms.

| candidate | insertion | decision | AUROC | ΔAUROC | d′ | R_ratio | **score-σ ratio vs Original** | **Spearman ρ vs Original ordering** | illumination share of **Good** score var | illumination share of **NG** score var |
|---|---|---|---|---|---|---|---|---|---|---|
| **Original** | (frozen reference) | REFERENCE | 0.76766 | 0 | 1.20676 | 1.0000 | 1.000 | **+1.000** | 0.4626 | **0.1120** |
| **C01** | input (photometric) | HOLD | 0.78043 | +0.01277 | 1.30552 | 1.0307 | 0.947 | **+0.904** | 0.3192 | 0.0981 |
| **C06** | post-concat (feature) | STOP | 0.48820 | −0.27946 | −0.05534 | 0.7529 | **0.031** | **−0.343** | 0.3230 | **0.3649** |
| **C07** | post-concat (feature) | STOP | 0.55555 | −0.21211 | 0.14363 | 0.9962 | 0.322 | +0.480 | 0.4373 | **0.4000** |
| **C08** | memory representation | STOP | 0.46197 | −0.30569 | −0.11035 | 0.4652 | 4.422 | **−0.402** | 0.2556 | 0.1745 |

### C06 SimpleNet — **degenerate, and its score is inverted**
* Final discriminator training loss settled at **exactly 1.0000 = 2·θ** (the margin-hinge floor with
  `dsc_margin = 0.5`): the discriminator stopped separating true features from its own synthetic
  Gaussian perturbations at all.
* Quantitatively: its score spread is **0.031× Original's** (≈32× collapsed) and it is
  **negatively rank-correlated with the truth (ρ = −0.343)**. AUROC 0.488 < 0.5 is therefore not
  "chance" — it is **systematically inverted**.
* Diagnosis: a 2-layer MLP discriminator trained with a `.5` margin loss on 1.23 M patches from a
  *frozen* 1536-d embedding has no gradient pressure to preserve anything; the hinge loss has a
  large flat region and the solution drifts to a near-constant, anti-correlated function. The
  synthetic-anomaly objective assumes the backbone is trainable/adapted (SimpleNet's own pipeline
  fine-tunes an adaptor jointly); transplanting only the discriminator onto a frozen embedding
  removes the signal it was designed to sharpen.

### C07 ReConPatch — **learned a representation that keeps ~half the ordering but destroys separation**
* The contrastive objective *did* optimise (loss 9.25 → 0.79), and ρ = +0.480 shows it retains a
  substantial part of Original's ordering — but `d′` collapses from 1.207 to **0.144** (−88%).
* Score spread 0.322× Original: the modulation compresses the feature space, and the
  normal-only pairwise/contextual pseudo-labels contain **no defect-supervision signal**, so the
  objective is free to collapse directions that carry defect information. Nothing in the paper's
  loss forbids it.

### C08 CRAD — **its reconstruction error is anti-correlated with defectness**
* This is the most instructive failure. Its score spread is **4.42× Original's** (it is *not*
  collapsed) yet `d′ = −0.110` and **ρ = −0.402**. The continuous memory reconstruction assigns
  **larger** error to normal patches than to defect patches: the learnable grid reconstruction is
  dominated by the illumination/specimen scale, so it "explains away" exactly the component that
  separates Good from NG — and inverts the residual.
* **Correction to an earlier shorthand:** C08's excellent `R_ratio = 0.465` is **not** produced by
  score compression (its variance is larger than Original's). `R_all` is computed on *z-scored*
  scores, i.e. cross-illumination dispersion **in units of the method's own Good-score spread**.
  A representation whose scored quantity simply **does not co-vary with illumination** — whether
  because it is degenerate or because it re-parameterised the problem — scores a low `R_all`
  **without any loss of illumination sensitivity in the image**. **`R_all` is therefore only
  meaningful jointly with `Δd′`/`ΔAUROC`**, which is exactly why the pre-registered rule stopped C08.

### C01 PIAD-Retinex — **the only non-failure, and the reason it is only a HOLD**
* ρ = **+0.904** and score spread **0.947× Original**: C01 is, to first order, *the same detector*
  with a modest reordering. It does not replace the decision mechanism — it only changes the pixels
  the frozen PatchCore sees.
* It is the only candidate that **buys** anything: `Δd′ = +0.0988` (+8.2% normal–defect separation)
  and `ΔAUROC = +0.0128` (inside the pre-registered HOLD band, never a "win").
* **A metric-tension worth recording:** on **raw** scores C01 **reduces** the illumination share of
  Good variance from 0.463 → **0.319 (−31%)**, yet the frozen primary robustness metric reports a
  **3.1% worsening** (`R_ratio = 1.0307`). The raw illumination share and `R_all` **disagree in
  sign** for C01. This is a metric-validity observation that should be resolved before any
  robustness claim is made, and it is the single most useful thing E5-1 produced.

---

## 2. Q2 — Is the failure consistent across illuminations?

**Yes, and completely so for every failing candidate.** Per-illumination ΔAUROC (10 illuminations):

| candidate | mean ΔAUROC | min | max | positive / negative | all-negative? | all-positive? |
|---|---|---|---|---|---|---|
| C01 | +0.01250 | −0.01400 | +0.04700 | **7 / 3** | no | no |
| C06 | −0.27890 | −0.35600 | −0.18600 | **0 / 10** | **YES** | no |
| C07 | −0.21030 | −0.32000 | −0.09400 | **0 / 10** | **YES** | no |
| C08 | −0.30750 | −0.36100 | −0.26700 | **0 / 10** | **YES** | no |

* The three STOP failures are **global, not illumination-specific**: there is no illumination
  condition in which any of them beats Original, so no rescue could be justified by
  condition-targeting.
* Conversely the modules **amplify** illumination dependence where it matters: the illumination
  share of the **NG** score variance jumps from Original's **0.112** to **0.365 (C06)**, **0.400
  (C07)**, **0.175 (C08)** — i.e. up to **3.6×** more illumination-driven anomaly scores. On this
  data the modules make the very failure mode the project targets *worse*.
* C01 is the opposite: mixed sign (7/3) and it does **not** amplify illumination in NG scores
  (0.098, marginally *below* Original).
* Honest caveat: a 10-illumination sign test at one view / one seed is a consistency check, not a
  significance test; the magnitudes (≈ −0.21 to −0.31 mean ΔAUROC) are far larger than any plausible
  noise, so the direction of the conclusion is robust even if the exact numbers are not.

---

## 3. Q3 — Did all three insertion points fail?

**No — and that asymmetry is the finding.**

| insertion point | candidate(s) | outcome | fitted? |
|---|---|---|---|
| **input (photometric)** | C01 | **HOLD** (ΔAUROC +0.013, Δd′ +0.099) | **training-free** (fixed pretrained Retinex) |
| **post-concat (feature)** | C06, C07 | **STOP ×2** (ΔAUROC −0.279 / −0.212; d′ −0.055 / +0.144) | fitted normal-only |
| **memory representation** | C08 | **STOP** (ΔAUROC −0.306, d′ −0.110) | fitted normal-only |

* The **feature-level** and **memory-level** insertion points failed **completely** (4/4 combined
  negative illuminations, all three AUROC < 0.56 and none above chance).
* The **input-level** insertion point did not fail, but it also did not deliver a robustness gain —
  it delivered *preservation*. On the frozen rule that is exactly a HOLD, not a GO.
* **The pattern that separates the two groups is not the insertion point but whether the module is
  fitted.** All three fitted modules failed; the only training-free module survived. A normal-only
  objective (synthetic-noise discrimination, contrastive pseudo-labels, or feature reconstruction)
  applied to a **frozen** embedding has an obvious degenerate optimum — "ignore the input" /
  "reproduce the dominant nuisance factor" — and nothing in these three losses forbids it. This is a
  falsifiable, mechanism-level explanation that is consistent with *all* the numbers above, including
  why C01 (which cannot collapse, having no trained parameters) is immune.

---

## 4. Q4 — Does this mean the "module insertion" route is globally unsupported?

**Within the tested envelope: yes, for fitted modules. The evidence is strong but bounded — state the
bounds explicitly.**

* Supported: *"Under this frozen protocol (M²AD Bird, view 120, seed 0, 1 200-image bank, frozen
  `wide_resnet50_2` layer2+layer3 embedding, normal-only fitting), **no fitted module inserted at
  the feature or memory level improved the robustness–preservation trade-off**; all three were
  strictly worse than the plain PatchCore baseline on both axes."* This is a **negative** result and
  it is genuine — it is not a tuning artefact, because nothing was tuned.
* **Also supported and arguably more important:** the E5-1 primary robustness metric `R_all` is
  **not sufficient** as a standalone objective — it can be lowered by a representation that simply
  stopped responding to illumination *and* to defects (C08: `R_ratio` 0.465 with AUROC 0.462,
  d′ −0.110, ρ −0.402). Any future round must read it jointly with `Δd′`/`ΔAUROC`, and the C01
  raw-vs-z sign disagreement (§1) should be resolved.
* **Not supported / do not over-claim:** that "module insertion cannot work in general". Counter-
  evidence exists inside this very round (C01's `Δd′` gain came from the input level with zero
  training). The defensible statement is narrower and mechanism-based: *the three fitted modules
  failed for identifiable reasons (degenerate optimum under normal-only fitting on a frozen
  embedding), not because insertion is impossible.*
* **Structural limit of the evidence:** 1 category × 1 view × 1 seed × 4 candidates × 1 backbone.
  Per E4-D1/E4-X precedent this cannot be extrapolated to other categories, views or seeds.

---

## 5. Q5 — Which family should the next stage move to?

Ranked by what this round actually evidenced:

1. **Input-level / training-free photometric normalisation (C01's family) — the only direction with
   any positive signal.** It is also the only one that cannot degenerate. Next step is **not** a new
   module: it is to **re-establish what C01 actually does** (ρ = 0.90 with Original; Δd′ up; raw
   illumination share down 31% while `R_all` says +3%). Until that sign disagreement is explained,
   no robustness claim from this branch is trustworthy.
2. **Robustness-metric re-validation before any further module work.** `R_all` alone is now known to
   be gameable; a replacement/composite objective must be pre-registered and validated on the frozen
   Original before it is used to judge any new candidate.
3. **Only then, if a module route is retried:** it must be one that is either **training-free** or
   whose objective **cannot** be satisfied by ignoring the input — i.e. explicitly constrained rather
   than purely normal-only-reconstructive.
4. **Explicitly deprioritised by this round:** synthetic-anomaly discriminators on frozen embeddings
   (C06), contrastive representation learning without defect-side anchoring (C07), and continuous
   memory reconstruction (C08). Re-running any of them, or their tuned variants, is forbidden by
   protocol §17.
5. **Closed and not to be reopened here:** B2, X6c, C2/C6 families, and the E5-0B HOLDs
   (C02/C04/C05/C09) and C10 top-up.

**GPU stays idle. No new experiment is created to use the machine.**

---

## 6. What this round advanced (paper ledger)

* **E5-1's negative result is now explained, not just reported**: all three fitted modules fail
  globally across illumination, two of them by **inverting** the anomaly ordering, and all three
  **amplify** illumination dependence in the NG score (0.112 → 0.17–0.40).
* **A measurement hazard is documented and quantified**: `R_all` can be reduced by a representation
  that responds neither to illumination nor to defects; and for C01 the raw illumination share and
  `R_all` disagree in sign.
* **The branch is closed with reason**: `Module Composition Screening` produced no survivor; the
  route forward is metric re-validation and input-level (training-free) work, not more module
  substitution.
* **Limits are on record**: 1 category × 1 view × 1 seed × 1 backbone; negative result, not proof of
  impossibility.

## 7. Next step

**Human review required.** This audit authorises **nothing automatic**: no E5-2 composition, no
rescues, no C10, no parameter or threshold tuning. If a next round is opened, the two pre-conditions
above (explain C01's metric sign disagreement; pre-register a non-gameable robustness objective)
should be satisfied first.
