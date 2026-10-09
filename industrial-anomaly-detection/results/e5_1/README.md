# E5-1 — Broad Module Composition Screening

**Stage:** Phase III — Module Composition Screening
**Protocol:** [`docs/E5_1_FROZEN_PROTOCOL.md`](../../docs/E5_1_FROZEN_PROTOCOL.md) — frozen at commit
`e96cd5a` **before** any candidate result existed (`protocol_sha256 = 32159221…6035d`).
**Verdict of this round: `0 GO / 1 HOLD / 3 STOP`** ⇒ protocol §17 case **C**.

---

## 1. Why (purpose)

E4-X killed B2 and X6c on M²AD Bird real illumination (`B2 = STOP`, `X6c = STOP`, no rescue).
E5-0/E5-0B reduced 17 candidate modules to 4 eligible ones with **four different insertion points**.
E5-1 asks exactly one question:

> Of these 4 mature modules with different mechanisms, which — if any — is worth advancing under
> real illumination conditions?

It is a **screening**, not a method hunt. No tuning, no rescue, no new thresholds.

## 2. Frozen protocol (summary)

| item | value |
|---|---|
| dataset | M²AD, category **Bird** (E4-0C screening winner) |
| split | official (`normal_ratio=0.6`, `seed=42`, NG → test) |
| frozen view | **120** (`VIEW_SELECTION_SEED=20261008`) — not changed |
| seed | 0 |
| bank | 1200 images = 10 normal train specimens × 12 views × 10 illuminations |
| scoring set | 700 images = test split @ view 120 (200 Good + 500 NG) |
| input | 256×256, ImageNet normalisation |
| backbone | `wide_resnet50_2` `layer2`+`layer3` → embedding **1536×32×32** (1024 patches/img) |
| Original reference | **reused from E4-X, 0 GPU** (AUROC 0.76766 / d′ 1.20676 / R_all 0.67543) |
| advance criteria | pre-registered in protocol §7 (GO-A / GO-B / STOP-s1,s2,s3,s6 / else HOLD) |

## 3. Candidates and provenance

| id | module | insertion | implementation status |
|---|---|---|---|
| C01 | PIAD-Retinex | input photometric | **official** `Kaichen-Yang/piad_baseline` @`67b816b`, vendored URetinex-Net (MIT © AndersonYong); PIAD's own path `Retinex.RL_separate.Separate` → reflectance `R` |
| C06 | SimpleNet | post-concat | **official** `DonaldRR/SimpleNet` @`351a2b8` (MIT © DonaldRR): `Projection` + `Discriminator`, synthetic anomalies only |
| C07 | ReConPatch | post-concat | **FROM-PAPER MINIMAL** (arXiv:2305.16713) — no official implementation located |
| C08 | CRAD | memory representation | **ADAPTED FROM CRAD** `tae-mo/CRAD` @`b5a1c472` (repo declares **no license**) — official class in our frozen feature space |

Declared engineering adaptations (protocol §12 deviations D1–D5): `ch_exp=864` kept absolute for C08
(ratio-preserving `6144` would need ≈4.15×10⁸ parameters); `Adam` instead of unavailable `AdamP`
for C07; brute-force kNN instead of unavailable `faiss`; `add_safe_globals` for the PIAD checkpoint
under torch 2.8; C01 Retinex applied at the frozen 256×256 input.

## 4. Results

| Candidate | AUROC | ΔAUROC | d′ | Δd′ | R_all | R_ratio | **Decision** |
|---|---|---|---|---|---|---|---|
| Original (reference) | 0.76766 | +0.00000 | 1.20676 | +0.00000 | 0.67543 | 1.0000 | REFERENCE |
| **C01 PIAD-Retinex** (input) | **0.78043** | **+0.01277** | **1.30552** | **+0.09876** | 0.69617 | 1.0307 | **HOLD** |
| C06 SimpleNet (post-concat) | 0.48820 | −0.27946 | −0.05534 | −1.26210 | 0.50851 | 0.7529 | **STOP** |
| C07 ReConPatch (post-concat) | 0.55555 | −0.21211 | 0.14363 | −1.06313 | 0.67286 | 0.9962 | **STOP** |
| C08 CRAD (memory) | 0.46197 | −0.30569 | −0.11035 | −1.31711 | 0.31423 | 0.4652 | **STOP** |

Ranking on four axes (protocol §9 — never AUROC alone):

| Candidate | Detection rank | Robustness rank | Preservation rank | Overall |
|---|---|---|---|---|
| C01 | **1** | 4 | **1** | **HOLD** |
| C06 | 3 | 2 | 3 | STOP |
| C07 | 2 | 3 | 2 | STOP |
| C08 | 4 | **1** | 4 | STOP |

### 4.1 Negative results (reported in full, nothing hidden)

* **C08 CRAD** — `R_ratio = 0.465` (by far the "most robust" number in the round) is **entirely
  hollow**: AUROC 0.462 is *below chance* and d′ is **negative** (−0.110). Collapsing all scores
  toward a constant trivially shrinks cross-illumination dispersion. This is exactly the
  `STOP-s2` failure mode ("improvement obtained mainly by sacrificing d′"). The training loss fell
  monotonically (0.0417 → 0.0382) without any detection capability emerging.
* **C06 SimpleNet** — AUROC 0.488, d′ −0.055. Its discriminator converged to a degenerate state
  (final training loss 1.0000 = exactly the margin loss floor `2·θ` with θ=0.5): the discriminator
  stopped separating true from synthetic-noise features at all. `R_ratio = 0.753` is again hollow.
* **C07 ReConPatch** — AUROC 0.556, d′ 0.144 (both far below Original). Its contrastive objective
  *did* learn (loss 9.25 → 0.79), but the learned modulation destroyed more signal than it removed
  illumination nuisance: `Δd′ = −1.063`. `R_ratio = 0.996` ⇒ no robustness gain either
  (`STOP-s1`).
* **No candidate reached GO.** Three of four candidates produced a *lower* AUROC **and** a *lower* d′
  than the plain PatchCore baseline, while showing apparently better `R_ratio` — i.e. **the frozen
  robustness metric `R_all` is trivially minimisable by degrading detection**. This is itself an
  important methodological finding (see §5).

### 4.2 The one positive signal — C01 (HOLD)

C01 PIAD-Retinex is the only candidate that is **not** a failure:

* `ΔAUROC = +0.01277` → inside the pre-registered HOLD band `[+0.01, +0.03)`; **never reported as a
  meaningful detection win** (protocol §7).
* `Δd′ = +0.09876` → a real improvement in normal-vs-defect separation (+8.2%).
* `R_ratio = 1.0307` → illumination robustness **slightly worse**, not better.

So the input-photometric route buys defect separation, not robustness — the opposite of what a
"robustness module" is supposed to do. Under the frozen rule it cannot be GO (`ΔAUROC < +0.03` and
`R_ratio > 0.90`).

Leave-one-illumination-out (informational — §15 only gates GO candidates; there is no GO):
`ΔAUROC` falls below `+0.01` when illumination **06** or **07** is removed, i.e. part of C01's
small gain is concentrated in two lighting conditions.

## 5. What this round advanced

1. **All three "representational" insertion points (post-concat ×2, memory ×1) actively hurt** on
   real illumination when the modules are trained normal-only on this frozen feature space. The
   input-photometric insertion is the only one that does not.
2. **A methodological hazard was exposed and is now on record**: `R_all` alone is not a valid
   robustness objective. Any module that compresses score variance (including a collapsed one)
   improves `R_ratio`. All future rounds must read `R_ratio` only jointly with `Δd′`/`ΔAUROC` —
   which is exactly what the frozen E5-1 rule already enforces, and why C08/C06 were stopped rather
   than celebrated.
3. **The E5-0/E5-0B composite hypothesis is not supported**: "swap in a mature module at a better
   insertion point" did not produce a module worth composing. The result strengthens the E4-X
   conclusion that the illumination robustness–defect preservation trade-off is not solved by
   architectural substitution on this feature space.

## 6. Runtime and resources

| stage | wall time |
|---|---|
| E5-1P preflight (incl. official-source recovery) | ≈1 h (over the 10–20 min estimate — see note) |
| C01 (Retinex prep 85 s + PatchCore fit 744.6 s + scoring 25.6 s) | 776.3 s (12.9 min) |
| C08 | 5569.1 s (92.8 min) |
| C07 | 640.3 s (10.7 min) |
| C06 | 524.1 s (8.7 min) |
| **total GPU wall** | **7510 s ≈ 2 h 05 min** |

* GPU: RTX 3090 24 GB. Peak VRAM: C01 14497 MB allocated / 18258 MB reserved; C08 15310 MB reserved.
* 1 worker throughout; no concurrent GPU-heavy fits.
* C08 is 74% of total GPU time (97.4 M parameters, 50 CRAD epochs at batch 16).
* **Preflight overrun note:** the 10–20 min estimate assumed candidate sources were already
  available. They were not: the server's GitHub reachability had changed since E4-D0, three
  official repositories were cloned, and C01's provenance chain was resolved — this was the single
  highest-risk item of the round and it is now closed with zero-invention provenance.

## 7. Sanity / integrity

| id | test | result |
|---|---|---|
| S1 | scope = 1200 bank / 240 val / 700 scoring, view 120, illuminations 01–10 | PASS |
| S3 | Original asset present, sha256 `4aebd115…0a5d` | PASS |
| S7 | bank ∩ scoring specimens = 0 | PASS |
| S8 | bank = train-split Good specimens only | PASS |
| S4/S5 | embedding shape `(1536,32,32)`; all embeddings and scores finite | PASS (runtime assertion) |
| S9 | no anomaly-label leakage (scores never used to fit) | PASS by construction |
| S10 | no threshold / hyper-parameter / view / seed search | PASS |

**Leakage: NO.** **Protocol deviations: 5, all declared and logged in protocol §12 (D1–D5) and all
made before any candidate result was read.**

## 8. Artifacts

```
results/e5_1/
├── README.md                     (this file)
├── progress.json                 per-candidate status/metrics/artifacts/git HEAD/seed/protocol hash
├── raw/
│   ├── C01.json  C01_per_image.csv  C01_weights.pt (n/a — PatchCore)
│   ├── C06.json  C06_per_image.csv  C06_weights.pt
│   ├── C07.json  C07_per_image.csv  C07_weights.pt
│   └── C08.json  C08_per_image.csv  C08_weights.pt
├── analysis/summary.csv, summary.md
├── figures/fig1_auroc_vs_robustness.png, fig2_deltas.png, fig3_illumination_stability.png
└── logs/e5_1_run.log, e5_1_run2.log, retinex_prep.log
```

## 9. Next step (§17 case C — **0 GO**)

**STOP the Module Composition branch.** Protocol §17 forbids `C01′`, `C01++`, `C07 tuned`,
`C08 rescue`, automatic C10, and any automatic composition. Route re-evaluation is required.

If a next round is authorised, the only evidence-backed direction from this round is the
**input-photometric** insertion point (C01, HOLD) — and even that must first explain why its
`Δd′` gain comes **without** any robustness gain, plus a principled look at whether `R_all` should
be replaced as the primary robustness objective (finding §5.2).

**Human review required before any further action.** No E5-2 was started.
