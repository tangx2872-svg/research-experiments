# E9 — Real-Illumination Robust Module Screening

```text
📍 PAPER STAGE          Phase III — Final Candidate Stability Validation
🎯 QUESTION             Is there a mature low-complexity illumination / photometric / feature /
                        score module that, alone or combined with Retinex, gives a STABLE
                        cross-category improvement over Original PatchCore?
🚦 EXIT                 NO SURVIVOR (Round-1 ADVANCE not confirmed at 3 seeds x 2 views)
```

## 1. Question

E5-1 (module insertion), E7 (photometric CLAHE), and E8 (CLAHE cross-category) all failed to
produce a cross-category-robust improvement. The only weak positive ever seen was Retinex (C01,
`ΔAUROC +0.0128` on Bird, but `R_ratio 1.03`). E9 audits the remaining mature, low-complexity
**illumination-normalisation / feature-normalisation** families for a stable cross-category signal.

## 2. Candidate pool (deduped against history)

Removed as duplicates of already-STOPped work (from `docs/MODULE_CANDIDATE_REGISTRY.md` +
E5-1/E7/E8): α-IN & all feature-reparameterisation variants, L2-residual rescaling (B2),
score-fusion (X6c), illumination-nuisance subspace projection, category×channel gating, CLAHE,
DINOv2/SoftPatch, SimpleNet/ReConPatch/CRAD. Score-level calibration is monotone (cannot change
AUROC) and was excluded as a standalone candidate.

| id | family | mechanism | dedup status |
|---|---|---|---|
| A0 | retinex | PIAD official reflectance R → PatchCore | reference (C01) |
| I01 | input | Gray-World (per-channel mean → global mean) | OPEN |
| I02 | input | Shades-of-Gray (p=6) | OPEN |
| I03 | input | Gamma (γ=0.8) | OPEN |
| I04 | input | Homomorphic filter (log-FFT high-pass, D0=32) | OPEN |
| F01 | feature | L2 feature normalisation (cosine kNN) | OPEN |
| A1 | combo | Retinex + L2 feature norm | OPEN |

## 3. Protocol

- Scope: **300-image bank** (E7-1 seed-0 rule) × **test @ view 120 (700)** — Round 1.
- Baseline: **E8 Original P00** (reused, 0 GPU), consumed from `l0_256`.
- Feature-level candidates (F01/A1) via runtime monkeypatch of `generate_embedding` (L2-norm per
  patch), keeping the frozen coreset/kNN/max-pool scoring.
- Smoke: Bird/seed0/12-bank+12-Good+12-NG subset, all 7 candidates → **7/7 PASS** (finite
  AUROC/d′, no NaN/Inf, no leakage, 25–36 s/candidate, ≤600 MB VRAM).

## 4. Round 1 results (3 cats × 7 candidates × seed0 × view 120 = 21 units)

| candidate | Bird Δ | Car Δ | Cube Δ | Mean | Δd′ | pos | verdict |
|---|---:|---:|---:|---:|---:|---:|---|
| **I03 Gamma** | +0.0078 | +0.0078 | -0.0044 | **+0.0037** | +0.0199 | 2/3 | ADVANCE |
| F01 L2-norm | -0.0033 | +0.0261 | -0.0065 | +0.0054 | +0.0042 | 1/3 | HOLD |
| I01 Gray-World | +0.0214 | -0.0017 | -0.0113 | +0.0028 | +0.0015 | 1/3 | HOLD |
| I04 Homomorphic | -0.0039 | +0.0133 | -0.0023 | +0.0024 | -0.0141 | 1/3 | HOLD |
| I02 Shades-of-Gray | +0.0352 | -0.0038 | -0.0333 | -0.0006 | -0.0064 | 1/3 | STOP |
| A0 Retinex | +0.0074 | -0.0226 | -0.0168 | -0.0107 | -0.0597 | 1/3 | STOP |
| A1 Retinex+L2 | +0.0068 | -0.0457 | -0.0497 | -0.0295 | -0.1166 | 1/3 | STOP |

Only **I03 (Gamma)** met the ADVANCE bar (2/3 positive + mean>0), but the effect is tiny
(+0.0037). Retinex (A0) and Shades-of-Gray (I02) reproduce the CLAHE pattern (Bird↑, Car/Cube↓).

## 5. Round 2 results (Top-2 = I03 + F01, 3 cats × 3 seeds × 2 views = 18 units)

| candidate | Bird (3 seeds) | Car (3 seeds) | Cube (3 seeds) | overall mean | verdict |
|---|---|---|---|---|---|
| I03 Gamma | **+0.0039** | -0.0019 | -0.0059 | **-0.0013** | NOT CONFIRMED |
| F01 L2-norm | -0.0344 | **+0.0407** | -0.0138 | -0.0025 | Car-specific |

- **I03 Gamma**: Round-1 positive Car (+0.0078) flips negative at seeds 1/2 (-0.0043/-0.0046);
  Cube is negative at all 3 seeds. Bird-only weak positive does not generalise.
- **F01 L2-norm**: strongly Car-specific (Car +0.04 consistent, Bird -0.034 consistent, Cube -0.014).

## 6. Verdict — NO SURVIVOR

Neither Round-1 ADVANCE survives the 3-seed × 2-view confirmation. Every low-complexity
illumination/feature-normalisation candidate shows **category-specific** effects (Bird-only,
Car-only, or negative), never a cross-category-consistent gain. This matches CLAHE (E8) and module
insertion (E5-1).

## 7. What this means for the paper

The evidence is now consistent across three independent rounds (E5-1, E8, E9): **no low-complexity
preprocessing / feature-normalisation / module-insertion transform produces a cross-category-robust
illumination-robustness gain over plain PatchCore on M²AD.** The route "add a cheap module to
PatchCore" is exhausted. A defensible next step is to either (a) accept the **null result** and
frame the paper as a careful negative study of illumination-robustness of PatchCore, or (b) pivot to
a method family that changes the *scoring objective* itself rather than preprocessing the input or
normalising features.

## 8. Cost

Round 1 ≈ 27 min (21 fits) · Round 2 ≈ 30 min (18 fits) · prep ≈ 15 min · smoke ≈ 7 min.
Peak VRAM ≈ 1.4 GB. No GPU rerun needed for archive.
