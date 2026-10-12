# E8 — CLAHE × PatchCore Cross-Category Stability Kill Test

```text
📍 PAPER STAGE          Phase III — Final Candidate Stability Validation
🎯 CLAIM UNDER TEST     training-free CLAHE preprocessing improves PatchCore robustness
                        under real illumination, while preserving anomaly discriminability
🚦 EXIT                 STOP — the Bird-only gain does not generalise across categories
```

## 1. Question

> Does the E7 CLAHE → PatchCore gain (mean +0.026 AUROC on Bird, 3/4 views) hold up
> **cross-category and cross-seed**, or was Bird a single-category accident?

## 2. Hypothesis

CLAHE (L-channel, `tile=(8,8)`, clip 1.5 / 2.0) suppresses illumination variation, so it should
help PatchCore on any multi-illumination object — not just Bird.

## 3. Protocol

Frozen **before any E8 result** in `docs/E8_FROZEN_PROTOCOL.md` (commit `f332ee1`,
sha256 `3d0fe8e0…`). Deviations D1–D3 logged pre-result in its §9.

| item | value |
|---|---|
| candidates | `P00` (Original PatchCore), `P02` (CLAHE mild 1.5), `P03` (CLAHE medium 2.0) |
| categories | **Bird, Car, Cube** (mechanical: E4-D0 official-order continuation) |
| seeds | 0, 1, 2 |
| views | `120, 090, 240, 330` (E7-2 frozen 4-view protocol) |
| bank | **300 images** (E7-1 frozen rule `random.Random(20261009+seed).sample(1200, 300)`), uniform across categories |
| PatchCore | `wide_resnet50_2`, layer2+layer3, coreset 0.1, k=9, α=0, input 256 (unchanged from E7) |
| primary metric | **image AUROC**; secondary AUPR, `d′`, pixel AUROC / AUPRO |
| matrix | 3 methods × 3 categories × 3 seeds × 4 views = **108 cells**, 27 fits |

## 4. Pipeline fidelity (Stage B regression gate)

Bird seed0 vs E7-1 reference (300-bank, view 120) — **PASS** (max |dev| = 0.0097, ranking preserved):

| method | E8 AUROC@120 | E7-1 ref | dev |
|---|---:|---:|---:|
| P00 | 0.77120 | 0.78089 | -0.0097 |
| P02 | 0.78987 | 0.79458 | -0.0047 |
| P03 | 0.79588 | 0.79591 | **-0.00003** |

P03 reproduces E7-1 essentially exactly ⇒ the E8 pipeline (cache-based inputs) is bit-faithful.

## 5. Result (108 cells)

| method | mean ΔAUROC | 95% bootstrap CI | positive categories | mean Δd′ | verdict |
|---|---:|---|---:|---:|---|
| P02 (CLAHE mild) | **-0.0150** | [-0.023, -0.006] | **1/3** | -0.057 | **STOP** |
| P03 (CLAHE medium) | **-0.0170** | [-0.027, -0.006] | **1/3** | -0.062 | **STOP** |

The 95% CI lies **entirely below zero** for both variants ⇒ the cross-category mean is reliably
negative, not merely null.

## 6. Category-wise behaviour (mean over seeds × views)

| category | P00 | P02 | ΔP02 | P03 | ΔP03 |
|---|---:|---:|---:|---:|---:|
| **Bird** | 0.7784 | 0.7914 | **+0.0130** | 0.7982 | **+0.0198** |
| **Car** | 0.6637 | 0.6273 | **-0.0364** | 0.6155 | **-0.0482** |
| **Cube** | 0.6025 | 0.5810 | **-0.0215** | 0.5799 | **-0.0225** |

- **Bird** is positive at 3/4 views (E7 reproduced).
- **Car and Cube are negative at 4/4 views** (0 positive views each).

Per-seed (mean ΔAUROC) is consistently negative for both variants (-0.014 … -0.018), so this is
not a seed artifact. Per-view means are negative at all 4 views for both variants.

## 7. Verdict — STOP

CLAHE's benefit is **category-specific**: it helps Bird but systematically degrades Car and Cube
(both AUROC and `d′`). This is the exact "Bird ↑ / Car ↓↓ / Cube ↓" pattern the protocol
pre-registered as STOP. **No candidate reached the pre-registered standard.**

## 8. Why STOP (mechanism, honest reading)

CLAHE stretches the L-channel contrast globally; whether that helps or hurts depends on the
object's texture/defect contrast. On Bird the defect contrast survives and illumination is
flattened; on Car/Cube the stretch appears to amplify nuisance variation relative to the defect
signal. The negative `Δd′` (-0.057 / -0.062) means CLAHE is *not* merely neutral on preservation —
it slightly damages normal-vs-defect separation on average.

## 9. Negative results (kept, not hidden)

- Car and Cube are **4/4 views negative** for both CLAHE strengths.
- `fig2_delta_heatmap.png` and `fig5` (when ≥4 categories) preserve these cells explicitly.

## 10. Next action

STOP the CLAHE route. Proceed to **E9 — Real-Illumination Robust Module/Combination Screening**
with the Retinex (C01) weak-positive signal as reference, and audit input-level / feature-level /
score-level normalisation families. Do **not** tune CLAHE further.

## 11. Runtime / cost

Stage A (prep, incl. streaming Car+Cube) ≈ 45 min · Stage B 3 fits ≈ 8 min · Stage C 24 fits ≈
70 min · analysis ≈ 2 min. **Peak VRAM ≈ 1188 MiB**. 27 fits total, no GPU rerun needed for archive.
