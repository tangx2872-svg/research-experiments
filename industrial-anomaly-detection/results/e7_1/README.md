# E7-1 — Lightweight Composition Broad Screening

**Stage:** Phase — Lightweight Module Composition Screening
**Protocol:** [`docs/E7_1_FROZEN_PROTOCOL.md`](../../docs/E7_1_FROZEN_PROTOCOL.md) — frozen at commit
**`af56dd3`**, **before any candidate final metric existed**.
**Role:** this round *implemented and executed* an upstream-frozen design. No candidate was added,
removed or retuned; no threshold was changed.

**Runtime: 19 min 15 s** (protocol §14 budget 30–60 min target / 90 min hard max).
GPU: RTX 3090, 1 worker, peak VRAM 14,497 MB alloc / 18,258 MB reserved (P00-family fits).

---

## 1. Setup (as frozen)

| item | value |
|---|---|
| dataset | M²AD **Bird**, seed **0**, view **120** |
| test set | **unreduced**: 700 images (200 Good + 500 NG), illuminations 01–10 |
| normal bank | **25 % subset** = **300 images** drawn from the E5-1 1200-image bank with `BANK_SUBSET_SEED=20261009` (§12-A1) |
| photometric caches | one-time, at **448×448** (§12-A2): Retinex (E5-1-verified PIAD `Separate`), CLAHE mild (1.5), CLAHE medium (2.0), Retinex→CLAHE-mild |
| metrics | **image AUROC (primary)**, image AUPR, `d′`; pixel AUROC / AUPRO where a map exists |
| gate | tiers A–D, relative to same-protocol P00 (§8) |

Scope sanity **5/5 PASS**: test unreduced (700) · bank subset 300/1200 · bank normal-only ·
**bank ∩ test specimen overlap = 0** (no leakage) · illuminations 01–10 present.

## 2. Stage 0 smoke (mandatory)

14/14 pipelines executed on 10 normal + 20 test images. Engineering checks
(shape / finite / dependency / normal-only / no-leakage) **PASS for all 14**.
`smoke.csv` records the run; `logs/smoke3.log` is the full log.

* **`score_direction` reported FAIL for P05/P06/P07/P08.** Root cause **investigated and resolved**:
  the mandated smoke bank is 10 images drawn from **10 distinct specimens**, which leaves DINOv2's
  nearest-neighbour reference with almost no coverage, so the direction check is not meaningful at
  that size. Re-measured at the real E7-1 bank scale (`logs/smoke_direction_diagnostic.json`):

  | backbone | AUROC | AUPR | d′ | Good | NG | direction |
  |---|---|---|---|---|---|---|
  | dinov2_vits14 | 0.71172 | 0.88318 | 0.9083 | 0.1973 | 0.2429 | **correct** |
  | dinov2_vitb14 | 0.73158 | 0.89527 | 1.0112 | 0.2183 | 0.2756 | **correct** |

  ⇒ **no implementation defect**; the smoke sample size is the cause. Recorded as an engineering
  finding, not hidden.

## 3. Results (full 700-image test set, 300-image bank)

| ID | pipeline | AUROC | ΔAUROC | AUPR | ΔAUPR | d′ | runtime | decision |
|---|---|---|---|---|---|---|---|---|
| **P03** | CLAHE medium → PatchCore | **0.79591** | **+0.01502** | 0.91479 | −0.00243 | 1.2175 | 106 s | **KEEP** |
| **P02** | CLAHE mild → PatchCore | **0.79458** | **+0.01369** | 0.91736 | +0.00013 | 1.2157 | 106 s | **KEEP** |
| P00 | Original PatchCore | 0.78089 | +0.00000 | 0.91723 | +0.00000 | 1.2258 | 131 s | REFERENCE |
| P04 | Retinex → CLAHE mild → PatchCore | 0.77662 | −0.00427 | 0.90993 | −0.00730 | 1.1328 | 108 s | STOP (D) |
| P01 | Retinex → PatchCore | 0.77483 | −0.00606 | 0.91197 | −0.00526 | 1.1445 | 108 s | STOP (D) |
| P10 | SoftPatch nearest | 0.74126 | −0.03963 | 0.84150 | −0.07572 | 0.8922 | 88 s | STOP (D) |
| P06 | DINOv2-B → NN | 0.73158 | −0.04931 | 0.89527 | −0.02196 | 1.0112 | 81 s | STOP (D) |
| P05 | DINOv2-S → NN | 0.71172 | −0.06917 | 0.88318 | −0.03405 | 0.9083 | 58 s | STOP (D) |
| P11 | SoftPatch LOF | 0.70932 | −0.07157 | 0.86058 | −0.05664 | 0.8014 | 104 s | STOP (D) |
| P13 | Retinex → SoftPatch LOF | 0.69566 | −0.08523 | 0.82117 | −0.09605 | 0.6638 | 80 s | STOP (D) |
| P09 | CLAHE mild → DINOv2-S → NN | 0.68634 | −0.09455 | 0.87089 | −0.04634 | 0.7993 | 30 s | STOP (D) |
| P08 | Retinex → DINOv2-B → NN | 0.68052 | −0.10037 | 0.86210 | −0.05513 | 0.7531 | 57 s | STOP (D) |
| P12 | Retinex → SoftPatch nearest | 0.67432 | −0.10657 | 0.79204 | −0.12518 | 0.2178 | 66 s | STOP (D) |
| P07 | Retinex → DINOv2-S → NN | 0.66963 | −0.11126 | 0.85639 | −0.06084 | 0.7240 | 32 s | STOP (D) |

`ADVANCE = []` · `KEEP = [P03, P02]` · `BACKUP = []` · `STOP = [11 pipelines]`

Pixel-level extras (where masks exist; 380 mask-bearing NG images):
P00 pixel AUROC 0.9723 / AUPRO 0.8432 · P02 0.9733 / 0.8348 · P03 0.9717 / 0.8277.

### 3.1 What the screen says

* **Only the CLAHE family clears the bar.** Mild and medium CLAHE both raise image AUROC by
  +1.4–1.5 pts, i.e. **tier B (KEEP)**. Neither reaches tier A (+0.020).
* **Retinex does not help**: P01 is −0.006 and every Retinex combination (P04/P07/P08/P12/P13) is
  worse than its best parent. This is consistent with E5-1/E6-A, where PIAD-Retinex bought
  `d′` but not AUROC.
* **DINOv2-S/B nearest-neighbour detectors are clearly behind PatchCore here** (−0.049 … −0.111).
  They are not "bad implementations" — the AnomalyDINO path reproduces its own protocol
  (448, L2-normalised kNN, `k=1`, `mean_top1p`) — they are simply weaker *on this dataset at this
  bank size*.
* **SoftPatch loses to plain PatchCore** (−0.040 nearest, −0.072 LOF). Its soft weighting is designed
  for **noisy** memory banks; on a clean bank it only perturbs the distance field.
* **A stronger `d′` does not imply a stronger AUROC**: P00 has the highest `d′` (1.2258) and P02 the
  second-lowest ΔAUPR (−0.0001 ≈ flat). The AUROC gains of P02/P03 come with a small AUPR/`d′` cost.

## 4. Composition synergy (protocol §9)

| A | B | A+B | ΔA(A) | ΔA(B) | ΔA(A+B) | synergy gain | positive? |
|---|---|---|---|---|---|---|---|
| P01 | P02 | P04 | −0.00606 | +0.01369 | −0.00427 | **−0.01796** | no |
| P01 | P05 | P07 | −0.00606 | −0.06917 | −0.11126 | −0.10520 | no |
| P01 | P06 | P08 | −0.00606 | −0.04931 | −0.10037 | −0.09431 | no |
| P02 | P05 | P09 | +0.01369 | −0.06917 | −0.09455 | −0.10824 | no |
| P01 | P10 | P12 | −0.00606 | −0.03963 | −0.10657 | −0.10051 | no |
| P01 | P11 | P13 | −0.00606 | −0.07157 | −0.08523 | −0.07917 | no |

**No composition is positive.** In every A+B cell the combination is *worse than its best parent* —
adding a module consistently destroys part of the other module's contribution. The single-module
CLAHE entries (P02/P03) therefore carry the whole signal of this round.

## 5. E7-1 → E7-2 gate (protocol §11)

```
ADVANCE = []          KEEP = [P03, P02]        BACKUP = []        STOP = 11 pipelines
SELECTED = [P03, P02]        proceed_to_e7_2 = True     (2 candidates, <= 4)
```

E7-2 was started with **P00 (reference) + P03 + P02**, full 1200-image bank, and the 4 frozen views
`['120', '090', '240', '330']` (primary 120 + 3 drawn by `VIEW_SEED=20261009`).

## 6. Engineering fixes made during the round (allowed by §G)

| # | issue | fix | method impact |
|---|---|---|---|
| E1 | `E7.Bar` had `.step()` but pipelines called `bar(i)` | added `__call__` | none |
| E2 | SoftPatch's `common.py` imports **faiss** at module level, absent in this environment | **installed `faiss-cpu` 1.15.1** from PyPI (AnomalyDINO/SoftPatch both declare it) | none — dependency fix |
| E3 | SoftPatch loader returned the *class* instead of the *module* | return the module | none |
| E4 | SoftPatch's effective coreset sampler is `WeightedGreedyCoresetSampler` (`softpatch.py:85`), not the CLI's `ApproximateGreedyCoresetSampler` | used the official class that consumes the weights | none — matches the official code path |
| E5 | `_compute_patch_weight` needs `feature_shape`; it was unset | set to the real `(32, 32)` embedding grid | none |
| E6 | Stage-0 smoke direction check spanned only 1 specimen per class | smoke sampling now takes one image per **distinct** specimen | none (diagnostic quality) |

No fix touched a method definition, architecture, loss, feature layer or scoring rule.

## 7. Assumptions carried from the protocol (§12) — all still open for designer confirmation

**A1 bank-subset base** (25 % of the E5-1 1200-image bank = 300 images) ·
**A2 cache resolution 448** · **A3 DINOv2 masking disabled** · **A4 no `τ_val` step** ·
**A5 SoftPatch mechanism applied to the shared PatchCloud representation** ·
**A6 faiss absent → exact torch kNN** (now moot for SoftPatch since faiss-cpu was installed, but the
SoftPatch/DINOv2 scoring paths already used exact kNN and were left unchanged to avoid a mid-round
change).

## 8. Limitations

* **One category, one view, one seed, one backbone family per pipeline group.** Nothing here
  extrapolates to other M²AD categories or to MVTec.
* **The bank is 300 images**, so the absolute AUROC of every pipeline is lower than the E5-1
  full-bank reference (P00: 0.78089 here vs 0.76766 under E5-1's *different* 1200-image bank at the
  same view — the two are **not** comparable; only within-E7 comparisons are valid).
* **Tier B margins are small** (+0.0137 / +0.0150 vs the +0.010 bar) and the CLAHE–baseline gap is
  the only effect this round found; E7-2 tests whether it survives a view change.
* `d′` and AUPR move slightly **against** the AUROC gain for P03, so the improvement is not uniform
  across metrics.

## 9. Artifacts

```
results/e7_1/
├── README.md (this file)   smoke.csv   summary.csv   synergy.csv   gate_e7_2.json
├── progress.json           raw/P00.json … raw/P13.json   raw/Pxx_per_image.csv
├── figures/ fig1_pipeline_auroc_ranking.png · fig2_delta_and_runtime.png · fig3_synergy.png
└── logs/ e7_1_run.log · smoke3.log · smoke_direction_diagnostic.json
scripts/e7_common.py · e7_pipelines.py · e7_1_runner.py · e7_2_runner.py · e7_figures.py
```

## 10. Next step

E7-2 automatically followed the gate at 2 candidates (protocol §11/§15) — see
[`results/e7_2/README.md`](../e7_2/README.md). **No E7-3, no multi-seed, no second category and no
parameter rescue is authorised.** Human review required before anything further.
