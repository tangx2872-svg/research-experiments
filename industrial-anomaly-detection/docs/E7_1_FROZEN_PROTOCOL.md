# E7-1 — Lightweight Composition Broad Screening (FROZEN PROTOCOL)

> **Status: FROZEN — written BEFORE any E7 candidate final metric was produced or read.**
> Frozen at git `c5b038a` + protocol commit (recorded in `results/e7_1/progress.json`).
>
> **Role boundary.** The candidate pool, module compositions, evaluation metrics and advance rules of
> this round were **frozen upstream**. This document does not design science — it records how the
> upstream design was implemented. No candidate was added, removed, or retuned; no threshold was
> changed. The only free parameters are the ones the upstream specification left open; **each such
> choice is listed in §12 as an explicit assumption** so it can be overridden by the designer.

```
CURRENT STAGE   = Phase: Lightweight Module Composition Screening
PRIMARY METRIC  = image-level AUROC          (E7-1 GO/STOP is driven by this alone)
SECONDARY       = image AUPR, d'
DIAGNOSTIC ONLY = R_all / illumination-sensitivity metrics  (MUST NOT decide E7-1 GO/STOP)
GPU WORK        = authorised for E7-1/E7-2 (pipeline execution only — never for design)
```

---

## 1. Purpose

Find a composition that is **literature-grounded, structurally simple, with a clear A/B division of
labour, and a stable improvement on standard anomaly-detection metrics** — good enough to support a
paper-sized incremental contribution. E7-1 is a **broad screen** of a frozen 14-pipeline registry.

## 2. Pipeline registry — FROZEN (no P14 may be added)

| ID | Pipeline | Family |
|---|---|---|
| **P00** | Original PatchCore | baseline |
| **P01** | PIAD Retinex → PatchCore | A-photometric |
| **P02** | CLAHE mild → PatchCore | A-photometric |
| **P03** | CLAHE medium → PatchCore | A-photometric |
| **P04** | PIAD Retinex → CLAHE mild → PatchCore | A+A |
| **P05** | DINOv2-**S** → nearest-neighbour anomaly detector | B-backbone |
| **P06** | DINOv2-**B** → nearest-neighbour anomaly detector | B-backbone |
| **P07** | PIAD Retinex → DINOv2-S → NN | A+B |
| **P08** | PIAD Retinex → DINOv2-B → NN | A+B |
| **P09** | CLAHE mild → DINOv2-S → NN | A+B |
| **P10** | SoftPatch **nearest** weighting | B-memory |
| **P11** | SoftPatch **LOF** weighting | B-memory |
| **P12** | PIAD Retinex → SoftPatch **nearest** | A+B |
| **P13** | PIAD Retinex → SoftPatch **LOF** | A+B |

`gaussian` weighting is explicitly **not** used (registry forbids a third weighting).

## 3. Implementation provenance (frozen)

| component | source | licence | verbatim facts used |
|---|---|---|---|
| **Retinex** | `third_party/piad_baseline/Retinex` @ `67b816b` (PIAD's vendored URetinex-Net), **already verified in E5-1** | MIT © 2022 AndersonYong | `Retinex.RL_separate.Separate` → reflectance `R` (PIAD's own call path). **Re-implementation forbidden; the mechanism is not touched.** |
| **DINOv2** | `facebookresearch/dinov2` via `torch.hub.load('facebookresearch/dinov2', name)` — exactly AnomalyDINO's own loader | Apache-2.0 (AnomalyDINO `b9d1c26`) | `resolution=448`, `Resize(smaller_edge=448, BICUBIC)`, `ToTensor`, ImageNet normalise, crop to multiple of `patch_size=14`; `model.get_intermediate_layers(x)[0]` (last-layer **patch tokens**, no CLS); **L2-normalise**; `knn_metric='L2_normalized'`; `k_neighbors=1` (**no k search**) |
| **SoftPatch** | `TencentYoutuResearch/AnomalyDetection-SoftPatch` @ `f8dc3e0` | ⚠️ **repository declares NO licence file** | official `_compute_nearest_distance` / `_compute_lof`, official coreset exclusion (`threshold=0.15`), official scoring `image_score = kNN distance × coreset_weight[matched index]`, `patch_maker.score` = **max over patches** |
| **CLAHE** | OpenCV `cv2.createCLAHE` | — | `mild: clipLimit=1.5`; `medium: clipLimit=2.0`; **both `tileGridSize=(8,8)`**. No further parameter search. |
| **NN search** | torch exact kNN (faiss is **not installed** in this environment) | — | numerically identical to `faiss.IndexFlatL2`; no method change |

## 4. Dataset, scope and the normal bank

| item | value |
|---|---|
| dataset | **M²AD, category Bird** |
| seed | **0** |
| view | **120** (identical to E5-1) |
| test set | **UNREDUCED**: all test images at view 120 = **700** (200 Good + 500 NG), illuminations 01–10 |
| labels / specimen IDs / illuminations | identical to E5-1 |
| **normal bank** | a frozen **25 % subset** of the normal memory/reference bank, drawn mechanically (§12-A1) |
| `BANK_SUBSET_SEED` | **20261009** |
| patch counts | **not** forced equal across pipelines — only the **normal source images** are shared |

**Shared-bank rule:** every pipeline that needs a normal reference uses **exactly the same source
image list**. Reference *structures* (coreset, memory bank, patch count) differ per method as each
official mechanism dictates.

## 5. Photometric caches (engineering, one-time)

Photometric transforms are applied **once** and cached, so 6 Retinex pipelines do not each pay for
Retinex:

* cache resolution **448 × 448** — the smallest size satisfying both consumers (DINOv2 needs
  smaller-edge 448; PatchCore then resizes to its frozen 256). Applied to the **original** image.
* caches: `retinex/`, `clahe_mild/`, `clahe_medium/`, `retinex+clahe_mild/`
* CLAHE is applied to the **L channel of LAB** (colour-preserving), then converted back to RGB.
* every backbones' own frozen preprocessing is applied **after** the cache.

## 6. Representation and scoring per family (frozen)

**PatchCore family (P00, P01, P02, P03, P04)** — the frozen E4-D1/E5-1 pipeline, unchanged:
`wide_resnet50_2`, layers `layer2`+`layer3`, input **256**, `coreset_sampling_ratio=0.1`,
`num_neighbors=9`, image score = PatchCore's max-over-patches (`α_l2=α_l3=0`).

**DINOv2 family (P05–P09)** — AnomalyDINO, unchanged:
input `448×448` (BICUBIC, ImageNet-norm, cropped to a multiple of 14 → 448), patch tokens from
`get_intermediate_layers(x)[0]`, **L2-normalised**, bank = concatenation of all bank-image patch
tokens, anomaly distance = **cosine distance to the 1-NN** (`faiss L2 on normalised vectors ÷ 2`),
image score = **`mean_top1p`** = mean of the largest 1 % of patch distances (max if that is < 1 patch).
Foreground masking: **disabled** (§12-A3). No k search (`k=1`).

**SoftPatch family (P10–P13)** — official mechanism on the same PatchCore representation:
patch weights via the official weight function over the bank patch features
(`nearest` = sum of the 2 smallest distances, +1; `LOF` = `lof_k=5`), then the official coreset
exclusion (`threshold=0.15`: patches above the 85th weight percentile are excluded from sampling),
coreset `0.1`, and the official score `image_score = kNN distance × coreset_weight[matched index]`,
aggregated by **max over patches**.

## 7. Metrics

`PRIMARY`: **image AUROC**. `SECONDARY`: image AUPR, `d'`. Naturally-available extras: pixel AUROC,
AUPRO when the pipeline yields an anomaly map (PatchCore = yes; DINOv2 = patch-distance map;
SoftPatch = yes). Per-image scores are saved for every pipeline.

`R_all` and the E6-A illumination-sensitivity metrics are **diagnostic only** and **must not** decide
E7-1 GO/STOP.

## 8. Decision rule (frozen; relative to P00 under the same protocol)

| tier | condition | verdict |
|---|---|---|
| **A** | `ΔAUROC ≥ +0.020` | **ADVANCE** |
| **B** | `+0.010 ≤ ΔAUROC < +0.020` | **KEEP** |
| **C** | `0 < ΔAUROC < +0.010` | `BACKUP` if AUPR **or** `d'` also improves, else **STOP** |
| **D** | `ΔAUROC ≤ 0` | **STOP** |

## 9. Composition synergy (P04, P07, P08, P09, P12, P13)

```
Synergy Gain = ΔAUROC(A+B) − max( ΔAUROC(A), ΔAUROC(B) )      using the matching single-module parents
```
`POSITIVE_COMPOSITION` iff `A+B > A` **and** `A+B > B`. A composition may advance on its **total**
ΔAUROC even if one parent alone is ≤ 0.

## 10. Ranking

Primary: **image AUROC**. Tie-breaks in order: AUPR → `d'` → **simpler pipeline** → lower runtime.
No custom robustness metric may be used for ranking.

## 11. E7-1 → E7-2 gate (frozen, mechanical)

Take `ADVANCE` + `KEEP`. If > 4 → keep the **AUROC top 4**. If < 4 → append `BACKUP` in order until
at most 4. **`STOP` never fills a slot.** If 0 candidates → **STOP, E7-2 is not started**.
If 1–4 candidates → E7-2 may start automatically.

## 12. Assumptions the upstream specification left open (each flagged for designer confirmation)

* **A1 — bank subset base.** "仅 normal memory/reference bank：使用冻结 25% subset" does not state the
  base. The **E5-1 bank (10 specimens / 1200 images)** is used as the base ⇒
  **bank = 300 images** = `random.Random(20261009).sample(sorted(1200 bank image paths), 300)`.
  *Why this choice:* (i) it is the literal "25 % subset of the normal memory/reference bank";
  (ii) it fits the stated **30–90 min** budget — with 7 specimens (840 images) the nine
  PatchCore-family fits alone are projected at ≈ 77 min, which would exceed the hard budget, so the
  budget itself is evidence for the smaller base.
  *Alternative reading*: 25 % of the 30 training specimens = **7 specimens / 840 images**. Measured
  runtimes are reported so the designer can re-run with the larger bank if desired.
* **A2 — photometric cache resolution 448.** Required for DINOv2; PatchCore pipelines therefore see
  `resize(transform(img, 448), 256)` instead of E5-1's `transform(resize(img, 256), 256)`.
  Uniform across all transformed pipelines and referenced to the same-protocol P00.
* **A3 — DINOv2 foreground masking disabled.** AnomalyDINO's `masking_default` is per-object and only
  defined for MVTec objects; M²AD is unsupported, so masking is **off** for all DINOv2 pipelines
  (uniform ⇒ fair). No other AnomalyDINO option is changed.
* **A4 — no `τ_val` step** in the PatchCore family (E5-1 computed one; it is unused by every metric).
* **A5 — SoftPatch representation.** The official SoftPatch *mechanism* is applied to the frozen
  PatchCore representation so that P00/P10/P11 share an identical embedding, which is required for a
  meaningful `ΔAUROC` and for composition synergy. SoftPatch's own `resize=256 / imagesize=224` crop
  stack is **not** replicated (it would break the comparison). Declared.
* **A6 — faiss absent** ⇒ exact torch kNN (numerically identical).

## 13. Stage 0 — smoke (mandatory, before any real run)

Every one of the 14 pipelines is smoke-tested on **10 normal + 20 test** images:
shape / finite values / device / dependency / **score direction** (anomaly > normal) / normal-only
bank / **no label leakage**. Budget **< 2 min per pipeline**.
Engineering fixes (device mismatch, import conflict, path/dependency issues) are permitted.
Any fix that would change **method definition, architecture, loss, feature layer or scoring** ⇒
**STOP and report.**

## 14. Runtime, ETA, checkpoint/resume

* Budget: E7-1 **30–60 min** target, **90 min** hard maximum. A pipeline projected at **> 15 min** is
  reported as **`SLOW_CANDIDATE`**; unless it has already passed 50 %, it is paused so that breadth is
  completed first.
* Live line (mandatory, heartbeat every 1–3 min), units and images never mixed:

```
E7-1 [8/14 | 57%]  Pipeline: P08 Retinex+DINOv2-B  Stage: scoring
Elapsed: 00:31:22   ETA: 00:24:10   Expected finish: 18:42
```

* Checkpoint: each pipeline is one **atomic unit** → `results/e7_1/raw/Pxx.json` +
  `results/e7_1/raw/Pxx_per_image.csv`, `results/e7_1/progress.json` updated immediately.
  States: `PENDING / RUNNING / DONE / FAILED / INTERRUPTED / SLOW_CANDIDATE`.
  On restart `DONE` is skipped; `INTERRUPTED` re-runs **only that pipeline**.
* Global hard stop (E7-1+E7-2): **4.5 h**. If exceeded, finish E7-1 breadth first, then E7-2 P1 units.

## 15. E7-2 (only if the §11 gate yields 1–4 candidates)

* Restore the **FULL normal bank** (E5-1's 1200-image bank).
* Views: **120** plus **3 more drawn mechanically** with `VIEW_SEED = 20261009` from the remaining
  available views — **frozen before any candidate result on those views is read**.
* Matrix: top ≤ 4 × 4 views × seed 0. `P00` needs a same-view reference (reused where compatible).
* Metrics: image AUROC primary; per candidate report mean/median AUROC, mean/median ΔAUROC,
  positive views / 4, worst-view AUROC and ΔAUROC; AUPR and `d'` secondary.
* Finalist rule: **Path A** `mean ΔAUROC ≥ +0.015` **and** `positive views ≥ 3/4`;
  **Path B** `mean ΔAUROC ≥ +0.010` **and** `positive views ≥ 3/4` **and** (if A+B) `Synergy Gain > 0`.
  At most **2 finalists**.

## 16. Forbidden this round

No candidate search · no added/removed candidate · no threshold change · no parameter rescue
(no Retinex tuning, no CLAHE search, no `k` search, no SoftPatch weighting search) ·
no E7-3 · no multi-seed · no second category · no paper writing.
`third_party/`, model caches, pretrained weights and large temporary embeddings are **never committed**.

## 17. Deviations / assumptions log (append-only)

| # | time | item | status |
|---|---|---|---|
| A1–A6 | at freeze | §12 assumptions | open for designer confirmation |
| E1 | during Stage 0 | `E7.Bar` lacked `__call__` (pipelines called `bar(i)`) | fixed, engineering only |
| E2 | during Stage 0 | SoftPatch `common.py` imports **faiss** at module level; absent in this environment | **installed `faiss-cpu` 1.15.1** (declared dependency of both upstream repos) — dependency fix permitted by §13 |
| E3 | during Stage 0 | SoftPatch module loader returned the class instead of the module | fixed, engineering only |
| E4 | during Stage 0 | official SoftPatch substitutes `WeightedGreedyCoresetSampler` (`softpatch.py:85`) for whatever sampler `main.py` builds, and that is the class implementing `set_sampling_weight` | used the official class — matches the official code path |
| E5 | during Stage 0 | `_compute_patch_weight` requires `feature_shape`; unset in our adapter | set to the real `(32,32)` embedding grid |
| E6 | during Stage 0 | smoke `score_direction` spanned only **1 specimen per class** (10 images = one specimen's 10 illuminations), making the check meaningless | smoke now takes **one image per distinct specimen**; additionally the direction was re-verified at the real bank scale (`results/e7_1/logs/smoke_direction_diagnostic.json`: dinov2_vits14 AUROC 0.71172, vitb14 0.73158, both with correct direction) ⇒ **no implementation defect** |
| V1 | before any E7-2 result | E7-2 additional views drawn mechanically per §15 with `VIEW_SEED=20261009` from the 11 remaining views | **frozen set = `['120', '090', '240', '330']`** (primary 120 + **090, 240, 330**). Drawn and recorded before any E7-2 metric was read. |
| O1 | before E7-2 run | PatchCore-family models are fitted on the **view-independent** normal bank, so fitting once per pipeline and scoring every view yields identical models to fitting per view | applied as an engineering optimisation (12 fits → 3); no method change |

**No metric definition, tier threshold, gate rule or pipeline definition was changed at any point.**

