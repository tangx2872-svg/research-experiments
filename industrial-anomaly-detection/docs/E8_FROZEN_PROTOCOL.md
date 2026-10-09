# E8 FROZEN PROTOCOL — Cross-Category CLAHE × PatchCore Stability Kill Test

```text
📍 PAPER STAGE
Phase III — Final Candidate Stability Validation

🎯 PAPER CLAIM UNDER TEST
A simple training-free local contrast enhancement (CLAHE) can improve the robustness of
PatchCore under real illumination variations while largely preserving anomaly discriminability.

🧪 WHY E8 IS NECESSARY
E7 showed that CLAHE-medium improved mean image AUROC by +0.0261 on Bird across 4 frozen
illumination views, with positive gains on 3/4 views.
However, this evidence currently comes from one category/domain.
E8 tests whether this gain survives category and seed changes.

🚦 EXIT
GO    → candidate enters full paper experiments.
HOLD  → effect exists but is category/condition dependent; narrow paper claim.
STOP  → E7 was insufficiently general/stable; stop CLAHE route.
```

> **This document is frozen BEFORE any E8 candidate result was produced or read.**
> E8 is a **validation experiment, not an optimisation experiment.**
> Nothing in §2–§7 may change after results exist. Any change must be logged in §12.

---

## 0. Question

> Does the E7 CLAHE → PatchCore candidate hold up **cross-category and cross-seed**, or was
> `Bird` a single-category accident?

This round **must not** be reported as proving "universal illumination robustness".

---

## 1. Restoration of project state (read-only audit, done before freeze)

| item | value |
|---|---|
| branch / HEAD | `research/industrial-anomaly` @ `1caecfc` (== `origin/research/industrial-anomaly`) |
| worktree | only 3 known non-research files (`monitor/exp_monitor.py`, 2 probe scripts) — not committed |
| GPU at freeze | RTX 3090 24 GB, idle (1 MiB / 0 %) |
| CPU / RAM | 14 vCPU / 755 GB |
| disk at freeze | 12 GB free → **26 GB after P0.5 cleanup** (§3.1) |
| E7 protocol | `docs/E7_1_FROZEN_PROTOCOL.md`, commit `af56dd3` |
| E7 results | `results/e7_1/` (300-image bank, view 120), `results/e7_2/` (1200-image bank, 4 views) |

### 1.1 Candidate definitions — mechanically recovered from E7 code (NOT from memory)

Read from `scripts/e7_1_runner.py::SPEC` and `scripts/e7_common.py`:

```text
P00 = {"family": "patchcore", "image": None}           # Original PatchCore
P02 = {"family": "patchcore", "image": "clahe_mild"}   # CLAHE mild  -> PatchCore
P03 = {"family": "patchcore", "image": "clahe_medium"} # CLAHE medium-> PatchCore
```

| frozen constant | value | source |
|---|---|---|
| PatchCore backbone | `wide_resnet50_2` (ImageNet pre-trained) | `scripts/e7_pipelines.py:41` |
| PatchCore layers | `["layer2", "layer3"]` | same |
| coreset sampling ratio | `0.1` | same |
| `num_neighbors` | `9` | same |
| `alpha_l2`, `alpha_l3` | `0.0`, `0.0` | same |
| model input size | `256` (`IMAGE_SIZE`) | `scripts/experiment1b_defect_sensitivity.py:85` |
| CLAHE colour space | Lab **L channel only** | `scripts/e7_common.py:173-179` |
| CLAHE `tileGridSize` | `(8, 8)` | `scripts/e7_common.py:37` |
| CLAHE `clipLimit` mild / medium | `1.5` / `2.0` | `scripts/e7_common.py:36` |
| CLAHE cache resolution | `448` (`CACHE_RES`) | `scripts/e7_common.py:34` |
| bank subset seed | `20261009` (`BANK_SUBSET_SEED`) | `scripts/e7_common.py:31` |
| bank subset fraction | `0.25` | `scripts/e7_common.py:32` |

**No E7 parameter is re-derived from memory.** P02 = CLAHE mild. P03 = CLAHE medium.
P03 is the **primary** candidate; P02 is the parameter-strength sanity / secondary candidate;
P00 is the **only** primary baseline.

### 1.2 Discrepancy report (code vs README)

None found. `docs/E7_1_FROZEN_PROTOCOL.md` / `results/e7_1/README.md` agree with the code for
`clipLimit ∈ {1.5, 2.0}`, `CACHE_RES = 448`, `tileGridSize = (8,8)`, Lab-L channel, coreset `0.1`,
`k = 9`, `layer2+layer3`, input `256`.

---

## 2. Frozen design (uniform across all categories)

### 2.1 Categories — mechanically selected, never by performance

All 10 M²AD classes share an **identical view set** (`000…330`, 12 views) and 10 illuminations,
so **no view mapping is needed**. Local-only inventory (from `meta_unsupervised.json`):

| category | train specimens | test Good | test NG | illum | test images @ 4 views | masks |
|---|---:|---:|---:|---:|---:|---:|
| **Bird** | 30 | 20 | 50 | 10 | 2800 | 4550 |
| **Car** | 30 | 20 | 50 | 10 | 2800 | 4850 |
| **Cube** | 30 | 20 | 50 | 10 | 2800 | 5530 |

**Selection rule (recorded before any download):** E4-D0 recorded the official class order
(dictionary order) `Bird, Car, Cube, Dice, Doll, Holder, Motor, Ring, Teapot, Tube` and chose the
first class with complete Good + NG + GT + metadata (Bird). E8 extends this **the same way**:
take the **next two** classes in that frozen official order that satisfy the same completeness
rule ⇒ **Car, Cube**. No performance information was consulted. Bird is retained for regression.

### 2.2 Methods

`P00`, `P02`, `P03` only. **No P04/P05 or any new pipeline may be added in E8.**

### 2.3 Seeds

`SEEDS = [0, 1, 2]`.

The seed controls the **bank draw** and the coreset stochastic component (torch/numpy seeding).
Seed must **not** change view, test set, preprocessing parameters or pipeline definition.

Bank for `(category, seed s)` — uniform for **every** category:

```text
pool      = E5-1/E7-1 frozen bank source of that category
            = 10 normal train specimens (stride rule) x 12 views x 10 illuminations = 1200 images
positions = sorted(random.Random(20261009 + s).sample(range(1200), 300))
bank      = [pool[i] for i in positions]        # 300 images
```

* `s = 0` ⇒ `random.Random(20261009)` ⇒ **byte-identical to the E7-1 frozen Bird bank** (this is the
  regression anchor of Stage B).
* The same rule is applied to Car and Cube from *their own* 1200-image pool.

**Bank size is 300 for every category.** `Bird = 1200` combined with `Car/Cube = 300` is
**forbidden** (§2.4 rationale).

### 2.4 Why bank = 300 (rationale, frozen pre-result)

| bank size | measured fit cost (E7) | 27 fits × 3 cats × 3 pipelines × 3 seeds |
|---|---|---|
| 300 (E7-1) | 106–131 s | ≈ 59 min ✔ |
| 1200 (E7-2) | 785–811 s | ≈ 5.9 h ✘ (would exceed the hard stop on fits alone) |

300 is also the bank size implied by the E7-1 rule (`BANK_SUBSET_FRACTION = 0.25`) and by this
round's 108-cell target. **Consequence (must be stated in every report):** the Stage B reference is
**E7-1** (300-bank, view 120), *not* E7-2 (1200-bank, 4 views); the two are not directly comparable.

### 2.5 Views

`VIEWS = ["120", "090", "240", "330"]` — the E7-2 frozen 4-view protocol, applied unchanged to
**all** categories (view IDs are common to all classes). Never re-selected after seeing results.

### 2.6 Metrics

| metric | role |
|---|---|
| image AUROC | **primary decision metric** |
| image AUPR | secondary (preservation) |
| `d′` (pooled-SD) | secondary (separation / preservation) |
| pixel AUROC, AUPRO | secondary, when the frozen pipeline supports it (E7-1 does) |
| `ΔX = X(method) − X(P00)` | computed **within the same category × seed × view cell** |

No new primary metric may be invented. Scoring protocol (`predict_one` / anomalib map→image score)
is unchanged from E7.

### 2.7 Cache design (fidelity claims)

Per image, stored once per category:

| cache | content | produce | consumed by |
|---|---|---|---|
| `L0_256` | `resize(original_1024, 256, BICUBIC, antialias)` as PNG | exact frozen resize | P00 (→ resize 256→256 is a no-op) |
| `L0_448` | `resize(original_1024, 448, BICUBIC, antialias)` as **lossless WebP** | exact frozen `_load_rgb448` | CLAHE source |
| `clahe_mild` | `CLAHE(L0_448, clip=1.5, tile=(8,8))` PNG @448 | E7 code path | P02 |
| `clahe_medium` | `CLAHE(L0_448, clip=2.0, tile=(8,8))` PNG @448 | E7 code path | P03 |

* **P02/P03 are bit-identical to E7** (identical source tensor, identical CLAHE call, lossless storage).
* **P00 differs from E7 in exactly one respect**: the frozen 1024→256 resize result is quantised to
  uint8 PNG and re-read, whereas E7 resized the float tensor in memory. This is a ≤0.5/255
  perturbation of the model input. **It is measured explicitly in Stage B** and reported as a
  deviation if non-negligible (§12 D2).
* Lossless storage verified: 1024 WebP-lossless round-trip is bit-exact (measured, 3/3 images).

### 2.8 Data acquisition

Non-Bird categories are **not** on local disk and their archives are 14.9–17.6 GB each while the
disk is 50 GB total ⇒ a full download is impossible. Acquisition is therefore
**streaming selective extraction** of the official Hugging Face mirror archives:

```text
host      : https://hf-mirror.com/datasets/ChengYuQi99/M2AD/resolve/main/<Category>.zip
member    : <Category>/<Good|NG>/<specimen>/A<view>_I<illum>.png
method    : sequential streaming read of the remote ZIP local file headers
            (members are DEFLATE with general-purpose flag 0 => sizes are in the local header,
             so no central-directory random access is required),
            decompress in flight, keep only members in the frozen need-list,
            discard every other member. disk cost = kept members only.
```

Need-list per category = bank pool (1200) ∪ test @ ≥1 frozen view. Verified reachable:
`hf-mirror.com` HTTP 200, HTTP Range → 206, ZIP member header parse OK.

---

## 3. Stages

### 3.1 P0.5 — disk safety audit (mandatory, done)

`du`-classified `/root/autodl-tmp`; only clearly recoverable trash was cleaned. Result:

| location | size | class | action |
|---|---:|---|---|
| `.Trash-0/files/bird.zip*` | 14.44 GB | trashed download of an already-extracted archive (SHA256 recorded, re-downloadable) | **truncated** → freed |
| `mvtec_ad_5cats.tar.gz` | 1.52 GB | unknown file outside the project | **untouched** |
| `results/experiment_1j`, `experiment_13p` | 3.2 GB | historical research results | **untouched** |
| `~/.cache/pip` | 198 MB | re-downloadable cache | untouched |
| `~/.cache/torch`, `~/.cache/huggingface` | 0.77 GB | backbone weights — still needed | untouched |

**Free space 12 GB → 26 GB** (target ≥18 GB met). Note: `.Trash-0` intercepts `rm` (renames
instead of unlinking), so the archive was **truncated** to release the blocks.

**Running guard:** the E8 driver checks free space before starting each new category. If free space
would drop below **3 GB**, it stops adding caches, saves state, and exits cleanly.

### 3.2 Stage A — data preparation

For each new category: stream → decode → write the 4 caches of §2.7 → per-category integrity gate.
The gate is **hard**: a failing category is not silently skipped.

```text
A1 file count      : cached image count == expected need-list size
A2 decode          : every cached file opens and has the expected shape/dtype
A3 mask alignment  : test records with mask_path resolve to existing GT files
A4 illumination    : cached test records cover exactly the 4 frozen views x 10 illuminations
A5 label alignment : Good/NG counts match the official metadata
```

### 3.3 Stage B — Bird regression gate (mandatory before the full matrix)

Run `Bird × seed 0 × {P00,P02,P03} × 4 views` and compare against the **E7-1** reference:

| method | E7-1 view 120 AUROC | E7-1 `d′` |
|---|---:|---:|
| P00 | 0.78089 | 1.22579 |
| P02 | 0.79458 | 1.21570 |
| P03 | 0.79591 | 1.21751 |

E7-2 (1200-bank) is recorded as a **secondary, non-comparable** reference:

| view | 120 | 090 | 240 | 330 | mean |
|---|---:|---:|---:|---:|---:|
| P00 | 0.76963 | 0.82964 | 0.76890 | 0.75465 | 0.78071 |
| P02 | 0.79543 | 0.81060 | 0.78714 | 0.79276 | 0.79648 |
| P03 | 0.81062 | 0.81536 | 0.80133 | 0.79992 | 0.80681 |

Verdict rule:

```text
PASS    : view-120 AUROC within +/-0.010 of E7-1 for all three methods
          AND method ranking P03 >= P02 >= P00 preserved
WARNING : max |deviation| <= 0.025, ranking preserved  -> record, continue to Stage C
FAIL    : |deviation| > 0.025, or the ranking flips     -> STOP E8, debug the pipeline
```

FAIL means the whole night is stopped: a wrong pipeline must not be run overnight.

### 3.4 Stage C — E8 core matrix

```text
3 categories (Bird, Car, Cube) x 3 methods (P00, P02, P03) x 3 seeds (0,1,2) x 4 views (frozen)
= 108 evaluation cells, 27 fits
```

**A model is fit once per `(category, method, seed)` and then scored on all 4 views.**
Re-fitting per view is forbidden. Stage B cells are reused, not recomputed.

Per cell recorded: `category, method, seed, view, bank_size, test_size, image_auroc, image_aupr,
d_prime, pixel_auroc, aupro, runtime_s, peak_vram_mb, status`.

### 3.5 E8 core verdict (fixed before results)

Per method (`P02`, `P03`), relative to `P00` **within the same category × seed × view**:

* category level: mean / median / std of `ΔAUROC`, seed consistency, view consistency
* cross-category: category win rate, seed win rate, worst-category `Δ`, worst-seed `Δ`
* preservation: `ΔAUPR`, `Δd′`, pixel `ΔAUROC` / `ΔAUPRO`

```text
GO    : >= 2/3 categories positive AND seed direction consistent AND robustness improved
        AND no obvious preservation collapse
HOLD  : small overall positive but category/view dependent
STOP  : <= 0 overall, or gains driven by a single category/seed, or multiple new categories
        clearly degrade
```

`R_all`-style "squashed score" robustness is **not** used as a primary criterion in E8:
E5-1 showed it can be trivially minimised while destroying detection. Robustness is read
**jointly with** `ΔAUROC` / `Δd′`.

---

## 4. Extensions (automatic, only after Stage C is complete)

Order of priority. Each is entered only if the remaining time to the hard stop allows it.

### Extension A — additional categories (highest priority)

Extend breadth using a **pre-frozen mechanical order**: `Dice → Doll → Holder`.
Never chosen by performance. Strategy per category: `stream → process → evaluate → release
recoverable cache`. Target **5 categories**, maximum **6** (Bird, Car, Cube + Dice, Doll, Holder).

### Extension B — winner confirmation

Only if `E8 ∈ {GO, HOLD}`. Keep `P00` + the best CLAHE variant only, extend to **seeds 0–4**
(5 seeds). Purpose: decide whether the small gain is noise. Failed methods get **no** extra seeds.

### Extension C — CLAHE parameter neighbourhood

Only if (i) E8 shows an overall positive CLAHE effect, and (ii) Extension A/B are complete or
resources remain. Frozen grid, no search:

```text
clipLimit in {1.25, 1.50, 1.75, 2.00, 2.25}   (tileGridSize stays (8,8))
```

Bayesian / Optuna / large-scale hyper-parameter search is **forbidden**. Purpose: decide whether
the gain occupies a **wide stable region** (`robust parameter region`) or is a single lucky point
(`fragile`).

---

## 5. Instrumentation (mandatory)

* **Dynamic progress**: `[completed/total] pct | elapsed | dynamic ETA | current unit`, ETA from the
  real mean/EMA wall-clock of completed units. Per-stage announcements with expected finish time.
* **Heartbeat** every 10–15 min: completed, elapsed, ETA, GPU util, VRAM, disk free, failures.
* **Crash-safe**: one JSON per `(category, method, seed)` written atomically (`.tmp` → `rename`)
  **before** the next unit starts; per-category flush of CSV/JSON/checkpoint/manifest on completion.
  Restart scans completed units and runs **only the missing ones**; already-`OK` units are skipped.
* **Failure records**: `category, method, seed, view, exception, timestamp` appended to
  `results/e8/failures.jsonl`; a failure does not abort unrelated units.

Files: `results/e8/run_manifest.json`, `results/e8/progress.json`, `results/e8/failures.jsonl`,
`results/e8/raw/<method>_<category>_seed<k>.json`, `results/e8/logs/`.

## 6. Time control

```text
start          : 2026-10-09 ~21:45
stage estimate : Stage A ~50 min | Stage B ~12 min | Stage C ~105 min | analysis ~10 min
                 => core total ~3 h
budget         : ~12 h (overnight)
hard deadline  : 2026-10-10 09:30 — no new large fit may start after this
wrap-up target : 2026-10-10 10:00 (analysis, figures, README, commit)
```

Partial Extension results are kept; nothing is discarded because the deadline arrived.

## 7. Deliverables

* `results/e8/README.md` (§19 checklist of the E8 brief: question … next experiment)
* figures: per-category P00/P02/P03; per-category Δ; per-view robustness; seed consistency;
  win/loss heatmap if Extension A ran
* updated root `README.md` and `PROJECT_STATE.md`
* commits: `experiment: freeze E8 CLAHE stability protocol` (this file) and
  `experiment: complete E8 CLAHE stability validation` (results)
* push to `origin/research/industrial-anomaly` only — **never** `main`.
  Never committed: model weights, caches, extracted datasets, `third_party/`.

## 8. Prohibitions

Searching new modules; new backbones; DINOv2 / SoftPatch / Retinex rescue; tuning CLAHE to find a
better point; choosing views or categories after seeing results; changing backbone/layers/scoring/
coreset ratio/input resolution/metric definitions; re-tuning after a category fails; reporting only
successful categories; adding favourable conditions to reach GO; deleting negative results;
infinite parameter rescue of a failed CLAHE variant; filling GPU time with already-done experiments.

**Failure is failure.**

---

## 9. Live log

| # | time | deviation | reason |
|---|---|---|---|
| — | — | *(none at freeze)* | — |
