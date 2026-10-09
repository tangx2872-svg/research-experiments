# E5-1 — Broad Module Composition Screening (FROZEN PROTOCOL)

> **Status: FROZEN — written BEFORE any E5-1 candidate result was produced or read.**
>
> Frozen at git `5094421` + protocol commit (see `results/e5_1/progress.json` → `protocol_commit`).
> Any deviation must be appended to §12. Nothing in §1–§11 may be edited after a result is read.

```
CURRENT STAGE          = Phase III — Module Composition Screening
METHOD SEARCH          = CLOSED (X6c frozen; E4-X: B2 = STOP, X6c = STOP — no rescue)
E4 DATASET             = M²AD (primary, E4-0C screening winner)
E4-D0 INTEGRITY AUDIT  = PASS
E5-0                   = 17 initial candidates → 8 shortlist
E5-0B                  = 8 shortlist → 4 eligible candidates
```

---

## 1. Purpose (single question)

> Of these 4 mature modules with **different insertion mechanisms**, which — if any —
> is worth advancing to the next round under **real illumination** conditions?

This round is **not** a hunt for the final method. It is a **screening**.

The only frozen question:

> Does any candidate beat **Original** on the frozen M²AD Bird real-illumination
> robustness–defect-preservation trade-off?

---

## 2. Frozen reference (Original) — DO NOT RE-RUN

Original is **reused from E4-X**, GPU units = 0.

| item | frozen value |
|---|---|
| source | `results/e4_d1_smoke/per_image.csv` (8400 rows, all views) filtered to view 120 → `results/e4_x/per_image_original.csv` (700 rows) |
| sha256 (e4_x file) | recorded in `results/e5_1/analysis/reference_sha256.txt` |
| image AUROC | **0.76766** |
| d′ | **1.20676** |
| R_all | **0.67543** |
| R_Good | 0.77887 |
| R_NG | 0.63405 |
| n_good / n_defect | 200 / 500 |

Re-run is permitted **only** if the asset is proven corrupt (SHA256 mismatch). No corruption
was found during Preflight (E5-1P §C PASS).

---

## 3. Frozen experimental scope

| item | frozen value | source |
|---|---|---|
| dataset | M²AD, category **Bird** | E4-0C / E4-D0 |
| official metadata | `data/m2ad/jsons_extract/meta_unsupervised.json` | E4-D0 |
| official split | `normal_ratio=0.6`, `seed=42`, NG → test | M²AD official |
| **frozen view** | **`120`** | `VIEW_SELECTION_SEED=20261008` (`scripts/e4x_common.py`); **禁止更换** |
| seed | **0** | frozen |
| image size | **256 × 256** | `experiment1b_defect_sensitivity.IMAGE_SIZE` |
| normalisation | ImageNet mean/std | frozen |
| backbone | `wide_resnet50_2`, `layer2` + `layer3` | frozen (`experiment5a_model`) |
| embedding | `concat(layer2(512@32×32), upsample(layer3(1024@16×16)))` = **1536 × 32 × 32** (1024 patches/img) | frozen |
| PatchCore | `coreset_sampling_ratio=0.1`, `num_neighbors=9`, `α_l2=α_l3=0` | frozen |
| **bank** | 10 normal train specimens (`sorted(train_specs)[::3][:10]`, pre-registered) → **1200 images** (12 views × 10 illuminations × 10 specimens) | `e4d1_m2ad_loader.bank_and_val_specimens` |
| val (τ only) | 2 normal train specimens (`[1::3][:2]`) → 240 images | idem |
| **scoring set** | test split at **view 120** → **700 images** (200 Good + 500 NG) | matches E4-X exactly |
| illuminations | `01`–`10` | M²AD official |

**Rationale for bank = 1200 (all views):** identical to the bank that produced the frozen Original
scores. Changing it would confound *bank size* with *candidate mechanism*. Frozen.

---

## 4. Candidates (frozen identities, provenance, insertion point)

### C01 — PIAD-Retinex  · insertion = **input photometric**

| field | value |
|---|---|
| official repo | `https://github.com/Kaichen-Yang/piad_baseline` @ `67b816b2a484317baa5b016cdce5dd52ec806ab1` |
| files used | `Retinex/RL_separate.py` (class `Separate`), `Retinex/network/*`, `Retinex/utils.py` |
| weights | `Retinex/ckpt/{init_low,unfolding,init_high,L_adjust}.pth` (committed in repo, 5.5 MB total) |
| **how PIAD itself calls it** | `pose_estimation.py:31` → `from Retinex.RL_separate import Separate` |
| **output used** | `Separate.forward()` → `R` (**reflectance / illumination-invariant**) — this is PIAD's own in-pipeline output; `Separate.run()` saves `Reflection` only. `I_enhance` (from `test.py`) is **NOT** used. |
| vendor license | `Retinex/LICENSE` = **MIT License, Copyright (c) 2022 AndersonYong** (PIAD vendors URetinex-Net) |
| pipeline | image → PIL(RGB) → `TF.to_tensor` → `Separate` on GPU → save reflectance PNG (8-bit) → **same frozen PatchCore pipeline** (new bank 1200, refit, score 700) |
| forbidden | CLAHE / gamma / histogram matching / extra denoise / custom enhancement / choosing a different URetinex repo |

`PROVENANCE = VERIFIED_OFFICIAL`. **No approximation, no re-implementation of the mechanism.**

### C06 — SimpleNet  · insertion = **post-concat** (supporting component)

| field | value |
|---|---|
| official repo | `https://github.com/DonaldRR/SimpleNet` @ `351a2b8d4e8cfc944dbccbf9bc6ceda930c6f26b` |
| files used | `simplenet.py` — `Projection`, `Discriminator`, `SimpleNet._train_discriminator`, `SimpleNet._predict` |
| license | `LICENSE` = **MIT License, Copyright (c) 2023 DonaldRR** |
| official hyper-params (`run.sh`) | `patchsize=3`, `pretrain_embed_dimension=1536`, `target_embed_dimension=1536`, `meta_epochs=40`, `gan_epochs=4`, `noise_std=0.015`, `dsc_hidden=1024`, `dsc_layers=2`, `dsc_margin=0.5`, `dsc_lr=0.0002`, `pre_proj=1`, `lr=1e-3` |
| frozen embedding | `target_embed_dimension` = **1536** = our frozen embedding dim ⇒ dimension-compatible |
| anomaly synthesis | `fake = true + N(0, (noise_std·1.1^k)·I)`, `mix_noise=1` — **synthetic only** |
| loss | `mean(clip(θ − s_true, 0)) + mean(clip(s_fake + θ, 0))`, `θ = dsc_margin` |
| inference score | `patch = −discriminator(features)`; image score = **max over patches** |
| role | **SUPPORTING_COMPONENT** — no novelty claim |

`ROLE = SUPPORTING_COMPONENT`. Any gain must be reported as *existing-method transfer*, never as a new contribution.

### C07 — ReConPatch  · insertion = **post-concat** (learned representation)

| field | value |
|---|---|
| official repo | **NONE CONFIRMED** (no official implementation located for this project) |
| implementation status | **FROM-PAPER MINIMAL IMPLEMENTATION** (must be disclosed wherever reported) |
| mechanism (frozen) | per-channel **linear modulation** `h = w ⊙ f + b` on frozen patch embeddings, trained by **normal-only contrastive** representation learning over patch pairs |
| constraints | no structures beyond the paper; no extra heads/losses; no architectural invention |
| inference score | kNN distance to the normal modal bank in modulated space (PatchCore-consistent), `num_neighbors=9` |
| **gate** | if the paper body cannot be retrieved to ground the mechanism **before** the run, C07 = `HOLD_UNGROUNDED` and is **not** run |

`IMPLEMENTATION = FROM-PAPER MINIMAL` — must never be written as `official reproduction`.

### C08 — CRAD  · insertion = **memory representation**

| field | value |
|---|---|
| official repo | `https://github.com/tae-mo/CRAD` @ `b5a1c472103479bcda0b09bf40f463afd56e8007` |
| files used | `models/reconstructions/crad.py` (class `CRAD`), `models/reconstructions/mobilenetv3.py` (`MobileBottleneck`), `utils/criterion_helper.py` (`FeatureMSELoss`) |
| license | **NO LICENSE FILE in repository** — recorded as an attribution/legal caveat (§11) |
| official hyper-params (`experiments/config.yaml`) | `local_resol=8`, `global_resol=4`, `ch_exp=864`, `layers=11`, `mse_lamb=0.3`, `cos_lamb=0.7`, `mse_coef=0.05`, `noise_std=0.05`; `max_epoch=50`; `grid_lr=0.1`; `net_lr=0.001`; `StepLR(step=40, gamma=0.1)`; `clip_max_norm=0.1`; loss = `FeatureMSELoss` (MSE) |
| adaptation | **ADAPTED FROM CRAD** — the module is applied to *our* frozen `1536×32×32` embedding instead of CRAD's own `efficientnet_b4 + MFCN` backbone, because the comparison is against PatchCore's discrete memory bank **in the same feature space**. |
| engineering adaptations (declared) | `ch = 1536` (ours) instead of `432`; `feature_size = (32, 32)`; `ch_exp` scaled `2×` → `3072` to preserve the official `ch_exp/ch = 2` ratio; `MobileBottleneck(ch, ch, …)` generalised from the hard-coded `432` |
| forbidden | claiming this is a full CRAD reproduction; changing `local_resol`/`global_resol`/`mse_lamb`/`cos_lamb`/`mse_coef`/`noise_std`/`layers` |

`IMPLEMENTATION = ADAPTED FROM CRAD (official code, adapted feature space)`.

### Not in this round

C10 is **not** added. C02 / C04 / C05 / C09 remain **HOLD**. B2 / X6c remain **STOP** (no rescue).

---

## 5. Shared-asset / recompute matrix

| asset | Original | C01 | C06 | C07 | C08 |
|---|---|---|---|---|---|
| frozen scores (view 120, 700) | REUSED | recomputed | recomputed | recomputed | recomputed |
| image list / view / split / masks / labels | REUSED | REUSED | REUSED | REUSED | REUSED |
| backbone `wide_resnet50_2` embedding (bank 1200 + test 700) | — | REUSED (on R images) | REUSED | REUSED | REUSED |
| PatchCore coreset bank | — | recomputed | n/a | n/a | n/a |
| normal patch-embedding cache | — | — | REUSED | REUSED | REUSED |
| test patch-embedding cache | — | — | REUSED | REUSED | REUSED |

**Rule:** a candidate may only reuse an asset when its mechanism is defined on that same asset.
C01 changes the *pixels* (input insertion) ⇒ its embedding is **recomputed**, never reused.
C06/C07/C08 are defined on the frozen patch embedding ⇒ embedding is **computed once and shared**.

Each candidate's `progress.json` entry records an explicit `REUSED` / `RECOMPUTED` list.

---

## 6. Pre-registered sanity tests (per candidate, BEFORE scoring)

| id | test | abort condition |
|---|---|---|
| S1 | dataset scope = 1200 bank / 240 val / 700 scoring, view 120, illuminations 01–10 | mismatch |
| S2 | official metadata + files present; official split used | missing |
| S3 | Original asset SHA256 matches recorded value | mismatch ⇒ STOP |
| S4 | tensor shape of candidate input/output is the expected frozen shape | mismatch |
| S5 | all candidate outputs finite (scores, params, loss) | non-finite ⇒ STOP |
| S6 | deterministic: same seed ⇒ identical scores on a 20-image replay | mismatch ⇒ STOP |
| S7 | train/test separation: bank ∩ scoring specimens = ∅ | overlap ⇒ **LEAKAGE ⇒ STOP** |
| S8 | normal-only: candidate fits on Good specimens of the **train** split only | violation ⇒ STOP |
| S9 | no anomaly-label leakage: fit never reads `object_anomaly`/`image_anomaly`/masks of the scoring set | violation ⇒ STOP |
| S10 | no threshold / hyper-parameter / view / seed search | any search ⇒ STOP |

**C01 extra:** output size == input size; value range within [0,1] after the official save path;
Retinex is fit-free (no test-set fitting is possible by construction).
**C06 extra:** synthetic anomalies are generated from **train normal features only** — never from
real test anomalies.
**C07 extra:** representation learning uses **normal train patches only**.
**C08 extra:** the learnable grids see **normal train features only**.

---

## 7. Frozen advance criteria (GO / HOLD / STOP) — pre-registered

Let, relative to Original:

```
ΔA  = AUROC_cand − AUROC_orig
Δd  = d'_cand  − d'_orig
ρ   = R_all_cand / R_all_orig        (R_ratio; smaller is better)
```

Semantics fixed in advance:
* **clear defect-preservation degradation** ⇔ `Δd < −0.10`
* **slight preservation loss** ⇔ `−0.10 ≤ Δd < −0.02`
* **no robustness gain** ⇔ `ρ > 0.95`
* `+0.001` / `+0.003` is **never** a meaningful improvement (`ΔA < +0.01` is never called a win)

```
GO-A  (Detection Win)   : ΔA ≥ +0.03  AND  Δd ≥ −0.10  AND  ρ ≤ 1.00
GO-B  (Robustness Win)  : ρ  ≤ 0.90  AND  Δd ≥ −0.10

STOP-s1 (both flat)     : ΔA < +0.01  AND  ρ > 0.95
STOP-s2 (one-axis harm) : (ΔA ≥ +0.03 AND ρ > 1.00)
                        OR (ρ ≤ 0.90 AND Δd < −0.10)
                        OR (ΔA ≥ +0.03 AND Δd < −0.10)      # gain mainly by sacrificing d′
STOP-s3 (indistinct)    : |ΔA| < 0.01 AND |ρ − 1| < 0.05 AND |Δd| < 0.05
STOP-s6 (single-illum.) : ΔA ≥ +0.03 AND ΔA_excl_best_illumination ≤ +0.01

decision = GO   if (GO-A or GO-B)
         = STOP if (s1 or s2 or s3 or s6)
         = HOLD otherwise
```

`HOLD` therefore covers, e.g. `+0.01 ≤ ΔA < +0.03`, and robustness clearly improved with a
slight preservation loss.

Implementation: `scripts/e5_1_metrics.py::decide_e5_1()` — thresholds hard-coded (no run-time editing).

---

## 8. Robustness sanity for any GO candidate (§15)

**Leave-one-illumination-out (L1oO), score-level, no extra fit:**
recompute `ΔA` after removing each illumination condition in turn.

* If the advantage disappears when a single illumination is removed
  (`ΔA` drops below `+0.01` for some removal) ⇒ flag **`SINGLE-CONDITION-DRIVEN`** and the
  candidate may **not** be reported as GO without that flag.

---

## 9. Metrics reported per candidate (§14)

Primary: `AUROC`, `d′`, `R_all`, `R_ratio`. Deltas: `ΔAUROC`, `Δd′`, `ΔR_all`.
Secondary (already frozen in `e4x_common.metric_block`): `R_Good`, `R_NG`, `R_all_median/p25/p75`,
`Rraw_*`, `per_illumination` (AUROC and d′ per illumination).
**No new primary metric may be invented after seeing results.**

Ranking is reported on **four axes**: Detection Rank (AUROC), Robustness Rank (R_ratio),
Preservation Rank (d′), and Overall Decision — never AUROC alone.

---

## 10. Runtime, GPU policy, progress and resume

* GPU: RTX 3090 24 GB. **Default 1 worker.** Expected peak 16–19 GB. No two GPU-heavy fits concurrently.
* Every candidate writes, on completion:
  * `results/e5_1/raw/<candidate>.json` (metrics + artifacts + git HEAD + seed + protocol hash)
  * `results/e5_1/raw/<candidate>_per_image.csv` (specimen, kind, illumination, score)
  * and updates `results/e5_1/progress.json`
* Statuses: `PENDING / RUNNING / DONE / FAILED / HOLD`. On restart, `DONE` is skipped, `FAILED` is
  reported without infinite retry, `RUNNING` with no live process ⇒ `INTERRUPTED` (restartable).
* Live progress line (mandatory, every ≤ 3 min heartbeat), ETA computed from **completed units of the
  same denominator** (never mixing unit counts with image counts):

```
E5-1 [2/4 | 50%]  Candidate: C08 CRAD  Stage: fitting 63%
Elapsed: 00:54:21   ETA: 00:47:10   Expected finish: 14:32
VRAM alloc=15120.0MB reserved=18258.0MB  RAM=2567.1MB
```

* Long runs are launched detached (`setsid nohup`); session name, PID, log path and progress path
  are reported. Output is not bound to the SSH/tool session.
* Order: **C01 → C08 → C07 → C06**. A successful earlier candidate never excuses skipping a later one.

---

## 11. Leakage guard (DO / DO NOT)

**DO**
* reuse the official split and official metadata;
* use only Good **train** specimens to fit anything;
* adapt resolution / loader / memory / batch size (engineering only);
* report negative and failed results.

**DO NOT**
* tune X6c/B2/C2/C6 or rescue them;
* use scoring-set labels, masks or scores to fit, select, or calibrate anything;
* pick categories / views / seeds from results;
* change `PASS` / `catastrophic` / `GO` thresholds after seeing results;
* drop a category (or illumination) because it looks bad;
* re-use the scoring set to choose α / λ / weights;
* present an adapted implementation as an official reproduction;
* call HOLD a success; call `+0.001` an improvement;
* add C10, restart C04/C05, or auto-compose candidates (`C01×C06`, …) in this round.

**Attribution / licensing caveat (must appear in the final report):**
C01 code+weights are MIT (© AndersonYong, vendored by PIAD). C06 is MIT (© DonaldRR).
**C08's repository declares no license** — used here for non-commercial academic comparison only,
with the code unmodified except for the declared feature-space adaptation. C07 has no official
implementation and is a disclosed from-paper minimal implementation.

`third_party/` is **git-ignored**: external sources are never committed.

---

## 12. Deviations log (append-only, empty at freeze)

| # | time | deviation | reason |
|---|---|---|---|
| D1 | 2026-10-09, pre-result | **§4 C08 `ch_exp` correction.** §4 stated "`ch_exp` scaled 2× → 3072 to preserve the official `ch_exp/ch = 2` ratio". Both the ratio and the value were **mis-stated at freeze time**. Reading `third_party/CRAD/models/reconstructions/crad.py` + `utils/misc_helper.py:212` + `models/necks/mfcn.py` shows the official **ch = 216** (`MobileBottleneck` input is hard-coded `432 == 2·ch`) and **`ch_exp = 864 == 4·ch`**. | Corrected decision: keep the **official absolute `ch_exp = 864`** and generalise only the hard-coded `432` → `2·ch`. Rationale: ratio-preserving `ch_exp = 4·1536 = 6144` yields ≈4.15×10⁸ parameters in the 11 `MobileBottleneck` blocks (weights + Adam states + activations ≫ 24 GB on the RTX 3090), i.e. it would make the candidate unrunnable. `ch_exp` is an internal width; all **mechanism-defining** quantities (`local_resol=8`, `global_resol=4`, `layers=11`, `mse_lamb=0.3`, `cos_lamb=0.7`, `mse_coef=0.05`, `noise_std=0.05`, loss, optimiser, schedule, epochs) remain exactly official. Logged **before any E5-1 candidate result was produced or read**. |
| D2 | 2026-10-09, pre-result | **§4 C07 optimiser.** `AdamP` (paper) is not installed in this environment (`ModuleNotFoundError: adamp`) and its official package is unavailable offline. | Substitute `torch.optim.Adam` with the same `lr=1e-5`, `weight_decay=1e-2` and cosine annealing. Disclosed as a from-paper-implementation substitution. |
| D3 | 2026-10-09, pre-result | **§4 C07 / CRAD inference:** `faiss` is not installed; ReConPatch's paper uses faiss for kNN. | Exact nearest-neighbour search in PyTorch (brute force) — numerically identical, no method change. |
| D4 | 2026-10-09, pre-result | **§4 C01 loading compatibility.** PyTorch 2.8 defaults `torch.load(weights_only=True)`; PIAD's official `ckpt/unfolding.pth` pickles an `argparse.Namespace` as `opts`. | `torch.serialization.add_safe_globals([argparse.Namespace])` before loading (official API). No PIAD code edited; Retinex mechanism untouched. |
| D5 | 2026-10-09, pre-result | **§4/§5 C01 resolution.** `Separate.forward` is invoked on `resize(img, 256)` — the same tensor the frozen Original pipeline feeds the backbone — rather than on the native 3648×5472/1024 source. | Keeps the C01 comparison at the frozen input resolution (declared engineering adaptation; the Retinex mechanism and its weights are unchanged, and the official uint8 save path is used). |

---

## 13. Exit rules for the round (§17)

| outcome | next |
|---|---|
| **≥ 2 GO** | `E5-2 — Composition Screening` over only the actual GO set (e.g. `C01×C06`, `C01×C08`, `C01×C07`) |
| **exactly 1 GO** | `Winner Confirmation` — multi-view / illumination confirmation. **No composition.** |
| **0 GO** | STOP the Module Composition branch; route re-evaluation. **No** `C01'` / `C01++` / `C07 tuned` / `C08 rescue`. |

After E5-1: **STOP.** No automatic E5-2, no auto-composition, no extra seeds/views, no tuning —
human review required first.
