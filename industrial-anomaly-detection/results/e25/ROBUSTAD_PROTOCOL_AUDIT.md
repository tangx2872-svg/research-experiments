# RobustAD — Protocol Audit (E25)

**Date**: 2026-10-18
**Dataset**: `AmazonScience/RobustAD` (HF dataset, CC-BY-4.0), accessed via `https://hf-mirror.com` (huggingface.co is unreachable from this server).
**Scope**: image-level anomaly detection only (localization masks exist but are NOT used).
**Status**: audit performed BEFORE any model run; no result in this file depends on model output.

---

## 0. Facts collected (source of truth)

| Item | Evidence |
|---|---|
| Dataset size (official) | **16.27 GB** total; PCB **6.18 GB** (1747 files), MetalParts **0.68 GB** (1912 files), PiledBags **9.41 GB** (1718 files) |
| Structure | `<Category>/<category>_data_dir_train/` + `..._data_dir_test0..N/`, each containing `normal/`, `anomaly/` and `metadata.jsonl` |
| Metadata schema | `{"file_name": "normal/<name>.jpg", "label": 0|1, "mask": null|"<path>"}` — mask only for anomalies |
| Label consistency | For **every** split: `normal/* → label 0`, `anomaly/* → label 1`; **0 inconsistencies** across all 15 metadata files |
| README / paper | Dataset card says only "multiple target domains with different shifts"; the paper is "COMING SOON" → **no official domain-name table exists** |

**Consequence**: domain semantics below are taken from **the official file names** (which encode the controlled condition, e.g. `Test_PCB_Lighting_Morning_Normal_35.jpg`), never from directory indices. Where a name is not self-explanatory, the label used in this project is exactly the literal token in the filename.

---

## 1. train 是哪个 source domain？

`<category>_data_dir_train` = **source domain**, condition token `resized_None_None` (PCB, `Set1`) / `None_None` (MetalParts, `Live_*`).
- PCB train: 255 = **180 normal + 75 anomaly**
- MetalParts train: 231 = **180 normal + 51 anomaly**
- PiledBags train: 257 files (classification only; not used this round)

→ The adapter / bank may use **only `label == 0`**. The 75/51 training anomalies are never used for training or model selection.

## 2. test0…testN 分别对应什么 target condition？

### PCB (6 test splits)

| split | condition token in official file names | n (normal/anomaly) | category |
|---|---|---|---|
| test0 | `resized_None_None` (same condition family as train) | 120 / 75 | **source test (no shift)** |
| test1 | `Lighting_{Morning, Evening, Afternoon}` | 120 / 90 | **illumination** |
| test2 | `White Balancing_{Evening, Morning, Afternoon} + AWB` | 120 / 60 | **lighting-related (white balance / AWB)** |
| test3 | `Rotation_{−5%, +5%}` | 80 / 60 | pose / viewpoint |
| test4 | `Position_{−10%, +10%}` | 80 / 60 | pose / translation |
| test5 | `Shadow_{Object Alternate Location, Light Alternate Position}` | 80 / 60 | **lighting-related (shadow geometry)** |

### MetalParts (7 test splits)

| split | condition token | n (normal/anomaly) | category |
|---|---|---|---|
| test0 | `None_None` (same condition family as train) | 120 / 51 | **source test** — ⚠️ **contaminated, see §6** |
| test1 | `Lighting_Ambient` | 120 / 80 | **illumination** |
| test2 | `Position_+5%` | 120 / 80 | pose / translation |
| test3 | `Rotation_+5` | 120 / 80 | pose / rotation |
| test4 | `Scale_+5%` | 120 / 80 | scale |
| test5 | `Background_Background` (background #1) | 60 / 40 | background |
| test6 | `Background_Background` (background #2) | 60 / 40 | background |

### PiledBags — NOT RUN
9.41 GB would exceed the remaining disk after PCB+MetalParts (6.86 GB of 13 GB free). Classification-only subset, not required by the primary protocol.

## 3. 哪些 target 明确属于 illumination / exposure / white balance / lighting-related？

- **PCB test1** (`Lighting_*`) — illumination.
- **PCB test2** (`White Balancing_* + AWB`) — white balance / auto-white-balance (lighting-related).
- **PCB test5** (`Shadow_*`) — shadow geometry (lighting-related).
- **MetalParts test1** (`Lighting_Ambient`) — illumination.
- Exposure: **no explicit exposure token** exists in the dataset. Nothing is classified as exposure.
- No domain is claimed to be "illumination-only" — the shifts are per-condition captures, not controlled single-variable renders.

## 4. 哪些属于 pose / background / blur / other？

- **Pose/view**: PCB test3 (rotation), PCB test4 (position), MetalParts test2 (position), MetalParts test3 (rotation).
- **Scale**: MetalParts test4.
- **Background**: MetalParts test5, test6.
- **Blur**: no explicit blur token; **not claimed**.
- **Source reference**: PCB test0 (clean, disjoint from train), MetalParts test0 (contaminated).

## 5. normal/anomaly 标签如何读取？

Read **only** `metadata.jsonl` (`label` 0/1) inside each split directory. Directory names happen to agree 100% with labels (verified), but the code reads metadata, not directory names, so the label source is unambiguous. `mask` is ignored (image-level task).

## 6. 是否存在信息泄漏风险？

**YES — one finding, MetalParts only.**

- PCB: train ↔ test0…test5 **name overlap = 0**; test ↔ test overlap = 0. Clean.
- MetalParts: `train` ∩ `test0` = **145 identical file names** (104 of them are `label==0`, 41 `label==1`); no overlap with test1–test6; no cross-test overlap.
  → MetalParts **test0 is not a held-out split** (its images are physically the same files as train). It is therefore **excluded from all target aggregates and from the source→target degradation metric**, and is reported only with `valid=0`.
  → Because the overlap is confined to test0, training/bank construction on train still leaks **nothing** into test1–test6. No bank exclusion is needed for the reported targets.

## 7. 是否可以严格执行 normal source train → target normal+anomaly test？

**Yes.**
- Bank/adapter: source train, `label == 0` only, no target data, no target labels, no target-based selection.
- Test: every target split evaluated with its own normal + anomaly images, once, with the frozen model.
- Model selection: none on RobustAD (strength/type frozen in E24 before E25 starts).

---

## 8. Protocol actually executed (frozen before results)

1. **Per category** (PCB, MetalParts): one model per category, trained only on source-train normals.
2. **Bank** = all source-train normals; **adapter training subset** = first 120, **validation** = last 60 (frozen `TRAIN_N=240/VAL_N=60` cannot be met with 180 source normals; the reduction is a bank-size constraint, **not** a tuning decision — no hyperparameter is changed).
3. **Frozen R1-ILA**: DINOv2-B final-layer patch tokens (1024×768), 448×448 bicubic resize, L2, linear residual adapter (768×768, zero-init), `L = L_cons + 0.1·L_preserve`, AdamW lr 1e-3, ≤300 steps patience 30, batch 8; scoring = L2 + 1-NN-max. **The single legal source of every method hyperparameter is `results/e17/final_method.json` (see §10). BETA, augmentation type and augmentation strength are NOT re-selected on RobustAD.**
4. **B0** = frozen DINOv2-B 1-NN-max on the same bank (no adapter).
5. **Metrics**: AUROC, AP per domain; mean/worst target AUROC; ΔAUROC = B1 − B0 (same images, same labels).
6. **Source→target degradation**: AUROC(test0) − AUROC(target) computed **for PCB only** (MetalParts test0 invalid). No metric name is redefined; the dataset's own "Average Relative Drop" definition is not published (paper "COMING SOON"), so it is **not** computed.
7. **PiledBags**: not run. The 9.41 GB archive is **not downloaded** and the volume has **6.1 GB free** (measured 2026-10-18), so it cannot be acquired in this round; documented, never imputed.

---

## 9. Naming rules for the E25 report

- Domains are named by their **literal condition tokens**; groups are `illumination-related` (PCB test1/test2/test5, MP test1), `pose` (PCB test3/test4, MP test2/test3), `scale` (MP test4), `background` (MP test5/test6), `source reference` (PCB test0; MP test0 invalid).
- The overall statement is **"real-world domain shift"**, never "illumination-only shift".

---

## 10. Frozen-configuration provenance (E25 amendment, 2026-10-18, before any GPU result)

**Decision (research-lead ruling, 2026-10-18)**: E25 does **not** wait for E24. The **only** legal source of method hyperparameters for E25 is the frozen artifact **`results/e17/final_method.json`**. `results/e24/` is neither created nor referenced, and no placeholder is fabricated.

Frozen configuration actually executed:

| Item | Value |
|---|---|
| Config source | `results/e17/final_method.json` (verdict `FREEZE_CURRENT_R1`) |
| Method | **R1-ILA** (illumination-consistent linear residual adaptation), `z' = z + W z`, W 768×768 zero-init, 589,824 trainable params |
| BETA | **0.1** (`L = L_cons + 0.1 · L_preserve`) |
| Augmentations | **brightness + contrast + gamma** |
| Strength | **{0.85, 1.15}** → 6 photometric variants |
| lr | **1e-3** (AdamW) |
| max_steps | **≤ 300** |
| patience | **30** (evaluated every 5 steps ⇒ ≤ 30-step stall window) |
| batch_size | **8** |

**Scope statement (binding for this round)**: E25 is an **independent external generalization test of an already-frozen method**. It is **NOT** permitted to re-tune, re-select or adjust **BETA**, **augmentation type**, or **augmentation strength** on the basis of **any** RobustAD test / target-domain result — including after seeing PCB or MetalParts numbers. If a result is unfavourable, the result is reported as-is; the method is not modified. Changing any of these values would require a new, separately pre-registered experiment (the planned E24 parameter-sensitivity study), which is out of scope here.

**Seed**: adapter training uses the E13-A convention `seed = 0` (`torch.manual_seed(0)` + `np.random.default_rng(0)`), one model per category. Bank construction is deterministic (all 180 source-train normals, sorted by `file_name`). No seed is chosen after seeing results, and no seed sweep is performed in this round.

**Metric definition (RobustAD has no `detectable` flag)**: plain image-level **AUROC** and **AP** over **all** images of the split, labels read only from `metadata.jsonl`. This differs from the M²AD `official` convention (`image_anomaly` + `detectable != False`) because the field does not exist in RobustAD; the two must not be compared numerically.

**Input geometry (added 2026-10-18, before any MetalParts result)**: §8.3's "448×448 bicubic resize" is applied **literally as a square resize to 448×448**. Rationale and evidence: PCB images are 2048×2048, so the short-side resize inherited from E13-A is already 448×448 and the two paths are **bit-identical** (verified: tensor and DINOv2 feature `max|Δ| = 0.0` on 8 images; PCB results were produced under this rule). MetalParts images are 2681×1500 (16:9) and a short-side resize yields 448×798 → 32×57 = **1824 tokens**, which is incompatible with the frozen 1024-token model/scorer; the square rule keeps 1024 tokens. Consequence to disclose: MetalParts inputs are anisotropically resized from 16:9, so absolute MetalParts levels are not comparable to PCB, while the **B0↔B1 comparison remains paired and internally fair** (both methods see the identical tensor). This is a preprocessing convention, not a tuned parameter, and it is recorded in `results/e25/geometry_<domain>.json`.

**Verdict rule**: E25 has **no pre-registered GO/STOP gate**. `final_verdict.json` is therefore a **descriptive** summary (per-split and per-group ΔAUROC/ΔAP, sign counts, paired bootstrap CIs), and must be reported as descriptive evidence, not as a hypothesis test with a pre-set threshold.
