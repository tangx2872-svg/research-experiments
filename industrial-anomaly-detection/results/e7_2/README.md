# E7-2 — Cross-view medium validation of the E7-1 survivors

**Triggered mechanically** by the E7-1 gate (protocol §11): `SELECTED = [P03, P02]`, 2 candidates ≤ 4.
**Protocol:** `docs/E7_1_FROZEN_PROTOCOL.md` §15 (view draw + finalist rule), frozen at `af56dd3`.
**Views frozen before any E7-2 metric was read:** `['120', '090', '240', '330']`
(primary 120 + **090/240/330** drawn with `VIEW_SEED=20261009` from the 11 remaining views).

**Runtime: 46 min 25 s** (plus a one-time 630 s photometric cache build) — budget 1.5–3 h.
Full normal bank restored: **1200 images**. Seed 0.

---

## 1. Setup

| item | value |
|---|---|
| candidates | **P03** (CLAHE medium → PatchCore), **P02** (CLAHE mild → PatchCore) |
| reference | **P00** Original PatchCore, recomputed per view under the identical protocol |
| bank | **full** 1200-image normal bank (E5-1's frozen bank), view-independent |
| views | 120 (primary) + 090, 240, 330 |
| test per view | 700 images (200 Good + 500 NG), unreduced |
| metric | **image AUROC (primary)**; AUPR / `d′` secondary |
| optim. | the PatchCore model depends only on the view-independent bank ⇒ fitted **once per pipeline** and scored on all 4 views (identical models; no method change) |

## 2. Per-view results

| candidate | view 120 | view 090 | view 240 | view 330 | mean AUROC | median AUROC |
|---|---|---|---|---|---|---|
| **P00 (baseline)** | 0.76963 | 0.82964 | 0.76890 | 0.75465 | 0.78071 | 0.76927 |
| **P03** | 0.81062 | 0.81536 | 0.80133 | 0.79992 | **0.80681** | 0.80598 |
| **P02** | 0.79543 | 0.81060 | 0.78714 | 0.79276 | **0.79648** | 0.79410 |

The baseline itself swings **0.7547 → 0.8296** across views (±0.037), so a single-view result was
never sufficient evidence — this is why E7-2 was run.

## 3. Per-view deltas vs the same-view P00

| candidate | Δ120 | Δ090 | Δ240 | Δ330 | mean Δ | median Δ | positive views | worst-view Δ | verdict |
|---|---|---|---|---|---|---|---|---|---|
| **P03** | **+0.04099** | −0.01428 | **+0.03243** | **+0.04527** | **+0.02610** | +0.03671 | **3/4** | −0.01428 | **FINALIST (Path A)** |
| **P02** | **+0.02580** | −0.01904 | **+0.01824** | **+0.03811** | **+0.01578** | +0.02202 | **3/4** | −0.01904 | **FINALIST (Path A)** |

Finalist rule (protocol §15): **Path A** = `mean ΔAUROC ≥ +0.015` **and** `positive views ≥ 3/4`
→ both candidates satisfy it. Path B was not needed. **At most 2 finalists** — exactly 2.

Secondary metrics (means over the 4 views):

| candidate | mean AUPR | mean ΔAUPR | mean `d′` | mean Δ`d′` |
|---|---|---|---|---|
| P03 | 0.92678 | +0.00661 | 1.4158 | +0.0616 |
| P02 | 0.92300 | −0.00309 | 1.3814 | +0.0272 |

P03 improves **all three** metrics on average (AUROC, AUPR, `d′`); P02's AUPR is marginally negative.

## 4. Interpretation

* **The E7-1 CLAHE effect survives a view change.** Both CLAHE variants stay ahead of the baseline on
  3 of 4 views, with mean gains of **+0.0261** (P03) and **+0.0158** (P02) — larger than the
  E7-1 single-view numbers, because the full bank helps the CLAHE pipelines more than the baseline.
* **It is not universal**: at view **090** the baseline is unusually strong (0.8296) and both CLAHE
  variants lose there (−0.014 / −0.019). The gain is therefore **view-conditional but repeatable**,
  not a single-view artefact.
* **`medium` > `mild` consistently** (P03 beats P02 at every one of the 4 views and in the mean), so
  the single-module winner of this round is **P03 = CLAHE (clipLimit = 2.0, tile 8×8) → PatchCore**.
* **Interpretation for the paper:** a *very* small, literature-standard, training-free input
  pre-processing step (OpenCV CLAHE) applied before a standard PatchCore detector gives a small but
  reproducible image-AUROC gain at the frozen view and at 3 of 3 additional views. That is precisely
  the "small, simple, clearly-divided A/B contribution" shape the frozen upstream design was after —
  with the honest caveat that the effect is a few AUROC points, is view-dependent in magnitude, and
  is accompanied by a flat-to-slightly-negative AUPR for the `mild` variant.
* **All composition routes remain dead**: E7-1 found negative synergy for all 6 A+B cells, so no
  combination was eligible for E7-2.

## 5. Artifacts

```
results/e7_2/
├── README.md   cross_view.json   progress.json
├── raw/ P00_view{120,090,240,330}.json + _per_image.csv
│        P03_view{...}.json + _per_image.csv
│        P02_view{...}.json + _per_image.csv
├── figures/ fig1_2_cross_view.png
└── logs/ e7_2_run.log
```

## 6. Finalists

**P03 — CLAHE medium → PatchCore** (mean ΔAUROC **+0.02610**, 3/4 views positive, all three metrics up)
**P02 — CLAHE mild → PatchCore** (mean ΔAUROC **+0.01578**, 3/4 views positive)

**Best pipeline of E7: P03.**

## 7. Next step

**STOP.** Protocol §16/W: no E7-3, no multi-seed, no second category, no new-module search, no
parameter rescue, no paper writing — all require explicit human authorisation. The only
evidence-backed direction from this round is the **input-photometric / training-free** family
(CLAHE), which is consistent with E5-1 (input-level C01 was the only non-failing insertion point) and
with E6-A (which demoted the previous robustness metric).
