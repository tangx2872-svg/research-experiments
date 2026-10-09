# E6-A — Asset Manifest (P0 audit)

**Audited:** 2026-10-09 · **GPU work: 0** · everything below is read from frozen on-disk artifacts.
**Purpose:** establish whether the seven methods' per-image scores are *apples-to-apples* before any
metric is defined or compared.

---

## 1. Method set and sources

| method | role | source file | sha256 | rows |
|---|---|---|---|---|
| **Original** | frozen reference (E4-X) | `results/e4_x/per_image_original.csv` | `4aebd115b91cd0e6ed6c69e7fcdb76ca2bc840e9253c65e477dfe1e3cc5b0a5d` | 700 |
| **B2** | E4-X STOP | `results/e4_x/per_image_b2.csv` | `4be33ccfb127985ddcabba0394c327c2f41e83e19f8ef866e9647a221f180a66` | 700 |
| **X6c** | E4-X STOP | `results/e4_x/per_image_x6c.csv` | `3c0b210244e39e387b7f4fb10baf4bddd87e79b1eeebc46e80d0d2c1ae95bc1c` | 700 |
| **C01** PIAD-Retinex | E5-1 HOLD | `results/e5_1/raw/C01_per_image.csv` | `7230e0da55a4ce1c27c171af30a064d149070d94c5637ffd760c3d0a83d9b4ca` | 700 |
| **C06** SimpleNet | E5-1 STOP | `results/e5_1/raw/C06_per_image.csv` | `986a2b1bc5cc200b5b1c1cae957cebef7e810e140265e1de681ec91e22bca3c8` | 700 |
| **C07** ReConPatch | E5-1 STOP | `results/e5_1/raw/C07_per_image.csv` | `7b41721d6574e605395e5e063add44d2d2756544b0bdd8291631f3d8372ed956` | 700 |
| **C08** CRAD | E5-1 STOP | `results/e5_1/raw/C08_per_image.csv` | `bced64ac08670517cc187db553d70897356b1c5928fb20a4549da0666b3854cf` | 700 |

Decision provenance: `results/e4_x/verdict.json` (`046c86c5…`, B2 = STOP, X6c = STOP, Original = REFERENCE)
and `results/e5_1/raw/<C>.json` (C01 = HOLD, C06/C07/C08 = STOP). Frozen E4-X metrics:
`results/e4_x/summary.csv` (`387568a8…`).

## 2. Field availability

| field | Original | B2 | X6c | C01 | C06 | C07 | C08 |
|---|---|---|---|---|---|---|---|
| `specimen` | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ |
| `kind` (Good/NG) | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ |
| `illumination` (01–10) | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ |
| `score` (raw, finite) | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ |
| `object_anomaly` | – | ✔ | – | – | – | – | – |
| `view` | – | ✔ | – | – | – | – | – |
| `image_path` | – | ✔ | – | – | – | – | – |
| normalised score | – | – | – | – | – | – | – |
| defect type (column) | – | – | – | – | – | – | – |

**Rules applied:** only *common* fields are compared. Missing columns are **never inferred**; a metric
that needs a missing field is reported `NOT_COMPUTABLE_FROM_EXISTING_ASSETS` instead of triggering GPU
work. `view` is **fixed at 120** for all seven files (frozen selection, `VIEW_SELECTION_SEED=20261008`).

## 3. Apples-to-apples verification (mechanical)

* All seven files have **700 rows** and exactly the same **700 unique `(specimen, illumination)` keys** —
  the key set of every method is *identical* to Original's.
* Class balance is identical everywhere: **200 Good / 500 NG**.
* 10 illuminations (`01`–`10`), **20 Good specimens**, **50 NG specimens**.
* ⇒ The comparison is **paired at the image level**: for every metric, method *m* and method *m'* are
  evaluated on the same images. This is required for M1/M2/M3/M5/M7/M8.

## 4. Additional fields reconstructible **without** inference

| field | how it is recovered | available for |
|---|---|---|
| **defect type** | NG `specimen` directory names are `<type>[_<type>…]_<instance>_<index>` (M²AD convention verified in E4-D0); tokens are extracted from the name — *no guessing of the image's lesion is involved, the type is literally the directory label* | all 7 methods (present in every file's `specimen` column) |
| **defect-type inventory @ view 120** | `damage` ×22, `hole` ×22, `scratch` ×22 (some specimens carry multiple tokens) | all 7 |
| **all-12-view scores** | `results/e4_d1_smoke/per_image.csv` (8 400 rows, Original only, includes `view`, `mask_path`, `image_anomaly`, `detectable`) | **Original only** — used solely for context/limits, never to build a cross-method metric |

## 5. Not available (recorded, not worked around)

| needed for | missing | status |
|---|---|---|
| patch-level illumination ranking (finer than image level) | no per-patch score dumps for the 7 methods | **M4 falls back to specimen-level ranking across illuminations** (see protocol §M4) — declared, not faked |
| cross-view illumination metric for C01/C06/C07/C08 | their per-image scores exist only at view 120 | `NOT_COMPUTABLE_FROM_EXISTING_ASSETS` — **GPU is not started to obtain it** |
| pixel/patch robustness metrics | not in these assets | out of scope for E6-A |

## 6. Conclusions of the audit

1. **PASS — apples-to-apples**: the seven methods form a fully paired, class-balanced, illumination-complete comparison at view 120.
2. **No field needs to be invented**, and no metric in E6-A requires GPU.
3. **Two declared limitations** are carried into the protocol: (a) M4 must operate at specimen level, not patch level; (b) no cross-view evidence exists for the E5-1 candidates, so E6-A can only validate *metrics*, not *method* view-generality.
