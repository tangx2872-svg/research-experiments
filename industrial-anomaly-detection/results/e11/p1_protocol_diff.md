# E11-P1 · Dual-Track Protocol Diff (code-verified)

Official repo `hustCYQ/M2AD` @ `a606aa0`, `data/ad_dataset.py` (read verbatim).

## THREE protocol differences (ordered by impact)

### 1. LABEL DEFINITION (dominant) — object_anomaly vs image_anomaly
| label | meaning | used by |
|---|---|---|
| `object_anomaly` | specimen-level: is the OBJECT defective? | **our E4-E10** (loader `kind` = specimen name numeric/NG) |
| `image_anomaly` | image-level: is the defect VISIBLE in THIS view/illum? | **official M2AD** (`ad_dataset.py` normal_idx/outlier_idx) |

Consequence (Bird view120): 50 NG specimens x 10 illum = 500 images, but only 380 have
`image_anomaly==1`; the other 120 (defect invisible at that illumination) are labeled NORMAL by
the official protocol but NG by ours.

### 2. DETECTABLE FILTER (minor) — `detectable != False`
- `detectable` is a **boolean** on NG (True/False) and empty-string `''` on Good.
- Filter drops only non-detectable NG (Bird 365 / Car 309 / Cube 222 full-test).
- Good samples all kept (`'' != False`).
- **Loader bug**: our `e4d1_m2ad_loader.load_records` stringifies `detectable` (`str(...)`), so
  `'False' != False` is True and the filter silently fails on loader records. E11 must use the
  RAW metadata boolean, not loader records.

### 3. VIEW/ILLUM SCOPE
- Ours: fixed view120 (or 4 frozen views), test 700/view.
- Official Invariant: all 12 views x 10 illum, single-image scoring.

## Verified answers (P1.2)
1. `detectable==False` only on NG (0 on Good). YES.
2. OK/Good all kept. YES (detectable='').
3. Other test filters: NONE (only class selection + detectable).
4. Separate image/pixel filter: NONE (single `data_all`, same set for both metrics).
5. Special aggregation: adeval standard image/pixel AUROC + AUPRO + AP.
6. Label/mask/path alignment: img_path/mask_path/object_anomaly/image_anomaly all present;
   image-level vs object-level labels differ (see #1).
