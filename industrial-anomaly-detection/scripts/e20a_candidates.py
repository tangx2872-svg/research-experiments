#!/usr/bin/env python3
"""E20-A part 3: machine-readable expansion candidate tables.

Outputs (results/e20/):
  - category_view_coverage.csv     : 10 categories x {train,test} x 12 views,
                                     official counts vs local cache presence
  - expansion_candidates.csv       : per-category expansion opportunity table
  - mini_experiment_proposals.csv  : the 3 ranked mini-experiment proposals

Read-only w.r.t. data/caches; writes only under results/e20/.
"""
import json
import os
import re
from collections import Counter

ROOT = "/root/autodl-tmp/research-experiments/industrial-anomaly-detection"
DATA = os.path.join(ROOT, "data", "m2ad")
OUT = os.path.join(ROOT, "results", "e20")
os.makedirs(OUT, exist_ok=True)

meta = json.load(open(os.path.join(DATA, "jsons_extract", "meta_unsupervised.json")))
NAME_RE = re.compile(r"^A(\d{3})_I(\d{2})\.png$", re.IGNORECASE)

CATS = ["Bird", "Car", "Cube", "Dice", "Doll", "Holder", "Motor", "Ring", "Teapot", "Tube"]
VIEWS_ALL = [f"{v:03d}" for v in range(0, 360, 30)]
VIEWS_CACHED = ["090", "120", "240", "330"]  # E8 frozen 4-view protocol

# ---------------------------------------------------------------- official counts
off = {}
for split in ("train", "test"):
    for cat in CATS:
        off[(split, cat)] = Counter()
        for e in meta[split][cat]:
            off[(split, cat)][(e["view"], e["image_anomaly"], str(e.get("detectable", "")))] += 1


def local_cache_count(kind, cat):
    """Count png files present in a cache dir for a category."""
    base = os.path.join(DATA, "e7_cache", kind, cat)
    n = 0
    if os.path.isdir(base):
        for _, _, files in os.walk(base):
            n += sum(1 for f in files if f.lower().endswith(".png"))
    return n


cache_counts = {}
for kind in ("l0_448", "l0_256", "clahe_mild", "clahe_medium"):
    for cat in CATS:
        cache_counts[(kind, cat)] = local_cache_count(kind, cat)

# extract GBs for Bird NG/G testing presence
extracted_bird = 0
ex = os.path.join(DATA, "extracted")
if os.path.isdir(ex):
    for _, _, files in os.walk(ex):
        extracted_bird += sum(1 for f in files if f.lower().endswith(".png"))

# ---------------------------------------------------------------- coverage matrix
# Per-(category, split, view) official counts. Local cache presence is an aggregate
# per category (cache dirs are view-agnostic); see expansion_candidates.csv columns
# l0_448_files / l0_256_files and inventory_crosscheck.json for the view120 detail.
with open(os.path.join(OUT, "category_view_coverage.csv"), "w") as f:
    f.write("category,split,view,n_official,n_anomaly1,n_detectable_false\n")
    for cat in CATS:
        for split in ("train", "test"):
            for v in VIEWS_ALL:
                c = off[(split, cat)]
                n = sum(cnt for (vv, aa, dd), cnt in c.items() if vv == v)
                n1 = sum(cnt for (vv, aa, dd), cnt in c.items() if vv == v and aa == 1)
                nf = sum(cnt for (vv, aa, dd), cnt in c.items() if vv == v and dd == "False")
                f.write(f"{cat},{split},{v},{n},{n1},{nf}\n")

# ---------------------------------------------------------------- per-category candidates
E8_EXT_ORDER = {"Dice": 1, "Doll": 2, "Holder": 3}

def test_counts(cat, views):
    c = off[("test", cat)]
    ok = sum(cnt for (vv, aa, dd), cnt in c.items() if vv in views and aa == 0)
    ng_kept = sum(cnt for (vv, aa, dd), cnt in c.items() if vv in views and aa == 1 and dd != "False")
    ng_false = sum(cnt for (vv, aa, dd), cnt in c.items() if vv in views and aa == 1 and dd == "False")
    return ok, ng_kept, ng_false, ok + ng_kept


rows = []
for cat in CATS:
    ok120, ng120, nf120, kept120 = test_counts(cat, ["120"])
    ok4, ng4, nf4, kept4 = test_counts(cat, VIEWS_CACHED)
    local_pixels = cat == "Bird"
    l448 = cache_counts[("l0_448", cat)]
    l256 = cache_counts[("l0_256", cat)]
    cl = cache_counts[("clahe_medium", cat)]
    used = cat in ("Bird", "Car", "Cube")
    if used:
        exp_type = "in_use (E16/E18)"
        acq = "local (cached)"
        acq_min = 0
    else:
        exp_type = "new independent category (type A)"
        acq = "hf-mirror streaming selective extraction (e8_fetch.py)"
        acq_min = 22  # ~45 min for 2 categories measured in E8 Stage A
    rows.append({
        "category": cat,
        "official_metadata_local": "yes",
        "pixels_local": "yes" if local_pixels else "no",
        "l0_448_files": l448,
        "l0_256_files": l256,
        "clahe_medium_files": cl,
        "test_v120_kept": kept120,
        "test_v120_filtered": nf120,
        "test_4view_kept": kept4,
        "test_4view_filtered": nf4,
        "used_in_e16_e18": "yes" if used else "no",
        "expansion_type": exp_type,
        "e8_pre_registered_ext_order": E8_EXT_ORDER.get(cat, ""),
        "acquisition_channel": acq,
        "acquisition_min_est": acq_min,
        "disk_gb_est_l0448_only": "" if used else 0.8,
        "disk_gb_est_4kinds": "" if used else 2.7,
    })

with open(os.path.join(OUT, "expansion_candidates.csv"), "w") as f:
    cols = list(rows[0].keys())
    f.write(",".join(cols) + "\n")
    for r in rows:
        f.write(",".join(str(r[c]) for c in cols) + "\n")

# ---------------------------------------------------------------- mini experiment proposals
proposals = [
    {
        "proposal_id": "M1",
        "title": "New-category generalisation of R1-ILA (Dice/Doll/Holder candidates)",
        "expansion_class": "A (new independent category)",
        "target_claim": "A1 extended: R1-ILA gain is not specific to Bird/Car/Cube; "
                        "illumination-shift dose-response (E16 hypothesis) reproduces on unseen categories",
        "why_current_insufficient": "A1/A2/A3 rest on 3 categories chosen by alphabetical "
                                    "completeness order (not performance, but still only 3); "
                                    "reviewer risk = '3-category coincidence'",
        "data_exists_and_compatible": "Official metadata local for all 10 cats; pixels via E8 "
                                      "streaming selective extraction (validated pipeline + hard gate A1-A5); "
                                      "frozen protocol (bank rule / view120 / official labels) unchanged",
        "fair_comparison": "F0 vs R1-ILA identical setup: same bank draw rule seed, same view120 "
                           "test, same 1-NN-max scoring, same official_metrics filter; "
                           "no history table -> drop E16 reproduction identity check (new script variant)",
        "metrics_statistics": "AUROC / AP / d' + illumination diagnostics A-E; 3 seeds; "
                              "mean delta + per-seed sign consistency + test-image paired bootstrap CI",
        "gpu_estimate": "~4.5 min/cell x 3 seeds x 1-2 cats = 14-28 min (F0+R1+diagnostics; E16 measured 4.4-4.6 min/cell)",
        "disk_estimate": "0.8 GB/cat (l0_448 only) or 2.7 GB/cat (4 cache kinds + masks)",
        "network_cost": "~22 min/cat (14.9-17.6 GB streamed @ ~16 MB/s; E8 measured ~45 min for Car+Cube)",
        "go_criteria": "3/3 seed dAUROC > 0 AND seed-mean >= +0.005; illumination-shift D ordering "
                       "consistent with gain sign across categories",
        "hold_criteria": "seed-mean in [0, +0.005) or 2/3 seeds positive",
        "stop_criteria": "seed-mean dAUROC < 0 (with ceiling check vs E13-B2 baseline-dependence)",
        "failure_interpretation": "If dAUROC ~ 0 with high frozen baseline -> consistent with "
                                  "baseline/headroom dependence (E13-B2), a boundary condition, not tuning target. "
                                  "If shift large but gain negative -> hypothesis damaged, report as-is. "
                                  "Forbidden: parameter change, seed cherry-picking, label edits",
        "risks": "hf-mirror reachability must be re-verified; streaming/gate failure; "
                 "category choice must be human-approved (do NOT auto-select)",
        "rank": 1,
    },
    {
        "proposal_id": "M2",
        "title": "4-view test-set extension on Bird/Car/Cube (zero data acquisition)",
        "expansion_class": "C (new test condition / larger evaluation split)",
        "target_claim": "A1/A3 robustness bound: conclusions are not an artefact of single view120; "
                        "cross-view stability of the R1-ILA gain",
        "why_current_insufficient": "E19 limitations #5 explicitly lists single-view as a known gap; "
                                    "geometric view change is a standard reviewer request",
        "data_exists_and_compatible": "100% already cached: l0_448 4-view test (Good 800 + NG 2000 "
                                      "per category) + tests split by specimen (no leakage); "
                                      "E8 frozen VIEWS = 090/120/240/330",
        "fair_comparison": "same F0/R1-ILA settings; only the test set is widened from view120 to "
                           "the frozen 4 views; bank/training untouched",
        "metrics_statistics": "4-view pooled AUROC/AP + per-view dAUROC (win count over 4 views) + "
                              "illumination x view stratification; statistical power: test size "
                              "~2700/cat shrinks the bootstrap delta-AUROC noise halfwidth from "
                              "~0.017-0.019 to ~0.009 (cf. E20-B: MVtec AD2 single scene 0.04-0.07)",
        "gpu_estimate": "~2.4-4x test extraction/scoring vs view120; pilot 1 seed x 3 cats ~30-40 min; "
                        "full 3 seeds x 3 cats ~1.5-2 h (needs a precise smoke measurement first)",
        "disk_estimate": "0 GB (no new files; results only)",
        "network_cost": "0",
        "go_criteria": "pooled 4-view seed-mean dAUROC >= 50% of the view120 value AND >= 6/8 "
                       "category-view cells positive (Bird+Car)",
        "hold_criteria": "positive but view-heterogeneous (1-2 views negative)",
        "stop_criteria": "pooled dAUROC < 0 or systematic view-driven reversal (e.g. only 120 positive)",
        "failure_interpretation": "If view120 is special -> report explicitly as a scope limitation; "
                                  "do NOT re-select views to restore the desired sign",
        "risks": "larger test inflates runtime; must NOT be framed as new categories",
        "rank": 2,
    },
    {
        "proposal_id": "M3",
        "title": "Bird full 12-view extension (local originals, zero network)",
        "expansion_class": "B (additional views, single category)",
        "target_claim": "A1/A3 robustness bound at maximum angular coverage (12 turntable views) "
                        "for the category with the strongest R1-ILA signal",
        "why_current_insufficient": "same single-view limitation; 12-view is the strongest available "
                                    "geometric-generalisation probe for one category",
        "data_exists_and_compatible": "Bird originals local (12 views x 10 illuminations); new l0_448 "
                                      "cache for the 8 uncovered test views must be generated locally "
                                      "(~5600 images, CPU); protocol unchanged",
        "fair_comparison": "same F0/R1-ILA, same bank; test = all 12 views x 10 illuminations",
        "metrics_statistics": "12-view pooled AUROC/AP + per-view dAUROC + illumination breakdown",
        "gpu_estimate": "cache build ~30-60 min CPU; GPU test extraction 12x vs view120; "
                        "3 seeds x 1 cat ~45-60 min (estimate, needs smoke)",
        "disk_estimate": "~2.2 GB (l0_448 only for newly generated views)",
        "network_cost": "0 (local originals)",
        "go_criteria": "12-view seed-mean dAUROC same sign as view120 AND >= 9/12 views positive",
        "hold_criteria": "positive overall with 1-3 negative views",
        "stop_criteria": "sign reversal vs view120 on the majority of views",
        "failure_interpretation": "report as category-specific view sensitivity; no view re-selection",
        "risks": "Bird-only (not a category expansion); cache generation must use frozen E8 writer code paths",
        "rank": 3,
    },
]

with open(os.path.join(OUT, "mini_experiment_proposals.csv"), "w") as f:
    cols = list(proposals[0].keys())
    f.write(",".join(cols) + "\n")
    for p in proposals:
        f.write(",".join('"' + str(p[c]).replace('"', "'") + '"' if isinstance(p[c], str) and "," in str(p[c]) else str(p[c]) for c in cols) + "\n")

print("wrote:")
print("  results/e20/category_view_coverage.csv")
print("  results/e20/expansion_candidates.csv")
print("  results/e20/mini_experiment_proposals.csv")
