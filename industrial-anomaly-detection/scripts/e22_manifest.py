#!/usr/bin/env python
"""E22-P0: freeze the full-category manifest from official metadata (READ-ONLY).

Outputs results/e22/category_manifest.csv with exactly the columns required by the
E22 execution protocol. Categories = all keys of meta_unsupervised.json (official order).
"""
from __future__ import annotations

import csv
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
OUTDIR = ROOT / "results" / "e22"
OUTDIR.mkdir(parents=True, exist_ok=True)

META = json.load(open(ROOT / "data/m2ad/jsons_extract/meta_unsupervised.json"))
EXISTING = {"Bird", "Car", "Cube"}

rows = []
for cat in META["train"].keys():  # official order
    tr = META["train"][cat]
    te = META["test"][cat]
    v120 = [e for e in te if e["view"] == "120"]
    kept = [e for e in v120 if e["detectable"] is not False]
    n_ok = sum(1 for e in kept if e["image_anomaly"] == 0)
    n_ng = sum(1 for e in kept if e["image_anomaly"] == 1)
    # local cache presence (l0_448 view120 test + pool sample)
    l448 = ROOT / "data/m2ad/e7_cache/l0_448" / cat
    n_cached = 0
    if l448.is_dir():
        for dp, _, fs in os.walk(l448):
            n_cached += sum(1 for f in fs if f.lower().endswith(".png"))
    is_existing = cat in EXISTING
    rows.append({
        "category": cat,
        "is_existing": int(is_existing),
        "is_new": int(not is_existing),
        "metadata_train_count": len(tr),
        "metadata_test_count": len(te),
        "view120_raw_count": len(v120),
        "view120_detectable_count": len(kept),
        "good_count": n_ok,
        "anomaly_count": n_ng,
        "local_pixel_status": ("original_present" if cat == "Bird" else
                               ("cache_present" if is_existing else "absent")),
        "local_feature_status": ("l0_448_cached" if is_existing else "absent"),
        "l0_448_total_files_local": n_cached,
    })

cols = ["category", "is_existing", "is_new", "metadata_train_count", "metadata_test_count",
        "view120_raw_count", "view120_detectable_count", "good_count", "anomaly_count",
        "local_pixel_status", "local_feature_status", "l0_448_total_files_local"]
with open(OUTDIR / "category_manifest.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=cols)
    w.writeheader()
    w.writerows(rows)

print(f"wrote {OUTDIR/'category_manifest.csv'}")
print(f"{'category':10s} {'existing':>8s} {'train':>6s} {'test':>6s} {'v120raw':>8s} "
      f"{'v120kept':>9s} {'good':>5s} {'anom':>5s}  pixel / feature")
for r in rows:
    print(f"{r['category']:10s} {r['is_existing']:>8d} {r['metadata_train_count']:>6d} "
          f"{r['metadata_test_count']:>6d} {r['view120_raw_count']:>8d} "
          f"{r['view120_detectable_count']:>9d} {r['good_count']:>5d} {r['anomaly_count']:>5d}  "
          f"{r['local_pixel_status']} / {r['local_feature_status']}")
new = [r["category"] for r in rows if r["is_new"]]
print(f"\nFROZEN NEW-CATEGORY LIST ({len(new)}): {', '.join(new)}")
