#!/usr/bin/env python3
"""E20-A part 2: precise cross-check of local caches vs official metadata split.

For each cache x cat x group, determine:
  - how many images fall into official train vs test split (by specimen + view + illum)
  - view120 / illumination coverage
  - overlap of Good specimens with train/test normal specimen sets

Read-only. Output: results/e20/inventory_crosscheck.json + prints.
"""
import json
import os
import re
from collections import Counter, defaultdict

ROOT = "/root/autodl-tmp/research-experiments/industrial-anomaly-detection"
DATA = os.path.join(ROOT, "data", "m2ad")
OUT = os.path.join(ROOT, "results", "e20")
NAME_RE = re.compile(r"^A(\d{3})_I(\d{2})\.png$", re.IGNORECASE)

meta = json.load(open(os.path.join(DATA, "jsons_extract", "meta_unsupervised.json")))

# Build official lookup: (cat, specimen, view, illum) -> (split, image_anomaly, detectable)
official = {}
for split in ("train", "test"):
    for cat, entries in meta[split].items():
        for e in entries:
            key = (cat, e["object_name"], e["view"], e["illumination"])
            official[key] = (split, e["image_anomaly"], str(e.get("detectable", "")))

# good vs ng specimen sets per cat (from official)
good_specs, ng_specs = {}, {}
for cat in meta["train"]:
    g = set()
    for e in meta["train"][cat]:
        if e["image_anomaly"] == 0:
            g.add(e["object_name"])
    good_specs[cat] = g
    n = set()
    for e in meta["test"][cat]:
        if e["image_anomaly"] == 1:
            n.add(e["object_name"])
        if e["image_anomaly"] == 0:
            g.add(e["object_name"])
    ng_specs[cat] = n


def crosscheck(cache_root, label):
    res = defaultdict(lambda: {
        "n": 0, "train": 0, "test": 0, "unmatched": 0,
        "view120": 0, "view120_train": 0, "view120_test": 0,
        "view120_test_det_t": 0, "view120_test_det_blank": 0, "view120_test_det_f": 0,
        "test_anom1": 0, "test_anom0": 0,
        "views": Counter(),
    })
    for dirpath, _, files in os.walk(cache_root):
        for fn in files:
            m = NAME_RE.match(fn)
            if not m:
                continue
            rel = os.path.relpath(os.path.join(dirpath, fn), cache_root)
            parts = rel.split(os.sep)
            cat, grp, spec = parts[0], parts[1] if len(parts) > 1 else "", parts[2] if len(parts) > 2 else ""
            view, illum = m.group(1), m.group(2)
            key = (cat, spec, view, illum)
            d = res[f"{cat}|{grp}"]
            d["n"] += 1
            d["views"][view] += 1
            o = official.get(key)
            if o is None:
                d["unmatched"] += 1
                continue
            split, anom, det = o
            d[split] += 1
            if view == "120":
                d["view120"] += 1
                d[f"view120_{split}"] += 1
                if split == "test":
                    if anom == 1:
                        d["test_anom1"] += 1
                        if det == "True":
                            d["view120_test_det_t"] += 1
                        elif det == "":
                            d["view120_test_det_blank"] += 1
                        else:
                            d["view120_test_det_f"] += 1
                    else:
                        d["test_anom0"] += 1
    return {k: {**v, "views": dict(v["views"])} for k, v in sorted(res.items())}


out = {}
for label, root in [
    ("e7_cache/l0_448", os.path.join(DATA, "e7_cache", "l0_448")),
    ("e7_cache/l0_256", os.path.join(DATA, "e7_cache", "l0_256")),
    ("extracted_raw", os.path.join(DATA, "extracted")),
]:
    out[label] = crosscheck(root, label)
    print(f"===== {label} =====")
    for k, v in out[label].items():
        print(
            f"  {k}: n={v['n']} train={v['train']} test={v['test']} unmatched={v['unmatched']} | "
            f"view120={v['view120']} (train={v['view120_train']} test={v['view120_test']}) "
            f"test_anom1={v['test_anom1']} test_anom0={v['test_anom0']} "
            f"detT={v['view120_test_det_t']} detBlank={v['view120_test_det_blank']} detF={v['view120_test_det_f']} "
            f"views={sorted(v['views'].keys())}"
        )

with open(os.path.join(OUT, "inventory_crosscheck.json"), "w") as f:
    json.dump(out, f, indent=1, ensure_ascii=False)
print("wrote results/e20/inventory_crosscheck.json")
