#!/usr/bin/env python3
"""E20-A: inventory of local M2AD data assets (read-only, no GPU, no downloads).

Scans:
  - data/m2ad/extracted/            (raw Bird images)
  - data/m2ad/e7_cache/*/           (l0_448, l0_256, clahe_*, retinex*)
  - data/m2ad/e9_cache/*/           (e9_*)
  - data/m2ad/_e4d1_cache/          (E4-D1 ImageFolder symlink farm)
  - data/m2ad/e8_gt/                (GT masks for Car/Cube)
  - data/m2ad/e5_1_retinex/         (E5-1 retinex bank/score/val)
  - data/m2ad/e15_farm/             (EfficientAD farm)
  - data/m2ad/jsons_extract/meta_unsupervised.json (official metadata, all 10 cats)

Outputs (results/e20/):
  - inventory_assets.json  : per-cache per-category per-view per-illum counts
  - inventory_summary.csv  : flat summary table
Writes nothing outside results/e20/ (mkdir allowed).
"""
import json
import os
import re
import sys
from collections import defaultdict, Counter

ROOT = "/root/autodl-tmp/research-experiments/industrial-anomaly-detection"
DATA = os.path.join(ROOT, "data", "m2ad")
OUT = os.path.join(ROOT, "results", "e20")
os.makedirs(OUT, exist_ok=True)

NAME_RE = re.compile(r"^A(\d{3})_I(\d{2})\.(png|jpg|jpeg|webp)$", re.IGNORECASE)


def classify_path(rel):
    """Return (cat, split_hint, specimen) from a relative path inside a cache root."""
    parts = rel.split(os.sep)
    cat = parts[0]
    if len(parts) >= 3:
        grp = parts[1]  # Good / NG / train / test / other
        spec = parts[2]
    else:
        grp = ""
        spec = ""
    return cat, grp, spec


def scan_dir(root, label, rec, max_files=None):
    """Scan a directory of images with A{view}_I{illum} names; fill rec[label]."""
    if not os.path.isdir(root):
        rec[label] = {"exists": False}
        return
    n = 0
    cat_grp = defaultdict(lambda: {"views": Counter(), "illums": Counter(), "specimens": set(), "n": 0})
    ext_counter = Counter()
    for dirpath, _, files in os.walk(root):
        for fn in files:
            m = NAME_RE.match(fn)
            if not m:
                continue
            n += 1
            rel = os.path.relpath(os.path.join(dirpath, fn), root)
            cat, grp, spec = classify_path(rel)
            key = f"{cat}|{grp}"
            d = cat_grp[key]
            d["views"][m.group(1)] += 1
            d["illums"][m.group(2)] += 1
            d["specimens"].add(spec)
            d["n"] += 1
            ext_counter[os.path.splitext(fn)[1].lower()] += 1
            if max_files and n >= max_files:
                break
        if max_files and n >= max_files:
            break
    rec[label] = {
        "exists": True,
        "total_images": n,
        "ext": dict(ext_counter),
        "groups": {
            k: {
                "n": v["n"],
                "views": sorted(v["views"].keys()),
                "illums": sorted(v["illums"].keys()),
                "n_specimens": len(v["specimens"]),
            }
            for k, v in sorted(cat_grp.items())
        },
    }


def scan_official_metadata(path, rec):
    """Official metadata: {train: {cat: [entries]}, test: {cat: [entries]}}."""
    if not os.path.exists(path):
        rec["official_metadata"] = {"exists": False}
        return
    meta = json.load(open(path))
    out = {}
    for split in ("train", "test"):
        out[split] = {}
        for cat, entries in meta.get(split, {}).items():
            views = Counter()
            illums = Counter()
            specs = set()
            n_anom = 0
            n_detectable = 0
            n_gt = 0
            for e in entries:
                views[e["view"]] += 1
                illums[e["illumination"]] += 1
                specs.add(e["object_name"])
                if e.get("image_anomaly") == 1:
                    n_anom += 1
                if str(e.get("detectable", "")).lower() == "true":
                    n_detectable += 1
                if e.get("mask_path"):
                    n_gt += 1
            out[split][cat] = {
                "n": len(entries),
                "n_specimens": len(specs),
                "views": sorted(views.keys()),
                "illums": sorted(illums.keys()),
                "n_image_anomaly1": n_anom,
                "n_detectable_true": n_detectable,
                "n_with_mask_path": n_gt,
            }
    rec["official_metadata"] = {"exists": True, "path": path, "summary": out}


def main():
    rec = {}
    # Raw extracted (Bird only)
    scan_dir(os.path.join(DATA, "extracted"), "extracted_raw", rec)

    # e7 caches
    e7 = os.path.join(DATA, "e7_cache")
    if os.path.isdir(e7):
        for sub in sorted(os.listdir(e7)):
            scan_dir(os.path.join(e7, sub), f"e7_cache/{sub}", rec)

    # e9 caches
    e9 = os.path.join(DATA, "e9_cache")
    if os.path.isdir(e9):
        for sub in sorted(os.listdir(e9)):
            scan_dir(os.path.join(e9, sub), f"e9_cache/{sub}", rec)

    # e8 gt (masks; names differ)
    gt = os.path.join(DATA, "e8_gt")
    if os.path.isdir(gt):
        cnt = {}
        for cat in sorted(os.listdir(gt)):
            p = os.path.join(gt, cat)
            if os.path.isdir(p):
                n = sum(len(f) for _, _, f in os.walk(p))
                cnt[cat] = n
        rec["e8_gt"] = {"exists": True, "mask_counts": cnt}

    # e4d1 symlink farm
    scan_dir(os.path.join(DATA, "_e4d1_cache"), "_e4d1_cache", rec)

    # e15 farm (ImageFolder style: train/good, test/good, test/anomaly)
    e15 = os.path.join(DATA, "e15_farm")
    if os.path.isdir(e15):
        cnt = defaultdict(int)
        for dirpath, _, files in os.walk(e15):
            rel = os.path.relpath(dirpath, e15)
            cnt[rel] = sum(1 for f in files if f.lower().endswith((".png", ".jpg", ".jpeg")))
        rec["e15_farm"] = {"exists": True, "dirs": {k: v for k, v in sorted(cnt.items())}}

    # e5_1 retinex
    r5 = os.path.join(DATA, "e5_1_retinex")
    if os.path.isdir(r5):
        structure = {}
        for sub in sorted(os.listdir(r5)):
            p = os.path.join(r5, sub)
            if os.path.isdir(p):
                items = sorted(os.listdir(p))[:10]
                n = sum(len(f) for _, _, f in os.walk(p))
                structure[sub] = {"n_files": n, "sample": items}
        rec["e5_1_retinex"] = {"exists": True, "structure": structure}

    # official metadata
    scan_official_metadata(os.path.join(DATA, "jsons_extract", "meta_unsupervised.json"), rec)

    # disk info
    st = os.statvfs("/root/autodl-tmp")
    rec["_disk"] = {
        "total_gb": round(st.f_blocks * st.f_frsize / 1e9, 2),
        "avail_gb": round(st.f_bavail * st.f_frsize / 1e9, 2),
        "used_gb": round((st.f_blocks - st.f_bfree) * st.f_frsize / 1e9, 2),
    }

    out_json = os.path.join(OUT, "inventory_assets.json")
    with open(out_json, "w") as f:
        json.dump(rec, f, indent=1, ensure_ascii=False)
    print(f"wrote {out_json}")

    # flat csv summary for caches
    csv_path = os.path.join(OUT, "inventory_summary.csv")
    with open(csv_path, "w") as f:
        f.write("cache,cat_group,n_images,views,illums,n_specimens\n")
        for label, d in sorted(rec.items()):
            if label.startswith("_") or not isinstance(d, dict) or "groups" not in d:
                continue
            for kg, g in d["groups"].items():
                f.write(
                    f"{label},{kg},{g['n']},{len(g['views'])},{len(g['illums'])},{g['n_specimens']}\n"
                )
    print(f"wrote {csv_path}")


if __name__ == "__main__":
    main()
