#!/usr/bin/env python
"""E4-D0 — M²AD integrity audit (CPU only, no GPU, no model).

Verifies that the REAL downloaded M²AD files agree with the official metadata
shipped in `jsons.zip` (`meta_unsupervised.json`) and with the static protocol
audit in `docs/E4_LL_IAD_M2AD_PROTOCOL_AUDIT.md`.

Outputs (written by the caller via --outdir):
  E4_M2AD_LOCAL_INVENTORY.csv        real tree inventory (local category only)
  E4_M2AD_ILLUMINATION_COVERAGE.csv  per-illumination Good/NG image counts
  E4_M2AD_PAIRING_AUDIT.csv          same-specimen + same-view + I01..I10 existence
  E4_M2AD_VIEW_COVERAGE.csv          per-specimen view x illumination completeness
  E4_M2AD_STRUCTURE_MISMATCH.csv     metadata vs real-file disagreements (if any)

Usage:
  python scripts/e4d0_m2ad_audit.py \
      --metadata data/m2ad/jsons_extract/meta_unsupervised.json \
      --tree-root data/m2ad/extracted/Bird \
      --category Bird --outdir docs
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

# ---------------------------------------------------------------- filename rules
# Confirmed against official metadata (E4-D0): img filename example `A060_I01.png`
#   view         = filename[1:4]   (3 chars, azimuth degrees: 000,030,...,330)
#   illumination = filename[6:8]   (2 chars, 01..10)
RE_IMG = re.compile(r"^(?P<prefix>.)(?P<view>\d{3})_(?P<ilit>I)?(?P<illum>\d{2})\.(?P<ext>png|jpg|jpeg|bmp|tif|tiff)$",
                    re.IGNORECASE)
IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def parse_image_name(name: str) -> dict | None:
    """Return {'view','illumination'} decoded with the official index rules, or None."""
    stem_ok = Path(name).suffix.lower() in IMG_EXTS
    if not stem_ok:
        return None
    # official rule: view = name[1:4], illumination = name[6:8]
    if len(name) < 8:
        return None
    return {"view": name[1:4], "illumination": name[6:8]}


def parse_mask_name(name: str) -> dict | None:
    """`A060_mask.png` -> {'view': '060'}  (per-(specimen,view) mask, shared across illumination)."""
    m = re.match(r"^(?P<prefix>.)(?P<view>\d{3})_mask\.(?P<ext>\w+)$", name, re.IGNORECASE)
    return {"view": m.group("view")} if m else None


def artifact_of(object_name: str) -> tuple[str, str]:
    """NG specimen dir name -> (defect_type, defect_subtype). Good dirs are numeric."""
    if object_name.isdigit():
        return ("good", "")
    parts = object_name.split("_")
    return (parts[0], object_name)


# ---------------------------------------------------------------- metadata load
def load_metadata(path: Path) -> list[dict]:
    d = json.loads(path.read_text())
    rows: list[dict] = []
    for split in ("train", "test"):
        for cls, items in d.get(split, {}).items():
            for r in items:
                rec = dict(r)
                rec["split"] = split
                rows.append(rec)
    return rows


# ---------------------------------------------------------------- real tree walk
def walk_tree(root: Path, category: str) -> list[dict]:
    """Read the REAL files on disk (no README inference)."""
    out: list[dict] = []
    cls_dir = root if root.name == category else root / category
    if not cls_dir.is_dir():
        # tree-root may already be <cat>/
        cls_dir = root
    for sub in ("Good", "NG", "GT"):
        d = cls_dir / sub
        if not d.is_dir():
            continue
        for spec_dir in sorted(p for p in d.iterdir() if p.is_dir()):
            for f in sorted(spec_dir.iterdir()):
                if not f.is_file():
                    continue
                out.append({
                    "sub": sub,
                    "object_name": spec_dir.name,
                    "filename": f.name,
                    "relpath": str(f.relative_to(cls_dir)),
                    "size_bytes": f.stat().st_size,
                })
    return out


def compare_metadata_vs_tree(meta_rows: list[dict], tree_rows: list[dict], category: str) -> list[dict]:
    meta_img = {r["img_path"].replace(f"{category}/", "", 1): r for r in meta_rows}
    meta_mask = {r["mask_path"].replace(f"{category}/", "", 1)
                 for r in meta_rows if r.get("mask_path")}
    meta_seg = {r["seg_path"].replace(f"{category}/", "", 1)
                for r in meta_rows if r.get("seg_path")}
    tree_img = {r["relpath"] for r in tree_rows if r["sub"] in ("Good", "NG")}
    # GT/ holds BOTH `*_mask.png` (pixel mask) and `*_seg.png` (defect-type segmentation).
    # The official metadata leaves `seg_path` EMPTY, so *_seg.png legitimately has no metadata entry.
    gt_all = [r["relpath"] for r in tree_rows if r["sub"] == "GT"]
    tree_mask = {p for p in gt_all if p.endswith("_mask.png")}
    tree_seg = {p for p in gt_all if p.endswith("_seg.png")}
    tree_gt_other = set(gt_all) - tree_mask - tree_seg

    mism: list[dict] = []
    for p in sorted(tree_img - set(meta_img)):
        mism.append({"kind": "file_on_disk_not_in_metadata", "path": p})
    for p in sorted(set(meta_img) - tree_img):
        mism.append({"kind": "in_metadata_not_on_disk", "path": p})
    for p in sorted(tree_mask - meta_mask):
        mism.append({"kind": "mask_on_disk_not_in_metadata", "path": p})
    for p in sorted(meta_mask - tree_mask):
        mism.append({"kind": "mask_in_metadata_not_on_disk", "path": p})
    for p in sorted(tree_gt_other):
        mism.append({"kind": "unexpected_gt_filename_pattern", "path": p})
    # *_seg.png is informational only (metadata seg_path is empty by design)
    for p in sorted(tree_seg):
        mism.append({"kind": "INFO_seg_file_no_metadata_entry", "path": p})

    # view/illumination decode agreement (metadata field vs filename rule)
    for r in meta_rows:
        fn = Path(r["img_path"]).name
        p = parse_image_name(fn)
        if p is None:
            mism.append({"kind": "filename_unparsable", "path": r["img_path"]})
            continue
        if p["view"] != str(r["view"]) or p["illumination"] != str(r["illumination"]):
            mism.append({"kind": "field_decode_disagreement", "path": r["img_path"],
                         "detail": f"file=({p['view']},{p['illumination']}) "
                                   f"meta=({r['view']},{r['illumination']})"})
    return mism


# ---------------------------------------------------------------- audits
def illumination_coverage(meta_rows: list[dict]) -> list[dict]:
    c: Counter = Counter()
    for r in meta_rows:
        kind = "Good" if r["object_anomaly"] == 0 else "NG"
        c[(r["illumination"], kind)] += 1
    illums = sorted({r["illumination"] for r in meta_rows})
    rows = []
    for i in illums:
        g, n = c[(i, "Good")], c[(i, "NG")]
        rows.append({"category": meta_rows[0]["cls_name"] if meta_rows else "",
                     "illumination_id": i, "Good_images": g, "NG_images": n, "total": g + n})
    return rows


def view_coverage(meta_rows: list[dict]) -> list[dict]:
    per = defaultdict(lambda: defaultdict(set))     # spec -> view -> {illum}
    kind = {}
    for r in meta_rows:
        per[r["object_name"]][r["view"]].add(r["illumination"])
        kind[r["object_name"]] = "Good" if r["object_anomaly"] == 0 else "NG"
    allv = sorted({r["view"] for r in meta_rows})
    alli = sorted({r["illumination"] for r in meta_rows})
    rows = []
    for spec in sorted(per):
        views = per[spec]
        n_full = sum(1 for v in allv if len(views.get(v, ())) == len(alli))
        n_imgs = sum(len(views.get(v, ())) for v in allv)
        rows.append({"specimen_id": spec, "kind": kind[spec],
                     "n_views": len(views), "n_illuminations_total": len(set().union(*views.values())) if views else 0,
                     "expected_configs": len(allv) * len(alli), "observed_images": n_imgs,
                     "complete_views": n_full, "complete": int(n_full == len(allv))})
    return rows


def pairing_audit(meta_rows: list[dict], n_good: int = 5, n_ng: int = 5, seed: int = 0) -> list[dict]:
    rnd = random.Random(seed)
    idx = {(r["object_name"], r["view"], r["illumination"]): r for r in meta_rows}
    allv = sorted({r["view"] for r in meta_rows})
    alli = sorted({r["illumination"] for r in meta_rows})
    goods = sorted({r["object_name"] for r in meta_rows if r["object_anomaly"] == 0})
    ngss = sorted({r["object_name"] for r in meta_rows if r["object_anomaly"] == 1})
    # selection rule recorded BEFORE looking at any result: random sample, fixed seed
    pick = [("Good", s) for s in rnd.sample(goods, min(n_good, len(goods)))] + \
           [("NG", s) for s in rnd.sample(ngss, min(n_ng, len(ngss)))]
    out = []
    for kind, spec in pick:
        views = sorted({r["view"] for r in meta_rows if r["object_name"] == spec})
        v = rnd.choice(views) if views else None   # seeded random pick among EXISTING views
        for i in alli:
            r = idx.get((spec, v, i))
            out.append({"category": meta_rows[0]["cls_name"] if meta_rows else "",
                        "status": kind, "specimen_id": spec, "view_id": v,
                        "illumination_id": i,
                        "image_path": r["img_path"] if r else "",
                        "exists": int(r is not None)})
    return out


def mask_audit(meta_rows: list[dict], extract_root: Path, n: int = 20, seed: int = 0) -> list[dict]:
    """§19: NG image -> GT mask. Checks existence, filename mapping, resolution, non-emptiness.

    `extract_root` is the directory that CONTAINS the category dir (e.g. .../extracted),
    because metadata paths are `<Category>/Good|NG|GT/...`.
    """
    import numpy as np
    from PIL import Image

    cand = [r for r in meta_rows if r.get("mask_path")]
    rnd = random.Random(seed)
    pick = rnd.sample(cand, min(n, len(cand)))
    rows = []
    for r in pick:
        img_p = extract_root / r["img_path"]
        mask_p = extract_root / r["mask_path"]
        rec = {"category": r["cls_name"], "image_relpath": r["img_path"], "mask_relpath": r["mask_path"],
               "img_exists": int(img_p.exists()), "mask_exists": int(mask_p.exists())}
        # mapping rule: image stem `A060_I01` -> mask stem `A060_mask`
        rec["mask_name_rule_ok"] = int(Path(r["img_path"]).stem[:-4] == Path(r["mask_path"]).stem.replace("_mask", ""))
        try:
            with Image.open(img_p) as im:
                rec["img_size"] = f"{im.size[0]}x{im.size[1]}"
            with Image.open(mask_p) as mk:
                rec["mask_size"] = f"{mk.size[0]}x{mk.size[1]}"
                rec["mask_mode"] = mk.mode
                arr = np.asarray(mk.convert("L"))
                rec["mask_nonempty"] = int(arr.max() > 0)
                rec["mask_fg_ratio"] = round(float((arr > 0).mean()), 6)
            rec["resolution_match"] = int(rec["img_size"] == rec["mask_size"])
        except Exception as e:  # noqa: BLE001
            rec["error"] = str(e)[:120]
        rows.append(rec)
    return rows


def split_audit(meta_rows: list[dict]) -> dict:
    tr = {(r["cls_name"], r["object_name"]) for r in meta_rows if r["split"] == "train"}
    te = {(r["cls_name"], r["object_name"]) for r in meta_rows if r["split"] == "test"}
    return {"n_train_specimens": len(tr), "n_test_specimens": len(te),
            "n_overlap": len(tr & te), "overlap": sorted(tr & te)[:20]}


# ---------------------------------------------------------------- io helpers
def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--metadata", required=True)
    ap.add_argument("--tree-root", default=None)
    ap.add_argument("--category", default="Bird")
    ap.add_argument("--outdir", default="docs")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    out = Path(a.outdir)
    all_rows = load_metadata(Path(a.metadata))
    print(f"[meta] total records = {len(all_rows)}")

    # ---- global split audit (all 10 categories)
    sa = split_audit(all_rows)
    print(f"[split] train specimens={sa['n_train_specimens']} test specimens={sa['n_test_specimens']} "
          f"overlap={sa['n_overlap']} {'<-- LEAKAGE' if sa['n_overlap'] else '(clean)'}")

    # ---- per-category record counts
    cnt = Counter((r["cls_name"], r["split"]) for r in all_rows)
    print("\n[per-class records] train / test")
    for cls in sorted({r["cls_name"] for r in all_rows}):
        print(f"  {cls:9s} {cnt[(cls,'train')]:6d} / {cnt[(cls,'test')]:6d}")

    # ---- local category focuses
    local = [r for r in all_rows if r["cls_name"] == a.category]
    print(f"\n[local] category={a.category} records={len(local)}")

    inv_rows, mism = [], []
    tree_rows = []
    if a.tree_root:
        tr_root = Path(a.tree_root)
        tree_rows = walk_tree(tr_root, a.category)
        print(f"[tree] real files found = {len(tree_rows)}")
        for r in tree_rows:
            dec = parse_image_name(r["filename"]) or parse_mask_name(r["filename"]) or {}
            inv_rows.append({"category": a.category, "sub": r["sub"],
                             "specimen_id": r["object_name"], "filename": r["filename"],
                             "view_id": dec.get("view", ""), "illumination_id": dec.get("illumination", ""),
                             "relpath": r["relpath"], "size_bytes": r["size_bytes"]})
        mism = compare_metadata_vs_tree(local, tree_rows, a.category)
        real_mism = [m for m in mism if not m["kind"].startswith("INFO_")]
        info_mism = [m for m in mism if m["kind"].startswith("INFO_")]
        print(f"[tree vs metadata] real mismatches = {len(real_mism)} ; informational = {len(info_mism)}")
        for m in real_mism[:10]:
            print("    MISMATCH:", m)
        ext = Counter(Path(r["filename"]).suffix.lower() for r in tree_rows)
        print(f"[tree] extensions: {dict(ext)}")
        gt = Counter(Path(r["filename"]).name.rsplit("_", 1)[-1] for r in tree_rows if r["sub"] == "GT")
        print(f"[tree] GT file kinds = {dict(gt)}")
        subs = Counter(r["sub"] for r in tree_rows)
        print(f"[tree] sub counts = {dict(subs)}")
        ma = mask_audit(local, tr_root.parent if tr_root.name == a.category else tr_root, seed=a.seed)
        write_csv(out / "E4_M2AD_MASK_AUDIT.csv", ma)
        ok = sum(r.get("mask_exists", 0) for r in ma)
        print(f"[mask] matched (exists) = {ok}/{len(ma)}; resolution_match = "
              f"{sum(r.get('resolution_match',0) for r in ma)}/{len(ma)}; nonempty = "
              f"{sum(r.get('mask_nonempty',0) for r in ma)}/{len(ma)}; name_rule_ok = "
              f"{sum(r.get('mask_name_rule_ok',0) for r in ma)}/{len(ma)}")

    write_csv(out / "E4_M2AD_LOCAL_INVENTORY.csv", inv_rows)
    write_csv(out / "E4_M2AD_ILLUMINATION_COVERAGE.csv", illumination_coverage(local))
    write_csv(out / "E4_M2AD_PAIRING_AUDIT.csv", pairing_audit(local, seed=a.seed))
    write_csv(out / "E4_M2AD_VIEW_COVERAGE.csv", view_coverage(local))
    write_csv(out / "E4_M2AD_STRUCTURE_MISMATCH.csv", mism)

    pa = pairing_audit(local, seed=a.seed)
    for kind in ("Good", "NG"):
        sub = [r for r in pa if r["status"] == kind]
        rate = sum(r["exists"] for r in sub) / len(sub) if sub else float("nan")
        print(f"[pairing] {kind} complete-illumination rate = {rate:.3f}  ({sum(r['exists'] for r in sub)}/{len(sub)})")

    vc = view_coverage(local)
    print(f"[view] specimens={len(vc)} complete(12x10)={sum(r['complete'] for r in vc)}")

    def _pair_rate(kind: str) -> float | None:
        sub = [r for r in pa if r["status"] == kind]
        return (sum(r["exists"] for r in sub) / len(sub)) if sub else None

    summary = {"category": a.category, "n_records": len(local),
               "n_tree_files": len(tree_rows), "n_mismatch": len([m for m in mism if not m["kind"].startswith("INFO_")]),
               "n_mismatch_informational": len([m for m in mism if m["kind"].startswith("INFO_")]),
               "split": sa,
               "pairing_rate_good": _pair_rate("Good"),
               "pairing_rate_ng": _pair_rate("NG"),
               "illuminations": sorted({r["illumination"] for r in local}),
               "views": sorted({r["view"] for r in local}),
               "defect_types": sorted({artifact_of(r["object_name"])[0]
                                       for r in local if r["object_anomaly"] == 1})}
    (out / "E4_M2AD_AUDIT_SUMMARY.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print("\n[summary]", json.dumps(summary, ensure_ascii=False)[:600])
    return 0


if __name__ == "__main__":
    sys.exit(main())
