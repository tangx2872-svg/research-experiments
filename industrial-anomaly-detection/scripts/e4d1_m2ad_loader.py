#!/usr/bin/env python
"""E4-D1 — M²AD data access layer (shared by benchmark / smoke runner).

Design notes
------------
* Reads the OFFICIAL metadata `meta_unsupervised.json` (from `jsons.zip`) for
  split / label / mask information; never invents a split.
* Decodes `view` / `illumination` from the REAL filename with the official rule
  verified in E4-D0:  `view = name[1:4]`, `illumination = name[6:8]`.
* Defect type is recovered from the NG specimen directory name
  (`<type>[_<type>...]_<instance>_<index>`); Good specimen dirs are numeric.
* Provides a minimal anomalib `Folder` datamodule over an EXPLICIT image list,
  so the memory bank can be bounded (see E4-D1 protocol freeze).

Nothing here touches the frozen method: no α, no threshold, no fusion rule.
"""
from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "data" / "m2ad" / "extracted"          # contains {Category}/
META_PATH = ROOT / "data" / "m2ad" / "jsons_extract" / "meta_unsupervised.json"

CATEGORY = "Bird"
VIEWS = ["000", "030", "060", "090", "120", "150", "180", "210", "240", "270", "300", "330"]
ILLUMINATIONS = [f"{i:02d}" for i in range(1, 11)]


# --------------------------------------------------------------------------- decode
def decode_name(filename: str) -> dict:
    """Official rule (E4-D0 verified, 0/119,760 mismatches)."""
    return {"view": filename[1:4], "illumination": filename[6:8]}


def specimen_kind(object_name: str) -> str:
    return "Good" if object_name.isdigit() else "NG"


def defect_type_tokens(object_name: str) -> list[str]:
    if object_name.isdigit():
        return []
    return [t for t in object_name.split("_") if not t.isdigit()]


# --------------------------------------------------------------------------- records
def load_records(category: str = CATEGORY, meta_path: Path = META_PATH,
                 require_exists: bool = True) -> list[dict]:
    """Flat record list with decoded metadata. Keeps official split/labels."""
    raw = json.loads(meta_path.read_text())
    out: list[dict] = []
    for split in ("train", "test"):
        for cls, items in raw.get(split, {}).items():
            if cls != category:
                continue
            for r in items:
                img_rel = r["img_path"]                       # e.g. Bird/Good/000/A000_I01.png
                p = DATA_ROOT / img_rel
                if require_exists and not p.exists():
                    continue
                dec = decode_name(Path(img_rel).name)
                out.append({
                    "split": split,
                    "category": cls,
                    "kind": r["object_name"] if False else specimen_kind(r["object_name"]),
                    "specimen": r["object_name"],
                    "view": dec["view"],
                    "illumination": dec["illumination"],
                    "object_anomaly": int(r["object_anomaly"]),
                    "image_anomaly": int(r["image_anomaly"]),
                    "detectable": str(r["detectable"]),
                    "defect_types": "|".join(defect_type_tokens(r["object_name"])),
                    "image_path": str(p),
                    "mask_path": str(DATA_ROOT / r["mask_path"]) if r["mask_path"] else "",
                })
    return out


def group_by_specimen(records: list[dict]) -> dict[str, list[dict]]:
    g: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        g[r["specimen"]].append(r)
    return dict(g)


# --------------------------------------------------------------------------- subsets
def bank_and_val_specimens(train_specs: list[str], n_bank: int = 10, n_val: int = 2) -> tuple[list[str], list[str]]:
    """PRE-REGISTERED, result-independent stride selection over sorted specimen ids.

    Registered in docs/E4_D1_PROTOCOL.md BEFORE any GPU run.
      bank = sorted(train_specs)[::3][:n_bank]   -> spread across the whole range
      val  = sorted(train_specs)[1::3][:n_val]
    Rationale: a contiguous prefix could fall entirely inside one visual sub-class
    (M2AD has 2 sub-classes per category); a stride avoids that without peeking at
    any model output.
    """
    s = sorted(train_specs)
    return s[::3][:n_bank], s[1::3][:n_val]


def select(records: list[dict], specimens: set[str] | None = None,
           views: list[str] | None = None, illuminations: list[str] | None = None) -> list[dict]:
    out = []
    for r in records:
        if specimens is not None and r["specimen"] not in specimens:
            continue
        if views is not None and r["view"] not in views:
            continue
        if illuminations is not None and r["illumination"] not in illuminations:
            continue
        out.append(r)
    return out


# --------------------------------------------------------------------------- datamodule
def _as_dir(image_paths: list[str], cache_name: str) -> str:
    """Return a DIRECTORY for anomalib `Folder`.

    anomalib 2.6.2's `Folder` treats every `normal_dir` entry as a DIRECTORY to
    glob, so a list of image FILES raises
    `RuntimeError: Found 0 DirType.NORMAL images in <file>`
    (verified empirically: E4-D1 benchmark run 1).

    Engineered adapter (no library modification, no data copy): a deterministic
    symlink farm. It does not change pixels, transforms, or the method.
    """
    import hashlib

    files = [Path(p) for p in image_paths]
    if not all(p.is_file() for p in files):
        return image_paths  # already dirs
    h = hashlib.md5("\n".join(sorted(str(p) for p in files)).encode()).hexdigest()[:10]
    farm = DATA_ROOT.parent / "_e4d1_cache" / f"{cache_name}_{h}_{len(files)}"
    if not farm.is_dir() or len(list(farm.iterdir())) != len(files):
        make_symlink_farm([str(p) for p in files], farm)
    return str(farm)


def make_folder_datamodule(image_paths: list[str], val_paths: list[str] | None = None,
                           batch_size: int = 16, num_workers: int = 0,
                           seed: int = 0, name: str = "m2ad_bank"):
    """anomalib `Folder` datamodule over EXPLICIT normal image lists.

    `val_paths` is supplied as `normal_test_dir` with `val_split_mode="same_as_test"`
    so Lightning always has a `val_dataloader` (it requests one even when no
    validation is performed). The actual tau_val is recomputed by the runner with
    the frozen `e1b.predict_one` path, NOT by Lightning's val step.
    """
    from anomalib.data import Folder

    normal_dir = _as_dir(image_paths, f"{name}_bank")
    kwargs: dict = {}
    if val_paths:
        kwargs = {"normal_test_dir": _as_dir(val_paths, f"{name}_val"),
                  "val_split_mode": "same_as_test"}
    else:
        kwargs = {"val_split_mode": "none"}

    dm = Folder(
        name=name,
        normal_dir=normal_dir,
        train_batch_size=batch_size,
        eval_batch_size=batch_size,
        num_workers=num_workers,
        test_split_mode="none",
        seed=seed,
        **kwargs,
    )
    dm.setup()
    return dm


def make_symlink_farm(image_paths: list[str], dest: Path) -> Path:
    """Fallback: deterministic symlink farm (cheap, no copies)."""
    dest.mkdir(parents=True, exist_ok=True)
    for i, p in enumerate(sorted(image_paths)):
        link = dest / f"{i:06d}_{Path(p).name}"
        if not link.exists():
            link.symlink_to(Path(p).resolve())
    return dest


# --------------------------------------------------------------------------- scope estimate
def estimate_scope(records: list[dict], n_bank: int = 10, n_val: int = 2,
                   scoring_views: list[str] | None = None) -> dict:
    train = [r for r in records if r["split"] == "train"]
    test = [r for r in records if r["split"] == "test"]
    train_specs = sorted({r["specimen"] for r in train})
    bank_s, val_s = bank_and_val_specimens(train_specs, n_bank, n_val)
    bank = select(records, set(bank_s))
    val = select(records, set(val_s))
    score = select(test, views=scoring_views)
    return {
        "n_train_specimens": len(train_specs),
        "bank_specimens": bank_s, "val_specimens": val_s,
        "n_bank_images": len(bank), "n_val_images": len(val),
        "n_scoring_images": len(score),
        "scoring_views": scoring_views or VIEWS,
        "n_test_specimens": len({r["specimen"] for r in test}),
    }


if __name__ == "__main__":
    recs = load_records()
    print(f"records (existing files) = {len(recs)}")
    for split in ("train", "test"):
        sub = [r for r in recs if r["split"] == split]
        print(f"  {split}: {len(sub)} images, {len({r['specimen'] for r in sub})} specimens")
    est = estimate_scope(recs)
    print(json.dumps(est, indent=2))
    print("\nsample record:", json.dumps(recs[0], ensure_ascii=False))
