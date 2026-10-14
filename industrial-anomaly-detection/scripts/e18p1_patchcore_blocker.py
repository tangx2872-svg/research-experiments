#!/usr/bin/env python
"""E18-P1 - evidence for the PatchCore Car/Cube blocker (zero GPU).

PatchCore's frozen pipeline (e7_pipelines.run_patchcore -> e1b.predict_one ->
preprocess_for_model) reads the ORIGINAL image path and applies
`TF.resize(to_tensor(img), [256,256], antialias=True)` (BILINEAR default).

The l0_256 cache is a BICUBIC 256x256 uint8 artefact (E11-P2 cache validation).
Substituting it would change the model input, i.e. a configuration deviation.
This script quantifies that difference and records why the 2 units cannot run.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torchvision.transforms.functional as TF  # noqa: E402
from PIL import Image  # noqa: E402

import e4d1_m2ad_loader as LD  # noqa: E402
import e8_common as E8  # noqa: E402

OUT = ROOT / "results" / "e18" / "filled" / "patchcore_input_blocker.json"


def main() -> int:
    CATS = ["Bird", "Car", "Cube"]
    avail = {}
    for cat in CATS:
        recs = LD.load_records(category=cat, require_exists=False)
        test = [r for r in LD.select(recs, views=["120"]) if r["split"] == "test"]
        orig = sum(1 for r in test if Path(r["image_path"]).exists())
        cache = sum(1 for r in test if E8.cache_path("l0_256", r).exists())
        avail[cat] = {"view120_test_n": len(test), "original_exists": orig,
                      "l0_256_cache_exists": cache}

    # quantify the two input paths on Bird (the only category with originals)
    diffs = []
    recs = LD.load_records(category="Bird", require_exists=False)
    test = [r for r in LD.select(recs, views=["120"]) if r["split"] == "test"][:20]
    for r in test:
        o = Image.open(r["image_path"]).convert("RGB")
        path_a = TF.resize(TF.to_tensor(o), [256, 256], antialias=True)          # E7-1 P00 path
        c = Image.open(E8.cache_path("l0_256", r)).convert("RGB")
        path_b = TF.resize(TF.to_tensor(c), [256, 256], antialias=True)          # cache path
        d = (path_a - path_b).abs()
        diffs.append({"rel": str(Path(r["image_path"]).relative_to(LD.DATA_ROOT)),
                      "max_abs": float(d.max()), "mean_abs": float(d.mean())})
    mx = max(x["max_abs"] for x in diffs)
    mn = float(np.mean([x["mean_abs"] for x in diffs]))

    payload = {
        "status": "BLOCKED",
        "units": ["PatchCore/Car", "PatchCore/Cube"],
        "reason": "PatchCore's frozen pipeline consumes the ORIGINAL 1024px image path; "
                  "data/m2ad/extracted/ contains only Bird. l0_256 is NOT a valid substitute.",
        "input_paths": {
            "E7_1_P00": "TF.resize(TF.to_tensor(original_1024), [256,256], antialias=True)  "
                        "[BILINEAR default]",
            "l0_256_cache": "BICUBIC 256x256 uint8 artefact (E11-P2 p2_cache_validation.md), "
                            "then a no-op resize"},
        "measured_difference_on_bird_20_images": {
            "max_abs_over_images": mx, "mean_abs_over_images": mn,
            "per_image": diffs},
        "conclusion": f"max|delta| = {mx:.6f} > 0 -> the cache is a different model input, "
                      f"so using it would be a configuration deviation (E18-P1 section 6).",
        "availability": avail,
        "options": [
            {"id": "A", "action": "download+extract M2AD car.zip / cube.zip from hf-mirror.com",
             "cost": "~14.4 GB zip each (Bird reference); 15 GB extracted each; 16 GB disk free -> "
                     "peak 29 GB with the zip kept, or ~15 GB with streaming extraction (leaves "
                     "~1 GB free, risky)",
             "time": "~20 min/category", "needs": "human approval (disk + network)"},
            {"id": "B", "action": "substitute l0_256 cache",
             "verdict": "REJECTED by measurement (see conclusion)"},
            {"id": "C", "action": "leave PatchCore official Car/Cube MISSING in the main table",
             "note": "the unified-scoring PatchCore row (E11-P2 PatchCore-emb + 1-NN-max) already "
                     "covers all 3 categories and is REUSE-complete"},
        ],
        "zero_gpu": True,
    }
    OUT.write_text(json.dumps(payload, indent=2))
    print(json.dumps({k: v for k, v in payload.items()
                      if k not in ("options", "input_paths")}, indent=2)[:1200])
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
