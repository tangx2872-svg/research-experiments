#!/usr/bin/env python
"""E18 - recompute official-protocol metrics for a legacy-labeled historical run.

E7-1 (git af56dd3) reported image metrics under the legacy `object_anomaly` labels
(200 Good / 500 NG at view120). The E11+ line uses the official protocol:
`image_anomaly` + `detectable != False`. The per-image SCORES are label-independent,
so the official AUROC/AP can be recomputed exactly from the saved per-image CSV.

Verified preconditions (asserted below):
  * bank of E7-1 P00 == E8.bank_for(pool, 0)  (bit-identical, sha256 33381cd3004bfe68)
  * test set == all 700 view120 images, seed 0
  * (specimen, illumination) -> meta record is a unique 1:1 mapping (700/700)

Zero GPU. Writes results/e18/recomputed/patchcore_bird_official.json
"""
from __future__ import annotations

import csv
import hashlib
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np  # noqa: E402
from sklearn.metrics import roc_auc_score, average_precision_score  # noqa: E402

import e4d1_m2ad_loader as LD  # noqa: E402
import e8_common as E8  # noqa: E402

CAT = "Bird"
P00_CSV = ROOT / "results" / "e7_1" / "raw" / "P00_per_image.csv"
P00_JSON = ROOT / "results" / "e7_1" / "raw" / "P00.json"
OUT = ROOT / "results" / "e18" / "recomputed" / "patchcore_bird_official.json"


def main() -> int:
    # --- precondition 1: bank identity
    recs = LD.load_records(category=CAT, require_exists=False)
    est = LD.estimate_scope(recs)
    pool = sorted(LD.select(recs, set(est["bank_specimens"])), key=lambda r: r["image_path"])
    assert len(pool) == 1200, f"pool {len(pool)} != 1200"
    bank_paths = [r["image_path"] for r in E8.bank_for(pool, 0)]
    sha = hashlib.sha256("\n".join(bank_paths).encode()).hexdigest()[:16]
    recon = sorted(random.Random(E8.BANK_SUBSET_SEED).sample(range(len(pool)), 300))
    assert [pool[i]["image_path"] for i in recon] == bank_paths, "E7-1 bank != E8.bank_for(pool,0)"
    assert sha == "33381cd3004bfe68", sha

    # --- precondition 2/3: 1:1 mapping to view120 metadata
    meta = json.load(open(ROOT / "data" / "m2ad" / "jsons_extract" / "meta_unsupervised.json"))
    key = {(r["object_name"], r["illumination"]): r
           for r in meta["test"][CAT] if r["view"] == "120"}
    rows = list(csv.DictReader(open(P00_CSV)))
    assert len(rows) == 700 and len(key) == 700
    unmatched = [r for r in rows if (r["specimen"], r["illumination"]) not in key]
    assert not unmatched, f"{len(unmatched)} unmatched"

    # --- official metrics from label-independent scores
    kept = [r for r in rows if key[(r["specimen"], r["illumination"])]["detectable"] != False]
    y = np.array([key[(r["specimen"], r["illumination"])]["image_anomaly"] for r in kept], int)
    s = np.array([float(r["score"]) for r in kept])
    auroc = float(roc_auc_score(y, s))
    ap = float(average_precision_score(y, s))

    # --- legacy metrics reproduced exactly (sanity that the mapping is the same run)
    y_leg = np.array([0 if r["kind"] == "Good" else 1 for r in rows], int)
    s_leg = np.array([float(r["score"]) for r in rows])
    leg_auroc = float(roc_auc_score(y_leg, s_leg))
    leg_ap = float(average_precision_score(y_leg, s_leg))
    hist = json.load(open(P00_JSON))["metrics"]
    assert abs(leg_auroc - hist["image_auroc"]) <= 5e-6, (leg_auroc, hist["image_auroc"])
    assert abs(leg_ap - hist["image_aupr"]) <= 5e-6

    out = {
        "method": "PatchCore (official scoring)", "category": CAT, "view": "120", "seed": 0,
        "bank_n": 300, "test_n": 700,
        "impl": "e7_pipelines.run_patchcore (wide_resnet50_2 layer2+layer3, input 256, "
                "coreset 0.1, k=9, max-pool)",
        "source_run": "results/e7_1/raw/P00.json (git af56dd3)",
        "source_scores": "results/e7_1/raw/P00_per_image.csv (label-independent per-image scores)",
        "official_protocol": {
            "labels": "image_anomaly", "filter": "detectable != False",
            "n_kept": len(kept), "n_ok": int((y == 0).sum()), "n_ng": int((y == 1).sum()),
            "n_filtered": len(rows) - len(kept), "auroc": auroc, "ap": ap},
        "legacy_protocol_reproduced": {
            "labels": "object_anomaly (200 Good / 500 NG)", "n": len(rows),
            "auroc": leg_auroc, "ap": leg_ap,
            "historical_reported_auroc": hist["image_auroc"],
            "historical_reported_ap": hist["image_aupr"]},
        "note": "Same bank (sha256 33381cd3004bfe68), same test set, same seed; only the label "
                "definition and the detectable filter differ. Scores are label-independent, so the "
                "official numbers are exact, not approximate.",
        "recomputed_zero_gpu": True,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2))
    print(json.dumps({"official": out["official_protocol"],
                      "legacy_reproduced": out["legacy_protocol_reproduced"]}, indent=2))
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
