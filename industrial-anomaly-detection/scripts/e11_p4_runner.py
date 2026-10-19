#!/usr/bin/env python
"""E11-P4-A/B - cross-view (090 vs 120) frozen-backbone diagnostic.

Same 300-bank (view-mixed) as E11-P2, tested on view090 and view120. Backbones: ResNet (PatchCore
embedding), DINOv2-S, DINOv2-B; unified 1-NN-max. Saves per-image rows (with object_anomaly) for P4-B.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

import e4d1_m2ad_loader as LD  # noqa: E402
import e8_common as E8  # noqa: E402
import e10_runner as R10  # noqa: E402
from e11_p2_runner import (load_dinov2, extract, nn_image_scores, route,  # noqa: E402
                           raw_test_map, official_metrics, BACKBONES)

VIEWS = ("090", "120")
BBS = ("resnet", "dinov2s", "dinov2b")


def build_p4_scope(category: str):
    recs = LD.load_records(category=category, require_exists=False)
    est = LD.estimate_scope(recs)
    bank_specs = set(est["bank_specimens"])
    bank = E8.bank_for(sorted(LD.select(recs, bank_specs), key=lambda r: r["image_path"]), 0)
    tests = {v: [r for r in LD.select(recs, views=[v]) if r["split"] == "test"] for v in VIEWS}
    return bank, tests, raw_test_map(category)


def run_p4(category: str, dev) -> dict:
    bank, tests, rawmap = build_p4_scope(category)
    rel = lambda r: str(Path(r["image_path"]).relative_to(LD.DATA_ROOT))
    out = {"category": category, "backbones": {}}
    for bb in BBS:
        kind = BACKBONES[bb]["input_kind"]
        bank_r = route(bank, kind)
        bank_paths = [r["image_path"] for r in bank_r]
        t0 = time.time()
        if bb == "resnet":
            tm = R10.fit_patchcore(bank_r, dev, f"p4_res_{category}", seed=0)
            model = tm
        else:
            model = load_dinov2(BACKBONES[bb]["hub"], dev)
        bank_f = extract(bb, model, bank_paths, dev, kind)
        entry = {"n_bank": len(bank), "views": {}}
        for v in VIEWS:
            test = tests[v]
            test_r = route(test, kind)
            test_f = extract(bb, model, [r["image_path"] for r in test_r], dev, kind)
            scores = nn_image_scores(bank_f, test_f, dev)
            rows = []
            for r, s in zip(test, scores):
                m = rawmap[rel(r)]
                rows.append({"specimen": r["specimen"], "kind": r["kind"],
                             "illumination": r["illumination"], "view": v, "score": float(s),
                             "image_anomaly": int(m["image_anomaly"]),
                             "object_anomaly": int(m["object_anomaly"]),
                             "detectable": m["detectable"]})
            met = official_metrics(rows)
            entry["views"][v] = {"metrics": met, "rows": rows,
                                 "peak_vram_mb": round(torch.cuda.max_memory_allocated() / 2**20, 1)}
        entry["runtime_s"] = round(time.time() - t0, 1)
        out["backbones"][bb] = entry
        print(f"    {bb}: " + "  ".join(
            f"view{v}={entry['views'][v]['metrics']['auroc']:.5f}" for v in VIEWS), flush=True)
        if bb != "resnet":
            del model
        torch.cuda.empty_cache()
    return out


def main() -> int:
    import argparse, csv
    ap = argparse.ArgumentParser()
    ap.add_argument("--categories", default="Bird,Car,Cube")
    args = ap.parse_args()
    dev = torch.device("cuda")
    cats = [c.strip() for c in args.categories.split(",") if c.strip()]
    outdir = ROOT / "results" / "e11"
    outdir.mkdir(parents=True, exist_ok=True)

    w = csv.writer(open(outdir / "p4_view_comparison.csv", "w", newline=""))
    w.writerow(["category", "view", "backbone", "auroc", "ap", "n_ok", "n_ng", "runtime_s",
                "peak_vram_mb", "status"])
    raw = {}
    for cat in cats:
        print(f"\n[E11-P4] RUN {cat}", flush=True)
        try:
            out = run_p4(cat, dev)
        except Exception as exc:
            import traceback; traceback.print_exc()
            out = {"status": "FAILED", "reason": f"{type(exc).__name__}: {exc}"}
        if out.get("status") == "FAILED":
            print(f"    FAILED: {out['reason'][:100]}", flush=True)
            continue
        raw[cat] = out
        for bb in BBS:
            entry = out["backbones"][bb]
            for v in VIEWS:
                m = entry["views"][v]["metrics"]
                w.writerow([cat, v, bb, f"{m['auroc']:.5f}", f"{m['ap']:.5f}",
                            m["n_ok"], m["n_ng"], entry["runtime_s"],
                            entry["views"][v]["peak_vram_mb"], "OK"])
    json.dump(raw, open(outdir / "p4_raw_results.json", "w"), indent=2, default=str)
    print(f"\n[E11-P4] done -> {outdir/'p4_view_comparison.csv'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
