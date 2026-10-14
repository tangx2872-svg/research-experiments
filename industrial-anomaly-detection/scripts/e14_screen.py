#!/usr/bin/env python
"""E14 - Anomaly-Oriented Representation Family Screening.

F0 = DINOv2-B final-layer 1-NN-max (reuse E13 R0, 0 GPU).
F1 = AnomalyDINO (DINOv2 448 raw features -> L2-normalized kNN k=1 -> mean_top1p).
F2 = SoftPatch (WideResNet50 PatchCore + official WeightedGreedyCoresetSampler soft weighting).
F3 = EfficientAD (SKIP this round: download+training cost).

Frozen protocol: results/e14/e14_protocol.json
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402

import e4d1_m2ad_loader as LD  # noqa: E402
import e8_common as E8  # noqa: E402
import e7_pipelines as PL  # noqa: E402
from e11_p2_runner import raw_test_map, official_metrics, route  # noqa: E402
from e13a_screen import load_dinov2_local, prep_image  # noqa: E402

OUTDIR = ROOT / "results" / "e14"
PARTIAL = OUTDIR / "e14_partial.json"
CSV = OUTDIR / "e14_screening.csv"
E13A_CSV = ROOT / "results" / "e13a" / "e13a_screening.csv"

FAMILIES = ["F1", "F2"]


def load_f0():
    """F0 = R0 (DINOv2-B 1-NN-max) reused from E13-A."""
    out = {}
    with open(E13A_CSV) as f:
        for r in csv.DictReader(f):
            if r["config"] == "R0":
                out[r["category"]] = {"auroc": float(r["auroc"]), "ap": float(r["ap"])}
    assert set(out) == {"Bird", "Car", "Cube"}
    return out


def load_scope(category):
    recs = LD.load_records(category=category, require_exists=False)
    est = LD.estimate_scope(recs)
    bank_specs = set(est["bank_specimens"])
    bank = E8.bank_for(sorted(LD.select(recs, bank_specs), key=lambda r: r["image_path"]), 0)
    test = [r for r in LD.select(recs, views=["120"]) if r["split"] == "test"]
    return bank, test


def _label_rows(test, scores, rawmap, rel):
    rows = []
    for r, s in zip(test, scores):
        m = rawmap[rel(r)]
        rows.append({"specimen": r["specimen"], "kind": r["kind"],
                     "illumination": r["illumination"], "view": "120", "score": float(s),
                     "image_anomaly": int(m["image_anomaly"]), "detectable": m["detectable"]})
    return rows


def run_f1(category, bank, test, dev, rawmap, rel):
    """AnomalyDINO: raw DINOv2 (no LayerNorm) -> L2-norm -> 1-NN cosine -> mean_top1p."""
    model = load_dinov2_local(dev).to(dev)
    bank_r = route(bank, "l0_448")
    test_r = route(test, "l0_448")
    t0 = time.time()
    feats = []
    with torch.inference_mode():
        for i, r in enumerate(bank_r, 1):
            t = prep_image(Image.open(r["image_path"]).convert("RGB"))
            tok = model.get_intermediate_layers(torch.from_numpy(t).to(dev).unsqueeze(0))[0].squeeze()
            feats.append(tok.float().cpu().numpy())
    bank_f = np.concatenate(feats, axis=0).astype("float32")
    bank_f /= np.maximum(np.linalg.norm(bank_f, axis=1, keepdims=True), 1e-12)
    bank_t = torch.from_numpy(bank_f).to(dev)
    extract_s = time.time() - t0
    t0 = time.time()
    scores = []
    with torch.inference_mode():
        for r in test_r:
            t = prep_image(Image.open(r["image_path"]).convert("RGB"))
            tok = model.get_intermediate_layers(torch.from_numpy(t).to(dev).unsqueeze(0))[0].squeeze().float()
            tok = tok / tok.norm(dim=1, keepdim=True).clamp_min(1e-12)
            best = None
            for k in range(0, bank_f.shape[0], 65536):
                sim = tok @ bank_t[k:k + 65536].T
                m = sim.max(dim=1).values
                best = m if best is None else torch.maximum(best, m)
            dist = (1.0 - best).clamp_min(0.0)
            scores.append(PL.mean_top1p(dist.cpu().numpy()))
    score_s = time.time() - t0
    del model, bank_t
    torch.cuda.empty_cache()
    rows = _label_rows(test, scores, rawmap, rel)
    met = official_metrics(rows)
    return {"family": "F1", "metrics": met, "extract_s": round(extract_s, 1),
            "score_s": round(score_s, 1), "n_bank_patches": int(bank_f.shape[0])}


def run_f2(category, bank, test, dev, rawmap, rel):
    """SoftPatch (official): WideResNet50 PatchCore + WeightedGreedyCoresetSampler soft weighting."""
    bank_r = route(bank, "l0_256")
    test_r = route(test, "l0_256")
    res = PL.run_softpatch("nearest", bank_r, test_r, dev, "F2")
    scores = [r["score"] for r in res["rows"]]
    rows = _label_rows(test, scores, rawmap, rel)
    met = official_metrics(rows)
    return {"family": "F2", "metrics": met,
            "extract_s": round(res["stages"].get("bank_embed_s", 0), 1),
            "score_s": round(res["stages"].get("scoring_s", 0), 1),
            "coreset_size": res["stages"].get("coreset_size")}


def run_family(fam, category, bank, test, dev, rawmap, rel):
    if fam == "F1":
        return run_f1(category, bank, test, dev, rawmap, rel)
    if fam == "F2":
        return run_f2(category, bank, test, dev, rawmap, rel)
    raise ValueError(fam)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--categories", default="Bird,Car,Cube")
    ap.add_argument("--families", default="F1,F2")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    dev = torch.device("cuda")
    cats = [c.strip() for c in args.categories.split(",") if c.strip()]
    fams = [f.strip() for f in args.families.split(",") if f.strip()]
    OUTDIR.mkdir(parents=True, exist_ok=True)
    f0 = load_f0()

    print(f"[E14] cats={cats} families={fams} smoke={args.smoke} | disk {E8.free_gb():.1f} GB | "
          f"GPU {E8.gpu_util()}", flush=True)
    print(f"[E14] F0 reuse (E13 R0): " + " ".join(f"{c}={f0[c]['auroc']:.5f}" for c in cats),
          flush=True)

    partial = {"experiment": "E14", "cells": {}, "f0": f0}
    if PARTIAL.exists():
        try:
            partial.update(json.loads(PARTIAL.read_text()))
        except json.JSONDecodeError:
            pass

    t_start = time.time()
    last_out = None
    total = len(cats) * len(fams)
    done_cnt = 0
    for cat in cats:
        bank, test = load_scope(cat)
        rel = lambda r: str(Path(r["image_path"]).relative_to(LD.DATA_ROOT))
        rawmap = raw_test_map(cat)
        for fam in fams:
            key = f"{cat}|{fam}"
            if not args.smoke and key in partial.get("cells", {}):
                print(f"[E14] SKIP {key} (done)", flush=True)
                done_cnt += 1
                continue
            print(f"\n[E14] RUN {cat} {fam}", flush=True)
            try:
                out = run_family(fam, cat, bank, test, dev, rawmap, rel)
                last_out = out
            except Exception as exc:
                import traceback
                traceback.print_exc()
                print(f"[E14] {cat} {fam} FAILED: {type(exc).__name__}: {exc}", flush=True)
                if args.smoke:
                    return 1
                continue
            if not args.smoke:
                partial["cells"][key] = {k: out[k] for k in
                                         ("family", "metrics", "extract_s", "score_s")}
                tmp = PARTIAL.with_suffix(".json.tmp")
                tmp.write_text(json.dumps(partial, indent=2, default=float))
                tmp.replace(PARTIAL)
            done_cnt += 1
            el = time.time() - t_start
            per = el / max(1, done_cnt)
            eta = per * (total - done_cnt)
            m = out["metrics"]
            d_au = m["auroc"] - f0[cat]["auroc"]
            d_ap = m["ap"] - f0[cat]["ap"]
            print(f"[E14] PROGRESS {done_cnt}/{total} | elapsed={el/60:.1f}m | ETA={eta/60:.1f}m | "
                  f"{cat} {fam} AUROC={m['auroc']:.5f} dAUROC={d_au:+.5f} "
                  f"AP={m['ap']:.5f} dAP={d_ap:+.5f}", flush=True)

    if args.smoke:
        print("\n[E14] SMOKE SUMMARY", flush=True)
        o = last_out
        ok = o is not None and all(np.isfinite(x) for x in (o["metrics"]["auroc"], o["metrics"]["ap"]))
        print(f"  finite_auroc_ap: {'PASS' if ok else 'FAIL'}", flush=True)
        print(f"[E14] SMOKE {'PASS' if ok else 'FAIL'} (no checkpoint/CSV written)", flush=True)
        return 0 if ok else 1

    write_csv(partial, f0)
    print(f"\n[E14] done -> {CSV}", flush=True)
    return 0


def write_csv(partial, f0):
    w = csv.writer(open(CSV, "w", newline=""))
    w.writerow(["category", "view", "family", "auroc", "ap", "d_auroc", "d_ap", "status"])
    for cat in ("Bird", "Car", "Cube"):
        w.writerow([cat, "120", "F0", f"{f0[cat]['auroc']:.5f}", f"{f0[cat]['ap']:.5f}",
                    "+0.00000", "+0.00000", "reused_e13"])
        for fam in FAMILIES:
            cell = partial.get("cells", {}).get(f"{cat}|{fam}")
            if not cell:
                w.writerow([cat, "120", fam, "", "", "", "", "MISSING"])
                continue
            m = cell["metrics"]
            w.writerow([cat, "120", fam, f"{m['auroc']:.5f}", f"{m['ap']:.5f}",
                        f"{m['auroc']-f0[cat]['auroc']:+.5f}", f"{m['ap']-f0[cat]['ap']:+.5f}", "OK"])


if __name__ == "__main__":
    raise SystemExit(main())
