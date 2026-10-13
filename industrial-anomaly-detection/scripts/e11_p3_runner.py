#!/usr/bin/env python
"""E11-P3.1 - Per-Illumination Normal Median Centering (frozen DINOv2-B, 1-NN-max).

Calibration = 20 non-bank train normal specimens (2400 imgs), specimen-disjoint from the 10-specimen
memory bank and the 70-specimen test. Per-illum median is fitted ONLY on that calibration set.
Controls: A raw, B per-illum median-centered, C global-median (sanity: AUROC == A).
"""
from __future__ import annotations

import json
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

import e4d1_m2ad_loader as LD  # noqa: E402
import e8_common as E8  # noqa: E402
from e11_p2_runner import (load_dinov2, extract, nn_image_scores, route,  # noqa: E402
                           raw_test_map, official_metrics, BACKBONES)

BB = "dinov2b"


def build_p3_scope(category: str):
    recs = LD.load_records(category=category, require_exists=False)
    est = LD.estimate_scope(recs)
    bank_specs = set(est["bank_specimens"])
    train_specs = sorted({r["specimen"] for r in recs if r["split"] == "train"})
    calib_specs = set(train_specs) - bank_specs
    bank = E8.bank_for(sorted(LD.select(recs, bank_specs), key=lambda r: r["image_path"]), 0)
    calib = [r for r in recs if r["split"] == "train" and r["specimen"] in calib_specs]
    test = [r for r in LD.select(recs, views=["120"]) if r["split"] == "test"]
    return bank, calib, test


def per_illum_metrics(rows: list, med: dict) -> list:
    g = defaultdict(lambda: {"norm": [], "anom": []})
    for r in rows:
        key = "anom" if (r["image_anomaly"] == 1 and r["detectable"] != False) else "norm"
        g[r["illumination"]][key].append(r["score"])
    out = []
    for ill in sorted(g):
        n = np.array(g[ill]["norm"]); a = np.array(g[ill]["anom"])
        rec = {"illumination": ill, "n_normal": int(n.size), "n_anom": int(a.size),
               "normal_median_raw": float(np.median(n)) if n.size else None,
               "normal_mean_raw": float(n.mean()) if n.size else None,
               "anom_mean_raw": float(a.mean()) if a.size else None,
               "anom_mean_cal": (float(a.mean()) - med.get(ill, 0.0)) if a.size else None,
               "median_used": med.get(ill, 0.0)}
        if n.size and a.size:
            y = np.concatenate([np.zeros(n.size), np.ones(a.size)])
            if len(set(y.tolist())) == 2:
                rec["auroc_raw"] = float(roc_auc_score(y, np.concatenate([n, a])))
                rec["auroc_cal"] = float(roc_auc_score(y, np.concatenate([n, a]) - med.get(ill, 0.0)))
        out.append(rec)
    return out


def run_p3(category: str, dev):
    bank, calib, test = build_p3_scope(category)
    bank_r, calib_r, test_r = route(bank, "l0_448"), route(calib, "l0_448"), route(test, "l0_448")
    model = load_dinov2(BACKBONES[BB]["hub"], dev)
    t0 = time.time()
    bank_f = extract(BB, model, [r["image_path"] for r in bank_r], dev, "l0_448")
    calib_f = extract(BB, model, [r["image_path"] for r in calib_r], dev, "l0_448")
    calib_scores = nn_image_scores(bank_f, calib_f, dev)
    test_f = extract(BB, model, [r["image_path"] for r in test_r], dev, "l0_448")
    test_scores = nn_image_scores(bank_f, test_f, dev)
    runtime = time.time() - t0

    calib_by_ill = defaultdict(list)
    for r, s in zip(calib, calib_scores):
        calib_by_ill[r["illumination"]].append(s)
    med = {ill: float(np.median(v)) for ill, v in calib_by_ill.items()}
    global_med = float(np.median(calib_scores))

    rel = lambda r: str(Path(r["image_path"]).relative_to(LD.DATA_ROOT))
    rawmap = raw_test_map(category)
    rows = []
    for r, s in zip(test, test_scores):
        m = rawmap[rel(r)]
        rows.append({"specimen": r["specimen"], "kind": r["kind"], "illumination": r["illumination"],
                     "view": r["view"], "score_raw": float(s),
                     "score_cal": float(s - med[r["illumination"]]),
                     "score_global": float(s - global_med),
                     "image_anomaly": int(m["image_anomaly"]), "detectable": m["detectable"]})

    met = {"raw": official_metrics([dict(r, score=r["score_raw"]) for r in rows]),
           "calibrated": official_metrics([dict(r, score=r["score_cal"]) for r in rows]),
           "global": official_metrics([dict(r, score=r["score_global"]) for r in rows])}
    diag = per_illum_metrics([dict(r, score=r["score_raw"]) for r in rows], med)
    norm_medians = [d["normal_median_raw"] for d in diag if d["normal_median_raw"] is not None]
    spread_raw = max(norm_medians) - min(norm_medians) if norm_medians else None

    return {"status": "OK", "category": category, "backbone": BB, "view": "120",
            "n_bank": len(bank), "n_calib": len(calib), "n_test": len(test),
            "per_illum_median": med, "global_median": global_med,
            "runtime_s": round(runtime, 1),
            "peak_vram_mb": round(torch.cuda.max_memory_allocated() / 2**20, 1),
            "metrics": met, "per_illumination": diag,
            "normal_median_spread_raw": round(spread_raw, 6) if spread_raw is not None else None,
            "rows": rows}


def main() -> int:
    import argparse, csv
    ap = argparse.ArgumentParser()
    ap.add_argument("--categories", default="Bird,Car,Cube")
    args = ap.parse_args()
    dev = torch.device("cuda")
    cats = [c.strip() for c in args.categories.split(",") if c.strip()]
    outdir = ROOT / "results" / "e11"
    outdir.mkdir(parents=True, exist_ok=True)

    w = csv.writer(open(outdir / "p3_results.csv", "w", newline=""))
    w.writerow(["category", "view", "backbone", "condition", "train_budget", "calib_budget",
                "n_ok", "n_ng", "auroc", "ap", "runtime", "peak_vram", "status"])
    raw = {}
    for cat in cats:
        print(f"\n[E11-P3.1] RUN {cat}", flush=True)
        try:
            out = run_p3(cat, dev)
        except Exception as exc:
            import traceback; traceback.print_exc()
            out = {"status": "FAILED", "reason": f"{type(exc).__name__}: {exc}"}
        if out["status"] == "OK":
            raw[cat] = out
            for cond in ("raw", "calibrated", "global"):
                m = out["metrics"][cond]
                w.writerow([cat, "120", "dinov2b", cond, out["n_bank"], out["n_calib"],
                            m["n_ok"], m["n_ng"], f"{m['auroc']:.5f}", f"{m['ap']:.5f}",
                            out["runtime_s"], out["peak_vram_mb"], "OK"])
            print(f"    raw={out['metrics']['raw']['auroc']:.5f}  "
                  f"calibrated={out['metrics']['calibrated']['auroc']:.5f}  "
                  f"global={out['metrics']['global']['auroc']:.5f}  "
                  f"spread_raw={out['normal_median_spread_raw']}", flush=True)
        else:
            w.writerow([cat, "120", "dinov2b", "raw", "", "", "", "", "", "", "", "", "FAILED"])
            print(f"    FAILED: {out.get('reason','')[:100]}", flush=True)
    json.dump(raw, open(outdir / "p3_raw_results.json", "w"), indent=2, default=str)
    print(f"\n[E11-P3.1] done -> {outdir/'p3_results.csv'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
