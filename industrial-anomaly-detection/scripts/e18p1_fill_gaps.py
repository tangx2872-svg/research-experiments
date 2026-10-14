#!/usr/bin/env python
"""E18-P1 - fill the 4 missing baseline cells under the frozen official protocol.

Units (exactly 4, no more):
  1. PatchCore official / Car   / seed0   (e7_pipelines.run_patchcore, E7-1 P00 config)
  2. PatchCore official / Cube  / seed0
  3. AnomalyDINO       / Cube   / seed0   (e14_screen.run_f1, verbatim)
  4. SoftPatch         / Cube   / seed0   (e14_screen.run_f2, verbatim)

No method configuration is changed. History is never overwritten: everything is
written into results/e18/filled/. Official labels only (image_anomaly + detectable).
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

import e4d1_m2ad_loader as LD  # noqa: E402
import e7_common as E7  # noqa: E402
import e8_common as E8  # noqa: E402
import e7_pipelines as PL  # noqa: E402
import e14_screen as E14  # noqa: E402
from e11_p2_runner import raw_test_map, official_metrics  # noqa: E402

OUTDIR = ROOT / "results" / "e18" / "filled"
PERIMG = OUTDIR / "per_image"
PARTIAL = ROOT / "results" / "e18" / "e18_partial.json"
SUMMARY = ROOT / "results" / "e18" / "filled_summary.csv"

BANK_SHA = {"Bird": "33381cd3004bfe68", "Car": "1d11f941de724bf8", "Cube": "3b0b7ea7b185e727"}
# detected-filtered label distribution, == E16 F0 reference (audit.md section 2)
EXPECT = {"Bird": (320, 333), "Car": (300, 356), "Cube": (260, 418)}
EXPECT_TEST_N = {"Bird": 700, "Car": 700, "Cube": 700}
EXPECT_FILTERED = {"Bird": 47, "Car": 44, "Cube": 22}
E16_F0 = ROOT / "results" / "e16" / "e16_partial.json"
UNITS = [("PatchCore", "Car"), ("PatchCore", "Cube"),
         ("AnomalyDINO", "Cube"), ("SoftPatch", "Cube")]


def e7_scope(category):
    """E7-1's frozen scope construction (E7.load_scope, parameterised by category)."""
    recs = LD.load_records(category=category, require_exists=False)
    est = LD.estimate_scope(recs)
    bank_source = sorted(LD.select(recs, set(est["bank_specimens"])), key=lambda r: r["image_path"])
    n_sub = max(1, int(E7.BANK_SUBSET_FRACTION * len(bank_source)))
    pick = sorted(random.Random(E7.BANK_SUBSET_SEED).sample(range(len(bank_source)), n_sub))
    bank = [bank_source[i] for i in pick]
    test = sorted([r for r in LD.select(recs, views=[E7.FROZEN_VIEW]) if r["split"] == "test"],
                  key=lambda r: r["image_path"])
    return bank, test


def verify_scope(category, bank, test, tag):
    sha = hashlib.sha256("\n".join(r["image_path"] for r in bank).encode()).hexdigest()[:16]
    if sha != BANK_SHA[category]:
        raise RuntimeError(f"BANK_HASH_MISMATCH {category}: {sha} != {BANK_SHA[category]}")
    if len(test) != 700:
        raise RuntimeError(f"TEST_COUNT_MISMATCH {category}: {len(test)} != 700")
    if len({(r["specimen"], r["illumination"]) for r in test}) != len(test):
        raise RuntimeError(f"TEST_MAPPING_DUPLICATE {category}")
    print(f"    [{tag}] bank sha256[:16]={sha} OK | test n={len(test)} unique-key OK", flush=True)


def official_rows(test, scores, category, tag):
    rawmap = raw_test_map(category)
    rows = []
    for r, s in zip(test, scores):
        rel = str(Path(r["image_path"]).relative_to(LD.DATA_ROOT))
        m = rawmap.get(rel)
        if m is None:
            raise RuntimeError(f"TEST_MAPPING_MISSING {category} {rel}")
        rows.append({"specimen": r["specimen"], "kind": r["kind"], "illumination": r["illumination"],
                     "score": float(s), "image_anomaly": int(m["image_anomaly"]),
                     "detectable": m["detectable"]})
    return rows


def recompute_assert(rows, met, category, tag):
    from sklearn.metrics import roc_auc_score, average_precision_score
    kept = [r for r in rows if r["detectable"] != False]
    y = np.array([r["image_anomaly"] for r in kept], int)
    s = np.array([r["score"] for r in kept], float)
    a2, p2 = float(roc_auc_score(y, s)), float(average_precision_score(y, s))
    if abs(a2 - met["auroc"]) >= 1e-12 or abs(p2 - met["ap"]) >= 1e-12:
        raise RuntimeError(f"RECOMPUTE_MISMATCH {category} {tag}: {a2} vs {met['auroc']}")
    n_all, n_ok, n_ng = len(rows), int((y == 0).sum()), int((y == 1).sum())
    n_filt = n_all - len(kept)
    if (n_ok, n_ng) != EXPECT[category]:
        raise RuntimeError(f"LABEL_DIST_MISMATCH {category}: (ok,ng)={(n_ok,n_ng)} "
                           f"!= F0 ref {EXPECT[category]}")
    if n_all != EXPECT_TEST_N[category] or n_filt != EXPECT_FILTERED[category]:
        raise RuntimeError(f"TEST_SCOPE_MISMATCH {category}: n={n_all} filtered={n_filt} "
                           f"!= ({EXPECT_TEST_N[category]}, {EXPECT_FILTERED[category]})")
    good = np.array([r["score"] for r in rows if r["kind"] == "Good"], float)
    ng = np.array([r["score"] for r in rows if r["kind"] == "NG"], float)
    if not (ng.mean() > good.mean()):
        raise RuntimeError(f"SCORE_DIRECTION {category} {tag}")
    print(f"    [{tag}] recompute OK (dAUROC={abs(a2-met['auroc']):.1e}, dAP={abs(p2-met['ap']):.1e})"
          f" | n={n_all} filtered={n_filt} kept={n_ok}+{n_ng} == F0 ref {EXPECT[category]}"
          f" | direction OK (NG {ng.mean():.4f} > Good {good.mean():.4f})", flush=True)


def _capture(fn, *a, **kw):
    """Run an E14 pipeline function verbatim while capturing the rows it used."""
    box = {}
    orig = E14.official_metrics

    def spy(rows):
        box["rows"] = rows
        return orig(rows)

    E14.official_metrics = spy
    try:
        out = fn(*a, **kw)
    finally:
        E14.official_metrics = orig
    if "rows" not in box:
        raise RuntimeError("capture failed: pipeline did not call official_metrics")
    return out, box["rows"]


def run_unit(method, category, dev):
    t0 = time.time()
    if method == "PatchCore":
        bank, test = e7_scope(category)
        verify_scope(category, bank, test, "PatchCore")
        res = PL.run_patchcore(bank, test, dev, f"e18p1_pc_{category}", want_maps=False)
        rows = official_rows(test, [r["score"] for r in res["rows"]], category, "PatchCore")
        cfg = {"impl": "e7_pipelines.run_patchcore", "backbone": "wide_resnet50_2 layer2+layer3",
               "input_size": 256, "coreset_sampling_ratio": 0.1, "num_neighbors": 9,
               "aggregation": "max-pool (official)", "source_run": "E7-1 P00 config"}
        stages = res["stages"]
    elif method == "AnomalyDINO":
        bank, test = E14.load_scope(category)
        verify_scope(category, bank, test, "AnomalyDINO")
        rawmap = raw_test_map(category)
        rel = lambda r: str(Path(r["image_path"]).relative_to(LD.DATA_ROOT))
        out, rows = _capture(E14.run_f1, category, bank, test, dev, rawmap, rel)
        cfg = {"impl": "e14_screen.run_f1", "backbone": "DINOv2-B raw (no LayerNorm)",
               "resolution": 448, "scoring": "L2 kNN k=1, image score = mean_top1p (official)",
               "source_run": "E14 F1 config"}
        stages = {"extract_s": out["extract_s"], "scoring_s": out["score_s"]}
    elif method == "SoftPatch":
        bank, test = E14.load_scope(category)
        verify_scope(category, bank, test, "SoftPatch")
        rawmap = raw_test_map(category)
        rel = lambda r: str(Path(r["image_path"]).relative_to(LD.DATA_ROOT))
        out, rows = _capture(E14.run_f2, category, bank, test, dev, rawmap, rel)
        cfg = {"impl": "e14_screen.run_f2 / e7_pipelines.run_softpatch",
               "backbone": "wide_resnet50_2 PatchCore (l0_256)",
               "scoring": "official WeightedGreedyCoresetSampler soft weighting (nearest)",
               "source_run": "E14 F2 config"}
        stages = {"extract_s": out["extract_s"], "scoring_s": out["score_s"],
                  "coreset_size": out.get("coreset_size")}
    else:
        raise ValueError(method)

    met = official_metrics([{"score": r["score"], "image_anomaly": r["image_anomaly"],
                             "detectable": r["detectable"]} for r in rows])
    recompute_assert(rows, met, category, method)
    return {"method": method, "category": category, "seed": 0, "metrics": met,
            "config": cfg, "stages": stages, "wall_s": round(time.time() - t0, 1),
            "rows": rows}


def dump_unit(res):
    tag = f"{res['method'].lower()}_{res['category'].lower()}"
    csv_path = PERIMG / f"{tag}_per_image.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["specimen", "kind", "illumination", "score", "image_anomaly", "detectable"])
        for r in res["rows"]:
            w.writerow([r["specimen"], r["kind"], r["illumination"], repr(r["score"]),
                        r["image_anomaly"], r["detectable"]])
    payload = {k: v for k, v in res.items() if k != "rows"}
    payload["per_image_csv"] = str(csv_path)
    payload["n_rows"] = len(res["rows"])
    json_path = OUTDIR / f"{tag}.json"
    json_path.write_text(json.dumps(payload, indent=2, default=float))
    print(f"    -> {json_path.name} + {csv_path.name} ({len(res['rows'])} rows)", flush=True)
    return payload


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--units", default="")
    args = ap.parse_args()
    dev = torch.device("cuda")
    OUTDIR.mkdir(parents=True, exist_ok=True)
    PERIMG.mkdir(parents=True, exist_ok=True)
    units = UNITS
    if args.units:
        want = {u.strip() for u in args.units.split(",") if u.strip()}
        units = [u for u in UNITS if f"{u[0]}|{u[1]}" in want]

    partial = {"experiment": "E18-P1", "units": {}}
    if PARTIAL.exists():
        try:
            old = json.loads(PARTIAL.read_text())
            if old.get("experiment") == "E18-P1":
                partial["units"] = old.get("units", {})
        except json.JSONDecodeError:
            pass

    print(f"[E18-P1] units={[f'{m}/{c}' for m, c in units]} | disk {E8.free_gb():.1f} GB | "
          f"seed 0 | official labels (image_anomaly + detectable)", flush=True)
    t_start = time.time()
    total, done = len(units), 0
    for method, cat in units:
        key = f"{method}|{cat}"
        if key in partial["units"]:
            print(f"[E18-P1] SKIP {key} (already done)", flush=True)
            done += 1
            continue
        print(f"\n[E18-P1] RUN {method} / {cat} / seed0", flush=True)
        t_unit = time.time()
        res = run_unit(method, cat, dev)
        payload = dump_unit(res)
        partial["units"][key] = payload
        tmp = PARTIAL.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(partial, indent=2, default=float))
        tmp.replace(PARTIAL)
        done += 1
        el = time.time() - t_start
        eta = el / done * (total - done)
        m = res["metrics"]
        msg = (f"[E18-P1] PROGRESS {done}/{total} | {100*done/total:.1f}% | current={method}/{cat} | "
               f"elapsed={el/60:.1f}m ETA={eta/60:.1f}m | AUROC={m['auroc']:.5f} AP={m['ap']:.5f} "
               f"n={m['n_ok']+m['n_ng']} | unit_wall={time.time()-t_unit:.0f}s")
        print(msg, flush=True)
        print(f"[E18-P1] HEARTBEAT ok unit={key}", flush=True)

    write_summary(partial)
    print(f"\n[E18-P1] done -> {SUMMARY}", flush=True)
    return 0


def write_summary(partial):
    f0 = json.loads(E16_F0.read_text())["cells"]
    with open(SUMMARY, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["method", "category", "seed", "AUROC", "AP", "n_ok", "n_ng", "n_filtered",
                    "F0_AUROC", "F0_AP", "dAUROC_vs_F0", "dAP_vs_F0", "wall_s", "status"])
        for method, cat in UNITS:
            cell = partial["units"].get(f"{method}|{cat}")
            if not cell:
                w.writerow([method, cat, 0, "", "", "", "", "", "", "", "", "", "", "MISSING"])
                continue
            m = cell["metrics"]
            f0c = f0[f"{cat}|0"]["f0"]
            w.writerow([method, cat, 0, f"{m['auroc']:.10f}", f"{m['ap']:.10f}",
                        m["n_ok"], m["n_ng"], m["n_filtered"],
                        f"{f0c['auroc']:.10f}", f"{f0c['ap']:.10f}",
                        f"{m['auroc']-f0c['auroc']:+.5f}", f"{m['ap']-f0c['ap']:+.5f}",
                        cell["wall_s"], "OK"])


if __name__ == "__main__":
    raise SystemExit(main())
