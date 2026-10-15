#!/usr/bin/env python
"""E21 - M2AD Multi-View Generalization Validation.

Frozen protocol: results/e21/protocol.md

Design (frozen):
  - categories Bird/Car/Cube x views {090,120,240,330} x seeds {0[,1,2]}
  - ONE adapter per (category, seed) trained on the frozen 300-bank (identical code path
    to e16_qualify.run_cell); the same trained adapter is then evaluated on all 4 views.
  - view120 is evaluated as an in-run REPRODUCTION CHECK against E16/E18/E19 history
    (tolerance 1e-6); reported view120 values are REUSED from history (source=REUSED_VERIFIED).
  - views 090/240/330 are new (source=NEW).

Reuses (imports only; NO historical runner is modified):
  e4d1_m2ad_loader (records/scope), e8_common (bank rule, cache_path),
  e11_p2_runner (raw_test_map, official_metrics, route, nn_image_scores),
  e13a_screen (model load, prep, adapters, forward, l2norm, _rows logic),
  e13b_screen.train_r1 (identical trainer/hyperparams).

Outputs: results/e21/{e21_partial.json, round1_seed0.csv, data_audit.csv}
Atomic partial writes + resume. Repro failure => STOP file + exit 2.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
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
from e11_p2_runner import raw_test_map, official_metrics, route, nn_image_scores  # noqa: E402
from e13a_screen import (load_dinov2_local, freeze_backbone, LinearAdapter,  # noqa: E402
                         adapted_feat_adapter, prep_image, forward_frozen_batch,
                         l2norm, D, N_PATCH, AUG_LIST, TRAIN_N, VAL_N)
from e13b_screen import train_r1  # noqa: E402

OUTDIR = ROOT / "results" / "e21"
PARTIAL = OUTDIR / "e21_partial.json"
R1_CSV = OUTDIR / "round1_seed0.csv"
AUDIT_CSV = OUTDIR / "data_audit.csv"
STOP_FILE = OUTDIR / "STOP_repro_fail.txt"

VIEWS = ["090", "120", "240", "330"]
TOL = 1e-6
LABEL_PROTOCOL = "official_image_anomaly|detectable!=False"
E16_PARTIAL = ROOT / "results" / "e16" / "e16_partial.json"

# S1/S2 target values as stated in the E21 execution protocol (cross-checked against
# e16_partial at runtime; mismatch > 1e-9 aborts before GPU work).
S1_AUROC_TARGET = {"Bird": 0.74725, "Car": 0.78552, "Cube": 0.85410}
S2_AUROC_TARGET = {"Bird": 0.75786, "Car": 0.80511, "Cube": 0.87010}


def sha256_of(seq) -> str:
    h = hashlib.sha256()
    for s in seq:
        h.update(str(s).encode())
        h.update(b"\n")
    return h.hexdigest()


def bank_hash(bank) -> str:
    return sha256_of([r["image_path"] for r in bank])


def adapter_hash(adapter) -> str:
    return hashlib.sha256(adapter.W.detach().cpu().numpy().astype("float32").tobytes()).hexdigest()


def rows21(test, scores, rawmap, rel, view):
    """Row builder with the correct view tag (e13a _rows hardcodes '120'; metrics are view-agnostic)."""
    rows = []
    for r, s in zip(test, scores):
        m = rawmap[rel(r)]
        rows.append({"specimen": r["specimen"], "kind": r["kind"], "illumination": r["illumination"],
                     "view": view, "score": float(s), "image_anomaly": int(m["image_anomaly"]),
                     "detectable": m["detectable"]})
    return rows


# ---------------------------------------------------------------- data audit
def data_audit():
    """3 cats x 4 views: raw/kept/good/anomaly/illum coverage; view120 vs history."""
    rows = []
    meta = json.load(open(ROOT / "data/m2ad/jsons_extract/meta_unsupervised.json"))
    hist = {"Bird": (700, 653, 320, 333), "Car": (700, 656, 300, 356), "Cube": (700, 678, 260, 418)}
    ok_all = True
    for cat in ("Bird", "Car", "Cube"):
        for v in VIEWS:
            ents = [e for e in meta["test"][cat] if e["view"] == v]
            kept = [e for e in ents if e["detectable"] is not False]
            y = [int(e["image_anomaly"]) for e in kept]
            n_ok = sum(1 for x in y if x == 0)
            n_ng = sum(1 for x in y if x == 1)
            ills = sorted({e["illumination"] for e in kept})
            # cache presence
            n_cache = 0
            cache_base = ROOT / "data/m2ad/e7_cache/l0_448" / cat
            for grp in ("Good", "NG"):
                d = cache_base / grp
                if d.is_dir():
                    for spec in os.listdir(d):
                        p = d / spec
                        if p.is_dir():
                            n_cache += len([f for f in os.listdir(p) if f.startswith(f"A{v}_")])
            row = {"category": cat, "view": v, "raw": len(ents), "kept": len(kept),
                   "good": n_ok, "anomaly": n_ng, "illum_groups": len(ills),
                   "illum_list": "|".join(ills), "cache_l0_448": n_cache,
                   "has_good_and_anomaly": int(n_ok > 0 and n_ng > 0)}
            if v == "120":
                h_raw, h_kept, h_ok, h_ng = hist[cat]
                row["hist_raw"], row["hist_kept"], row["hist_good"], row["hist_anomaly"] = h_raw, h_kept, h_ok, h_ng
                row["matches_history"] = int(len(ents) == h_raw and len(kept) == h_kept
                                             and n_ok == h_ok and n_ng == h_ng)
                if not row["matches_history"]:
                    ok_all = False
            if row["has_good_and_anomaly"] != 1 or len(ills) != 10:
                ok_all = False
            rows.append(row)
    OUTDIR.mkdir(parents=True, exist_ok=True)
    cols = ["category", "view", "raw", "kept", "good", "anomaly", "illum_groups", "illum_list",
            "cache_l0_448", "has_good_and_anomaly", "hist_raw", "hist_kept", "hist_good",
            "hist_anomaly", "matches_history"]
    with open(AUDIT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in cols})
    print(f"[E21 audit] wrote {AUDIT_CSV} | all_checks_pass={ok_all}", flush=True)
    for r in rows:
        print(f"  {r['category']} v{r['view']}: raw={r['raw']} kept={r['kept']} "
              f"good={r['good']} anomaly={r['anomaly']} illum={r['illum_groups']} "
              f"cache={r['cache_l0_448']}", flush=True)
    return ok_all


# ---------------------------------------------------------------- one cell
def run_cell_multiview(cat: str, seed: int, dev, smoke=False):
    """Train ONE frozen R1-ILA for (cat, seed); evaluate on all 4 views.

    Bank/aug/training code path is line-for-line identical to e16_qualify.run_cell.
    """
    t0 = time.time()
    torch.cuda.reset_peak_memory_stats()
    recs = LD.load_records(category=cat, require_exists=False)
    est = LD.estimate_scope(recs)
    bank_specs = set(est["bank_specimens"])
    pool = sorted(LD.select(recs, bank_specs), key=lambda r: r["image_path"])
    bank = E8.bank_for(pool, seed)
    bank_r = route(bank, "l0_448")
    rel = lambda r: str(Path(r["image_path"]).relative_to(LD.DATA_ROOT))
    rawmap = raw_test_map(cat)

    model = load_dinov2_local(dev).to(dev)
    freeze_backbone(model)

    n_bank = min(60, len(bank_r)) if smoke else len(bank_r)
    bank_r, bank = bank_r[:n_bank], bank[:n_bank]

    bank_tensors = np.stack([prep_image(Image.open(r["image_path"]).convert("RGB")) for r in bank_r])
    aug_tensors = np.empty((len(bank_r), len(AUG_LIST), 3, 448, 448), dtype="float32")
    for i, r in enumerate(bank_r):
        img = Image.open(r["image_path"]).convert("RGB")
        for k, (fam, lv) in enumerate(AUG_LIST):
            aug_tensors[i, k] = prep_image(img, fam, lv)

    bank_feat = forward_frozen_batch(model, bank_tensors, dev)
    aug_feat = forward_frozen_batch(model, aug_tensors.reshape(-1, 3, 448, 448), dev)
    aug_feat = aug_feat.reshape(len(bank_r), len(AUG_LIST), N_PATCH, D)
    print(f"  [{cat}|{seed}] bank+aug features done ({len(bank_r)} imgs)", flush=True)

    # ---- train R1 (identical sequence to e16_qualify.run_cell)
    train_n = 40 if smoke else TRAIN_N
    val_n = 20 if smoke else VAL_N
    train_feat, train_aug = bank_feat[:train_n], aug_feat[:train_n]
    val_feat, val_aug = bank_feat[train_n:train_n + val_n], aug_feat[train_n:train_n + val_n]
    torch.manual_seed(seed)
    adapter = LinearAdapter().to(dev)
    n_params = sum(p.numel() for p in adapter.parameters())
    assert n_params == 589824, f"trainable params {n_params} != 589824"
    trep = train_r1(adapter, train_feat, train_aug, val_feat, val_aug, dev, seed)

    # ---- F0 / R1 bank features
    b0 = bank_feat.reshape(-1, D).astype("float32")
    b0 /= np.maximum(np.linalg.norm(b0, axis=1, keepdims=True), 1e-12)
    bank_af = adapted_feat_adapter(adapter, bank_feat, dev)
    bn = l2norm(torch.from_numpy(bank_af).to(dev)).cpu().numpy().astype("float32")

    views_out = {}
    for v in VIEWS:
        test = [r for r in LD.select(recs, views=[v]) if r["split"] == "test"]
        if smoke:
            step = max(1, len(test) // 40)
            test = test[::step][:40]  # uniform subsample so both labels are present
        else:
            assert len(test) == 700, f"{cat} v{v}: expected 700 test records, got {len(test)}"
        test_r = route(test, "l0_448")
        test_tensors = np.stack([prep_image(Image.open(r["image_path"]).convert("RGB"))
                                 for r in test_r])
        test_feat = forward_frozen_batch(model, test_tensors, dev)
        t0f = test_feat.reshape(-1, D).astype("float32")
        t0f /= np.maximum(np.linalg.norm(t0f, axis=1, keepdims=True), 1e-12)
        s0 = nn_image_scores(b0, t0f, dev)
        met0 = official_metrics(rows21(test, s0, rawmap, rel, v))
        test_af = adapted_feat_adapter(adapter, test_feat, dev)
        tn = l2norm(torch.from_numpy(test_af).to(dev)).cpu().numpy().astype("float32")
        s1 = nn_image_scores(bn, tn, dev)
        met1 = official_metrics(rows21(test, s1, rawmap, rel, v))
        views_out[v] = {
            "n": len(test), "n_ok": met0["n_ok"], "n_ng": met0["n_ng"],
            "n_filtered": met0["n_filtered"],
            "auroc_f0": met0["auroc"], "ap_f0": met0["ap"],
            "auroc_r1": met1["auroc"], "ap_r1": met1["ap"],
            "scores_f0": [float(x) for x in s0], "scores_r1": [float(x) for x in s1],
            "labels": [int(m["image_anomaly"]) for m in [rawmap[rel(r)] for r in test]],
            "detectable_raw": [m["detectable"] if m["detectable"] is not False else False
                               for m in [rawmap[rel(r)] for r in test]],
            "specimen": [r["specimen"] for r in test],
            "illumination": [r["illumination"] for r in test],
        }
        print(f"  [{cat}|{seed}] view {v}: F0 AUROC={met0['auroc']:.5f} "
              f"R1 AUROC={met1['auroc']:.5f}", flush=True)
        del test_tensors, test_feat, test_af, tn
        torch.cuda.empty_cache()

    runtime = time.time() - t0
    peak = torch.cuda.max_memory_allocated() / 2 ** 20
    out = {"category": cat, "seed": seed, "smoke": bool(smoke),
           "bank_hash": bank_hash(bank), "adapter_hash": adapter_hash(adapter),
           "trainable_params": n_params, "steps_used": trep["steps_used"],
           "runtime_s": round(runtime, 1), "peak_gpu_mb": round(peak, 1),
           "views": views_out, "repro": None}

    # ---- reproduction check on view120 vs E16 history (values from e16_partial)
    if not smoke:
        e16 = json.load(open(E16_PARTIAL))["cells"][f"{cat}|{seed}"]
        h_hist = e16
        v120 = views_out["120"]
        d_f0 = max(abs(a - b) for a, b in zip(v120["scores_f0"], h_hist["test_scores_f0"]))
        d_r1 = max(abs(a - b) for a, b in zip(v120["scores_r1"], h_hist["test_scores_r1"]))
        auroc_d_f0 = abs(v120["auroc_f0"] - h_hist["f0"]["auroc"])
        auroc_d_r1 = abs(v120["auroc_r1"] - h_hist["r1"]["auroc"])
        ap_d_f0 = abs(v120["ap_f0"] - h_hist["f0"]["ap"])
        ap_d_r1 = abs(v120["ap_r1"] - h_hist["r1"]["ap"])
        repro_pass = (d_f0 <= TOL and d_r1 <= TOL and auroc_d_f0 <= TOL
                      and auroc_d_r1 <= TOL and ap_d_f0 <= TOL and ap_d_r1 <= TOL)
        out["repro"] = {"view": "120", "n": len(v120["scores_f0"]),
                        "max_abs_diff_scores_f0": d_f0, "max_abs_diff_scores_r1": d_r1,
                        "auroc_diff_f0": auroc_d_f0, "auroc_diff_r1": auroc_d_r1,
                        "ap_diff_f0": ap_d_f0, "ap_diff_r1": ap_d_r1,
                        "s1_target_auroc_f0": S1_AUROC_TARGET[cat],
                        "s2_target_auroc_r1": S2_AUROC_TARGET[cat],
                        "repro_pass": bool(repro_pass)}
        if not repro_pass:
            OUTDIR.mkdir(parents=True, exist_ok=True)
            with open(STOP_FILE, "w") as f:
                f.write(f"E21 STOP: reproduction failure at {cat}|{seed} (view120)\n")
                f.write(json.dumps(out["repro"], indent=2))
            print(f"!! REPRO FAIL {cat}|{seed}: {out['repro']}", flush=True)

    del model, bank_tensors, aug_tensors, bank_feat, aug_feat, bank_af
    del adapter
    torch.cuda.empty_cache()
    return out


# ---------------------------------------------------------------- partial io
def load_partial():
    if PARTIAL.exists():
        return json.load(open(PARTIAL))
    return {"experiment": "E21 multiview validation", "protocol": "results/e21/protocol.md",
            "views": VIEWS, "cells": {}}


def save_partial(d):
    OUTDIR.mkdir(parents=True, exist_ok=True)
    tmp = PARTIAL.with_suffix(".json.tmp")
    with open(tmp, "w") as f:
        json.dump(d, f, indent=1)
    os.replace(tmp, PARTIAL)


# ---------------------------------------------------------------- report
def write_round1_csv():
    """CSV from partial; view120 rows use REUSED history values (source=REUSED_VERIFIED)."""
    d = load_partial()
    e16 = json.load(open(E16_PARTIAL))["cells"]
    rows = []
    for cat in ("Bird", "Car", "Cube"):
        cell = d["cells"].get(f"{cat}|0")
        if cell is None:
            continue
        for v in VIEWS:
            vo = cell["views"][v]
            if v == "120":
                h = e16[f"{cat}|0"]
                rows.append({
                    "category": cat, "view": v, "seed": 0,
                    "n_good": vo["n_ok"], "n_anomaly": vo["n_ng"],
                    "auroc_f0": f"{h['f0']['auroc']:.6f}", "auroc_r1": f"{h['r1']['auroc']:.6f}",
                    "d_auroc": f"{h['r1']['auroc'] - h['f0']['auroc']:+.6f}",
                    "ap_f0": f"{h['f0']['ap']:.6f}", "ap_r1": f"{h['r1']['ap']:.6f}",
                    "d_ap": f"{h['r1']['ap'] - h['f0']['ap']:+.6f}",
                    "runtime_s": cell["runtime_s"], "bank_hash": cell["bank_hash"],
                    "adapter_hash": cell["adapter_hash"], "label_protocol": LABEL_PROTOCOL,
                    "source": "REUSED_VERIFIED"})
            else:
                rows.append({
                    "category": cat, "view": v, "seed": 0,
                    "n_good": vo["n_ok"], "n_anomaly": vo["n_ng"],
                    "auroc_f0": f"{vo['auroc_f0']:.6f}", "auroc_r1": f"{vo['auroc_r1']:.6f}",
                    "d_auroc": f"{vo['auroc_r1'] - vo['auroc_f0']:+.6f}",
                    "ap_f0": f"{vo['ap_f0']:.6f}", "ap_r1": f"{vo['ap_r1']:.6f}",
                    "d_ap": f"{vo['ap_r1'] - vo['ap_f0']:+.6f}",
                    "runtime_s": cell["runtime_s"], "bank_hash": cell["bank_hash"],
                    "adapter_hash": cell["adapter_hash"], "label_protocol": LABEL_PROTOCOL,
                    "source": "NEW"})
    cols = ["category", "view", "seed", "n_good", "n_anomaly", "auroc_f0", "auroc_r1", "d_auroc",
            "ap_f0", "ap_r1", "d_ap", "runtime_s", "bank_hash", "adapter_hash",
            "label_protocol", "source"]
    with open(R1_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"[E21 report] wrote {R1_CSV} ({len(rows)} rows)", flush=True)
    return rows


def preflight_target_check():
    """Verify protocol-stated S1/S2 display targets match e16_partial full-precision values.

    Tolerance 1e-5 = display-rounding level (protocol quotes 4-5 dp; e.g. Cube F0
    0.85410 is the 4dp rounding of 0.8540945896209056 whose 5dp form is 0.85409).
    The actual run-time reproduction tolerance remains TOL=1e-6 against e16_partial.
    """
    cells = json.load(open(E16_PARTIAL))["cells"]
    bad = []
    for cat in ("Bird", "Car", "Cube"):
        c = cells[f"{cat}|0"]
        f0 = c["f0"]["auroc"]
        r1 = c["r1"]["auroc"]
        if abs(f0 - S1_AUROC_TARGET[cat]) > 1e-5:
            bad.append((cat, "S1", f0, S1_AUROC_TARGET[cat]))
        if abs(r1 - S2_AUROC_TARGET[cat]) > 1e-5:
            bad.append((cat, "S2", r1, S2_AUROC_TARGET[cat]))
    if bad:
        print("!! protocol-stated targets do NOT match e16_partial (1e-5):")
        for b in bad:
            print("  ", b)
        return False
    print("[E21 preflight] S1/S2 display targets consistent with e16_partial full precision "
          "(<=1e-5); run-time tolerance vs history = 1e-6", flush=True)
    for cat in ("Bird", "Car", "Cube"):
        c = cells[f"{cat}|0"]
        print(f"  {cat}: F0={c['f0']['auroc']:.10f} R1={c['r1']['auroc']:.10f}", flush=True)
    return True


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--categories", default="Bird,Car,Cube")
    ap.add_argument("--seeds", default="0")
    ap.add_argument("--audit-only", action="store_true")
    ap.add_argument("--report-only", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--force", action="store_true", help="recompute cells even if present")
    args = ap.parse_args()

    OUTDIR.mkdir(parents=True, exist_ok=True)
    if args.report_only:
        write_round1_csv()
        return 0

    ok = data_audit()
    if not ok:
        print("!! data audit FAILED -> STOP (no GPU work)")
        return 3
    if not args.smoke and not preflight_target_check():
        print("!! preflight target mismatch -> STOP")
        return 3
    if args.audit_only:
        return 0

    dev = torch.device("cuda")
    cats = [c.strip() for c in args.categories.split(",") if c.strip()]
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    d = load_partial()
    total = len(cats) * len(seeds)
    done = 0
    t_start = time.time()
    for cat in cats:
        for seed in seeds:
            key = f"{cat}|{seed}"
            if key in d["cells"] and not args.force and not args.smoke:
                print(f"[E21] skip existing cell {key}", flush=True)
                done += 1
                continue
            print(f"[E21] === cell {key} ({done + 1}/{total}) ===", flush=True)
            cell = run_cell_multiview(cat, seed, dev, smoke=args.smoke)
            if not args.smoke:
                if cell["repro"] is None or not cell["repro"]["repro_pass"]:
                    print(f"!! {key} reproduction FAILED -> partial saved, STOP.", flush=True)
                    d["cells"][key] = cell
                    save_partial(d)
                    return 2
                d["cells"][key] = cell
                save_partial(d)
            done += 1
            el = time.time() - t_start
            eta = el / done * (total - done)
            print(f"[E21 STATUS] completed={done}/{total} elapsed={el:.0f}s "
                  f"avg={el / done:.0f}s/cell eta={eta:.0f}s", flush=True)
    if not args.smoke:
        write_round1_csv()
    print("[E21] run finished", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
