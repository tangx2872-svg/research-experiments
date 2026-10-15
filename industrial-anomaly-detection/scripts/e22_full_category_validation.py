#!/usr/bin/env python
"""E22 - M2AD Full-Category Multi-Seed Validation (frozen protocol: results/e22/protocol.md).

Matrix: 10 categories x 3 seeds x {F0, frozen R1-ILA} x view120 = 30 cells.
  - Bird/Car/Cube (9 cells): HISTORICAL_REUSED from results/e16/e16_partial.json
    (reuse is guarded by bank-hash / label-list / metric recomputation checks).
  - Dice..Tube (21 cells): NEW computation by this script.

Provenance (IMPORTANT - no frozen file is modified by E22):
  run_cell_e22 is a COPY of e16_qualify.run_cell (the frozen E16 code path) with exactly
  three deliberate differences:
    (a) the `hist[(cat, seed)]` reproduction assertion is REMOVED from the cell body
        (the 7 new categories have no history) and replaced by an optional comparison
        against e16_partial performed by the CALLER (repro_compare);
    (b) bank_hash / adapter_hash / trainable_params / label-provenance / timestamp are
        added to the returned dict;
    (c) a HARD assertion that every evaluation label comes from the official metadata
        (raw_test_map) - never from Good/NG directory names.
  Empirical equivalence of the copy is proven by `--repro` Gate A: Bird|0 executed
  through BOTH the frozen e16_qualify.run_cell and this copy, in one process, must be
  bit-identical (max|delta score| == 0).

Reused as imports only: e4d1_m2ad_loader, e8_common, e11_p2_runner (scoring/metrics),
e13a_screen (model/prep/adapters/aug), e13b_screen.train_r1, e16_qualify (diagnostics).

CLI:
  python scripts/e22_full_category_validation.py --preflight
  python scripts/e22_full_category_validation.py --import-history
  python scripts/e22_full_category_validation.py --repro          # GPU reproduction gate
  python scripts/e22_full_category_validation.py --matrix         # 21 new cells (resume)
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402

import e4d1_m2ad_loader as LD  # noqa: E402
import e8_common as E8  # noqa: E402
import e13a_screen as E13A  # noqa: E402
from e11_p2_runner import raw_test_map, official_metrics, route, nn_image_scores  # noqa: E402
from e13a_screen import (load_dinov2_local, freeze_backbone, LinearAdapter,  # noqa: E402
                         adapted_feat_adapter, prep_image, forward_frozen_batch,
                         l2norm, _rows, D, N_PATCH, AUG_LIST, TRAIN_N, VAL_N)
from e13b_screen import train_r1  # noqa: E402
from e16_qualify import (compute_diagnostics, d_prime, bootstrap_ci,  # noqa: E402
                         robustness_feat, robustness_score_norm, adapted_feat_aug)

OUTDIR = ROOT / "results" / "e22"
CELLDIR = OUTDIR / "cells"
LIVE = OUTDIR / "live_results.csv"
STATUS = OUTDIR / "status.json"
STOP_REPRO = OUTDIR / "STOP_repro_fail.txt"
PREFLIGHT_CSV = OUTDIR / "data_preflight.csv"
PROTOCOL = OUTDIR / "protocol.md"
E16_PARTIAL = ROOT / "results" / "e16" / "e16_partial.json"
E21_PARTIAL = ROOT / "results" / "e21" / "e21_partial.json"

HIST_CATS = ["Bird", "Car", "Cube"]
NEW_CATS = ["Dice", "Doll", "Holder", "Motor", "Ring", "Teapot", "Tube"]
ALL_CATS = HIST_CATS + NEW_CATS
SEEDS = [0, 1, 2]
TOL = 1e-6
ROUNDS = {0: "A", 1: "B", 2: "C"}

_LOCK = threading.Lock()
_T0 = time.time()
_RUNTIMES: list[float] = []


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def hms(s: float) -> str:
    s = int(max(0, s))
    return f"{s // 3600}h{s % 3600 // 60}m{s % 60}s"


def gpu_stats() -> dict:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total",
             "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=15)
        u, m, mt = [x.strip() for x in out.stdout.strip().split(",")[:3]]
        return {"util_pct": int(u), "mem_used_mb": int(m), "mem_total_mb": int(mt)}
    except Exception as exc:  # noqa: BLE001
        return {"error": f"{type(exc).__name__}: {exc}"}


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


def write_json_atomic(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1, default=float)
    os.replace(tmp, path)


def cell_path(cat: str, seed: int) -> Path:
    return CELLDIR / f"{cat}_seed{seed}.json"


# ------------------------------------------------------------------ frozen-parameter guards
def assert_frozen_hyperparams() -> None:
    """Hard runtime assertions: the E22 run must use exactly the E17-frozen values."""
    fm = json.loads((ROOT / "results" / "e17" / "final_method.json").read_text())["parameters"]
    checks = {
        "BETA": (E13A.BETA, fm["BETA"]),
        "LR_ADAPTER": (E13A.LR_ADAPTER, fm["LR_ADAPTER"]),
        "MAX_STEPS": (E13A.MAX_STEPS, fm["MAX_STEPS"]),
        "PATIENCE_STEPS": (E13A.PATIENCE_STEPS, fm["PATIENCE_STEPS"]),
        "EVAL_EVERY": (E13A.EVAL_EVERY, fm["EVAL_EVERY"]),
        "BATCH_IMGS": (E13A.BATCH_IMGS, fm["BATCH_IMGS"]),
        "TRAIN_N": (E13A.TRAIN_N, fm["TRAIN_N"]),
        "VAL_N": (E13A.VAL_N, fm["VAL_N"]),
    }
    bad = {k: v for k, v in checks.items() if v[0] != v[1]}
    if bad:
        raise RuntimeError(f"frozen hyperparameter mismatch vs results/e17/final_method.json: {bad}")
    if list(AUG_LIST) != [(f, lv) for f in ["brightness", "contrast", "gamma"] for lv in [0.85, 1.15]]:
        raise RuntimeError(f"AUG_LIST changed: {AUG_LIST}")
    if (D, N_PATCH) != (768, 1024):
        raise RuntimeError(f"D/N_PATCH changed: {D}/{N_PATCH}")
    if E8.BANK_SUBSET_SEED != 20261009 or E8.BANK_SIZE != 300:
        raise RuntimeError(f"bank rule changed: seed={E8.BANK_SUBSET_SEED} size={E8.BANK_SIZE}")
    print(f"[E22] frozen hyperparams OK: BETA={E13A.BETA} LR={E13A.LR_ADAPTER} "
          f"steps={E13A.MAX_STEPS}/{E13A.PATIENCE_STEPS} batch={E13A.BATCH_IMGS} "
          f"train/val={TRAIN_N}/{VAL_N} bank={E8.BANK_SIZE} seed0={E8.BANK_SUBSET_SEED} "
          f"aug={AUG_LIST}", flush=True)


def labels_from_metadata(test, rawmap, rel) -> tuple[list, list, list, dict]:
    """Hard label-provenance gate: labels ONLY from official metadata image_anomaly."""
    labels, detect, ill = [], [], []
    n_ng_dir_but_normal = 0
    for r in test:
        k = rel(r)
        if k not in rawmap:
            raise RuntimeError(f"label provenance violation: {k} not in official metadata")
        m = rawmap[k]
        lab = int(m["image_anomaly"])
        labels.append(lab)
        detect.append(m["detectable"] is not False)
        ill.append(r["illumination"])
        if "/NG/" in k and lab == 0:
            n_ng_dir_but_normal += 1
    prov = {"label_source": "meta_unsupervised.json:image_anomaly",
            "n_records": len(test), "n_ng_dir_but_image_anomaly_0": n_ng_dir_but_normal,
            "n_detectable": int(sum(detect)), "n_filtered": int(len(detect) - sum(detect))}
    return labels, detect, ill, prov


# ------------------------------------------------------------------ one cell (copy of e16_qualify.run_cell)
def run_cell_e22(cat: str, seed: int, dev, smoke: bool = False) -> dict:
    """Frozen cell code path (copy of e16_qualify.run_cell; see module docstring)."""
    t0 = time.time()
    torch.cuda.reset_peak_memory_stats()
    recs = LD.load_records(category=cat, require_exists=False)
    est = LD.estimate_scope(recs)
    bank_specs = set(est["bank_specimens"])
    pool = sorted(LD.select(recs, bank_specs), key=lambda r: r["image_path"])
    bank = E8.bank_for(pool, seed)
    test = [r for r in LD.select(recs, views=["120"]) if r["split"] == "test"]
    bank_r = route(bank, "l0_448")
    test_r = route(test, "l0_448")
    rel = lambda r: str(Path(r["image_path"]).relative_to(LD.DATA_ROOT))
    rawmap = raw_test_map(cat)

    missing = [r["image_path"] for r in bank_r + test_r if not Path(r["image_path"]).exists()]
    if missing:
        raise RuntimeError(f"{cat}|{seed}: {len(missing)} cache files missing, e.g. {missing[:2]}")

    labels, detect_flags, _, prov = labels_from_metadata(test, rawmap, rel)

    model = load_dinov2_local(dev).to(dev)
    freeze_backbone(model)

    n_bank = min(60, len(bank_r)) if smoke else len(bank_r)
    n_test = min(40, len(test_r)) if smoke else len(test_r)
    bank_r, test_r = bank_r[:n_bank], test_r[:n_test]
    bank, test = bank[:n_bank], test[:n_test]

    bank_tensors = np.stack([prep_image(Image.open(r["image_path"]).convert("RGB")) for r in bank_r])
    test_tensors = np.stack([prep_image(Image.open(r["image_path"]).convert("RGB")) for r in test_r])
    aug_tensors = np.empty((len(bank_r), len(AUG_LIST), 3, 448, 448), dtype="float32")
    for i, r in enumerate(bank_r):
        img = Image.open(r["image_path"]).convert("RGB")
        for k, (fam, lv) in enumerate(AUG_LIST):
            aug_tensors[i, k] = prep_image(img, fam, lv)

    bank_feat = forward_frozen_batch(model, bank_tensors, dev)
    test_feat = forward_frozen_batch(model, test_tensors, dev)
    aug_feat = forward_frozen_batch(model, aug_tensors.reshape(-1, 3, 448, 448), dev)
    aug_feat = aug_feat.reshape(len(bank_r), len(AUG_LIST), N_PATCH, D)

    # ---- F0
    b0 = bank_feat.reshape(-1, D).astype("float32")
    b0 /= np.maximum(np.linalg.norm(b0, axis=1, keepdims=True), 1e-12)
    t0f = test_feat.reshape(-1, D).astype("float32")
    t0f /= np.maximum(np.linalg.norm(t0f, axis=1, keepdims=True), 1e-12)
    s0 = nn_image_scores(b0, t0f, dev)
    met0 = official_metrics(_rows(test, s0, rawmap, rel)) if not smoke else None

    # ---- R1
    train_n = 40 if smoke else TRAIN_N
    val_n = 20 if smoke else VAL_N
    train_feat, train_aug = bank_feat[:train_n], aug_feat[:train_n]
    val_feat, val_aug = bank_feat[train_n:train_n + val_n], aug_feat[train_n:train_n + val_n]
    torch.manual_seed(seed)
    adapter = LinearAdapter().to(dev)
    n_params = sum(p.numel() for p in adapter.parameters())
    if n_params != 589824:
        raise RuntimeError(f"trainable params {n_params} != 589824")
    trep = train_r1(adapter, train_feat, train_aug, val_feat, val_aug, dev, seed)
    bank_af = adapted_feat_adapter(adapter, bank_feat, dev)
    test_af = adapted_feat_adapter(adapter, test_feat, dev)
    bn = l2norm(torch.from_numpy(bank_af).to(dev)).cpu().numpy()
    tn = l2norm(torch.from_numpy(test_af).to(dev)).cpu().numpy()
    s1 = nn_image_scores(bn.astype("float32"), tn.astype("float32"), dev)
    met1 = official_metrics(_rows(test, s1, rawmap, rel)) if not smoke else None

    if smoke:
        del model, bank_tensors, test_tensors, aug_tensors, bank_feat, test_feat, aug_feat
        torch.cuda.empty_cache()
        return {"smoke": True, "auroc_f0": float(np.mean(s0)), "auroc_r1": float(np.mean(s1)),
                "steps": trep["steps_used"], "shape_ok": bank_af.shape[1] == D}

    # ---- d_prime
    dp0 = d_prime(s0, labels)
    dp1 = d_prime(s1, labels)

    # ---- robustness (bank normals; feature + score level)
    aug_af = adapted_feat_aug(adapter, aug_feat, dev)
    rob_f0 = robustness_feat(bank_feat, aug_feat)
    rob_f1 = robustness_feat(bank_af.reshape(len(bank_r), N_PATCH, D), aug_af)
    denom = float(np.std(s0)) if np.std(s0) > 1e-12 else 1.0
    rob_s0 = robustness_score_norm(b0, aug_feat, dev, denom)
    rob_s1 = robustness_score_norm(bn, aug_af, dev, denom)

    illums = [r["illumination"] for r in test]
    diag = compute_diagnostics(bank_feat, test_feat, bank_af, test_af, aug_feat, aug_af,
                               illums, labels)
    ci0 = bootstrap_ci(s0, labels)
    ci1 = bootstrap_ci(s1, labels)
    runtime = time.time() - t0
    peak = torch.cuda.max_memory_allocated() / 2 ** 20

    out = {"category": cat, "seed": seed, "view": "120", "source": "NEW",
           "n_test": len(test), "n_ok": met0["n_ok"], "n_ng": met0["n_ng"],
           "n_filtered": met0["n_filtered"],
           "f0": {"auroc": met0["auroc"], "ap": met0["ap"], "d_prime": dp0,
                  "robustness_feat": rob_f0, "robustness_score": rob_s0, "auroc_ci": ci0},
           "r1": {"auroc": met1["auroc"], "ap": met1["ap"], "d_prime": dp1,
                  "robustness_feat": rob_f1, "robustness_score": rob_s1, "auroc_ci": ci1},
           "delta_auroc": met1["auroc"] - met0["auroc"],
           "delta_ap": met1["ap"] - met0["ap"],
           "bank_hash": bank_hash(bank), "bank_size": len(bank),
           "adapter_hash": adapter_hash(adapter), "trainable_params": n_params,
           "steps_used": trep["steps_used"], "runtime_s": round(runtime, 1),
           "peak_gpu_mb": round(peak, 1), "diagnostics": diag,
           "scores_f0": [float(x) for x in s0], "scores_r1": [float(x) for x in s1],
           "labels": labels, "detectable_raw": detect_flags,
           "specimen": [r["specimen"] for r in test], "kind": [r["kind"] for r in test],
           "illumination": illums, "label_provenance": prov,
           "repro": None, "status": "OK", "timestamp": now_iso()}

    del model, bank_tensors, test_tensors, aug_tensors, bank_feat, test_feat, aug_feat
    del bank_af, test_af, adapter
    torch.cuda.empty_cache()
    return out


def repro_compare(cell: dict, ref: dict) -> dict:
    """Compare a fresh cell against a stored e16_partial cell (per-image scores + metrics)."""
    d_f0 = max(abs(a - b) for a, b in zip(cell["scores_f0"], ref["test_scores_f0"]))
    d_r1 = max(abs(a - b) for a, b in zip(cell["scores_r1"], ref["test_scores_r1"]))
    return {"n": len(cell["scores_f0"]),
            "max_abs_diff_scores_f0": d_f0, "max_abs_diff_scores_r1": d_r1,
            "auroc_diff_f0": abs(cell["f0"]["auroc"] - ref["f0"]["auroc"]),
            "auroc_diff_r1": abs(cell["r1"]["auroc"] - ref["r1"]["auroc"]),
            "ap_diff_f0": abs(cell["f0"]["ap"] - ref["f0"]["ap"]),
            "ap_diff_r1": abs(cell["r1"]["ap"] - ref["r1"]["ap"]),
            "labels_identical": cell["labels"] == [int(x) for x in ref["labels"]],
            "repro_pass": bool(d_f0 <= TOL and d_r1 <= TOL
                               and abs(cell["f0"]["auroc"] - ref["f0"]["auroc"]) <= TOL
                               and abs(cell["r1"]["auroc"] - ref["r1"]["auroc"]) <= TOL
                               and abs(cell["f0"]["ap"] - ref["f0"]["ap"]) <= TOL
                               and abs(cell["r1"]["ap"] - ref["r1"]["ap"]) <= TOL
                               and cell["labels"] == [int(x) for x in ref["labels"]])}


# ------------------------------------------------------------------ preflight (CPU, no GPU)
def data_preflight() -> bool:
    meta = json.loads((ROOT / "data/m2ad/jsons_extract/meta_unsupervised.json").read_text())
    rows, ok_all = [], True
    for cat in ALL_CATS:
        recs = LD.load_records(category=cat, require_exists=False)
        est = LD.estimate_scope(recs)
        pool = sorted(LD.select(recs, set(est["bank_specimens"])), key=lambda r: r["image_path"])
        bank = E8.bank_for(pool, 0)
        test = [r for r in LD.select(recs, views=["120"]) if r["split"] == "test"]
        ents = [e for e in meta["test"][cat] if e["view"] == "120"]
        kept = [e for e in ents if e["detectable"] is not False]
        y = [int(e["image_anomaly"]) for e in kept]
        rel = lambda r: str(Path(r["image_path"]).relative_to(LD.DATA_ROOT))
        rawmap = raw_test_map(cat)
        miss = [r["image_path"] for r in route(bank, "l0_448") + route(test, "l0_448")
                if not Path(r["image_path"]).exists()]
        ill = sorted({e["illumination"] for e in kept})
        reasons = []
        if len(test) != len(ents):
            reasons.append(f"test count {len(test)} != metadata {len(ents)}")
        if any(rel(r) not in rawmap for r in test):
            reasons.append("test record missing from metadata")
        if miss:
            reasons.append(f"{len(miss)} cache files missing")
        if len(ill) != 10:
            reasons.append(f"illumination groups {len(ill)} != 10")
        if y.count(0) < 2 or y.count(1) < 2:
            reasons.append(f"class balance {y.count(0)}/{y.count(1)}")
        if len(bank) != E8.BANK_SIZE or len(pool) != 1200:
            reasons.append(f"pool/bank {len(pool)}/{len(bank)} != 1200/300")
        ok = not reasons
        ok_all &= ok
        rows.append({"category": cat, "hist_or_new": "hist" if cat in HIST_CATS else "new",
                     "pool": len(pool), "bank": len(bank), "test_raw": len(test),
                     "test_kept": len(kept), "good": y.count(0), "anomaly": y.count(1),
                     "n_filtered": len(ents) - len(kept), "illum_groups": len(ill),
                     "cache_missing": len(miss), "bank_hash_seed0": bank_hash(bank),
                     "pass": int(ok), "reasons": "; ".join(reasons)})
    OUTDIR.mkdir(parents=True, exist_ok=True)
    cols = list(rows[0])
    with open(PREFLIGHT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    for r in rows:
        print(f"[E22 preflight] {r['category']:7s} pool={r['pool']} bank={r['bank']} "
              f"test={r['test_raw']} kept={r['test_kept']} good={r['good']} ng={r['anomaly']} "
              f"illum={r['illum_groups']} cache_miss={r['cache_missing']} "
              f"{'PASS' if r['pass'] else 'FAIL ' + r['reasons']}", flush=True)
    print(f"[E22 preflight] wrote {PREFLIGHT_CSV} | ALL={'PASS' if ok_all else 'FAIL'}", flush=True)
    return ok_all


# ------------------------------------------------------------------ historical import (CPU)
def import_history() -> bool:
    """Materialise the 9 Bird/Car/Cube cells as HISTORICAL_REUSED with reuse verification."""
    e16 = json.loads(E16_PARTIAL.read_text())["cells"]
    e21 = json.loads(E21_PARTIAL.read_text())["cells"] if E21_PARTIAL.exists() else {}
    all_ok = True
    for cat in HIST_CATS:
        recs = LD.load_records(category=cat, require_exists=False)
        est = LD.estimate_scope(recs)
        pool = sorted(LD.select(recs, set(est["bank_specimens"])), key=lambda r: r["image_path"])
        for seed in SEEDS:
            bank = E8.bank_for(pool, seed)
            bh = bank_hash(bank)
            src = e16[f"{cat}|{seed}"]
            test = [r for r in LD.select(recs, views=["120"]) if r["split"] == "test"]
            rel = lambda r: str(Path(r["image_path"]).relative_to(LD.DATA_ROOT))
            rawmap = raw_test_map(cat)
            labels, detect, ill, prov = labels_from_metadata(test, rawmap, rel)
            rows = [{"score": s, "image_anomaly": lab, "detectable": d, "illumination": il}
                    for s, lab, d, il in zip(src["test_scores_f0"], labels, detect, ill)]
            m0 = official_metrics(rows)
            rows1 = [dict(r, score=s) for r, s in zip(rows, src["test_scores_r1"])]
            m1 = official_metrics(rows1)
            checks = {
                "labels_match_e16": labels == [int(x) for x in src["labels"]],
                "illum_match_e16": ill == list(src["illumination"]),
                "specimen_match_e16": [r["specimen"] for r in test] == list(src["specimen"]),
                "n_test": len(test) == len(src["test_scores_f0"]),
                "metrics_recompute_f0": abs(m0["auroc"] - src["f0"]["auroc"]) <= 1e-12
                                        and abs(m0["ap"] - src["f0"]["ap"]) <= 1e-12,
                "metrics_recompute_r1": abs(m1["auroc"] - src["r1"]["auroc"]) <= 1e-12
                                        and abs(m1["ap"] - src["r1"]["ap"]) <= 1e-12,
                "bank_hash_matches_e21": (seed != 0) or (
                    e21.get(f"{cat}|0", {}).get("bank_hash") == bh),
                "n_ok_ng": m0["n_ok"] == src["f0"].get("n_ok", m0["n_ok"])
                           if "n_ok" in src["f0"] else True,
            }
            ok = all(checks.values())
            all_ok &= ok
            cell = {"category": cat, "seed": seed, "view": "120", "source": "HISTORICAL_REUSED",
                    "n_test": len(test), "n_ok": m0["n_ok"], "n_ng": m0["n_ng"],
                    "n_filtered": m0["n_filtered"],
                    "f0": dict(src["f0"]), "r1": dict(src["r1"]),
                    "delta_auroc": src["r1"]["auroc"] - src["f0"]["auroc"],
                    "delta_ap": src["r1"]["ap"] - src["f0"]["ap"],
                    "bank_hash": bh, "bank_size": len(bank), "adapter_hash": None,
                    "trainable_params": 589824, "steps_used": src["steps_used"],
                    "runtime_s": src["runtime_s"], "peak_gpu_mb": src["peak_gpu_mb"],
                    "diagnostics": src["diagnostics"],
                    "scores_f0": src["test_scores_f0"], "scores_r1": src["test_scores_r1"],
                    "labels": labels, "detectable_raw": detect,
                    "specimen": src["specimen"], "kind": [r["kind"] for r in test],
                    "illumination": ill, "label_provenance": prov,
                    "reuse_verification": checks,
                    "repro": {"note": "E16 stored per-image scores; E21 independently verified "
                                      "bit-identical view120 reproduction for seed0 in a fresh "
                                      "process; E22 --repro re-executes seed0 (and Car|1).",
                              "repro_pass": bool(ok)},
                    "source_ref": "results/e16/e16_partial.json",
                    "status": "OK" if ok else "REUSE_CHECK_FAILED",
                    "timestamp": now_iso()}
            write_json_atomic(cell_path(cat, seed), cell)
            print(f"[E22 history] {cat}|{seed} bank={bh[:12]} reuse_checks="
                  f"{'PASS' if ok else json.dumps(checks)}", flush=True)
    print(f"[E22 history] imported 9 historical cells | ALL={'PASS' if all_ok else 'FAIL'}",
          flush=True)
    return all_ok


# ------------------------------------------------------------------ live csv / status
def refresh_live_csv() -> None:
    rows = []
    for cat in ALL_CATS:
        for seed in SEEDS:
            p = cell_path(cat, seed)
            if not p.exists():
                continue
            c = json.loads(p.read_text())
            rows.append({
                "category": cat, "seed": seed, "view": c.get("view", "120"),
                "n_good": c["n_ok"], "n_anomaly": c["n_ng"], "n_filtered": c["n_filtered"],
                "auroc_f0": f"{c['f0']['auroc']:.6f}", "auroc_r1": f"{c['r1']['auroc']:.6f}",
                "d_auroc": f"{c['r1']['auroc'] - c['f0']['auroc']:+.6f}",
                "ap_f0": f"{c['f0']['ap']:.6f}", "ap_r1": f"{c['r1']['ap']:.6f}",
                "d_ap": f"{c['r1']['ap'] - c['f0']['ap']:+.6f}",
                "d_prime_f0": f"{c['f0']['d_prime']:.4f}", "d_prime_r1": f"{c['r1']['d_prime']:.4f}",
                "rob_feat_f0": f"{c['f0']['robustness_feat']:.6f}",
                "rob_feat_r1": f"{c['r1']['robustness_feat']:.6f}",
                "rob_score_f0": f"{c['f0']['robustness_score']:.6f}",
                "rob_score_r1": f"{c['r1']['robustness_score']:.6f}",
                "bank_hash": c["bank_hash"], "adapter_hash": c.get("adapter_hash") or "",
                "steps_used": c.get("steps_used", ""), "runtime_s": c.get("runtime_s", ""),
                "peak_gpu_mb": c.get("peak_gpu_mb", ""), "timestamp": c.get("timestamp", ""),
                "status": c.get("status", ""), "source": c.get("source", "")})
    cols = ["category", "seed", "view", "n_good", "n_anomaly", "n_filtered", "auroc_f0",
            "auroc_r1", "d_auroc", "ap_f0", "ap_r1", "d_ap", "d_prime_f0", "d_prime_r1",
            "rob_feat_f0", "rob_feat_r1", "rob_score_f0", "rob_score_r1", "bank_hash",
            "adapter_hash", "steps_used", "runtime_s", "peak_gpu_mb", "timestamp", "status",
            "source"]
    tmp = LIVE.with_suffix(".csv.tmp")
    with open(tmp, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    os.replace(tmp, LIVE)


def write_status(state: dict) -> None:
    write_json_atomic(STATUS, state)


def start_heartbeat(shared: dict, interval: int = 60):
    stop = threading.Event()

    def loop():
        while not stop.wait(interval):
            with _LOCK:
                shared["last_heartbeat"] = now_iso()
                shared["elapsed_s"] = round(time.time() - _T0, 1)
                shared["gpu"] = gpu_stats()
                shared["disk_free_gb"] = round(E8.free_gb(), 2)
                write_status(shared)
                n_done, n_tot = shared["n_completed"], shared["total_cells"]
                cur = shared.get("current") or {}
                mean_rt = (sum(_RUNTIMES) / len(_RUNTIMES)) if _RUNTIMES else float("nan")
                eta = (n_tot - n_done) * mean_rt if _RUNTIMES else float("nan")
                print(f"[E22 STATUS] stage={shared['stage']} completed {n_done}/{n_tot} | "
                      f"{100 * n_done / max(1, n_tot):.1f}% | current={cur.get('category')}"
                      f"/seed{cur.get('seed')} cell_elapsed={hms(time.time() - cur.get('t0', time.time()))} "
                      f"| elapsed={hms(shared['elapsed_s'])} last_cell={hms(_RUNTIMES[-1]) if _RUNTIMES else 'n/a'} "
                      f"mean_cell={hms(mean_rt) if _RUNTIMES else 'n/a'} ETA={hms(eta) if _RUNTIMES else 'n/a'} "
                      f"| gpu={shared['gpu'].get('util_pct', '?')}%/{shared['gpu'].get('mem_used_mb', '?')}MB "
                      f"disk={shared['disk_free_gb']}GB", flush=True)
    t = threading.Thread(target=loop, daemon=True)
    t.start()
    return stop


# ------------------------------------------------------------------ commands
def cmd_repro(dev) -> int:
    """Gate A (bit-level, both code paths on Bird|0) + Gate B (1e-6 vs e16, 4 cells)."""
    import e16_qualify as E16  # frozen original; imported ONLY for the equivalence gate
    ref = json.loads(E16_PARTIAL.read_text())["cells"]
    hist = E16.load_history()
    results = {}

    print("\n[E22 repro] Gate A: Bird|0 through frozen e16_qualify.run_cell ...", flush=True)
    frozen_out = E16.run_cell("Bird", 0, dev, hist)
    print("[E22 repro] Gate A: Bird|0 through E22 copy ...", flush=True)
    e22_out = run_cell_e22("Bird", 0, dev)
    d_f0 = max(abs(a - b) for a, b in zip(frozen_out["test_scores_f0"], e22_out["scores_f0"]))
    d_r1 = max(abs(a - b) for a, b in zip(frozen_out["test_scores_r1"], e22_out["scores_r1"]))
    gate_a = {"max_abs_diff_scores_f0": d_f0, "max_abs_diff_scores_r1": d_r1,
              "bit_identical": bool(d_f0 == 0.0 and d_r1 == 0.0),
              "frozen_path_repro_vs_history": bool(frozen_out["repro_ok"]),
              "e22_path_vs_history": repro_compare(e22_out, ref["Bird|0"])}
    results["GateA_Bird0_two_code_paths"] = gate_a
    print(f"[E22 repro] Gate A: max|d score| F0={d_f0:.3e} R1={d_r1:.3e} "
          f"bit_identical={gate_a['bit_identical']} "
          f"frozen_repro_ok={gate_a['frozen_path_repro_vs_history']}", flush=True)

    print("\n[E22 repro] Gate B: Bird|0, Car|0, Cube|0, Car|1 vs e16_partial (tol 1e-6) ...",
          flush=True)
    for cat, seed in [("Bird", 0), ("Car", 0), ("Cube", 0), ("Car", 1)]:
        if cat == "Bird" and seed == 0:
            cell = e22_out
        else:
            cell = run_cell_e22(cat, seed, dev)
        cmp_ = repro_compare(cell, ref[f"{cat}|{seed}"])
        results[f"GateB_{cat}{seed}"] = cmp_
        cell["repro"] = cmp_
        print(f"[E22 repro] {cat}|{seed}: dF0={cmp_['max_abs_diff_scores_f0']:.3e} "
              f"dR1={cmp_['max_abs_diff_scores_r1']:.3e} labels_identical="
              f"{cmp_['labels_identical']} pass={cmp_['repro_pass']}", flush=True)

    ok = gate_a["bit_identical"] and gate_a["frozen_path_repro_vs_history"] \
        and gate_a["e22_path_vs_history"]["repro_pass"] \
        and all(v["repro_pass"] for k, v in results.items() if k.startswith("GateB"))
    write_json_atomic(OUTDIR / "repro_gate.json",
                      {"timestamp": now_iso(), "tolerance": TOL, "results": results,
                       "all_pass": bool(ok)})
    if not ok:
        STOP_REPRO.write_text(f"E22 reproduction gate FAILED at {now_iso()}\n"
                              + json.dumps(results, indent=1, default=float))
        print("!! E22 REPRODUCTION GATE FAILED -> STOP (no matrix run)", flush=True)
        return 2
    print("[E22 repro] ALL GATES PASS -> matrix may start", flush=True)
    return 0


def cmd_matrix(dev, cats: list, seeds: list, smoke: bool = False) -> int:
    if not PROTOCOL.exists():
        print("!! protocol.md missing - freeze the protocol first", flush=True)
        return 3
    assert_frozen_hyperparams()
    if smoke:
        cat, seed = cats[0], seeds[0]
        print(f"[E22 smoke] {cat}|{seed} (no persistence)", flush=True)
        out = run_cell_e22(cat, seed, dev, smoke=True)
        print(f"[E22 smoke] OK: auroc_f0_mean={out['auroc_f0']:.4f} "
              f"auroc_r1_mean={out['auroc_r1']:.4f} steps={out['steps']} "
              f"shape_ok={out['shape_ok']}", flush=True)
        return 0
    e16 = json.loads(E16_PARTIAL.read_text())["cells"]
    OUTDIR.mkdir(parents=True, exist_ok=True)
    CELLDIR.mkdir(parents=True, exist_ok=True)

    todo = [(c, s) for s in seeds for c in cats]       # seed-major = Round A/B/C
    total = len(todo)
    shared = {"task": "E22 M2AD full-category multi-seed validation",
              "protocol": "results/e22/protocol.md", "stage": "matrix",
              "total_cells": total, "n_completed": 0, "completed_cells": [],
              "current": None, "started": now_iso(), "last_heartbeat": now_iso(),
              "pid": os.getpid(), "gpu": gpu_stats(), "disk_free_gb": round(E8.free_gb(), 2),
              "eta_s": None, "last_output": None, "host": "autodl"}
    hb = start_heartbeat(shared)
    failures: dict[str, int] = {}
    try:
        for i, (cat, seed) in enumerate(todo):
            key = f"{cat}|{seed}"
            p = cell_path(cat, seed)
            if p.exists():
                cell0 = json.loads(p.read_text())
                if cell0.get("status") == "OK" and cell0.get("source") == "NEW":
                    print(f"[E22] SKIP {key} (already complete)", flush=True)
                    shared["completed_cells"].append(key)
                    shared["n_completed"] = len(shared["completed_cells"])
                    with _LOCK:
                        write_status(shared)
                    continue
            t0 = time.time()
            with _LOCK:
                shared["current"] = {"category": cat, "seed": seed, "t0": t0,
                                     "round": ROUNDS[seed], "key": key, "pid": os.getpid()}
                write_status(shared)
            print(f"\n[E22] === RUN {key} (round {ROUNDS[seed]}, {i + 1}/{total}) ===", flush=True)
            try:
                cell = run_cell_e22(cat, seed, dev)
                # optional per-cell reproduction compare (3 historical cats have refs)
                if f"{key}" in e16:
                    cell["repro"] = repro_compare(cell, e16[key])
                    print(f"[E22] {key} repro_ok={cell['repro']['repro_pass']} "
                          f"dR1={cell['repro']['max_abs_diff_scores_r1']:.3e}", flush=True)
                write_json_atomic(p, cell)
            except Exception as exc:  # noqa: BLE001
                failures[key] = failures.get(key, 0) + 1
                print(f"!! [E22] CELL FAILED {key}: {type(exc).__name__}: {exc} "
                      f"(attempt {failures[key]}/2)", flush=True)
                write_json_atomic(OUTDIR / "cell_errors.json",
                                  {k: v for k, v in failures.items()})
                if failures[key] >= 2:
                    shared["stage"] = "ENGINEERING_BLOCKED"
                    shared["current"] = {"category": cat, "seed": seed, "blocked": True}
                    with _LOCK:
                        write_status(shared)
                    print(f"!! [E22] ENGINEERING BLOCKED at {key} after 2 failures -> STOP",
                          flush=True)
                    return 4
                continue
            rt = cell["runtime_s"]
            _RUNTIMES.append(rt)
            shared["completed_cells"].append(key)
            shared["n_completed"] = len(shared["completed_cells"])
            shared["last_output"] = (f"{key} dAUROC={cell['delta_auroc']:+.5f} "
                                     f"dAP={cell['delta_ap']:+.5f} "
                                     f"(F0 {cell['f0']['auroc']:.5f} -> R1 {cell['r1']['auroc']:.5f})")
            refresh_live_csv()
            el = time.time() - _T0
            mean_rt = sum(_RUNTIMES) / len(_RUNTIMES)
            eta = mean_rt * (total - shared["n_completed"])
            shared["eta_s"] = round(eta, 1)
            shared["current"] = None
            shared["last_heartbeat"] = now_iso()
            shared["gpu"] = gpu_stats()
            shared["disk_free_gb"] = round(E8.free_gb(), 2)
            with _LOCK:
                write_status(shared)
            print(f"[E22 STATUS] completed {shared['n_completed']}/{total} | "
                  f"{100 * shared['n_completed'] / max(1, total):.1f}% | current={key} "
                  f"| round={ROUNDS[seed]} | elapsed={hms(el)} last_cell={hms(rt)} "
                  f"mean_cell={hms(mean_rt)} ETA={hms(eta)} | gpu="
                  f"{shared['gpu'].get('util_pct', '?')}%/{shared['gpu'].get('mem_used_mb', '?')}MB "
                  f"disk={shared['disk_free_gb']}GB | {shared['last_output']}", flush=True)
    finally:
        hb.set()
        shared["stage"] = "done" if shared["n_completed"] == total else "stopped"
        shared["current"] = None
        shared["finished"] = now_iso()
        write_status(shared)
    refresh_live_csv()
    print(f"\n[E22] matrix finished: {shared['n_completed']}/{total} cells | live csv {LIVE}",
          flush=True)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--preflight", action="store_true")
    ap.add_argument("--import-history", action="store_true")
    ap.add_argument("--repro", action="store_true")
    ap.add_argument("--matrix", action="store_true")
    ap.add_argument("--categories", default=",".join(NEW_CATS))
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    if args.preflight:
        return 0 if data_preflight() else 3
    if args.import_history:
        return 0 if import_history() else 3

    dev = torch.device("cuda")
    if args.repro:
        return cmd_repro(dev)
    if args.matrix:
        cats = [c.strip() for c in args.categories.split(",") if c.strip()]
        seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
        return cmd_matrix(dev, cats, seeds, smoke=args.smoke)
    ap.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
