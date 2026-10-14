#!/usr/bin/env python
"""E16 - R1 Signal Qualification.

Frozen protocol: results/e16/protocol.md

Reuses F0/R1 AUROC/AP from E13 (source=REUSED) and performs a DETERMINISTIC
reproduction run to obtain the metrics E13 never saved (d_prime, illumination
robustness, runtime, peak GPU). The reproduction MUST match history to 1e-6,
otherwise the cell is flagged REPRO_FAIL and the run stops.

No new candidates. No tuning. No backbone/scoring/bank/resolution changes.
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
from e11_p2_runner import raw_test_map, official_metrics, route, nn_image_scores  # noqa: E402
from e13a_screen import (load_dinov2_local, freeze_backbone, LinearAdapter,  # noqa: E402
                         adapter_forward, prep_image, forward_frozen_batch,
                         adapted_feat_adapter, l2norm, val_adapter_loss, _rows,
                         D, N_PATCH, AUG_LIST, TRAIN_N, VAL_N, EXTRACT_CHUNK)
from e13b_screen import train_r1  # noqa: E402  (seed-aware R1 trainer, identical hyperparams)

OUTDIR = ROOT / "results" / "e16"
PARTIAL = OUTDIR / "e16_partial.json"
RAW_CSV = OUTDIR / "metrics_raw.csv"
SUM_CSV = OUTDIR / "summary.csv"
DIAG_CSV = OUTDIR / "diagnostics.csv"
VERDICT = OUTDIR / "verdict.json"
CATS = ["Bird", "Car", "Cube"]
SEEDS = [0, 1, 2]
TOL = 1e-6
ROB_CHUNK = 50


# ------------------------------------------------------------------ history (reuse table)
def load_history():
    """-> {(cat, seed): {'f0': (auroc, ap), 'r1': (auroc, ap)}}

    NOTE: read the authoritative FULL-PRECISION partial JSONs, not the human-facing
    *_screening.csv / *_results.csv (those are formatted at 5 decimals and lose ~4e-6,
    which is larger than the pre-registered 1e-6 reproduction tolerance).
    """
    h = {}
    a = json.loads((ROOT / "results" / "e13a" / "e13a_partial.json").read_text())
    for cat in CATS:
        c = a["cells"][f"{cat}|R1"]
        h[(cat, 0)] = {"f0": (a["sanity"][cat]["auroc"], a["sanity"][cat]["ap"]),
                       "r1": (c["auroc"], c["ap"])}
    b = json.loads((ROOT / "results" / "e13b" / "e13b_partial.json").read_text())
    for cat in CATS:
        for s in (1, 2):
            c = b["cells"][f"{cat}|{s}"]
            h[(cat, s)] = {"f0": (c["r0"]["auroc"], c["r0"]["ap"]),
                           "r1": (c["r1"]["auroc"], c["r1"]["ap"])}
    missing = [(c, s) for c in CATS for s in SEEDS if (c, s) not in h]
    assert not missing, f"history missing: {missing}"
    return h


def load_scope(category):
    recs = LD.load_records(category=category, require_exists=False)
    est = LD.estimate_scope(recs)
    bank_specs = set(est["bank_specimens"])
    bank = E8.bank_for(sorted(LD.select(recs, bank_specs), key=lambda r: r["image_path"]), 0)
    test = [r for r in LD.select(recs, views=["120"]) if r["split"] == "test"]
    return bank, test, recs


# ------------------------------------------------------------------ metrics
def d_prime(scores, labels):
    s = np.asarray(scores, float)
    y = np.asarray(labels, int)
    ng, ok = s[y == 1], s[y == 0]
    if len(ng) < 2 or len(ok) < 2:
        return float("nan")
    denom = np.sqrt(0.5 * (ng.var(ddof=1) + ok.var(ddof=1)))
    return float((ng.mean() - ok.mean()) / max(denom, 1e-12))


def _l2_last(x):
    return x / x.norm(dim=-1, keepdim=True).clamp_min(1e-12)


def robustness_feat(bank_feat, aug_feat, chunk=50):
    """mean_{i,k} 1-cos(F(x_i), F(T_k(x_i))). bank_feat (N,P,D); aug_feat (N,K,P,D)."""
    N, K, P, D_ = aug_feat.shape
    acc, cnt = 0.0, 0
    with torch.inference_mode():
        for i in range(0, N, chunk):
            b = _l2_last(torch.from_numpy(bank_feat[i:i + chunk]).float())      # (c,P,D)
            a = _l2_last(torch.from_numpy(aug_feat[i:i + chunk]).float())       # (c,K,P,D)
            cos = (b.unsqueeze(1) * a).sum(dim=-1)                              # (c,K,P)
            acc += float((1.0 - cos).sum().item())
            cnt += cos.numel()
    return acc / max(1, cnt)


def robustness_score_norm(bank_norm_feat, aug_norm_feat, dev, denom):
    """mean anomaly score of perturbed normal images (vs the same bank), normalized."""
    bank_t = torch.from_numpy(bank_norm_feat.reshape(-1, D)).to(dev)
    N, K = aug_norm_feat.shape[0], aug_norm_feat.shape[1]
    P = aug_norm_feat.shape[2] if aug_norm_feat.ndim == 4 else N_PATCH
    vals = []
    with torch.inference_mode():
        for i in range(N):
            q = torch.from_numpy(aug_norm_feat[i].reshape(-1, D)).to(dev)
            q = q / q.norm(dim=1, keepdim=True).clamp_min(1e-12)
            best = None
            for c0 in range(0, bank_t.shape[0], 65536):
                sim = q @ bank_t[c0:c0 + 65536].T
                m = sim.max(dim=1).values
                best = m if best is None else torch.maximum(best, m)
            dist = (1.0 - best).clamp_min(0.0)
            for k in range(K):
                vals.append(float(mean_top1p_local(dist[k * P:(k + 1) * P].cpu().numpy())))
    return float(np.mean(vals) / max(denom, 1e-12))


def _img_mean(feat):
    """(N,P,D) -> (N,D) mean over patches, L2-normalized."""
    v = feat.mean(axis=1)
    return v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-12)


def _mean_pairwise_cos(v):
    """mean off-diagonal cosine among rows of (N,D)."""
    if len(v) < 2:
        return float("nan")
    s = v @ v.T
    n = len(v)
    return float((s.sum() - np.trace(s)) / (n * (n - 1)))


def compute_diagnostics(bank_feat, test_feat, bank_af, test_af, aug_feat, aug_af,
                        illumination, labels):
    """Protocol 5 A-E. All cheap (image-mean vectors + patch-level shift)."""
    N, P, Dd = bank_feat.shape
    M = test_feat.shape[0]
    b_v, t_v = _img_mean(bank_feat), _img_mean(test_feat)
    ba_v, ta_v = _img_mean(bank_af.reshape(N, P, Dd)), _img_mean(test_af.reshape(M, P, Dd))
    diag = {}
    # A. normal feature dispersion
    diag["A_disp_pairwise_cos_F0"] = _mean_pairwise_cos(b_v)
    diag["A_disp_pairwise_cos_R1"] = _mean_pairwise_cos(ba_v)
    diag["A_disp_patch_std_F0"] = float(bank_feat.reshape(-1, D).std(axis=0).mean())
    diag["A_disp_patch_std_R1"] = float(bank_af.reshape(-1, D).std(axis=0).mean())
    # B. illumination-group centroid distance (test)
    ill = np.asarray(illumination)
    cents = []
    for u in sorted(set(ill.tolist())):
        m = ill == u
        if m.sum() >= 1:
            cents.append(t_v[m].mean(axis=0))
    cents = np.asarray(cents)
    cents = cents / np.maximum(np.linalg.norm(cents, axis=1, keepdims=True), 1e-12)
    diag["B_illum_group_centroid_dist_F0"] = float(1.0 - _mean_pairwise_cos(cents))
    cents1 = []
    for u in sorted(set(ill.tolist())):
        m = ill == u
        if m.sum() >= 1:
            cents1.append(ta_v[m].mean(axis=0))
    cents1 = np.asarray(cents1)
    cents1 = cents1 / np.maximum(np.linalg.norm(cents1, axis=1, keepdims=True), 1e-12)
    diag["B_illum_group_centroid_dist_R1"] = float(1.0 - _mean_pairwise_cos(cents1))
    # C. good-vs-defect separation (image-mean centroid cosine distance)
    y = np.asarray(labels, int)
    if (y == 0).sum() and (y == 1).sum():
        g0, g1 = t_v[y == 0].mean(axis=0), t_v[y == 1].mean(axis=0)
        g0 /= max(np.linalg.norm(g0), 1e-12)
        g1 /= max(np.linalg.norm(g1), 1e-12)
        diag["C_good_ng_centroid_dist_F0"] = float(1.0 - float(g0 @ g1))
        h0, h1 = ta_v[y == 0].mean(axis=0), ta_v[y == 1].mean(axis=0)
        h0 /= max(np.linalg.norm(h0), 1e-12)
        h1 /= max(np.linalg.norm(h1), 1e-12)
        diag["C_good_ng_centroid_dist_R1"] = float(1.0 - float(h0 @ h1))
    # D. illumination shift magnitude (patch level, bank)
    K = aug_feat.shape[1]
    ref0 = np.repeat(bank_feat[:, None], K, axis=1).reshape(-1, Dd)
    sh0 = np.linalg.norm((aug_feat - bank_feat[:, None]).reshape(-1, Dd), axis=1) / \
        np.maximum(np.linalg.norm(ref0, axis=1), 1e-12)
    diag["D_illum_shift_F0"] = float(sh0.mean())
    bank_af_r = bank_af.reshape(N, P, Dd)
    ref1 = np.repeat(bank_af_r[:, None], aug_af.shape[1], axis=1).reshape(-1, Dd)
    sh1 = np.linalg.norm((aug_af - bank_af_r[:, None]).reshape(-1, Dd), axis=1) / \
        np.maximum(np.linalg.norm(ref1, axis=1), 1e-12)
    diag["D_illum_shift_R1"] = float(sh1.mean())
    # E. adaptation-induced centroid drift (test)
    c0, c1 = t_v.mean(axis=0), ta_v.mean(axis=0)
    diag["E_adapt_centroid_drift"] = float(np.linalg.norm(c1 - c0) / max(np.linalg.norm(c0), 1e-12))
    return diag


def bootstrap_ci(scores, labels, n_boot=2000, seed=0):
    """bootstrap CI of AUROC (returns [lo, hi] 2.5/97.5 percentile)."""
    from sklearn.metrics import roc_auc_score
    s = np.asarray(scores, float)
    y = np.asarray(labels, int)
    rng = np.random.default_rng(seed)
    n = len(s)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if len(set(y[idx].tolist())) < 2:
            continue
        vals.append(roc_auc_score(y[idx], s[idx]))
    if not vals:
        return [float("nan"), float("nan")]
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


def mean_top1p_local(dist):
    flat = np.asarray(dist).flatten()
    n = int(len(flat) * 0.01)
    if n == 0:
        return float(np.max(flat))
    return float(np.mean(np.sort(flat)[::-1][:n]))


# ------------------------------------------------------------------ one cell
def run_cell(cat, seed, dev, hist, smoke=False):
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

    # ---- reproduction identity check
    h = hist[(cat, seed)]
    r0_ok = abs(met0["auroc"] - h["f0"][0]) <= TOL and abs(met0["ap"] - h["f0"][1]) <= TOL
    r1_ok = abs(met1["auroc"] - h["r1"][0]) <= TOL and abs(met1["ap"] - h["r1"][1]) <= TOL

    # ---- d_prime
    labels = [int(m["image_anomaly"]) for m in
              [rawmap[rel(r)] for r in test]]
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
    out = {"category": cat, "seed": seed,
           "f0": {"auroc": met0["auroc"], "ap": met0["ap"], "d_prime": dp0,
                  "robustness_feat": rob_f0, "robustness_score": rob_s0,
                  "auroc_ci": ci0},
           "r1": {"auroc": met1["auroc"], "ap": met1["ap"], "d_prime": dp1,
                  "robustness_feat": rob_f1, "robustness_score": rob_s1,
                  "auroc_ci": ci1},
           "repro_ok": bool(r0_ok and r1_ok), "repro_r0_ok": bool(r0_ok), "repro_r1_ok": bool(r1_ok),
           "steps_used": trep["steps_used"], "runtime_s": round(runtime, 1),
           "peak_gpu_mb": round(peak, 1), "diagnostics": diag,
           "test_scores_f0": [float(x) for x in s0], "test_scores_r1": [float(x) for x in s1],
           "labels": labels, "specimen": [r["specimen"] for r in test],
           "illumination": illums}
    del model, bank_tensors, test_tensors, aug_tensors, bank_feat, test_feat, aug_feat
    del bank_af, test_af, adapter
    torch.cuda.empty_cache()
    return out


def adapted_feat_aug(adapter, aug_feat, dev, chunk=20):
    N, K, P, D_ = aug_feat.shape
    outs = np.empty((N, K, P, D_), dtype="float32")
    with torch.inference_mode():
        for i in range(0, N, chunk):
            z = torch.from_numpy(aug_feat[i:i + chunk].reshape(-1, D_)).to(dev)
            outs[i:i + chunk] = adapter_forward(adapter, z).cpu().numpy().reshape(-1, K, P, D_)
    return outs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--categories", default="Bird,Car,Cube")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    dev = torch.device("cuda")
    cats = [c.strip() for c in args.categories.split(",") if c.strip()]
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    OUTDIR.mkdir(parents=True, exist_ok=True)
    hist = load_history()

    print(f"[E16] cats={cats} seeds={seeds} smoke={args.smoke} | disk {E8.free_gb():.1f} GB",
          flush=True)
    assert (OUTDIR / "protocol.md").exists(), "protocol.md must be frozen first"

    partial = {"experiment": "E16", "cells": {}}
    if PARTIAL.exists() and not args.smoke:
        try:
            partial.update(json.loads(PARTIAL.read_text()))
        except json.JSONDecodeError:
            pass

    t_start = time.time()
    total = len(cats) * len(seeds)
    done = 0
    for cat in cats:
        for seed in seeds:
            key = f"{cat}|{seed}"
            if not args.smoke and key in partial.get("cells", {}):
                print(f"[E16] SKIP {key} (done)", flush=True)
                done += 1
                continue
            print(f"\n[E16] RUN {cat} seed{seed}", flush=True)
            out = run_cell(cat, seed, dev, hist, smoke=args.smoke)
            if args.smoke:
                print(f"[E16] SMOKE OK: auroc_f0_mean={out['auroc_f0']:.4f} "
                      f"auroc_r1_mean={out['auroc_r1']:.4f} steps={out['steps']} "
                      f"shape_ok={out['shape_ok']}", flush=True)
                return 0
            if not out["repro_ok"]:
                print(f"[E16] REPRO_FAIL {key}: r0_ok={out['repro_r0_ok']} "
                      f"r1_ok={out['repro_r1_ok']} | reproduced f0={out['f0']['auroc']:.6f} "
                      f"r1={out['r1']['auroc']:.6f} vs hist f0={hist[(cat,seed)]['f0'][0]:.6f} "
                      f"r1={hist[(cat,seed)]['r1'][0]:.6f}", flush=True)
                return 2
            partial["cells"][key] = out
            tmp = PARTIAL.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(partial, indent=2, default=float))
            tmp.replace(PARTIAL)
            done += 1
            el = time.time() - t_start
            d_au = out["r1"]["auroc"] - out["f0"]["auroc"]
            d_ap = out["r1"]["ap"] - out["f0"]["ap"]
            print(f"[E16] STATUS {done}/{total} | {100*done/total:.1f}% | current={cat}/seed{seed} "
                  f"| elapsed={el/60:.1f}m ETA={el/max(1,done)*(total-done)/60:.1f}m | "
                  f"dAUROC={d_au:+.5f} dAP={d_ap:+.5f} repro_ok=True", flush=True)

    write_csvs(partial, hist)
    print(f"\n[E16] done -> {RAW_CSV}", flush=True)
    return 0


def write_csvs(partial, hist):
    cells = partial["cells"]
    with open(RAW_CSV, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["category", "seed", "method", "AUROC", "AP", "d_prime",
                    "robustness_feat", "robustness_score", "runtime_s", "peak_gpu_mb",
                    "source", "repro_ok"])
        for c in CATS:
            for s in SEEDS:
                cell = cells.get(f"{c}|{s}")
                if not cell:
                    continue
                h = hist[(c, s)]
                for meth, histvals in (("F0", h["f0"]), ("R1", h["r1"])):
                    m = cell["f0"] if meth == "F0" else cell["r1"]
                    w.writerow([c, s, meth, f"{histvals[0]:.5f}", f"{histvals[1]:.5f}",
                                f"{m['d_prime']:.4f}", f"{m['robustness_feat']:.6f}",
                                f"{m['robustness_score']:.6f}",
                                cell["runtime_s"] if meth == "R1" else 0.0,
                                cell["peak_gpu_mb"] if meth == "R1" else 0.0,
                                "REUSED", cell["repro_ok"]])
    with open(SUM_CSV, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["category", "seed", "AUROC_F0", "AUROC_R1", "AP_F0", "AP_R1",
                    "d_prime_F0", "d_prime_R1", "rob_F0", "rob_R1", "dScore_F0", "dScore_R1",
                    "delta_AUROC", "delta_AP", "delta_d_prime", "delta_rob",
                    "delta_dScore", "runtime_s", "source"])
        for c in CATS:
            for s in SEEDS:
                cell = cells.get(f"{c}|{s}")
                if not cell:
                    continue
                h = hist[(c, s)]
                w.writerow([c, s, f"{h['f0'][0]:.5f}", f"{h['r1'][0]:.5f}",
                            f"{h['f0'][1]:.5f}", f"{h['r1'][1]:.5f}",
                            f"{cell['f0']['d_prime']:.4f}", f"{cell['r1']['d_prime']:.4f}",
                            f"{cell['f0']['robustness_feat']:.6f}",
                            f"{cell['r1']['robustness_feat']:.6f}",
                            f"{cell['f0']['robustness_score']:.6f}",
                            f"{cell['r1']['robustness_score']:.6f}",
                            f"{h['r1'][0]-h['f0'][0]:+.5f}", f"{h['r1'][1]-h['f0'][1]:+.5f}",
                            f"{cell['r1']['d_prime']-cell['f0']['d_prime']:+.4f}",
                            f"{cell['r1']['robustness_feat']-cell['f0']['robustness_feat']:+.6f}",
                            f"{cell['r1']['robustness_score']-cell['f0']['robustness_score']:+.6f}",
                            cell["runtime_s"], "REUSED+NEW"])
    with open(DIAG_CSV, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["category", "seed", "metric", "F0", "R1", "delta_R1_minus_F0"])
        for c in CATS:
            for s in SEEDS:
                cell = cells.get(f"{c}|{s}")
                if not cell or "diagnostics" not in cell:
                    continue
                d = cell["diagnostics"]
                seen = set()
                for k in sorted(d):
                    if k.endswith("_F0"):
                        base = k[:-3]
                        v1 = d.get(base + "_R1")
                        if v1 is None:
                            continue
                        seen.add(base)
                        w.writerow([c, s, base, f"{d[k]:.6f}", f"{v1:.6f}", f"{v1-d[k]:+.6f}"])
                for k in sorted(d):
                    if k.endswith("_F0") or k.endswith("_R1"):
                        continue
                    w.writerow([c, s, k, "", f"{d[k]:.6f}", ""])


if __name__ == "__main__":
    raise SystemExit(main())
