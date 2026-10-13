#!/usr/bin/env python
"""E13-B - R1 (Linear Residual Adapter) cross-seed stability validation.

Reuses E13-A R1 hyperparameters exactly. seed0 is REUSED from E13-A
(results/e13a/e13a_screening.csv); this runner computes only seed1/seed2
(6 new cells: 3 categories x 2 seeds). R0 baseline is recomputed per seed
(the frozen 1-NN-max baseline depends on the per-seed 300-bank).

Frozen protocol: results/e13b/e13b_protocol.json
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
from e11_p2_runner import (raw_test_map, official_metrics, route,  # noqa: E402
                           nn_image_scores)
from e13a_screen import (load_dinov2_local, freeze_backbone, LinearAdapter,  # noqa: E402
                         adapter_forward, prep_image, forward_frozen_batch,
                         adapted_feat_adapter, compute_loss, l2norm, collapse_stats,
                         is_collapsed, val_adapter_loss, _rows, load_p5_baseline,
                         D, N_PATCH, AUG_LIST, BATCH_IMGS, MAX_STEPS, PATIENCE_STEPS,
                         EVAL_EVERY, LR_ADAPTER, TRAIN_N, VAL_N)

OUTDIR = ROOT / "results" / "e13b"
PARTIAL = OUTDIR / "e13b_partial.json"
CSV = OUTDIR / "e13b_results.csv"
NEW_SEEDS = [1, 2]


def train_r1(adapter, train_feat, train_aug, val_feat, val_aug, dev, seed):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    opt = torch.optim.AdamW(adapter.parameters(), lr=LR_ADAPTER)
    N = train_feat.shape[0]
    K = train_aug.shape[1]
    before = collapse_stats(adapter_forward(adapter, torch.from_numpy(
        train_feat[:50].reshape(-1, D)).to(dev)).detach().cpu().numpy())
    best_val = float("inf")
    no_improve = 0
    report = {"steps_used": 0, "best_val": float("nan"), "loss_finite": True,
              "grad_present": False}

    for step in range(MAX_STEPS):
        idx = rng.integers(0, N, size=BATCH_IMGS)
        aidx = rng.integers(0, K, size=BATCH_IMGS)
        z = torch.from_numpy(train_feat[idx].reshape(-1, D)).to(dev)
        za = torch.from_numpy(train_aug[idx, aidx].reshape(-1, D)).to(dev)
        opt.zero_grad()
        z_orig = adapter_forward(adapter, z)
        z_aug = adapter_forward(adapter, za)
        loss, _, _ = compute_loss(z_orig, z_aug, z, "cons_preserve")
        if not torch.isfinite(loss):
            report["loss_finite"] = False
            break
        loss.backward()
        g = [p.grad for p in adapter.parameters() if p.grad is not None]
        report["grad_present"] = len(g) > 0 and all(x is not None for x in g)
        opt.step()

        if (step + 1) % EVAL_EVERY == 0:
            v = val_adapter_loss(adapter, val_feat, val_aug, "cons_preserve", dev)
            if v < best_val - 1e-6:
                best_val = v
                no_improve = 0
            else:
                no_improve += 1
            if no_improve * EVAL_EVERY >= PATIENCE_STEPS:
                break
        report["steps_used"] = step + 1

    after = collapse_stats(adapter_forward(adapter, torch.from_numpy(
        train_feat[:50].reshape(-1, D)).to(dev)).detach().cpu().numpy())
    report["before"] = before
    report["after"] = after
    report["best_val"] = best_val if best_val < float("inf") else float("nan")
    return report


def run_cell(category, seed, dev):
    recs = LD.load_records(category=category, require_exists=False)
    est = LD.estimate_scope(recs)
    bank_specs = set(est["bank_specimens"])
    pool = sorted(LD.select(recs, bank_specs), key=lambda r: r["image_path"])
    bank = E8.bank_for(pool, seed)
    test = [r for r in LD.select(recs, views=["120"]) if r["split"] == "test"]
    bank_r = route(bank, "l0_448")
    test_r = route(test, "l0_448")
    rel = lambda r: str(Path(r["image_path"]).relative_to(LD.DATA_ROOT))
    rawmap = raw_test_map(category)

    t0 = time.time()
    bank_tensors = np.stack([prep_image(Image.open(r["image_path"]).convert("RGB"))
                             for r in bank_r], axis=0)
    test_tensors = np.stack([prep_image(Image.open(r["image_path"]).convert("RGB"))
                             for r in test_r], axis=0)
    aug_tensors = np.empty((len(bank), len(AUG_LIST), 3, 448, 448), dtype="float32")
    for i, r in enumerate(bank_r):
        img = Image.open(r["image_path"]).convert("RGB")
        for k, (fam, lv) in enumerate(AUG_LIST):
            aug_tensors[i, k] = prep_image(img, fam, lv)
    prep_s = time.time() - t0

    t0 = time.time()
    model = load_dinov2_local(dev).to(dev)
    freeze_backbone(model)
    bank_feat = forward_frozen_batch(model, bank_tensors, dev)
    test_feat = forward_frozen_batch(model, test_tensors, dev)
    aug_feat = forward_frozen_batch(model, aug_tensors.reshape(-1, 3, 448, 448), dev)
    aug_feat = aug_feat.reshape(len(bank), len(AUG_LIST), N_PATCH, D)
    extract_s = time.time() - t0

    # R0 baseline (per-seed bank)
    bank_s0 = bank_feat.reshape(-1, D).astype("float32")
    bank_s0 = bank_s0 / np.maximum(np.linalg.norm(bank_s0, axis=1, keepdims=True), 1e-12)
    test_s0 = test_feat.reshape(-1, D).astype("float32")
    test_s0 = test_s0 / np.maximum(np.linalg.norm(test_s0, axis=1, keepdims=True), 1e-12)
    s0_scores = nn_image_scores(bank_s0, test_s0, dev)
    s0_met = official_metrics(_rows(test, s0_scores, rawmap, rel))

    # R1 training + scoring
    train_feat, train_aug = bank_feat[:TRAIN_N], aug_feat[:TRAIN_N]
    val_feat, val_aug = bank_feat[TRAIN_N:TRAIN_N + VAL_N], aug_feat[TRAIN_N:TRAIN_N + VAL_N]
    torch.manual_seed(seed)
    adapter = LinearAdapter().to(dev)
    rep = train_r1(adapter, train_feat, train_aug, val_feat, val_aug, dev, seed)
    bank_af = adapted_feat_adapter(adapter, bank_feat, dev)
    test_af = adapted_feat_adapter(adapter, test_feat, dev)
    n_params_ = sum(p.numel() for p in adapter.parameters())
    del adapter
    torch.cuda.empty_cache()

    bank_n = l2norm(torch.from_numpy(bank_af).to(dev)).cpu().numpy()
    test_n = l2norm(torch.from_numpy(test_af).to(dev)).cpu().numpy()
    scores = nn_image_scores(bank_n.astype("float32"), test_n.astype("float32"), dev)
    met = official_metrics(_rows(test, scores, rawmap, rel))

    del model, bank_tensors, test_tensors, aug_tensors, bank_feat, test_feat, aug_feat
    torch.cuda.empty_cache()
    return {"category": category, "seed": seed,
            "r0": {"auroc": s0_met["auroc"], "ap": s0_met["ap"]},
            "r1": {"auroc": met["auroc"], "ap": met["ap"]},
            "d_auroc": met["auroc"] - s0_met["auroc"],
            "d_ap": met["ap"] - s0_met["ap"],
            "trainable_params": n_params_, "steps_used": rep["steps_used"],
            "std_ratio": rep["after"]["std"] / max(rep["before"]["std"], 1e-12),
            "pairwise_cos": rep["after"]["pairwise_cos"],
            "collapsed": is_collapsed(rep), "loss_finite": rep["loss_finite"],
            "prep_s": round(prep_s, 1), "extract_s": round(extract_s, 1)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--categories", default="Bird,Car,Cube")
    ap.add_argument("--seeds", default="1,2")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    dev = torch.device("cuda")
    cats = [c.strip() for c in args.categories.split(",") if c.strip()]
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    OUTDIR.mkdir(parents=True, exist_ok=True)

    print(f"[E13-B] cats={cats} seeds={seeds} smoke={args.smoke} | disk {E8.free_gb():.1f} GB | "
          f"GPU {E8.gpu_util()}", flush=True)

    partial = {"experiment": "E13-B", "cells": {}}
    if PARTIAL.exists():
        try:
            partial.update(json.loads(PARTIAL.read_text()))
        except json.JSONDecodeError:
            pass

    t_start = time.time()
    last_out = None
    total_new = len(cats) * len(seeds)
    done_cnt = 0
    for cat in cats:
        for seed in seeds:
            key = f"{cat}|{seed}"
            if not args.smoke and key in partial.get("cells", {}):
                print(f"[E13-B] SKIP {key} (done)", flush=True)
                done_cnt += 1
                continue
            print(f"\n[E13-B] RUN {cat} seed{seed}", flush=True)
            try:
                out = run_cell(cat, seed, dev)
                last_out = out
            except Exception as exc:
                import traceback
                traceback.print_exc()
                print(f"[E13-B] {cat} seed{seed} FAILED: {type(exc).__name__}: {exc}", flush=True)
                if args.smoke:
                    return 1
                continue
            if not args.smoke:
                partial["cells"][key] = {k: out[k] for k in
                                         ("category", "seed", "r0", "r1", "d_auroc", "d_ap",
                                          "trainable_params", "steps_used", "std_ratio",
                                          "pairwise_cos", "collapsed", "loss_finite")}
                tmp = PARTIAL.with_suffix(".json.tmp")
                tmp.write_text(json.dumps(partial, indent=2, default=float))
                tmp.replace(PARTIAL)
            done_cnt += 1
            el = time.time() - t_start
            per = el / max(1, done_cnt)
            eta = per * (total_new - done_cnt)
            print(f"[E13-B] PROGRESS new={done_cnt}/{total_new} | elapsed={el/60:.1f}m | "
                  f"ETA={eta/60:.1f}m | {cat} seed{seed} AUROC={out['r1']['auroc']:.5f} "
                  f"dAUROC={out['d_auroc']:+.5f} dAP={out['d_ap']:+.5f}", flush=True)

    if args.smoke:
        print("\n[E13-B] SMOKE CHECK SUMMARY", flush=True)
        ok = True
        checks = {}
        o = last_out
        if o is None:
            checks["0_run"] = False
            ok = False
        else:
            checks["0_run"] = True
            checks["1_r0_finite"] = np.isfinite(o["r0"]["auroc"])
            checks["2_adapter_params"] = o["trainable_params"] == 589824
            checks["3_loss_finite"] = o["loss_finite"]
            checks["4_no_collapse"] = not o["collapsed"]
            checks["5_checkpoint"] = True
        for k, v in checks.items():
            print(f"  {k}: {'PASS' if v else 'FAIL'}", flush=True)
            ok = ok and v
        print(f"[E13-B] SMOKE {'PASS' if ok else 'FAIL'} (no checkpoint/CSV written)", flush=True)
        return 0 if ok else 1

    write_csv(partial)
    print(f"\n[E13-B] done -> {CSV}", flush=True)
    return 0


def write_csv(partial):
    w = csv.writer(open(CSV, "w", newline=""))
    w.writerow(["category", "seed", "r0_auroc", "r0_ap", "r1_auroc", "r1_ap",
                "d_auroc", "d_ap", "steps_used", "std_ratio", "pairwise_cos", "status"])
    for cat in ("Bird", "Car", "Cube"):
        for seed in NEW_SEEDS:
            cell = partial.get("cells", {}).get(f"{cat}|{seed}")
            if not cell:
                continue
            w.writerow([cat, seed, f"{cell['r0']['auroc']:.5f}", f"{cell['r0']['ap']:.5f}",
                        f"{cell['r1']['auroc']:.5f}", f"{cell['r1']['ap']:.5f}",
                        f"{cell['d_auroc']:+.5f}", f"{cell['d_ap']:+.5f}",
                        cell["steps_used"], f"{cell['std_ratio']:.3f}",
                        f"{cell['pairwise_cos']:.3f}",
                        "STOP_COLLAPSE" if cell["collapsed"] else "OK"])


if __name__ == "__main__":
    raise SystemExit(main())
