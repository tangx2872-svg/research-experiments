#!/usr/bin/env python
"""E17 - Lightweight Refinement: BETA sweep around the qualified R1.

Frozen protocol: results/e17/protocol.md

Single search axis = BETA (weight of L_preserve in L = l_cons + BETA*l_preserve).
Injected at runtime via `e13a_screen.BETA = v` (compute_loss reads the module global),
so NO frozen code is modified.

C0 (BETA=0.1) is the qualified current R1: its metrics are REUSED from E16 and it is
only recomputed as a 1e-6 reproduction gate.
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
import e13a_screen as E13A  # noqa: E402
from e11_p2_runner import raw_test_map, official_metrics, route, nn_image_scores  # noqa: E402
from e13a_screen import (load_dinov2_local, freeze_backbone, LinearAdapter,  # noqa: E402
                         prep_image, forward_frozen_batch, adapted_feat_adapter,
                         l2norm, collapse_stats, is_collapsed, _rows,
                         D, N_PATCH, AUG_LIST, TRAIN_N, VAL_N)
from e13b_screen import train_r1  # noqa: E402
from e16_qualify import (d_prime, robustness_feat, robustness_score_norm,  # noqa: E402
                         _l2_last, _img_mean, _mean_pairwise_cos)

OUTDIR = ROOT / "results" / "e17"
PARTIAL = OUTDIR / "e17_partial.json"
E16 = ROOT / "results" / "e16" / "e16_partial.json"
CATS = ["Bird", "Car", "Cube"]
CONFIGS = [("C0", 0.1), ("C1", 0.0), ("C2", 0.05), ("C3", 0.2), ("C4", 0.5)]
CHALLENGERS = ["C1", "C2", "C3", "C4"]
TOL = 1e-6
ROB_CHUNK = 50


def load_e16():
    p = json.loads(E16.read_text())
    assert p.get("cells"), "E16 reference incomplete"
    return p["cells"]


def scope(category, seed):
    recs = LD.load_records(category=category, require_exists=False)
    est = LD.estimate_scope(recs)
    bank = E8.bank_for(sorted(LD.select(recs, set(est["bank_specimens"])),
                              key=lambda r: r["image_path"]), seed)
    test = [r for r in LD.select(recs, views=["120"]) if r["split"] == "test"]
    return bank, test


def illuminate_shift(bank_feat, aug_feat, bank_af, aug_af):
    K = aug_feat.shape[1]
    P, Dd = bank_feat.shape[1], bank_feat.shape[2]
    ref0 = np.repeat(bank_feat[:, None], K, axis=1).reshape(-1, Dd)
    s0 = (np.linalg.norm((aug_feat - bank_feat[:, None]).reshape(-1, Dd), axis=1)
          / np.maximum(np.linalg.norm(ref0, axis=1), 1e-12)).mean()
    bar = bank_af.reshape(bank_af.shape[0] // P, P, Dd)
    ref1 = np.repeat(bar[:, None], aug_af.shape[1], axis=1).reshape(-1, Dd)
    s1 = (np.linalg.norm((aug_af - bar[:, None]).reshape(-1, Dd), axis=1)
          / np.maximum(np.linalg.norm(ref1, axis=1), 1e-12)).mean()
    return float(s0), float(s1)


def run_category_seed(cat, seed, dev, configs, e16c, gate):
    """Extract features ONCE, then train/score each BETA config on the cached features."""
    t_start = time.time()
    bank, test = scope(cat, seed)
    bank_r, test_r = route(bank, "l0_448"), route(test, "l0_448")
    rel = lambda r: str(Path(r["image_path"]).relative_to(LD.DATA_ROOT))
    rawmap = raw_test_map(cat)

    model = load_dinov2_local(dev).to(dev)
    freeze_backbone(model)
    bank_t = np.stack([prep_image(Image.open(r["image_path"]).convert("RGB")) for r in bank_r])
    test_t = np.stack([prep_image(Image.open(r["image_path"]).convert("RGB")) for r in test_r])
    aug_t = np.empty((len(bank_r), len(AUG_LIST), 3, 448, 448), dtype="float32")
    for i, r in enumerate(bank_r):
        img = Image.open(r["image_path"]).convert("RGB")
        for k, (fam, lv) in enumerate(AUG_LIST):
            aug_t[i, k] = prep_image(img, fam, lv)

    t0 = time.time()
    bank_feat = forward_frozen_batch(model, bank_t, dev)
    test_feat = forward_frozen_batch(model, test_t, dev)
    aug_feat = forward_frozen_batch(model, aug_t.reshape(-1, 3, 448, 448), dev)
    aug_feat = aug_feat.reshape(len(bank_r), len(AUG_LIST), N_PATCH, D)
    extract_s = time.time() - t0
    del model, bank_t, test_t, aug_t
    torch.cuda.empty_cache()

    # F0 test-score std (reused from E16; frozen & deterministic) -> robustness normalizer
    s0_std = float(np.std(np.array(e16c["test_scores_f0"]))) or 1.0

    labels = [int(rawmap[rel(r)]["image_anomaly"]) for r in test]
    train_feat, train_aug = bank_feat[:TRAIN_N], aug_feat[:TRAIN_N]
    val_feat, val_aug = bank_feat[TRAIN_N:TRAIN_N + VAL_N], aug_feat[TRAIN_N:TRAIN_N + VAL_N]

    results = {}
    for cfg, beta in configs:
        t0 = time.time()
        E13A.BETA = beta                    # single search axis (module global read at call time)
        torch.manual_seed(seed)
        adapter = LinearAdapter().to(dev)
        rep = train_r1(adapter, train_feat, train_aug, val_feat, val_aug, dev, seed)
        bank_af = adapted_feat_adapter(adapter, bank_feat, dev)
        test_af = adapted_feat_adapter(adapter, test_feat, dev)
        aug_af = adapted_feat_aug(adapter, aug_feat, dev)
        n_params = sum(p.numel() for p in adapter.parameters())
        del adapter
        torch.cuda.empty_cache()
        E13A.BETA = 0.1                     # restore the qualified value

        bn = l2norm(torch.from_numpy(bank_af).to(dev)).cpu().numpy().astype("float32")
        tn = l2norm(torch.from_numpy(test_af).to(dev)).cpu().numpy().astype("float32")
        s = nn_image_scores(bn, tn, dev)
        met = official_metrics(_rows(test, s, rawmap, rel))
        dp = d_prime(s, labels)
        rf = robustness_feat(bank_af.reshape(len(bank_feat), N_PATCH, D), aug_af)
        rs = robustness_score_norm(bn, aug_af, dev, s0_std)
        _, shift1 = illuminate_shift(bank_feat, aug_feat, bank_af, aug_af)
        results[cfg] = {
            "beta": beta, "auroc": met["auroc"], "ap": met["ap"], "d_prime": dp,
            "robustness_feat": rf, "robustness_score": rs, "illum_shift": shift1,
            "trainable_params": n_params, "steps_used": rep["steps_used"],
            "std_ratio": rep["after"]["std"] / max(rep["before"]["std"], 1e-12),
            "pairwise_cos": rep["after"]["pairwise_cos"],
            "collapsed": bool(is_collapsed(rep)), "loss_finite": bool(rep["loss_finite"]),
            "runtime_s": round(time.time() - t0, 1)}
        print(f"    [{cat}|seed{seed}] {cfg} beta={beta:<5} AUROC={met['auroc']:.5f} "
              f"AP={met['ap']:.5f} d'={dp:.3f} rob_s={rs:.4f} shift={shift1:.4f} "
              f"steps={rep['steps_used']} collapse={results[cfg]['collapsed']}", flush=True)
        if gate and cfg == "C0":
            # 1e-6 reproduction gate: BETA-injection machinery must reproduce qualified R1
            ok = (abs(met["auroc"] - e16c["r1"]["auroc"]) <= TOL
                  and abs(met["ap"] - e16c["r1"]["ap"]) <= TOL)
            print(f"    [GATE] {cat}|seed{seed} C0 repro "
                  f"{'PASS' if ok else 'FAIL'} | auroc {met['auroc']:.7f} vs "
                  f"e16 {e16c['r1']['auroc']:.7f} | ap {met['ap']:.7f} vs "
                  f"{e16c['r1']['ap']:.7f}", flush=True)
            if not ok:
                raise RuntimeError(f"REPRO_GATE_FAIL {cat}|seed{seed}")

    gate_out = None
    if gate:
        c0 = results["C0"]
        ok = (abs(c0["auroc"] - e16c["r1"]["auroc"]) <= TOL
              and abs(c0["ap"] - e16c["r1"]["ap"]) <= TOL)
        gate_out = {"ok": bool(ok), "auroc": c0["auroc"], "e16_auroc": e16c["r1"]["auroc"],
                    "ap": c0["ap"], "e16_ap": e16c["r1"]["ap"]}
    return {"category": cat, "seed": seed, "extract_s": round(extract_s, 1),
            "wall_s": round(time.time() - t_start, 1), "configs": results, "gate": gate_out}


def adapted_feat_aug(adapter, aug_feat, dev, chunk=20):
    N, K, P, D_ = aug_feat.shape
    outs = np.empty((N, K, P, D_), dtype="float32")
    with torch.inference_mode():
        for i in range(0, N, chunk):
            z = torch.from_numpy(aug_feat[i:i + chunk].reshape(-1, D_)).to(dev)
            outs[i:i + chunk] = (z + adapter(z)).cpu().numpy().reshape(-1, K, P, D_)
    return outs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--round", default="1", choices=["1", "2"])
    ap.add_argument("--challengers", default="")
    ap.add_argument("--categories", default="Bird,Car,Cube")
    ap.add_argument("--seeds", default="")
    args = ap.parse_args()
    dev = torch.device("cuda")
    OUTDIR.mkdir(parents=True, exist_ok=True)
    assert (OUTDIR / "protocol.md").exists(), "freeze protocol first"
    e16 = load_e16()
    cats = [c.strip() for c in args.categories.split(",") if c.strip()]

    if args.round == "1":
        seeds, configs = [0], CONFIGS
    else:
        seeds = [int(s) for s in (args.seeds or "0,1,2").split(",") if s.strip()]
        names = [c.strip() for c in args.challengers.split(",") if c.strip()]
        assert 0 < len(names) <= 2, "Round 2 allows at most 2 challengers"
        configs = [c for c in CONFIGS if c[0] in names] + [("C0", 0.1)]

    partial = {"experiment": "E17", "round": args.round, "cells": {}}
    if PARTIAL.exists():
        try:
            old = json.loads(PARTIAL.read_text())
            if old.get("round") == args.round:
                partial["cells"] = old.get("cells", {})
        except json.JSONDecodeError:
            pass

    print(f"[E17] round={args.round} cats={cats} seeds={seeds} "
          f"configs={[c[0] for c in configs]} | disk {E8.free_gb():.1f} GB", flush=True)

    t_start = time.time()
    units = [(c, s) for c in cats for s in seeds]
    done = 0
    for cat, seed in units:
        key = f"{cat}|{seed}"
        if key in partial["cells"]:
            print(f"[E17] SKIP {key}", flush=True)
            done += 1
            continue
        print(f"\n[E17] RUN {cat} seed{seed}", flush=True)
        try:
            out = run_category_seed(cat, seed, dev, configs, e16[key], gate=(args.round == "1"))
        except Exception as exc:
            import traceback
            traceback.print_exc()
            print(f"[E17] {key} FAILED: {type(exc).__name__}: {exc}", flush=True)
            return 1
        if out["gate"] is not None and not out["gate"]["ok"]:
            print(f"[E17] REPRO_GATE_FAIL {key}: C0 auroc={out['gate']['auroc']:.7f} vs "
                  f"e16={out['gate']['e16_auroc']:.7f}", flush=True)
            return 2
        partial["cells"][key] = out
        tmp = PARTIAL.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(partial, indent=2, default=float))
        tmp.replace(PARTIAL)
        done += 1
        el = time.time() - t_start
        eta = el / done * (len(units) - done)
        print(f"[E17] STATUS {done}/{len(units)} | {100*done/len(units):.1f}% | "
              f"current={cat}/seed{seed} | elapsed={el/60:.1f}m ETA={eta/60:.1f}m", flush=True)

    write_csvs(partial, args.round)
    print(f"\n[E17] round {args.round} done -> {OUTDIR / ('round' + args.round + '.csv')}",
          flush=True)
    return 0


def write_csvs(partial, rnd):
    cells = partial["cells"]
    out = OUTDIR / f"round{rnd}.csv"
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["category", "seed", "config", "beta", "AUROC", "AP", "d_prime",
                    "robustness_feat", "robustness_score", "illum_shift", "steps_used",
                    "std_ratio", "pairwise_cos", "collapsed", "runtime_s"])
        for cat in CATS:
            for key, cell in cells.items():
                if not key.startswith(cat + "|"):
                    continue
                for cfg, r in cell["configs"].items():
                    w.writerow([cat, cell["seed"], cfg, r["beta"], f"{r['auroc']:.5f}",
                                f"{r['ap']:.5f}", f"{r['d_prime']:.4f}",
                                f"{r['robustness_feat']:.6f}", f"{r['robustness_score']:.6f}",
                                f"{r['illum_shift']:.6f}", r["steps_used"],
                                f"{r['std_ratio']:.3f}", f"{r['pairwise_cos']:.3f}",
                                r["collapsed"], r["runtime_s"]])


if __name__ == "__main__":
    raise SystemExit(main())
