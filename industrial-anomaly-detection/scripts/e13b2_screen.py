#!/usr/bin/env python
"""E13-B2 - Bank sensitivity + extended stability sweep for R1.

Reuses the R1 cell computation from e13b_screen.run_cell (identical method/loss/lr/
augmentation/scoring). Only the output dir and seed range differ. Two stages:
  Stage A = Cube x seed3-7, then decide STABLE/BORDERLINE/UNSTABLE.
  Stage B = Bird/Car x seed3-7 (if not UNSTABLE).

Frozen protocol: results/e13b2/e13b2_protocol.json
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

import e8_common as E8  # noqa: E402
from e13b_screen import run_cell  # noqa: E402

OUTDIR = ROOT / "results" / "e13b2"
PARTIAL = OUTDIR / "e13b2_partial.json"
CSV = OUTDIR / "e13b2_results.csv"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--categories", default="Cube")
    ap.add_argument("--seeds", default="3,4,5,6,7")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    dev = torch.device("cuda")
    cats = [c.strip() for c in args.categories.split(",") if c.strip()]
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    OUTDIR.mkdir(parents=True, exist_ok=True)

    print(f"[E13-B2] cats={cats} seeds={seeds} smoke={args.smoke} | disk {E8.free_gb():.1f} GB | "
          f"GPU {E8.gpu_util()}", flush=True)

    partial = {"experiment": "E13-B2", "cells": {}}
    if PARTIAL.exists():
        try:
            partial.update(json.loads(PARTIAL.read_text()))
        except json.JSONDecodeError:
            pass

    t_start = time.time()
    total = len(cats) * len(seeds)
    done_cnt = 0
    for cat in cats:
        for seed in seeds:
            key = f"{cat}|{seed}"
            if not args.smoke and key in partial.get("cells", {}):
                print(f"[E13-B2] SKIP {key} (done)", flush=True)
                done_cnt += 1
                continue
            print(f"\n[E13-B2] RUN {cat} seed{seed}", flush=True)
            try:
                out = run_cell(cat, seed, dev)
            except Exception as exc:
                import traceback
                traceback.print_exc()
                print(f"[E13-B2] {cat} seed{seed} FAILED: {type(exc).__name__}: {exc}", flush=True)
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
            eta = per * (total - done_cnt)
            print(f"[E13-B2] PROGRESS new={done_cnt}/{total} | elapsed={el/60:.1f}m | "
                  f"ETA={eta/60:.1f}m | {cat} seed{seed} R0={out['r0']['auroc']:.5f} "
                  f"R1={out['r1']['auroc']:.5f} dAUROC={out['d_auroc']:+.5f} "
                  f"dAP={out['d_ap']:+.5f}", flush=True)

    if args.smoke:
        print("[E13-B2] SMOKE OK (no checkpoint/CSV written)", flush=True)
        return 0

    write_csv(partial)
    print(f"\n[E13-B2] done -> {CSV}", flush=True)
    return 0


def write_csv(partial):
    w = csv.writer(open(CSV, "w", newline=""))
    w.writerow(["category", "seed", "r0_auroc", "r0_ap", "r1_auroc", "r1_ap",
                "d_auroc", "d_ap", "steps_used", "std_ratio", "pairwise_cos", "status"])
    for key, cell in sorted(partial.get("cells", {}).items()):
        cat, seed = key.split("|")
        w.writerow([cat, seed, f"{cell['r0']['auroc']:.5f}", f"{cell['r0']['ap']:.5f}",
                    f"{cell['r1']['auroc']:.5f}", f"{cell['r1']['ap']:.5f}",
                    f"{cell['d_auroc']:+.5f}", f"{cell['d_ap']:+.5f}",
                    cell["steps_used"], f"{cell['std_ratio']:.3f}",
                    f"{cell['pairwise_cos']:.3f}",
                    "STOP_COLLAPSE" if cell["collapsed"] else "OK"])


if __name__ == "__main__":
    raise SystemExit(main())
