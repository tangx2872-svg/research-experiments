#!/usr/bin/env python
"""E20-B auxiliary analysis (CPU only) -- statistical resolution of image-level AUROC
at the MVTec AD 2 public-test sample sizes.

What it does
------------
Takes the *already frozen* per-image scores from E16 (M2AD view120, seed 0:
frozen F0 = `test_scores_f0`, R1-ILA = `test_scores_r1`, labels) and repeatedly
subsamples (bootstrap, with replacement) down to each MVTec AD 2 candidate scene's
public-test size (n_good, n_bad).  For every draw it computes

    auroc_f0, auroc_r1, delta = auroc_r1 - auroc_f0   (paired: same draw, same images)

The spread of `delta` answers one question for the preflight report:

    "if the true effect size were the M2AD one, how reliably could the MVTec AD 2
     public test set resolve it?"

Caveats that MUST travel with any use of these numbers
-----------------------------------------------------
* This is NOT MVTec AD 2 data. It resamples M2AD score distributions as a template.
  It estimates *sampling noise*, not the AD2 effect size.
* Bootstrap is with replacement and ignores image correlation across illumination
  groups; it is a lower bound on realistic uncertainty, not an upper bound.
* No model is trained; zero GPU.

Writes: results/e20/auroc_power_sim.json  (+ prints a compact table)
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parents[1]
E16P = ROOT / "results" / "e16" / "e16_partial.json"
OUT = ROOT / "results" / "e20"
OUT.mkdir(parents=True, exist_ok=True)

CATS = ["Bird", "Car", "Cube"]
# MVTec AD 2 candidate scenes: (n_good, n_bad) in test_public, from the official paper
AD2_TARGETS = {
    "sheet_metal": (24, 90),
    "vial": (35, 105),
    "fruit_jelly": (20, 60),
    "ad2_pooled_3_scenes": (79, 255),   # good/bad sums of the three candidates
}
N_BOOT = 4000
SEED = 20261015


def fast_auroc(scores, y):
    r = rankdata(scores)
    n1 = int(y.sum())
    n0 = len(y) - n1
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0)


def main():
    cells = json.loads(E16P.read_text())["cells"]
    rng = np.random.default_rng(SEED)
    res = {}
    for cat in CATS:
        c = cells[f"{cat}|0"]
        s0 = np.asarray(c["test_scores_f0"], float)
        s1 = np.asarray(c["test_scores_r1"], float)
        y = np.asarray(c["labels"], int)
        ig, ib = np.where(y == 0)[0], np.where(y == 1)[0]
        auroc_f0_full = fast_auroc(s0, y)
        auroc_r1_full = fast_auroc(s1, y)
        res[cat] = {
            "m2ad_full_n": {"good": int(len(ig)), "bad": int(len(ib))},
            "m2ad_full_auroc": {"f0": float(auroc_f0_full), "r1": float(auroc_r1_full),
                                "delta": float(auroc_r1_full - auroc_f0_full)},
            "scenes": {},
        }
        targets = dict(AD2_TARGETS)
        targets["m2ad_reference_full"] = (int(len(ig)), int(len(ib)))
        for scene, (ng, nb) in targets.items():
            d, a0, a1 = np.empty(N_BOOT), np.empty(N_BOOT), np.empty(N_BOOT)
            for b in range(N_BOOT):
                gi = rng.choice(ig, ng, replace=True)
                bi = rng.choice(ib, nb, replace=True)
                idx = np.concatenate([gi, bi])
                yy = y[idx]
                a0[b] = fast_auroc(s0[idx], yy)
                a1[b] = fast_auroc(s1[idx], yy)
                d[b] = a1[b] - a0[b]
            res[cat]["scenes"][scene] = {
                "n_good": ng, "n_bad": nb,
                "delta_mean": float(d.mean()),
                "delta_sd": float(d.std(ddof=1)),
                "delta_p2p5": float(np.percentile(d, 2.5)),
                "delta_p97p5": float(np.percentile(d, 97.5)),
                "delta_halfwidth95": float((np.percentile(d, 97.5)
                                            - np.percentile(d, 2.5)) / 2.0),
                "auroc_f0_halfwidth95": float((np.percentile(a0, 97.5)
                                               - np.percentile(a0, 2.5)) / 2.0),
                "p_delta_gt0": float((d > 0).mean()),
            }
    meta = {
        "source": "results/e16/e16_partial.json (M2AD view120, seed 0, frozen)",
        "method": ("paired bootstrap resample of per-image scores down to MVTec AD 2 "
                   "public-test sizes; sampling-noise estimate only, NOT AD2 data"),
        "n_boot": N_BOOT, "rng_seed": SEED,
        "ad2_targets": AD2_TARGETS,
    }
    (OUT / "auroc_power_sim.json").write_text(
        json.dumps({"meta": meta, "results": res}, indent=1))

    print(f"{'template':<6} {'scene':<12} {'n(g/b)':<8} {'dAUROC mean±sd':<18} "
          f"{'dHalf95':<8} {'AUROC half95':<12} {'P(d>0)':<7}")
    for cat in CATS:
        for scene, v in res[cat]["scenes"].items():
            print(f"{cat:<6} {scene:<12} {v['n_good']:>2}/{v['n_bad']:<4} "
                  f"{v['delta_mean']:+.4f} ± {v['delta_sd']:.4f} "
                  f"{'':<3}{v['delta_halfwidth95']:.4f} "
                  f"{'':<3}{v['auroc_f0_halfwidth95']:.4f} "
                  f"{'':<3}{v['p_delta_gt0']:.3f}")
    print(f"\nwrote {OUT / 'auroc_power_sim.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
