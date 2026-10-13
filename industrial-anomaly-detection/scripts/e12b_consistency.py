#!/usr/bin/env python
"""E12-B - compute the 'BEFORE' illumination consistency (fixed equal-weight sum fusion).

The trained-gate 'AFTER' consistency is already in e12b_training.csv (final_cons).
This helper re-extracts bank-normal features + their 6 mild augmentations and computes,
per category, the illumination consistency of the FIXED equal-weight fusion
(F_ref = normalize(0.5*n(f9) + 0.5*n(ffinal))), which is the gate initialization point.

Also computes the final-only (S0) consistency as a secondary reference.
Output: results/e12b/e12b_consistency.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

import e4d1_m2ad_loader as LD  # noqa: E402
import e8_common as E8  # noqa: E402
from e11_p2_runner import route  # noqa: E402
from e12b_screen import (extract_originals, extract_augmented, load_dinov2_local,  # noqa: E402
                         _norm, N_PATCH, D)

OUT = ROOT / "results" / "e12b" / "e12b_consistency.json"


def fixed_ref_consistency(f9, ff, aug_f9, aug_ff, dev):
    """1 - mean cos(F_ref(x), F_ref(T_illum(x))) over all (img, aug) pairs, F_ref = equal-weight sum."""
    N = f9.shape[0]
    K = aug_f9.shape[1]
    acc = 0.0
    cnt = 0
    with torch.inference_mode():
        for i in range(N):
            r9 = torch.from_numpy(f9[i].reshape(N_PATCH, D)).to(dev)
            rf = torch.from_numpy(ff[i].reshape(N_PATCH, D)).to(dev)
            F_ref = _norm(0.5 * _norm(r9) + 0.5 * _norm(rf))
            for k in range(K):
                a9 = torch.from_numpy(aug_f9[i, k].reshape(N_PATCH, D)).to(dev)
                af = torch.from_numpy(aug_ff[i, k].reshape(N_PATCH, D)).to(dev)
                F_aug = _norm(0.5 * _norm(a9) + 0.5 * _norm(af))
                acc += float((1.0 - (F_ref * F_aug).sum(dim=1)).mean().item())
                cnt += 1
    return acc / max(1, cnt)


def final_only_consistency(ff, aug_ff, dev):
    """1 - mean cos(n_final(x), n_final(T_illum(x)))."""
    N = ff.shape[0]
    K = aug_ff.shape[1]
    acc = 0.0
    cnt = 0
    with torch.inference_mode():
        for i in range(N):
            rf = torch.from_numpy(ff[i].reshape(N_PATCH, D)).to(dev)
            F = _norm(rf)
            for k in range(K):
                af = torch.from_numpy(aug_ff[i, k].reshape(N_PATCH, D)).to(dev)
                F_aug = _norm(af)
                acc += float((1.0 - (F * F_aug).sum(dim=1)).mean().item())
                cnt += 1
    return acc / max(1, cnt)


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--categories", default="Bird,Car,Cube")
    args = ap.parse_args()
    cats = [c.strip() for c in args.categories.split(",") if c.strip()]
    dev = torch.device("cuda")
    model = load_dinov2_local(dev)

    out = {}
    for cat in cats:
        recs = LD.load_records(category=cat, require_exists=False)
        est = LD.estimate_scope(recs)
        bank_specs = set(est["bank_specimens"])
        bank = E8.bank_for(sorted(LD.select(recs, bank_specs), key=lambda r: r["image_path"]), 0)
        bank_r = route(bank, "l0_448")
        print(f"[E12-B CONS] {cat}: extract {len(bank)} bank normals + aug", flush=True)
        f9, ff = extract_originals(model, bank_r, dev, log_every=100)
        aug_f9, aug_ff = extract_augmented(model, bank_r, dev, log_every=100)
        before_ref = fixed_ref_consistency(f9, ff, aug_f9, aug_ff, dev)
        before_s0 = final_only_consistency(ff, aug_ff, dev)
        out[cat] = {"fixed_equal_weight_consistency": round(before_ref, 6),
                    "final_only_consistency": round(before_s0, 6)}
        print(f"[E12-B CONS] {cat}: equal-weight before={before_ref:.5f} "
              f"final-only before={before_s0:.5f}", flush=True)
        del f9, ff, aug_f9, aug_ff
        torch.cuda.empty_cache()

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2))
    print(f"[E12-B CONS] wrote {OUT}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
