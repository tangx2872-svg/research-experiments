#!/usr/bin/env python
"""E5-1 C01 — PIAD-Retinex input preparation (reflectance images).

PROVENANCE (frozen in `docs/E5_1_FROZEN_PROTOCOL.md` §4, verified at E5-1P):
  * repo    : github.com/Kaichen-Yang/piad_baseline @ 67b816b2a484317baa5b016cdce5dd52ec806ab1
  * module  : `Retinex/RL_separate.py` -> class `Separate`  (this is exactly what PIAD's own
              `pose_estimation.py:31` imports)
  * weights : `Retinex/ckpt/{init_low.pth, unfolding.pth}`  (committed in the repo)
  * license : `Retinex/LICENSE` = MIT, Copyright (c) 2022 AndersonYong  (PIAD vendors URetinex-Net)
  * output  : `Separate.forward()` -> R (reflectance / illumination-invariant).
              `I_enhance` from `test.py` is NOT used.

NO modification of the Retinex mechanism. Only the resolution is handled to match the frozen
pipeline: Original feeds `resize(img, 256)`; C01 feeds `Retinex(resize(img, 256))`.
Saving uses the official `np_save_TensorImg` path (x255, clip, uint8 PNG).

Writes ONLY to `data/m2ad/e5_1_retinex/` (git-ignored: under `data/`).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "third_party" / "piad_baseline"))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torchvision.transforms.functional as TF  # noqa: E402
from PIL import Image  # noqa: E402

import e4d1_m2ad_loader as LD  # noqa: E402

IMAGE_SIZE = 256
RETINEX_ROOT = ROOT / "third_party" / "piad_baseline" / "Retinex"
OUT_ROOT = ROOT / "data" / "m2ad" / "e5_1_retinex"


class _Opts:
    """Minimal opts object required by `Separate` (official signature)."""

    def __init__(self, retinex_root: Path) -> None:
        self.Decom_model_low_path = str(retinex_root / "ckpt" / "init_low.pth")
        self.unfolding_model_path = str(retinex_root / "ckpt" / "unfolding.pth")
        self.adjust_model_path = str(retinex_root / "ckpt" / "L_adjust.pth")
        self.ratio = 5          # official default; unused by `Separate`
        self.gpu_id = 0


def _vram() -> tuple[float, float]:
    return (torch.cuda.memory_allocated() / 2**20, torch.cuda.memory_reserved() / 2**20)


def build_separate():
    """Instantiate PIAD's official `Separate` (verbatim import path).

    ENGINEERING ADAPTATION (loading compatibility only, mechanism untouched):
    the official `ckpt/unfolding.pth` pickles an `argparse.Namespace` as `opts`.
    PyTorch >= 2.6 defaults `torch.load(weights_only=True)`, which rejects it.
    We allowlist that single global via the official API instead of editing PIAD code.
    """
    import argparse

    torch.serialization.add_safe_globals([argparse.Namespace])

    from Retinex.RL_separate import Separate  # official module (PIAD's own import path)
    from Retinex.utils import np_save_TensorImg  # official save helper

    opts = _Opts(RETINEX_ROOT)
    model = Separate(opts).cuda().eval()
    return model, np_save_TensorImg


def prepare(records: list[dict], out_dir: Path, tag: str, log_every: int = 200) -> dict:
    from Retinex.utils import np_save_TensorImg  # noqa: F401  (official helper, re-exported)

    model, save_fn = build_separate()
    out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    manifest = []
    total = len(records)
    for i, r in enumerate(records, 1):
        src = Path(r["image_path"])
        rel = src.relative_to(LD.DATA_ROOT)                      # Bird/Good/000/A000_I01.png
        dst = out_dir / rel
        dst.parent.mkdir(parents=True, exist_ok=True)

        img = Image.open(src).convert("RGB")
        img = TF.resize(img, [IMAGE_SIZE, IMAGE_SIZE], antialias=True)
        tensor = TF.to_tensor(img).unsqueeze(0)                  # [0,1] (1,3,256,256)

        R, _L, _t = model(tensor)                                # official `Separate.forward`
        if tuple(R.shape) != (1, 3, IMAGE_SIZE, IMAGE_SIZE):
            raise RuntimeError(f"C01 sanity S4 shape: {tuple(R.shape)}")
        if not torch.isfinite(R).all():
            raise RuntimeError(f"C01 sanity S5 non-finite: {src}")
        save_fn(R, str(dst))                                     # official save (uint8 PNG)

        manifest.append({"src": str(src), "dst": str(dst), "size": list(R.shape)})
        if i % log_every == 0 or i == total:
            el = time.time() - t0
            rate = i / el if el > 0 else 0.0
            remain = (total - i) / rate if rate > 0 else float("inf")
            a, rv = _vram()
            print(f"  [retinex {tag}] {i:5d}/{total:<5d} {100.0*i/total:5.1f}%  "
                  f"elapsed={time.strftime('%H:%M:%S', time.gmtime(el))}  "
                  f"ETA={time.strftime('%H:%M:%S', time.localtime(time.time()+remain))} "
                  f"(remain {remain/60:.1f} min)  {rate:6.1f} img/s  "
                  f"VRAM alloc={a:7.1f}MB reserved={rv:7.1f}MB", flush=True)

    info = {"tag": tag, "n_images": total, "seconds": round(time.time() - t0, 2),
            "images_per_s": round(total / max(1e-9, time.time() - t0), 3),
            "out_dir": str(out_dir), "image_size": IMAGE_SIZE,
            "peak_vram_alloc_mb": round(torch.cuda.max_memory_allocated() / 2**20, 1),
            "peak_vram_reserved_mb": round(torch.cuda.max_memory_reserved() / 2**20, 1)}
    (out_dir / "_prep_info.json").write_text(json.dumps(info, indent=2))
    return info


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--views", default="all", help="'all' = bank uses all 12 views (frozen); "
                                                   "'120' = only the frozen scoring view")
    ap.add_argument("--what", default="both", choices=["bank", "score", "both"])
    args = ap.parse_args()

    assert torch.cuda.is_available(), "C01 prep requires CUDA (official `Separate` calls .cuda())"
    recs = LD.load_records()
    est = LD.estimate_scope(recs)
    bank = sorted(LD.select(recs, set(est["bank_specimens"])), key=lambda r: r["image_path"])
    score = sorted([r for r in LD.select(recs, views=["120"]) if r["split"] == "test"],
                   key=lambda r: r["image_path"])
    print(f"[scope] bank={len(bank)} scoring(view 120)={len(score)}", flush=True)
    if len(bank) != 1200 or len(score) != 700:
        print("[ABORT] frozen scope mismatch (expect bank=1200, score=700)")
        return 3

    res = {}
    if args.what in ("bank", "both"):
        res["bank"] = prepare(bank, OUT_ROOT / "bank", "bank")
    if args.what in ("score", "both"):
        res["score"] = prepare(score, OUT_ROOT / "score", "score")
    print(json.dumps(res, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
