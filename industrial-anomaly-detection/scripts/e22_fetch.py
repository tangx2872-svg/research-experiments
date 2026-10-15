#!/usr/bin/env python
"""E22-P0/P1: acquire the 7 new M2AD categories with streaming selective extraction,
writing ONLY the l0_448 cache required by the frozen R1-ILA train/eval protocol.

Equivalence contract (established by the E22-P0 code audit of E16/E17):
  - E16/E17 read ONLY the l0_448 cache for train+eval:
      * bank pool = 10 train specimens (sorted(train_specs)[::3][:10]) x 12 views x 10 illum
                    = 1200 images (E8.bank_for draws 300 from these by frozen seed rule)
      * test      = view120 test split (official metadata; detectable filter at eval time)
      * R1 augmentation (brightness/contrast/gamma) is applied to the 448 image
        (e13a_screen.prep_image(img, fam, lv)) -> original resolution is NEVER needed
  - l0_448 generation path is identical to e8_fetch.write_image_caches:
        TF.resize(rgb, [448, 448], antialias=True, BICUBIC) -> WebP lossless q100 method 0
  - masks are NOT fetched (E22 is image-level only).

Engineering-only differences vs e8_fetch.py (no protocol change):
  - stops reading the archive as soon as ALL wanted members are extracted
  - only l0_448 is written (l0_256 / CLAHE caches are not part of any E16/E22 path)
  - per-member decode/cache failure stops the run (no silent skips)
  - results/e22/status.json = master record (written by main only)
  - results/e22/stream_progress.json = live per-member progress of the current category
    (fix 2026-10-15: before this fix the streaming state was written to status.json too,
     transiently clobbering the master record; the first E22 run used that earlier revision —
     see results/e22/README.md, no data impact, resume scan is cache-based)

CLI:
  python scripts/e22_fetch.py --categories Dice        # real fetch
  python scripts/e22_fetch.py --categories Dice --dry-run   # scope + skip scan only
"""
from __future__ import annotations

import argparse
import io
import json
import struct
import sys
import time
import zlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np  # noqa: E402
import torchvision.transforms.functional as TF  # noqa: E402
from PIL import Image  # noqa: E402

import e4d1_m2ad_loader as LD  # noqa: E402
import e8_common as E8  # noqa: E402
import e8_fetch as F8  # noqa: E402  (ZipStream, stream header parsing, _webp_lossless)

OUTDIR = ROOT / "results" / "e22"
STATUS = OUTDIR / "status.json"                 # master record (main only)
STREAM_STATUS = OUTDIR / "stream_progress.json"  # live progress of the current category
LOGDIR = OUTDIR / "logs"
TARGET_VIEW = "120"


def hms(s: float) -> str:
    s = int(s)
    return f"{s//3600}h{s%3600//60}m{s%60}s"


def free_gb() -> float:
    st = __import__("os").statvfs(str(ROOT))
    return st.f_bavail * st.f_frsize / 1e9


def u448_path(rel: str) -> Path:
    return E8.CACHE_ROOT / "l0_448" / Path(rel)


def build_e22_scope(category: str) -> dict:
    """pool (1200) + view120 test, official metadata authoritative."""
    scope = F8.build_scope(category)  # pool + 4-view test, mask paths redirected
    pool = scope["pool"]
    test120 = [r for r in scope["test"] if r["view"] == TARGET_VIEW]
    return {"pool": pool, "test": test120}


def write_l0448(rel: str, rgb: Image.Image) -> None:
    """l0_448-only writer; resize/webp parameters identical to write_image_caches."""
    dst = u448_path(rel)
    dst.parent.mkdir(parents=True, exist_ok=True)
    a448 = np.asarray(TF.resize(rgb, [E8.CACHE_RES, E8.CACHE_RES], antialias=True,
                                interpolation=TF.InterpolationMode.BICUBIC))
    F8._webp_lossless(a448, dst)


def stream_e22(category: str, wanted: set, on_member, state: dict, bar_every: int = 250):
    """Sequential remote-ZIP reader with early stop once all wanted members are kept."""
    url = f"{F8.HF_BASE}/{category}.zip"
    zs = F8.ZipStream(url)
    seen = kept = 0
    names_sample = []
    try:
        while True:
            hdr = zs.read_exact(30)
            if hdr is None or hdr[:4] != b"PK\x03\x04":
                break
            (_sig, _ver, flag, method, _t, _d, _crc, csize, _usize, nlen, elen) = struct.unpack(
                "<IHHHHHIIIHH", hdr)
            if flag & 0x08:
                raise RuntimeError(f"{category}: member uses a data descriptor (flag=0x08); "
                                   f"unsupported by streaming reader")
            name = zs.read_exact(nlen).decode("utf-8", "replace")
            if elen:
                zs.read_exact(elen)
            data = zs.read_exact(csize) if csize else b""
            seen += 1
            if len(names_sample) < 20:
                names_sample.append(name)
            if name in wanted:
                raw = zlib.decompress(data, -15) if method == 8 else data
                on_member(name, raw)
                kept += 1
            if bar_every and seen % bar_every == 0:
                el = time.time() - state["t0"]
                state["seen"] = seen
                state["kept"] = kept
                state["streamed_bytes"] = zs.total
                print(f"[E22 DATA] {category}: members={seen} kept={kept}/{len(wanted)} "
                      f"streamed={zs.total/2**30:.2f} GB  {(zs.total/2**20)/max(el,1):.1f} MB/s  "
                      f"elapsed={hms(el)}  free={free_gb():.1f} GB", flush=True)
                write_status(state, path=STREAM_STATUS)
            if kept >= len(wanted):
                print(f"[E22 DATA] {category}: ALL {kept} wanted members kept -> early stop "
                      f"(streamed {zs.total/2**30:.2f} GB of archive)", flush=True)
                break
    finally:
        zs.close()
    return seen, kept, zs.total, names_sample


def write_status(state: dict, path: Path | None = None) -> None:
    dst = path or STATUS
    OUTDIR.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(".json.tmp")
    with open(tmp, "w") as f:
        json.dump(state, f, indent=1)
    tmp.replace(dst)


def verify_category(category: str, pool: list, test120: list) -> dict:
    """Post-fetch gate: every wanted l0_448 file present + 448x448 RGB."""
    missing = []
    bad_shape = []
    for r in pool + test120:
        rel = F8.rel_of(r)
        p = u448_path(rel)
        if not p.exists():
            missing.append(rel)
            continue
        if len(bad_shape) < 5:
            with Image.open(p) as im:
                if im.size != (E8.CACHE_RES, E8.CACHE_RES) or im.mode not in ("RGB", "P"):
                    bad_shape.append((rel, im.size, im.mode))
    return {"n_expected": len(pool) + len(test120), "n_missing": len(missing),
            "missing_sample": missing[:10], "bad_shape": bad_shape}


def fetch_category(category: str, dry_run: bool = False) -> dict:
    scope = build_e22_scope(category)
    pool, test120 = scope["pool"], scope["test120"] if "test120" in scope else scope["test"]
    wanted = {F8.rel_of(r) for r in pool + test120}
    # resume scan: drop members whose cache already exists
    present = 0
    for rel in list(wanted):
        if u448_path(rel).exists():
            wanted.discard(rel)
            present += 1
    print(f"[E22 DATA] {category}: pool={len(pool)} test120={len(test120)} "
          f"wanted={len(wanted)}+{present} present  free={free_gb():.1f} GB", flush=True)
    info = {"pool": len(pool), "test120": len(test120), "n_wanted_total": len(pool) + len(test120),
            "n_present_before": present, "n_missing_before": len(wanted)}
    if dry_run:
        return info
    if not wanted:
        info["mode"] = "already_complete"
        return info

    state = {"t0": time.time(), "category": category, "seen": 0, "kept": 0,
             "streamed_bytes": 0, "wanted_initial": len(wanted)}
    def on_member(name: str, raw: bytes) -> None:
        try:
            img = Image.open(io.BytesIO(raw)).convert("RGB")
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(f"{category}: cannot decode {name}: {exc}") from exc
        write_l0448(name, img)

    seen, kept, nbytes, names = stream_e22(category, wanted, on_member, state)
    info.update({"mode": "streamed", "members_seen": seen, "members_kept": kept,
                 "streamed_gb": round(nbytes / 2**30, 2), "seconds": round(time.time() - state["t0"], 1),
                 "name_sample": names[:8]})
    return info


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--categories", required=True)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    cats = [c.strip() for c in args.categories.split(",") if c.strip()]

    OUTDIR.mkdir(parents=True, exist_ok=True)
    LOGDIR.mkdir(parents=True, exist_ok=True)
    state = {"task": "E22", "categories": {}, "current": None, "updated": time.time()}
    if STATUS.exists():
        try:
            state = json.load(open(STATUS))
        except Exception:  # noqa: BLE001
            pass

    for cat in cats:
        state["current"] = cat
        write_status(state)
        t0 = time.time()
        try:
            info = fetch_category(cat, dry_run=args.dry_run)
        except Exception as exc:  # noqa: BLE001
            state["categories"][cat] = {"fetch": "FAILED", "error": f"{type(exc).__name__}: {exc}",
                                        "seconds": round(time.time() - t0, 1)}
            write_status(state)
            print(f"[E22 DATA] {cat}: FETCH FAILED -> {exc}", flush=True)
            return 2
        if args.dry_run:
            print(f"[E22 DATA] {cat}: dry-run OK {info}", flush=True)
            continue
        scope = build_e22_scope(cat)
        gate = verify_category(cat, scope["pool"], scope["test"])
        status = "done" if gate["n_missing"] == 0 and not gate["bad_shape"] else "GATE_FAILED"
        state["categories"][cat] = {**info, "fetch": status, "gate": gate,
                                    "seconds": round(time.time() - t0, 1),
                                    "free_gb_after": round(free_gb(), 2)}
        write_status(state)
        print(f"[E22 DATA] {cat}: {status} expected={gate['n_expected']} "
              f"missing={gate['n_missing']} bad_shape={len(gate['bad_shape'])} "
              f"time={hms(time.time()-t0)} free={free_gb():.1f} GB", flush=True)
        if status != "done":
            return 3
    state["current"] = None
    write_status(state)
    print("[E22 DATA] all requested categories finished", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
