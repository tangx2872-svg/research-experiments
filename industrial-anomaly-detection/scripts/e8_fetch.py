#!/usr/bin/env python
"""E8 - Stage A data preparation.

Acquires the non-local M2AD categories with STREAMING SELECTIVE EXTRACTION of the official
Hugging Face mirror archives (protocol s2.8): members are DEFLATE with general-purpose flag 0,
so every local file header carries its own sizes and no central-directory random access is
needed. We read the archive sequentially, decompress in flight, keep only the frozen need-list
and DISCARD the original bytes - the four photometric caches of protocol s2.7 are written instead.

Nothing here changes the algorithm: the cache content is produced by the frozen E7 code paths.
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
import requests  # noqa: E402
import torchvision.transforms.functional as TF  # noqa: E402
from PIL import Image  # noqa: E402

import e4d1_m2ad_loader as LD  # noqa: E402
import e7_common as E7  # noqa: E402
import e8_common as E8  # noqa: E402

MASK_ROOT = ROOT / "data" / "m2ad" / "e8_gt"
HF_BASE = "https://hf-mirror.com/datasets/ChengYuQi99/M2AD/resolve/main"
KINDS = ["l0_256", "l0_448", "clahe_mild", "clahe_medium"]


# ------------------------------------------------------------------ scope for any category
def build_scope(category: str) -> dict:
    """Official metadata is authoritative; masks are re-pointed at the E8 GT root when the
    original extracted tree is not present."""
    recs = LD.load_records(category=category, require_exists=False)
    for r in recs:
        r["mask_member"] = ""
        if r["mask_path"]:
            rel = str(Path(r["mask_path"]).relative_to(LD.DATA_ROOT))
            r["mask_member"] = rel
            orig = Path(r["mask_path"])
            r["mask_path"] = str(orig) if orig.exists() else str(MASK_ROOT / rel)
    est = LD.estimate_scope(recs)
    pool = sorted(LD.select(recs, set(est["bank_specimens"])), key=lambda r: r["image_path"])
    test = sorted([r for r in LD.select(recs, views=E8.VIEWS) if r["split"] == "test"],
                  key=lambda r: r["image_path"])
    return {"category": category, "recs": recs, "est": est, "pool": pool, "test": test}


def rel_of(rec: dict, key: str = "image_path") -> str:
    return str(Path(rec[key]).relative_to(LD.DATA_ROOT))


def need_list(scope: dict) -> set:
    """Members required from the archive: full bank pool + unreduced 4-view test + their masks."""
    names = set()
    for r in scope["pool"] + scope["test"]:
        names.add(rel_of(r))
    for r in scope["test"]:
        if r.get("mask_member"):
            names.add(r["mask_member"])
    return names


# ------------------------------------------------------------------ remote sequential reader
class ZipStream:
    """Sequential reader over a remote ZIP delivered by an HTTP streaming response."""

    def __init__(self, url: str, chunk: int = 1 << 20, timeout: int = 300):
        self.resp = requests.get(url, stream=True, timeout=timeout)
        self.resp.raise_for_status()
        self.it = self.resp.iter_content(chunk_size=chunk)
        self.buf = bytearray()
        self.total = 0

    def _fill(self, n: int) -> bool:
        while len(self.buf) < n:
            try:
                c = next(self.it)
            except StopIteration:
                return False
            if c:
                self.buf += c
        return True

    def read_exact(self, n: int):
        if not self._fill(n):
            return None
        out = bytes(self.buf[:n])
        del self.buf[:n]
        self.total += n
        return out

    def close(self) -> None:
        try:
            self.resp.close()
        except Exception:  # noqa: BLE001
            pass


def stream_extract(category: str, wanted: set, on_member, bar=None, max_members: int | None = None):
    """Read the category archive sequentially; call on_member(name, raw) for wanted members.

    Returns (members_seen, members_kept, bytes_streamed).
    """
    url = f"{HF_BASE}/{category}.zip"
    zs = ZipStream(url)
    seen = kept = 0
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
            if name in wanted:
                raw = zlib.decompress(data, -15) if method == 8 else data
                on_member(name, raw)
                kept += 1
            if bar is not None and seen % 250 == 0:
                bar(seen, extra=f"kept={kept}/{len(wanted)} streamed={zs.total/2**30:.2f} GB")
            if max_members is not None and seen >= max_members:
                break
    finally:
        zs.close()
    return seen, kept, zs.total


# ------------------------------------------------------------------ cache writers (protocol s2.7)
def _webp_lossless(arr: np.ndarray, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr).save(dst, "WEBP", lossless=True, quality=100, method=0)


def _png(arr: np.ndarray, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr).save(dst, "PNG", compress_level=3)


def write_image_caches(rel: str, rgb: Image.Image, skip_existing: bool = True) -> int:
    """Produce the four frozen caches for one image from its decoded RGB original.

    Existing caches are left untouched (resume-friendly, and lets E7's CLAHE caches be reused).
    Returns the number of cache files written.
    """
    base = Path(rel)
    need = [k for k in KINDS
            if not (skip_existing and (E8.CACHE_ROOT / k / base).exists())]
    if not need:
        return 0
    written = 0
    if "l0_256" in need:
        a256 = np.asarray(TF.resize(rgb, [256, 256], antialias=True,
                                    interpolation=TF.InterpolationMode.BICUBIC))
        _png(a256, E8.CACHE_ROOT / "l0_256" / base)
        written += 1
    if any(k in need for k in ("l0_448", "clahe_mild", "clahe_medium")):
        a448 = np.asarray(TF.resize(rgb, [E8.CACHE_RES] * 2, antialias=True,
                                    interpolation=TF.InterpolationMode.BICUBIC))
        if "l0_448" in need:
            _webp_lossless(a448, E8.CACHE_ROOT / "l0_448" / base)
            written += 1
        if "clahe_mild" in need:
            _png(E7._apply_clahe(a448, E7.CLIP_MILD), E8.CACHE_ROOT / "clahe_mild" / base)
            written += 1
        if "clahe_medium" in need:
            _png(E7._apply_clahe(a448, E7.CLIP_MEDIUM), E8.CACHE_ROOT / "clahe_medium" / base)
            written += 1
    return written


def write_mask(rel: str, raw: bytes) -> None:
    dst = MASK_ROOT / Path(rel)
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_bytes(raw)


# ------------------------------------------------------------------ per-category preparation
def _log(msg: str) -> None:
    print(f"[E8 PREP] {msg}", flush=True)


def prepare_category(category: str, max_members: int | None = None) -> dict:
    scope = build_scope(category)
    wanted = need_list(scope)
    n_img = len([r for r in scope["pool"] + scope["test"]])
    _log(f"{category}: need {len(wanted)} members ({n_img} images + "
         f"{len(wanted)-n_img} masks), pool={len(scope['pool'])}, test={len(scope['test'])}")
    t0 = time.time()

    local_ok = all(Path(r["image_path"]).exists() for r in scope["pool"] + scope["test"])
    if local_ok:
        _log(f"{category}: originals present locally -> building caches locally")
        done = 0
        for r in scope["pool"] + scope["test"]:
            rel = rel_of(r)
            if all((E8.CACHE_ROOT / k / rel).exists() for k in KINDS):
                done += 1
                continue
            write_image_caches(rel, Image.open(r["image_path"]).convert("RGB"))
            done += 1
            if done % 250 == 0:
                el = time.time() - t0
                _log(f"{category}: {done}/{n_img} images cached  {done/el:.1f} img/s  "
                     f"elapsed={E8.hms(el)}  ETA={E8.hms((n_img-done)/(done/el))}  "
                     f"disk {E8.free_gb():.1f} GB")
        seen = kept = 0
        stats = {"mode": "local", "images": n_img, "masks": 0, "seconds": time.time() - t0}
    else:
        _log(f"{category}: originals not local -> streaming selective extraction")
        for r in scope["pool"] + scope["test"]:
            rel = rel_of(r)
            if all((E8.CACHE_ROOT / k / rel).exists() for k in KINDS):
                wanted.discard(rel)
        for r in scope["test"]:
            if r.get("mask_member") and Path(r["mask_path"]).exists():
                wanted.discard(r["mask_member"])
        _log(f"{category}: {len(wanted)} members still missing after cache scan")

        n_mask = 0
        state = {"i": 0}

        def on_member(name: str, raw: bytes) -> None:
            nonlocal n_mask
            if name.endswith(".png") and "/GT/" in name:
                write_mask(name, raw)
                n_mask += 1
                return
            try:
                img = Image.open(io.BytesIO(raw)).convert("RGB")
            except Exception as exc:  # noqa: BLE001
                raise RuntimeError(f"{category}: cannot decode {name}: {exc}") from exc
            write_image_caches(name, img)
            state["i"] += 1

        def bar(seen, extra=""):
            el = time.time() - t0
            _log(f"{category}: stream {seen} members  images={state['i']}/{n_img}  "
                 f"masks={n_mask}  {seen/el:.0f} mem/s  elapsed={E8.hms(el)}  "
                 f"disk {E8.free_gb():.1f} GB  {extra}")

        seen, kept, nbytes = stream_extract(category, wanted, on_member, bar=bar,
                                            max_members=max_members)
        stats = {"mode": "stream", "members_seen": seen, "members_kept": kept,
                 "bytes_streamed": nbytes, "images": state["i"], "masks": n_mask,
                 "seconds": time.time() - t0}

    stats["category"] = category
    stats["elapsed_hms"] = E8.hms(stats["seconds"])
    stats["need_members"] = len(wanted)
    _log(f"{category}: prep done in {stats['elapsed_hms']}  ({stats})")
    return stats


# ------------------------------------------------------------------ integrity gate (protocol s3.2)
def gate_category(category: str) -> dict:
    scope = build_scope(category)
    pool, test = scope["pool"], scope["test"]
    res = {}

    expected = len(pool) + len(test)
    missing = []
    for kind in KINDS:
        for r in pool + test:
            if not (E8.CACHE_ROOT / kind / rel_of(r)).exists():
                missing.append(f"{kind}/{rel_of(r)}")
    res["A1_file_count"] = {"pass": len(missing) == 0,
                            "detail": f"{category}: {expected} images x {len(KINDS)} kinds; "
                                      f"{len(missing)} missing" + (f" e.g. {missing[:2]}" if missing else "")}

    bad = []
    for kind in KINDS:
        for r in (pool + test)[::37]:
            p = E8.CACHE_ROOT / kind / rel_of(r)
            if not p.exists():
                continue
            try:
                arr = np.asarray(Image.open(p).convert("RGB"))
            except Exception as exc:  # noqa: BLE001
                bad.append(f"{kind}/{rel_of(r)}: {exc}")
                continue
            want = (256, 256) if kind == "l0_256" else (E8.CACHE_RES, E8.CACHE_RES)
            if arr.shape[:2] != want or arr.dtype != np.uint8:
                bad.append(f"{kind}/{rel_of(r)}: shape {arr.shape} dtype {arr.dtype}")
    res["A2_decode"] = {"pass": not bad, "detail": f"{category}: sampled decode check, "
                                                   f"{len(bad)} bad" + (f" e.g. {bad[:2]}" if bad else "")}

    with_mask = [r for r in test if r["mask_path"]]
    miss_m = [r["mask_path"] for r in with_mask if not Path(r["mask_path"]).exists()]
    res["A3_mask_alignment"] = {"pass": not miss_m,
                                "detail": f"{category}: {len(with_mask)}/{len(test)} test records carry a "
                                          f"mask; {len(miss_m)} missing"}

    views = sorted({r["view"] for r in test})
    ills = sorted({r["illumination"] for r in test})
    res["A4_illumination"] = {"pass": views == sorted(E8.VIEWS) and ills == [f"{i:02d}" for i in range(1, 11)],
                              "detail": f"{category}: views={views} illum={ills}"}

    n_good = sum(1 for r in test if r["kind"] == "Good")
    n_ng = sum(1 for r in test if r["kind"] == "NG")
    res["A5_label_alignment"] = {"pass": n_good == 800 and n_ng == 2000,
                                 "detail": f"{category}: test Good={n_good} (expect 800), "
                                           f"NG={n_ng} (expect 2000), total={len(test)}"}
    return res


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--categories", default="Car,Cube")
    ap.add_argument("--max-members", type=int, default=None, help="smoke: stop after N members")
    ap.add_argument("--gate-only", action="store_true")
    args = ap.parse_args()
    cats = [c.strip() for c in args.categories.split(",") if c.strip()]
    out = {}
    for c in cats:
        if not args.gate_only:
            out[c] = {"prep": prepare_category(c, max_members=args.max_members)}
        g = gate_category(c)
        out.setdefault(c, {})["gate"] = g
        for k, v in g.items():
            print(f"  {k}: {'PASS' if v['pass'] else 'FAIL'}  {v['detail']}", flush=True)
        if not all(v["pass"] for v in g.values()):
            print(f"[E8 PREP] {c}: GATE FAILED - stopping for this category (not silently skipped)",
                  flush=True)
            return 4
    (E8.OUT).mkdir(parents=True, exist_ok=True)
    (E8.OUT / "prep.json").write_text(json.dumps(out, indent=2, default=float))
    print(f"[E8 PREP] wrote {E8.OUT/'prep.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
