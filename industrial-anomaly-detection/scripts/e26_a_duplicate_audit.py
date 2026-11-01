#!/usr/bin/env python
"""E26-A: RobustAD near-duplicate audit of the MetalParts train/test0 name overlap.

Read-only audit. Decodes images, computes byte identity, pixel identity, pHash, dHash,
SSIM and a heuristic classification, then writes a manifest, a full table, a top-20
shortlist and two contact sheets.

Nothing is written back to the dataset; no experiment is re-run; no training happens.

Pair definition (copied from the E25 overlap audit, not re-invented):
    MetalParts `metal_parts_data_dir_train` vs `metal_parts_data_dir_test0`,
    joined on the exact string `file_name` from each split's metadata.jsonl.

Usage
-----
    python scripts/e26_a_duplicate_audit.py [--data-root data/robustad]
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.fftpack import dct
from skimage.metrics import structural_similarity as skimage_ssim

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = ROOT / "data" / "robustad"
OUT = ROOT / "results" / "e26" / "robustad_duplicate_audit"
DOMAIN = "MetalParts"
SPLIT_A = "train"
SPLIT_B = "test0"
DIRS = {"train": "metal_parts_data_dir_train", "test0": "metal_parts_data_dir_test0"}

# Auxiliary classification thresholds. They are a screening heuristic for human review,
# NOT a leak verdict: identical industrial parts under different lighting/pose are
# expected to score high and must not be called duplicates.
TH_NEAR = {"phash": 6, "dhash": 8, "ssim": 0.95}
TH_SCENE = {"ssim": 0.80, "phash": 12, "dhash": 14}


def read_metadata(split_dir: Path) -> dict:
    rows = [json.loads(line) for line in open(split_dir / "metadata.jsonl")]
    return {r["file_name"]: r for r in rows}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def phash(path: Path, hash_size: int = 8, highfreq_factor: int = 4) -> int:
    """Standard DCT pHash: 32x32 gray -> DCT -> low 8x8 (DC dropped) -> median bits."""
    img = Image.open(path).convert("L").resize(
        (hash_size * highfreq_factor, hash_size * highfreq_factor), Image.LANCZOS)
    arr = np.asarray(img, dtype=np.float64)
    dctd = dct(dct(arr, axis=0, norm="ortho"), axis=1, norm="ortho")
    low = dctd[:hash_size, :hash_size].flatten()[1:]          # drop DC term
    bits = low > np.median(low)
    val = 0
    for b in bits:
        val = (val << 1) | int(b)
    return val


def dhash(path: Path, hash_size: int = 8) -> int:
    """Difference hash: 9x8 gray, bit = left pixel brighter than right pixel."""
    img = Image.open(path).convert("L").resize((hash_size + 1, hash_size), Image.LANCZOS)
    arr = np.asarray(img, dtype=np.int16)
    bits = arr[:, 1:] > arr[:, :-1]
    val = 0
    for b in bits.flatten():
        val = (val << 1) | int(b)
    return val


def hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


def pair_ssim(path_a: Path, path_b: Path) -> tuple[float, str]:
    """Grayscale SSIM at native resolution; only resizes when the sizes differ."""
    a = Image.open(path_a).convert("L")
    b = Image.open(path_b).convert("L")
    if a.size == b.size:
        policy = "native"
    else:
        b = b.resize(a.size, Image.LANCZOS)
        policy = "b_resized_to_a_LANCZOS"
    aa = np.asarray(a, dtype=np.float64)
    bb = np.asarray(b, dtype=np.float64)
    win = min(7, min(aa.shape))
    if win % 2 == 0:
        win -= 1
    val = skimage_ssim(aa, bb, data_range=255.0, gaussian_weights=True,
                       sigma=1.5, use_sample_covariance=False, win_size=max(3, win))
    return float(val), policy


def pixel_identity(path_a: Path, path_b: Path) -> bool:
    a = Image.open(path_a).convert("RGB")
    b = Image.open(path_b).convert("RGB")
    if a.size != b.size:
        return False
    return bool(np.array_equal(np.asarray(a), np.asarray(b)))


def mad_64_gray(path_a: Path, path_b: Path) -> float:
    a = Image.open(path_a).convert("L").resize((64, 64), Image.LANCZOS)
    b = Image.open(path_b).convert("L").resize((64, 64), Image.LANCZOS)
    return float(np.abs(np.asarray(a, np.float64) - np.asarray(b, np.float64)).mean())


def classify(sha_equal: bool, pixel_equal: bool, ph: int, dh: int, ss: float) -> str:
    if sha_equal or pixel_equal:
        return "EXACT_DUPLICATE"
    if ph <= TH_NEAR["phash"] and dh <= TH_NEAR["dhash"] and ss >= TH_NEAR["ssim"]:
        return "VERY_LIKELY_NEAR_DUPLICATE"
    if ss >= TH_SCENE["ssim"] or (ph <= TH_SCENE["phash"] and dh <= TH_SCENE["dhash"]):
        return "SIMILAR_SCENE_OR_OBJECT"
    return "NO_OBVIOUS_DUPLICATE"


def build_pairs(data_root: Path) -> tuple[list[dict], dict]:
    tr_dir = data_root / DOMAIN / DIRS["train"]
    t0_dir = data_root / DOMAIN / DIRS["test0"]
    tr, t0 = read_metadata(tr_dir), read_metadata(t0_dir)
    shared = sorted(set(tr) & set(t0))
    pairs = [{
        "pair_id": f"MP_{i:03d}",
        "domain": DOMAIN,
        "filename": name,
        "label_a": tr[name]["label"],
        "label_b": t0[name]["label"],
        "split_a": SPLIT_A,
        "split_b": SPLIT_B,
        "path_a": str(Path(DIRS["train"]) / name),
        "path_b": str(Path(DIRS["test0"]) / name),
    } for i, name in enumerate(shared, 1)]
    meta = {
        "n_train_entries": len(tr),
        "n_test0_entries": len(t0),
        "n_name_overlap": len(shared),
        "n_label0_pairs": sum(1 for p in pairs if p["label_a"] == 0),
        "n_label1_pairs": sum(1 for p in pairs if p["label_a"] == 1),
        "n_same_label": sum(1 for p in pairs if p["label_a"] == p["label_b"]),
        "train_dir": str(tr_dir.relative_to(ROOT)),
        "test0_dir": str(t0_dir.relative_to(ROOT)),
    }
    return pairs, meta


def compute(pairs: list[dict], data_root: Path) -> list[dict]:
    rows = []
    for p in pairs:
        pa = data_root / DOMAIN / p["path_a"]
        pb = data_root / DOMAIN / p["path_b"]
        rec = dict(p)
        rec["exists_a"], rec["exists_b"] = pa.exists(), pb.exists()
        if not (rec["exists_a"] and rec["exists_b"]):
            rows.append(rec)
            continue
        sha_a, sha_b = sha256_file(pa), sha256_file(pb)
        wa, ha = Image.open(pa).size
        wb, hb = Image.open(pb).size
        ph_a, ph_b = phash(pa), phash(pb)
        dh_a, dh_b = dhash(pa), dhash(pb)
        ss, policy = pair_ssim(pa, pb)
        rec.update({
            "sha256_a": sha_a,
            "sha256_b": sha_b,
            "sha256_equal": sha_a == sha_b,
            "bytes_a": pa.stat().st_size,
            "bytes_b": pb.stat().st_size,
            "width_a": wa, "height_a": ha, "width_b": wb, "height_b": hb,
            "exact_pixel_equal": pixel_identity(pa, pb),
            "phash_a": f"{ph_a:016x}", "phash_b": f"{ph_b:016x}",
            "phash_hamming": hamming(ph_a, ph_b),
            "dhash_a": f"{dh_a:016x}", "dhash_b": f"{dh_b:016x}",
            "dhash_hamming": hamming(dh_a, dh_b),
            "ssim": round(ss, 6),
            "ssim_resize_policy": policy,
            "mean_abs_diff_gray_64": round(mad_64_gray(pa, pb), 3),
            "lpips": "SKIPPED",
        })
        rec["hamming_sum"] = rec["phash_hamming"] + rec["dhash_hamming"]
        rec["auto_class"] = classify(rec["sha256_equal"], rec["exact_pixel_equal"],
                                     rec["phash_hamming"], rec["dhash_hamming"], ss)
        rows.append(rec)
    return rows


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def contact_sheet(rows: list[dict], data_root: Path, path: Path, title: str,
                  thumb_w: int = 620, dpi: int = 180) -> None:
    cols = [("A", "path_a", "split_a"), ("B", "path_b", "split_b")]
    thumb_w = 620
    thumb_h = int(thumb_w * 1500 / 2681)          # keep the 16:9 frame
    pad, text_w = 14, 700
    row_h = thumb_h + 2 * pad
    width = 2 * thumb_w + text_w + 4 * pad
    height = int(72 + row_h * len(rows) + pad)
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 19)
        small = ImageFont.truetype("DejaVuSans.ttf", 17)
    except OSError:
        font = small = ImageFont.load_default()
    draw.text((pad, 18), title, fill="black", font=font)

    for i, r in enumerate(rows):
        y = 72 + i * row_h
        for j, (tag, key, split_key) in enumerate(cols):
            x = pad + j * (thumb_w + pad)
            img = Image.open(data_root / DOMAIN / r[key]).convert("RGB")
            img.thumbnail((thumb_w, thumb_h), Image.LANCZOS)
            canvas.paste(img, (x, y + pad))
            draw.text((x + 4, y + pad - 2), f"{tag} ({r[split_key]})", fill="black", font=small)
        tx = pad + 2 * (thumb_w + pad)
        draw.multiline_text(
            (tx, y + pad),
            f"rank {r['rank']:02d}  {r['pair_id']}\n"
            f"{r['filename'].split('/')[-1]}\n"
            f"{r['filename'].split('/')[0]}  label={r['label_a']}\n"
            f"pHash d={r['phash_hamming']}  dHash d={r['dhash_hamming']}\n"
            f"SSIM={r['ssim']:.4f}  LPIPS=SKIPPED\n"
            f"MAD64={r['mean_abs_diff_gray_64']:.2f}\n"
            f"auto: {r['auto_class']}",
            fill="black", font=small, spacing=5)
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path, dpi=(dpi, dpi))
    print(f"[E26-A] wrote {path.name} ({canvas.size[0]}x{canvas.size[1]})", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path, default=DEFAULT_DATA)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    pairs, meta = build_pairs(args.data_root)
    print(f"[E26-A] pairs={len(pairs)} (train {meta['n_train_entries']}, "
          f"test0 {meta['n_test0_entries']})", flush=True)
    rows = compute(pairs, args.data_root)

    write_csv(OUT / "pair_manifest.csv", pairs,
              ["pair_id", "domain", "filename", "label_a", "label_b", "split_a", "split_b",
               "path_a", "path_b"])

    measured = [r for r in rows if "ssim" in r]
    ranked = sorted(measured, key=lambda r: (r["hamming_sum"], -r["ssim"]))
    for i, r in enumerate(ranked, 1):
        r["rank"] = i
    all_rows = ranked + [r for r in rows if "ssim" not in r]

    fields = ["rank", "pair_id", "domain", "filename", "label_a", "label_b", "split_a", "split_b",
              "path_a", "path_b", "exists_a", "exists_b", "sha256_a", "sha256_b", "sha256_equal",
              "bytes_a", "bytes_b", "width_a", "height_a", "width_b", "height_b",
              "exact_pixel_equal", "phash_a", "phash_b", "phash_hamming",
              "dhash_a", "dhash_b", "dhash_hamming", "hamming_sum", "ssim",
              "ssim_resize_policy", "mean_abs_diff_gray_64", "lpips", "auto_class"]
    write_csv(OUT / "all_pairs.csv", all_rows, fields)
    write_csv(OUT / "top20.csv", ranked[:20], fields)

    contact_sheet(ranked[:10], args.data_root, OUT / "top01_10_contact_sheet.png",
                  "RobustAD MetalParts train/test0 - top-10 most similar name-matched pairs")
    contact_sheet(ranked[10:20], args.data_root, OUT / "top11_20_contact_sheet.png",
                  "RobustAD MetalParts train/test0 - ranks 11-20")

    ss = np.array([r["ssim"] for r in measured], dtype=float)
    ph = np.array([r["phash_hamming"] for r in measured], dtype=int)
    dh = np.array([r["dhash_hamming"] for r in measured], dtype=int)
    mad = np.array([r["mean_abs_diff_gray_64"] for r in measured], dtype=float)
    summary = {
        "date": "2026-11-01",
        "experiment": "E26-A",
        "pair_definition": "same exact `file_name` string in MetalParts train metadata.jsonl "
                           "and test0 metadata.jsonl (the E25 definition, re-derived)",
        "counts": {
            **meta,
            "n_pairs_measured": len(measured),
            "n_missing_on_disk": len(rows) - len(measured),
            "n_sha256_equal": int(sum(r["sha256_equal"] for r in measured)),
            "n_pixel_equal": int(sum(bool(r["exact_pixel_equal"]) for r in measured)),
            "n_same_dimensions": int(sum(r["width_a"] == r["width_b"] and r["height_a"] == r["height_b"]
                                        for r in measured)),
        },
        "class_counts": {c: sum(1 for r in measured if r["auto_class"] == c)
                         for c in ("EXACT_DUPLICATE", "VERY_LIKELY_NEAR_DUPLICATE",
                                   "SIMILAR_SCENE_OR_OBJECT", "NO_OBVIOUS_DUPLICATE")},
        "distributions": {
            "ssim": {"min": float(ss.min()), "p05": float(np.percentile(ss, 5)),
                     "median": float(np.median(ss)), "max": float(ss.max())},
            "phash_hamming": {"min": int(ph.min()), "median": float(np.median(ph)),
                              "max": int(ph.max())},
            "dhash_hamming": {"min": int(dh.min()), "median": float(np.median(dh)),
                              "max": int(dh.max())},
            "mean_abs_diff_gray_64": {"min": float(mad.min()), "median": float(np.median(mad)),
                                      "max": float(mad.max())},
        },
        "thresholds_auxiliary": {"near_duplicate": TH_NEAR, "similar_scene": TH_SCENE},
        "ranking_rule": "ascending (pHash Hamming + dHash Hamming), tie-break SSIM descending; "
                        "a review order, not a leak verdict",
        "ssim_policy": "grayscale, native resolution; only resized (b -> a, LANCZOS) when the two "
                       "sizes differ; gaussian_weights=True, sigma=1.5, data_range=255",
        "lpips": "SKIPPED - the package is absent from the frozen environment and installing it "
                 "would add dependencies; pHash + dHash + SSIM were used instead",
        "environment": {"python": "3.12.3", "numpy": np.__version__, "pillow": Image.__version__},
    }
    (OUT / "audit_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary["counts"], indent=1), flush=True)
    print(json.dumps(summary["class_counts"], indent=1), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
