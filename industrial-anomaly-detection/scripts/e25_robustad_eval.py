#!/usr/bin/env python
"""E25 - RobustAD external generalization test of the FROZEN R1-ILA method.

Protocol (frozen before any GPU result): results/e25/ROBUSTAD_PROTOCOL_AUDIT.md §8 + §10.
Legal hyperparameter source (E25 amendment, 2026-10-18): results/e17/final_method.json ONLY.
results/e24/ is neither created nor referenced.

Design
------
* One model per category (PCB, MetalParts). Bank = ALL source-train normals (180),
  adapter train subset = first 120, validation = last 60 (bank-size constraint documented
  in the protocol; no hyperparameter is changed).
* B0 = frozen DINOv2-B final-layer patch tokens + L2 + 1-NN-max on the same bank.
* B1 = frozen R1-ILA (z' = z + Wz, BETA=0.1, aug brightness/contrast/gamma x {0.85,1.15},
  lr 1e-3, <=300 steps, patience 30, batch 8, seed 0) on the same bank.
* Labels come ONLY from each split's metadata.jsonl (never from directory names).
* MetalParts test0 is excluded from all aggregates (145-file overlap with train) and is
  stored with valid=0.
* Metrics are plain image-level AUROC/AP over all images (RobustAD has no `detectable`
  field; this is NOT comparable to the M2AD `official` convention).
* Per-domain CSV is rewritten after EVERY split, so a crash never loses a finished domain.
* Resume: an existing results/e25/cells/<domain>_<split>.json with status OK is skipped.

Reused imports only (no frozen file is modified):
  e13a_screen  -> DINOv2 loader, prep_image, LinearAdapter, adapted_feat_adapter,
                  forward_frozen_batch, l2norm, AUG_LIST, constants
  e13b_screen  -> train_r1 (the E13-A/E13-B/E22 frozen R1 training loop)
  e11_p2_runner-> nn_image_scores (1-NN-max, identical to every earlier experiment)

CLI
  python scripts/e25_robustad_eval.py --smoke
  python scripts/e25_robustad_eval.py --domain PCB
  python scripts/e25_robustad_eval.py --domain MetalParts
  python scripts/e25_robustad_eval.py --all
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torchvision.transforms as T  # noqa: E402
from PIL import Image  # noqa: E402

import e13a_screen as E13A  # noqa: E402
from e13a_screen import (load_dinov2_local, freeze_backbone, LinearAdapter,  # noqa: E402
                         adapted_feat_adapter, prep_image, forward_frozen_batch,
                         l2norm, D, N_PATCH, AUG_LIST)
from e13b_screen import train_r1  # noqa: E402
from e11_p2_runner import nn_image_scores  # noqa: E402

DATA = ROOT / "data" / "robustad"
OUTDIR = ROOT / "results" / "e25"
CELLDIR = OUTDIR / "cells"
PARTIAL = OUTDIR / "partial.json"
FINAL_METHOD = ROOT / "results" / "e17" / "final_method.json"

# domain -> (directory prefix, split -> condition group, invalid splits)
DOMAINS = {
    "PCB": {
        "prefix": "pcb_data_dir_",
        "group": {"test0": "source_reference", "test1": "illumination",
                  "test2": "illumination_related_white_balance", "test3": "pose",
                  "test4": "pose", "test5": "illumination_related_shadow"},
        "invalid": {},
    },
    "MetalParts": {
        "prefix": "metal_parts_data_dir_",
        "group": {"test0": "source_reference_INVALID", "test1": "illumination",
                  "test2": "pose", "test3": "pose", "test4": "scale",
                  "test5": "background", "test6": "background"},
        "invalid": {"test0": "145 file names overlap source train (not held-out)"},
    },
}

BANK_TRAIN_N = 120          # documented bank-size constraint, NOT a tuned hyperparameter
BANK_VAL_N = 60
EXTRACT_BS = 8              # images per forward pass (DINOv2-B @448)
AUG_BS = 6                  # images per augmented forward pass (x6 variants = 36 tensors)
BOOT_N = 3000
BOOT_RNG = 20261018


# ------------------------------------------------------------------ infrastructure
def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def hms(s: float) -> str:
    s = int(max(0, s))
    return f"{s // 3600}h{s % 3600 // 60}m{s % 60}s"


def write_json_atomic(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1, default=float)
    os.replace(tmp, path)


def gpu_stats() -> dict:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,utilization.gpu,memory.used,memory.total",
             "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=15)
        name, u, m, mt = [x.strip() for x in out.stdout.strip().split(",")[:4]]
        return {"name": name, "util_pct": int(u), "mem_used_mb": int(m),
                "mem_total_mb": int(mt)}
    except Exception as exc:  # noqa: BLE001
        return {"error": f"{type(exc).__name__}: {exc}"}


def sha256_of(seq) -> str:
    h = hashlib.sha256()
    for s in seq:
        h.update(str(s).encode())
        h.update(b"\n")
    return h.hexdigest()


def free_gb() -> float:
    st = os.statvfs(str(ROOT))
    return st.f_bavail * st.f_frsize / 2 ** 30


# ------------------------------------------------------------------ frozen-config guards
def assert_frozen_config() -> dict:
    """Hard runtime assertion: E25 must run exactly the E17-frozen R1-ILA values."""
    fm = json.loads(FINAL_METHOD.read_text())["parameters"]
    checks = {
        "BETA": (E13A.BETA, fm["BETA"]),
        "LR_ADAPTER": (E13A.LR_ADAPTER, fm["LR_ADAPTER"]),
        "MAX_STEPS": (E13A.MAX_STEPS, fm["MAX_STEPS"]),
        "PATIENCE_STEPS": (E13A.PATIENCE_STEPS, fm["PATIENCE_STEPS"]),
        "EVAL_EVERY": (E13A.EVAL_EVERY, fm["EVAL_EVERY"]),
        "BATCH_IMGS": (E13A.BATCH_IMGS, fm["BATCH_IMGS"]),
    }
    bad = {k: v for k, v in checks.items() if v[0] != v[1]}
    if bad:
        raise RuntimeError(f"frozen hyperparameter mismatch vs results/e17/final_method.json: {bad}")
    if list(AUG_LIST) != [(f, lv) for f in ["brightness", "contrast", "gamma"]
                          for lv in [0.85, 1.15]]:
        raise RuntimeError(f"AUG_LIST changed: {AUG_LIST}")
    if (D, N_PATCH) != (768, 1024):
        raise RuntimeError(f"D/N_PATCH changed: {D}/{N_PATCH}")
    print(f"[E25] frozen config OK: BETA={E13A.BETA} lr={E13A.LR_ADAPTER} "
          f"steps={E13A.MAX_STEPS}/patience{E13A.PATIENCE_STEPS} batch={E13A.BATCH_IMGS} "
          f"aug={AUG_LIST} | documented deviation: TRAIN_N={BANK_TRAIN_N} "
          f"(E17 has {fm['TRAIN_N']}) because source train has 180 normals only", flush=True)
    return fm


# ------------------------------------------------------------------ image prep (square 448)
def prep(img: Image.Image, family=None, level=None) -> np.ndarray:
    """Frozen `prep_image` body with the geometric step made explicit-square.

    Protocol §8.3 states "448×448 bicubic resize". E13-A's `prep_image` calls
    `T.Resize(448)` (short-side resize): identical for square inputs (all PCB images
    are 2048x2048) but produces 32x57 = 1824 tokens for MetalParts (2681x1500),
    which is incompatible with the frozen 1024-token model/scorer.
    For square inputs this is bit-identical to `E13A.prep_image` (asserted at runtime
    by `geometry_audit`); only the geometric target size is made explicit.
    """
    t = T.Compose([T.Resize((448, 448), interpolation=T.InterpolationMode.BICUBIC,
                            antialias=True), T.ToTensor()])(img)
    if family is not None:
        t = E13A._aug(t, family, level)
    t = T.Normalize(mean=E13A.IMAGENET_MEAN, std=E13A.IMAGENET_STD)(t)
    t = t[:, :t.shape[1] - t.shape[1] % 14, :t.shape[2] - t.shape[2] % 14]
    return t.numpy().astype("float32")


def geometry_audit(domain: str, n_check: int = 20) -> dict:
    """Record original image sizes and prove `prep` == `E13A.prep_image` on square images."""
    from collections import Counter
    out = {"domain": domain, "splits": {}, "timestamp": now_iso()}
    sizes_all: Counter = Counter()
    for split, d in split_dirs(domain).items():
        files = sorted([p for p in (d / "normal").glob("*")]
                       + [p for p in (d / "anomaly").glob("*")])
        sizes = Counter(Image.open(f).size for f in files[:80])
        out["splits"][split] = {"n_files": len(files),
                                "sizes_sample": {f"{w}x{h}": c for (w, h), c in sizes.items()}}
        sizes_all.update(sizes)
    out["n_square_sampled"] = sum(c for (w, h), c in sizes_all.items() if w == h)
    out["n_nonsquare_sampled"] = sum(c for (w, h), c in sizes_all.items() if w != h)
    probe = []
    for split, d in split_dirs(domain).items():
        probe += sorted((d / "normal").glob("*"))[:(n_check // 5) + 1]
    diffs, shape_mismatch = [], 0
    for f in probe[:n_check]:
        im = Image.open(f).convert("RGB")
        a, b = prep(im), prep_image(im)
        if a.shape != b.shape:
            shape_mismatch += 1
            continue
        diffs.append(float(np.abs(a - b).max()))
    out["bit_identical_vs_e13a"] = {
        "n_checked": len(probe[:n_check]), "n_shape_mismatch": shape_mismatch,
        "n_compared": len(diffs), "max_abs_diff": max(diffs) if diffs else 0.0,
        "identical": bool(shape_mismatch == 0 and max(diffs or [0.0]) == 0.0),
        "note": "shape mismatch is expected iff the domain has non-square images; then the "
                "square 448x448 resize IS the documented geometry rule, not an accident"}
    write_json_atomic(OUTDIR / f"geometry_{domain}.json", out)
    print(f"[E25] geometry audit {domain}: square_sampled={out['n_square_sampled']} "
          f"nonsquare_sampled={out['n_nonsquare_sampled']} | prep vs E13A "
          f"max|d|={out['bit_identical_vs_e13a']['max_abs_diff']:.3e}", flush=True)
    return out


# ------------------------------------------------------------------ data loader
def split_dirs(domain: str) -> dict:
    root = DATA / domain
    prefix = DOMAINS[domain]["prefix"]
    out = {}
    for p in sorted(root.iterdir()):
        if not p.is_dir() or not p.name.startswith(prefix):
            continue
        out[p.name[len(prefix):]] = p
    return out


def load_split(path: Path) -> list[dict]:
    """Read metadata.jsonl only; labels are NEVER taken from directory names."""
    meta = path / "metadata.jsonl"
    if not meta.exists():
        raise RuntimeError(f"metadata.jsonl missing in {path}")
    recs, dir_mismatch = [], 0
    for line in meta.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        fp = path / r["file_name"]
        if not fp.exists():
            raise RuntimeError(f"file listed in metadata but missing on disk: {fp}")
        lab = int(r["label"])
        if lab not in (0, 1):
            raise RuntimeError(f"illegal label {lab} in {meta}")
        if lab != (1 if r["file_name"].startswith("anomaly/") else 0):
            dir_mismatch += 1
        recs.append({"path": fp, "file_name": r["file_name"], "label": lab,
                     "rel": f"{path.name}/{r['file_name']}"})
    if dir_mismatch:
        raise RuntimeError(f"{meta}: {dir_mismatch} label/folder mismatches")
    if len(recs) != len({r["rel"] for r in recs}):
        raise RuntimeError(f"duplicate file_name entries in {meta}")
    recs.sort(key=lambda r: r["file_name"])          # deterministic order everywhere
    return recs


def source_normals(domain: str) -> list[dict]:
    recs = [r for r in load_split(split_dirs(domain)["train"]) if r["label"] == 0]
    if len(recs) != 180:
        raise RuntimeError(f"{domain}: expected 180 source-train normals, got {len(recs)}")
    return recs


# ------------------------------------------------------------------ features
def bank_feats(model, bank: list[dict], dev) -> np.ndarray:
    out = np.empty((len(bank), N_PATCH, D), dtype="float32")
    t0 = time.time()
    for i0 in range(0, len(bank), EXTRACT_BS):
        chunk = bank[i0:i0 + EXTRACT_BS]
        tens = np.stack([prep(Image.open(r["path"]).convert("RGB")) for r in chunk])
        out[i0:i0 + len(chunk)] = forward_frozen_batch(model, tens, dev)
        print(f"    bank feat {min(i0 + EXTRACT_BS, len(bank))}/{len(bank)} "
              f"({time.time() - t0:.0f}s)", flush=True)
    return out


def aug_feats(model, bank: list[dict], dev) -> np.ndarray:
    K = len(AUG_LIST)
    out = np.empty((len(bank), K, N_PATCH, D), dtype="float32")
    t0 = time.time()
    for i0 in range(0, len(bank), AUG_BS):
        chunk = bank[i0:i0 + AUG_BS]
        tens = np.empty((len(chunk) * K, 3, 448, 448), dtype="float32")
        for j, r in enumerate(chunk):
            img = Image.open(r["path"]).convert("RGB")
            for k, (fam, lv) in enumerate(AUG_LIST):
                tens[j * K + k] = prep(img, fam, lv)
        f = forward_frozen_batch(model, tens, dev)
        out[i0:i0 + len(chunk)] = f.reshape(len(chunk), K, N_PATCH, D)
        print(f"    aug feat {min(i0 + AUG_BS, len(bank))}/{len(bank)} "
              f"({time.time() - t0:.0f}s)", flush=True)
    return out


def test_feats(model, recs: list[dict], dev) -> np.ndarray:
    out = np.empty((len(recs), N_PATCH, D), dtype="float32")
    for i0 in range(0, len(recs), EXTRACT_BS):
        chunk = recs[i0:i0 + EXTRACT_BS]
        tens = np.stack([prep(Image.open(r["path"]).convert("RGB")) for r in chunk])
        out[i0:i0 + len(chunk)] = forward_frozen_batch(model, tens, dev)
    return out


def norm_rows(feat: np.ndarray) -> np.ndarray:
    x = feat.reshape(-1, D).astype("float32")
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)


# ------------------------------------------------------------------ metrics
def metrics(scores: np.ndarray, labels: list[int]) -> dict:
    from sklearn.metrics import roc_auc_score, average_precision_score
    y = np.asarray(labels, dtype=int)
    s = np.asarray(scores, dtype=float)
    if len(set(y.tolist())) != 2:
        raise RuntimeError(f"split has a single class: {sorted(set(y.tolist()))}")
    return {"n_ok": int((y == 0).sum()), "n_ng": int((y == 1).sum()),
            "auroc": float(roc_auc_score(y, s)),
            "ap": float(average_precision_score(y, s))}


# ------------------------------------------------------------------ one domain
def run_domain(domain: str, dev, args) -> dict:
    t_dom = time.time()
    torch.cuda.reset_peak_memory_stats()
    bank = source_normals(domain)
    splits = split_dirs(domain)
    geom = geometry_audit(domain)
    targets = [s for s in sorted(splits) if s != "train"]
    print(f"\n[E25] === DOMAIN {domain} === bank={len(bank)} normals | "
          f"splits={targets} | disk {free_gb():.1f} GB", flush=True)

    model = load_dinov2_local(dev).to(dev)
    freeze_backbone(model)

    bank_f = bank_feats(model, bank, dev)
    bank_aug = aug_feats(model, bank, dev)

    # ---- B0 on the shared bank
    b0_bank = norm_rows(bank_f)
    bank_hash = sha256_of([r["rel"] for r in bank])
    print(f"[E25] bank hash {bank_hash[:16]} | B0 bank ready", flush=True)

    # ---- B1 adapter (frozen R1-ILA)
    torch.manual_seed(0)
    adapter = LinearAdapter().to(dev)
    n_params = sum(p.numel() for p in adapter.parameters())
    if n_params != 589824:
        raise RuntimeError(f"trainable params {n_params} != 589824")
    tr = bank_f[:BANK_TRAIN_N]
    tr_aug = bank_aug[:BANK_TRAIN_N]
    va = bank_f[BANK_TRAIN_N:BANK_TRAIN_N + BANK_VAL_N]
    va_aug = bank_aug[BANK_TRAIN_N:BANK_TRAIN_N + BANK_VAL_N]
    t_tr = time.time()
    trep = train_r1(adapter, tr, tr_aug, va, va_aug, dev, 0)
    print(f"[E25] adapter trained in {time.time() - t_tr:.0f}s | "
          f"steps={trep['steps_used']} best_val={trep['best_val']:.6f} "
          f"loss_finite={trep['loss_finite']} pairwise_cos={trep['after']['pairwise_cos']:.4f}",
          flush=True)
    if not trep["loss_finite"]:
        raise RuntimeError(f"{domain}: adapter loss became non-finite -> STOP")
    bank_af = adapted_feat_adapter(adapter, bank_f, dev)
    b1_bank = l2norm(torch.from_numpy(bank_af).to(dev)).cpu().numpy().astype("float32")
    adapter_hash = hashlib.sha256(
        adapter.W.detach().cpu().numpy().astype("float32").tobytes()).hexdigest()

    rows = []
    for split in targets:
        key = f"{domain}|{split}"
        cp = CELLDIR / f"{domain}_{split}.json"
        if cp.exists() and not args.force:
            old = json.loads(cp.read_text())
            if old.get("status") == "OK":
                print(f"[E25] SKIP {key} (already complete)", flush=True)
                rows.append(summary_row(old))
                write_domain_csv(domain, rows, complete=False)
                continue

        t0 = time.time()
        recs = load_split(splits[split])
        labels = [r["label"] for r in recs]
        tf = test_feats(model, recs, dev)

        s0 = nn_image_scores(b0_bank, norm_rows(tf), dev)
        m0 = metrics(s0, labels)

        taf = adapted_feat_adapter(adapter, tf, dev)
        s1 = nn_image_scores(b1_bank, l2norm(torch.from_numpy(taf).to(dev)).cpu().numpy()
                             .astype("float32"), dev)
        m1 = metrics(s1, labels)

        invalid_reason = DOMAINS[domain]["invalid"].get(split)
        cell = {"domain": domain, "split": split, "valid": 0 if invalid_reason else 1,
                "invalid_reason": invalid_reason,
                "condition_group": DOMAINS[domain]["group"][split],
                "condition_token_sample": recs[0]["file_name"].split("/")[-1],
                "n_total": len(recs), "n_ok": m0["n_ok"], "n_ng": m0["n_ng"],
                "f0": m0, "r1": m1,
                "d_auroc": m1["auroc"] - m0["auroc"], "d_ap": m1["ap"] - m0["ap"],
                "scores_f0": [float(x) for x in s0],
                "scores_r1": [float(x) for x in s1],
                "labels": labels, "files": [r["rel"] for r in recs],
                "bank_hash": bank_hash, "bank_size": len(bank),
                "adapter_hash": adapter_hash, "trainable_params": n_params,
                "steps_used": trep["steps_used"], "seed": 0,
                "config_source": "results/e17/final_method.json",
                "runtime_s": round(time.time() - t0, 1),
                "peak_gpu_mb": round(torch.cuda.max_memory_allocated() / 2 ** 20, 1),
                "timestamp": now_iso(), "status": "OK"}
        write_json_atomic(cp, cell)
        rows.append(summary_row(cell))
        # persist after EVERY split (protocol requirement)
        write_domain_csv(domain, rows, complete=False)
        print(f"[E25] {key:20s} valid={cell['valid']} n={cell['n_ok']}/{cell['n_ng']} "
              f"AUROC {m0['auroc']:.5f} -> {m1['auroc']:.5f} ({cell['d_auroc']:+.5f}) | "
              f"AP {m0['ap']:.5f} -> {m1['ap']:.5f} ({cell['d_ap']:+.5f}) | "
              f"{cell['runtime_s']:.0f}s", flush=True)

    write_domain_csv(domain, rows, complete=True)
    del model, adapter, bank_f, bank_aug, bank_af, b0_bank, b1_bank
    torch.cuda.empty_cache()
    print(f"[E25] DOMAIN {domain} done in {hms(time.time() - t_dom)} -> "
          f"{OUTDIR / f'e25_results_{domain}.csv'}", flush=True)
    return {"domain": domain, "n_splits": len(targets), "bank_hash": bank_hash,
            "steps_used": trep["steps_used"], "adapter_hash": adapter_hash,
            "runtime_s": round(time.time() - t_dom, 1),
            "geometry": {"square_sampled": geom["n_square_sampled"],
                         "nonsquare_sampled": geom["n_nonsquare_sampled"],
                         "prep_vs_e13a_identical": geom["bit_identical_vs_e13a"]["identical"]},
            "peak_gpu_mb": round(torch.cuda.max_memory_allocated() / 2 ** 20, 1)}


CSV_COLS = ["domain", "split", "condition_group", "valid", "n_normal", "n_anomaly",
            "auroc_f0", "ap_f0", "auroc_r1", "ap_r1", "d_auroc", "d_ap",
            "steps_used", "runtime_s", "peak_gpu_mb", "status"]


def summary_row(cell: dict) -> dict:
    return {"domain": cell["domain"], "split": cell["split"],
            "condition_group": cell["condition_group"], "valid": cell["valid"],
            "n_normal": cell["n_ok"], "n_anomaly": cell["n_ng"],
            "auroc_f0": cell["f0"]["auroc"], "ap_f0": cell["f0"]["ap"],
            "auroc_r1": cell["r1"]["auroc"], "ap_r1": cell["r1"]["ap"],
            "d_auroc": cell["d_auroc"], "d_ap": cell["d_ap"],
            "steps_used": cell["steps_used"], "runtime_s": cell["runtime_s"],
            "peak_gpu_mb": cell["peak_gpu_mb"], "status": cell["status"]}


def write_domain_csv(domain: str, rows: list[dict], complete: bool) -> None:
    path = OUTDIR / f"e25_results_{domain}.csv"
    tmp = path.with_suffix(".csv.tmp")
    with open(tmp, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLS)
        w.writeheader()
        for r in rows:
            w.writerow({**r,
                        "auroc_f0": f"{r['auroc_f0']:.6f}", "ap_f0": f"{r['ap_f0']:.6f}",
                        "auroc_r1": f"{r['auroc_r1']:.6f}", "ap_r1": f"{r['ap_r1']:.6f}",
                        "d_auroc": f"{r['d_auroc']:+.6f}", "d_ap": f"{r['d_ap']:+.6f}"})
    os.replace(tmp, path)
    if complete:
        print(f"[E25] wrote {path} ({len(rows)} splits, COMPLETE)", flush=True)


# ------------------------------------------------------------------ smoke
def smoke(dev, domain: str = "PCB") -> int:
    """Minimal end-to-end gate: loader/labels/bank/features/B0/R1/scoring/metrics."""
    checks: dict[str, bool] = {}
    print(f"\n[E25 SMOKE] domain={domain} 1) loader + label provenance", flush=True)
    bank = source_normals(domain)[:24]
    checks["bank_24_normals"] = len(bank) == 24 and all(r["label"] == 0 for r in bank)
    recs_all = load_split(split_dirs(domain)["test1"])
    recs = ([r for r in recs_all if r["label"] == 0][:6]
            + [r for r in recs_all if r["label"] == 1][:6])
    checks["labels_from_metadata"] = all(r["label"] in (0, 1) for r in recs)
    checks["labels_both_classes"] = len({r["label"] for r in recs}) == 2

    print("[E25 SMOKE] 2) GPU + DINOv2-B", flush=True)
    checks["cuda"] = torch.cuda.is_available()
    model = load_dinov2_local(dev).to(dev)
    freeze_backbone(model)
    checks["backbone_frozen"] = not any(p.requires_grad for p in model.parameters())

    print("[E25 SMOKE] 3) feature extraction", flush=True)
    bf = bank_feats(model, bank, dev)
    tf = test_feats(model, recs, dev)
    checks["feature_shape"] = bf.shape == (len(bank), N_PATCH, D) and tf.shape[0] == len(recs)
    checks["feature_finite"] = bool(np.isfinite(bf).all() and np.isfinite(tf).all())

    print("[E25 SMOKE] 4) B0 baseline scoring", flush=True)
    s0 = nn_image_scores(norm_rows(bf), norm_rows(tf), dev)
    checks["b0_scores_finite"] = bool(np.isfinite(s0).all()) and len(s0) == len(recs)
    ref = load_split(split_dirs(domain)["test1"])
    s0_full = nn_image_scores(norm_rows(bf), norm_rows(test_feats(model, ref, dev)), dev)
    m0 = metrics(s0_full, [r["label"] for r in ref])
    checks["b0_metrics_valid"] = 0.0 <= m0["auroc"] <= 1.0 and 0.0 <= m0["ap"] <= 1.0

    print("[E25 SMOKE] 5) R1-ILA training + scoring", flush=True)
    torch.manual_seed(0)
    adapter = LinearAdapter().to(dev)
    checks["adapter_params"] = sum(p.numel() for p in adapter.parameters()) == 589824
    af = aug_feats(model, bank, dev)
    checks["aug_shape"] = af.shape == (len(bank), len(AUG_LIST), N_PATCH, D)
    trep = train_r1(adapter, bf[:16], af[:16], bf[16:24], af[16:24], dev, 0)
    checks["r1_loss_finite"] = bool(trep["loss_finite"])
    checks["r1_grad_seen"] = bool(trep["grad_present"])
    checks["r1_steps"] = trep["steps_used"] > 0
    b_af = adapted_feat_adapter(adapter, bf, dev)
    t_af = adapted_feat_adapter(adapter, tf, dev)
    s1 = nn_image_scores(l2norm(torch.from_numpy(b_af).to(dev)).cpu().numpy().astype("float32"),
                         l2norm(torch.from_numpy(t_af).to(dev)).cpu().numpy().astype("float32"),
                         dev)
    checks["r1_scores_finite"] = bool(np.isfinite(s1).all()) and len(s1) == len(recs)
    checks["r1_no_collapse"] = trep["after"]["pairwise_cos"] <= 0.98

    print("[E25 SMOKE] 6) output paths", flush=True)
    OUTDIR.mkdir(parents=True, exist_ok=True)
    probe = OUTDIR / ".write_probe"
    probe.write_text("ok")
    checks["output_writable"] = probe.read_text() == "ok"
    probe.unlink()
    checks["disk_ge_1gb"] = free_gb() > 1.0

    print("\n[E25 SMOKE] CHECK SUMMARY", flush=True)
    for k, v in checks.items():
        print(f"  {k}: {'PASS' if v else 'FAIL'}", flush=True)
    ok = all(checks.values())
    print(f"[E25 SMOKE] {'PASS' if ok else 'FAIL'} (no cell/CSV written)", flush=True)
    write_json_atomic(OUTDIR / "smoke_check.json",
                      {"timestamp": now_iso(), "checks": checks, "pass": bool(ok),
                       "b0_auroc_test1_subset_full": m0["auroc"],
                       "steps_used": trep["steps_used"]})
    return 0 if ok else 1


# ------------------------------------------------------------------ manifest / main
def write_manifest(dev) -> None:
    import sklearn
    git = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"],
                         capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "-C", str(ROOT), "status", "--short"],
                           capture_output=True, text=True).stdout.strip().splitlines()
    write_json_atomic(OUTDIR / "run_manifest.json", {
        "experiment": "E25 RobustAD external generalization of frozen R1-ILA",
        "protocol": "results/e25/ROBUSTAD_PROTOCOL_AUDIT.md (§8 + §10 amendment)",
        "config_source": "results/e17/final_method.json",
        "method": "R1-ILA", "BETA": E13A.BETA, "augmentations": "brightness+contrast+gamma",
        "strength": [0.85, 1.15], "lr": E13A.LR_ADAPTER,
        "max_steps": E13A.MAX_STEPS, "patience": E13A.PATIENCE_STEPS,
        "batch_size": E13A.BATCH_IMGS, "EVAL_EVERY": E13A.EVAL_EVERY,
        "seed": 0, "bank_train_n": BANK_TRAIN_N, "bank_val_n": BANK_VAL_N,
        "bank_rule": "all source-train normals (180), sorted by file_name; "
                     "train = first 120, val = last 60",
        "input_geometry": "square bicubic resize to 448x448 (protocol §8.3); identical to "
                          "E13-A for square inputs, asserted per domain in geometry_*.json",
        "metric": "plain image-level AUROC/AP over all images (no detectable filter; "
                  "RobustAD has no such field)",
        "git_commit": git, "git_dirty_files": dirty, "python": sys.version.split()[0],
        "torch": torch.__version__, "sklearn": sklearn.__version__,
        "gpu": gpu_stats(), "disk_free_gb": round(free_gb(), 2),
        "started": now_iso(), "pid": os.getpid(), "note": "results/e24/ not used and not created"})


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--smoke-domain", default="PCB")
    ap.add_argument("--domain", default="")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    dev = torch.device("cuda")
    OUTDIR.mkdir(parents=True, exist_ok=True)
    CELLDIR.mkdir(parents=True, exist_ok=True)

    if args.smoke:
        return smoke(dev, args.smoke_domain)

    assert_frozen_config()
    write_manifest(dev)
    print(f"[E25] gpu={gpu_stats()} disk={free_gb():.1f} GB", flush=True)

    todo = list(DOMAINS) if args.all else [d.strip() for d in args.domain.split(",") if d.strip()]
    if not todo:
        ap.print_help()
        return 1
    partial = {"experiment": "E25", "domains": {}, "started": now_iso()}
    if PARTIAL.exists():
        try:
            partial.update(json.loads(PARTIAL.read_text()))
        except json.JSONDecodeError:
            pass
    for dom in todo:
        if dom not in DOMAINS:
            print(f"[E25] unknown domain {dom}", flush=True)
            return 1
        info = run_domain(dom, dev, args)
        partial["domains"][dom] = info
        write_json_atomic(PARTIAL, partial)
        print(f"[E25] partial updated -> {PARTIAL}", flush=True)
    print(f"\n[E25] finished {todo} at {now_iso()}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
