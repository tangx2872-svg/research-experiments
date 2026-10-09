#!/usr/bin/env python
"""E7 — shared scope, photometric caches, metrics, progress/checkpoint.

Protocol: `docs/E7_1_FROZEN_PROTOCOL.md` (frozen at commit af56dd3 BEFORE any candidate metric).

FROZEN UPSTREAM DESIGN — this module implements it, it does not design it.
Only the free parameters listed as assumptions A1-A6 in the protocol are chosen here.
"""
from __future__ import annotations

import csv
import hashlib
import json
import random
import resource
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

import e4d1_m2ad_loader as LD  # noqa: E402
import e4x_common as C  # noqa: E402

SEED = 0
FROZEN_VIEW = "120"
BANK_SUBSET_SEED = 20261009
BANK_SUBSET_FRACTION = 0.25          # A1
VIEW_SEED = 20261009                 # E7-2
CACHE_RES = 448                      # A2
CACHE_ROOT = ROOT / "data" / "m2ad" / "e7_cache"
CLIP_MILD, CLIP_MEDIUM = 1.5, 2.0    # frozen upstream
TILE = (8, 8)                        # frozen upstream

PIPELINES = ["P00", "P01", "P02", "P03", "P04", "P05", "P06", "P07", "P08", "P09",
             "P10", "P11", "P12", "P13"]
PIPELINE_DESC = {
    "P00": "Original PatchCore", "P01": "Retinex->PatchCore", "P02": "CLAHE mild->PatchCore",
    "P03": "CLAHE medium->PatchCore", "P04": "Retinex->CLAHE mild->PatchCore",
    "P05": "DINOv2-S->NN", "P06": "DINOv2-B->NN", "P07": "Retinex->DINOv2-S->NN",
    "P08": "Retinex->DINOv2-B->NN", "P09": "CLAHE mild->DINOv2-S->NN",
    "P10": "SoftPatch nearest", "P11": "SoftPatch LOF",
    "P12": "Retinex->SoftPatch nearest", "P13": "Retinex->SoftPatch LOF",
}
# A+X parents used for synergy (protocol section 9)
SYNERGY_PARENT = {"P04": ("P01", "P02"), "P07": ("P01", "P05"), "P08": ("P01", "P06"),
                  "P09": ("P02", "P05"), "P12": ("P01", "P10"), "P13": ("P01", "P11")}


# ------------------------------------------------------------------ progress / misc
def ram_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def vram() -> tuple:
    return (torch.cuda.memory_allocated() / 2**20, torch.cuda.memory_reserved() / 2**20)


class Bar:
    def __init__(self, tag: str, stage: str, total: int, unit: str = "img"):
        self.tag, self.stage, self.total, self.unit = tag, stage, total, unit
        self.t0 = time.time()
        self.done = 0

    def __call__(self, done=None, extra=""):
        return self.step(done, extra)

    def step(self, done=None, extra=""):
        self.done = done if done is not None else self.done + 1
        el = time.time() - self.t0
        rate = self.done / el if el > 0 else 0.0
        remain = (self.total - self.done) / rate if rate > 0 else float("inf")
        a, r = vram()
        print(f"  E7 [{self.tag}] {self.stage} {self.done}/{self.total} "
              f"{100.0*self.done/max(1,self.total):5.1f}%  elapsed={time.strftime('%H:%M:%S', time.gmtime(el))}  "
              f"ETA={time.strftime('%H:%M:%S', time.gmtime(remain)) if np.isfinite(remain) else 'n/a'}"
              f"  finish~{time.strftime('%H:%M', time.localtime(time.time()+remain)) if np.isfinite(remain) else 'n/a'}"
              f"  {rate:6.1f} {self.unit}/s  VRAM alloc={a:7.1f}MB res={r:7.1f}MB  RAM={ram_mb():7.1f}MB {extra}",
              flush=True)


def progress_json(path: Path) -> dict:
    if path.exists():
        try:
            return json.loads(path.read_text())
        except json.JSONDecodeError:
            pass
    return {"experiment": path.parent.name, "pipelines": {}}


def save_progress(path: Path, st: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(st, indent=2, default=float))


def write_csv(path: Path, rows: list, fields=None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    fields = fields or list(rows[0].keys())
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader(); w.writerows(rows)


def sha256(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


# ------------------------------------------------------------------ scope (frozen)
def load_scope() -> dict:
    recs = LD.load_records()
    est = LD.estimate_scope(recs)
    # E5-1's frozen bank source: the 10 specimens / 1200 images that produced the frozen scores
    bank_source = sorted(LD.select(recs, set(est["bank_specimens"])), key=lambda r: r["image_path"])
    n_sub = max(1, int(BANK_SUBSET_FRACTION * len(bank_source)))       # A1: 25% of 1200 = 300
    pick = sorted(random.Random(BANK_SUBSET_SEED).sample(range(len(bank_source)), n_sub))
    bank_subset = [bank_source[i] for i in pick]
    test120 = sorted([r for r in LD.select(recs, views=[FROZEN_VIEW]) if r["split"] == "test"],
                     key=lambda r: r["image_path"])
    test_all = sorted([r for r in LD.select(recs, views=None) if r["split"] == "test"],
                      key=lambda r: r["image_path"])
    return {"recs": recs, "est": est, "bank_source": bank_source, "bank_subset": bank_subset,
            "test120": test120, "test_all": test_all,
            "bank_specimens": sorted({r["specimen"] for r in bank_subset}),
            "bank_views": sorted({r["view"] for r in bank_subset}),
            "bank_illuminations": sorted({r["illumination"] for r in bank_subset})}


def sanity_scope(scope: dict) -> dict:
    res = {}
    res["S1_test_unreduced"] = {"pass": len(scope["test120"]) == 700,
                                "detail": f"test@view120 = {len(scope['test120'])} (expect 700)"}
    res["S1b_bank_subset"] = {"pass": len(scope["bank_subset"]) == 300,
                              "detail": f"bank subset = {len(scope['bank_subset'])} of "
                                        f"{len(scope['bank_source'])} (25% by BANK_SUBSET_SEED="
                                        f"{BANK_SUBSET_SEED})"}
    res["S2_normal_only"] = {"pass": all(r["kind"] == "Good" and r["split"] == "train"
                                         for r in scope["bank_subset"]),
                             "detail": "bank = train-split Good images only"}
    bank_specs = {r["specimen"] for r in scope["bank_subset"]}
    test_specs = {r["specimen"] for r in scope["test120"]}
    res["S3_no_leakage"] = {"pass": not (bank_specs & test_specs),
                            "detail": f"|bank specimens n test specimens| = {len(bank_specs & test_specs)}"}
    res["S4_illuminations"] = {"pass": sorted({r["illumination"] for r in scope["test120"]}) == C.ILLUMINATIONS,
                               "detail": f"test illuminations = {sorted({r['illumination'] for r in scope['test120']})}"}
    return res


# ------------------------------------------------------------------ photometric caches
def _rel(rec: dict) -> Path:
    return Path(rec["image_path"]).relative_to(LD.DATA_ROOT)


def cache_dir(kind: str) -> Path:
    return CACHE_ROOT / kind


def _load_rgb448(path: Path) -> np.ndarray:
    from PIL import Image
    import torchvision.transforms.functional as TF
    img = Image.open(path).convert("RGB")
    t = TF.resize(img, [CACHE_RES, CACHE_RES], antialias=True,
                  interpolation=TF.InterpolationMode.BICUBIC)
    return np.asarray(t)


def _apply_clahe(rgb: np.ndarray, clip: float) -> np.ndarray:
    import cv2
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    cl = cv2.createCLAHE(clipLimit=clip, tileGridSize=TILE)
    l2 = cl.apply(l)
    return cv2.cvtColor(cv2.merge([l2, a, b]), cv2.COLOR_LAB2RGB)


def _retinex_model():
    """PIAD's official vendored `Separate` (unchanged since E5-1)."""
    import argparse
    torch.serialization.add_safe_globals([argparse.Namespace])
    rroot = ROOT / "third_party" / "piad_baseline"
    if str(rroot) not in sys.path:
        sys.path.insert(0, str(rroot))
    from Retinex.RL_separate import Separate
    from Retinex.utils import np_save_TensorImg

    class _O:
        def __init__(self):
            self.Decom_model_low_path = str(rroot / "Retinex" / "ckpt" / "init_low.pth")
            self.unfolding_model_path = str(rroot / "Retinex" / "ckpt" / "unfolding.pth")
            self.adjust_model_path = str(rroot / "Retinex" / "ckpt" / "L_adjust.pth")
            self.ratio = 5
            self.gpu_id = 0
    return Separate(_O()).cuda().eval(), np_save_TensorImg


def ensure_cache(kind: str, records: list, bar=None) -> int:
    """Compute only the missing PNGs for a photometric transform. Returns the number written."""
    from PIL import Image
    out = cache_dir(kind)
    todo = [r for r in records if not (out / _rel(r)).exists()]
    if not todo:
        return 0
    model = save_fn = None
    if kind.startswith("retinex"):
        model, save_fn = _retinex_model()
    written = 0
    for i, r in enumerate(todo, 1):
        src = Path(r["image_path"])
        dst = out / _rel(r)
        dst.parent.mkdir(parents=True, exist_ok=True)
        if kind == "retinex":
            arr = _load_rgb448(src)
            t = torch.from_numpy(arr).float().permute(2, 0, 1).unsqueeze(0) / 255.0
            R, _L, _ = model(t)
            save_fn(R, str(dst))
        elif kind == "clahe_mild":
            Image.fromarray(_apply_clahe(_load_rgb448(src), CLIP_MILD)).save(dst)
        elif kind == "clahe_medium":
            Image.fromarray(_apply_clahe(_load_rgb448(src), CLIP_MEDIUM)).save(dst)
        elif kind == "retinex_clahe_mild":
            base = cache_dir("retinex") / _rel(r)
            assert base.exists(), "retinex cache must be built first"
            Image.fromarray(_apply_clahe(np.asarray(Image.open(base).convert("RGB")), CLIP_MILD)).save(dst)
        else:
            raise ValueError(f"unknown cache kind {kind}")
        written += 1
        if bar is not None and (i % 25 == 0 or i == len(todo)):
            bar.step(i, extra=f"kind={kind} written={written}")
    return written


def cached_image_path(kind: str | None, rec: dict) -> str:
    """Return the on-disk image path a pipeline should consume (None = original)."""
    if kind is None:
        return rec["image_path"]
    if kind == "retinex_clahe_mild":
        return str(cache_dir("retinex_clahe_mild") / _rel(rec))
    return str(cache_dir(kind) / _rel(rec))


# ------------------------------------------------------------------ metrics (protocol section 7)
def image_aupr(good: np.ndarray, defect: np.ndarray) -> float:
    from sklearn.metrics import average_precision_score
    y = np.concatenate([np.zeros(len(good)), np.ones(len(defect))])
    s = np.concatenate([good, defect])
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(average_precision_score(y, s))


def evaluate(rows: list, maps=None, masks=None, want_pixel: bool = False) -> dict:
    from experiment1h_runner import image_auroc
    g = np.array([r["score"] for r in rows if r["kind"] == "Good"], dtype=float)
    d = np.array([r["score"] for r in rows if r["kind"] == "NG"], dtype=float)
    out = {"n_good": int(len(g)), "n_defect": int(len(d)),
           "image_auroc": float(image_auroc(g, d)), "image_aupr": image_aupr(g, d),
           "d_prime": float(C.dprime(g, d)),
           "good_mean": float(g.mean()), "good_std": float(g.std(ddof=1)),
           "ng_mean": float(d.mean()), "ng_std": float(d.std(ddof=1))}
    out["pixel_auroc"] = float("nan"); out["aupro"] = float("nan")
    if want_pixel and maps is not None:
        good_maps = [m for m, r in zip(maps, rows) if r["kind"] == "Good"]
        def_maps = [m for m, r in zip(maps, rows) if r["kind"] == "NG"]
        def_masks = [mk for mk, r in zip(masks, rows) if r["kind"] == "NG"]
        try:
            from experiment1h_runner import pixel_auroc_from_maps
            out["pixel_auroc"] = float(pixel_auroc_from_maps(good_maps, def_maps, def_masks))
        except Exception as exc:  # noqa: BLE001
            out["pixel_auroc_error"] = f"{type(exc).__name__}: {str(exc)[:80]}"
        try:
            from experiment5a_h_runner import compute_aupro
            out["aupro"] = float(compute_aupro(good_maps, def_maps, def_masks))
        except Exception as exc:  # noqa: BLE001
            out["aupro_error"] = f"{type(exc).__name__}: {str(exc)[:80]}"
    return out


def tier(d_auroc: float, aupr_delta: float, dprime_delta: float) -> dict:
    """Frozen upstream tier rule (protocol section 8)."""
    if d_auroc >= 0.020:
        return {"tier": "A", "decision": "ADVANCE"}
    if d_auroc >= 0.010:
        return {"tier": "B", "decision": "KEEP"}
    if d_auroc > 0.0:
        if (aupr_delta > 0) or (dprime_delta > 0):
            return {"tier": "C", "decision": "BACKUP"}
        return {"tier": "C", "decision": "STOP"}
    return {"tier": "D", "decision": "STOP"}


def synergy(metrics_by_id: dict, comp: str) -> dict:
    """Protocol section 9."""
    a, b = SYNERGY_PARENT[comp]
    da = metrics_by_id[a]["d_auroc"]; db = metrics_by_id[b]["d_auroc"]
    dab = metrics_by_id[comp]["d_auroc"]
    return {"A": a, "B": b, "A_plus_B": comp,
            "d_auroc_A": da, "d_auroc_B": db, "d_auroc_AB": dab,
            "synergy_gain": dab - max(da, db),
            "positive_composition": bool(dab > da and dab > db)}


def gate_e7_2(metrics_by_id: dict, max_n: int = 4) -> dict:
    """Protocol section 11: ADVANCE+KEEP, top-max_n by AUROC, then BACKUP fill."""
    order = sorted(metrics_by_id.items(), key=lambda kv: -kv[1]["image_auroc"])
    adv = [k for k, v in order if v["decision"] == "ADVANCE"]
    keep = [k for k, v in order if v["decision"] == "KEEP"]
    back = [k for k, v in order if v["decision"] == "BACKUP"]
    chosen = (adv + keep)[:max_n]
    filled_from = None
    if len(chosen) < max_n and back:
        need = max_n - len(chosen)
        chosen = chosen + back[:need]
        filled_from = "BACKUP"
    return {"advance": adv, "keep": keep, "backup": back,
            "stop": [k for k, v in order if v["decision"] == "STOP"],
            "selected": chosen, "filled_from_backup": filled_from,
            "proceed_to_e7_2": len(chosen) > 0}
