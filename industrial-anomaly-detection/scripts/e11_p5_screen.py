#!/usr/bin/env python
"""E11-P5 - Final Lightweight Screening (frozen DINOv2-B, 1-NN-max baseline).

Candidates (frozen config `results/e11/p5_module_config.json`):
  D1a/D1b = K-means prototype bank (1024 / 4096 centroids) -> 1-NN-max against centroids.
  D2a/D2b = k-NN local consistency (k=3 / k=5) -> mean of k nearest cosine distances, max over patches.
  D3a/D3b = global(CLS 1-NN) + local(patch 1-NN-max) fusion, alpha_global = 0.3 / 0.5.
Reuses a single 448 forward (get_intermediate_layers return_class_token=True -> patches + CLS).
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torchvision.transforms as T  # noqa: E402
from PIL import Image  # noqa: E402

import e4d1_m2ad_loader as LD  # noqa: E402
import e8_common as E8  # noqa: E402
from e11_p2_runner import load_dinov2, raw_test_map, official_metrics, route, BACKBONES  # noqa: E402

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
CHUNK = 8
BANK_CHUNK = 65536


def _prep(img: Image.Image, model, dev):
    tf = T.Compose([T.Resize(448, interpolation=T.InterpolationMode.BICUBIC, antialias=True),
                    T.ToTensor(), T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)])
    t = tf(img)
    ps = model.patch_size
    return t[:, :t.shape[1] - t.shape[1] % ps, :t.shape[2] - t.shape[2] % ps].to(dev)


def extract_p5(model, path: str, dev):
    """-> (patches (1024,768) float32 numpy, cls (768,) float32 numpy), both raw (unnormalised)."""
    img = Image.open(path).convert("RGB")
    with torch.inference_mode():
        t = _prep(img, model, dev)
        r = model.get_intermediate_layers(t.unsqueeze(0), n=1, norm=True, return_class_token=True)[0]
        patches = r[0][0].float().cpu().numpy()
        cls = r[1][0].float().cpu().numpy()
    return patches, cls


def l2norm(x: torch.Tensor) -> torch.Tensor:
    return x / x.norm(dim=1, keepdim=True).clamp_min(1e-12)


def knn_scores(bank_t: torch.Tensor, test_t: torch.Tensor, k: int, dev) -> np.ndarray:
    """Image score = max over patches of (mean of k nearest cosine distances). bank_t/test_t L2-normalised."""
    n_bank, n_test = bank_t.shape[0], test_t.shape[0]
    n_img = n_test // 1024
    scores = np.empty(n_img, dtype="float32")
    with torch.inference_mode():
        for i in range(0, n_test, 1024 * CHUNK):
            tt = test_t[i:i + 1024 * CHUNK]
            top = None
            for j in range(0, n_bank, BANK_CHUNK):
                dist = 1.0 - tt @ bank_t[j:j + BANK_CHUNK].T
                top = dist if top is None else torch.cat([top, dist], dim=1)
                top = torch.topk(top, k=k, dim=1, largest=False).values
            patch = top.mean(dim=1)
            img = patch.view(-1, 1024).max(dim=1).values
            j0 = i // 1024
            scores[j0:j0 + img.shape[0]] = img.cpu().numpy()
    return scores


def global_scores(bank_cls: torch.Tensor, test_cls: torch.Tensor, dev) -> np.ndarray:
    """Per-image CLS 1-NN cosine distance. bank_cls (N,768), test_cls (M,768) L2-normalised."""
    sim = test_cls @ bank_cls.T
    return (1.0 - sim.max(dim=1).values).cpu().numpy()


def kmeans_centroids(bank_patches: np.ndarray, K: int) -> np.ndarray:
    """Cosine K-means on L2-normalised patches (Euclidean on unit sphere ~ cosine). Returns (K, 768) centroids."""
    import faiss
    km = faiss.Kmeans(bank_patches.shape[1], K, niter=12, verbose=False, seed=0, gpu=False)
    km.train(np.ascontiguousarray(bank_patches, dtype="float32"))
    c = np.ascontiguousarray(km.centroids, dtype="float32")
    return c / np.maximum(np.linalg.norm(c, axis=1, keepdims=True), 1e-12)


# ------------------------------------------------------------------ run
def run_category(category: str, dev) -> dict:
    recs = LD.load_records(category=category, require_exists=False)
    est = LD.estimate_scope(recs)
    bank_specs = set(est["bank_specimens"])
    bank = E8.bank_for(sorted(LD.select(recs, bank_specs), key=lambda r: r["image_path"]), 0)
    test = [r for r in LD.select(recs, views=["120"]) if r["split"] == "test"]
    bank_r = route(bank, "l0_448")
    test_r = route(test, "l0_448")
    model = load_dinov2(BACKBONES["dinov2b"]["hub"], dev)

    t0 = time.time()
    bank_p, bank_c, test_p, test_c = [], [], [], []
    for r in bank_r:
        p, c = extract_p5(model, r["image_path"], dev)
        bank_p.append(p); bank_c.append(c)
    for r in test_r:
        p, c = extract_p5(model, r["image_path"], dev)
        test_p.append(p); test_c.append(c)
    bank_p = np.concatenate(bank_p, axis=0); bank_c = np.stack(bank_c)
    test_p = np.concatenate(test_p, axis=0); test_c = np.stack(test_c)
    extract_s = time.time() - t0

    bank_pn = bank_p / np.maximum(np.linalg.norm(bank_p, axis=1, keepdims=True), 1e-12)
    bank_pt = torch.from_numpy(bank_pn).to(dev)
    test_pt = l2norm(torch.from_numpy(test_p).to(dev))
    bank_ct = l2norm(torch.from_numpy(bank_c).to(dev))
    test_ct = l2norm(torch.from_numpy(test_c).to(dev))

    t0 = time.time()
    local = knn_scores(bank_pt, test_pt, 1, dev)
    base_s = time.time() - t0
    t0 = time.time()
    glob = global_scores(bank_ct, test_ct, dev)
    glob_s = time.time() - t0

    results = {"baseline": {"score": local, "score_s": base_s}}
    for key, K in (("D1a", 1024), ("D1b", 4096)):
        t0 = time.time()
        cents = kmeans_centroids(bank_pn, K)
        cents_t = torch.from_numpy(cents).to(dev)
        results[key] = {"score": knn_scores(cents_t, test_pt, 1, dev), "score_s": time.time() - t0}
        del cents_t
        torch.cuda.empty_cache()
    for key, k in (("D2a", 3), ("D2b", 5)):
        t0 = time.time()
        results[key] = {"score": knn_scores(bank_pt, test_pt, k, dev), "score_s": time.time() - t0}
    for key, alpha in (("D3a", 0.3), ("D3b", 0.5)):
        results[key] = {"score": (1 - alpha) * local + alpha * glob, "score_s": 0.0}

    rel = lambda r: str(Path(r["image_path"]).relative_to(LD.DATA_ROOT))
    rawmap = raw_test_map(category)
    out = {"category": category, "extract_s": round(extract_s, 1),
           "peak_vram_mb": round(torch.cuda.max_memory_allocated() / 2**20, 1), "candidates": {}}
    for key, res in results.items():
        rows = []
        for r, s in zip(test, res["score"]):
            m = rawmap[rel(r)]
            rows.append({"specimen": r["specimen"], "kind": r["kind"],
                         "illumination": r["illumination"], "view": "120", "score": float(s),
                         "image_anomaly": int(m["image_anomaly"]), "detectable": m["detectable"]})
        met = official_metrics(rows)
        out["candidates"][key] = {"metrics": met, "score_s": round(res["score_s"], 1)}
        print(f"    {key:9s} auroc={met['auroc']:.5f} ap={met['ap']:.5f}", flush=True)
    return out


def main() -> int:
    import argparse, csv
    ap = argparse.ArgumentParser()
    ap.add_argument("--categories", default="Bird,Car,Cube")
    args = ap.parse_args()
    dev = torch.device("cuda")
    cats = [c.strip() for c in args.categories.split(",") if c.strip()]
    outdir = ROOT / "results" / "e11"
    outdir.mkdir(parents=True, exist_ok=True)

    w = csv.writer(open(outdir / "p5_screening.csv", "w", newline=""))
    w.writerow(["category", "view", "module", "auroc", "ap", "d_auroc", "d_ap",
                "extract_s", "score_s", "peak_vram_mb", "status"])
    raw = {}
    for cat in cats:
        print(f"\n[E11-P5] RUN {cat}", flush=True)
        try:
            out = run_category(cat, dev)
        except Exception as exc:
            import traceback; traceback.print_exc()
            out = {"status": "FAILED", "reason": f"{type(exc).__name__}: {exc}"}
        if out.get("status") == "FAILED":
            print(f"    FAILED: {out['reason'][:100]}", flush=True)
            continue
        raw[cat] = out
        base = out["candidates"]["baseline"]["metrics"]
        for key in ["baseline", "D1a", "D1b", "D2a", "D2b", "D3a", "D3b"]:
            c = out["candidates"][key]
            w.writerow([cat, "120", key, f"{c['metrics']['auroc']:.5f}", f"{c['metrics']['ap']:.5f}",
                        f"{c['metrics']['auroc']-base['auroc']:+.5f}", f"{c['metrics']['ap']-base['ap']:+.5f}",
                        out["extract_s"], c["score_s"], out["peak_vram_mb"], "OK"])
    json.dump(raw, open(outdir / "p5_module_raw.json", "w"), indent=2, default=str)
    print(f"\n[E11-P5] done -> {outdir/'p5_screening.csv'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
