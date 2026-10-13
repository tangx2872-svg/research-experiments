#!/usr/bin/env python
"""E11-P2 - Frozen Representation Diagnostic (no training).

Compares frozen backbones on the SAME test set, SAME official image_anomaly + detectable labels,
SAME unified 1-NN scoring (L2-normalised features -> nearest bank patch -> max over patches).

Backbones: ResNet (wide_resnet50_2 L2+L3 = PatchCore embedding), DINOv2-S, DINOv2-B.
Train budget: 300 normal images (E7-1 seed-0 bank). Test: Bird/Car/Cube view120 (700 img).
Inputs: ResNet <- l0_256; DINOv2 <- l0_448 (both lossless RGB caches, verified in P2.2).
"""
from __future__ import annotations

import argparse
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
import e7_common as E7  # noqa: E402
import e8_common as E8  # noqa: E402
import e8_fetch as F  # noqa: E402
import e10_runner as R10  # noqa: E402  (fit_patchcore, extract_embedding)

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
BANK_BUDGET = 300
SEED = 0
CHUNK = 8          # test images per GPU batch
BANK_CHUNK = 65536

BACKBONES = {
    "resnet":  {"input_kind": "l0_256", "dim": 1536, "desc": "wide_resnet50_2 L2+L3"},
    "dinov2s": {"input_kind": "l0_448", "dim": 384,  "hub": "dinov2_vits14", "desc": "DINOv2 ViT-S/14"},
    "dinov2b": {"input_kind": "l0_448", "dim": 768,  "hub": "dinov2_vitb14", "desc": "DINOv2 ViT-B/14"},
}


def raw_test_map(category: str) -> dict:
    """img_path (rel) -> raw metadata record (boolean detectable, image_anomaly)."""
    meta = json.load(open("data/m2ad/jsons_extract/meta_unsupervised.json"))
    return {r["img_path"]: r for r in meta["test"][category]}


def official_scope(category: str, seed: int = 0):
    """bank (300 normal) + test (view120), both from the frozen E7-1 scope; labels from RAW metadata."""
    scope = F.build_scope(category)
    bank = E8.bank_for(scope["pool"], seed)
    test = [r for r in scope["test"] if r["view"] == "120"]
    rawmap = raw_test_map(category)
    return bank, test, rawmap


def route(records: list, kind: str) -> list:
    return [dict(r, image_path=str(E8.cache_path(kind, r))) for r in records]


def load_dinov2(name: str, dev):
    model = torch.hub.load("facebookresearch/dinov2", name)
    model.eval().to(dev)
    return model


def dinov2_embed(model, path: str, dev):
    img = Image.open(path).convert("RGB")
    tf = T.Compose([T.Resize(size=E7.CACHE_RES, interpolation=T.InterpolationMode.BICUBIC,
                             antialias=True),
                    T.ToTensor(), T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)])
    t = tf(img)
    _, h, w = t.shape
    ps = model.patch_size
    t = t[:, :h - h % ps, :w - w % ps].to(dev)
    with torch.inference_mode():
        tok = model.get_intermediate_layers(t.unsqueeze(0), n=1)[0][0]  # (1024, C) patch tokens; CLS already excluded
    return tok.float()  # (1024, C)


def extract(bb: str, model_or_tm, paths: list, dev, kind: str):
    """Return concatenated patch features (N*1024, C), L2-normalised, float32."""
    out = []
    with torch.inference_mode():
        for i, p in enumerate(paths, 1):
            if bb == "resnet":
                emb = R10.extract_embedding(model_or_tm, __import__("experiment1b_defect_sensitivity",
                                                fromlist=["load_image_as_tensor"]).load_image_as_tensor(p), dev)
            else:
                emb = dinov2_embed(model_or_tm, p, dev)
            emb = emb / emb.norm(dim=1, keepdim=True).clamp_min(1e-12)
            out.append(emb.cpu().numpy().astype("float32"))
            if i % 100 == 0:
                print(f"    {bb} embed {i}/{len(paths)}", flush=True)
    return np.concatenate(out, axis=0)


def nn_image_scores(bank_f: np.ndarray, test_f: np.ndarray, dev):
    """Unified 1-NN: cosine distance to nearest bank patch, image score = max over patches."""
    bank_t = torch.from_numpy(bank_f).to(dev)
    n_bank = bank_t.shape[0]
    n_test = test_f.shape[0]
    n_img = n_test // 1024
    scores = np.empty(n_img, dtype="float32")
    with torch.inference_mode():
        for i in range(0, n_test, 1024 * CHUNK):
            tt = torch.from_numpy(test_f[i:i + 1024 * CHUNK]).to(dev)   # (B*1024, C)
            best = None
            for k in range(0, n_bank, BANK_CHUNK):
                sim = tt @ bank_t[k:k + BANK_CHUNK].T                     # cosine similarity
                m = sim.max(dim=1).values
                best = m if best is None else torch.maximum(best, m)
            dist = (1.0 - best).clamp_min(0.0)                            # (B*1024,)
            dist = dist.view(-1, 1024).max(dim=1).values                  # image score = max
            j0 = i // 1024
            scores[j0:j0 + dist.shape[0]] = dist.cpu().numpy()
    return scores


# ------------------------------------------------------------------ evaluation
def official_metrics(rows: list) -> dict:
    """rows: list of dicts with score / image_anomaly / detectable (raw boolean)."""
    from sklearn.metrics import roc_auc_score, average_precision_score
    kept = [r for r in rows if r["detectable"] != False]           # official filter (boolean)
    y = np.array([r["image_anomaly"] for r in kept], dtype=int)
    s = np.array([r["score"] for r in kept], dtype=float)
    auroc = float(roc_auc_score(y, s)) if len(set(y.tolist())) == 2 else float("nan")
    ap = float(average_precision_score(y, s)) if len(set(y.tolist())) == 2 else float("nan")
    return {"n_ok": int((y == 0).sum()), "n_ng": int((y == 1).sum()),
            "auroc": auroc, "ap": ap, "n_filtered": int(len(rows) - len(kept))}


# ------------------------------------------------------------------ run one backbone on one category
def run_backbone(bb: str, category: str, dev, smoke: bool = False) -> dict:
    spec = BACKBONES[bb]
    bank, test, rawmap = official_scope(category)
    if smoke:
        good = [r for r in test if r["kind"] == "Good"][:12]
        ng = [r for r in test if r["kind"] == "NG"][:12]
        test = good + ng
        bank = bank[:20]
    kind = spec["input_kind"]
    bank_r = route(bank, kind)
    test_r = route(test, kind)
    missing = [r["image_path"] for r in bank_r + test_r if not Path(r["image_path"]).exists()]
    if missing:
        return {"status": "FAILED", "reason": f"{len(missing)} cache images missing (e.g. {missing[:2]})"}

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    if bb == "resnet":
        tm = R10.fit_patchcore(bank_r, dev, f"p2_res_{category}", seed=SEED)
        model = tm
    else:
        model = load_dinov2(spec["hub"], dev)
    fit_s = time.time() - t0

    t0 = time.time()
    bank_f = extract(bb, model, [r["image_path"] for r in bank_r], dev, kind)
    test_f = extract(bb, model, [r["image_path"] for r in test_r], dev, kind)
    emb_s = time.time() - t0

    t0 = time.time()
    scores = nn_image_scores(bank_f, test_f, dev)
    score_s = time.time() - t0

    rel = lambda r: str(Path(r["image_path"]).relative_to(LD.DATA_ROOT))
    rows = []
    for r, s in zip(test, scores):
        meta = rawmap[rel(r)]
        rows.append({"specimen": r["specimen"], "kind": r["kind"],
                     "illumination": r["illumination"], "view": r["view"], "score": float(s),
                     "image_anomaly": int(meta["image_anomaly"]),
                     "detectable": meta["detectable"]})
    met = official_metrics(rows)
    peak = round(torch.cuda.max_memory_allocated() / 2**20, 1)
    del model, bank_f, test_f
    torch.cuda.empty_cache()
    return {"status": "OK", "backbone": bb, "category": category, "view": "120",
            "train_budget": len(bank), "n_bank_patches": int(bank_f.shape[0]) if False else len(bank) * 1024,
            "fit_s": round(fit_s, 1), "embed_s": round(emb_s, 1), "score_s": round(score_s, 1),
            "runtime_s": round(fit_s + emb_s + score_s, 1), "peak_vram_mb": peak,
            "metrics": met, "rows": rows}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--categories", default="Bird,Car,Cube")
    ap.add_argument("--backbones", default="resnet,dinov2s,dinov2b")
    args = ap.parse_args()
    dev = torch.device("cuda")
    cats = [c.strip() for c in args.categories.split(",") if c.strip()]
    bbs = [b.strip() for b in args.backbones.split(",") if b.strip()]
    print(f"[E11-P2] backbones={bbs} cats={cats} smoke={args.smoke} | "
          f"disk {E8.free_gb():.1f} GB", flush=True)

    outdir = ROOT / "results" / "e11"
    outdir.mkdir(parents=True, exist_ok=True)
    import csv
    csv_path = outdir / "p2_representation_results.csv"
    w = csv.writer(open(csv_path, "w", newline=""))
    w.writerow(["category", "view", "backbone", "scoring", "train_budget", "protocol",
                "n_ok", "n_ng", "seed", "auroc", "ap", "runtime", "peak_vram", "status"])

    raw_json = {}
    for cat in cats:
        for bb in bbs:
            print(f"\n[E11-P2] {'SMOKE' if args.smoke else 'RUN'} {bb}/{cat}", flush=True)
            try:
                out = run_backbone(bb, cat, dev, smoke=args.smoke)
            except Exception as exc:
                import traceback
                traceback.print_exc()
                out = {"status": "FAILED", "reason": f"{type(exc).__name__}: {exc}"}
            if out["status"] == "OK":
                m = out["metrics"]
                w.writerow([cat, "120", bb, "1-NN-max", out["train_budget"], "official_image_anomaly",
                            m["n_ok"], m["n_ng"], SEED, f"{m['auroc']:.5f}", f"{m['ap']:.5f}",
                            out["runtime_s"], out["peak_vram_mb"], "OK"])
                raw_json[f"{bb}|{cat}"] = out
                print(f"    AUROC={m['auroc']:.5f} AP={m['ap']:.5f} n_ok={m['n_ok']} n_ng={m['n_ng']} "
                      f"(filtered {m['n_filtered']})  runtime={out['runtime_s']:.0f}s "
                      f"VRAM={out['peak_vram_mb']:.0f}MB", flush=True)
            else:
                w.writerow([cat, "120", bb, "1-NN-max", "", "official_image_anomaly",
                            "", "", SEED, "", "", "", "", f"FAILED: {out.get('reason','')[:80]}"])
                print(f"    FAILED: {out.get('reason','')[:100]}", flush=True)

    if not args.smoke:
        json.dump(raw_json, open(outdir / "p2_raw_results.json", "w"), indent=2, default=str)
    print(f"\n[E11-P2] done -> {csv_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
