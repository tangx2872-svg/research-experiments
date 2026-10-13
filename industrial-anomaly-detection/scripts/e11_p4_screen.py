#!/usr/bin/env python
"""E11-P4 Round 1 - Lightweight Module Screening (frozen DINOv2-B, 1-NN-max).

Candidates (frozen config `results/e11/p4_module_config.json`):
  baseline = block11 L2-norm
  A1/A2    = concat(last block11, block9 / block7) L2-norm
  B1/B2    = 3x3 / 5x5 avg-pool smoothing of block11, L2-norm
  C1/C2    = concat(448 last, 224 / 336 last upsampled to 32x32) L2-norm
Reuses a single 448 forward (get_intermediate_layers n=5 -> blocks 7..11); only C needs extra 224/336 forwards.
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
import torch.nn.functional as F  # noqa: E402
import torchvision.transforms as T  # noqa: E402
from PIL import Image  # noqa: E402

import e4d1_m2ad_loader as LD  # noqa: E402
import e8_common as E8  # noqa: E402
from e11_p2_runner import load_dinov2, raw_test_map, official_metrics, BACKBONES  # noqa: E402

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
CHUNK = 8
BANK_CHUNK = 65536
CONFIGS = ["baseline", "A1", "A2", "B1", "B2", "C1", "C2"]
LAYER_KEYS = ["l7", "l8", "l9", "l10", "l11"]


def _prep(img: Image.Image, sz: int, model, dev):
    tf = T.Compose([T.Resize(sz, interpolation=T.InterpolationMode.BICUBIC, antialias=True),
                    T.ToTensor(), T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)])
    t = tf(img)
    ps = model.patch_size
    return t[:, :t.shape[1] - t.shape[1] % ps, :t.shape[2] - t.shape[2] % ps].to(dev)


def extract_ms(model, path: str, dev) -> dict:
    """One image -> dict of numpy patch-token matrices {l7..l11:(1024,768), s224:(256,768), s336:(576,768)}."""
    img = Image.open(path).convert("RGB")
    with torch.inference_mode():
        t448 = _prep(img, 448, model, dev)
        layers = model.get_intermediate_layers(t448.unsqueeze(0), n=5, norm=True)  # blocks 7..11
        out = {k: layers[i][0].float().cpu().numpy() for i, k in enumerate(LAYER_KEYS)}
        t224 = _prep(img, 224, model, dev)
        out["s224"] = model.get_intermediate_layers(t224.unsqueeze(0), n=1, norm=True)[0][0].float().cpu().numpy()
        t336 = _prep(img, 336, model, dev)
        out["s336"] = model.get_intermediate_layers(t336.unsqueeze(0), n=1, norm=True)[0][0].float().cpu().numpy()
    return out


def smooth(x: torch.Tensor, k: int) -> torch.Tensor:
    n_img = x.shape[0] // 1024
    xx = x.reshape(n_img, 32, 32, -1).permute(0, 3, 1, 2)
    xx = F.avg_pool2d(xx, kernel_size=k, stride=1, padding=k // 2)
    return xx.permute(0, 2, 3, 1).reshape(n_img * 1024, -1)


def candidate_tensor(stack: dict, cfg: str, dev) -> torch.Tensor:
    """stack: dict key->numpy (M,C). Returns L2-normalized torch (M,C) on dev."""
    def t(k):
        return torch.from_numpy(stack[k]).to(dev)
    if cfg == "baseline":
        x = t("l11")
    elif cfg == "A1":
        x = torch.cat([t("l11"), t("l9")], dim=1)
    elif cfg == "A2":
        x = torch.cat([t("l11"), t("l7")], dim=1)
    elif cfg in ("B1", "B2"):
        x = smooth(t("l11"), 3 if cfg == "B1" else 5)
    elif cfg == "C1":
        x = torch.cat([t("l11"), upsample_grid(t("s224"), 16)], dim=1)
    elif cfg == "C2":
        x = torch.cat([t("l11"), upsample_grid(t("s336"), 24)], dim=1)
    else:
        raise ValueError(cfg)
    return x / x.norm(dim=1, keepdim=True).clamp_min(1e-12)


def upsample_grid(x: torch.Tensor, H: int) -> torch.Tensor:
    """x: (N_img * H*H, C) -> (N_img*1024, C) nearest-neighbour upsampled to 32x32."""
    n_img = x.shape[0] // (H * H)
    xx = x.reshape(n_img, H, H, -1).permute(0, 3, 1, 2)
    xx = F.interpolate(xx, size=(32, 32), mode="nearest")
    return xx.permute(0, 2, 3, 1).reshape(n_img * 1024, -1)


# ------------------------------------------------------------------ scoring
def score_candidate(bank_t: torch.Tensor, test_t: torch.Tensor, dev) -> np.ndarray:
    n_bank, n_test = bank_t.shape[0], test_t.shape[0]
    n_img = n_test // 1024
    scores = np.empty(n_img, dtype="float32")
    with torch.inference_mode():
        for i in range(0, n_test, 1024 * CHUNK):
            tt = test_t[i:i + 1024 * CHUNK]
            best = None
            for k in range(0, n_bank, BANK_CHUNK):
                sim = tt @ bank_t[k:k + BANK_CHUNK].T
                m = sim.max(dim=1).values
                best = m if best is None else torch.maximum(best, m)
            dist = (1.0 - best).clamp_min(0.0).view(-1, 1024).max(dim=1).values
            j0 = i // 1024
            scores[j0:j0 + dist.shape[0]] = dist.cpu().numpy()
    return scores


def run_category(category: str, dev) -> dict:
    from e11_p2_runner import route
    recs = LD.load_records(category=category, require_exists=False)
    est = LD.estimate_scope(recs)
    bank_specs = set(est["bank_specimens"])
    bank = E8.bank_for(sorted(LD.select(recs, bank_specs), key=lambda r: r["image_path"]), 0)
    test = [r for r in LD.select(recs, views=["120"]) if r["split"] == "test"]
    bank_r = route(bank, "l0_448")
    test_r = route(test, "l0_448")
    model = load_dinov2(BACKBONES["dinov2b"]["hub"], dev)

    keys = LAYER_KEYS + ["s224", "s336"]
    t0 = time.time()
    bank_ms = {k: [] for k in keys}
    for r in bank_r:
        m = extract_ms(model, r["image_path"], dev)
        for k in keys:
            bank_ms[k].append(m[k])
    test_ms = {k: [] for k in keys}
    for r in test_r:
        m = extract_ms(model, r["image_path"], dev)
        for k in keys:
            test_ms[k].append(m[k])
    bank_stack = {k: np.concatenate(v, axis=0) for k, v in bank_ms.items()}
    test_stack = {k: np.concatenate(v, axis=0) for k, v in test_ms.items()}
    del bank_ms, test_ms
    extract_s = time.time() - t0

    rel = lambda r: str(Path(r["image_path"]).relative_to(LD.DATA_ROOT))
    rawmap = raw_test_map(category)
    out = {"category": category, "extract_s": round(extract_s, 1), "candidates": {}}
    for cfg in CONFIGS:
        t0 = time.time()
        bank_t = candidate_tensor(bank_stack, cfg, dev)
        test_t = candidate_tensor(test_stack, cfg, dev)
        scores = score_candidate(bank_t, test_t, dev)
        rows = []
        for r, s in zip(test, scores):
            m = rawmap[rel(r)]
            rows.append({"specimen": r["specimen"], "kind": r["kind"],
                         "illumination": r["illumination"], "view": "120", "score": float(s),
                         "image_anomaly": int(m["image_anomaly"]), "detectable": m["detectable"]})
        met = official_metrics(rows)
        out["candidates"][cfg] = {"metrics": met,
                                  "score_s": round(time.time() - t0, 1),
                                  "peak_vram_mb": round(torch.cuda.max_memory_allocated() / 2**20, 1),
                                  "dim": bank_t.shape[1]}
        del bank_t, test_t
        torch.cuda.empty_cache()
        print(f"    {cfg:9s} auroc={met['auroc']:.5f} ap={met['ap']:.5f} dim={out['candidates'][cfg]['dim']}",
              flush=True)
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

    w = csv.writer(open(outdir / "p4_module_screening.csv", "w", newline=""))
    w.writerow(["category", "view", "module", "auroc", "ap", "d_auroc", "d_ap", "dim",
                "extract_s", "score_s", "peak_vram_mb", "status"])
    raw = {}
    for cat in cats:
        print(f"\n[E11-P4] RUN {cat}", flush=True)
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
        for cfg in CONFIGS:
            c = out["candidates"][cfg]
            d_au = c["metrics"]["auroc"] - base["auroc"]
            d_ap = c["metrics"]["ap"] - base["ap"]
            w.writerow([cat, "120", cfg, f"{c['metrics']['auroc']:.5f}", f"{c['metrics']['ap']:.5f}",
                        f"{d_au:+.5f}", f"{d_ap:+.5f}", c["dim"], out["extract_s"], c["score_s"],
                        c["peak_vram_mb"], "OK"])
    json.dump(raw, open(outdir / "p4_module_raw.json", "w"), indent=2, default=str)
    print(f"\n[E11-P4] done -> {outdir/'p4_module_screening.csv'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
