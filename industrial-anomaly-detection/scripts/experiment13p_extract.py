"""Experiment 13-P — illumination-response paired-difference extraction（train/good only）。

每 unit = 一个 (category, sampling-seed)：抽取 N 张 train/good，对每张图计算
  dF_l,k(x,u) = F_l(T_k(x))[u] - F_l(x)[u]      （严格同一空间位置）
并**在线累加**逐 (layer, perturbation) 的一阶/二阶矩（避免落盘巨大的 per-patch 样本）：
  sum_d (C,), sum_ddT (C,C), count   ->  centered cov = sum_ddT - sum_d sum_d^T / n
  clean_sum (C,), clean_count        -> mu_normal（供 defect overlap 的 normal 对照）
输出：results/experiment_13p/raw/<category>/seed_<s>/stats.npz + meta.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "experiments" / "2026-09-30_illumination_sensitivity_exploration" / "falpha_patchcore"))

EXP = ROOT / "results" / "experiment_13p"
PERT = [("T1", "gamma", 0.8), ("T2", "gamma", 1.2), ("T3", "brightness", 0.8), ("T4", "brightness", 1.2)]
LAYERS = ["layer2", "layer3"]
BATCH = 16


def _sample(cat: str, seed: int, n: int) -> list:
    files = sorted(p.name for p in (ROOT / "data" / "mvtec_ad" / cat / "train" / "good").glob("*.png"))
    rng = np.random.RandomState(1000 + int(seed))
    idx = rng.choice(len(files), size=min(n, len(files)), replace=False)
    return [files[i] for i in sorted(idx.tolist())]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--category", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n", type=int, default=64)
    a = ap.parse_args()

    import experiment1_illumination_tradeoff as e1
    import experiment1b_defect_sensitivity as e1b
    from anomalib.models.image.patchcore.torch_model import PatchcoreModel

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    m = PatchcoreModel(backbone="wide_resnet50_2", layers=LAYERS, pre_trained=True,
                       num_neighbors=9).eval().to(dev)
    for p in m.parameters():
        p.requires_grad_(False)

    names = _sample(a.category, a.seed, a.n)
    paths = [ROOT / "data" / "mvtec_ad" / a.category / "train" / "good" / nm for nm in names]
    C = {"layer2": 512, "layer3": 1024}
    sum_d = {(l, p[0]): torch.zeros(C[l], dtype=torch.float64, device=dev) for l in LAYERS for p in PERT}
    sum_ddT = {(l, p[0]): torch.zeros(C[l], C[l], dtype=torch.float64, device=dev) for l in LAYERS for p in PERT}
    cnt = {(l, p[0]): 0 for l in LAYERS for p in PERT}
    clean_sum = {l: torch.zeros(C[l], dtype=torch.float64, device=dev) for l in LAYERS}
    clean_cnt = {l: 0 for l in LAYERS}

    t0 = time.time()
    with torch.no_grad():
        for i in range(0, len(paths), BATCH):
            raw = [e1b.load_image_as_tensor(p) for p in paths[i:i + BATCH]]
            x0 = torch.stack([e1b.preprocess_for_model(t, dev) for t in raw])
            f0 = m.feature_extractor(x0)
            f0 = {k: m.feature_pooler(v) for k, v in f0.items()}
            for l in LAYERS:
                clean_sum[l] += f0[l].reshape(-1, C[l]).double().sum(0)
                clean_cnt[l] += int(f0[l].reshape(-1, C[l]).shape[0])
            for pid, itype, level in PERT:
                xs = torch.stack([e1b.preprocess_for_model(e1.apply_photometric(t, itype, level), dev)
                                  for t in raw])
                fs = m.feature_extractor(xs)
                fs = {k: m.feature_pooler(v) for k, v in fs.items()}
                for l in LAYERS:
                    d = (fs[l] - f0[l]).reshape(-1, C[l]).double()
                    sum_d[(l, pid)] += d.sum(0)
                    sum_ddT[(l, pid)] += d.transpose(0, 1) @ d
                    cnt[(l, pid)] += int(d.shape[0])
                del xs, fs
            del x0, f0, raw
            print("  [%s seed%d] batch %d/%d  %.1fs" % (
                a.category, a.seed, i // BATCH + 1, (len(paths) + BATCH - 1) // BATCH,
                time.time() - t0), flush=True)
    runtime = time.time() - t0
    del m
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    out = {f"sum_d_{l}_{pid}": sum_d[(l, pid)].cpu().numpy() for l in LAYERS for pid, _, _ in PERT}
    out.update({f"sum_ddT_{l}_{pid}": sum_ddT[(l, pid)].cpu().numpy() for l in LAYERS for pid, _, _ in PERT})
    out.update({f"clean_sum_{l}": clean_sum[l].cpu().numpy() for l in LAYERS})
    d = EXP / "raw" / a.category / f"seed_{a.seed}"
    d.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(d / "stats.npz", **out)
    meta = {"category": a.category, "seed": a.seed, "n_requested": a.n, "n_used": len(names),
            "n_train_good_total": len(list((ROOT / "data" / "mvtec_ad" / a.category / "train" / "good").glob("*.png"))),
            "sampled_filenames": names,
            "sampled_filenames_sha256": hashlib.sha256("\n".join(names).encode()).hexdigest(),
            "counts": {f"{l}_{pid}": cnt[(l, pid)] for l in LAYERS for pid, _, _ in PERT},
            "clean_counts": {l: clean_cnt[l] for l in LAYERS},
            "runtime_seconds": runtime, "device": str(dev),
            "perturbations": PERT, "layers": LAYERS, "batch": BATCH, "ts": time.time()}
    (d / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    print("[13P] %s seed%d done: n=%d runtime=%.1fs rows/img=%d"
          % (a.category, a.seed, len(names), runtime, cnt[("layer2", "T1")] // len(names)), flush=True)


if __name__ == "__main__":
    main()
