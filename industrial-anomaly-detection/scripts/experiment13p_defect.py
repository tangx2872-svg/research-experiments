"""Experiment 13-P — Diagnostic E: defect overlap（只读；**在 illumination subspace 冻结之后**）。

冻结前已完成：analysis/illumination_subspace_freeze.json（K=8, PRIMARY_centered, seed0）。
本脚本不重估 subspace、不选 K、不选 augmentation。

每 unit = 一个 category：抽取 test 图像（defect 各类型 + good）的 pooled patch 特征，
对 d = F_patch - mu_normal(category,layer)（mu_normal 来自 train/good，已由抽取阶段给出）累加
  S = Σ d dᵀ (C×C)  与  Σ||d||²  —— 足以精确计算任意子空间 U 的 mean overlap = trace(Uᵀ S U)/Σ||d||²。
输出：analysis/defect_stats_<cat>.npz（per layer × subset(defect type / good)）
"""
from __future__ import annotations

import argparse
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
LAYERS = ["layer2", "layer3"]
BATCH = 16


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--category", required=True)
    a = ap.parse_args()
    cat = a.category

    import experiment1b_defect_sensitivity as e1b
    from anomalib.models.image.patchcore.torch_model import PatchcoreModel

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    m = PatchcoreModel(backbone="wide_resnet50_2", layers=LAYERS, pre_trained=True,
                       num_neighbors=9).eval().to(dev)
    for p in m.parameters():
        p.requires_grad_(False)

    # mu_normal 来自 train/good（seed0 的 clean_sum / clean_count）
    z = np.load(EXP / "raw" / cat / "seed_0" / "stats.npz")
    meta = json.loads((EXP / "raw" / cat / "seed_0" / "meta.json").read_text())
    mu = {}
    for l in LAYERS:
        n = float(meta["clean_counts"][l])
        mu[l] = torch.as_tensor(z[f"clean_sum_{l}"] / n, dtype=torch.float64, device=dev)

    test_root = ROOT / "data" / "mvtec_ad" / cat / "test"
    subsets = []
    for dt in sorted(p.name for p in test_root.iterdir() if p.is_dir()):
        for img in sorted((test_root / dt).glob("*.png")):
            subsets.append((dt, img, 0))          # 0 = defect
    for img in sorted((test_root / "good").glob("*.png")):
        subsets.append(("good", img, 1))          # 1 = normal control

    C = {"layer2": 512, "layer3": 1024}
    S, NORM, CNT = {}, {}, {}
    for l in LAYERS:
        for tag in {s[0] for s in subsets}:
            S[(l, tag)] = torch.zeros(C[l], C[l], dtype=torch.float64, device=dev)
            NORM[(l, tag)] = 0.0
            CNT[(l, tag)] = 0
    t0 = time.time()
    with torch.no_grad():
        for i in range(0, len(subsets), BATCH):
            chunk = subsets[i:i + BATCH]
            x = torch.stack([e1b.preprocess_for_model(e1b.load_image_as_tensor(p), dev)
                             for _, p, _ in chunk])
            f = m.feature_extractor(x)
            f = {k: m.feature_pooler(v) for k, v in f.items()}
            for l in LAYERS:
                d = (f[l].reshape(-1, C[l]).double() - mu[l])          # (n_patch, C)
                for j, (tag, _, _) in enumerate(chunk):
                    sl = slice(j * 1024 if l == "layer2" else j * 256,
                               (j + 1) * 1024 if l == "layer2" else (j + 1) * 256)
                    dj = d[sl]
                    S[(l, tag)] += dj.transpose(0, 1) @ dj
                    NORM[(l, tag)] += float(dj.pow(2).sum())
                    CNT[(l, tag)] += int(dj.shape[0])
            del x, f
    out = {}
    for (l, tag), v in S.items():
        out[f"S_{l}_{tag}"] = v.cpu().numpy()
        out[f"norm_{l}_{tag}"] = np.array([NORM[(l, tag)]])
        out[f"cnt_{l}_{tag}"] = np.array([CNT[(l, tag)]])
    np.savez_compressed(EXP / "analysis" / f"defect_stats_{cat}.npz", **out)
    print("[13P-E] %s done: %d images, %d tags, %.1fs" % (cat, len(subsets),
          len({s[0] for s in subsets}), time.time() - t0), flush=True)
    for l in LAYERS:
        print("   %s: %s" % (l, {t: CNT[(l, t)] for t in sorted({s[0] for s in subsets})}))


if __name__ == "__main__":
    main()
