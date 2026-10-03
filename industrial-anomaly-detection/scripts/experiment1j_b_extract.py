"""Experiment 1J-B — Geometry → NN → Score Transmission Intervention 的提取。

对 frozen 9 primary defects × 3 seeds，从 1J-A 缓存的 Layer3 F0/F1（npz）做
radius-controlled intervention X(β)=μ1 + target_radius(β)*D1_norm，β∈{0,0.5,1}，
然后用 frozen M0（α=0 normal Layer3 bank）算 NN distance，输出 per-image
radius / NN / score_proxy。

纯 CPU/numpy 后处理，不重新 forward / fit / GPU。

输出（results/experiment_1j_b/）：
  raw/per_image.csv            逐 (defect, seed, β, image) 的 radius / NN / score_proxy
  sanity_checks.csv            reconstruction sanity
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

# 复用 1J-A 常量与 NN 统计（layer_nn_stats 是 (C,H,W) 输入版）
from experiment1j_extract import (  # noqa: E402
    FROZEN_PRIMARY,
    FAMILY_OF,
    SEEDS,
    layer_nn_stats,
)

J_OUT = ROOT / "results" / "experiment_1j"
B_OUT = ROOT / "results" / "experiment_1j_b"
RAW_ROOT = J_OUT / "raw"
BANK_ROOT = J_OUT / "banks"
EPS = 1e-12
BETAS = [0.0, 0.5, 1.0]


def rms_radius(X: np.ndarray) -> float:
    """X: (P,C)。RMS distance to centroid。"""
    mu = X.mean(axis=0, keepdims=True)
    Xc = X - mu
    return float(np.sqrt(np.mean(np.sum(Xc ** 2, axis=1))))


def intervene(X0: np.ndarray, X1: np.ndarray, beta: float) -> np.ndarray:
    """radius-controlled intervention。

    X0/X1: (P,C) 已 reshape 的 layer3 feature（α=0 / α=1）。
    返回 X(β)：(P,C)。
    """
    mu1 = X1.mean(axis=0, keepdims=True)
    D1 = X1 - mu1
    r0 = rms_radius(X0)
    r1 = rms_radius(X1)
    D1_norm = D1 / (r1 + EPS)
    target = r0 + beta * (r1 - r0)
    return mu1 + target * D1_norm


def feat_to_X(feat: np.ndarray) -> np.ndarray:
    """(C,H,W) -> (P,C)。"""
    C, H, W = feat.shape
    return feat.reshape(C, H * W).T.astype(np.float32)


def nn_max(feat: np.ndarray, bank: np.ndarray) -> float:
    """PatchCore num_neighbors=1 口径的 image score = max patch NN distance。

    与 layer_nn_stats 同用 einsum 展开欧氏距离，但返回 max 而非分布统计。
    """
    C, H, W = feat.shape
    X = feat.reshape(C, H * W).T.astype(np.float32)  # (P, C)
    P = X.shape[0]
    bank_norms = np.einsum("nc,nc->n", bank, bank)
    best = np.empty(P, dtype=np.float32)
    CHUNK = 512
    for s in range(0, P, CHUNK):
        e = min(s + CHUNK, P)
        xb = X[s:e]
        x_norms = np.einsum("pc,pc->p", xb, xb)
        d2 = -2.0 * (xb @ bank.T) + bank_norms[None, :]
        d2 += x_norms[:, None]
        np.maximum(d2, 0.0, out=d2)
        best[s:e] = np.sqrt(d2).min(axis=1)
    return float(best.max())


def main() -> None:
    (B_OUT / "raw").mkdir(parents=True, exist_ok=True)

    pi_fields = ["category", "defect_type", "family", "seed", "image_path",
                 "beta", "radius", "mean_nn", "std_nn", "median_nn", "score_proxy_max", "score_proxy_std"]
    pi_csv = B_OUT / "raw" / "per_image.csv"
    sanity_fields = ["category", "defect_type", "seed", "image_path", "beta",
                     "recon_maxdiff", "radius_target", "radius_actual"]
    sanity_csv = B_OUT / "sanity_checks.csv"

    pi_rows: list[dict] = []
    sanity_rows: list[dict] = []
    total_imgs = 0

    for fam, defects in FROZEN_PRIMARY.items():
        for cat, dt in defects:
            for seed in SEEDS:
                # 加载 frozen M0（layer3 bank）
                bank_path = BANK_ROOT / f"{cat}_seed{seed}_layer3.npy"
                if not bank_path.exists():
                    print(f"[WARN] missing bank {bank_path}", flush=True)
                    continue
                bank = np.load(bank_path).astype(np.float32)

                layer_dir = RAW_ROOT / cat / dt / f"seed{seed}" / "layer3"
                a0_files = sorted(layer_dir.glob("*_a0.npz"))
                for a0_path in a0_files:
                    stem = a0_path.stem.split("_a0")[0]
                    a1_path = layer_dir / f"{stem}_a1.npz"
                    if not a1_path.exists():
                        print(f"[WARN] missing a1 for {a0_path}", flush=True)
                        continue

                    F0 = np.load(a0_path)["feat"].astype(np.float32)  # (C,H,W)
                    F1 = np.load(a1_path)["feat"].astype(np.float32)
                    X0 = feat_to_X(F0)
                    X1 = feat_to_X(F1)

                    # 每个 β 构造干预 feature 并算 NN
                    for beta in BETAS:
                        Xb = intervene(X0, X1, beta)

                        # radius（干预后）
                        radius = rms_radius(Xb)

                        # NN -> M0
                        C = F0.shape[0]
                        H = F0.shape[1]
                        W = F0.shape[2]
                        feat_b = Xb.T.reshape(C, H, W).astype(np.float32)  # (C,H,W)
                        ns = layer_nn_stats(feat_b, bank)
                        score_max = nn_max(feat_b, bank)

                        img_png = ROOT / "data" / "mvtec_ad" / cat / "test" / dt / f"{stem}.png"
                        pi_rows.append({
                            "category": cat, "defect_type": dt, "family": fam,
                            "seed": seed, "image_path": str(img_png), "beta": beta,
                            "radius": radius,
                            "mean_nn": ns["mean_nn_distance"],
                            "std_nn": ns["std_nn_distance"],
                            "median_nn": ns["median_nn_distance"],
                            # score proxy：PatchCore num_neighbors=1 语义 = max patch NN；
                            # std 为 dispersion 分量（descriptive）
                            "score_proxy_max": score_max,
                            "score_proxy_std": ns["std_nn_distance"],
                        })

                    # reconstruction sanity（β=1 需复现 X1）
                    Xb1 = intervene(X0, X1, 1.0)
                    recon_diff = float(np.max(np.abs(Xb1 - X1)))
                    r_target_1 = rms_radius(X0) + 1.0 * (rms_radius(X1) - rms_radius(X0))
                    sanity_rows.append({
                        "category": cat, "defect_type": dt, "seed": seed,
                        "image_path": str(ROOT / "data" / "mvtec_ad" / cat / "test" / dt / f"{stem}.png"),
                        "beta": 1.0, "recon_maxdiff": recon_diff,
                        "radius_target": r_target_1, "radius_actual": rms_radius(Xb1),
                    })

                    total_imgs += 1

    # 写 CSV（覆盖重建，避免 id() 去重坑）
    with open(pi_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=pi_fields)
        w.writeheader()
        w.writerows(pi_rows)

    with open(sanity_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=sanity_fields)
        w.writeheader()
        w.writerows(sanity_rows)

    # 打印 sanity 汇总
    maxdiff = max(r["recon_maxdiff"] for r in sanity_rows)
    print(f"[DONE] total defect images: {total_imgs}")
    print(f"  per_image rows: {len(pi_rows)}")
    print(f"  sanity rows: {len(sanity_rows)}")
    print(f"  recon maxdiff (β=1): {maxdiff:.3e}  (需 < 1e-5)")


if __name__ == "__main__":
    main()
