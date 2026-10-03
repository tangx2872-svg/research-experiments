"""从 raw/*.npz + banks/*.npy 重建 1J-A 的两个 CSV。

背景：extract.py 的 flush_pi/flush_nn 用了脆弱的 id() 去重，导致 ~50% 行丢失。
npz（1848 个，完整）与 bank（30 个，完整）都正确，因此无需重新跑 GPU 提取，
直接从 npz 重算 geometry + NN 即可重建完整 CSV。

输出：
  per_image/experiment1j_geometry_per_image.csv
  nn/experiment1j_nn_per_image.csv
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

# 复用 extract.py 的 geometry_of / layer_nn_stats / FROZEN_PRIMARY 常量
from experiment1j_extract import (  # noqa: E402
    FROZEN_PRIMARY,
    FAMILY_OF,
    geometry_of,
    layer_nn_stats,
    SEEDS,
    LAYERS,
    ALPHAS,
)

OUT_ROOT = ROOT / "results" / "experiment_1j"
RAW_ROOT = OUT_ROOT / "raw"
BANKS_ROOT = OUT_ROOT / "banks"

pi_fields = ["category", "defect_type", "family", "seed", "image_path", "layer", "alpha",
             "mdc", "rms_radius", "normalized_pr", "pca1_ratio", "feature_norm"]
nn_fields = ["category", "defect_type", "family", "seed", "layer", "alpha",
             "mean_nn_distance", "std_nn_distance", "median_nn_distance"]


def main():
    pi_rows: list[dict] = []
    nn_rows: list[dict] = []
    missing_bank = 0
    total = 0

    for fam, defects in FROZEN_PRIMARY.items():
        for cat, dt in defects:
            for seed in SEEDS:
                # 加载该 (cat, seed) 的两个 layer bank
                banks = {}
                for layer in LAYERS:
                    bp = BANKS_ROOT / f"{cat}_seed{seed}_{layer}.npy"
                    if bp.exists():
                        banks[layer] = np.load(bp)
                    else:
                        banks[layer] = None
                        missing_bank += 1

                for layer in LAYERS:
                    layer_dir = RAW_ROOT / cat / dt / f"seed{seed}" / layer
                    if not layer_dir.exists():
                        continue
                    for npz_path in sorted(layer_dir.glob("*.npz")):
                        # 从文件名解析 alpha：000_a0.npz -> alpha=0.0
                        stem_img, a_str = npz_path.stem.split("_a", 1)
                        alpha = float(a_str)
                        total += 1
                        feat = np.load(npz_path)["feat"].astype(np.float32)  # (C,H,W)
                        # 重建原始 png 路径，与 extract.py 的 image_path 口径一致
                        png_path = (ROOT / "data" / "mvtec_ad" / cat / "test" / dt /
                                    f"{stem_img}.png")

                        g = geometry_of(feat)
                        pi_rows.append({
                            "category": cat, "defect_type": dt, "family": fam,
                            "seed": seed, "image_path": str(png_path),
                            "layer": layer, "alpha": alpha, **g,
                        })

                        bank = banks[layer]
                        if bank is not None:
                            ns = layer_nn_stats(feat, bank)
                            nn_rows.append({
                                "category": cat, "defect_type": dt, "family": fam,
                                "seed": seed, "layer": layer, "alpha": alpha, **ns,
                            })

    # 写 CSV（覆盖重建）
    pi_csv = OUT_ROOT / "per_image" / "experiment1j_geometry_per_image.csv"
    nn_csv = OUT_ROOT / "nn" / "experiment1j_nn_per_image.csv"
    pi_csv.parent.mkdir(parents=True, exist_ok=True)
    nn_csv.parent.mkdir(parents=True, exist_ok=True)

    with open(pi_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=pi_fields)
        w.writeheader()
        w.writerows(pi_rows)

    with open(nn_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=nn_fields)
        w.writeheader()
        w.writerows(nn_rows)

    print(f"[DONE] 重建完成。")
    print(f"  npz 处理数: {total}")
    print(f"  pi rows: {len(pi_rows)}")
    print(f"  nn rows: {len(nn_rows)}")
    print(f"  missing bank: {missing_bank}")


if __name__ == "__main__":
    main()
