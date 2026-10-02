"""Experiment 1E：Cross-Category Validation 的统一 runner。

复用 Experiment 1B 的底层科学计算（PatchCore / α-IN / fit / predict / split /
area），只改变：
  - category 参数化（由 --category 传入）
  - α 端点可配置（--alphas，smoke test 用 "0 1"，正式 Pilot 用完整 5 档）
  - 输出 1E 统一 sample-level schema（含 good_mean/good_std/z_score/area 字段）

核心科学定义【与 1B/1C/1D 完全一致，禁止修改】：
  - PatchCore: backbone=wide_resnet50_2, layers=[layer2,layer3],
    coreset_sampling_ratio=0.1, num_neighbors=9
  - α-IN: F_alpha = (1-α)F + α·IN(F), IN=InstanceNorm(affine=False),
    位于 generate_embedding concat 后 reshape 前；α=0 直接返回原始 feature
  - validation split: 每 category 从 train/good 固定 seed 划 VAL_SIZE=20 张，
    剩余进 memory bank
  - z-score: z_i(α) = (s_i(α) - μ_good(α)) / σ_good(α)，
    μ_good/σ_good 用【同 category 同 seed 同 α】的 test/good 分布，σ 用 ddof=1
  - d': d' = (μ_D - μ_G) / sqrt((σ_D² + σ_G²)/2)
  - area: GT mask pixels / total pixels（用原始分辨率 H×W）

输出目录：results/experiment_1e/<category>/seed_<seed>/
  sample_level.csv    —— 统一 sample-level schema
  group_level.csv     —— group-level metrics（含 d'）
  config.json         —— 本次运行配置

用法（smoke test）：
  python scripts/experiment1e_runner.py --category hazelnut --seed 0 --alphas "0 1"

用法（正式 Pilot 单 category 单 seed）：
  python scripts/experiment1e_runner.py --category cable --seed 0 --alphas "0 0.25 0.5 0.75 1"
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

# 复用 1B 模块的底层实现（不改任何科学计算）
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import experiment1b_defect_sensitivity as e1b  # noqa: E402

DATA_ROOT = e1b.DATA_ROOT
EPS = 1e-9  # log_area_ratio 的 ε，固定并记录


def compute_area(mask_path: Path) -> tuple[int, float, float]:
    """返回 (area_pixels, area_ratio, log_area_ratio)。

    area_ratio 用 mask 原始分辨率计算（GT 像素 / 总像素）。
    """
    m = e1b.load_mask(mask_path)
    h, w = m.shape
    area_pixels = int(m.sum())
    area_ratio = float(area_pixels) / float(h * w)
    log_area = math.log(area_ratio + EPS)
    return area_pixels, area_ratio, log_area


def run_category_seed(
    category: str,
    seed: int,
    alphas: list[float],
    out_dir: Path,
) -> None:
    """对单个 (category, seed) 跑指定 α 端点，输出统一 schema。"""
    out_dir.mkdir(parents=True, exist_ok=True)

    # 让 1B 的 fit_model 用本次运行目录的 logs（fit_model 内部引用 LOGS_DIR 全局）
    logs_dir = out_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    e1b.LOGS_DIR = logs_dir

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 前置守卫
    if not (DATA_ROOT / category).is_dir():
        raise FileNotFoundError(
            f"category '{category}' 数据不存在: {DATA_ROOT / category}"
        )

    defect_types = e1b.discover_defect_types(category)
    if not defect_types:
        raise RuntimeError(f"category '{category}' 无 defect type")

    val_ids, train_ids = e1b.make_validation_split(category=category, seed=seed)
    print(f"[{category} seed={seed}] val={len(val_ids)} train={len(train_ids)}")

    test_root = DATA_ROOT / category / "test"
    good_paths = sorted((test_root / "good").glob("*.png"))
    defect_paths: dict[str, list[Path]] = {
        dt: sorted((test_root / dt).glob("*.png")) for dt in defect_types
    }

    # 预计算 area（GT mask，原始分辨率）
    area_cache: dict[str, tuple[int, float, float]] = {}
    for dt in defect_types:
        for p in defect_paths[dt]:
            mask_path = DATA_ROOT / category / "ground_truth" / dt / f"{p.stem}_mask.png"
            area_cache[str(p)] = compute_area(mask_path) if mask_path.exists() else (0, float("nan"), float("nan"))

    sample_rows: list[dict] = []
    group_rows: list[dict] = []
    resource_rows: list[dict] = []

    for alpha in alphas:
        print(f"\n{'='*60}\n[{category} seed={seed} alpha={alpha:g}] fit\n{'='*60}")
        t0 = time.time()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        lightning_model, torch_model, datamodule = e1b.fit_model(
            alpha, category, train_ids, seed=seed
        )
        e1b._move_model_to_device(torch_model, device)

        # ---- Phase 8 资源记录：coreset_size / peak_gpu ----
        coreset_size = "not_measured"
        try:
            if hasattr(torch_model, "memory_bank") and isinstance(torch_model.memory_bank, torch.Tensor):
                coreset_size = int(torch_model.memory_bank.shape[0])
        except Exception:
            coreset_size = "not_measured"
        peak_gpu_mb = "not_measured"
        if torch.cuda.is_available():
            try:
                peak_gpu_mb = round(torch.cuda.max_memory_allocated() / (1024 * 1024), 2)
            except Exception:
                peak_gpu_mb = "not_measured"
        resource_rows.append({
            "category": category,
            "seed": seed,
            "alpha": alpha,
            "runtime_seconds": round(time.time() - t0, 2),
            "train_good_count": len(train_ids),
            "validation_good_count": len(val_ids),
            "memory_bank_train_count": len(train_ids),
            "coreset_size": coreset_size,
            "peak_gpu_memory_allocated_mb": peak_gpu_mb,
            "status": "OK",
        })

        # ---- 推理 test/good → 得到 contemporaneous good 分布 ----
        good_scores = []
        for p in good_paths:
            img = e1b.load_image_as_tensor(p)
            score, _ = e1b.predict_one(torch_model, img, device)
            good_scores.append(score)

        mu_good = float(np.mean(good_scores))
        sigma_good = float(np.std(good_scores, ddof=1))
        print(f"  good n={len(good_scores)} mu={mu_good:.4f} sigma={sigma_good:.4f} (ddof=1)")

        # ---- 记录 good 样本（z 用同 α good 分布）----
        for p, s in zip(good_paths, good_scores):
            z = (s - mu_good) / sigma_good if sigma_good > 0 else float("nan")
            sample_rows.append({
                "category": category,
                "defect_type": "good",
                "image_path": str(p),
                "seed": seed,
                "alpha": alpha,
                "is_good": 1,
                "gt_label": 0,
                "anomaly_score": s,
                "good_mean": mu_good,
                "good_std": sigma_good,
                "z_score": z,
                "mask_path": "",
                "area_pixels": "",
                "area_ratio": "",
                "log_area_ratio": "",
            })

        # ---- 推理 defect ----
        for dt in defect_types:
            scores = []
            for p in defect_paths[dt]:
                img = e1b.load_image_as_tensor(p)
                score, _ = e1b.predict_one(torch_model, img, device)
                scores.append(score)
                ap, ar, lar = area_cache[str(p)]
                z = (score - mu_good) / sigma_good if sigma_good > 0 else float("nan")
                mask_path = DATA_ROOT / category / "ground_truth" / dt / f"{p.stem}_mask.png"
                sample_rows.append({
                    "category": category,
                    "defect_type": dt,
                    "image_path": str(p),
                    "seed": seed,
                    "alpha": alpha,
                    "is_good": 0,
                    "gt_label": 1,
                    "anomaly_score": score,
                    "good_mean": mu_good,
                    "good_std": sigma_good,
                    "z_score": z,
                    "mask_path": str(mask_path) if mask_path.exists() else "",
                    "area_pixels": ap,
                    "area_ratio": ar,
                    "log_area_ratio": lar,
                })

            # group-level 统计（当前 dt × α）
            s_arr = np.array(scores)
            mu_d = float(s_arr.mean())
            sigma_d = float(s_arr.std(ddof=1))
            mean_gap = mu_d - mu_good
            pooled_std = math.sqrt((sigma_d**2 + sigma_good**2) / 2.0)
            d_prime = mean_gap / pooled_std if pooled_std > 0 else float("nan")
            mean_z = float(np.mean([(x - mu_good) / sigma_good for x in scores])) if sigma_good > 0 else float("nan")
            group_rows.append({
                "category": category,
                "defect_type": dt,
                "seed": seed,
                "alpha": alpha,
                "n_good": len(good_scores),
                "n_defect": len(scores),
                "mean_good": mu_good,
                "std_good": sigma_good,
                "mean_defect": mu_d,
                "std_defect": sigma_d,
                "mean_gap": mean_gap,
                "mean_z": mean_z,
                "d_prime": d_prime,
            })

        # 释放显存
        del lightning_model, torch_model, datamodule
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ---- 写 sample-level CSV ----
    sample_fields = ["category", "defect_type", "image_path", "seed", "alpha",
                     "is_good", "gt_label", "anomaly_score", "good_mean", "good_std",
                     "z_score", "mask_path", "area_pixels", "area_ratio", "log_area_ratio"]
    with open(out_dir / "sample_level.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=sample_fields)
        w.writeheader()
        w.writerows(sample_rows)

    # ---- 写 group-level CSV ----
    group_fields = ["category", "defect_type", "seed", "alpha", "n_good", "n_defect",
                    "mean_good", "std_good", "mean_defect", "std_defect",
                    "mean_gap", "mean_z", "d_prime"]
    with open(out_dir / "group_level.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=group_fields)
        w.writeheader()
        w.writerows(group_rows)

    # ---- 写 config ----
    config = {
        "experiment": "Experiment 1E Cross-Category Validation",
        "category": category,
        "seed": seed,
        "alphas": alphas,
        "defect_types": defect_types,
        "backbone": "wide_resnet50_2",
        "layers": ["layer2", "layer3"],
        "coreset_sampling_ratio": 0.1,
        "num_neighbors": 9,
        "image_size": e1b.IMAGE_SIZE,
        "val_size": e1b.VAL_SIZE,
        "train_size": len(train_ids),
        "n_test_good": len(good_paths),
        "log_area_epsilon": EPS,
        "z_normalization": "test/good contemporaneous (same seed+alpha), ddof=1",
        "d_prime": "pooled_std sqrt((sd^2+sg^2)/2)",
    }
    with open(out_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    # ---- 写 resource CSV（Phase 8 资源记录）----
    resource_fields = ["category", "seed", "alpha", "runtime_seconds",
                       "train_good_count", "validation_good_count",
                       "memory_bank_train_count", "coreset_size",
                       "peak_gpu_memory_allocated_mb", "status"]
    with open(out_dir / "resource.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=resource_fields)
        w.writeheader()
        w.writerows(resource_rows)

    print(f"\n[done] {category} seed={seed}: sample_rows={len(sample_rows)}, "
          f"group_rows={len(group_rows)}")
    print(f"[done] 输出目录 = {out_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Experiment 1E cross-category runner")
    parser.add_argument("--category", type=str, required=True, help="MVTec AD category")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--alphas", type=str, default="0 1",
                        help="空格分隔的 α 列表，如 '0 1'（smoke）或 '0 0.25 0.5 0.75 1'（正式）")
    parser.add_argument("--out-root", type=str, default=None,
                        help="输出根目录（默认 results/experiment_1e）")
    args = parser.parse_args()

    alphas = [float(x) for x in args.alphas.split() if x.strip()]
    out_root = Path(args.out_root) if args.out_root else PROJECT_ROOT / "results" / "experiment_1e"
    out_dir = out_root / args.category / f"seed_{args.seed}"

    run_category_seed(category=args.category, seed=args.seed, alphas=alphas, out_dir=out_dir)


if __name__ == "__main__":
    main()
