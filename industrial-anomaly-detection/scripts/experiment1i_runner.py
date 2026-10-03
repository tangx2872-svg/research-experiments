"""Experiment 1I — Spatial-Statistics Control：单 (category, seed, phase) runner。

复用 1H 的底层工具（数据预处理 / validation split / mask / area / predict_one），
模型换成 FAlphaMatched256Patchcore（layer2 用 matched256 IN 统计）。

每 (category, seed, phase) 为最小单元：跑 α ∈ {0, 1}。
α=0 与 1H layer2 α=0 理论 bit-wise 等价（不调用 IN），orchestrator 做等价 sanity。

输出（results/experiment_1i/<category>/seed_<seed>/phase_<phase>/）：
  sample_level.csv  group_level.csv  config.json  resource.csv

primary 指标与 1H/1H-S 对齐：std_defect（defect 类 anomaly score 的样本标准差），
Δstd = std(α=1) − std(α=0)。

用法：
  python scripts/experiment1i_runner.py --category bottle --seed 0 --phase 00 --alphas "0 1"
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

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
ARCHIVE_FALPHA = (
    PROJECT_ROOT
    / "experiments"
    / "2026-09-30_illumination_sensitivity_exploration"
    / "falpha_patchcore"
)
if str(ARCHIVE_FALPHA) not in sys.path:
    sys.path.insert(0, str(ARCHIVE_FALPHA))

import experiment1b_defect_sensitivity as e1b  # noqa: E402
from falpha_matched256_patchcore import FAlphaMatched256Patchcore  # noqa: E402

from anomalib.data import MVTecAD  # noqa: E402
from anomalib.engine import Engine  # noqa: E402

DATA_ROOT = e1b.DATA_ROOT
EPS = 1e-9
PHASES = ["00", "01", "10", "11"]


def fit_matched256_model(alpha: float, category: str, train_ids: list[str],
                         phase: str, seed: int = 0, logs_dir: Path | None = None):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    datamodule = MVTecAD(
        root=str(DATA_ROOT),
        category=category,
        train_batch_size=16,
        eval_batch_size=16,
        num_workers=0,
        seed=seed,
    )
    datamodule.setup()

    keep_names = {n for n in train_ids}
    td = datamodule.train_data
    df = td._samples
    df = df[df["image_path"].apply(lambda p: str(Path(str(p)).name) in keep_names)].reset_index(drop=True)
    td._samples = df
    if hasattr(td, "_num_samples"):
        td._num_samples = len(df)

    model = FAlphaMatched256Patchcore(
        backbone="wide_resnet50_2",
        layers=["layer2", "layer3"],
        coreset_sampling_ratio=0.1,
        num_neighbors=9,
        alpha=alpha,
        phase=phase,
        visualizer=False,
    )

    default_root = str(logs_dir / f"fit_alpha_{alpha:g}") if logs_dir else None
    engine = Engine(
        enable_progress_bar=False,
        logger=False,
        barebones=True,
        default_root_dir=default_root or "results",
    )
    engine.fit(model=model, datamodule=datamodule)

    torch_model = model.model
    torch_model.eval()
    return model, torch_model, datamodule


def compute_area(mask_path: Path) -> tuple[int, float, float]:
    m = e1b.load_mask(mask_path)
    h, w = m.shape
    area_pixels = int(m.sum())
    area_ratio = float(area_pixels) / float(h * w)
    log_area = math.log(area_ratio + EPS)
    return area_pixels, area_ratio, log_area


def image_auroc(scores_good: np.ndarray, scores_defect: np.ndarray) -> float:
    y = np.concatenate([np.zeros(len(scores_good)), np.ones(len(scores_defect))])
    s = np.concatenate([scores_good, scores_defect])
    if len(np.unique(y)) < 2:
        return float("nan")
    order = np.argsort(s)
    y_sorted = y[order]
    n_pos = int(y.sum())
    n_neg = len(y) - n_pos
    ranks = np.arange(1, len(y) + 1)
    rank_pos = ranks[y_sorted == 1]
    return float((rank_pos.sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def run_condition(category: str, seed: int, phase: str, alphas: list[float],
                  out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    logs_dir = out_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not (DATA_ROOT / category).is_dir():
        raise FileNotFoundError(f"category '{category}' 数据不存在")

    defect_types = e1b.discover_defect_types(category)
    if not defect_types:
        raise RuntimeError(f"category '{category}' 无 defect type")

    val_ids, train_ids = e1b.make_validation_split(category=category, seed=seed)
    print(f"[{category} seed={seed} phase={phase}] val={len(val_ids)} train={len(train_ids)}")

    test_root = DATA_ROOT / category / "test"
    good_paths = sorted((test_root / "good").glob("*.png"))
    defect_paths = {dt: sorted((test_root / dt).glob("*.png")) for dt in defect_types}

    area_cache = {}
    for dt in defect_types:
        for p in defect_paths[dt]:
            mask_path = DATA_ROOT / category / "ground_truth" / dt / f"{p.stem}_mask.png"
            area_cache[str(p)] = compute_area(mask_path) if mask_path.exists() else (0, float("nan"), float("nan"))

    sample_rows: list[dict] = []
    group_rows: list[dict] = []
    resource_rows: list[dict] = []

    for alpha in alphas:
        print(f"\n[phase={phase} alpha={alpha:g}] fit")
        t0 = time.time()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        lightning_model, torch_model, datamodule = fit_matched256_model(
            alpha, category, train_ids, phase, seed=seed, logs_dir=logs_dir)
        e1b._move_model_to_device(torch_model, device)

        peak_gpu_mb = "not_measured"
        if torch.cuda.is_available():
            try:
                peak_gpu_mb = round(torch.cuda.max_memory_allocated() / (1024 * 1024), 2)
            except Exception:
                pass
        resource_rows.append({
            "category": category, "seed": seed, "phase": phase, "alpha": alpha,
            "runtime_seconds": round(time.time() - t0, 2),
            "train_good_count": len(train_ids),
            "peak_gpu_memory_allocated_mb": peak_gpu_mb,
            "status": "OK",
        })

        good_scores = []
        for p in good_paths:
            img = e1b.load_image_as_tensor(p)
            score, _ = e1b.predict_one(torch_model, img, device)
            good_scores.append(score)
        mu_good = float(np.mean(good_scores))
        sigma_good = float(np.std(good_scores, ddof=1))

        for p, s in zip(good_paths, good_scores):
            z = (s - mu_good) / sigma_good if sigma_good > 0 else float("nan")
            sample_rows.append({
                "category": category, "defect_type": "good", "image_path": str(p),
                "seed": seed, "phase": phase, "alpha": alpha,
                "is_good": 1, "gt_label": 0, "anomaly_score": s,
                "good_mean": mu_good, "good_std": sigma_good, "z_score": z,
                "mask_path": "", "area_pixels": "", "area_ratio": "", "log_area_ratio": "",
            })

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
                    "category": category, "defect_type": dt, "image_path": str(p),
                    "seed": seed, "phase": phase, "alpha": alpha,
                    "is_good": 0, "gt_label": 1, "anomaly_score": score,
                    "good_mean": mu_good, "good_std": sigma_good, "z_score": z,
                    "mask_path": str(mask_path) if mask_path.exists() else "",
                    "area_pixels": ap, "area_ratio": ar, "log_area_ratio": lar,
                })
            s_arr = np.array(scores)
            mu_d = float(s_arr.mean())
            sigma_d = float(s_arr.std(ddof=1))
            median_d = float(np.median(s_arr))
            mean_gap = mu_d - mu_good
            pooled_std = math.sqrt((sigma_d**2 + sigma_good**2) / 2.0)
            d_prime = mean_gap / pooled_std if pooled_std > 0 else float("nan")
            mean_z = float(np.mean([(x - mu_good) / sigma_good for x in scores])) if sigma_good > 0 else float("nan")
            img_auc = image_auroc(np.array(good_scores), s_arr)
            group_rows.append({
                "category": category, "defect_type": dt, "seed": seed, "phase": phase, "alpha": alpha,
                "n_good": len(good_scores), "n_defect": len(scores),
                "mean_good": mu_good, "std_good": sigma_good,
                "mean_defect": mu_d, "std_defect": sigma_d, "median_defect": median_d,
                "mean_gap": mean_gap, "mean_z": mean_z, "d_prime": d_prime,
                "image_auroc": img_auc,
            })

        del lightning_model, torch_model, datamodule
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    sample_fields = ["category", "defect_type", "image_path", "seed", "phase", "alpha",
                     "is_good", "gt_label", "anomaly_score", "good_mean", "good_std",
                     "z_score", "mask_path", "area_pixels", "area_ratio", "log_area_ratio"]
    with open(out_dir / "sample_level.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=sample_fields)
        w.writeheader()
        w.writerows(sample_rows)

    group_fields = ["category", "defect_type", "seed", "phase", "alpha", "n_good", "n_defect",
                    "mean_good", "std_good", "mean_defect", "std_defect", "median_defect",
                    "mean_gap", "mean_z", "d_prime", "image_auroc"]
    with open(out_dir / "group_level.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=group_fields)
        w.writeheader()
        w.writerows(group_rows)

    config = {
        "experiment": "Experiment 1I Spatial-Statistics Control",
        "category": category, "seed": seed, "phase": phase, "alphas": alphas,
        "defect_types": defect_types,
        "backbone": "wide_resnet50_2", "layers": ["layer2", "layer3"],
        "coreset_sampling_ratio": 0.1, "num_neighbors": 9,
        "image_size": e1b.IMAGE_SIZE, "val_size": e1b.VAL_SIZE,
        "train_size": len(train_ids), "n_test_good": len(good_paths),
        "IN_affine": False,
        "intervention": "layer2 matched256 (stride-2 sampled statistics applied to full 32x32)",
        "phase": phase,
        "z_normalization": "contemporaneous test/good (same seed+phase+alpha), ddof=1",
    }
    with open(out_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    resource_fields = ["category", "seed", "phase", "alpha", "runtime_seconds",
                       "train_good_count", "peak_gpu_memory_allocated_mb", "status"]
    with open(out_dir / "resource.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=resource_fields)
        w.writeheader()
        for r in resource_rows:
            r = {k: r.get(k, "") for k in resource_fields}
            w.writerow(r)

    print(f"[done] {category} seed={seed} phase={phase}: sample={len(sample_rows)} group={len(group_rows)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Experiment 1I matched256 runner")
    parser.add_argument("--category", type=str, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--phase", type=str, required=True, choices=PHASES)
    parser.add_argument("--alphas", type=str, default="0 1")
    parser.add_argument("--out-root", type=str, default=None)
    args = parser.parse_args()

    alphas = [float(x) for x in args.alphas.split() if x.strip()]
    out_root = Path(args.out_root) if args.out_root else PROJECT_ROOT / "results" / "experiment_1i"
    out_dir = out_root / args.category / f"seed_{args.seed}" / f"phase_{args.phase}"

    run_condition(category=args.category, seed=args.seed, phase=args.phase,
                  alphas=alphas, out_dir=out_dir)


if __name__ == "__main__":
    main()
