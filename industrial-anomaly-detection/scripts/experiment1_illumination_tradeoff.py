"""Experiment 1：Synthetic Illumination × α-IN Mechanism Screening（第一轮低成本筛查）。

研究问题（待验证假设，非结论）：
  随着 PatchCore 特征的 illumination/style invariance 增强（α 增大），
  正常产品在简单 photometric perturbation 下是否更不容易被误报；
  同时真实缺陷的异常响应是否下降，且这种下降是否与 defect type 有关。

重要边界（必须保留）：
  本实验仅使用 brightness / gamma 等 synthetic photometric perturbation，
  不能模拟真实工业光照（specular reflection / highlight / shadow /
  illumination direction 等）。因此任何结果只能表述为
  "Synthetic illumination screening suggests..."，不能表述为真实光照结论。

方法（复用归档的 F_alpha PatchCore）：
  F_alpha = (1-α) F + α·IN(F)，IN 为 InstanceNorm(affine=False)。
  α=0 等价原始 PatchCore baseline。

实验矩阵：
  alpha ∈ {0, 0.25, 0.5, 0.75, 1.0}

  Robustness side（仅正常 test 图，20 张 good）：
    original / brightness 0.7 / brightness 1.3 / gamma 0.7 / gamma 1.3

  Sensitivity side（仅原始真实 defect 图，不做人为光照）：
    broken_large(20) / broken_small(22) / contamination(21)

输出（results/experiment1_illumination_tradeoff/）：
  raw_results.csv     每图 × 每 α × 每 condition 的原始 pred_score
  summary_results.csv 分 defect_type / illumination condition 的汇总
  metrics.json        每 α 的 image/pixel 指标（用于 α=0 baseline 校验）
  figures/*.png       核心图

运行方式（项目根目录，industrial-ad 环境）：
  python scripts/experiment1_illumination_tradeoff.py --smoke   # 先跑 smoke test
  python scripts/experiment1_illumination_tradeoff.py           # 全量
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torchvision.transforms import functional as TF

# 复用归档实验中的 F_alpha PatchCore 实现
PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARCHIVE_FALPHA = (
    PROJECT_ROOT
    / "experiments"
    / "2026-09-30_illumination_sensitivity_exploration"
    / "falpha_patchcore"
)
if str(ARCHIVE_FALPHA) not in sys.path:
    sys.path.insert(0, str(ARCHIVE_FALPHA))

from anomalib.data import MVTecAD  # noqa: E402
from anomalib.engine import Engine  # noqa: E402

from falpha_patchcore import FAlphaPatchcore, FAlphaPatchcoreModel  # noqa: E402

# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------
DATA_ROOT = PROJECT_ROOT / "data" / "mvtec_ad"
RESULTS_ROOT = PROJECT_ROOT / "results" / "experiment1_illumination_tradeoff"
FIGURES_DIR = RESULTS_ROOT / "figures"
LOGS_DIR = RESULTS_ROOT / "logs"

ALPHAS = [0.0, 0.25, 0.5, 0.75, 1.0]

# illumination condition 定义：(名称, 类型, 参数)
# brightness: 像素值线性缩放；gamma: 像素值指数变换；original: 原图。
NORMAL_CONDITIONS = [
    ("original", "identity", 1.0),
    ("brightness", "brightness", 0.7),
    ("brightness", "brightness", 1.3),
    ("gamma", "gamma", 0.7),
    ("gamma", "gamma", 1.3),
]

DEFECT_TYPES = ["broken_large", "broken_small", "contamination"]

# 与 Patchcore.configure_pre_processor 一致的预处理参数
IMAGE_SIZE = 256
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# 检测阈值：取 α=0 正常图 original 分数的分位数（见 build_threshold）
THRESHOLD_QUANTILE = 1.0  # 用 max（最保守：所有正常图都判正常）


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------
def load_image_as_tensor(path: Path) -> torch.Tensor:
    """加载图片为 [0,1] 范围的 float tensor (C, H, W)，尺寸为原始尺寸。"""
    from PIL import Image

    img = Image.open(path).convert("RGB")
    t = TF.to_tensor(img)  # [0,1], (C,H,W)
    return t


def apply_photometric(
    img: torch.Tensor,
    illum_type: str,
    level: float,
) -> torch.Tensor:
    """在原始 [0,1] 像素空间施加 photometric perturbation。

    - brightness: img * level（线性缩放像素值，模拟曝光/亮度）
    - gamma:      img ** (1/level) 当 level>1 时提亮，level<1 时压暗
                  （与常见 gamma correction 约定一致：gamma<1 变亮、>1 变暗）
    - identity:   原图不变
    """
    img = img.clone()
    if illum_type == "brightness":
        img = img * level
    elif illum_type == "gamma":
        # level 为 gamma 系数：output = input^(1/gamma)
        img = img.pow(1.0 / level)
    elif illum_type == "identity":
        pass
    else:
        raise ValueError(f"未知 illumination type: {illum_type}")
    return img.clamp(0.0, 1.0)


def preprocess_for_model(img: torch.Tensor, device: torch.device) -> torch.Tensor:
    """对齐 Patchcore.configure_pre_processor：Resize(256, antialias) + Normalize(ImageNet)。"""
    img = TF.resize(img, [IMAGE_SIZE, IMAGE_SIZE], antialias=True)
    img = TF.normalize(img, mean=IMAGENET_MEAN, std=IMAGENET_STD)
    return img.to(device)


def defect_type_from_path(path: Path) -> str:
    return path.parent.name


# ---------------------------------------------------------------------------
# 模型 fit
# ---------------------------------------------------------------------------
def fit_model(alpha: float, seed: int = 0, limit_train: int | None = None):
    """Fit 一个 FAlphaPatchcore，返回 (lightning_model, torch_model, datamodule)。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    datamodule = MVTecAD(
        root=str(DATA_ROOT),
        category="bottle",
        train_batch_size=16,
        eval_batch_size=16,
        num_workers=0,
        seed=seed,
    )
    datamodule.setup()

    model = FAlphaPatchcore(
        backbone="wide_resnet50_2",
        layers=["layer2", "layer3"],
        coreset_sampling_ratio=0.1,
        num_neighbors=9,
        alpha=alpha,
        visualizer=False,
    )

    engine = Engine(
        enable_progress_bar=False,
        logger=False,
        barebones=True,
        default_root_dir=str(LOGS_DIR / f"fit_alpha_{alpha:g}"),
    )

    engine.fit(model=model, datamodule=datamodule)

    # 取 fit 好的 torch 模型（含 memory bank），置为 eval
    torch_model: FAlphaPatchcoreModel = model.model
    torch_model.eval()

    return model, torch_model, datamodule


def _move_model_to_device(torch_model: FAlphaPatchcoreModel, device: torch.device) -> None:
    """把 fit 后的 torch 模型移到目标设备（fit 结束 Lightning 会把它移回 CPU）。

    memory_bank 是普通 tensor 属性（非 registered buffer），``.to()`` 不会移动它，
    需显式处理。
    """
    torch_model.to(device)
    if hasattr(torch_model, "memory_bank") and isinstance(
        torch_model.memory_bank, torch.Tensor
    ):
        torch_model.memory_bank = torch_model.memory_bank.to(device)


def collect_test_paths(limit: int | None = None) -> dict[str, list[Path]]:
    """收集测试图路径，按 defect_type 分组。返回 {defect_type: [paths]}。"""
    test_root = DATA_ROOT / "bottle" / "test"
    groups: dict[str, list[Path]] = {}
    for d in ["good"] + DEFECT_TYPES:
        paths = sorted((test_root / d).glob("*.png"))
        if limit is not None:
            paths = paths[:limit]
        groups[d] = paths
    return groups


# ---------------------------------------------------------------------------
# 主实验
# ---------------------------------------------------------------------------
def run_screening(
    alphas: list[float],
    smoke_limit: int | None = None,
    seed: int = 0,
) -> None:
    """核心筛查：对每个 α fit 一次，然后对正常图(含扰动)和缺陷图做推理。"""
    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_paths = collect_test_paths(limit=smoke_limit)

    raw_rows: list[dict] = []
    metrics_per_alpha: dict[float, dict] = {}

    # 记录阈值（用 α=0 正常图 original 的 max score 作为保守阈值）
    threshold_by_alpha: dict[float, float] = {}

    for alpha in alphas:
        print(f"\n{'=' * 70}\n[alpha={alpha:g}] fit + predict\n{'=' * 70}")
        lightning_model, torch_model, datamodule = fit_model(alpha, seed=seed)

        # ---- 用 engine.test 得到每 α 的聚合指标（α=0 用于 baseline 校验）----
        # 注意：engine.test 结束会把模型移回 CPU，因此必须在推理前、engine.test 后再 move 到 device。
        try:
            test_results = engine_test(lightning_model, datamodule, alpha)
            metrics_per_alpha[alpha] = test_results
        except Exception as exc:  # noqa: BLE001
            print(f"  [warn] engine.test 失败（继续）：{exc}")
            metrics_per_alpha[alpha] = {"error": str(exc)}

        _move_model_to_device(torch_model, device)

        # ---- Robustness side：正常图 × perturbation ----
        normal_scores: dict[str, list[float]] = {}  # condition -> scores
        for p in test_paths["good"]:
            img = load_image_as_tensor(p)
            for cond_name, illum_type, level in NORMAL_CONDITIONS:
                perturbed = apply_photometric(img, illum_type, level)
                inp = preprocess_for_model(perturbed, device)
                with torch.no_grad():
                    out = torch_model(inp.unsqueeze(0))
                score = float(out.pred_score.item())
                raw_rows.append(
                    {
                        "image_path": str(p),
                        "sample_name": p.name,
                        "gt_type": "good",
                        "is_anomaly": 0,
                        "illumination_type": illum_type,
                        "illumination_level": level,
                        "alpha": alpha,
                        "pred_score": score,
                    }
                )
                normal_scores.setdefault(f"{cond_name}", []).append(score)

        # ---- Sensitivity side：真实 defect 图（原图，不做扰动）----
        for dt in DEFECT_TYPES:
            for p in test_paths[dt]:
                img = load_image_as_tensor(p)
                inp = preprocess_for_model(img, device)
                with torch.no_grad():
                    out = torch_model(inp.unsqueeze(0))
                score = float(out.pred_score.item())
                raw_rows.append(
                    {
                        "image_path": str(p),
                        "sample_name": p.name,
                        "gt_type": dt,
                        "is_anomaly": 1,
                        "illumination_type": "identity",
                        "illumination_level": 1.0,
                        "alpha": alpha,
                        "pred_score": score,
                    }
                )

        # ---- 阈值：α=0 正常图 original 的 max score ----
        if alpha == 0.0:
            orig_scores = normal_scores["original"]
            threshold_by_alpha[0.0] = float(max(orig_scores))
            print(f"  [threshold] α=0 normal original max score = {threshold_by_alpha[0.0]:.6f}")

        # ---- 释放显存 ----
        del lightning_model, torch_model, datamodule
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ---- 保存 raw CSV ----
    raw_csv = RESULTS_ROOT / "raw_results.csv"
    fieldnames = [
        "image_path", "sample_name", "gt_type", "is_anomaly",
        "illumination_type", "illumination_level", "alpha", "pred_score",
    ]
    with open(raw_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(raw_rows)

    # ---- 保存 metrics.json ----
    with open(RESULTS_ROOT / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics_per_alpha, f, indent=2, ensure_ascii=False, default=str)

    # ---- 保存 threshold ----
    with open(RESULTS_ROOT / "threshold.json", "w", encoding="utf-8") as f:
        json.dump(threshold_by_alpha, f, indent=2, ensure_ascii=False)

    print(f"\n[done] raw_results.csv 记录数 = {len(raw_rows)}")
    print(f"[done] 输出目录: {RESULTS_ROOT}")


def engine_test(lightning_model, datamodule, alpha: float) -> dict:
    """用 engine.test 得到聚合指标，返回 dict。"""
    engine = Engine(
        enable_progress_bar=False,
        logger=False,
        barebones=True,
        default_root_dir=str(LOGS_DIR / f"test_alpha_{alpha:g}"),
    )
    results = engine.test(model=lightning_model, datamodule=datamodule)
    metrics: dict = {}
    if results:
        first = results[0]
        for k, v in first.items():
            if hasattr(v, "item"):
                metrics[k] = float(v.item())
            elif isinstance(v, (int, float)):
                metrics[k] = float(v)
            else:
                metrics[k] = str(v)
    metrics["alpha"] = alpha
    return metrics


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Experiment 1 illumination × α-IN screening")
    parser.add_argument("--smoke", action="store_true", help="smoke test：少量样本")
    parser.add_argument("--smoke-limit", type=int, default=5, help="smoke test 每类样本数")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if args.smoke:
        print(f"[smoke test] 每类样本数 = {args.smoke_limit}")
        run_screening(alphas=[0.0, 0.5, 1.0], smoke_limit=args.smoke_limit, seed=args.seed)
    else:
        run_screening(alphas=ALPHAS, smoke_limit=None, seed=args.seed)


if __name__ == "__main__":
    main()
