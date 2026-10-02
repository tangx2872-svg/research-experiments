"""Experiment 1B：Defect-Specific α Sensitivity Screening（正式 Mini Experiment）。

研究问题（待验证假设，非结论）：
  当 PatchCore 内部逐渐增强 InstanceNorm 统计干预（α 增大）时，
  不同 defect type（broken_large / broken_small / contamination）的
  异常检测能力是否表现出不同的变化趋势？

方法（复用归档的 F_alpha PatchCore，一行未改）：
  F_alpha = (1-α) F + α·IN(F)，IN 为 InstanceNorm(affine=False)。
  α=0 等价原始 PatchCore baseline。

与 Experiment 1 的关键区别：
  本实验【不做任何 synthetic illumination perturbation】，
  只研究内部 representation probe（α）对不同 defect type 的影响。

核心协议（预注册，不可静默修改）：
  1. 独立 validation split：从 train/good(209) 中固定 seed 划出 VAL_SIZE=20 张，
     剩余 189 张进入 memory bank。validation 只用于定阈值，不进 bank，不参与训练。
  2. 每个 α 独立 fit：用同一 189 张 train/good 建对应 α 的 memory bank。
  3. 阈值规则 tau_alpha：每 α 下取该 α 的 validation normal 分数的 max（最保守，
     验证集 FPR=0）。三类 defect 共享同一个 tau_alpha。
  4. 不用 test 数据、不用 defect 标签定阈值。

输出（results/experiment_1b/）：
  config/     experiment_config.json + validation_split.csv
  raw/        all_sample_scores.csv（每图 × 每 α 的 raw score + anomaly_map 统计）
  summary/    defect_wise_metrics.csv + area_analysis.csv
  figures/    auroc/recall/fpr/score_distribution/trajectories/area 图
  heatmaps/   代表性样本五档 alpha 的 anomaly map
  logs/       每 α 运行日志

运行方式（项目根目录，industrial-ad 环境）：
  python scripts/experiment1b_defect_sensitivity.py --smoke   # smoke test（少量样本）
  python scripts/experiment1b_defect_sensitivity.py           # 全量五档
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
from PIL import Image
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
# 结果根目录：默认 experiment_1b；Experiment 1C 会通过 --results-root 覆盖为按 seed 分目录。
RESULTS_ROOT = PROJECT_ROOT / "results" / "experiment_1b"

ALPHAS = [0.0, 0.25, 0.5, 0.75, 1.0]
# 默认 category（历史 1B/1C 均用 bottle）。Experiment 1E 会通过 --category 传入
# 其他 category，defect types 由 test/ 目录自动发现（见 discover_defect_types）。
DEFAULT_CATEGORY = "bottle"

SEED = 0
VAL_SIZE = 20            # 从 209 train/good 划出的 validation 数
TRAIN_SIZE = 189         # 进入 memory bank 的正常图数（209 - 20）

# 与 Patchcore.configure_pre_processor 一致的预处理参数
IMAGE_SIZE = 256
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# 阈值规则：validation normal 分数的 max（最保守，验证集 FPR=0）
THRESHOLD_RULE = "max_validation_score"


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------
def discover_defect_types(category: str) -> list[str]:
    """自动扫描 <DATA_ROOT>/<category>/test/ 下的 defect type 目录。

    排除 "good"（正常类），其余按字典序固定排序，保证不同运行可复现。
    仅返回实际存在的目录（含 *.png），不硬编码任何 category 的缺陷类型。
    若 category/test 目录不存在（数据未下载），返回空列表（由调用方处理）。
    """
    test_root = DATA_ROOT / category / "test"
    if not test_root.is_dir():
        return []
    types = sorted(
        d.name for d in test_root.iterdir()
        if d.is_dir() and d.name != "good" and any(d.glob("*.png"))
    )
    return types


def make_validation_split(category: str, seed: int = SEED, val_size: int = VAL_SIZE) -> tuple[list[str], list[str]]:
    """从 <category>/train/good 固定 seed 划分 (validation_ids, train_ids)。

    返回文件名列表（如 '000.png'）。validation 不进 memory bank。
    """
    train_root = DATA_ROOT / category / "train" / "good"
    all_names = sorted(p.name for p in train_root.glob("*.png"))
    rng = random.Random(seed)
    shuffled = all_names[:]
    rng.shuffle(shuffled)
    val_ids = sorted(shuffled[:val_size])
    train_ids = sorted(shuffled[val_size:])
    return val_ids, train_ids


def load_image_as_tensor(path: Path) -> torch.Tensor:
    img = Image.open(path).convert("RGB")
    return TF.to_tensor(img)  # [0,1] (C,H,W)


def preprocess_for_model(img: torch.Tensor, device: torch.device) -> torch.Tensor:
    img = TF.resize(img, [IMAGE_SIZE, IMAGE_SIZE], antialias=True)
    img = TF.normalize(img, mean=IMAGENET_MEAN, std=IMAGENET_STD)
    return img.to(device)


def load_mask(path: Path) -> np.ndarray:
    """加载 GT mask，返回 (H, W) 0/1 数组。"""
    mask = Image.open(path).convert("L")
    return (np.array(mask) > 127).astype(np.uint8)


def defect_area_ratio(mask_path: Path) -> float:
    m = load_mask(mask_path)
    return float(m.sum()) / float(m.size)


# ---------------------------------------------------------------------------
# 模型 fit（每 α 独立，用固定的 train_ids 建 bank）
# ---------------------------------------------------------------------------
def fit_model(alpha: float, category: str, train_ids: list[str], seed: int = SEED):
    """用 train_ids 建 memory bank 的 FAlphaPatchcore。返回 (lightning_model, torch_model)。"""
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

    # ---- 用 train_ids 过滤 train_data 的 _samples DataFrame（最干净的子集方式）----
    train_root = DATA_ROOT / category / "train" / "good"
    keep_names = {n for n in train_ids}
    td = datamodule.train_data
    df = td._samples
    # image_path 列是完整图片路径，用文件名匹配
    df = df[df["image_path"].apply(lambda p: str(Path(str(p)).name) in keep_names)].reset_index(drop=True)
    td._samples = df
    # 同步更新样本数量相关属性（若存在）
    if hasattr(td, "_num_samples"):
        td._num_samples = len(df)

    print(f"  [fit] train_data 子集大小 = {len(datamodule.train_data)}（应为 {len(train_ids)}）")

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

    torch_model: FAlphaPatchcoreModel = model.model
    torch_model.eval()

    return model, torch_model, datamodule


def _move_model_to_device(torch_model: FAlphaPatchcoreModel, device: torch.device) -> None:
    torch_model.to(device)
    if hasattr(torch_model, "memory_bank") and isinstance(torch_model.memory_bank, torch.Tensor):
        torch_model.memory_bank = torch_model.memory_bank.to(device)


# ---------------------------------------------------------------------------
# 推理：对单张图得到 (score, anomaly_map)
# ---------------------------------------------------------------------------
def predict_one(torch_model, img_tensor, device) -> tuple[float, np.ndarray]:
    inp = preprocess_for_model(img_tensor, device)
    with torch.no_grad():
        out = torch_model(inp.unsqueeze(0))
    score = float(out.pred_score.item())
    amap = out.anomaly_map[0].detach().cpu().numpy() if hasattr(out, "anomaly_map") else None
    return score, amap


# ---------------------------------------------------------------------------
# 主实验
# ---------------------------------------------------------------------------
def run_screening(
    smoke_limit: int | None = None,
    seed: int = SEED,
    results_root: Path | None = None,
    category: str = DEFAULT_CATEGORY,
) -> None:
    """运行五档 α 筛查。results_root 为空则用模块默认 RESULTS_ROOT；
    Experiment 1C 传入按 seed 分目录的 results_root，实现多 seed 隔离。
    category 指定 MVTec AD 类别，defect types 由 test/ 目录自动发现。"""
    global RESULTS_ROOT, CONFIG_DIR, RAW_DIR, SUMMARY_DIR, FIGURES_DIR, HEATMAPS_DIR, LOGS_DIR
    if results_root is not None:
        RESULTS_ROOT = Path(results_root)
    CONFIG_DIR = RESULTS_ROOT / "config"
    RAW_DIR = RESULTS_ROOT / "raw"
    SUMMARY_DIR = RESULTS_ROOT / "summary"
    FIGURES_DIR = RESULTS_ROOT / "figures"
    HEATMAPS_DIR = RESULTS_ROOT / "heatmaps"
    LOGS_DIR = RESULTS_ROOT / "logs"

    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    for d in [CONFIG_DIR, RAW_DIR, SUMMARY_DIR, FIGURES_DIR, HEATMAPS_DIR, LOGS_DIR]:
        d.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---- 前置守卫：category 数据目录必须存在（数据未下载时给出清晰报错）----
    if not (DATA_ROOT / category).is_dir():
        raise FileNotFoundError(
            f"category '{category}' 数据目录不存在: {DATA_ROOT / category}。"
            f"请先通过 MVTec 官方渠道下载该 category 并解压到 {DATA_ROOT}/。"
        )

    # ---- defect types：自动发现（排除 good，固定排序）----
    defect_types = discover_defect_types(category)
    if not defect_types:
        raise RuntimeError(f"category '{category}' 的 test/ 目录下未发现任何 defect type")

    # ---- validation split ----
    val_ids, train_ids = make_validation_split(category=category, seed=seed)
    print(f"[split] validation={len(val_ids)} train={len(train_ids)} (seed={seed}, category={category})")

    # 保存 validation split
    with open(CONFIG_DIR / "validation_split.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["split", "filename"])
        for n in train_ids:
            w.writerow(["train", n])
        for n in val_ids:
            w.writerow(["validation", n])

    # ---- 收集路径 ----
    val_paths = [DATA_ROOT / category / "train" / "good" / n for n in val_ids]
    test_root = DATA_ROOT / category / "test"
    test_paths: dict[str, list[Path]] = {}
    for d in ["good"] + defect_types:
        paths = sorted((test_root / d).glob("*.png"))
        if smoke_limit is not None:
            paths = paths[:smoke_limit]
        test_paths[d] = paths

    # ---- 预计算 defect area（用全量 GT mask）----
    area_by_sample: dict[str, float] = {}
    for dt in defect_types:
        for p in sorted((test_root / dt).glob("*.png")):
            mask_path = DATA_ROOT / category / "ground_truth" / dt / f"{p.stem}_mask.png"
            area_by_sample[str(p)] = defect_area_ratio(mask_path) if mask_path.exists() else float("nan")

    raw_rows: list[dict] = []
    threshold_by_alpha: dict[float, float] = {}
    # 保存 anomaly map（用于 heatmap）：{alpha: {sample_path: amap}}
    amaps_by_alpha: dict[float, dict[str, np.ndarray]] = {}

    for alpha in ALPHAS:
        print(f"\n{'='*70}\n[alpha={alpha:g}] fit + predict\n{'='*70}")
        lightning_model, torch_model, datamodule = fit_model(alpha, category, train_ids, seed=seed)
        _move_model_to_device(torch_model, device)

        amaps_by_alpha[alpha] = {}

        # ---- validation normal → 定阈值 ----
        val_scores: list[float] = []
        for p in val_paths:
            img = load_image_as_tensor(p)
            score, amap = predict_one(torch_model, img, device)
            val_scores.append(score)
            raw_rows.append({
                "split": "validation", "sample_id": p.stem, "image_path": str(p),
                "defect_type": "good", "is_anomaly": 0, "alpha": alpha,
                "anomaly_score": score, "gt_label": 0,
            })
        tau = float(max(val_scores))
        threshold_by_alpha[alpha] = tau
        print(f"  [threshold] α={alpha:g} validation max score = {tau:.6f}")

        # ---- test/good ----
        for p in test_paths["good"]:
            img = load_image_as_tensor(p)
            score, amap = predict_one(torch_model, img, device)
            raw_rows.append({
                "split": "test", "sample_id": p.stem, "image_path": str(p),
                "defect_type": "good", "is_anomaly": 0, "alpha": alpha,
                "anomaly_score": score, "gt_label": 0,
            })
            amaps_by_alpha[alpha][str(p)] = amap

        # ---- test/defect ----
        for dt in defect_types:
            for p in test_paths[dt]:
                img = load_image_as_tensor(p)
                score, amap = predict_one(torch_model, img, device)
                raw_rows.append({
                    "split": "test", "sample_id": p.stem, "image_path": str(p),
                    "defect_type": dt, "is_anomaly": 1, "alpha": alpha,
                    "anomaly_score": score, "gt_label": 1,
                })
                amaps_by_alpha[alpha][str(p)] = amap

        # ---- 释放显存 ----
        del lightning_model, torch_model, datamodule
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ---- 保存 raw CSV ----
    raw_csv = RAW_DIR / "all_sample_scores.csv"
    fieldnames = ["split", "sample_id", "image_path", "defect_type", "is_anomaly",
                  "alpha", "anomaly_score", "gt_label"]
    with open(raw_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(raw_rows)

    # ---- 保存 config ----
    config = {
        "experiment": "Experiment 1B: Defect-Specific α Sensitivity Screening",
        "dataset": f"MVTecAD {category}",
        "category": category,
        "alphas": ALPHAS,
        "defect_types": defect_types,
        "backbone": "wide_resnet50_2",
        "layers": ["layer2", "layer3"],
        "coreset_sampling_ratio": 0.1,
        "num_neighbors": 9,
        "input_size": IMAGE_SIZE,
        "batch_size": 16,
        "seed": seed,
        "val_size": VAL_SIZE,
        "train_size": len(train_ids),
        "threshold_rule": THRESHOLD_RULE,
        "num_test": {d: len(test_paths[d]) for d in ["good"] + defect_types},
        "IN_position": "generate_embedding concat 后、reshape 前，affine=False",
        "alpha0_shortcut": "alpha==0 直接返回原始 feature，跳过 IN",
    }
    with open(CONFIG_DIR / "experiment_config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    # ---- 保存 threshold ----
    with open(SUMMARY_DIR / "thresholds.json", "w", encoding="utf-8") as f:
        json.dump({str(k): v for k, v in threshold_by_alpha.items()}, f, indent=2, ensure_ascii=False)

    # ---- 保存 area analysis ----
    area_csv = SUMMARY_DIR / "area_analysis.csv"
    with open(area_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["image_path", "defect_type", "defect_area_ratio"])
        for p, r in area_by_sample.items():
            w.writerow([p, Path(p).parent.name, r])

    # ---- 保存 anomaly maps（numpy）供后续 heatmap ----
    np.savez_compressed(
        RAW_DIR / "anomaly_maps.npz",
        **{f"{a:g}|{p}": amap for a, d in amaps_by_alpha.items() for p, amap in d.items() if amap is not None},
    )

    print(f"\n[done] raw 记录数 = {len(raw_rows)}")
    print(f"[done] 输出目录 = {RESULTS_ROOT}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Experiment 1B defect-specific α sensitivity screening")
    parser.add_argument("--smoke", action="store_true", help="smoke test：少量样本")
    parser.add_argument("--smoke-limit", type=int, default=5, help="smoke test 每类样本数")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--category", type=str, default=DEFAULT_CATEGORY, help="MVTec AD category")
    parser.add_argument(
        "--results-root",
        type=str,
        default=None,
        help="结果输出根目录（默认 results/experiment_1b）。Experiment 1C 按 seed 分目录时传入。",
    )
    args = parser.parse_args()

    results_root = Path(args.results_root) if args.results_root else None
    if args.smoke:
        print(f"[smoke test] 每类样本数 = {args.smoke_limit}")
        run_screening(smoke_limit=args.smoke_limit, seed=args.seed, results_root=results_root, category=args.category)
    else:
        run_screening(smoke_limit=None, seed=args.seed, results_root=results_root, category=args.category)


if __name__ == "__main__":
    main()
