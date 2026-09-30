"""F_alpha PatchCore 实验：可控的 instance-normalization 混合。

================================================================================
研究问题（待验证假设）
================================================================================
在工业异常检测中，随着模型对 style / illumination-sensitive feature 的抑制
程度增加，不同 defect type 的异常检测性能是否表现出不同的变化规律？

实验控制变量 alpha 定义为：

    F_alpha = (1 - alpha) * F + alpha * F_IN

其中：
  - F    是 PatchCore 从 backbone 提取并 concat 后的原始 multi-scale embedding；
  - F_IN = InstanceNorm(F) 是 F 的 instance-normalized 版本；
  - alpha = 0 表示完全使用原始 feature（等价于原始 PatchCore baseline）；
  - alpha = 1 表示完全使用 instance-normalized feature。

重要说明（与研究假设一致）：
  alpha 不表示「去除了百分之多少的真实物理光照」，而仅作为
  style / illumination-sensitive feature suppression strength 的实验控制变量。

================================================================================
技术设计要点（基于 Anomalib 2.6.2 的 PatchCore 实现）
================================================================================
1. feature 提取层：
   PatchcoreModel 通过 TimmFeatureExtractor 从 ``wide_resnet50_2`` 的
   ``layer2`` / ``layer3`` 提取 feature map，再经 AvgPool2d(3,1,1) 局部聚合，
   最后在 generate_embedding() 中把 layer3 上采样到 layer2 分辨率后沿 channel
   concat，得到 embedding F，shape 为 (B, 1536, 32, 32)（输入 256x256 时）。

2. Instance Normalization 插入位置：
   在 ``generate_embedding`` 完成 concat 之后、``reshape_embedding`` 之前，
   对整张 concat 后的 embedding 做 InstanceNorm（affine=False，逐样本逐通道
   在空间维度 (H, W) 上归一化）。

3. 为什么这个位置是正确且最小的：
   - F 是真正进入 memory bank / nearest-neighbor 的 feature，直接对它做 IN
     最贴合「F → F_IN」的定义；
   - train(fit) 与 test 都走同一个 forward()，因此两者经历完全一致的
     feature transformation，memory bank 与 query 处在同一特征空间；
   - InstanceNorm 是逐样本、无参数的确定性变换，天然满足 train/test 一致；
   - affine=False 保证 IN 纯粹是归一化，不引入任何可学习参数。

4. 不会破坏 pipeline：
   - 归一化只改变 feature 的 scale/分布，PatchCore 后续的
     reshape → memory bank → k-center-greedy → nearest-neighbor(欧氏距离)
     全部照常工作；memory bank 与 query 被同等变换，相对距离关系一致；
   - 阈值由 PostProcessor 在 validation 阶段自适应计算，会随 anomaly score
     的 scale 变化自动调整，无需人工干预。

5. 已知局限（记录在案，供后续分析）：
   InstanceNorm 不仅移除 style/illumination 信息，还会把 feature 缩放到单位
   方差。因此 F_alpha 在增大 alpha 时，除了抑制 style/illumination，也伴随
   feature 整体 scale 的下降。这是一个需要在下阶段正式分析时注意的混淆因素。

================================================================================
可重复性
================================================================================
- PatchCore 的 coreset 采样（KCenterGreedy）使用 ``torch.randint`` 选择初始点，
  且 SparseRandomProjection 含随机投影，因此必须固定随机种子。
- 本模块在 run_experiment 中固定 python/numpy/torch(CPU+CUDA) 随机种子。
================================================================================
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from anomalib.data import MVTecAD
from anomalib.engine import Engine
from anomalib.metrics import AUPR, AUROC, F1Score
from anomalib.metrics.evaluator import Evaluator
from anomalib.models import Patchcore
from anomalib.models.image.patchcore.torch_model import PatchcoreModel

# 归档实验目录与共享数据所在的项目根目录
EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DATA_ROOT = PROJECT_ROOT / "data" / "mvtec_ad"
DEFAULT_RESULTS_ROOT = EXPERIMENT_ROOT / "reruns" / "falpha_experiment"

# 候选 alpha（实验控制变量全集）
CANDIDATE_ALPHAS = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]


# ---------------------------------------------------------------------------
# 模型定义
# ---------------------------------------------------------------------------
class FAlphaPatchcoreModel(PatchcoreModel):
    """PatchcoreModel + 可控 instance-normalization 混合。

    通过覆写 ``generate_embedding``，在 concat 后的 multi-scale embedding ``F``
    上施加：

        F_alpha = (1 - alpha) * F + alpha * F_IN

    ``alpha == 0`` 时完全跳过 IN，等价于原始 PatchCore（bit-wise 一致）。
    """

    def __init__(self, *args, alpha: float = 0.0, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.alpha = float(alpha)

    def generate_embedding(self, features: dict[str, torch.Tensor]) -> torch.Tensor:
        embedding = super().generate_embedding(features)
        if self.alpha > 0.0:
            f_in = F.instance_norm(embedding)
            embedding = (1.0 - self.alpha) * embedding + self.alpha * f_in
        return embedding


class FAlphaPatchcore(Patchcore):
    """Lightning 版 PatchCore，暴露 ``alpha`` 参数并额外记录 AP(AUPR) 指标。"""

    def __init__(self, *args, alpha: float = 0.0, **kwargs) -> None:
        self.alpha = float(alpha)
        super().__init__(*args, **kwargs)

        # 用带 alpha 的 torch 模型替换默认 PatchcoreModel，并保持精度一致。
        dtype = next(self.model.parameters()).dtype
        self.model = FAlphaPatchcoreModel(
            backbone=self.model.backbone,
            layers=self.model.layers,
            pre_trained=True,
            num_neighbors=self.model.num_neighbors,
            alpha=self.alpha,
        ).to(dtype=dtype)

    @staticmethod
    def configure_evaluator() -> Evaluator:
        """默认指标 + 图像/像素级 AP（AUPR）。"""
        image_auroc = AUROC(fields=["pred_score", "gt_label"], prefix="image_")
        image_f1 = F1Score(fields=["pred_label", "gt_label"], prefix="image_")
        pixel_auroc = AUROC(fields=["anomaly_map", "gt_mask"], prefix="pixel_", strict=False)
        pixel_f1 = F1Score(fields=["pred_mask", "gt_mask"], prefix="pixel_", strict=False)
        image_ap = AUPR(fields=["pred_score", "gt_label"], prefix="image_")
        pixel_ap = AUPR(fields=["anomaly_map", "gt_mask"], prefix="pixel_", strict=False)
        return Evaluator(
            test_metrics=[image_auroc, image_f1, pixel_auroc, pixel_f1, image_ap, pixel_ap],
        )


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------
class _LimitedDataset:
    """限制长度的 dataset 包装器，其余属性（如 collate_fn）透传给底层 dataset。"""

    def __init__(self, dataset, n: int) -> None:
        self._dataset = dataset
        self._n = n

    def __len__(self) -> int:
        return self._n

    def __getitem__(self, idx: int):
        if idx >= self._n:
            raise IndexError(idx)
        return self._dataset[idx]

    def __getattr__(self, name: str):
        return getattr(self._dataset, name)


class LimitedMVTecAD(MVTecAD):
    """限制训练集规模的 MVTecAD（仅用于低显存 smoke test / 调试）。

    覆写 ``_setup``，在父类创建好 ``train_data`` 后把它截断为前 ``limit_train`` 张。
    由于 ``AnomalibDataModule.setup`` 可能被 Trainer 重复调用并重建 ``train_data``，
    在 ``_setup`` 内限制可保证每次重建后都保持一致。
    """

    def __init__(self, *args, limit_train: int | None = None, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.limit_train = limit_train

    def _setup(self, _stage: str | None = None) -> None:
        super()._setup(_stage)
        if self.limit_train is not None and hasattr(self, "train_data"):
            n = min(self.limit_train, len(self.train_data))
            self.train_data = _LimitedDataset(self.train_data, n)


def _tensor_to_float(value: Any) -> Any:
    """把 tensor 转成 python float/int，便于 JSON 序列化。"""
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return value.item()
        return value.tolist()
    if isinstance(value, (list, tuple)):
        return [_tensor_to_float(v) for v in value]
    if isinstance(value, dict):
        return {k: _tensor_to_float(v) for k, v in value.items()}
    return value


def _defect_type_from_path(image_path: str | Path) -> str:
    """从 MVTec 的图片路径推断 defect_type（good / broken_large / ...）。"""
    return Path(image_path).parent.name


def _gt_label_from_path(image_path: str | Path) -> int:
    """MVTec AD 的 ground-truth label：``good`` -> 0(正常)，其余 -> 1(异常)。"""
    defect = _defect_type_from_path(image_path).lower()
    return 0 if defect in {"good", "normal"} else 1


# ---------------------------------------------------------------------------
# 实验主函数
# ---------------------------------------------------------------------------
def run_experiment(
    category: str = "bottle",
    alpha: float = 0.0,
    root: str | Path = DEFAULT_DATA_ROOT,
    coreset_sampling_ratio: float = 0.1,
    num_neighbors: int = 9,
    backbone: str = "wide_resnet50_2",
    layers: list[str] | None = None,
    train_batch_size: int = 16,
    eval_batch_size: int = 16,
    num_workers: int = 0,
    seed: int = 0,
    output_dir: str | Path | None = None,
    save_predictions: bool = True,
    limit_train: int | None = None,
) -> dict:
    """运行一次 F_alpha PatchCore 实验（fit + test，可选 per-image 预测保存）。

    返回包含参数与指标的结果字典；若提供 ``output_dir`` 则额外落盘：
      - metrics.json      聚合指标 + 关键参数
      - predictions.jsonl 每张测试图的 per-image 记录
    """
    layers = layers or ["layer2", "layer3"]

    # ---- 固定随机种子（KCenterGreedy 用 torch.randint，SparseRandomProjection 含随机投影）----
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # ---- 数据 ----
    datamodule_kwargs = dict(
        root=str(root),
        category=category,
        train_batch_size=train_batch_size,
        eval_batch_size=eval_batch_size,
        num_workers=num_workers,
        seed=seed,
    )
    if limit_train is not None:
        datamodule_kwargs["limit_train"] = limit_train
        datamodule = LimitedMVTecAD(**datamodule_kwargs)
    else:
        datamodule = MVTecAD(**datamodule_kwargs)
    datamodule.setup()

    # ---- 模型 ----
    model = FAlphaPatchcore(
        backbone=backbone,
        layers=layers,
        coreset_sampling_ratio=coreset_sampling_ratio,
        num_neighbors=num_neighbors,
        alpha=alpha,
        visualizer=False,
    )

    # ---- engine（default_root_dir 指向实验输出目录，避免污染 notebooks/results）----
    out_dir = Path(output_dir) if output_dir is not None else None
    engine = Engine(
        enable_progress_bar=False,
        logger=False,
        barebones=True,
        default_root_dir=str(out_dir) if out_dir is not None else "results",
    )

    # ---- fit + test ----
    engine.fit(model=model, datamodule=datamodule)
    test_results = engine.test(model=model, datamodule=datamodule)

    metrics: dict[str, Any] = {}
    if test_results:
        first = test_results[0]
        for k, v in first.items():
            metrics[k] = float(v) if hasattr(v, "item") else _tensor_to_float(v)

    result: dict[str, Any] = {
        "dataset": "MVTecAD",
        "category": category,
        "alpha": alpha,
        "backbone": backbone,
        "layers": list(layers),
        "coreset_sampling_ratio": coreset_sampling_ratio,
        "num_neighbors": num_neighbors,
        "train_batch_size": train_batch_size,
        "eval_batch_size": eval_batch_size,
        "seed": seed,
        "limit_train": limit_train,
        "num_train": len(datamodule.train_data),
        "num_test": len(datamodule.test_data),
        "metrics": metrics,
    }

    # ---- 可选：保存 per-image 预测记录 ----
    records: list[dict[str, Any]] = []
    if save_predictions:
        predictions = engine.predict(model=model, datamodule=datamodule)
        for batch in predictions:
            for i in range(len(batch.image_path)):
                path = Path(str(batch.image_path[i]))
                defect_type = _defect_type_from_path(path)
                # 优先取 batch 自带的 gt_label，缺失时按 MVTec 目录规则推断
                gt_label = None
                if getattr(batch, "gt_label", None) is not None:
                    gt_label = int(batch.gt_label[i].item())
                else:
                    gt_label = _gt_label_from_path(path)

                record: dict[str, Any] = {
                    "dataset": "MVTecAD",
                    "category": category,
                    "defect_type": defect_type,
                    "alpha": alpha,
                    "image_path": str(path),
                    "gt_label": gt_label,
                    "anomaly_score": float(batch.pred_score[i].item()),
                }
                # 尽量附上 pred_label / 像素级统计，字段缺失时不伪造
                if getattr(batch, "pred_label", None) is not None:
                    record["pred_label"] = int(batch.pred_label[i].item())
                if getattr(batch, "anomaly_map", None) is not None:
                    am = batch.anomaly_map[i]
                    record["anomaly_map_mean"] = float(am.mean().item())
                    record["anomaly_map_max"] = float(am.max().item())
                records.append(record)

    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        if records:
            with open(out_dir / "predictions.jsonl", "w", encoding="utf-8") as f:
                for r in records:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")

    result["num_records"] = len(records)
    return result


# ---------------------------------------------------------------------------
# CLI 入口
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="F_alpha PatchCore 实验")
    parser.add_argument("--category", default="bottle")
    parser.add_argument("--alpha", type=float, default=0.0)
    parser.add_argument("--root", default=str(DEFAULT_DATA_ROOT))
    parser.add_argument("--coreset-sampling-ratio", type=float, default=0.1)
    parser.add_argument("--num-neighbors", type=int, default=9)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--no-save-predictions", action="store_true")
    parser.add_argument("--limit-train", type=int, default=None)
    args = parser.parse_args()

    out_dir = args.output_dir or (DEFAULT_RESULTS_ROOT / args.category / f"alpha_{args.alpha}")
    result = run_experiment(
        category=args.category,
        alpha=args.alpha,
        root=args.root,
        coreset_sampling_ratio=args.coreset_sampling_ratio,
        num_neighbors=args.num_neighbors,
        seed=args.seed,
        output_dir=out_dir,
        save_predictions=not args.no_save_predictions,
        limit_train=args.limit_train,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
