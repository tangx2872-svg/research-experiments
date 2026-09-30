"""F_alpha PatchCore 最小 smoke test。

仅验证：数据可读、模型可运行、feature transformation 生效、
PatchCore pipeline 未被破坏、结果能正常输出。

不进行「全部类别 x 6 个 alpha」的大规模实验。

运行方式（在项目根目录，使用 industrial-ad 环境）：

    python experiments/2026-09-30_illumination_sensitivity_exploration/falpha_patchcore/smoke_test.py

或指定环境解释器：

    D:/miniconda/envs/industrial-ad/python.exe experiments/2026-09-30_illumination_sensitivity_exploration/falpha_patchcore/smoke_test.py
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import numpy as np
import torch

# 从同目录导入归档模型，数据路径由模型定位
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from anomalib.models.image.patchcore.torch_model import PatchcoreModel  # noqa: E402

from falpha_patchcore import (  # noqa: E402
    FAlphaPatchcoreModel,
    run_experiment,
)

RESULTS_ROOT = PROJECT_ROOT.parent / "reruns" / "smoke_test"


def check_environment() -> dict:
    import torchvision

    return {
        "python": sys.version.split()[0],
        "pytorch": torch.__version__,
        "torchvision": torchvision.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_version": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }


def check_falpha_transform() -> bool:
    """数值校验：alpha=0 等价 baseline，alpha=1 的 feature 确实被 IN 改变。

    PatchCore 在 ``training=True`` 时 forward 返回 embedding；在 eval 且
    memory bank 为空时会走最近邻分支并报错，因此这里必须用 ``.train()``。
    """
    torch.manual_seed(0)
    x = torch.randn(2, 3, 256, 256)

    base = FAlphaPatchcoreModel(
        backbone="wide_resnet50_2", layers=["layer2", "layer3"], alpha=0.0
    ).train()
    ref = PatchcoreModel(
        backbone="wide_resnet50_2", layers=["layer2", "layer3"]
    ).train()
    mixed = FAlphaPatchcoreModel(
        backbone="wide_resnet50_2", layers=["layer2", "layer3"], alpha=1.0
    ).train()

    with torch.no_grad():
        f_base = base(x)
        f_ref = ref(x)
        f_mixed = mixed(x)

    # 训练态返回 embedding：shape (B*H*W, C) = (2048, 1536)
    same_shape = f_base.shape == f_mixed.shape == f_ref.shape
    # alpha=0 必须与原版 PatchcoreModel 完全一致（bit-wise）
    baseline_match = torch.equal(f_base, f_ref)
    # alpha=1 时，concat 后每个通道在空间维上应近似零均值、单位方差
    emb_mixed = f_mixed.view(2, 32, 32, -1).permute(0, 3, 1, 2)  # (B, C, H, W)
    ch_mean = emb_mixed.mean(dim=(2, 3)).abs().max().item()
    ch_std = emb_mixed.std(dim=(2, 3)).mean().item()
    differ = not torch.allclose(f_base, f_mixed)

    print("  [check] embedding shape 一致:", same_shape, "->", tuple(f_base.shape))
    print("  [check] alpha=0 与原版 PatchCore 完全一致:", baseline_match)
    print(f"  [check] alpha=1 后 per-channel 空间均值 |max| = {ch_mean:.2e} (应≈0)")
    print(f"  [check] alpha=1 后 per-channel 空间标准差 mean = {ch_std:.4f} (应≈1)")
    print("  [check] alpha=0 与 alpha=1 的 feature 不同:", differ)

    return same_shape and baseline_match and differ and ch_mean < 1e-3 and abs(ch_std - 1.0) < 0.1


def main() -> None:
    print("=" * 70)
    print("F_alpha PatchCore — smoke test")
    print("=" * 70)

    # 1. 环境
    print("\n[1] 环境检查")
    env = check_environment()
    for k, v in env.items():
        print(f"  {k}: {v}")
    assert env["cuda_available"], "CUDA 不可用，请检查 GPU/驱动环境"

    # 2. feature transformation 数值校验
    print("\n[2] F_alpha feature transformation 数值校验")
    ok = check_falpha_transform()
    assert ok, "F_alpha 数值校验未通过"

    # 3. 小规模实验：bottle + alpha=0 + alpha=1.0
    #    使用 limit_train=64 限制训练集，规避当前 GPU 被图形应用争用导致的显存不足；
    #    仅用于验证 pipeline 可跑通，正式实验用全量（limit_train=None，需关闭图形应用）。
    SMOKE_TRAIN = 64
    print("\n[3] 运行 smoke test 实验（bottle, alpha=0.0 与 alpha=1.0）")
    print(f"    （训练集限制为 {SMOKE_TRAIN} 张正常图，仅验证 pipeline，非正式结果）")
    for alpha in (0.0, 1.0):
        out_dir = RESULTS_ROOT / "bottle" / f"alpha_{alpha:g}"
        print(f"\n  --- alpha = {alpha:g} ---")
        result = run_experiment(
            category="bottle",
            alpha=alpha,
            seed=0,
            output_dir=out_dir,
            save_predictions=True,
            limit_train=SMOKE_TRAIN,
        )
        metrics = result["metrics"]
        print(f"  训练图数: {result['num_train']}, 测试图数: {result['num_test']}")
        print(f"  保存记录数: {result['num_records']}")
        for k in sorted(metrics):
            print(f"  {k}: {metrics[k]:.4f}")

    print("\n" + "=" * 70)
    print("smoke test 完成。结果保存在:", RESULTS_ROOT)
    print("=" * 70)


if __name__ == "__main__":
    main()
