"""Experiment 1I — Spatial-Statistics Control 的模型定义。

在 PatchCore 的 generate_embedding 内部，对 layer2 施加 **Matched-256 Statistics**：
只用 stride-2 采样的 256 个 spatial position 估计 InstanceNorm 的 μ/σ，但归一化
作用于完整 32×32 feature map（**绝不 resize**）。

与 1H 的 FAlphaLayerPatchcoreModel 区别仅在于：layer2 的 IN 统计量从 256 个
stride-2 采样位置估计（而非全部 1024 个），以匹配 layer3 的 16×16=256 统计支持。

phase ∈ {"00","01","10","11"}：
  phase_00: F[:, :, 0::2, 0::2]
  phase_01: F[:, :, 0::2, 1::2]
  phase_10: F[:, :, 1::2, 0::2]
  phase_11: F[:, :, 1::2, 1::2]

IN 定义与 1E/1H 一致：F_alpha = (1-α)F + α·IN(F)，IN 逐样本逐通道在采样位置上估计
μ/σ，affine=False（无参数）。α=0 直接返回原始 feature（跳过 IN）。

继承 1H 的 FAlphaLayerPatchcore（Lightning 包装复用 AP 指标与 model 替换逻辑），
只把 torch_model 换成本模块的 FAlphaMatched256Model。
"""

from __future__ import annotations

import torch

from anomalib.models.image.patchcore.torch_model import PatchcoreModel

from falpha_layer_patchcore import FAlphaLayerPatchcore  # noqa: E402  (Lightning 包装复用)

PHASES = ["00", "01", "10", "11"]
_PHASE_SLICE = {
    "00": (slice(None), slice(None), slice(0, None, 2), slice(0, None, 2)),
    "01": (slice(None), slice(None), slice(0, None, 2), slice(1, None, 2)),
    "10": (slice(None), slice(None), slice(1, None, 2), slice(0, None, 2)),
    "11": (slice(None), slice(None), slice(1, None, 2), slice(1, None, 2)),
}


def matched256_instance_norm(f: torch.Tensor, phase: str, eps: float = 1e-5) -> torch.Tensor:
    """用 stride-2 采样的 256 个 spatial position 估计 μ/σ，作用于完整 feature map。

    Args:
        f: (B, C, H, W)，layer2 的 32×32 feature。
        phase: "00"/"01"/"10"/"11"。
        eps: 与 torch F.instance_norm 默认一致（1e-5）。

    Returns:
        归一化后的完整 (B, C, H, W) tensor。
    """
    assert phase in _PHASE_SLICE, f"invalid phase: {phase}"
    s = _PHASE_SLICE[phase]
    subset = f[s]  # (B, C, H/2, W/2) = (B, C, 16, 16)
    mu = subset.mean(dim=(2, 3), keepdim=True)      # (B, C, 1, 1)
    var = subset.var(dim=(2, 3), keepdim=True, unbiased=False)  # 与 instance_norm 一致（biased）
    return (f - mu) / torch.sqrt(var + eps)


class FAlphaMatched256Model(PatchcoreModel):
    """PatchcoreModel + 可指定 phase 的 matched256 α-IN（只作用于 layer2）。

    覆写 generate_embedding：layer2 用 matched256 统计归一化，layer3 保持原始
    （本实验不干预 layer3）。α=0 直接返回原始 feature（与 1H layer2 α=0 bit-wise 等价）。
    """

    def __init__(self, *args, alpha: float = 0.0, phase: str = "00", **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.alpha = float(alpha)
        self.phase = phase
        assert phase in _PHASE_SLICE, f"invalid phase: {phase}"

    def _alpha_mix(self, f: torch.Tensor) -> torch.Tensor:
        """(1-α)F + α·IN_256(F)；α=0 直接返回原始。"""
        if self.alpha > 0.0:
            f_in = matched256_instance_norm(f, self.phase)
            return (1.0 - self.alpha) * f + self.alpha * f_in
        return f

    def generate_embedding(self, features: dict[str, torch.Tensor]) -> torch.Tensor:
        """layer2 用 matched256 α-IN，其余与 1H layer2 干预版 generate_embedding 一致。"""
        layer2 = features[self.layers[0]]
        layer2 = self._alpha_mix(layer2)

        processed = [layer2]
        for layer_name in self.layers[1:]:
            layer_embedding = features[layer_name]
            layer_embedding = torch.nn.functional.interpolate(
                layer_embedding, size=layer2.shape[-2:], mode="bilinear")
            processed.append(layer_embedding)

        return torch.cat(processed, 1)


class FAlphaMatched256Patchcore(FAlphaLayerPatchcore):
    """Lightning 版 PatchCore，额外暴露 phase，torch_model 用 FAlphaMatched256Model。"""

    def __init__(self, *args, alpha: float = 0.0, phase: str = "00", **kwargs) -> None:
        self.phase = phase
        super().__init__(*args, alpha=alpha, intervention_location="layer2", **kwargs)
        dtype = next(self.model.parameters()).dtype
        self.model = FAlphaMatched256Model(
            backbone=self.model.backbone,
            layers=self.model.layers,
            pre_trained=True,
            num_neighbors=self.model.num_neighbors,
            alpha=self.alpha,
            phase=self.phase,
        ).to(dtype=dtype)
