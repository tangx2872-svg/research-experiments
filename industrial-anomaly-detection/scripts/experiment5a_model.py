"""Experiment 5A — Geometry-Guided Layer-wise Normalization 的模型定义。

FAlphaDualLayerPatchcoreModel：PatchcoreModel + 每层独立 α-IN。

  layer2: F2' = (1−α₂)F2 + α₂·IN(F2)   （concat 前，512×32×32）
  layer3: F3' = (1−α₃)F3 + α₃·IN(F3)   （upsample 前，1024×16×16）

IN 定义与 1B/1E/1H 完全一致：F.instance_norm(f)（affine=False）。
α=0 走严格 short-circuit（直接返回原始 feature，与
FAlphaLayerPatchcoreModel._alpha_mix 相同语义），因此
dual(α₂=0, α₃=0) 与原始 PatchcoreModel.generate_embedding 数值等价。

不修改 1G–1J 任何已有代码；本模块为 5A 独立新增。
"""

from __future__ import annotations

import torch
from torch.nn import functional as F

from anomalib.models.image.patchcore.torch_model import PatchcoreModel

from falpha_patchcore import FAlphaPatchcore  # noqa: E402  (Lightning 包装复用)


class FAlphaDualLayerPatchcoreModel(PatchcoreModel):
    """PatchcoreModel + layer2/layer3 独立 α-IN。"""

    def __init__(self, *args, alpha_l2: float = 0.0, alpha_l3: float = 0.0, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.alpha_l2 = float(alpha_l2)
        self.alpha_l3 = float(alpha_l3)

    @staticmethod
    def _alpha_mix(f: torch.Tensor, alpha: float) -> torch.Tensor:
        """(1−α)F + α·IN(F)；α=0 直接返回原始（严格 short-circuit）。"""
        if alpha > 0.0:
            f_in = F.instance_norm(f)
            return (1.0 - alpha) * f + alpha * f_in
        return f

    def generate_embedding(self, features: dict[str, torch.Tensor]) -> torch.Tensor:
        """每层独立 α-IN 版 generate_embedding。

        结构与 PatchcoreModel / FAlphaLayerPatchcoreModel.generate_embedding
        完全一致：layer2 在 concat 前、layer3 在 upsample 前施加干预。
        """
        layer2 = features[self.layers[0]]
        layer2 = self._alpha_mix(layer2, self.alpha_l2)

        processed = [layer2]
        for layer_name in self.layers[1:]:
            layer_embedding = features[layer_name]
            # layer3 干预：在 upsample 之前（保持该层原始 16×16 统计）
            if layer_name == "layer3":
                layer_embedding = self._alpha_mix(layer_embedding, self.alpha_l3)
            layer_embedding = F.interpolate(layer_embedding, size=layer2.shape[-2:], mode="bilinear")
            processed.append(layer_embedding)

        return torch.cat(processed, 1)


class FAlphaDualLayerPatchcore(FAlphaPatchcore):
    """Lightning 版 PatchCore，torch_model 用 FAlphaDualLayerPatchcoreModel。

    其余（AP 指标、model 替换逻辑）复用 FAlphaPatchcore，与 1H 的
    FAlphaLayerPatchcore 包装方式一致。
    """

    def __init__(self, *args, alpha_l2: float = 0.0, alpha_l3: float = 0.0, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # 用 dual-layer torch 模型替换默认 FAlphaPatchcoreModel
        dtype = next(self.model.parameters()).dtype
        self.model = FAlphaDualLayerPatchcoreModel(
            backbone=self.model.backbone,
            layers=self.model.layers,
            pre_trained=True,
            num_neighbors=self.model.num_neighbors,
            alpha_l2=alpha_l2,
            alpha_l3=alpha_l3,
        ).to(dtype=dtype)
