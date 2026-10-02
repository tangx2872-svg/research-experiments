"""Experiment 1H — Layer-Specific Normalization Sensitivity 的模型定义。

在 PatchCore 的 generate_embedding 内部，按 intervention_location 对指定层
在 concat 之前做 α-IN。三种 location：

  layer2      : 只对 features["layer2"] 做 IN（concat 前）
  layer3      : 只对 features["layer3"] 做 IN（upsample 前）
  post_concat : 对 concat 后 embedding 做 IN（与 Experiment 1E 完全一致，reference）

IN 定义与 1E 一致：F_alpha = (1-α)F + α·IN(F)，IN = F.instance_norm(affine=False)。
α=0 直接返回原始 feature，跳过 IN（三个 location 理论 bit-wise 等价）。

依赖 falpha_patchcore.FAlphaPatchcore（Lightning 包装）复用其 AP 指标与 model 替换逻辑，
只把 torch_model 换成本模块的 FAlphaLayerPatchcoreModel。
"""

from __future__ import annotations

import torch
from torch.nn import functional as F

from anomalib.models.image.patchcore.torch_model import PatchcoreModel

from falpha_patchcore import FAlphaPatchcore  # noqa: E402  (Lightning 包装复用)


class FAlphaLayerPatchcoreModel(PatchcoreModel):
    """PatchcoreModel + 可指定层的 α-IN。

    覆写 generate_embedding，在 concat 之前对指定层施加 (1-α)F + α·IN(F)。
    """

    def __init__(self, *args, alpha: float = 0.0, intervention_location: str = "post_concat", **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.alpha = float(alpha)
        self.intervention_location = intervention_location
        assert intervention_location in ("layer2", "layer3", "post_concat"), \
            f"invalid intervention_location: {intervention_location}"

    def _alpha_mix(self, f: torch.Tensor) -> torch.Tensor:
        """对单个 feature tensor 做 (1-α)F + α·IN(F)；α=0 直接返回原始。"""
        if self.alpha > 0.0:
            f_in = F.instance_norm(f)
            return (1.0 - self.alpha) * f + self.alpha * f_in
        return f

    def generate_embedding(self, features: dict[str, torch.Tensor]) -> torch.Tensor:
        """layer-specific 干预版 generate_embedding。

        与 PatchcoreModel.generate_embedding 结构一致，仅在 concat 前按 location
        对指定层做 α-IN。layer3 在 upsample 之前做（保持该层原始 16×16 统计）。
        """
        # layer2 干预：取出 layer2，若 location==layer2 则 α-IN
        layer2 = features[self.layers[0]]
        if self.intervention_location == "layer2":
            layer2 = self._alpha_mix(layer2)

        # 逐层处理（从第二层起，先 upsample 到 layer2 分辨率再 concat）
        processed = [layer2]
        for layer_name in self.layers[1:]:
            layer_embedding = features[layer_name]
            # layer3 干预：在 upsample 之前做 IN
            if self.intervention_location == "layer3" and layer_name == "layer3":
                layer_embedding = self._alpha_mix(layer_embedding)
            layer_embedding = F.interpolate(layer_embedding, size=layer2.shape[-2:], mode="bilinear")
            processed.append(layer_embedding)

        embedding = torch.cat(processed, 1)

        # post_concat 干预：与 1E 完全一致（reference）
        if self.intervention_location == "post_concat":
            embedding = self._alpha_mix(embedding)

        return embedding


class FAlphaLayerPatchcore(FAlphaPatchcore):
    """Lightning 版 PatchCore，额外暴露 intervention_location，torch_model 用
    FAlphaLayerPatchcoreModel，其余（AP 指标、model 替换）复用 FAlphaPatchcore。
    """

    def __init__(self, *args, alpha: float = 0.0, intervention_location: str = "post_concat", **kwargs) -> None:
        self.intervention_location = intervention_location
        super().__init__(*args, alpha=alpha, **kwargs)
        # 用带 location 的 torch 模型替换默认 FAlphaPatchcoreModel
        dtype = next(self.model.parameters()).dtype
        self.model = FAlphaLayerPatchcoreModel(
            backbone=self.model.backbone,
            layers=self.model.layers,
            pre_trained=True,
            num_neighbors=self.model.num_neighbors,
            alpha=self.alpha,
            intervention_location=self.intervention_location,
        ).to(dtype=dtype)
