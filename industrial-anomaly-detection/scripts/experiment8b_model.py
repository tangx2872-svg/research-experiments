"""Experiment 8B Phase B — channel-masked PatchCore（diagnostic probe 用）。

只把 embedding 换成 **layer 内 channel 子集**（α=0，无任何 IN / 无训练 / 无 target 统计）；
bank / coreset(0.1) / kNN(9) / illumination / scoring / 指标全部沿用 5A-H 同一代码路径，
经 experiment7ao_runner 的同款 monkey-patch 方式注入。

mask 为 None 时等价于原始 PatchCore（FULL）。数学上 mask 在 upsample 之前或之后施加等价
（bilinear interpolate 是逐 channel 线性算子），这里按 5A/5A-H 的历史位置施加于 concat 前。
"""

from __future__ import annotations

import torch
from torch.nn import functional as F

from anomalib.models.image.patchcore.torch_model import PatchcoreModel

from falpha_patchcore import FAlphaPatchcore  # noqa: E402


class MaskedPatchcoreModel(PatchcoreModel):
    """PatchCore with per-layer channel selection (index_select) inside generate_embedding."""

    def __init__(self, *args, mask_l2=None, mask_l3=None, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.mask_l2 = None if mask_l2 is None else torch.as_tensor(mask_l2, dtype=torch.long)
        self.mask_l3 = None if mask_l3 is None else torch.as_tensor(mask_l3, dtype=torch.long)

    def _mask_of(self, layer: str):
        if layer == self.layers[0] or layer == "layer2":
            return self.mask_l2
        if layer == "layer3":
            return self.mask_l3
        return None

    def generate_embedding(self, features: dict[str, torch.Tensor]) -> torch.Tensor:
        first = self.layers[0]
        emb = features[first]
        m = self._mask_of(first)
        if m is not None:
            emb = emb.index_select(1, m.to(emb.device))
        outs = [emb]
        for name in self.layers[1:]:
            le = features[name]
            m = self._mask_of(name)
            if m is not None:
                le = le.index_select(1, m.to(le.device))
            outs.append(F.interpolate(le, size=emb.shape[-2:], mode="bilinear"))
        return torch.cat(outs, 1)


class MaskedPatchcore(FAlphaPatchcore):
    """Lightning 版 PatchCore，torch_model 换成 MaskedPatchcoreModel（alpha 恒为 0）。"""

    def __init__(self, *args, mask_l2=None, mask_l3=None, **kwargs) -> None:
        super().__init__(*args, alpha=0.0, **kwargs)
        dtype = next(self.model.parameters()).dtype
        self.model = MaskedPatchcoreModel(
            backbone=self.model.backbone,
            layers=self.model.layers,
            pre_trained=True,
            num_neighbors=self.model.num_neighbors,
            mask_l2=mask_l2,
            mask_l3=mask_l3,
        ).to(dtype=dtype)
