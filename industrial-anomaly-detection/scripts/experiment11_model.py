"""Experiment 11 — 在 Exp10 的 RecoveryPatchcoreModel 上新增一个 step（不改 Exp10 文件）。

新增 `residual_gated`：
    F_out = F + beta * g_j * IN(F) * rms_spatial(F)
其中 g_j 为**冻结的** per-channel 权重向量（来自 train/good 的 normal 统计，见 experiment11_candidates）。
beta=0.25 且 g_j ≡ 1 时严格退化为 C2 的 L2 算子（`residual`）。
"""
from __future__ import annotations

import torch
from torch.nn import functional as F

from falpha_patchcore import FAlphaPatchcore  # noqa: E402
from experiment10_model import RecoveryPatchcoreModel as _Base  # noqa: E402


class AdaptivePatchcoreModel(_Base):
    @classmethod
    def _step(cls, f: torch.Tensor, st: dict) -> torch.Tensor:
        kind = st.get("kind")
        if kind == "residual_gated":
            beta = float(st["beta"])
            if beta == 0.0:
                return f
            g = torch.as_tensor(st["g"], dtype=f.dtype, device=f.device).view(1, -1, 1, 1)
            return f + beta * g * F.instance_norm(f) * cls._rms(f)
        return super()._step(f, st)


class AdaptivePatchcore(FAlphaPatchcore):
    """Lightning 包装：torch_model 用 AdaptivePatchcoreModel。"""

    def __init__(self, *args, spec: dict | None = None, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        dtype = next(self.model.parameters()).dtype
        self.model = AdaptivePatchcoreModel(
            backbone=self.model.backbone, layers=self.model.layers, pre_trained=True,
            num_neighbors=self.model.num_neighbors, spec=spec).to(dtype=dtype)
