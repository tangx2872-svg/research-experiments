"""Experiment 14 — 在 Exp11 AdaptivePatchcoreModel 之上新增唯一一个 step：`inss`（Family B）。

    F' = F - lam * P_illum(F - mu),   P_illum = U U^T  (U: (C,K) 来自 Exp13-P 冻结 basis)

约束（Exp13-P 冻结）：只作用于 **layer2**（pre-concat，与 13-P 的 pooled 空间一致）；
U 与 mu 均来自 **train/good**；不重估 subspace；不读 test/defect。
lam=0 时严格 short-circuit（与 Original 逐位一致）。
"""
from __future__ import annotations

import numpy as np
import torch

from falpha_patchcore import FAlphaPatchcore  # noqa: E402
from experiment11_model import AdaptivePatchcoreModel as _Base  # noqa: E402


class InssPatchcoreModel(_Base):
    @classmethod
    def _step(cls, f: torch.Tensor, st: dict) -> torch.Tensor:
        if st.get("kind") == "inss":
            lam = float(st["lam"])
            if lam == 0.0:
                return f
            U = torch.as_tensor(np.asarray(st["U"], dtype=np.float32), dtype=f.dtype, device=f.device)  # (K,C)
            mu = torch.as_tensor(np.asarray(st["mu"], dtype=np.float32), dtype=f.dtype, device=f.device)
            b, c = f.shape[0], f.shape[1]
            d = f - mu.view(1, c, 1, 1)
            dd = d.reshape(b, c, -1)
            proj = U.transpose(0, 1) @ (U @ dd)          # (B,C,N): P_illum(d)
            return f - lam * proj.reshape(f.shape)
        return super()._step(f, st)


class InssPatchcore(FAlphaPatchcore):
    def __init__(self, *args, spec: dict | None = None, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        dtype = next(self.model.parameters()).dtype
        self.model = InssPatchcoreModel(
            backbone=self.model.backbone, layers=self.model.layers, pre_trained=True,
            num_neighbors=self.model.num_neighbors, spec=spec).to(dtype=dtype)
