"""Experiment 10 — 新增两个 deterministic step（在 7A-O 模型之上扩展，不改历史代码）。

Family B（dual-branch concatenation，**与 7A-O Family B / M10 不同**）：
  7A-O B1 = cat([F, gamma*N_hat(F)])                       —— 与 F 的**标准化副本**拼接
  Exp10 B1 = cat([F, (1-a)F + a*IN(F)])                     —— 与 F 的**robust 分支**拼接（T1 用的同一算子）
  Exp10 B2 = cat([F, (1-a)F + a*IN(F)*rms(F)])              —— 尺度匹配版（只用 per-sample RMS，不用任何 target 统计）

两条都是「保留 Original block + 追加 robust block」，用于检验 preservation 能否被补回。
"""

from __future__ import annotations

import torch
from torch.nn import functional as F

from falpha_patchcore import FAlphaPatchcore  # noqa: E402
from experiment7ao_model import ModulePatchcoreModel as _BaseModel  # noqa: E402


class RecoveryPatchcoreModel(_BaseModel):
    """继承 7A-O 的 step 机制，只新增 concat_robust / concat_robust_scaled。"""

    @classmethod
    def _step(cls, f: torch.Tensor, st: dict) -> torch.Tensor:
        kind = st.get("kind")
        if kind == "concat_robust":
            a = float(st["alpha"])
            if a <= 0.0:
                return f
            return torch.cat([f, (1.0 - a) * f + a * F.instance_norm(f)], dim=1)
        if kind == "concat_robust_scaled":
            a = float(st["alpha"])
            if a <= 0.0:
                return f
            return torch.cat([f, (1.0 - a) * f + a * F.instance_norm(f) * cls._rms(f)], dim=1)
        return super()._step(f, st)


class RecoveryPatchcore(FAlphaPatchcore):
    """Lightning 包装：torch_model 用 RecoveryPatchcoreModel。"""

    def __init__(self, *args, spec: dict | None = None, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        dtype = next(self.model.parameters()).dtype
        self.model = RecoveryPatchcoreModel(
            backbone=self.model.backbone,
            layers=self.model.layers,
            pre_trained=True,
            num_neighbors=self.model.num_neighbors,
            spec=spec,
        ).to(dtype=dtype)
