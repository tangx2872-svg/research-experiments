"""Experiment 7A-O — 可插拔 feature module 的 PatchCore 模型定义。

只替换 generate_embedding 内的**确定性、无训练、无 target 统计**的 feature 变换；
bank / coreset / kNN / illumination / scoring 全部沿用 5A-H 代码路径。

支持的 step（全部在 layer2 于 concat 之前、layer3 于 upsample 之前施加，与 5A/5A-H 位置一致）：

  alpha_in          (1-a)F + a*IN(F)                       a=0 严格 short-circuit（与历史等价）
  residual          F + lam*N_hat                          N=IN(F), N_hat = N*rms_spatial(F)
  energy            (F + lam*N_hat) * rms_spatial(F)/rms_spatial(F + lam*N_hat)
  concat_dual       cat([F, gamma*N_hat], dim=1)           channel dim 翻倍
  altnorm_strength  (1-s)F + s*N_alt(F)                    N_alt ∈ {layernorm_all_dims, groupnorm_channels}
  compose           steps 依次施加

注意：F + lam*(IN(F)-F) 与 alpha interpolation 数学等价，**不在**本模块中实现（protocol 明确禁止）。
"""

from __future__ import annotations

import torch
from torch.nn import functional as F

from anomalib.models.image.patchcore.torch_model import PatchcoreModel

from falpha_patchcore import FAlphaPatchcore  # noqa: E402


class ModulePatchcoreModel(PatchcoreModel):
    """PatchCore with a deterministic per-layer feature module spec."""

    def __init__(self, *args, spec: dict | None = None, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.spec = dict(spec or {})

    # ---------------- primitives ----------------
    @staticmethod
    def _rms(f: torch.Tensor) -> torch.Tensor:
        dims = tuple(range(2, f.dim()))
        return f.pow(2).mean(dim=dims, keepdim=True).sqrt()

    @classmethod
    def _altnorm(cls, f: torch.Tensor, st: dict) -> torch.Tensor:
        eps = float(st.get("eps", 1e-5))
        which = st["altnorm"]
        if which == "layernorm_all_dims":
            dims = tuple(range(1, f.dim()))
            mu = f.mean(dim=dims, keepdim=True)
            var = f.var(dim=dims, unbiased=False, keepdim=True)
            return (f - mu) / (var + eps).sqrt()
        if which == "groupnorm_channels":
            g = int(st["groups"])
            b, c = f.shape[0], f.shape[1]
            assert c % g == 0, f"groups {g} must divide channels {c}"
            shape = (b, g, c // g) + tuple(f.shape[2:])
            fg = f.reshape(shape)
            dims = tuple(range(2, fg.dim()))
            mu = fg.mean(dim=dims, keepdim=True)
            var = fg.var(dim=dims, unbiased=False, keepdim=True)
            return ((fg - mu) / (var + eps).sqrt()).reshape(f.shape)
        raise ValueError(f"unknown altnorm {which}")

    @classmethod
    def _step(cls, f: torch.Tensor, st: dict) -> torch.Tensor:
        kind = st["kind"]
        if kind == "alpha_in":
            a = float(st["alpha"])
            if a <= 0.0:
                return f
            return (1.0 - a) * f + a * F.instance_norm(f)
        if kind == "residual":
            lam = float(st["lambda"])
            if lam == 0.0:
                return f
            n_hat = F.instance_norm(f) * cls._rms(f)
            return f + lam * n_hat
        if kind == "energy":
            lam = float(st["lambda"])
            if lam == 0.0:
                return f
            out = f + lam * (F.instance_norm(f) * cls._rms(f))
            return out * (cls._rms(f) / cls._rms(out).clamp_min(float(st.get("eps", 1e-5))))
        if kind == "concat_dual":
            g = float(st["gamma"])
            n_hat = F.instance_norm(f) * cls._rms(f)
            return torch.cat([f, g * n_hat], dim=1)
        if kind == "altnorm_strength":
            s = float(st["strength"])
            if s <= 0.0:
                return f
            return (1.0 - s) * f + s * cls._altnorm(f, st)
        if kind == "compose":
            for sub in st["steps"]:
                f = cls._step(f, sub)
            return f
        raise ValueError(f"unknown step kind {kind}")

    # ---------------- embedding ----------------
    def generate_embedding(self, features: dict[str, torch.Tensor]) -> torch.Tensor:
        layer2 = features[self.layers[0]]
        layer2 = self._step(layer2, self.spec["l2"])
        processed = [layer2]
        for layer_name in self.layers[1:]:
            le = features[layer_name]
            if layer_name == "layer3":
                le = self._step(le, self.spec["l3"])
            le = F.interpolate(le, size=layer2.shape[-2:], mode="bilinear")
            processed.append(le)
        return torch.cat(processed, 1)


class ModulePatchcore(FAlphaPatchcore):
    """Lightning 包装：把 torch_model 换成 ModulePatchcoreModel。"""

    def __init__(self, *args, spec: dict | None = None, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        dtype = next(self.model.parameters()).dtype
        self.model = ModulePatchcoreModel(
            backbone=self.model.backbone,
            layers=self.model.layers,
            pre_trained=True,
            num_neighbors=self.model.num_neighbors,
            spec=spec,
        ).to(dtype=dtype)
