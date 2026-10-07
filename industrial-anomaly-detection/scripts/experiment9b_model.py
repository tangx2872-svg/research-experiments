"""Experiment 9B — 候选模型定义（M7 / M8 / M9 / M11；M10 走历史复用）。只新增本文件。

M7  — Layer-Selective Normalization：逐层独立 α-IN（复用既有 FAlphaDualLayerPatchcoreModel）。
M8  — Normal-Only Channel Gate：由 train-only s_c 构造软门
      F'_c = (1 - a_c) * F_c + a_c * IN(F)_c
M9  — Soft Channel Weighting：s_c 的绝对标度映射（M3 hard projection 的温和替代）。
M11 — Layer x Channel Gate：层级基强度 (a_bar_L2, a_bar_L3) x 逐层 channel 门。

冻结约束（见 results/experiment_9b_screening/config.json）：
  * 只用 train/good 估计 s_c；不使用 test defect mask / test label / test defect statistics
  * 无梯度训练；无 learnable 参数；IN = F.instance_norm(affine=False)
  * 施加位置与全部历史 α-IN 实验一致：layer2 在 concat 之前、layer3 在 upsample 之前
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import torch
from torch.nn import functional as F

from experiment5a_model import FAlphaDualLayerPatchcore, FAlphaDualLayerPatchcoreModel

# 与 5A/5A-H/7A-O/9A 逐字一致的冻结光照定义
SHIFTS = [("brightness_0.7", "brightness", 0.7),
          ("brightness_1.3", "brightness", 1.3),
          ("gamma_0.7", "gamma", 0.7),
          ("gamma_1.3", "gamma", 1.3)]

GATE_EPS = 1e-8
S_EPS = 1e-12


def _as_gate(x):
    if x is None:
        return torch.zeros(0, dtype=torch.float32)
    return torch.as_tensor(x, dtype=torch.float32)


class ChannelGatePatchcoreModel(FAlphaDualLayerPatchcoreModel):
    """F'_c = (1 - a_c) * F_c + a_c * instance_norm(F)_c（逐层、逐 channel 软门）。

    gate 全 0 时严格 short-circuit 返回原始 feature（保证 gate=0 与原始 PatchCore 等价）；
    gate 为常数 c 时数学上等价于 uniform α=c 的 α-IN（CPU sanity S4 验证）。
    """

    def __init__(self, *args, gate_l2=None, gate_l3=None, **kwargs) -> None:
        super().__init__(*args, alpha_l2=0.0, alpha_l3=0.0, **kwargs)
        self.register_buffer("gate2", _as_gate(gate_l2))
        self.register_buffer("gate3", _as_gate(gate_l3))

    @staticmethod
    def _gated_mix(f: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        if a.numel() == 0:
            return f
        if float(a.max()) == 0.0:
            return f                      # 严格 short-circuit（bit-exact）
        w = a.to(f.dtype).view(1, -1, 1, 1)
        return (1.0 - w) * f + w * F.instance_norm(f)

    def generate_embedding(self, features: dict) -> torch.Tensor:
        layer2 = self._gated_mix(features[self.layers[0]], self.gate2)
        processed = [layer2]
        for layer_name in self.layers[1:]:
            le = features[layer_name]
            if layer_name == "layer3":
                le = self._gated_mix(le, self.gate3)
            le = F.interpolate(le, size=layer2.shape[-2:], mode="bilinear")
            processed.append(le)
        return torch.cat(processed, 1)


class ChannelGatePatchcore(FAlphaDualLayerPatchcore):
    """Lightning 包装：把 torch_model 换成 ChannelGatePatchcoreModel。"""

    def __init__(self, *args, gate_l2=None, gate_l3=None, **kwargs) -> None:
        super().__init__(*args, alpha_l2=0.0, alpha_l3=0.0, **kwargs)
        dtype = next(self.model.parameters()).dtype
        self.model = ChannelGatePatchcoreModel(
            backbone=self.model.backbone, layers=self.model.layers, pre_trained=True,
            num_neighbors=self.model.num_neighbors,
            gate_l2=gate_l2, gate_l3=gate_l3).to(dtype=dtype)


# ---------------------------------------------------------------------------
# normal-only illumination sensitivity s_c
# ---------------------------------------------------------------------------
@torch.no_grad()
def _pooled(model, x: torch.Tensor):
    feats = model.feature_extractor(x)
    return (model.feature_pooler(feats["layer2"]), model.feature_pooler(feats["layer3"]))


@torch.no_grad()
def estimate_illum_sensitivity(category: str, train_ids, device, data_root, batch: int = 16):
    """s_c = mean_{image,shift} mean_spatial |pooled(T(x))_c - pooled(x)_c|（**仅 train/good**）。"""
    from anomalib.models.image.patchcore.torch_model import PatchcoreModel

    import experiment1_illumination_tradeoff as e1
    import experiment1b_defect_sensitivity as e1b

    m = PatchcoreModel(backbone="wide_resnet50_2", layers=["layer2", "layer3"],
                       pre_trained=True, num_neighbors=9)
    m.eval().to(device)
    for p in m.parameters():
        p.requires_grad_(False)

    s2 = torch.zeros(512, dtype=torch.float64, device=device)
    s3 = torch.zeros(1024, dtype=torch.float64, device=device)
    paths = [Path(data_root) / category / "train" / "good" / nm for nm in train_ids]
    n_seen = 0
    for i in range(0, len(paths), batch):
        chunk = paths[i:i + batch]
        raw = [e1b.load_image_as_tensor(p) for p in chunk]
        x0 = torch.stack([e1b.preprocess_for_model(t, device) for t in raw])
        f0 = _pooled(m, x0)
        for _, itype, level in SHIFTS:
            xs = torch.stack([e1b.preprocess_for_model(
                e1.apply_photometric(t, itype, level), device) for t in raw])
            fs = _pooled(m, xs)
            for li, (a, b) in enumerate(zip(f0, fs)):
                d = (b - a).abs().mean(dim=(0, 2, 3)).double()   # (C,) per-image spatial mean
                if li == 0:
                    s2 += d          # BUGFIX(9B): 必须是 (C,) 向量累加；d.sum() 会把标量广播到全部 channel
                else:
                    s3 += d
            del xs, fs
        del x0, f0, raw
        n_seen += len(chunk)
    denom = float(n_seen * len(SHIFTS))
    s2, s3 = (s2 / denom).cpu(), (s3 / denom).cpu()
    del m
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {"layer2": s2.numpy(), "layer3": s3.numpy(), "n_train": n_seen}


def sensitivity_hash(category: str, seed: int, train_ids) -> str:
    payload = ",".join([category, str(seed), "9b-v2"] + sorted(train_ids))
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def _stat_dict(s):
    import numpy as np
    q = np.percentile(s, [0, 1, 5, 50, 90, 95, 99, 100])
    return {"C": int(s.shape[0]), "min": float(q[0]), "p1": float(q[1]), "p5": float(q[2]),
            "p50": float(q[3]), "p90": float(q[4]), "p95": float(q[5]), "p99": float(q[6]),
            "max": float(q[7]), "mean": float(s.mean()), "std": float(s.std())}


def load_or_estimate_sensitivity(cache_dir, category: str, seed: int, train_ids, device,
                                 data_root, force: bool = False):
    """落盘复用：每个 (category, seed) 只估一次，M8/M9/M11 共用。返回 (s, meta)。"""
    import numpy as np

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    h = sensitivity_hash(category, seed, train_ids)
    npz_p = cache_dir / f"illum_sensitivity_{category}_seed{seed}_{h}.npz"
    json_p = cache_dir / f"illum_sensitivity_{category}_seed{seed}_{h}.json"
    if npz_p.exists() and json_p.exists() and not force:
        z = np.load(npz_p)
        meta = json.loads(json_p.read_text())
        meta["reused"] = True
        return {"layer2": z["s2"], "layer3": z["s3"]}, meta
    est = estimate_illum_sensitivity(category, train_ids, device, data_root)
    np.savez_compressed(npz_p, s2=est["layer2"], s3=est["layer3"])
    meta = {"category": category, "seed": seed, "hash": h, "n_train": est["n_train"],
            "shifts": [list(x) for x in SHIFTS], "reused": False,
            "source": "train/good only (normal-only)", "npz": str(npz_p),
            "created_at": time.time(),
            "stats": {ln: _stat_dict(np.asarray(est[ln])) for ln in ("layer2", "layer3")}}
    json_p.write_text(json.dumps(meta, indent=2))
    return {"layer2": est["layer2"], "layer3": est["layer3"]}, meta


# ---------------------------------------------------------------------------
# 预注册 gate 映射（全部在 config.json 中冻结，不读 test 数据）
# ---------------------------------------------------------------------------
def zscore(s):
    import numpy as np
    return (s - s.mean()) / (s.std() + GATE_EPS)


def make_gate(s, map_name: str, **kw):
    """返回 (gate: float32 (C,), 记录用 stats)。三种映射均为冻结定义。"""
    import numpy as np
    s = np.asarray(s, dtype=np.float64)
    if map_name in ("m8_zmap", "m9_powmap") and float(s.std()) <= 0.0:
        raise ValueError("degenerate sensitivity s_c (per-channel std == 0) — refuse to build gate")
    if map_name == "m8_zmap":                        # a = clip(a_bar + beta * z_c, 0, 1)
        a = np.clip(kw["a_bar"] + kw["beta"] * zscore(s), 0.0, 1.0)
        desc = "clip(a_bar + beta*z_c, 0, 1); a_bar=%g beta=%g" % (kw["a_bar"], kw["beta"])
    elif map_name == "m9_powmap":                    # w = clip((s_c / p90_c)**gamma, 0, 1)
        p90 = float(np.percentile(s, 90))
        a = np.clip((s / (p90 + S_EPS)) ** kw["gamma"], 0.0, 1.0)
        desc = "clip((s_c/p90_c)**gamma, 0, 1); p90_c=%.6g gamma=%g" % (p90, kw["gamma"])
    elif map_name == "const":                        # 常数 gate（仅 sanity 用）
        a = np.full_like(s, float(kw["value"]))
        desc = "constant gate = %g" % kw["value"]
    else:
        raise ValueError(map_name)
    st = {"map": map_name, "desc": desc, "mean": float(a.mean()), "std": float(a.std()),
          "min": float(a.min()), "max": float(a.max()),
          "frac_ge_1": float((a >= 1.0 - 1e-12).mean()),
          "frac_le_0": float((a <= 1e-12).mean())}
    return a.astype(np.float32), st
