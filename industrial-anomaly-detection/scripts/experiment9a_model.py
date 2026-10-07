"""Experiment 9A — Method Screening v2：M1 / M3 两个新候选的模型定义。

只新增本文件，不修改任何历史实验代码。bank / coreset(0.1) / kNN(9) / illumination /
scoring / 指标 全部沿用 5A-H 同一代码路径（经 experiment5a_h_runner.run_config）。

M1 — Illumination-Robust Representation（per-image photometric standardization）
  在模型输入空间（已 resize(256)+ImageNet-normalize 的 y）做 per-image / per-channel
  mean-std 标准化，把每张图每通道统计对齐到 **train 正常图参考统计** (mu_ref, sigma_ref)：

      x' = (x - mu_c(x)) * (sigma_ref_c / sigma_c(x)) + mu_ref_c

  由 N(x) = (x - m_c)/s_c 可得闭式（无需反归一化）：
      y' = (sigma_ref_c / (sigma_c(y)*s_c)) * (y - mu_c(y)) + (mu_ref_c - m_c)/s_c

  零可学习参数、零 target 统计；同一变换施加于 train/val/test/defect/光照偏移全部图像。

M3 — Nuisance-Suppressed Representation（illumination 子空间投影抑制）
  逐层把 pooled pre-concat 特征投影到「光照差分协方差」top-k 特征向量的正交补：

      F' = F - U_k U_k^T F

  U_k (C x k) 由 **train 正常图** clean vs 4 个冻结光照偏移（brightness/gamma ±30%）
  差分矩阵的协方差特征分解得到（仅 train，零 target 统计，k 冻结 = 16）。
  施加位置与 5A/5A-H/7A-O 一致：layer2 在 concat 前、layer3 在 upsample 前。
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
from torch.nn import functional as F

from experiment5a_model import FAlphaDualLayerPatchcore, FAlphaDualLayerPatchcoreModel

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# 与 5A/5A-H/7A-O 冻结定义逐字一致（±30% brightness/gamma）
SHIFTS = [("brightness_0.7", "brightness", 0.7),
          ("brightness_1.3", "brightness", 1.3),
          ("gamma_0.7", "gamma", 0.7),
          ("gamma_1.3", "gamma", 1.3)]

K_NUISANCE = 16  # 冻结：每层抑制的 nuisance 方向数
M1_EPS = 1e-6


# ---------------------------------------------------------------------------
# M1
# ---------------------------------------------------------------------------
class M1IllumStandardModel(FAlphaDualLayerPatchcoreModel):
    """α=0 + 输入空间 per-image/per-channel 标准化到 train 参考统计。"""

    def __init__(self, *args, mu_ref=None, sigma_ref=None, **kwargs):
        super().__init__(*args, alpha_l2=0.0, alpha_l3=0.0, **kwargs)
        self.register_buffer("imagenet_mean",
                             torch.tensor(IMAGENET_MEAN, dtype=torch.float32).view(1, 3, 1, 1))
        self.register_buffer("imagenet_std",
                             torch.tensor(IMAGENET_STD, dtype=torch.float32).view(1, 3, 1, 1))
        self.register_buffer("mu_ref",
                             torch.tensor(mu_ref, dtype=torch.float32).view(1, 3, 1, 1))
        self.register_buffer("sigma_ref",
                             torch.tensor(sigma_ref, dtype=torch.float32).view(1, 3, 1, 1))

    def m1_correct(self, y: torch.Tensor) -> torch.Tensor:
        """y: (B,3,H,W) ImageNet-normalized -> 对齐到参考统计后的同空间张量。"""
        dt = y.dtype
        mu = y.mean(dim=(2, 3), keepdim=True)
        sd = y.std(dim=(2, 3), unbiased=False, keepdim=True)
        s = self.imagenet_std.to(dt)
        m = self.imagenet_mean.to(dt)
        scale = self.sigma_ref.to(dt) / (sd * s).clamp_min(M1_EPS)
        shift = (self.mu_ref.to(dt) - m) / s
        return scale * (y - mu) + shift

    def forward(self, input_tensor):
        return super().forward(self.m1_correct(input_tensor))


class M1IllumStandardPatchcore(FAlphaDualLayerPatchcore):
    """Lightning 包装：把 torch_model 换成 M1IllumStandardModel。"""

    def __init__(self, *args, mu_ref=None, sigma_ref=None, **kwargs):
        super().__init__(*args, alpha_l2=0.0, alpha_l3=0.0, **kwargs)
        dtype = next(self.model.parameters()).dtype
        self.model = M1IllumStandardModel(
            backbone=self.model.backbone, layers=self.model.layers, pre_trained=True,
            num_neighbors=self.model.num_neighbors,
            mu_ref=mu_ref, sigma_ref=sigma_ref).to(dtype=dtype)


# ---------------------------------------------------------------------------
# M3
# ---------------------------------------------------------------------------
def _as_basis(x):
    if x is None:
        return torch.zeros(0, 0, dtype=torch.float32)
    return torch.as_tensor(x, dtype=torch.float32)


class M3NuisanceSuppressModel(FAlphaDualLayerPatchcoreModel):
    """α=0 + 逐层把 illumination 差分协方差 top-k 方向投影掉。"""

    def __init__(self, *args, basis_l2=None, basis_l3=None, **kwargs):
        super().__init__(*args, alpha_l2=0.0, alpha_l3=0.0, **kwargs)
        self.register_buffer("U2", _as_basis(basis_l2))
        self.register_buffer("U3", _as_basis(basis_l3))

    @staticmethod
    def _project(f: torch.Tensor, U: torch.Tensor) -> torch.Tensor:
        if U.numel() == 0:
            return f
        b, c = f.shape[0], f.shape[1]
        f2 = f.reshape(b, c, -1)
        u = U.to(f.dtype)
        coef = torch.matmul(u.transpose(0, 1).unsqueeze(0), f2)  # (B,k,N)
        rem = torch.matmul(u.unsqueeze(0), coef)                 # (B,C,N)
        return (f2 - rem).reshape(f.shape)

    def generate_embedding(self, features):
        layer2 = self._project(features[self.layers[0]], self.U2)
        processed = [layer2]
        for layer_name in self.layers[1:]:
            le = features[layer_name]
            if layer_name == "layer3":
                le = self._project(le, self.U3)
            le = F.interpolate(le, size=layer2.shape[-2:], mode="bilinear")
            processed.append(le)
        return torch.cat(processed, 1)


class M3NuisanceSuppressPatchcore(FAlphaDualLayerPatchcore):
    """Lightning 包装：把 torch_model 换成 M3NuisanceSuppressModel。"""

    def __init__(self, *args, basis_l2=None, basis_l3=None, **kwargs):
        super().__init__(*args, alpha_l2=0.0, alpha_l3=0.0, **kwargs)
        dtype = next(self.model.parameters()).dtype
        self.model = M3NuisanceSuppressModel(
            backbone=self.model.backbone, layers=self.model.layers, pre_trained=True,
            num_neighbors=self.model.num_neighbors,
            basis_l2=basis_l2, basis_l3=basis_l3).to(dtype=dtype)


# ---------------------------------------------------------------------------
# M4（等价性 sanity 专用，非新方法）
# ---------------------------------------------------------------------------
class M4ResidualCompensationModel(FAlphaDualLayerPatchcoreModel):
    """M4 residual compensation 的**字面实现**（仅用于数值等价性检验）：

        F_r   = (1 - α')·F + α'·IN(F)
        F_out = F_r + λ·(F - F_r)

    代数上 F_out = (1 - α)F + α·IN(F)，其中 α = (1-λ)·α'。
    这里故意保留两步组合形式（而非直接代入 α），使 M4 sanity unit 与
    直接 effective-α implementation 构成**两条独立代码路径**，可做数值对照。
    """

    def __init__(self, *args, alpha_prime: float = 0.0, lam: float = 0.5, **kwargs) -> None:
        super().__init__(*args, alpha_l2=float(alpha_prime), alpha_l3=float(alpha_prime), **kwargs)
        self.lam = float(lam)

    def _residual(self, f: torch.Tensor, alpha_prime: float) -> torch.Tensor:
        f_r = self._alpha_mix(f, alpha_prime)
        return f_r + self.lam * (f - f_r)

    def generate_embedding(self, features: dict[str, torch.Tensor]) -> torch.Tensor:
        layer2 = self._residual(features[self.layers[0]], self.alpha_l2)
        processed = [layer2]
        for layer_name in self.layers[1:]:
            le = features[layer_name]
            if layer_name == "layer3":
                le = self._residual(le, self.alpha_l3)
            le = F.interpolate(le, size=layer2.shape[-2:], mode="bilinear")
            processed.append(le)
        return torch.cat(processed, 1)


class M4ResidualCompensationPatchcore(FAlphaDualLayerPatchcore):
    """Lightning 包装：torch_model 用 M4ResidualCompensationModel。"""

    def __init__(self, *args, alpha_prime: float = 0.0, lam: float = 0.5, **kwargs) -> None:
        super().__init__(*args, alpha_l2=alpha_prime, alpha_l3=alpha_prime, **kwargs)
        dtype = next(self.model.parameters()).dtype
        self.model = M4ResidualCompensationModel(
            backbone=self.model.backbone, layers=self.model.layers, pre_trained=True,
            num_neighbors=self.model.num_neighbors,
            alpha_prime=alpha_prime, lam=lam).to(dtype=dtype)


# ---------------------------------------------------------------------------
# M1 参考统计（CPU-only，仅 train 正常图）
# ---------------------------------------------------------------------------
def reference_stats(category: str, train_ids, data_root, image_size: int = 256):
    """返回 (mu_ref, sigma_ref)：每通道 [0,1] 空间统计，resize(256) 后计算。"""
    from torchvision.transforms import functional as TF

    import experiment1b_defect_sensitivity as e1b

    acc = torch.zeros(3, dtype=torch.float64)
    sq = torch.zeros(3, dtype=torch.float64)
    n = 0
    for nm in train_ids:
        img = e1b.load_image_as_tensor(Path(data_root) / category / "train" / "good" / nm)
        img = TF.resize(img, [image_size, image_size], antialias=True).double()
        acc += img.sum(dim=(1, 2))
        sq += (img ** 2).sum(dim=(1, 2))
        n += int(img.shape[1] * img.shape[2])
    mu = acc / n
    var = (sq / n - mu ** 2).clamp_min(0.0)
    return mu.float().tolist(), var.sqrt().clamp_min(1e-8).float().tolist()


# ---------------------------------------------------------------------------
# M3 nuisance basis 估计（仅 train 正常图；GPU 前向一次，结果落盘复用）
# ---------------------------------------------------------------------------
@torch.no_grad()
def _pooled_features(model, x: torch.Tensor):
    feats = model.feature_extractor(x)
    return (model.feature_pooler(feats["layer2"]), model.feature_pooler(feats["layer3"]))


@torch.no_grad()
def estimate_nuisance_basis(category: str, train_ids, device, data_root,
                            k: int = K_NUISANCE, batch: int = 16):
    """返回 {"layer2": U(C,k), "layer3": U(C,k), "eig_topk": {...}}。

    协方差 = Σ (F_shift - F_clean)^T (F_shift - F_clean)，对 train 正常图 × 4 个冻结偏移求和，
    在 pooled/pre-concat 特征空间做（与 generate_embedding 的输入一致）。
    """
    import numpy as np
    from anomalib.models.image.patchcore.torch_model import PatchcoreModel

    import experiment1_illumination_tradeoff as e1
    import experiment1b_defect_sensitivity as e1b

    m = PatchcoreModel(backbone="wide_resnet50_2", layers=["layer2", "layer3"],
                       pre_trained=True, num_neighbors=9)
    m.eval().to(device)
    for p in m.parameters():
        p.requires_grad_(False)

    cov2 = torch.zeros(512, 512, dtype=torch.float64, device=device)
    cov3 = torch.zeros(1024, 1024, dtype=torch.float64, device=device)
    paths = [Path(data_root) / category / "train" / "good" / nm for nm in train_ids]
    n_seen = 0
    for i in range(0, len(paths), batch):
        chunk = paths[i:i + batch]
        raw = [e1b.load_image_as_tensor(p) for p in chunk]
        x0 = torch.stack([e1b.preprocess_for_model(t, device) for t in raw])
        f0 = _pooled_features(m, x0)
        for _, itype, level in SHIFTS:
            xs = torch.stack([e1b.preprocess_for_model(
                e1.apply_photometric(t, itype, level), device) for t in raw])
            fs = _pooled_features(m, xs)
            for li in (0, 1):
                a, b = f0[li], fs[li]
                d = (b - a).reshape(-1, b.shape[1]).double()
                cov = d.transpose(0, 1) @ d
                if li == 0:
                    cov2 += cov
                else:
                    cov3 += cov
            del xs, fs
        del x0, f0, raw
        n_seen += len(chunk)
    out = {"n_train": n_seen, "per_layer": {}}
    for li, (lname, cov) in enumerate((("layer2", cov2), ("layer3", cov3))):
        evals, evecs = torch.linalg.eigh(cov)          # 升序
        top = evals[-k:]
        out[lname] = evecs[:, -k:].float().cpu().numpy()  # (C,k) 正交基
        out["per_layer"][lname] = {
            "C": int(cov.shape[0]), "k": int(k),
            "topk_eig": [float(v) for v in top.flip(0).cpu().numpy()],
            "trace": float(torch.diagonal(cov).sum().cpu()),
            "topk_share_of_trace": float(top.sum().cpu() / torch.diagonal(cov).sum().cpu()),
        }
    del cov2, cov3, m
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return out


def basis_hash(category: str, seed: int, train_ids, k: int) -> str:
    import hashlib

    payload = ",".join([category, str(seed), str(k), "v1"] + sorted(train_ids))
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def load_or_estimate_basis(cache_dir, category: str, seed: int, train_ids, device,
                           data_root, k: int = K_NUISANCE, force: bool = False):
    """落盘复用。返回 (basis{layer2,layer3}, meta)。"""
    import numpy as np

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    h = basis_hash(category, seed, train_ids, k)
    stem = f"nuisance_basis_{category}_seed{seed}_k{k}_{h}"
    npz_p, json_p = cache_dir / f"{stem}.npz", cache_dir / f"{stem}.json"
    if npz_p.exists() and json_p.exists() and not force:
        z = np.load(npz_p)
        meta = json.loads(json_p.read_text())
        meta["reused"] = True
        return {"layer2": z["U2"], "layer3": z["U3"]}, meta
    est = estimate_nuisance_basis(category, train_ids, device, data_root, k=k)
    np.savez_compressed(npz_p, U2=est["layer2"], U3=est["layer3"])
    meta = {"category": category, "seed": seed, "k": k, "hash": h,
            "n_train": est["n_train"], "per_layer": est["per_layer"],
            "shifts": [list(s) for s in SHIFTS], "reused": False,
            "npz": str(npz_p), "created_at": __import__("time").time()}
    json_p.write_text(json.dumps(meta, indent=2))
    return {"layer2": est["layer2"], "layer3": est["layer3"]}, meta
