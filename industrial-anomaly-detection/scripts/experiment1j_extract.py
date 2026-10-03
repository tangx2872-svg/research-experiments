"""Experiment 1J-A — Full Primary Layer-wise Feature Geometry Probe 的 feature 提取。

对 frozen 9 primary defects × 3 seeds × 2 layers × 2 alpha，纯 forward 提取
layer2 / layer3 的**分离** feature（非 post_concat），逐 image 计算几何统计量。

与 1H 严格对齐：
  - 数据 split：make_validation_split(category, seed)，train=189 / val=20；
  - 预处理：load_image_as_tensor + preprocess_for_model（resize 256 + ImageNet normalize）；
  - 模型：FAlphaLayerPatchcore（intervention_location = layer2 / layer3），backbone wide_resnet50_2；
  - hook 位置：generate_embedding 内，concat 前，layer2 = features["layer2"]（512,32,32），
    layer3 = features["layer3"]（1024,16,16），α-IN 施加于 upsample 前（layer3 保持原始 16×16）。

几何统计（每张 defect 图，逐 layer，逐 alpha）：
  mdc, rms_radius, normalized_pr, pca1_ratio, feature_norm

layer-specific NN（secondary cross-chain）：
  对 layer2 / layer3 各自 feature 构建 layer-specific normal bank（复用 PatchCore coreset），
  计算 defect patch -> bank 的 NN 距离分布（mean/std/median）。

输出目录结构（results/experiment_1j/）：
  per_image/experiment1j_geometry_per_image.csv
  raw/{category}/{defect}/{seed}/{layer}/alpha{...}.npz   （分离 feature，可选缓存）
  banks/{category}/{seed}/{layer}_alpha{...}.npy          （layer-specific bank）
  nn/{category}/{seed}/{layer}/alpha{...}.csv             （逐 image NN 距离统计）
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "experiments" / "2026-09-30_illumination_sensitivity_exploration" / "falpha_patchcore"))

import experiment1b_defect_sensitivity as e1b  # noqa: E402
from experiment1b_defect_sensitivity import make_validation_split  # noqa: E402
from falpha_layer_patchcore import FAlphaLayerPatchcore, FAlphaLayerPatchcoreModel  # noqa: E402

DATA_ROOT = e1b.DATA_ROOT
OUT_ROOT = ROOT / "results" / "experiment_1j"
EPS = 1e-12
SEEDS = [0, 1, 2]
LAYERS = ["layer2", "layer3"]
ALPHAS = [0.0, 1.0]

# frozen 9 primary defects（来自 selection.csv，硬编码为只读常量，禁止运行时改）
FROZEN_PRIMARY = {
    "shrink": [("cable", "bent_wire"), ("hazelnut", "print"), ("bottle", "contamination")],
    "neutral": [("screw", "manipulated_front"), ("screw", "thread_side"), ("screw", "thread_top")],
    "expand": [("grid", "glue"), ("grid", "metal_contamination"), ("grid", "thread")],
}
FAMILY_OF = {}
for fam, lst in FROZEN_PRIMARY.items():
    for cat, dt in lst:
        FAMILY_OF[(cat, dt)] = fam


def list_defect_images(category: str, defect_type: str) -> list[Path]:
    d = DATA_ROOT / category / "test" / defect_type
    return sorted(d.glob("*.png"))


# ---------------------------------------------------------------------------
# feature hook：concat 前抓分离的 layer2/layer3（已施加对应 location 的 α-IN）
# ---------------------------------------------------------------------------
def hook_layer_features(torch_model: FAlphaLayerPatchcoreModel, img_tensor: torch.Tensor,
                        device: torch.device) -> dict[str, np.ndarray]:
    """对单张图，抓取 layer2 / layer3 在 generate_embedding 内 concat 前的分离 feature。

    复用 torch_model 自身的 generate_embedding 逻辑（含 intervention_location 的 α-IN），
    但在其内部捕获 concat 前的 layer2（干预后）、layer3（干预后，upsample 前）。

    返回 {"layer2": (512,32,32), "layer3": (1024,16,16)} 的 float32 numpy 数组。
    """
    captured: dict[str, torch.Tensor] = {}

    orig_gen = torch_model.generate_embedding

    def hooked(self, features):
        # 直接复用 FAlphaLayerPatchcoreModel.generate_embedding 的源码逻辑，
        # 但在此处捕获 concat 前的分离 feature。
        layer2 = features[self.layers[0]]
        if self.intervention_location == "layer2":
            layer2 = self._alpha_mix(layer2)
        captured["layer2"] = layer2.detach().clone()

        processed = [layer2]
        for layer_name in self.layers[1:]:
            layer_embedding = features[layer_name]
            if self.intervention_location == "layer3" and layer_name == "layer3":
                layer_embedding = self._alpha_mix(layer_embedding)
            captured["layer3"] = layer_embedding.detach().clone()  # upsample 前的 layer3
            layer_embedding = torch.nn.functional.interpolate(
                layer_embedding, size=layer2.shape[-2:], mode="bilinear")
            processed.append(layer_embedding)
        embedding = torch.cat(processed, 1)
        if self.intervention_location == "post_concat":
            embedding = self._alpha_mix(embedding)
        return embedding

    torch_model.generate_embedding = hooked.__get__(torch_model)

    try:
        inp = e1b.preprocess_for_model(img_tensor, device).unsqueeze(0)
        with torch.no_grad():
            torch_model(inp)
    finally:
        torch_model.generate_embedding = orig_gen

    out = {}
    for k, t in captured.items():
        out[k] = t[0].detach().cpu().numpy().astype(np.float32)  # (C,H,W)
    return out


# ---------------------------------------------------------------------------
# 几何统计量
# ---------------------------------------------------------------------------
def geometry_of(feat: np.ndarray) -> dict[str, float]:
    """feat: (C,H,W) -> X: (P,C)。返回 mdc / rms_radius / normalized_pr / pca1_ratio / feature_norm。"""
    C, H, W = feat.shape
    X = feat.reshape(C, H * W).T.astype(np.float64)  # (P, C)
    P = X.shape[0]
    mu = X.mean(axis=0, keepdims=True)
    Xc = X - mu
    dist = np.linalg.norm(Xc, axis=1)

    mdc = float(dist.mean())
    rms_radius = float(np.sqrt(np.mean(dist ** 2)))
    feature_norm = float(np.mean(np.linalg.norm(X, axis=1)))

    # 谱分解（用 Xc 的奇异值，避免显式构造 C×C covariance）
    # Xc: (P,C)；用 torch.linalg.svdvals 求奇异值，等价于 sqrt(eigenvalues of Xc^T Xc)
    Xc_t = torch.from_numpy(Xc)
    s = torch.linalg.svdvals(Xc_t)  # 奇异值 σ_i
    lam = (s ** 2).numpy()  # eigenvalues of covariance (未除 P-1，比例不变)
    lam = lam[lam > 1e-12]
    if lam.size == 0:
        pr = pca1 = 0.0
    else:
        lam_sum = float(lam.sum())
        pr = float((lam_sum ** 2) / float((lam ** 2).sum() + EPS))
        pca1 = float(lam[0] / lam_sum)
    # normalized PR
    npr = float(pr / min(P - 1, C)) if min(P - 1, C) > 0 else 0.0

    return {
        "mdc": mdc,
        "rms_radius": rms_radius,
        "normalized_pr": npr,
        "pca1_ratio": pca1,
        "feature_norm": feature_norm,
    }


# ---------------------------------------------------------------------------
# layer-specific NN（secondary）
# ---------------------------------------------------------------------------
def build_layer_bank(normal_feats: np.ndarray, coreset_ratio: float = 0.1,
                     device: torch.device | None = None) -> np.ndarray:
    """从 normal train feature 构建 layer-specific bank，复用 anomalib 的 KCenterGreedy。

    normal_feats: (N, C) 所有 normal 图的某层 spatial feature flatten 后拼起来。
    用 PatchCore 同款 KCenterGreedy 采样 coreset_ratio 比例（GPU 加速 + SparseRandomProjection 降维）。
    """
    from anomalib.models.image.patchcore.torch_model import KCenterGreedy
    N = normal_feats.shape[0]
    n_bank = max(1, int(N * coreset_ratio))
    if n_bank >= N:
        return normal_feats.astype(np.float32)
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    emb = torch.from_numpy(normal_feats.astype(np.float32)).to(device)
    sampler = KCenterGreedy(embedding=emb, sampling_ratio=coreset_ratio)
    idxs = sampler.select_coreset_idxs()
    bank = normal_feats[np.array(idxs)].astype(np.float32)
    del emb
    return bank


def layer_nn_stats(feat: np.ndarray, bank: np.ndarray) -> dict[str, float]:
    """defect 图某层 feature (C,H,W) 到 bank 的 NN 距离分布统计（欧氏距离，PatchCore 同口径）。

    用 ||x-y||^2 = ||x||^2 - 2<x,y> + ||y||^2 展开，只产生 (chunk, N_bank) 的中间张量，
    避免 (chunk, N_bank, C) 的三维广播导致 OOM。
    """
    C, H, W = feat.shape
    X = feat.reshape(C, H * W).T.astype(np.float32)  # (P, C)
    P = X.shape[0]
    bank_norms = np.einsum("nc,nc->n", bank, bank)  # (N_bank,)
    nn = np.empty(P, dtype=np.float32)
    CHUNK = 512  # 只产生 (512, N_bank) 中间张量，可控
    for s in range(0, P, CHUNK):
        e = min(s + CHUNK, P)
        xb = X[s:e]  # (chunk, C)
        x_norms = np.einsum("pc,pc->p", xb, xb)  # (chunk,)
        # (chunk, N_bank) = -2<x,y> + ||y||^2
        d2 = -2.0 * (xb @ bank.T) + bank_norms[None, :]
        d2 += x_norms[:, None]
        np.maximum(d2, 0.0, out=d2)  # 数值稳定
        nn[s:e] = np.sqrt(d2).min(axis=1)
    return {
        "mean_nn_distance": float(nn.mean()),
        "std_nn_distance": float(nn.std(ddof=1)) if P > 1 else 0.0,
        "median_nn_distance": float(np.median(nn)),
    }


# ---------------------------------------------------------------------------
# fit 模型（复用 1H 逻辑）
# ---------------------------------------------------------------------------
def fit_layer_model(alpha: float, category: str, train_ids: list[str],
                    intervention_location: str, seed: int, logs_dir: Path):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    from anomalib.data import MVTecAD
    from anomalib.engine import Engine

    datamodule = MVTecAD(
        root=str(DATA_ROOT), category=category,
        train_batch_size=16, eval_batch_size=16, num_workers=0, seed=seed,
    )
    datamodule.setup()

    train_root = DATA_ROOT / category / "train" / "good"
    keep_names = {n for n in train_ids}
    td = datamodule.train_data
    df = td._samples
    df = df[df["image_path"].apply(lambda p: str(Path(str(p)).name) in keep_names)].reset_index(drop=True)
    td._samples = df
    if hasattr(td, "_num_samples"):
        td._num_samples = len(df)

    model = FAlphaLayerPatchcore(
        backbone="wide_resnet50_2", layers=["layer2", "layer3"],
        coreset_sampling_ratio=0.1, num_neighbors=9,
        alpha=alpha, intervention_location=intervention_location, visualizer=False,
    )
    engine = Engine(enable_progress_bar=False, logger=False, barebones=True,
                    default_root_dir=str(logs_dir / f"fit_{intervention_location}_a{alpha:g}"))
    engine.fit(model=model, datamodule=datamodule)
    torch_model = model.model
    torch_model.eval()
    return model, torch_model, datamodule


def extract_normal_layer_features(torch_model: FAlphaLayerPatchcoreModel, category: str,
                                  seed: int, device: torch.device) -> dict[str, np.ndarray]:
    """对 normal train 图提取各层 feature（用于构建 layer-specific bank）。

    返回 {"layer2": (N*1024, 512), "layer3": (N*256, 1024)} flatten 后的 normal feature。
    """
    _, train_ids = make_validation_split(category, seed)
    normal_root = DATA_ROOT / category / "train" / "good"
    l2_list, l3_list = [], []
    for name in train_ids:
        p = normal_root / name
        img = e1b.load_image_as_tensor(p)
        feats = hook_layer_features(torch_model, img, device)
        # 注意：build bank 用 α=0 的原始 normal feature（未 IN）。用 intervention 不影响的原始层 feature。
        l2 = feats["layer2"]  # (512,32,32)
        l3 = feats["layer3"]  # (1024,16,16)
        l2_list.append(l2.reshape(l2.shape[0], -1).T)
        l3_list.append(l3.reshape(l3.shape[0], -1).T)
    l2 = np.concatenate(l2_list, axis=0).astype(np.float32)
    l3 = np.concatenate(l3_list, axis=0).astype(np.float32)
    return {"layer2": l2, "layer3": l3}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--category", default=None, help="只跑某个 category（调试）")
    ap.add_argument("--seed", type=int, default=None, help="只跑某个 seed（调试）")
    ap.add_argument("--max-images", type=int, default=None, help="每 defect 最多图数（smoke）")
    ap.add_argument("--no-resume", action="store_true")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device = {device}")

    logs_dir = OUT_ROOT / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    e1b.LOGS_DIR = logs_dir
    (OUT_ROOT / "per_image").mkdir(parents=True, exist_ok=True)
    (OUT_ROOT / "raw").mkdir(parents=True, exist_ok=True)
    (OUT_ROOT / "banks").mkdir(parents=True, exist_ok=True)
    (OUT_ROOT / "nn").mkdir(parents=True, exist_ok=True)

    pi_fields = ["category", "defect_type", "family", "seed", "image_path", "layer", "alpha",
                 "mdc", "rms_radius", "normalized_pr", "pca1_ratio", "feature_norm"]
    pi_csv = OUT_ROOT / "per_image" / "experiment1j_geometry_per_image.csv"
    pi_rows: list[dict] = []
    pi_written = set()

    nn_fields = ["category", "defect_type", "family", "seed", "layer", "alpha",
                 "mean_nn_distance", "std_nn_distance", "median_nn_distance"]
    nn_csv = OUT_ROOT / "nn" / "experiment1j_nn_per_image.csv"
    nn_rows: list[dict] = []
    nn_written = set()

    def flush_pi():
        # pi_rows 在每次 flush 后都会被 clear，所以这里的所有行都是新的、未写入的，
        # 直接全部写入即可（不要用 id() 去重：clear 后新 dict 会复用已释放的内存地址，
        # 导致 id(r) 命中旧记录而被错误跳过、丢失行）。
        if pi_rows:
            exist = pi_csv.exists()
            with open(pi_csv, "a", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=pi_fields)
                if not exist:
                    w.writeheader()
                w.writerows(pi_rows)
            pi_rows.clear()

    def flush_nn():
        if nn_rows:
            exist = nn_csv.exists()
            with open(nn_csv, "a", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=nn_fields)
                if not exist:
                    w.writeheader()
                w.writerows(nn_rows)
            nn_rows.clear()

    # 按 (category, seed) 分组（同一 category 共享 train_ids / 模型 / bank）
    # 需要处理的有序 (cat, seed) 列表
    cat_seeds: list[tuple[str, int]] = []
    seen = set()
    for fam, defects in FROZEN_PRIMARY.items():
        for cat, dt in defects:
            if args.category and cat != args.category:
                continue
            for seed in SEEDS:
                if args.seed is not None and seed != args.seed:
                    continue
                if (cat, seed) not in seen:
                    seen.add((cat, seed))
                    cat_seeds.append((cat, seed))

    for cat, seed in cat_seeds:
        val_ids, train_ids = make_validation_split(cat, seed)
        print(f"\n[{cat} seed={seed}] train={len(train_ids)}", flush=True)
        logs_dir_seed = logs_dir / f"{cat}_seed{seed}"
        logs_dir_seed.mkdir(parents=True, exist_ok=True)

        # 该 category 的所有 frozen defects（用于后面按模型遍历）
        cat_defects = [(fam, cat, dt) for fam, defects in FROZEN_PRIMARY.items()
                       for c2, dt in defects if c2 == cat]

        # ---- 内存管理核心：同一时刻 GPU 上只保留一个模型 ----
        # 三个模型依次 fit/使用/释放：
        #   (1) α=0（location 任意，干预恒等）：建 bank + 提取 layer2/layer3 的 α=0 feature
        #   (2) α=1 location=layer2：提取 layer2 的 α=1 feature
        #   (3) α=1 location=layer3：提取 layer3 的 α=1 feature
        # 每个模型服务哪些 (loc, alpha) 单元：
        model_units: list[tuple[float, str, list[tuple[str, float]]]] = [
            (0.0, "layer2", [("layer2", 0.0), ("layer3", 0.0)]),
            (1.0, "layer2", [("layer2", 1.0)]),
            (1.0, "layer3", [("layer3", 1.0)]),
        ]

        banks = None
        for alpha, fit_loc, units in model_units:
            print(f"  [fit] alpha={alpha} fit_location={fit_loc} -> units={units}", flush=True)
            _, m, _ = fit_layer_model(alpha, cat, train_ids, fit_loc, seed, logs_dir_seed)
            e1b._move_model_to_device(m, device)
            try:
                if alpha == 0.0:
                    # α=0 模型：建 layer-specific bank（原始 normal feature）
                    normal_feats = extract_normal_layer_features(m, cat, seed, device)
                    banks = {}
                    for bl in LAYERS:
                        bank = build_layer_bank(normal_feats[bl], device=device)
                        banks[bl] = bank
                        np.save(OUT_ROOT / "banks" / f"{cat}_seed{seed}_{bl}.npy", bank)
                    del normal_feats
                    print(f"    banks: layer2={banks['layer2'].shape} layer3={banks['layer3'].shape}", flush=True)

                # 用当前模型提取它负责的所有 (loc, alpha) 单元的 feature
                for loc, a in units:
                    for fam, c2, dt in cat_defects:
                        imgs = list_defect_images(cat, dt)
                        if args.max_images:
                            imgs = imgs[:args.max_images]
                        for img_path in imgs:
                            stem = img_path.stem
                            # 断点续跑：npz 已存在则跳过
                            raw_dir = OUT_ROOT / "raw" / cat / dt / f"seed{seed}" / loc
                            npz_path = raw_dir / f"{stem}_a{a:g}.npz"
                            if npz_path.exists() and not args.no_resume:
                                continue
                            img = e1b.load_image_as_tensor(img_path)
                            feats = hook_layer_features(m, img, device)
                            feat = feats[loc]
                            g = geometry_of(feat)
                            pi_rows.append({
                                "category": cat, "defect_type": dt, "family": fam,
                                "seed": seed, "image_path": str(img_path),
                                "layer": loc, "alpha": a, **g,
                            })
                            raw_dir.mkdir(parents=True, exist_ok=True)
                            np.savez_compressed(npz_path, feat=feat.astype(np.float16))
                            if banks is not None:
                                ns = layer_nn_stats(feat, banks[loc])
                                nn_rows.append({
                                    "category": cat, "defect_type": dt, "family": fam,
                                    "seed": seed, "layer": loc, "alpha": a, **ns,
                                })
                            flush_pi()
                            flush_nn()
            finally:
                # 释放当前模型，避免显存累积
                del m
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    print("\n[DONE] 1J-A feature extraction complete.")
    print(f"  per_image rows: {len(pi_written)}")
    print(f"  nn rows: {len(nn_written)}")


if __name__ == "__main__":
    main()
