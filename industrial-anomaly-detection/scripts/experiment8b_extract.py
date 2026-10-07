"""Experiment 8B Phase A — 特征提取（8B-A，最小必要 GPU 补测）。

只做 frozen backbone forward + 统计聚合：
  - 不 fit / 不建 bank / 不训练 / 不改任何已冻结协议；
  - 表示 = α=0 PatchCore pretrained（wide_resnet50_2, layer2/layer3），layer3 不 upsample。

产出（per (category, seed)）：
  normal_stats/{cat}_seed{s}.npz   : mu_c / sigma_c / n_patches（layer2 + layer3, 全 train/good）
  illum/{cat}_seed{s}.npz          : paired illumination per-image per-channel |Δf|（12 cond + identity）
  defect/{cat}_seed{s}.npz         : per-image defect-region per-channel patch mean（全部 defect type）
  sanity_extract.json              : S1/S3/S6/S13 的实测证据

用法：
  python -u scripts/experiment8b_extract.py --mode smoke
  python -u scripts/experiment8b_extract.py --mode full [--out-root ...]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import experiment1b_defect_sensitivity as e1b  # noqa: E402
import experiment1_illumination_tradeoff as e1  # noqa: E402

DATA_ROOT = e1b.DATA_ROOT
BANKS_1J = ROOT / "results" / "experiment_1j" / "banks"
RAW_1J = ROOT / "results" / "experiment_1j" / "raw"

CATEGORIES = ["bottle", "cable", "grid", "hazelnut", "screw"]
SEEDS = [0, 1, 2]
LAYERS = ["layer2", "layer3"]
CDIM = {"layer2": 512, "layer3": 1024}
GRID = {"layer2": (32, 32), "layer3": (16, 16)}
BATCH = 16
OVERLAP_DEFECT = 0.5

FROZEN_PRIMARY = {
    "cable": ["bent_wire"], "hazelnut": ["print"], "bottle": ["contamination"],
    "screw": ["manipulated_front", "thread_side", "thread_top"],
    "grid": ["glue", "metal_contamination", "thread"],
}

# 冻结条件集：12 个计入 I + identity（仅 S6 校验）
FAMILIES = ["brightness", "gamma"]
LEVELS = [0.5, 0.7, 0.9, 1.1, 1.3, 1.5]
CONDITIONS = [(f, lv) for f in FAMILIES for lv in LEVELS]      # 12
IDENTITY = [(f, 1.0) for f in FAMILIES]                         # S6 only
ALL_CONDITIONS = CONDITIONS + IDENTITY
SEVERITY_OF = {0.5: "strong", 1.5: "strong", 0.7: "medium", 1.3: "medium",
               0.9: "mild", 1.1: "mild"}


# ---------------------------------------------------------------------------
# 模型 / 预处理
# ---------------------------------------------------------------------------
def build_model(device: torch.device):
    from anomalib.models.image.patchcore.torch_model import PatchcoreModel
    m = PatchcoreModel(backbone="wide_resnet50_2", layers=["layer2", "layer3"],
                       pre_trained=True, num_neighbors=9)
    m.eval().to(device)
    for p in m.parameters():
        p.requires_grad_(False)
    return m


@torch.no_grad()
def features_of(model, batch: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """batch: (B,3,256,256) 已预处理 -> (layer2 (B,512,32,32), layer3 (B,1024,16,16))。

    必须复现 PatchcoreModel.forward 的约定：feature_extractor 之后、generate_embedding 之前
    对所有层特征施加 model.feature_pooler = AvgPool2d(3,1,1)。
    （1J/1H 历史资产在 generate_embedding 内部捕获，因此是 pooled 之后的特征；
    S13 实测若跳过 pooler 会 corr≈0.89 / max|Δ|≈26 —— 已由 sanity 捕获并修正。）
    """
    feats = model.feature_extractor(batch)
    return (model.feature_pooler(feats["layer2"]), model.feature_pooler(feats["layer3"]))


def load_preprocessed(paths: list[Path], device: torch.device) -> torch.Tensor:
    tensors = [e1b.preprocess_for_model(e1b.load_image_as_tensor(p), device) for p in paths]
    return torch.stack(tensors, 0)


def load_mask_256(mask_path: Path) -> np.ndarray:
    m = np.asarray(Image.open(mask_path).convert("L"), dtype=np.float32) / 255.0
    if m.shape != (256, 256):
        m = np.asarray(Image.fromarray((m * 255).astype(np.uint8)).resize((256, 256), Image.BILINEAR),
                       dtype=np.float32) / 255.0
    return (m > 0.5).astype(np.float32)


def mask_overlap_grid(mask: np.ndarray, gh: int, gw: int) -> np.ndarray:
    h, w = mask.shape
    out = np.zeros((gh, gw), dtype=np.float32)
    for i in range(gh):
        h0, h1 = int(i * h / gh), int((i + 1) * h / gh)
        for j in range(gw):
            w0, w1 = int(j * w / gw), int((j + 1) * w / gw)
            out[i, j] = float(mask[h0:h1, w0:w1].mean())
    return out


def defect_patch_index(mask: np.ndarray, layer: str) -> tuple[np.ndarray, bool]:
    """返回 (defect patch 的 flatten 索引, fallback_flag)。冻结规则见 README §2.4。"""
    gh, gw = GRID[layer]
    ov = mask_overlap_grid(mask, gh, gw).reshape(-1)
    idx = np.where(ov > OVERLAP_DEFECT)[0]
    if idx.size == 0:
        idx = np.array([int(np.argmax(ov))], dtype=np.int64)
        return idx, True
    return idx, False


# ---------------------------------------------------------------------------
# 单元 1：normal 统计（全 train/good）
# ---------------------------------------------------------------------------
@torch.no_grad()
def normal_stats(model, device, train_ids: list[str], category: str, log) -> dict:
    root = DATA_ROOT / category / "train" / "good"
    acc = {ly: {"s": torch.zeros(CDIM[ly], dtype=torch.float64, device=device),
                "s2": torch.zeros(CDIM[ly], dtype=torch.float64, device=device),
                "n": 0} for ly in LAYERS}
    for i in range(0, len(train_ids), BATCH):
        chunk = train_ids[i:i + BATCH]
        x = load_preprocessed([root / n for n in chunk], device)
        l2, l3 = features_of(model, x)
        for ly, t in (("layer2", l2), ("layer3", l3)):
            C, B_, H, W = t.shape[1], t.shape[0], t.shape[2], t.shape[3]
            X = t.permute(1, 0, 2, 3).reshape(C, B_ * H * W).double()
            acc[ly]["s"] += X.sum(dim=1)
            acc[ly]["s2"] += (X * X).sum(dim=1)
            acc[ly]["n"] += X.shape[1]
    out = {}
    for ly in LAYERS:
        n = acc[ly]["n"]
        mu = (acc[ly]["s"] / n).cpu().numpy()
        var = torch.clamp(acc[ly]["s2"] / n - (acc[ly]["s"] / n) ** 2, min=0.0)
        out[ly] = {"mu": mu.astype(np.float64), "sigma": var.sqrt().cpu().numpy().astype(np.float64),
                   "n_patches": int(n)}
        log(f"    normal {ly}: n_patches={n} sigma median={np.median(out[ly]['sigma']):.4f}")
    return out


# ---------------------------------------------------------------------------
# 单元 2：paired illumination（test/good 图像 x 与其扰动 T(x)）
# ---------------------------------------------------------------------------
@torch.no_grad()
def illum_paired(model, device, good_paths: list[Path], out_dir: Path) -> dict:
    n_img = len(good_paths)
    n_cond = len(ALL_CONDITIONS)
    delta = {ly: np.zeros((n_cond, n_img, CDIM[ly]), dtype=np.float32) for ly in LAYERS}
    orig_mean = {ly: np.zeros((n_img, CDIM[ly]), dtype=np.float32) for ly in LAYERS}
    pooled_abs = {ly: np.zeros((n_cond, CDIM[ly]), dtype=np.float64) for ly in LAYERS}
    pooled_n = 0

    for s in range(0, n_img, 8):
        chunk = good_paths[s:s + 8]
        x = load_preprocessed(chunk, device)
        l2o, l3o = features_of(model, x)
        for ly, t in (("layer2", l2o), ("layer3", l3o)):
            C = t.shape[1]
            orig_mean[ly][s:s + len(chunk)] = t.reshape(t.shape[0], C, -1).mean(2).cpu().numpy()
        pooled_n += l2o.shape[0] * l2o.shape[2] * l2o.shape[3]

        for ci, (fam, lv) in enumerate(ALL_CONDITIONS):
            pert = []
            for p in chunk:
                img = e1b.load_image_as_tensor(p)
                pert.append(e1b.preprocess_for_model(e1.apply_photometric(img, fam, lv), device))
            xp = torch.stack(pert, 0)
            l2p, l3p = features_of(model, xp)
            for ly, fo, fp in (("layer2", l2o, l2p), ("layer3", l3o, l3p)):
                d = (fp - fo).abs()                       # (B,C,H,W)
                C = d.shape[1]
                per_img = d.reshape(d.shape[0], C, -1).mean(2)     # (B,C) mean over patches
                delta[ly][ci, s:s + len(chunk)] = per_img.cpu().numpy()
                pooled_abs[ly][ci] += d.reshape(d.shape[0], C, -1).sum(2).double().sum(0).cpu().numpy()

    payload = {"cond_family": np.array([c[0] for c in ALL_CONDITIONS]),
               "cond_level": np.array([c[1] for c in ALL_CONDITIONS], dtype=np.float64),
               "cond_is_identity": np.array([c in IDENTITY for c in ALL_CONDITIONS]),
               "image_names": np.array([p.name for p in good_paths]),
               "pooled_n_patches": np.array([pooled_n])}
    for ly in LAYERS:
        payload[f"delta_{ly}"] = delta[ly]
        payload[f"orig_mean_{ly}"] = orig_mean[ly]
        payload[f"pooled_abs_{ly}"] = pooled_abs[ly]
    np.savez_compressed(out_dir / "illum.npz", **payload)

    ident = [i for i, c in enumerate(ALL_CONDITIONS) if c in IDENTITY]
    s6 = {ly: float(np.abs(delta[ly][ident]).max()) for ly in LAYERS}
    return {"n_images": n_img, "n_conditions_total": n_cond,
            "s6_identity_max_abs_delta": s6}


# ---------------------------------------------------------------------------
# 单元 3：defect-region per-image per-channel patch mean（全部 defect type）
# ---------------------------------------------------------------------------
@torch.no_grad()
def defect_features(model, device, category: str, seed: int, defect_paths: dict, out_dir: Path,
                    max_per_type: int | None, log) -> dict:
    rows = {ly: [] for ly in LAYERS}
    meta = {"image_path": [], "defect_type": [], "frozen_primary": [],
            "fallback_layer2": [], "fallback_layer3": [], "n_defect_patch_layer2": [],
            "n_defect_patch_layer3": [], "mask_area_ratio": []}
    s13 = []
    for dt, paths in defect_paths.items():
        use = paths[:max_per_type] if max_per_type else paths
        for i in range(0, len(use), BATCH):
            chunk = use[i:i + BATCH]
            x = load_preprocessed(chunk, device)
            l2, l3 = features_of(model, x)
            for k, p in enumerate(chunk):
                mp = DATA_ROOT / category / "ground_truth" / dt / f"{p.stem}_mask.png"
                m = load_mask_256(mp)
                meta["image_path"].append(str(p))
                meta["defect_type"].append(dt)
                meta["frozen_primary"].append(dt in FROZEN_PRIMARY.get(category, []))
                meta["mask_area_ratio"].append(float(m.mean()))
                for ly, t in (("layer2", l2), ("layer3", l3)):
                    C, H, W = t.shape[1], t.shape[2], t.shape[3]
                    flat = t[k].reshape(C, H * W)                    # (C, HW)
                    idx, fb = defect_patch_index(m, ly)
                    rows[ly].append(flat[:, torch.from_numpy(idx).to(device)].mean(1).cpu().numpy())
                    meta[f"fallback_{ly}"].append(bool(fb))
                    meta[f"n_defect_patch_{ly}"].append(int(idx.size))
            # S13：与 1J npz 逐元素比对（仅 frozen primary defect，最多 3 张）
            if dt in FROZEN_PRIMARY.get(category, []):
                s13 += s13_check(category, dt, seed, chunk, l2, l3, max_n=3, log=log)
    payload = {f"m_{ly}": np.stack(rows[ly]).astype(np.float32) for ly in LAYERS}
    payload.update({k: np.array(v) for k, v in meta.items()})
    payload["layer2_channel_dim"] = np.array([CDIM["layer2"]])
    payload["layer3_channel_dim"] = np.array([CDIM["layer3"]])
    np.savez_compressed(out_dir / "defect.npz", **payload)
    return {"n_defect_images": len(meta["image_path"]),
            "n_fallback_layer2": int(sum(meta["fallback_layer2"])),
            "n_fallback_layer3": int(sum(meta["fallback_layer3"])),
            "s13": s13}


# ---------------------------------------------------------------------------
# S13：新提取 feature vs 1J 历史资产（逐元素）
# ---------------------------------------------------------------------------
def s13_check(category: str, dt: str, seed: int, chunk: list[Path],
              l2: torch.Tensor, l3: torch.Tensor, max_n: int = 3, log=print) -> list:
    out = []
    for k, p in enumerate(chunk[:max_n]):
        for ly, t in (("layer2", l2), ("layer3", l3)):
            npy = RAW_1J / category / dt / f"seed{seed}" / ly / f"{p.stem}_a0.npz"
            if not npy.exists():
                out.append({"category": category, "defect_type": dt, "seed": seed, "layer": ly,
                            "image": p.name, "status": "MISSING_1J_ASSET"})
                continue
            ref = np.load(npy)["feat"].astype(np.float32)
            ours = t[k].detach().cpu().numpy().astype(np.float32)
            same_shape = list(ref.shape) == list(ours.shape)
            if same_shape:
                d = np.abs(ours - ref)
                f16_frac = float((np.float16(ours) == np.float16(ref)).mean())
                maxabs, medabs = float(d.max()), float(np.median(d))
                p99 = float(np.percentile(d, 99))
                corr = float(np.corrcoef(ours.ravel(), ref.ravel())[0, 1])
            else:
                f16_frac = maxabs = medabs = p99 = corr = float("nan")
            # S13 判据（PRE-RUN AMENDMENT A1，见 README §5）：容差化，因为 GPU 结果与 batch 组成
            # 相关（实测同图 batch1 vs batch8 max|Δ|=6.5e-3），bit-exact 不可达。
            ok = bool(same_shape and corr >= 0.9999 and medabs <= 1e-3
                      and p99 <= 5e-3 and maxabs <= 0.05)
            out.append({"category": category, "defect_type": dt, "seed": seed, "layer": ly,
                        "image": p.name, "status": "OK", "same_shape": bool(same_shape),
                        "shape_ref": list(ref.shape), "shape_new": list(ours.shape),
                        "float16_identical_fraction": f16_frac, "max_abs_diff": maxabs,
                        "median_abs_diff": medabs, "p99_abs_diff": p99, "corr": corr,
                        "s13_pass": ok})
    f = [r for r in out if r["status"] == "OK"]
    if f:
        log(f"    S13 {category}/{dt}: n={len(f)} pass={sum(r['s13_pass'] for r in f)}/{len(f)} "
            f"corr_min={min(r['corr'] for r in f):.6f} "
            f"max|d|_max={max(r['max_abs_diff'] for r in f):.3e} "
            f"median|d|_max={max(r['median_abs_diff'] for r in f):.3e}")
    return out


# ---------------------------------------------------------------------------
# S5：image<->mask alignment（用 1J layer bank 的 patch NN distance，检验 top-5% 与 mask 重叠）
# ---------------------------------------------------------------------------
def s5_alignment(category: str, seed: int, layer: str, feat: np.ndarray, mask: np.ndarray,
                 n_random: int = 200) -> dict:
    bank_path = BANKS_1J / f"{category}_seed{seed}_{layer}.npy"
    if not bank_path.exists():
        return {"status": "MISSING_BANK"}
    bank = np.load(bank_path).astype(np.float32)
    C, H, W = feat.shape
    X = feat.reshape(C, H * W).T.astype(np.float32)
    bn = np.einsum("nc,nc->n", bank, bank)
    d2 = -2.0 * (X @ bank.T) + bn[None, :] + np.einsum("pc,pc->p", X, X)[:, None]
    np.maximum(d2, 0.0, out=d2)
    nn = np.sqrt(d2).min(axis=1)
    gh, gw = GRID[layer]
    ov = mask_overlap_grid(mask, gh, gw).reshape(-1)
    k = max(1, int(0.05 * nn.size))
    top = np.argsort(-nn)[:k]
    overlap_top = float(ov[top].mean())
    rng = np.random.default_rng(80003)
    rnd = np.array([float(ov[rng.choice(nn.size, size=k, replace=False)].mean())
                    for _ in range(n_random)])
    return {"status": "OK", "n_patches": int(nn.size), "k_top": int(k),
            "mask_overlap_of_top5_anomaly_patches": overlap_top,
            "mask_overlap_random_mean": float(rnd.mean()),
            "mask_overlap_random_std": float(rnd.std(ddof=1)),
            "z": (float((overlap_top - rnd.mean()) / rnd.std(ddof=1))
                  if rnd.std(ddof=1) > 0 else None)}


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def run_cell(model, device, category: str, seed: int, out_root: Path, max_train, max_good,
             max_per_type, smoke: bool, log) -> dict:
    cell = out_root / "cells" / f"{category}_seed{seed}"
    cell.mkdir(parents=True, exist_ok=True)
    val_ids, train_ids = e1b.make_validation_split(category=category, seed=seed)
    if max_train:
        train_ids = train_ids[:max_train]
    good_paths = sorted((DATA_ROOT / category / "test" / "good").glob("*.png"))
    if max_good:
        good_paths = good_paths[:max_good]
    defect_paths = {dt: sorted((DATA_ROOT / category / "test" / dt).glob("*.png"))
                    for dt in e1b.discover_defect_types(category)}

    t0 = time.time()
    ns = normal_stats(model, device, train_ids, category, log)
    payload = {f"{ly}_{k}": v for ly in LAYERS for k, v in ns[ly].items()}
    for ly in LAYERS:
        payload[f"eps_{ly}"] = np.array([1e-3 * float(np.median(ns[ly]["sigma"]))])
    np.savez_compressed(cell / "normal_stats.npz", **payload)

    ill = illum_paired(model, device, good_paths, cell)
    log(f"    illum: {ill['n_images']} imgs x {ill['n_conditions_total']} conds; "
        f"S6 identity max|delta|={ill['s6_identity_max_abs_delta']}")

    dfc = defect_features(model, device, category, seed, defect_paths, cell, max_per_type, log)
    log(f"    defect: {dfc['n_defect_images']} imgs; fallback l2={dfc['n_fallback_layer2']} "
        f"l3={dfc['n_fallback_layer3']}")

    s5 = {}
    if smoke:
        for dt in FROZEN_PRIMARY.get(category, [])[:1]:
            for p in defect_paths[dt][:2]:
                img_t = e1b.preprocess_for_model(e1b.load_image_as_tensor(p), device).unsqueeze(0)
                l2, l3 = features_of(model, img_t)
                m = load_mask_256(DATA_ROOT / category / "ground_truth" / dt / f"{p.stem}_mask.png")
                for ly, t in (("layer2", l2), ("layer3", l3)):
                    s5[f"{dt}/{p.name}/{ly}"] = s5_alignment(category, seed, ly, t[0].cpu().numpy(), m)
    return {"category": category, "seed": seed,
            "n_train": len(train_ids), "n_good": len(good_paths),
            "n_defect_images": dfc["n_defect_images"],
            "n_fallback_layer2": dfc["n_fallback_layer2"], "n_fallback_layer3": dfc["n_fallback_layer3"],
            "illum": ill, "s5": s5, "s13": dfc["s13"],
            "wallclock_seconds": round(time.time() - t0, 1)}


def main() -> None:
    import os
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["smoke", "full"], default="smoke")
    ap.add_argument("--out-root", default="")
    ap.add_argument("--cells", default="", help="逗号分隔 cat:seed（覆盖 mode 全集）")
    ap.add_argument("--no-resume", action="store_true")
    args = ap.parse_args()

    if args.out_root:
        out_root = Path(args.out_root).resolve()
    else:
        out_root = ROOT / "results" / "experiment8b" / ("smoke" if args.mode == "smoke" else "extract")
    (out_root / "logs").mkdir(parents=True, exist_ok=True)
    (out_root / "logs" / "extract_pid.txt").write_text(str(os.getpid()))

    def log(msg: str) -> None:
        print(msg, flush=True)

    if args.cells:
        cells = [(t.split(":")[0], int(t.split(":")[1])) for t in args.cells.split(",") if t]
    elif args.mode == "smoke":
        cells = [("bottle", 0)]
    else:
        cells = [(c, s) for c in CATEGORIES for s in SEEDS]

    smoke = args.mode == "smoke"
    max_train = 6 if smoke else None
    max_good = 8 if smoke else None
    max_per_type = 3 if smoke else None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"[8B extract] mode={args.mode} device={device} cells={cells} out={out_root.relative_to(ROOT)}")
    model = build_model(device)

    results = []
    for cat, seed in cells:
        cell_dir = out_root / "cells" / f"{cat}_seed{seed}"
        if (not args.no_resume) and all((cell_dir / f).exists()
                                       for f in ("normal_stats.npz", "illum.npz", "defect.npz")):
            log(f"[{cat}:{seed}] resume skip")
            continue
        log(f"\n[{cat}:{seed}] start")
        results.append(run_cell(model, device, cat, seed, out_root, max_train, max_good,
                                max_per_type, smoke, log))
        (out_root / "sanity_extract.json").write_text(json.dumps(results, indent=2, ensure_ascii=False))
    log(f"\n[8B extract] done. cells processed={len(results)}")


if __name__ == "__main__":
    main()
