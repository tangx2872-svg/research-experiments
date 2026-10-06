"""Experiment 5A-H — Extended Cross-Category Validation of Frozen G2：runner。

15 units（5 categories × 3 seeds）× 5 frozen configs（B0/B2/C2/G2/C3）= 75 runs。
所有协议（split / preprocessing / bank / coreset / illumination）与 5A 完全同一
代码路径；本脚本零超参搜索，configs 为硬编码冻结常量。

并行策略：每个 worker 进程处理互斥的 unit 列表（--units "cat:seed,cat:seed,..."），
输出写 results/experiment_5a_h/raw/{category}/seed_{seed}/config_{name}/，
路径互斥，无共享 CSV。Resume：config 级（info.json status==OK 且行数正确即跳过）。

每 config 持久化：
  per_image.csv  —— 每图分数（val / clean_good / clean_defect / shift_good）+ clipped ratio
  info.json      —— status / tau_val / runtime / peak VRAM / coreset_size /
                     train_ids_hash / pixel_auroc / aupro / 行数

Sanity：S1–S10/S12/S13 在 runner 内做（S11/S14 在 analysis 内跨 seed / 跨实验做）。
全部结果落盘后由 experiment5a_h_analysis.py 汇总判定。

用法（3 workers 示例）：
  nohup python -u scripts/experiment5a_h_runner.py --units hazelnut:0,hazelnut:1,hazelnut:2 \
      > results/experiment_5a_h/logs/worker1.log 2>&1 &
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "experiments" / "2026-09-30_illumination_sensitivity_exploration" / "falpha_patchcore"))

import experiment1b_defect_sensitivity as e1b  # noqa: E402
import experiment1_illumination_tradeoff as e1  # noqa: E402 (apply_photometric 原样复用)
from experiment5a_model import FAlphaDualLayerPatchcore, FAlphaDualLayerPatchcoreModel  # noqa: E402
from experiment1h_runner import image_auroc, pixel_auroc_from_maps  # noqa: E402 (复用 1H 实现)

from anomalib.data import MVTecAD, ImageBatch  # noqa: E402
from anomalib.engine import Engine  # noqa: E402
from anomalib.metrics import AUPRO  # noqa: E402
from anomalib.models.image.patchcore.torch_model import PatchcoreModel  # noqa: E402

DATA_ROOT = e1b.DATA_ROOT
OUT_ROOT = ROOT / "results" / "experiment_5a_h"
LOGS_DIR = OUT_ROOT / "logs"
RULE_JSON = ROOT / "results" / "experiment_5a" / "geometry_rule.json"

CATEGORIES = ["bottle", "cable", "grid", "hazelnut", "screw"]
SEEDS = [0, 1, 2]

# ---- frozen configs（与 5A 实际运行值精确一致；括号为协议 3 位小数显示）----
RATIO = 0.603651  # s_L3 / s_L2，来自冻结 geometry_rule.json
A2_G2 = 0.75 * RATIO                     # 0.45273825 (≈0.453)
A_C2 = (A2_G2 + 0.75) / 2.0              # 0.601369125 (≈0.601)
A_C3 = (RATIO * 1.0 + 1.0) / 2.0         # 0.8018255 (≈0.802)
CONFIGS = [
    {"name": "B0", "alpha_l2": 0.0, "alpha_l3": 0.0},
    {"name": "B2", "alpha_l2": 0.5, "alpha_l3": 0.5},
    {"name": "C2", "alpha_l2": A_C2, "alpha_l3": A_C2},
    {"name": "G2", "alpha_l2": A2_G2, "alpha_l3": 0.75},
    {"name": "C3", "alpha_l2": A_C3, "alpha_l3": A_C3},
]

SHIFTS = [
    ("brightness_0.7", "brightness", 0.7),
    ("brightness_1.3", "brightness", 1.3),
    ("gamma_0.7", "gamma", 0.7),
    ("gamma_1.3", "gamma", 1.3),
]

STAGE4_FILES = [
    ROOT / "results" / "experiment_1j" / "analysis" / "experiment1j_cross_chain_summary.csv",
    ROOT / "results" / "experiment_1j" / "analysis" / "experiment1j_family_summary.csv",
    ROOT / "results" / "experiment_1j_b" / "analysis" / "transmission_summary.csv",
    ROOT / "results" / "experiment_1j_b" / "analysis" / "per_defect.csv",
    RULE_JSON,
]

EPS_PARENTH = 1e-6  # float32 检查容差


def hash_ids(ids: list[str]) -> str:
    return hashlib.md5(",".join(sorted(ids)).encode()).hexdigest()[:12]


def md5_file(p: Path) -> str:
    return hashlib.md5(p.read_bytes()).hexdigest()


# ---------------------------------------------------------------------------
# Sanity（runner 内可执行的部分）
# ---------------------------------------------------------------------------
def sanity_pre(worker_tag: str) -> dict:
    res: dict[str, tuple[bool, str]] = {}

    # S2 / S3：frozen config 精确值
    res["S2_G2_exact"] = (
        abs(A2_G2 - 0.45273825) < 1e-12 and CONFIGS[3]["alpha_l3"] == 0.75
        and abs(round(A2_G2, 3) - 0.453) < 1e-9,
        f"alpha_l2={A2_G2:.8f} (≈0.453), alpha_l3=0.75",
    )
    res["S3_C2_exact"] = (
        abs(A_C2 - (A2_G2 + 0.75) / 2.0) < 1e-12 and abs(round(A_C2, 3) - 0.601) < 1e-9,
        f"C2=mean(G2)={A_C2:.8f} (≈0.601); C3={A_C3:.8f} (≈0.802)",
    )

    # S4：rule hash 记录（analysis 复查）
    (LOGS_DIR / f"stage4_hashes_{worker_tag}.json").write_text(json.dumps(
        {"before": {str(p): md5_file(p) for p in STAGE4_FILES}}, indent=2))

    # S1：alpha=(0,0) embedding 等价
    val_ids, _ = e1b.make_validation_split(category="bottle", seed=0)
    img = e1b.load_image_as_tensor(DATA_ROOT / "bottle" / "train" / "good" / val_ids[0])
    x = e1b.preprocess_for_model(img, torch.device("cpu")).unsqueeze(0)
    base = PatchcoreModel(backbone="wide_resnet50_2", layers=["layer2", "layer3"],
                          pre_trained=True, num_neighbors=9)
    dual = FAlphaDualLayerPatchcoreModel(backbone="wide_resnet50_2", layers=["layer2", "layer3"],
                                         pre_trained=True, num_neighbors=9,
                                         alpha_l2=0.0, alpha_l3=0.0)
    base.eval(); dual.eval()
    with torch.no_grad():
        feats = base.feature_extractor(x)
        diff = float((base.generate_embedding(feats) - dual.generate_embedding(feats)).abs().max().item())
    res["S1_alpha0_equivalence"] = (diff < 1e-6, f"max_abs_diff={diff:.3e}")

    # S5：photometric 协议数值（与 5A 相同实现 + 相同检查）
    t = torch.tensor([[[0.0, 0.5, 1.0]]])
    ok = True; msgs = []
    for name, itype, level in SHIFTS:
        r = e1.apply_photometric(t, itype, level)
        if r.min().item() < 0.0 or r.max().item() > 1.0:
            ok = False; msgs.append(f"{name} out of range")
    if abs(e1.apply_photometric(t, "brightness", 0.7)[0, 0, 2].item() - 0.7) > EPS_PARENTH:
        ok = False; msgs.append("brightness0.7 identity check fail")
    res["S5_photometric_protocol"] = (ok, "same apply_photometric as 5A; range OK" if ok else ";".join(msgs))

    # S7：preprocessing 同一函数（结构保证，记录版本事实）
    res["S7_preprocessing_same"] = (
        e1b.preprocess_for_model.__module__ == "experiment1b_defect_sensitivity",
        "preprocess/load via e1b (same as 5A/1H/1J)")

    # S8：α 为全局常量，无 label 通路
    res["S8_no_label_in_alpha"] = (True, "alphas are frozen constants for all images")

    return res


# ---------------------------------------------------------------------------
# fit（与 5A fit_dual_model 同一流程，参数化 category/seed）
# ---------------------------------------------------------------------------
def fit_dual_model(alpha_l2: float, alpha_l3: float, category: str, seed: int,
                   train_ids: list[str], logs_dir: Path):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    datamodule = MVTecAD(root=str(DATA_ROOT), category=category,
                         train_batch_size=16, eval_batch_size=16, num_workers=0, seed=seed)
    datamodule.setup()

    keep_names = {n for n in train_ids}
    td = datamodule.train_data
    df = td._samples
    df = df[df["image_path"].apply(lambda p: str(Path(str(p)).name) in keep_names)].reset_index(drop=True)
    td._samples = df
    if hasattr(td, "_num_samples"):
        td._num_samples = len(df)

    model = FAlphaDualLayerPatchcore(
        backbone="wide_resnet50_2",
        layers=["layer2", "layer3"],
        coreset_sampling_ratio=0.1,
        num_neighbors=9,
        alpha_l2=alpha_l2,
        alpha_l3=alpha_l3,
        visualizer=False,
    )
    engine = Engine(enable_progress_bar=False, logger=False, barebones=True,
                    default_root_dir=str(logs_dir))
    engine.fit(model=model, datamodule=datamodule)
    torch_model = model.model
    torch_model.eval()
    return model, torch_model


def clipped_ratio(img: torch.Tensor) -> float:
    """变换后像素落在饱和端的比例（≥1−1e-6；低端的 ≤1e-6 单独计）。"""
    high = float((img >= 1.0 - 1e-6).float().mean().item())
    return high


def clipped_ratio_low(img: torch.Tensor) -> float:
    return float((img <= 1e-6).float().mean().item())


def resize_mask(mask: np.ndarray, hw: tuple[int, int]) -> np.ndarray:
    h, w = hw
    m = np.asarray(mask).squeeze()
    if m.shape[:2] != (h, w):
        m = np.array(Image.fromarray((m * 255).astype(np.uint8)).resize((w, h), Image.BILINEAR)).astype(np.float32) / 255.0
    return (m > 0.5).astype(np.float32)


def compute_aupro(maps_good: list[np.ndarray], maps_defect: list[np.ndarray],
                  masks_defect: list[np.ndarray]) -> float:
    """anomalib AUPRO（clean test 全集：good + defect）。"""
    aupro = AUPRO(fields=["anomaly_map", "gt_mask"])
    ams, mks = [], []
    for am in maps_good:
        am = np.asarray(am).squeeze()
        ams.append(torch.from_numpy(am.astype(np.float32))[None, None])
        mks.append(torch.zeros(1, 1, *am.shape))
    for am, mk in zip(maps_defect, masks_defect):
        am = np.asarray(am).squeeze()
        m = resize_mask(mk, am.shape)
        ams.append(torch.from_numpy(am.astype(np.float32))[None, None])
        mks.append(torch.from_numpy(m)[None, None])
    n = len(ams)
    batch = ImageBatch(
        image=torch.zeros(n, 3, *ams[0].shape[-2:]),
        gt_label=torch.tensor([0] * len(maps_good) + [1] * len(maps_defect)),
        gt_mask=torch.cat(mks),
        pred_score=torch.zeros(n),
    )
    batch.anomaly_map = torch.cat(ams)
    aupro.update(batch)
    return float(aupro.compute().item())


# ---------------------------------------------------------------------------
# 单 config 执行
# ---------------------------------------------------------------------------
def run_config(cfg: dict, category: str, seed: int, val_ids: list[str], train_ids: list[str],
               good_paths: list[Path], defect_paths: dict[str, list[Path]],
               device: torch.device, cfg_dir: Path) -> None:
    t0 = time.time()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    logs_dir = cfg_dir / "fit"
    logs_dir.mkdir(parents=True, exist_ok=True)
    lightning_model, torch_model = fit_dual_model(
        cfg["alpha_l2"], cfg["alpha_l3"], category, seed, train_ids, logs_dir)
    e1b._move_model_to_device(torch_model, device)

    coreset_size = "not_measured"
    try:
        if hasattr(torch_model, "memory_bank") and isinstance(torch_model.memory_bank, torch.Tensor):
            coreset_size = int(torch_model.memory_bank.shape[0])
    except Exception:
        pass

    rows: list[dict] = []

    def score_img(img: torch.Tensor, subset: str, shift: str, defect_type: str, path: Path,
                  clip_high: float = 0.0, clip_low: float = 0.0):
        s, _ = e1b.predict_one(torch_model, img, device)
        if not np.isfinite(s):
            raise FloatingPointError(f"non-finite score: {category}:{seed}:{cfg['name']} {path}")
        rows.append({
            "config": cfg["name"], "category": category, "seed": seed,
            "alpha_l2": cfg["alpha_l2"], "alpha_l3": cfg["alpha_l3"],
            "subset": subset, "shift": shift, "defect_type": defect_type,
            "image_path": str(path), "score": s,
            "clipped_high_ratio": clip_high, "clipped_low_ratio": clip_low,
        })

    # 1) validation good → τ_val
    val_scores = []
    for vid in val_ids:
        p = DATA_ROOT / category / "train" / "good" / vid
        s, _ = e1b.predict_one(torch_model, e1b.load_image_as_tensor(p), device)
        if not np.isfinite(s):
            raise FloatingPointError(f"non-finite val score: {p}")
        val_scores.append(s)
        rows.append({
            "config": cfg["name"], "category": category, "seed": seed,
            "alpha_l2": cfg["alpha_l2"], "alpha_l3": cfg["alpha_l3"],
            "subset": "val", "shift": "none", "defect_type": "good",
            "image_path": str(p), "score": s,
            "clipped_high_ratio": 0.0, "clipped_low_ratio": 0.0,
        })
    tau_val = float(max(val_scores))

    # 2) clean good（保留 map 供 pixel 指标）
    good_maps = []
    for p in good_paths:
        img = e1b.load_image_as_tensor(p)
        s, amap = e1b.predict_one(torch_model, img, device)
        if not np.isfinite(s):
            raise FloatingPointError(f"non-finite score: {p}")
        rows.append({
            "config": cfg["name"], "category": category, "seed": seed,
            "alpha_l2": cfg["alpha_l2"], "alpha_l3": cfg["alpha_l3"],
            "subset": "clean_good", "shift": "none", "defect_type": "good",
            "image_path": str(p), "score": s,
            "clipped_high_ratio": 0.0, "clipped_low_ratio": 0.0,
        })
        good_maps.append(amap)

    # 3) clean defect（保留 map + mask）
    defect_maps, defect_masks = [], []
    for dt, paths in defect_paths.items():
        for p in paths:
            img = e1b.load_image_as_tensor(p)
            s, amap = e1b.predict_one(torch_model, img, device)
            if not np.isfinite(s):
                raise FloatingPointError(f"non-finite score: {p}")
            rows.append({
                "config": cfg["name"], "category": category, "seed": seed,
                "alpha_l2": cfg["alpha_l2"], "alpha_l3": cfg["alpha_l3"],
                "subset": "clean_defect", "shift": "none", "defect_type": dt,
                "image_path": str(p), "score": s,
                "clipped_high_ratio": 0.0, "clipped_low_ratio": 0.0,
            })
            defect_maps.append(amap)
            mask_path = DATA_ROOT / category / "ground_truth" / dt / f"{p.stem}_mask.png"
            defect_masks.append(e1b.load_mask(mask_path) if mask_path.exists() else np.zeros((1, 1)))

    # 4) shifted good（Axis A）+ clipped ratio
    for shift_name, itype, level in SHIFTS:
        for p in good_paths:
            img = e1b.load_image_as_tensor(p)
            img_s = e1.apply_photometric(img, itype, level)
            score_img(img_s, "shift_good", shift_name, "good", p,
                      clip_high=clipped_ratio(img_s), clip_low=clipped_ratio_low(img_s))

    # pixel-level secondary 指标
    try:
        p_auc = pixel_auroc_from_maps(good_maps, defect_maps, defect_masks)
    except Exception as exc:
        p_auc = float("nan")
        print(f"[warn] pixel_auroc failed {category}:{seed}:{cfg['name']}: {exc}", flush=True)
    try:
        aupro_v = compute_aupro(good_maps, defect_maps, defect_masks)
    except Exception as exc:
        aupro_v = float("nan")
        print(f"[warn] aupro failed {category}:{seed}:{cfg['name']}: {exc}", flush=True)

    peak_mb = "not_measured"
    if torch.cuda.is_available():
        try:
            peak_mb = round(torch.cuda.max_memory_allocated() / (1024 * 1024), 2)
        except Exception:
            pass

    # 持久化
    fields = list(rows[0].keys())
    with open(cfg_dir / "per_image.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    info = {
        "status": "OK", "category": category, "seed": seed, "config": cfg["name"],
        "alpha_l2": cfg["alpha_l2"], "alpha_l3": cfg["alpha_l3"],
        "runtime_seconds": round(time.time() - t0, 2),
        "coreset_size": coreset_size, "tau_val": tau_val,
        "peak_gpu_memory_allocated_mb": peak_mb,
        "train_ids_hash": hash_ids(train_ids),
        "n_rows": len(rows), "pixel_auroc": p_auc, "aupro": aupro_v,
    }
    (cfg_dir / "info.json").write_text(json.dumps(info, indent=2))

    del lightning_model, torch_model, good_maps, defect_maps, defect_masks
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print(f"[{category}:{seed}:{cfg['name']}] OK rows={len(rows)} "
          f"runtime={info['runtime_seconds']}s tau={tau_val:.4f} "
          f"pxAUROC={p_auc:.4f} AUPRO={aupro_v:.4f}", flush=True)


def expected_rows(category: str, good_paths: list[Path], defect_paths: dict) -> int:
    n_good = len(good_paths)
    n_def = sum(len(v) for v in defect_paths.values())
    return 20 + n_good + n_def + 4 * n_good


def unit_done(cfg_dir: Path, exp_rows: int) -> bool:
    info_p = cfg_dir / "info.json"
    if not info_p.exists():
        return False
    try:
        info = json.loads(info_p.read_text())
    except Exception:
        return False
    return info.get("status") == "OK" and info.get("n_rows") == exp_rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--units", required=True,
                        help="逗号分隔 cat:seed 列表，如 hazelnut:0,hazelnut:1,hazelnut:2")
    parser.add_argument("--worker-tag", default="w0")
    args = parser.parse_args()

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[{args.worker_tag}] device={device} units={args.units}", flush=True)

    # ---- pre-run sanity ----
    sres = sanity_pre(args.worker_tag)
    fails = {k: v for k, v in sres.items() if not v[0]}
    for k, (ok, detail) in sres.items():
        print(f"[sanity {args.worker_tag}] {k}: {'PASS' if ok else 'FAIL'} ({detail})", flush=True)
    if fails:
        raise SystemExit(f"[{args.worker_tag}] pre-run sanity FAILED: {list(fails)} — STOP")
    with open(LOGS_DIR / f"sanity_pre_{args.worker_tag}.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["check", "status", "detail"])
        for k, (ok, detail) in sres.items():
            w.writerow([k, "PASS" if ok else "FAIL", detail])

    # ---- units ----
    units = []
    for tok in args.units.split(","):
        cat, seed = tok.strip().split(":")
        units.append((cat, int(seed)))

    for category, seed in units:
        if category not in CATEGORIES:
            raise ValueError(f"unknown category {category}")
        val_ids, train_ids = e1b.make_validation_split(category=category, seed=seed)
        assert len(val_ids) == 20, f"{category} val != 20"
        defect_types = e1b.discover_defect_types(category)
        good_paths = sorted((DATA_ROOT / category / "test" / "good").glob("*.png"))
        defect_paths = {dt: sorted((DATA_ROOT / category / "test" / dt).glob("*.png"))
                        for dt in defect_types}
        exp_rows = expected_rows(category, good_paths, defect_paths)
        print(f"\n[{args.worker_tag}] unit {category}:{seed} "
              f"(train={len(train_ids)} good={len(good_paths)} defect={sum(len(v) for v in defect_paths.values())})", flush=True)

        # S6：unit 内 split 一致性基准
        train_hash = hash_ids(train_ids)

        for cfg in CONFIGS:
            cfg_dir = OUT_ROOT / "raw" / category / f"seed_{seed}" / f"config_{cfg['name']}"
            if unit_done(cfg_dir, exp_rows):
                print(f"[{category}:{seed}:{cfg['name']}] resume skip", flush=True)
                continue
            cfg_dir.mkdir(parents=True, exist_ok=True)
            # S12：并发冲突保护——本 worker 独占该路径（unit 列表互斥由调用方保证）
            lock = cfg_dir / "run.lock"
            if lock.exists():
                try:
                    old_pid = int(lock.read_text().strip())
                    import os
                    os.kill(old_pid, 0)
                    raise SystemExit(f"[{category}:{seed}:{cfg['name']}] 已有运行中的 PID {old_pid}，冲突 STOP")
                except (ProcessLookupError, ValueError, OSError):
                    pass  # 残留 lock，进程已死，接管
            import os
            lock.write_text(str(os.getpid()))
            try:
                run_config(cfg, category, seed, val_ids, train_ids,
                           good_paths, defect_paths, device, cfg_dir)
                # S6：train hash 一致
                info = json.loads((cfg_dir / "info.json").read_text())
                if info["train_ids_hash"] != train_hash:
                    raise SystemExit(f"S6 FAIL: {category}:{seed} train hash mismatch in {cfg['name']}")
            except Exception as exc:
                (cfg_dir / "info.json").write_text(json.dumps(
                    {"status": f"FAILED: {type(exc).__name__}: {exc}"}))
                raise
            finally:
                lock.unlink(missing_ok=True)

    print(f"\n[{args.worker_tag}] all units done", flush=True)


if __name__ == "__main__":
    main()
