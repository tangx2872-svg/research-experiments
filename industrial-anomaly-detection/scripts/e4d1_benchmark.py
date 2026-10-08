#!/usr/bin/env python
"""E4-D1 — micro-benchmark BEFORE freezing the smoke scope.

Measures, with the FROZEN Original (alpha=0) configuration and IMAGE_SIZE=256:
  1. whether anomalib `Folder` accepts an explicit list of image FILES
  2. memory-bank fit time and peak VRAM vs number of bank images
  3. per-image scoring time (forward + NN search)
  4. implied safe bank size and total runtime for candidate scoring scopes

GPU only. Does NOT run B2 / X6c. Writes JSON to results/e4_d1_smoke/benchmark/.
"""
from __future__ import annotations

import json
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

import experiment1b_defect_sensitivity as e1b  # noqa: E402
from experiment5a_model import FAlphaDualLayerPatchcore  # noqa: E402
import e4d1_m2ad_loader as LD  # noqa: E402

from anomalib.engine import Engine  # noqa: E402

OUT = ROOT / "results" / "e4_d1_smoke" / "benchmark"


def vram_mb() -> tuple[float, float]:
    return (torch.cuda.memory_allocated() / 2**20, torch.cuda.memory_reserved() / 2**20)


def fit_bank(image_paths: list[str], seed: int, tag: str, batch_size: int = 16,
             val_paths: list[str] | None = None):
    """Fit an Original (alpha_l2=alpha_l3=0) PatchCore over the given normals."""
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()

    t0 = time.time()
    dm = LD.make_folder_datamodule(image_paths, val_paths=val_paths, batch_size=batch_size,
                                   num_workers=0, seed=seed, name=f"m2ad_{tag}")
    dm_time = time.time() - t0
    n_train = len(dm.train_data)

    model = FAlphaDualLayerPatchcore(
        backbone="wide_resnet50_2", layers=["layer2", "layer3"],
        coreset_sampling_ratio=0.1, num_neighbors=9,
        alpha_l2=0.0, alpha_l3=0.0, visualizer=False)
    engine = Engine(enable_progress_bar=False, logger=False, barebones=True,
                    num_sanity_val_steps=0, limit_val_batches=0,
                    default_root_dir=str(OUT / f"fit_{tag}"))
    t1 = time.time()
    engine.fit(model=model, datamodule=dm)
    fit_time = time.time() - t1

    tm = model.model
    tm.eval()
    coreset = int(tm.memory_bank.shape[0]) if hasattr(tm, "memory_bank") else -1
    alloc, reserved = vram_mb()
    return tm, {"tag": tag, "n_bank_images_requested": len(image_paths), "n_train_data": n_train,
                "datamodule_setup_s": round(dm_time, 2),
                "fit_s": round(fit_time, 2), "coreset_size": coreset,
                "vram_alloc_mb": round(alloc, 1), "vram_reserved_mb": round(reserved, 1),
                "peak_alloc_mb": round(torch.cuda.max_memory_allocated() / 2**20, 1),
                "peak_reserved_mb": round(torch.cuda.max_memory_reserved() / 2**20, 1)}


def score_images(tm, recs: list[dict], device, label: str) -> dict:
    e1b._move_model_to_device(tm, device)
    torch.cuda.synchronize()
    t0 = time.time()
    scores, maps_bytes = [], 0
    for r in recs:
        img = e1b.load_image_as_tensor(Path(r["image_path"]))
        s, amap = e1b.predict_one(tm, img, device)
        scores.append(s)
        maps_bytes += int(np.asarray(amap).nbytes) if amap is not None else 0
    torch.cuda.synchronize()
    dt = time.time() - t0
    return {"label": label, "n_images": len(recs), "seconds": round(dt, 3),
            "s_per_image": round(dt / max(1, len(recs)), 4),
            "images_per_s": round(len(recs) / max(dt, 1e-9), 3),
            "anomaly_map_MB_per_image": round(maps_bytes / max(1, len(recs)) / 2**20, 4),
            "score_mean": round(float(np.mean(scores)), 4) if scores else None,
            "score_std": round(float(np.std(scores)), 4) if scores else None}


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    assert torch.cuda.is_available(), "CUDA required for the benchmark"
    dev = torch.device("cuda")
    props = torch.cuda.get_device_properties(0)
    print(f"[gpu] {props.name}  {props.total_memory/2**30:.1f} GiB", flush=True)

    recs = LD.load_records()
    est = LD.estimate_scope(recs)
    print(f"[scope] bank={est['n_bank_images']} val={est['n_val_images']} "
          f"scoring={est['n_scoring_images']}", flush=True)

    bank = LD.select(recs, set(est["bank_specimens"]))
    test = LD.select(recs, views=None)
    # deterministic, result-independent benchmark order (sorted by path)
    bank = sorted(bank, key=lambda r: r["image_path"])
    test = sorted(test, key=lambda r: r["image_path"])

    results = {"gpu": {"name": props.name, "total_MiB": round(props.total_memory / 2**20, 1)},
               "scope": est, "fits": [], "scores": []}

    # ---- fit scaling: small banks only (measure MB/image and extrapolate)
    for n in (50, 100, 200):
        tm, info = fit_bank([r["image_path"] for r in bank[:n]], seed=0, tag=f"b{n}")
        info["coreset_per_bank_image"] = round(info["coreset_size"] / max(1, n), 3)
        results["fits"].append(info)
        print(f"[fit] n={n:4d} fit={info['fit_s']:7.2f}s coreset={info['coreset_size']:7d} "
              f"peak_alloc={info['peak_alloc_mb']:8.1f} MB peak_reserved={info['peak_reserved_mb']:8.1f} MB",
              flush=True)
        if n == 200:
            tm200 = tm
            # ---- scoring benchmark on 100 images (mixed good/NG, all views/illums)
            sample = test[::len(test) // 100][:100]
            si = score_images(tm200, sample, dev, "200bank_100imgs")
            results["scores"].append(si)
            print(f"[score] {si['n_images']} imgs in {si['seconds']}s -> "
                  f"{si['s_per_image']} s/img, {si['images_per_s']} img/s "
                  f"(amp {si['anomaly_map_MB_per_image']} MB/img)", flush=True)

    # ---- extrapolation
    f200 = next(f for f in results["fits"] if f["tag"] == "b200")
    peak_per_img = f200["peak_alloc_mb"] / 200.0
    free_mb = props.total_memory / 2**20 - 3_000       # keep >=3 GiB headroom
    max_bank = int(free_mb / peak_per_img)
    results["extrapolation"] = {
        "peak_alloc_MB_per_bank_image": round(peak_per_img, 3),
        "coreset_size_per_bank_image": f200["coreset_per_bank_image"],
        "max_bank_images_for_3GiB_headroom": max_bank,
        "note": "peak_alloc scales ~linearly with bank size (embedding_store + vstack)",
        "fit_s_at_1200_bank_est": round(f200["fit_s"] / 200.0 * 1200, 1),
    }
    sp = results["scores"][0]["s_per_image"] if results["scores"] else None
    if sp:
        for n_img, label in ((est["n_scoring_images"], "full_test_8400"),
                             (70 * 4 * 10, "views_000_090_180_270_2800"),
                             (70 * 10, "single_view_700")):
            results["extrapolation"][f"score_minutes_{label}"] = round(n_img * sp / 60.0, 1)

    (OUT / "benchmark.json").write_text(json.dumps(results, indent=2))
    print("\n===== EXTRAPOLATION =====", flush=True)
    print(json.dumps(results["extrapolation"], indent=2), flush=True)
    print(f"\n[written] {OUT/'benchmark.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
