"""Experiment 8B Phase B — score-level selection probe runner（复用 5A-H 完整评估路径）。

零改动复用 `experiment5a_h_runner.run_config`（bank / coreset 0.1 / kNN 9 / illumination /
scoring / per_image.csv 全部同一代码路径），只 monkey-patch fit，把 embedding 换成
channel-masked embedding（α=0）。

units：{cat}:{seed}:{MASK}，MASK 来自 results/experiment8b/probe/masks.json（由 analyze 冻结生成）。
输出：results/experiment8b/probe/raw/{cat}/seed_{seed}/{MASK}/{per_image.csv,info.json,module_meta.json}

用法（3 workers 示例，unit 列表互斥）：
  nohup python -u scripts/experiment8b_probe_runner.py --units "bottle:0:FULL25,..." --worker-tag w1 \
      > results/experiment8b/logs/probe_w1.log 2>&1 &
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "experiments" / "2026-09-30_illumination_sensitivity_exploration" / "falpha_patchcore"))

import experiment1b_defect_sensitivity as e1b  # noqa: E402
import experiment5a_h_runner as h  # noqa: E402
from experiment8b_model import MaskedPatchcore  # noqa: E402
from anomalib.data import MVTecAD  # noqa: E402
from anomalib.engine import Engine  # noqa: E402

PROBE_OUT = ROOT / "results" / "experiment8b" / "probe"
MASKS_JSON = PROBE_OUT / "masks.json"
LOGS_DIR = ROOT / "results" / "experiment8b" / "logs"

CAPTURED: dict = {}


def patched_fit_dual_model(alpha_l2, alpha_l3, category, seed, train_ids, logs_dir):
    """与 experiment5a_h_runner.fit_dual_model 同构，仅模型类换成 MaskedPatchcore(α=0)。"""
    name = CAPTURED["mask_name"]
    mask = CAPTURED["mask"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    datamodule = MVTecAD(root=str(h.DATA_ROOT), category=category,
                         train_batch_size=16, eval_batch_size=16, num_workers=0, seed=seed)
    datamodule.setup()
    keep_names = {n for n in train_ids}
    td = datamodule.train_data
    df = td._samples
    df = df[df["image_path"].apply(lambda p: str(Path(str(p)).name) in keep_names)].reset_index(drop=True)
    td._samples = df
    if hasattr(td, "_num_samples"):
        td._num_samples = len(df)
    model = MaskedPatchcore(backbone="wide_resnet50_2", layers=["layer2", "layer3"],
                            coreset_sampling_ratio=0.1, num_neighbors=9, visualizer=False,
                            mask_l2=mask.get("layer2"), mask_l3=mask.get("layer3"))
    engine = Engine(enable_progress_bar=False, logger=False, barebones=True,
                    default_root_dir=str(logs_dir))
    engine.fit(model=model, datamodule=datamodule)
    torch_model = model.model
    torch_model.eval()
    try:
        CAPTURED["embed_dim"] = int(torch_model.memory_bank.shape[1])
    except Exception:
        CAPTURED["embed_dim"] = -1
    return model, torch_model


h.fit_dual_model = patched_fit_dual_model


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--units", required=True, help="逗号分隔 cat:seed:MASK")
    ap.add_argument("--worker-tag", default="w0")
    args = ap.parse_args()

    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    (LOGS_DIR / f"probe_pid_{args.worker_tag}.txt").write_text(str(os.getpid()))
    masks = json.loads(MASKS_JSON.read_text())
    units = []
    for tok in args.units.split(","):
        c, s, name = tok.strip().split(":")
        units.append((c, int(s), name))
    print(f"[{args.worker_tag}] PID={os.getpid()} n_units={len(units)}", flush=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for category, seed, name in units:
        val_ids, train_ids = e1b.make_validation_split(category=category, seed=seed)
        assert len(val_ids) == 20
        mask = masks[f"{category}:{seed}"][name]
        defect_types = e1b.discover_defect_types(category)
        good_paths = sorted((h.DATA_ROOT / category / "test" / "good").glob("*.png"))
        defect_paths = {dt: sorted((h.DATA_ROOT / category / "test" / dt).glob("*.png"))
                        for dt in defect_types}
        exp_rows = h.expected_rows(category, good_paths, defect_paths)
        cfg = {"name": name, "alpha_l2": 0.0, "alpha_l3": 0.0}
        cfg_dir = PROBE_OUT / "raw" / category / f"seed_{seed}" / name
        if h.unit_done(cfg_dir, exp_rows):
            print(f"[{category}:{seed}:{name}] resume skip", flush=True)
            continue
        cfg_dir.mkdir(parents=True, exist_ok=True)
        lock = cfg_dir / "run.lock"
        if lock.exists():
            try:
                old = int(lock.read_text().strip())
                os.kill(old, 0)
                raise SystemExit(f"[{category}:{seed}:{name}] running PID {old} — conflict STOP")
            except (ProcessLookupError, ValueError, OSError):
                pass
        lock.write_text(str(os.getpid()))
        CAPTURED.clear()
        CAPTURED.update({"mask_name": name, "mask": mask, "embed_dim": -1})
        t0 = time.time()
        print(f"\n[{args.worker_tag}] unit {category}:{seed}:{name} "
              f"(l2={len(mask['layer2'])}, l3={len(mask['layer3'])}, rows={exp_rows})", flush=True)
        try:
            h.run_config(cfg, category, seed, val_ids, train_ids,
                         good_paths, defect_paths, device, cfg_dir)
        except Exception as exc:
            (cfg_dir / "info.json").write_text(json.dumps(
                {"status": f"FAILED: {type(exc).__name__}: {exc}"}))
            raise
        finally:
            lock.unlink(missing_ok=True)
        info = json.loads((cfg_dir / "info.json").read_text())
        (cfg_dir / "module_meta.json").write_text(json.dumps({
            "experiment": "8B", "mask_name": name, "category": category, "seed": seed,
            "n_channels_l2": len(mask["layer2"]), "n_channels_l3": len(mask["layer3"]),
            "embedding_dim": CAPTURED.get("embed_dim", -1),
            "alpha": 0.0, "coreset_size": info.get("coreset_size"),
            "tau_val": info.get("tau_val"), "runtime_seconds": info.get("runtime_seconds"),
            "wallclock_seconds": round(time.time() - t0, 2),
            "oracle_probe_only": True,
            "probe_note": "selection uses test defect masks (oracle); diagnostic evidence only",
        }, indent=2))
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    print(f"\n[{args.worker_tag}] all units done", flush=True)


if __name__ == "__main__":
    main()
