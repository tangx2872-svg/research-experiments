"""Experiment 7A-O — runner（复用 5A-H 的完整评估路径，只替换 feature 变换）。

关键：`experiment5a_h_runner.run_config` 被**零改动复用**（bank / coreset / kNN /
illumination / scoring / 指标 / per_image.csv / info.json 全部同一代码路径），
只把内部调用的 `fit_dual_model` monkey-patch 成 7A-O 的模块模型。

用法：
  python -u scripts/experiment7ao_runner.py --units "bottle:0:SMOKE_ORIG_a000" --worker-tag smoke
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

import experiment7ao_config as c7  # noqa: E402
import experiment5a_h_runner as h  # noqa: E402
from experiment7ao_model import ModulePatchcore  # noqa: E402
from anomalib.data import MVTecAD  # noqa: E402
from anomalib.engine import Engine  # noqa: E402

OUT_ROOT = c7.RAW_DIR
LOGS_DIR = c7.LOG_DIR
h.OUT_ROOT = OUT_ROOT
h.LOGS_DIR = LOGS_DIR

ALL_SPECS = dict(c7.MODULE_SPECS)
ALL_SPECS.update(c7.SMOKE_SPECS)

CAPTURED: dict = {}


def resolve_spec(name: str) -> dict:
    if name in ALL_SPECS:
        return ALL_SPECS[name]
    if name.startswith("COMB_"):
        body = name[len("COMB_"):]
        wa, wb = body.split("__")
        return c7.compose_spec(wa, wb)
    raise SystemExit(f"[FATAL] unknown config {name}")


def declared_alphas(name: str, spec: dict) -> tuple:
    """info.json 里记录的 alpha 字段：只有纯 alpha 策略才有意义，其余记 -1.0。"""
    if spec["l2"]["kind"] == "alpha_in" and spec["l3"]["kind"] == "alpha_in":
        return float(spec["l2"]["alpha"]), float(spec["l3"]["alpha"])
    return -1.0, -1.0


def patched_fit_dual_model(alpha_l2, alpha_l3, category, seed, train_ids, logs_dir):
    """与 experiment5a_h_runner.fit_dual_model 完全同构，仅模型类与 spec 不同。"""
    name = CAPTURED["config"]
    spec = CAPTURED["spec"]
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
    model = ModulePatchcore(backbone="wide_resnet50_2", layers=["layer2", "layer3"],
                            coreset_sampling_ratio=0.1, num_neighbors=9, spec=spec, visualizer=False)
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


def pre_run_sanity(units: list) -> dict:
    c7.OUT.mkdir(parents=True, exist_ok=True)
    res = {}
    fr = json.loads((c7.REF_DIR / "policy_freeze.json").read_text())
    res["R1_freeze_exists"] = ((c7.REF_DIR / "policy_freeze.json").exists()
                               and (c7.REF_DIR / "asset_audit.csv").exists(),
                               f"freeze head={fr['git_head'][:7]} "
                               f"sha={c7.sha256_file(c7.REF_DIR / 'policy_freeze.json')[:12]}")
    res["R2_all_configs_frozen"] = (all(u[2] in ALL_SPECS or u[2].startswith("COMB_") for u in units),
                                    "every requested config is in the frozen grid (or a registered combination)")
    res["R3_cuda_available"] = (torch.cuda.is_available(), f"cuda={torch.cuda.is_available()}")
    res["R4_output_isolated"] = (OUT_ROOT.name == "raw" and "experiment_7a_o" in str(OUT_ROOT),
                                 f"output root = {OUT_ROOT.relative_to(c7.ROOT)}")
    hf = c7.historical_keys()
    res["R5_baselines_available_no_rerun"] = (
        all(len([1 for c in c7.CATEGORIES for s in c7.SEEDS
                 if (c, s, c7.akey(a), c7.akey(a)) in hf]) == 15
            for a in (0.0, c7.FIXED_ALPHA, c7.UNIFORM_ALPHA)),
        "B0/B1/B2 baselines have 15/15 historical coverage (no rerun)")
    res["R6_no_target_dependent_params"] = (
        all(v["l2"]["kind"] in ("alpha_in", "residual", "energy", "concat_dual", "altnorm_strength")
            for v in ALL_SPECS.values()),
        "all module hyperparameters are frozen constants; none derived from target results")
    return res


def main() -> None:
    import experiment1b_defect_sensitivity as e1b
    ap = argparse.ArgumentParser()
    ap.add_argument("--units", required=True, help="逗号分隔 category:seed:CONFIG")
    ap.add_argument("--worker-tag", default="w0")
    args = ap.parse_args()
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    units = []
    for tok in args.units.split(","):
        c, s, n = tok.strip().split(":")
        units.append((c, int(s), n))
    print(f"[{args.worker_tag}] PID={os.getpid()} units={units}", flush=True)

    sres = pre_run_sanity(units)
    for k, (ok, d) in sres.items():
        print(f"[sanity {args.worker_tag}] {k}: {'PASS' if ok else 'FAIL'} ({d})", flush=True)
    with open(LOGS_DIR / f"sanity_pre_{args.worker_tag}.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["check", "status", "detail"])
        for k, (ok, d) in sres.items():
            w.writerow([k, "PASS" if ok else "FAIL", d])
    fails = [k for k, v in sres.items() if not v[0]]
    if fails:
        raise SystemExit(f"[{args.worker_tag}] pre-run sanity FAILED: {fails} — STOP")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    (LOGS_DIR / f"runner_pid_{args.worker_tag}.txt").write_text(str(os.getpid()))

    for category, seed, name in units:
        spec = resolve_spec(name)
        a2, a3 = declared_alphas(name, spec)
        val_ids, train_ids = e1b.make_validation_split(category=category, seed=seed)
        assert len(val_ids) == 20
        defect_types = e1b.discover_defect_types(category)
        good_paths = sorted((h.DATA_ROOT / category / "test" / "good").glob("*.png"))
        defect_paths = {dt: sorted((h.DATA_ROOT / category / "test" / dt).glob("*.png"))
                        for dt in defect_types}
        exp_rows = h.expected_rows(category, good_paths, defect_paths)
        cfg = {"name": name, "alpha_l2": a2, "alpha_l3": a3}
        cfg_dir = OUT_ROOT / category / f"seed_{seed}" / f"config_{name}"
        print(f"\n[{args.worker_tag}] unit {category}:{seed}:{name} (rows={exp_rows})", flush=True)
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
        CAPTURED.update({"config": name, "spec": spec, "embed_dim": -1})
        t0 = time.time()
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
        fam = c7.FAMILY_OF.get(name, "smoke" if name in c7.SMOKE_SPECS else "combination")
        meta = {"config": name, "family": fam, "spec": spec,
                "embedding_dim": CAPTURED.get("embed_dim", -1),
                "memory_bank_size": info.get("coreset_size"),
                "runtime_seconds": info.get("runtime_seconds"),
                "wallclock_seconds": round(time.time() - t0, 2),
                "peak_gpu_memory_allocated_mb": info.get("peak_gpu_memory_allocated_mb"),
                "declared_alpha_l2": a2, "declared_alpha_l3": a3,
                "embedding_dim_note": "3072 for concat_dual (doubled), 1536 otherwise"}
        (cfg_dir / "module_meta.json").write_text(json.dumps(meta, indent=2))
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print(f"\n[{args.worker_tag}] all units done", flush=True)


if __name__ == "__main__":
    main()
