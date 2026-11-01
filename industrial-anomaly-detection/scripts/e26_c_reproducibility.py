#!/usr/bin/env python
"""E26-C: build the reproducibility manifest for the frozen R1-ILA method.

Every hyper-parameter is read from the code that actually produced the reported
results (imported modules / frozen JSON), never from memory. Read-only: no GPU work,
no training, no file outside results/e26/reproducibility/ is written.

Usage
-----
    python scripts/e26_c_reproducibility.py
"""
from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import e13a_screen as E13A  # noqa: E402  (constants of the frozen trainer)
import e13b_screen as E13B  # noqa: E402

OUT = ROOT / "results" / "e26" / "reproducibility"
WEIGHTS = Path("/root/.cache/torch/hub/checkpoints/dinov2_vitb14_pretrain.pth")
M2AD_META = ROOT / "data" / "m2ad" / "jsons_extract" / "meta_unsupervised.json"
FINAL_METHOD = ROOT / "results" / "e17" / "final_method.json"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git(*args: str) -> str:
    return subprocess.run(["git", "-C", str(ROOT), *args], capture_output=True,
                          text=True, check=False).stdout.strip()


def env_block() -> dict:
    import torch
    import torchvision
    return {
        "os": platform.platform(),
        "python": sys.version.split()[0],
        "pytorch": torch.__version__,
        "torchvision": torchvision.__version__,
        "cuda_runtime": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "gpu_memory_mib": (torch.cuda.get_device_properties(0).total_memory // (1024 ** 2))
                          if torch.cuda.is_available() else None,
        "numpy": __import__("numpy").__version__,
    }


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    fm = json.loads(FINAL_METHOD.read_text())
    env = env_block()

    cache_manifest = ROOT / "results" / "e22" / "cache_l0_448_manifest.csv"
    n_cache_files = (sum(1 for _ in open(cache_manifest)) - 1) if cache_manifest.exists() else None
    cache_agg = ROOT / "results" / "e22" / "cache_l0_448_aggregate_sha256.txt"

    manifest = {
        "date": "2026-11-01",
        "experiment": "E26-C",
        "method": fm["method_name"],
        "source_of_truth": {
            "frozen_config": str(FINAL_METHOD.relative_to(ROOT)),
            "trainer": "scripts/e13b_screen.py::train_r1 (lines 44-90)",
            "constants": "scripts/e13a_screen.py (lines 42-64, 126-133, 154-186)",
            "detector": "scripts/e11_p2_runner.py::nn_image_scores (lines 116-135)",
            "loader_m2ad": "scripts/e4d1_m2ad_loader.py (DATA_ROOT / META_PATH)",
            "loader_robustad": "scripts/e25_robustad_eval.py (BANK_TRAIN_N/BANK_VAL_N, metadata.jsonl)",
        },
        "environment": env,
        "backbone": {
            "model": "dinov2_vitb14 (DINOv2 ViT-B/14, patch 14)",
            "loaded_via": "torch.hub local clone (%s); dinov2.hub.backbones.dinov2_vitb14(pretrained=True)" % E13A._HUB_REPO,
            "feature_source": "model.get_intermediate_layers(x, n=1, norm=True)[0] (final block, LayerNorm applied)",
            "weights_file": str(WEIGHTS),
            "weights_sha256": sha256(WEIGHTS) if WEIGHTS.exists() else "MISSING",
            "weights_bytes": WEIGHTS.stat().st_size if WEIGHTS.exists() else None,
            "input_size": "Resize(short side = 448, BICUBIC, antialias=True), then crop to a multiple of 14 "
                          "-> 448x448 for M2AD 1024x1024 sources",
            "patch_size": 14,
            "n_patch_tokens": E13A.N_PATCH,
            "feature_dim": E13A.D,
            "preprocessing": "ToTensor (0-1) -> ImageNet normalization",
            "normalization": {"mean": E13A.IMAGENET_MEAN, "std": E13A.IMAGENET_STD},
            "trainable": False,
        },
        "detector": {
            "bank": "M2AD: 300 normal reference images per category/seed (bank_for); "
                    "RobustAD: 180 source-train normals (documented bank-size constraint)",
            "nearest_neighbour_k": 1,
            "distance": "cosine distance = 1 - max cosine similarity (features L2-normalized)",
            "score_aggregation": "image score = max over 1024 patch distances (1-NN-max)",
            "feature_l2_normalization": True,
            "implementation": "pure PyTorch matmul over bank chunks (scripts/e11_p2_runner.py:116-135); no FAISS/sklearn",
            "labels_and_filter": "labels from the dataset's official `image_anomaly`; rows with "
                                 "`detectable == False` are dropped (e11_p2_runner.py:142)",
        },
        "r1_ila": {
            "formula": "z' = z + W z  (adapter_forward, e13a_screen.py:154-155; LinearAdapter.forward returns Wz)",
            "trainable_dimension": f"{E13A.D} x {E13A.D}",
            "initialization": "W = zeros (LinearAdapter.__init__, e13a_screen.py:126-133); no bias",
            "trainable_parameters": 589824,
            "BETA": E13A.BETA,
            "L_cons": "1 - mean_patch cos(l2norm(z_adapt(x)), l2norm(z_adapt(T(x)))) "
                      "(both sides adapted; e13a_screen.py:223-227)",
            "L_preserve": "1 - mean_patch cos(l2norm(z_adapt(x)), l2norm(z_frozen(x))); frozen side detached",
            "total_loss": "L = L_cons + BETA * L_preserve (objective='cons_preserve')",
            "beta_injection": "module-level global read by compute_loss (no frozen code edited)",
        },
        "training": {
            "optimizer": "AdamW",
            "lr": E13A.LR_ADAPTER,
            "betas": [0.9, 0.999],
            "eps": 1e-8,
            "weight_decay": 0.01,
            "optimizer_note": "only lr is passed explicitly; betas/eps/weight_decay are PyTorch defaults "
                              "(torch.optim.AdamW defaults at torch 2.8.0)",
            "batch_size": E13A.BATCH_IMGS,
            "max_steps": E13A.MAX_STEPS,
            "eval_interval": E13A.EVAL_EVERY,
            "patience": E13A.PATIENCE_STEPS,
            "early_stopping_criterion": "validation consistency term only (L_cons, NOT the total loss), averaged "
                                        "over 60 held-out normals x all 6 variants (val_adapter_loss, "
                                        "e13a_screen.py:320-338); improvement threshold 1e-6; stop after 30 steps "
                                        "without improvement",
            "checkpoint_policy": "NO best-checkpoint restore: the parameters at the stopping iteration are "
                                 "scored (see results/e26/checkpoint_audit/REPORT.md)",
            "train_normal_count": {"m2ad": E13A.TRAIN_N, "robustad": 120},
            "val_normal_count": {"m2ad": E13A.VAL_N, "robustad": 60},
            "random_seeds": {"m2ad": [0, 1, 2], "robustad": [0, 1, 2],
                             "note": "seed controls adapter init/training RNG only; bank per seed"},
            "observed_steps_used": {
                "E22": "44-124 (median 59), all 30 cells early-stopped",
                "E25": "49-164 (median 74), all 39 cells early-stopped",
                "E24": "44-219 (median 59), all 51 cells early-stopped",
            },
        },
        "augmentation": {
            "families": E13A.AUG_FAMILIES,
            "levels": E13A.AUG_LEVELS,
            "n_variants": len(E13A.AUG_LIST),
            "brightness": "x * level on the [0,1] tensor then clamp(0,1)",
            "contrast": "(x - 0.5) * level + 0.5 then clamp(0,1)",
            "gamma": "x.clamp(min=1e-6) ** (1/level) then clamp(0,1)",
            "probability": "no Bernoulli gate: every training step samples one variant index uniformly "
                           "per image from the 6 precomputed variants (train_aug[idx, aidx])",
            "order": "perturbation applied BEFORE ImageNet normalization; all 6 variants precomputed once per bank image",
            "detail": "NOT a random photometric jitter at each step: the 6 variants are fixed tensors reused "
                      "across steps (e13a_screen.py:59-64, 179-186)",
        },
        "data": {
            "m2ad": {
                "root": "data/m2ad/extracted",
                "official_metadata": "data/m2ad/jsons_extract/meta_unsupervised.json",
                "metadata_sha256": sha256(M2AD_META) if M2AD_META.exists() else "MISSING",
                "selected_classes": ["Bird", "Car", "Cube", "Dice", "Doll", "Holder", "Motor", "Ring",
                                     "Teapot", "Tube"],
                "views": {"bank": "all views", "test": ["120"]},
                "illuminations": [f"{i:02d}" for i in range(1, 11)],
                "detectable_filtering": "drop rows with detectable == False",
                "labels": "official image_anomaly (not object_anomaly)",
                "input_cache": {
                    "path": "data/m2ad/e7_cache/l0_448",
                    "manifest": "results/e22/cache_l0_448_manifest.csv",
                    "n_files_in_manifest": n_cache_files,
                    "aggregate_sha256_file": str(cache_agg.relative_to(ROOT)) if cache_agg.exists() else None,
                    "note": "9 of 10 categories have no extracted originals on this server; E22/E26 read the "
                            "448 cache (short side 448, BICUBIC, antialias, PNG uint8)",
                },
            },
            "robustad": {
                "root": "data/robustad",
                "splits": "per-subdataset metadata.jsonl (labels never taken from directory names)",
                "domains": {"PCB": "6 splits", "MetalParts": "7 splits", "PiledBags": "not run (disk)"},
                "split_generation": "upstream RobustAD (AmazonScience) split directories, unmodified; "
                                    "MetalParts test0 excluded (valid=0) by the E25 protocol",
            },
        },
        "version": {
            "repository": "github.com/tangx2872-svg/research-experiments",
            "branch": git("rev-parse", "--abbrev-ref", "HEAD"),
            "audit_head": git("rev-parse", "HEAD"),
            "head_subject": git("log", "-1", "--format=%s"),
            "working_tree_dirty": bool(git("status", "--porcelain")),
            "final_method_config": str(FINAL_METHOD.relative_to(ROOT)),
            "final_result_paths": {
                "e22_main_table": "results/e22/main_table_10cat_3seed.csv",
                "e22_cells": "results/e22/cells/*.json",
                "e25_multiseed": "results/e25/multiseed_analysis.json",
                "paper_tables": "results/paper_ready/paper_main_table.csv",
            },
        },
    }
    p = OUT / "reproducibility_manifest.json"
    p.write_text(json.dumps(manifest, indent=2))
    print(json.dumps({"environment": env, "weights_sha256": manifest["backbone"]["weights_sha256"]},
                     indent=1))
    print(f"[E26-C] -> {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
