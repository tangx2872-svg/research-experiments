"""Experiment 8B — 冻结记录（pre-run freeze）。

把 README 的 PRE-RUN PLAN 落成可校验的 JSON（含 git HEAD / 环境 / 冻结阈值 / CASE 规则），
并写 sha256。任何 threshold / ranking / ratio 事后修改都会导致 hash 不一致。

用法: python -u scripts/experiment8b_freeze.py
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "experiment8b" / "reference"
README = ROOT / "experiments" / "experiment8b" / "README.md"


def git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True).stdout.strip()


def build() -> dict:
    import torch
    import anomalib
    return {
        "experiment": "8B",
        "title": "Defect-Illumination Feature Separability Probe",
        "date": "2026-10-07",
        "paper_phase": "Phase 5 - Solution Selection",
        "hypothesis": ("Defect sensitivity and illumination sensitivity may be heterogeneously "
                       "distributed across pretrained feature dimensions, leaving a potentially "
                       "selectable subspace that is simultaneously defect-sensitive and "
                       "illumination-stable."),
        "hypothesis_is_not_conclusion": True,
        "git": {
            "branch": git("branch", "--show-current"),
            "head": git("rev-parse", "HEAD"),
            "origin_main": git("rev-parse", "origin/main"),
            "status_short": git("status", "--short"),
        },
        "env": {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "cuda_available": bool(torch.cuda.is_available()),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "anomalib": getattr(anomalib, "__version__", "2.6.2"),
        },
        "representation": {
            "backbone": "wide_resnet50_2", "layers": ["layer2", "layer3"],
            "alpha": 0.0, "no_fit": True,
            "layer2_shape": [512, 32, 32], "layer3_shape": [1024, 16, 16],
            "preprocess": "experiment1b.load_image_as_tensor + preprocess_for_model (resize256 + ImageNet norm)",
            "split": "experiment1b.make_validation_split(cat, seed) -> val 20 / train rest",
        },
        "frozen_metrics": {
            "D_c": ("per-image defect-region patch mean m_{i,c}; per-image equal weight; "
                    "D_signed = (mean_i m_{i,c} - mu_c)/sigma_eff_c ; D_abs = |D_signed| ; "
                    "ranking uses D_abs"),
            "normal_reference": "full train/good patch population (mu_c, sigma_c, ddof=0)",
            "I_c": ("paired same-image/same-location mean_patch |f_c(T(x))-f_c(x)| per image, then "
                    "per-image equal weight, divided by sigma_eff_c"),
            "sigma_eff": "sqrt(sigma_c^2 + eps^2)",
            "eps": "1e-3 * median_c(sigma_c) per (cat, seed, layer), frozen",
            "D_secondary_variant": "pooled-SD sqrt((sigma_c^2 + sd_between_image(m_i,c)^2)/2)",
            "I_secondary_variant": "patch-pooled aggregation",
            "illumination_implementation": "experiment1_illumination_tradeoff.apply_photometric (reused)",
            "illumination_conditions": {
                "families": ["brightness", "gamma"],
                "levels": [0.5, 0.7, 0.9, 1.1, 1.3, 1.5],
                "severity": {"mild": [0.9, 1.1], "medium": [0.7, 1.3], "strong": [0.5, 1.5]},
                "identity_check_only": [1.0],
                "n_conditions_counted": 12,
            },
            "mask_rule": "GT mask -> resize256 BILINEAR -> >0.5 -> grid overlap ratio (1G convention); "
                         "defect patch = overlap > 0.5",
            "mask_fallback": "if zero patches with overlap>0.5, use the single max-overlap grid cell "
                             "(frozen pre-run; driven by measured coverage in asset audit)",
            "degenerate_rule": "sigma_c < 1e-6 flagged, NOT removed; one pre-registered robustness "
                               "variant excludes them",
        },
        "selection": {
            "granularity": "within-layer z-scores (z_D on D_abs, z_I on I_c)",
            "proposed_scalar": "s_c = z_D - z_I (only scalar version allowed)",
            "variants": ["PROPOSED", "D-ONLY", "I-ONLY", "PARETO", "FULL", "RANDOM"],
            "ratios": [0.25, 0.5],
            "ratio_scope": "within each layer separately (layer-balanced)",
            "metric_level_null_draws": 1000,
            "metric_level_null_seed": "8B0001",
            "score_level_random_draws_per_ratio": 10,
            "score_level_random_seed": "8B0002",
            "forbidden": ["D/I ratio", "D-0.5I", "2D-I", "log", "nonlinear", "lambda grid",
                          "selection ratio grid", "ranking formula grid"],
        },
        "case_rules": {
            "cells": "15 = 5 categories x 3 seeds; primary ratio 25%",
            "C1_defect_up": "z_D(PROPOSED25) >= +2 in >=12/15 cells AND mean D_sel > mean D_full",
            "C2_illum_down": "z_I(PROPOSED25) <= -2 in >=12/15 cells AND mean I_sel < mean I_full",
            "C3_category": ">=4/5 categories with >=2/3 seeds winning",
            "C4_seed": ">=2/3 seeds with >=3/5 categories winning AND 3-seed pooled ranking wins in >=4/5 categories",
            "C5_negative_transfer": "0 categories with D_sel < D_full or I_sel > I_full; tolerance 2% relative slack each axis",
            "C6_null": "6a mean(z_D) >= +3 and mean(z_I) <= -3; "
                       "6b score-level probe: (i) win vs FULL >=10/15 (ii) win vs RANDOM-mean >=12/15 "
                       "(iii) mean(d'_prop-d'_FULL)>=0 and mean(|dz|_prop-|dz|_FULL)<=0 "
                       "(iv) mean(d'_prop-mu_d'(RAND))>0 and mean(|dz|_prop-mu_|dz|(RAND))<0",
            "C7_not_layer_identity": "L2-only and L3-only each win (z_D>=2 and z_I<=-2) in >=9/15 cells",
            "decision_order": [
                "any primary sanity S1-S13 FAIL -> CASE_D / INVALID",
                "C1&C2&C3&C4&C5&C6a&C6b&C7 -> CASE_A / GO",
                "C1&C2 true but any of C3-C7 false -> CASE_B / HOLD (C6b failure alone forces CASE_B)",
                "else directional cells >= 8/15 -> CASE_B / HOLD",
                "else -> CASE_C / STOP (record coupling if pooled rho_Spearman(|D|,I) >= 0.5)",
            ],
        },
        "leakage_policy": {
            "8B_A": "oracle/diagnostic (uses test defect masks); never a final method performance",
            "ranking": "computed from target-side D/I of the same (cat, seed): maximum oracle",
            "score_level_probe": "labelled oracle_probe_only=true everywhere",
            "8B_B": "normal-only proxy development FORBIDDEN this round",
        },
        "sanity": [f"S{i}" for i in range(1, 14)],
        "amendments": [{
            "id": "A1",
            "date": "2026-10-07",
            "scope": "S13 criterion only (measurement validity), before reading any 8B target result",
            "change": ("S13 from bit-exact (max|delta| <= 1e-6) to numeric tolerance: corr >= 0.9999, "
                       "median|delta| <= 1e-3, p99|delta| <= 5e-3, max|delta| <= 0.05"),
            "reason": ("anomalib PatchcoreModel.forward applies feature_pooler=AvgPool2d(3,1,1) between "
                       "feature_extractor and generate_embedding; historic 1J/1H assets are pooled features. "
                       "8B extractor initially skipped the pooler (corr 0.894, max|delta| 3.4e+01). After the "
                       "fix corr=1.000000, median|delta|=1.8e-4, but GPU batch-composition dependent "
                       "non-determinism (batch1 vs batch8 max|delta|=6.5e-3, TF32 on) makes bit-exactness "
                       "unreachable."),
            "does_not_touch": ["D_c definition", "I_c definition", "CASE thresholds",
                               "selection rules", "ratios"],
            "impact_on_history": "none (1J assets unmodified; only the new 8B extractor was fixed)",
        }],
        "gpu_plan": {
            "phase_A": {"units": 15, "content": "feature extraction only (no bank, no fit)",
                        "estimated_wallclock": "10-20 min single worker", "vram_gb": 3},
            "phase_B": {"units": 420, "content": "channel-masked PatchCore full evaluation path",
                        "estimated_wallclock": "~2 h with 3 workers (~5.8 h single)", "vram_gb_per_worker": 2.6},
        },
        "target_results_read": False,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fr = build()
    fp = OUT / "protocol_freeze.json"
    fp.write_text(json.dumps(fr, indent=2, ensure_ascii=False))
    sha = hashlib.sha256(fp.read_bytes()).hexdigest()
    (OUT / "protocol_freeze.sha256").write_text(sha + "  protocol_freeze.json\n")
    rsha = hashlib.sha256(README.read_bytes()).hexdigest()
    (OUT / "readme_prerun.sha256").write_text(rsha + "  experiments/experiment8b/README.md\n")
    print(f"freeze      : {fp.relative_to(ROOT)}")
    print(f"sha256      : {sha}")
    print(f"readme sha  : {rsha}")
    print(f"target_results_read: {fr['target_results_read']}")


if __name__ == "__main__":
    main()
