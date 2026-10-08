"""Experiment 8B — S5（image ↔ mask alignment）全 15 cells 复算。

只读数据与冻结模型，重算 S5 并写 results/experiment8b/sanity/S5_alignment.csv，
不修改任何 extract/probe 结果资产。

判据（PRE-RUN）：top-5% patch-NN 距离的格点与 defect mask（overlap>0.5）重叠比例，
需显著高于随机（z > 3）。

用法：python -u scripts/experiment8b_s5_alignment.py
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "experiments" / "2026-09-30_illumination_sensitivity_exploration" / "falpha_patchcore"))

import experiment1b_defect_sensitivity as e1b  # noqa: E402
import experiment8b_extract as e8b  # noqa: E402
from anomalib.data import MVTecAD  # noqa: E402
from anomalib.models.image.patchcore.torch_model import PatchcoreModel  # noqa: E402


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PatchcoreModel(backbone="wide_resnet50_2", layers=["layer2", "layer3"],
                           pre_trained=True, num_neighbors=9).eval().to(device)
    rows = []
    for category in e8b.CATEGORIES:
        for seed in e8b.SEEDS:
            defect_types = sorted(p.name for p in (e8b.DATA_ROOT / category / "test").iterdir()
                                  if p.is_dir() and p.name != "good")
            dt = e8b.FROZEN_PRIMARY.get(category, [defect_types[0]])[0]
            imgs = sorted((e8b.DATA_ROOT / category / "test" / dt).glob("*.png"))[:2]
            for p in imgs:
                img_t = e1b.preprocess_for_model(e1b.load_image_as_tensor(p), device).unsqueeze(0)
                l2, l3 = e8b.features_of(model, img_t)
                m = e8b.load_mask_256(e8b.DATA_ROOT / category / "ground_truth" / dt / f"{p.stem}_mask.png")
                for ly, t in (("layer2", l2), ("layer3", l3)):
                    r = e8b.s5_alignment(category, seed, ly, t[0].cpu().numpy(), m)
                    r["category"] = category
                    r["seed"] = seed
                    r["layer"] = ly
                    r["defect_type"] = dt
                    r["image"] = p.name
                    rows.append(r)
    out = ROOT / "results" / "experiment8b" / "sanity"
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "S5_alignment.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    ok = [r for r in rows if r.get("status") == "OK"]
    zs = [r["z"] for r in ok]
    summary = {"n_rows": len(rows), "n_ok": len(ok),
               "z_min": float(min(zs)), "z_max": float(max(zs)),
               "overlap_min": float(min(r["mask_overlap_of_top5_anomaly_patches"] for r in ok)),
               "pass": bool(min(zs) > 3.0)}
    (out / "S5_alignment_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
