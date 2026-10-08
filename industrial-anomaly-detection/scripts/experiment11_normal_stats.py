"""Experiment 11A — normal-only statistics（**只用 train/good**，无 test/defect 数据）。

对每类训练图（batch=16, no_grad）提取 **pooled** layer2/layer3 特征，累计 per-channel 统计：
  f_rms_j  : 通道空间 RMS（对图像取均值）
  in_mag_j : ||IN(F)_j - F_j|| 的空间 RMS（C2 补偿项的方向响应幅度）
  ratio_j  : in_mag_j / f_rms_j（residual/original ratio）
  var_j    : 通道空间方差（对图像取均值）
  disp_j   : 通道跨图像离散度（跨图像 std / 均值幅度）
标量（每类一行）：S_IN_L2/L3、q=S_IN_L2/S_IN_L3、RMS_L2/L3、RATIO_L2、C2_ENERGY_RATIO、
PR_VAR_L2（通道能量 participation ratio）、DISP_L2。
输出：stats/normal_statistics.csv + stats/per_channel_<cat>.npz
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "experiments" / "2026-09-30_illumination_sensitivity_exploration" / "falpha_patchcore"))

EXP = ROOT / "results" / "experiment_11_c2_adaptive"
CATS = ["bottle", "cable", "hazelnut", "screw", "grid"]
BETA = 0.25


def main() -> None:
    import experiment1b_defect_sensitivity as e1b
    from anomalib.models.image.patchcore.torch_model import PatchcoreModel

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PatchcoreModel(backbone="wide_resnet50_2", layers=["layer2", "layer3"],
                           pre_trained=True, num_neighbors=9).eval().to(dev)
    (EXP / "stats").mkdir(parents=True, exist_ok=True)
    rows = []
    for cat in CATS:
        imgs = sorted((ROOT / "data" / "mvtec_ad" / cat / "train" / "good").glob("*.png"))
        acc, chmean = {}, []
        with torch.no_grad():
            for i in range(0, len(imgs), 16):
                x = torch.stack([e1b.preprocess_for_model(e1b.load_image_as_tensor(p), dev)
                                 for p in imgs[i:i + 16]])
                feats = model.feature_extractor(x)
                feats = {k: model.feature_pooler(v) for k, v in feats.items()}
                for layer in ("layer2", "layer3"):
                    f = feats[layer]
                    d = F.instance_norm(f) - f
                    cur = acc.setdefault(layer, {"rms": None, "inmag": None, "var": None, "n": 0})
                    rms = f.pow(2).mean(dim=(2, 3)).sqrt().sum(0)
                    imag = d.pow(2).mean(dim=(2, 3)).sqrt().sum(0)
                    var = f.var(dim=(2, 3), unbiased=False).sum(0)
                    cur["rms"] = rms if cur["rms"] is None else cur["rms"] + rms
                    cur["inmag"] = imag if cur["inmag"] is None else cur["inmag"] + imag
                    cur["var"] = var if cur["var"] is None else cur["var"] + var
                    cur["n"] += int(x.shape[0])
                    if layer == "layer2":
                        chmean.append(f.mean(dim=(2, 3)).cpu())
        L2, L3 = acc["layer2"], acc["layer3"]
        rms2 = (L2["rms"] / L2["n"]).cpu().numpy()
        inmag2 = (L2["inmag"] / L2["n"]).cpu().numpy()
        var2 = (L2["var"] / L2["n"]).cpu().numpy()
        rms3 = (L3["rms"] / L3["n"]).cpu().numpy()
        inmag3 = (L3["inmag"] / L3["n"]).cpu().numpy()
        ratio2 = inmag2 / (rms2 + 1e-8)
        cm = torch.cat(chmean, 0).numpy()
        disp = cm.std(0) / (np.abs(cm).mean(0) + 1e-8)
        pr = float(var2.sum() ** 2 / (var2 ** 2).sum())
        rows.append({"category": cat, "n_train_images": int(L2["n"]),
                     "S_IN_L2": float(inmag2.mean()), "S_IN_L3": float(inmag3.mean()),
                     "q_L2_over_L3": float(inmag2.mean() / (inmag3.mean() + 1e-12)),
                     "RMS_L2": float(rms2.mean()), "RMS_L3": float(rms3.mean()),
                     "RATIO_L2": float(ratio2.mean()),
                     "C2_ENERGY_RATIO": float(BETA * ratio2.mean()),
                     "PR_VAR_L2": pr, "PR_VAR_L2_frac": float(pr / len(var2)),
                     "DISP_L2": float(disp.mean())})
        np.savez(EXP / "stats" / f"per_channel_{cat}.npz", rms=rms2, inmag=inmag2, var=var2,
                 ratio=ratio2, disp=disp, chmean_abs=np.abs(cm).mean(0))
        print("[11A] %-9s n=%3d S_IN_L2=%.4f S_IN_L3=%.4f q=%.4f RATIO=%.4f PR=%.1f DISP=%.4f"
              % (cat, L2["n"], rows[-1]["S_IN_L2"], rows[-1]["S_IN_L3"],
                 rows[-1]["q_L2_over_L3"], rows[-1]["RATIO_L2"], pr, rows[-1]["DISP_L2"]), flush=True)
    with open(EXP / "stats" / "normal_statistics.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print("[11A] written stats/normal_statistics.csv")


if __name__ == "__main__":
    main()
