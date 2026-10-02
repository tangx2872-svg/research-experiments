"""
Experiment 1G — Step 1 收尾: patch-role 可视化校验

随机选 3 张 defect 图，把 32×32 网格的 patch role 可视化：
  defect (红) / boundary (黄) / background (不标)
叠加在原始 GT mask 上，验证 patch→GT 空间映射正确。
"""

import csv
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "experiment_1g"
FIG = OUT / "roi_check"
FIG.mkdir(parents=True, exist_ok=True)


def load_overlap(cat, dt, stem):
    d = np.load(OUT / "embeddings" / f"{cat}_{dt}_{stem}_a0_seed0.npz")
    return d["overlap"]  # (32,32)


def vis_one(cat, dt, stem):
    overlap = load_overlap(cat, dt, stem)
    h, w = overlap.shape

    # 读取原图 + GT mask
    img_path = ROOT / "data" / "mvtec_ad" / cat / "test" / dt / f"{stem}.png"
    mask_path = ROOT / "data" / "mvtec_ad" / cat / "ground_truth" / dt / f"{stem}_mask.png"
    img = Image.open(img_path).convert("RGB").resize((256, 256))
    mask = Image.open(mask_path).convert("L").resize((256, 256))
    mask = np.array(mask) > 127

    arr = np.array(img).astype(np.float32)
    # GT mask 用半透明青色叠加
    arr[mask] = arr[mask] * 0.5 + np.array([0, 200, 200]) * 0.5

    # 画网格 patch role
    ph, pw = 256 / h, 256 / w
    for i in range(h):
        for j in range(w):
            ov = overlap[i, j]
            if ov > 0.5:    # defect
                color = np.array([255, 0, 0])
            elif ov > 0:    # boundary
                color = np.array([255, 255, 0])
            else:
                continue
            y0, x0 = int(i * ph), int(j * pw)
            y1, x1 = int((i + 1) * ph), int((j + 1) * pw)
            arr[y0:y1, x0:x1] = arr[y0:y1, x0:x1] * 0.5 + color * 0.5

    out_p = FIG / f"patch_role_{cat}_{dt}_{stem}.png"
    Image.fromarray(arr.astype(np.uint8)).save(out_p)
    print(f"已保存 {out_p}  (defect={(overlap>0.5).sum()} patch, boundary={((overlap>0)&(overlap<=0.5)).sum()})")


def main():
    # 选 3 个代表（shrink / neutral / expand 各一）
    picks = [
        ("bottle", "contamination", "000"),
        ("cable", "bent_wire", "000"),
        ("grid", "metal_contamination", "000"),
    ]
    for cat, dt, stem in picks:
        try:
            vis_one(cat, dt, stem)
        except FileNotFoundError as e:
            print(f"跳过 {cat}/{dt}/{stem}: {e}")


if __name__ == "__main__":
    main()
