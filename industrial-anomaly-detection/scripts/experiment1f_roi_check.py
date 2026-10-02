"""
Experiment 1F — Step 2: ROI 可视化校验

随机抽 3 张图，把 defect mask（红）+ surrounding ring（绿）叠加在原图上，
验证 ROI 构造是否正确。这是分析前的关键防伪步骤。
"""

import sys
from pathlib import Path
import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from experiment1f_attribute_extractor import DILATION_PX, load_sample_paths

OUT_DIR = ROOT / "results" / "experiment_1f" / "roi_check"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# 手动选 3 个代表性样本（不同 category + 不同 defect 形态）
PICKS = [
    ("hazelnut", "print"),
    ("cable", "bent_wire"),
    ("grid", "metal_contamination"),
]

for cat, dt in PICKS:
    samples = load_sample_paths(cat)
    # 找该 defect type 的第一个样本
    s = next(x for x in samples if x["defect_type"] == dt)

    img = cv2.imread(s["image_path"], cv2.IMREAD_COLOR)
    mask = cv2.imread(s["mask_path"], cv2.IMREAD_GRAYSCALE)
    bin_mask = (mask > 0).astype(np.uint8)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                       (DILATION_PX * 2 + 1, DILATION_PX * 2 + 1))
    dilated = cv2.dilate(bin_mask, kernel)
    ring = (dilated - bin_mask).astype(np.uint8)

    vis = img.copy()
    # 缺陷 mask 红色
    vis[bin_mask > 0] = (0, 0, 255)
    # ring 绿色（半透明叠加）
    green = np.zeros_like(vis)
    green[ring > 0] = (0, 255, 0)
    vis = cv2.addWeighted(vis, 0.7, green, 0.3, 0)

    # 标注
    cv2.putText(vis, f"{cat}/{dt}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 0), 2)

    out_p = OUT_DIR / f"roi_check_{cat}_{dt}.png"
    cv2.imwrite(str(out_p), vis)
    print(f"已保存 {out_p}  (defect={bin_mask.sum()}px, ring={ring.sum()}px)")

print("\nROI 可视化完成，请人工检查红色=defect、绿色=ring 是否正确。")
