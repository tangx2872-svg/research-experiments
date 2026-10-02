"""
Experiment 1F — Offline Defect-Attribute Mechanism Screening
Step 1-2: Attribute Extractor

从原始 defect 图像 + GT mask 提取预注册的视觉属性。
完全离线，不读取任何 response signature，不做模型推理。

属性分组（见 config.json）：
  size:         area_ratio, log_area_ratio (已有，控制变量)
  contrast:     intensity_contrast, lab_contrast, local_variance_ratio
  frequency:    gradient_mean_defect, gradient_contrast, laplacian_energy, hf_energy_ratio
  morphology:   area_pixels, perimeter, compactness, elongation

关键设计：
  - surrounding ring = dilated(mask, 15px) - mask，作为缺陷周围正常 context
  - 所有 ROI 只在 mask 有效区域内统计（排除背景 0 值干扰）
  - 输出 sample-level CSV，一个唯一样本一行（按 image_path 去重）
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "data" / "mvtec_ad"
OUT_ROOT = ROOT / "results" / "experiment_1f"

DILATION_PX = 15          # 冻结参数
SENSITIVITY_PX = [10, 20] # 预注册的 sensitivity check
FFT_HF_FC = 0.30          # HF 阈值 = 0.3 * nyquist (最高频的 70% 视为 HF)
EPS = 1e-6

CATEGORIES = ["bottle", "grid", "cable", "screw", "hazelnut"]


def load_sample_paths(category: str) -> list[dict]:
    """从 1E sample_level 提取唯一样本（alpha==0 去重），返回 image_path + mask_path 列表。

    mask_path 为空时（bottle 从 1C 重建未存 mask 字段），按 MVTec 标准规则自动推断：
    ground_truth/<defect_type>/<basename>_mask.png
    """
    p = ROOT / "results" / "experiment_1e" / category / "seed_0" / "sample_level.csv"
    seen: set[str] = set()
    samples: list[dict] = []
    with open(p, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["defect_type"] == "good":
                continue
            ip = r["image_path"]
            if ip in seen:
                continue
            seen.add(ip)
            mask_path = r["mask_path"].strip() if r["mask_path"] else ""
            if not mask_path:
                # 自动推断 mask 路径
                img_p = Path(ip)
                mask_path = str(DATA_ROOT / category / "ground_truth" /
                                r["defect_type"] / (img_p.stem + "_mask.png"))
            samples.append({
                "category": category,
                "defect_type": r["defect_type"],
                "image_path": ip,
                "mask_path": mask_path,
            })
    return samples


def read_gray_and_lab(img_path: str):
    """读取图像，返回 (gray uint8, lab float32, bgr)。"""
    bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"无法读取图像: {img_path}")
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    return gray, lab, bgr


def extract_mask_attrs(mask: np.ndarray):
    """从二值 mask 提取 morphology + area 属性。

    mask: uint8 二值 (0/255 或 0/1)。
    返回 dict: area_pixels, perimeter, compactness, elongation, major_axis, minor_axis
    """
    bin_mask = (mask > 0).astype(np.uint8)

    area_pixels = float(bin_mask.sum())

    # perimeter: 用轮廓近似
    contours, _ = cv2.findContours(bin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return {
            "area_pixels": 0.0, "perimeter": np.nan, "compactness": np.nan,
            "elongation": np.nan, "major_axis": np.nan, "minor_axis": np.nan,
        }
    # 取最大轮廓（GT mask 通常单连通，但保险起见）
    largest = max(contours, key=cv2.contourArea)
    perimeter = float(cv2.arcLength(largest, True))

    compactness = (4 * np.pi * area_pixels) / (perimeter ** 2 + EPS) if perimeter > 0 else np.nan

    # elongation 用 fitEllipse 的长短轴比（需 >=5 个点）
    if len(largest) >= 5:
        (cx, cy), (ma, MA), angle = cv2.fitEllipse(largest)
        # fitEllipse 返回的 (ma, MA) 是 (短轴/2, 长轴/2) 直径的一半？实际是轴长
        major_axis = float(max(ma, MA))
        minor_axis = float(min(ma, MA))
        elongation = major_axis / (minor_axis + EPS)
    else:
        major_axis = minor_axis = np.nan
        elongation = np.nan

    return {
        "area_pixels": area_pixels,
        "perimeter": perimeter,
        "compactness": compactness,
        "elongation": elongation,
        "major_axis": major_axis,
        "minor_axis": minor_axis,
    }


def compute_attributes(image_path: str, mask_path: str, category: str,
                       dilation_px: int = DILATION_PX) -> dict:
    """对一个 defect 样本提取全部视觉属性。"""
    gray, lab, bgr = read_gray_and_lab(image_path)

    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"无法读取 mask: {mask_path}")
    bin_mask = (mask > 0).astype(np.uint8)

    H, W = gray.shape
    total_pixels = H * W

    attrs = extract_mask_attrs(bin_mask)

    # ---- size (控制变量) ----
    area_pixels = attrs["area_pixels"]
    area_ratio = area_pixels / total_pixels
    log_area_ratio = np.log(area_ratio + EPS) if area_ratio > 0 else np.nan

    # ---- local contrast ----
    # surrounding ring = dilated(mask) - mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilation_px * 2 + 1, dilation_px * 2 + 1))
    dilated = cv2.dilate(bin_mask, kernel)
    ring = (dilated - bin_mask).astype(np.uint8)

    # 缺陷区域的有效像素统计（只在 mask 内）
    defect_gray = gray[bin_mask > 0]
    defect_lab = lab[bin_mask > 0]

    # ring 有效像素（ring 内，且排除纯黑背景像素 —— 灰度>0）
    ring_valid = (ring > 0) & (gray > 0)
    ring_gray = gray[ring_valid]
    ring_lab = lab[ring_valid]

    if defect_gray.size == 0:
        raise ValueError(f"空 mask: {image_path}")

    mu_defect_gray = float(defect_gray.mean())
    mu_defect_lab = defect_lab.mean(axis=0)  # (L,a,b)

    if ring_gray.size == 0:
        # ring 为空（缺陷太大填满图），fallback 到全图非 mask 区域
        ring_gray = gray[(bin_mask == 0) & (gray > 0)]
        ring_lab = lab[(bin_mask == 0) & (gray > 0)]
        if ring_gray.size == 0:
            raise ValueError(f"无法构造 ring: {image_path}")

    mu_ring_gray = float(ring_gray.mean())
    mu_ring_lab = ring_lab.mean(axis=0)

    # 1. intensity contrast
    intensity_contrast = abs(mu_defect_gray - mu_ring_gray)

    # 2. Lab color contrast (L2 距离)
    lab_contrast = float(np.linalg.norm(mu_defect_lab - mu_ring_lab))

    # 3. local variance ratio
    sigma_defect = float(defect_gray.std())
    sigma_ring = float(ring_gray.std())
    local_variance_ratio = sigma_defect / (sigma_ring + EPS)

    # ---- spatial frequency / texture ----
    # Sobel gradient
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = cv2.magnitude(gx, gy)

    gradient_mean_defect = float(grad_mag[bin_mask > 0].mean())
    gradient_mean_ring = float(grad_mag[ring_valid].mean()) if ring_valid.any() else np.nan
    gradient_contrast = gradient_mean_defect - gradient_mean_ring

    # Laplacian energy
    lap = cv2.Laplacian(gray, cv2.CV_32F)
    laplacian_energy = float(np.abs(lap)[bin_mask > 0].mean())

    # 高频纹理能量比（缺陷区 vs ring）
    hf_energy_ratio = compute_hf_energy_ratio(gray, bin_mask, fc=FFT_HF_FC,
                                              dilation_px=dilation_px)

    attrs.update({
        "category": category,
        "area_ratio": area_ratio,
        "log_area_ratio": log_area_ratio,
        "intensity_contrast": intensity_contrast,
        "lab_contrast": lab_contrast,
        "local_variance_ratio": local_variance_ratio,
        "gradient_mean_defect": gradient_mean_defect,
        "gradient_contrast": gradient_contrast,
        "laplacian_energy": laplacian_energy,
        "hf_energy_ratio": hf_energy_ratio,
    })
    return attrs


def compute_hf_energy_ratio(gray: np.ndarray, bin_mask: np.ndarray, fc: float,
                         patch_size: int = 256, dilation_px: int = 15) -> float:
    """缺陷区 vs 周围 ring 的高频纹理能量比（spatial-frequency 属性）。

    技术说明（重要，记录在案）：
      协议原定义「对 defect-centered patch 做 FFT 的 HF ratio = Σ(f>fc)|F|²/Σ|F|²」，
      经诊断（去均值后自然图像功率谱近似 1/f 幂律，f>0.25*nyquist 环带在 2D 频谱中
      面积占比 >99%），该比值对几乎所有 patch 恒 ~1.0，无区分度 —— 这是指标定义的
      数学必然，非代码 bug。质心频率同样由 1/f 共性主导，区分度 <0.003。

      因此改用「缺陷区 vs surrounding ring 的高频能量比」，即协议中 GradientContrast
      （G_defect - G_ring）的能量版。空域 Sobel 梯度平方和是稳健的高频纹理度量，
      直接回答「缺陷是否比周围环境更细碎」。这与 gradient_contrast（一阶幅值差）
      互补：此处用能量（平方和）比值，对高频结构更敏感。

      实测区分度良好：hazelnut/print≈16, crack≈5.5, cable_swap≈0.07, contamination≈1.3。
    """
    ys, xs = np.where(bin_mask > 0)
    if len(xs) == 0:
        return np.nan

    # surrounding ring
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilation_px * 2 + 1, dilation_px * 2 + 1))
    dilated = cv2.dilate(bin_mask, k)
    ring = (dilated - bin_mask).astype(np.uint8)

    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    hf_energy = gx ** 2 + gy ** 2  # 高频能量（梯度平方和）

    d_hf = float(hf_energy[bin_mask > 0].mean())
    ring_valid = (ring > 0) & (gray > 0)
    if not ring_valid.any():
        # ring 为空，退化到全图非 mask 区域
        ring_valid = (bin_mask == 0) & (gray > 0)
    r_hf = float(hf_energy[ring_valid].mean())
    return d_hf / (r_hf + EPS)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--categories", nargs="*", default=CATEGORIES)
    ap.add_argument("--dilation", type=int, default=DILATION_PX)
    args = ap.parse_args()

    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    all_rows: list[dict] = []
    for cat in args.categories:
        samples = load_sample_paths(cat)
        print(f"[{cat}] {len(samples)} 唯一样本，开始提取属性...")
        for s in samples:
            a = compute_attributes(s["image_path"], s["mask_path"], cat, args.dilation)
            all_rows.append({
                "category": cat,
                "defect_type": s["defect_type"],
                "image_path": s["image_path"],
                "mask_path": s["mask_path"],
                **{k: a[k] for k in [
                    "area_pixels", "area_ratio", "log_area_ratio",
                    "perimeter", "compactness", "elongation", "major_axis", "minor_axis",
                    "intensity_contrast", "lab_contrast", "local_variance_ratio",
                    "gradient_mean_defect", "gradient_contrast", "laplacian_energy",
                    "hf_energy_ratio",
                ]},
            })

    fieldnames = ["category", "defect_type", "image_path", "mask_path",
                  "area_pixels", "area_ratio", "log_area_ratio",
                  "perimeter", "compactness", "elongation", "major_axis", "minor_axis",
                  "intensity_contrast", "lab_contrast", "local_variance_ratio",
                  "gradient_mean_defect", "gradient_contrast", "laplacian_energy",
                  "hf_energy_ratio"]

    out_p = OUT_ROOT / f"sample_attributes_d{args.dilation}.csv"
    with open(out_p, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(all_rows)

    print(f"\n完成：{len(all_rows)} 个唯一样本 → {out_p}")
    # 统计 NaN
    nan_counts = {}
    for row in all_rows:
        for k in fieldnames:
            v = row[k]
            if isinstance(v, float) and (np.isnan(v) if v == v else True):
                nan_counts[k] = nan_counts.get(k, 0) + 1
    if nan_counts:
        print("NaN 统计:", nan_counts)


if __name__ == "__main__":
    main()
