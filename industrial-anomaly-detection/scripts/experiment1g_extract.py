"""
Experiment 1G — Step 1: 提取 pre/post-IN embedding + patch-role 映射

对每个 (selected defect type, alpha)：
  1. fit FAlphaPatchcore（复用 1B fit_model）
  2. 对每张 defect 图，截获 generate_embedding 前后（pre-IN F, post-IN F_alpha）的
     concat embedding（reshape 前，shape (1,1536,H,W)）
  3. 计算每个 patch 与 GT mask 的 overlap → patch_role（defect/boundary/background）
  4. 保存 patch_level.csv

关键：embedding 网格 H×W（1024 输入时 = 32×32），patch 的 receptive field 对应
原图 32×32 像素块。GT mask 下采样到 H×W 网格算 overlap。

输出 patch_level.csv 每行一个 patch：
  category, defect_type, image_path, seed, alpha, patch_id, h, w,
  gt_overlap, patch_role,
  feat_pre (1536-dim 太宽，不存全量，存统计量 + 单独存 npz),
  feat_alpha (同上)
实际：全量 embedding 存 npz，CSV 存 patch_id/role/overlap 索引。
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "experiments" / "2026-09-30_illumination_sensitivity_exploration" / "falpha_patchcore"))

from experiment1b_defect_sensitivity import (  # noqa: E402
    fit_model, make_validation_split, _move_model_to_device, load_mask,
    DATA_ROOT,
)
from falpha_patchcore import FAlphaPatchcoreModel  # noqa: E402

import experiment1b_defect_sensitivity as e1b  # noqa: E402

OUT_ROOT = ROOT / "results" / "experiment_1g"
ALPHAS = [0.0, 0.25, 0.5, 0.75, 1.0]
SEED = 0
OVERLAP_DEFECT = 0.5  # >50% 视为 defect patch


def load_selection(selection: str = "primary") -> list[dict]:
    rows = list(csv.DictReader(open(OUT_ROOT / "selection.csv", encoding="utf-8")))
    return [r for r in rows if r["selection"] == selection]


def list_defect_images(category: str, defect_type: str) -> list[Path]:
    """返回该 defect type 的 test 图（按文件名排序，与 1E 一致）。"""
    d = DATA_ROOT / category / "test" / defect_type
    return sorted(d.glob("*.png"))


def category_complete(category: str, defects: list[dict], out_npz: Path,
                      max_images: int | None) -> bool:
    """判断某 category 的所有 (defect, img, alpha) 的 npz 是否已全部产出。

    用于断点续跑：完整则跳过整个 category（含 fit，因为 model 无法从 bank.npy
    直接重建，fit 与 embedding 提取是该 category 的原子操作）。
    """
    for d in defects:
        dt = d["defect_type"]
        imgs = list_defect_images(category, dt)
        if max_images:
            imgs = imgs[:max_images]
        for img_path in imgs:
            for alpha in ALPHAS:
                stem = f"{category}_{dt}_{img_path.stem}_a{alpha:g}_seed{SEED}"
                if not (out_npz / f"{stem}.npz").exists():
                    return False
    return True


def mask_overlap_grid(mask: np.ndarray, grid_h: int, grid_w: int) -> np.ndarray:
    """GT mask（已 resize 到 256×256 的 0/1 数组）下采样到 (grid_h, grid_w) overlap 比例。"""
    h_img, w_img = mask.shape
    overlap = np.zeros((grid_h, grid_w), dtype=np.float32)
    for gh in range(grid_h):
        for gw in range(grid_w):
            h0 = int(gh * h_img / grid_h)
            h1 = int((gh + 1) * h_img / grid_h)
            w0 = int(gw * w_img / grid_w)
            w1 = int((gw + 1) * w_img / grid_w)
            block = mask[h0:h1, w0:w1]
            overlap[gh, gw] = float(block.mean())
    return overlap


def assign_role(overlap: float) -> str:
    if overlap > OVERLAP_DEFECT:
        return "defect"
    elif overlap > 0.0:
        return "boundary"
    else:
        return "background"


def extract_embeddings(torch_model: FAlphaPatchcoreModel, img_path: Path,
                       device: torch.device) -> tuple[np.ndarray, np.ndarray, int, int]:
    """截获 pre-IN 和 post-IN 的 embedding（reshape 前）。

    通过临时 monkey-patch generate_embedding 捕获 concat 后（IN 前）和 IN 后的 embedding。
    返回 (feat_pre, feat_alpha, H, W)，shape (H*W, D)。

    预处理与 1B 完全一致：resize 256 + ImageNet normalize（保证 embedding 与 1E 可比）。
    """
    captured = {}
    orig_gen = torch_model.generate_embedding

    def hooked(self, features):
        from anomalib.models.image.patchcore.torch_model import PatchcoreModel
        emb_pre = PatchcoreModel.generate_embedding(self, features)
        captured["pre"] = emb_pre.detach().clone()
        if self.alpha > 0.0:
            import torch.nn.functional as F
            f_in = F.instance_norm(emb_pre)
            emb_post = (1.0 - self.alpha) * emb_pre + self.alpha * f_in
        else:
            emb_post = emb_pre
        captured["post"] = emb_post.detach().clone()
        return emb_post

    torch_model.generate_embedding = hooked.__get__(torch_model)

    # 读图 + 预处理（与 1B preprocess_for_model 等价：to_tensor -> resize 256 -> normalize）
    from torchvision.transforms import functional as TF
    from PIL import Image
    img = Image.open(img_path).convert("RGB")
    img = TF.to_tensor(img)
    img = TF.resize(img, [256, 256], antialias=True)
    img = TF.normalize(img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    inp = img.unsqueeze(0).to(device)

    with torch.no_grad():
        torch_model(inp)

    pre = captured["pre"]   # (1, D, H, W)
    post = captured["post"]
    torch_model.generate_embedding = orig_gen

    D, H, W = pre.shape[1], pre.shape[2], pre.shape[3]
    feat_pre = pre[0].permute(1, 2, 0).reshape(-1, D).cpu().numpy()   # (H*W, D)
    feat_alpha = post[0].permute(1, 2, 0).reshape(-1, D).cpu().numpy()
    return feat_pre, feat_alpha, H, W


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selection", default="primary")
    ap.add_argument("--category", default=None, help="只跑某个 category（调试）")
    ap.add_argument("--max_images", type=int, default=None, help="每 defect 最多图数（smoke）")
    ap.add_argument("--no-resume", action="store_true", help="禁用断点续跑，从头重做")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device = {device}")

    # 初始化 1B 的 LOGS_DIR 全局（fit_model 内部引用）
    logs_dir = OUT_ROOT / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    e1b.LOGS_DIR = logs_dir

    sel = load_selection(args.selection)
    # 按 category 分组
    by_cat: dict[str, list[dict]] = {}
    for r in sel:
        by_cat.setdefault(r["category"], []).append(r)

    out_npz = OUT_ROOT / "embeddings"
    out_npz.mkdir(parents=True, exist_ok=True)

    csv_fields = ["category", "defect_type", "image_path", "seed", "alpha", "patch_id",
                  "h", "w", "gt_overlap", "patch_role", "npz"]
    csv_path = OUT_ROOT / "patch_level.csv"

    def rebuild_patch_csv() -> int:
        """从 embeddings/*.npz 全量重建 patch_level.csv（npz 是唯一真相源）。

        不依赖内存 patch_rows，因此断点续跑/跳过的 category 也能被正确覆盖。
        按 selection 的 (category, defect_type, img_stem, alpha) 正向枚举，避免 stem 反解。
        返回写入的 patch 行数。
        """
        rows: list[dict] = []
        for cat, defects in by_cat.items():
            for d in defects:
                dt = d["defect_type"]
                for img_path in list_defect_images(cat, dt):
                    img_stem = img_path.stem
                    for alpha in ALPHAS:
                        stem = f"{cat}_{dt}_{img_stem}_a{alpha:g}_seed{SEED}"
                        npz_path = out_npz / f"{stem}.npz"
                        if not npz_path.exists():
                            continue
                        data = np.load(npz_path)
                        overlap = data["overlap"]  # (H, W)
                        H, W = overlap.shape
                        for pid in range(H * W):
                            h, w = divmod(pid, W)
                            ov = float(overlap[h, w])
                            rows.append({
                                "category": cat, "defect_type": dt,
                                "image_path": str(img_path), "seed": f"seed{SEED}",
                                "alpha": alpha, "patch_id": pid, "h": h, "w": w,
                                "gt_overlap": ov, "patch_role": assign_role(ov),
                                "npz": f"{stem}.npz",
                            })
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=csv_fields)
            w.writeheader()
            w.writerows(rows)
        return len(rows)

    for cat, defects in by_cat.items():
        if args.category and cat != args.category:
            continue
        # 断点续跑：该 category 的 npz 已齐全则跳过（含 fit，model 无法从 bank 重建）
        if not args.no_resume and category_complete(cat, defects, out_npz, args.max_images):
            print(f"\n[{cat}] 已完整（{sum(len(list_defect_images(cat, d['defect_type'])) for d in defects)} 图），跳过")
            continue
        # 每个 category 只需 fit 一次 per alpha（同一 category 共享 train_ids）
        train_ids, _ = make_validation_split(cat, seed=SEED)
        models: dict[float, FAlphaPatchcoreModel] = {}
        print(f"\n[{cat}] fit {len(ALPHAS)} alpha...")
        bank_dir = OUT_ROOT / "memory_banks"
        bank_dir.mkdir(parents=True, exist_ok=True)
        for alpha in ALPHAS:
            _, torch_model, _ = fit_model(alpha, cat, train_ids, seed=SEED)
            _move_model_to_device(torch_model, device)
            torch_model.eval()
            models[alpha] = torch_model
            # 保存 memory_bank（供 matched / frozen-M0 分析）
            mb = torch_model.memory_bank.detach().cpu().numpy()
            np.save(bank_dir / f"bank_{cat}_a{alpha:g}_seed{SEED}.npy", mb)
            print(f"    bank_{cat}_a{alpha:g}: {mb.shape}")

        for d in defects:
            dt = d["defect_type"]
            imgs = list_defect_images(cat, dt)
            if args.max_images:
                imgs = imgs[:args.max_images]
            print(f"  [{cat}/{dt}] {len(imgs)} 张图")

            for img_path in imgs:
                mask_path = DATA_ROOT / cat / "ground_truth" / dt / (img_path.stem + "_mask.png")
                # 用 1B load_mask 读 mask，然后 resize 到 256 与 embedding 网格对齐
                mask_bin = load_mask(mask_path).astype(np.float32)
                from PIL import Image
                mask_bin = np.array(Image.fromarray((mask_bin * 255).astype(np.uint8))
                                    .resize((256, 256), Image.BILINEAR)).astype(np.float32) / 255.0

                # 每 alpha 提取 embedding
                for alpha in ALPHAS:
                    tm = models[alpha]
                    feat_pre, feat_alpha, H, W = extract_embeddings(tm, img_path, device)

                    # patch-role（overlap 与 alpha 无关，算一次即可）
                    if alpha == 0.0:
                        overlap = mask_overlap_grid(mask_bin, H, W)

                    # 保存 npz
                    stem = f"{cat}_{dt}_{img_path.stem}_a{alpha:g}_seed{SEED}"
                    np.savez_compressed(
                        out_npz / f"{stem}.npz",
                        feat_pre=feat_pre, feat_alpha=feat_alpha, overlap=overlap,
                    )

        # 释放模型显存
        for m in models.values():
            del m
        torch.cuda.empty_cache()

    # 从所有 npz 全量重建 patch_level.csv（唯一真相源，覆盖断点/跳过）
    n_rows = rebuild_patch_csv()
    print(f"\n完成：{n_rows} patch 行 → patch_level.csv")
    print(f"embedding npz 目录：{out_npz}")


if __name__ == "__main__":
    main()
