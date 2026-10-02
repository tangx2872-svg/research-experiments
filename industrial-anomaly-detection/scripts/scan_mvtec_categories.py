"""数据扫描脚本：检查 MVTec AD 各 category 的数据完整性。

用途：Experiment 1E Pilot 前盘点数据。只读，不训练、不修改任何文件。

对每个 category 输出：
  - train/good 数量
  - test/good 数量
  - defect type 名称（自动发现，排除 good，字典序固定）
  - 每个 defect type 的 test 样本数
  - GT mask 计数 / 缺失 mask 数
  - 图像分辨率（取 test/good 第一张）
  - test 总数

用法：
  python scripts/scan_mvtec_categories.py \
      --root data/mvtec_ad \
      --categories bottle cable hazelnut screw grid
"""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image


def scan_category(root: Path, category: str) -> dict:
    cat_root = root / category
    train_good_dir = cat_root / "train" / "good"
    test_root = cat_root / "test"
    gt_root = cat_root / "ground_truth"

    train_good = sorted(train_good_dir.glob("*.png")) if train_good_dir.is_dir() else []
    test_good = sorted((test_root / "good").glob("*.png")) if (test_root / "good").is_dir() else []

    # 自动发现 defect types（排除 good）
    defect_types = sorted(
        d.name for d in test_root.iterdir()
        if d.is_dir() and d.name != "good" and any(d.glob("*.png"))
    ) if test_root.is_dir() else []

    defects = {}
    for dt in defect_types:
        imgs = sorted((test_root / dt).glob("*.png"))
        masks = sorted((gt_root / dt).glob("*_mask.png")) if (gt_root / dt).is_dir() else []
        # 按 stem 匹配 mask
        missing = [p.stem for p in imgs if not (gt_root / dt / f"{p.stem}_mask.png").exists()]
        defects[dt] = {
            "n_test": len(imgs),
            "n_mask": len(masks),
            "n_missing_mask": len(missing),
            "missing": missing,
        }

    # 图像分辨率（取 test/good 第一张）
    resolution = None
    if test_good:
        with Image.open(test_good[0]) as im:
            resolution = list(im.size)  # [W, H]

    total_test = len(test_good) + sum(v["n_test"] for v in defects.values())

    return {
        "category": category,
        "exists": cat_root.is_dir(),
        "train_good": len(train_good),
        "test_good": len(test_good),
        "defect_types": defect_types,
        "defects": defects,
        "resolution": resolution,
        "total_test": total_test,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="扫描 MVTec AD 各 category 数据完整性")
    parser.add_argument("--root", type=str, default="data/mvtec_ad")
    parser.add_argument("--categories", type=str, default="bottle cable hazelnut screw grid",
                        help="category 列表，可用空格或逗号分隔")
    args = parser.parse_args()

    root = Path(args.root)
    # 同时支持空格和逗号分隔
    categories = [c for c in args.categories.replace(",", " ").split() if c]

    print(f"root = {root.resolve()}")
    print(f"categories = {categories}\n")

    for cat in categories:
        r = scan_category(root, cat)
        print(f"{'='*60}")
        print(f"category: {cat}")
        if not r["exists"]:
            print("  [缺失] 目录不存在，请先下载数据")
            continue
        print(f"  train/good : {r['train_good']}")
        print(f"  test/good  : {r['test_good']}")
        print(f"  resolution : {r['resolution']} (W,H)")
        print(f"  total test : {r['total_test']}")
        print(f"  defect types ({len(r['defect_types'])}):")
        for dt in r["defect_types"]:
            d = r["defects"][dt]
            warn = "  [缺失 mask]" if d["n_missing_mask"] > 0 else ""
            print(f"    - {dt:24s} n_test={d['n_test']:>3}  n_mask={d['n_mask']:>3}"
                  f"  missing={d['n_missing_mask']}{warn}")
        if any(d["n_missing_mask"] > 0 for d in r["defects"].values()):
            print("  [警告] 存在缺失 GT mask，详见上方标注")
        print()


if __name__ == "__main__":
    main()
