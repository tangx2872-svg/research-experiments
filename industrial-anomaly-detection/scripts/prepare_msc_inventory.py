#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
MSC-AD 数据落地流水线：下载 → 解压 → 生成 dataset inventory（不训练模型）。

用法（三选一）：
  1. 已有压缩包：  python scripts/prepare_msc_inventory.py --archive <path/to/mscad.zip>
  2. 已有解压目录：python scripts/prepare_msc_inventory.py --dir <path/to/mscad/>
  3. 给定下载 URL：python scripts/prepare_msc_inventory.py --url <http://...> [--out-dir data/msc_ad]

流程：
  [可选] 下载 → 解压 → 递归扫描目录结构 → 统计 surface/defect/illumination/resolution/图片数量
  → 输出 data/msc_ad/inventory.json + data/msc_ad/inventory_report.md

设计原则：
  - 只读数据、只做侦察，不训练、不改模型、不改 PatchCore。
  - 不依赖具体目录结构先验；用启发式规则从目录名/文件名推断 surface、defect、
    illumination、resolution 等维度，推断结果标注 confidence，未知项显式标"unknown"。
  - 幂等：重复运行覆盖 inventory 产物，不删除任何原始数据。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from urllib.parse import unquote

# 项目内 data 根目录（相对脚本上级）
DATA_ROOT = Path(__file__).resolve().parents[1] / "data"

# 支持的图片扩展名
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
# 标注/mask 扩展名（pixel-level ground truth 常见后缀）
MASK_HINTS = ("mask", "gt", "groundtruth", "ground_truth", "label", "seg", "annot")
# 压缩包扩展名
ARCHIVE_EXTS = {".zip", ".tar", ".gz", ".tgz", ".xz", ".txz", ".rar", ".7z", ".tar.gz", ".tar.xz"}


def log(msg: str) -> None:
    print(f"[inventory] {msg}", flush=True)


def file_sha256(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


# ---------------------------------------------------------------- 下载 ----
def download(url: str, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    # 从 URL 推断文件名
    name = Path(unquote(url.rstrip("/").split("?")[0])).name or "msc_ad_download.bin"
    dest = out_dir / name
    if dest.exists() and dest.stat().st_size > 0:
        log(f"已存在，跳过下载：{dest}")
        return dest
    log(f"下载 {url} -> {dest}")
    # 优先 curl（带断点续传），失败回退 python urllib
    try:
        subprocess.run(
            ["curl", "-L", "-C", "-", "-o", str(dest), url],
            check=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        import urllib.request

        log("curl 不可用或失败，改用 urllib 下载")
        with urllib.request.urlopen(url) as r, dest.open("wb") as f:
            shutil.copyfileobj(r, f)
    log(f"下载完成：{dest} ({dest.stat().st_size / 1e6:.1f} MB)")
    return dest


# ---------------------------------------------------------------- 解压 ----
def extract(archive: Path, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    log(f"解压 {archive} -> {out_dir}")
    ext = archive.suffix.lower()
    name_lower = archive.name.lower()

    if name_lower.endswith((".tar.gz", ".tgz")) or ext in {".tar", ".gz", ".tgz"}:
        shutil.unpack_archive(str(archive), str(out_dir))
    elif ext in {".zip", ".xz", ".txz"} or name_lower.endswith(".tar.xz"):
        shutil.unpack_archive(str(archive), str(out_dir))
    elif ext == ".rar":
        log("检测到 .rar，尝试用 unrar 解压")
        subprocess.run(["unrar", "x", "-o+", str(archive), str(out_dir)], check=True)
    elif ext == ".7z":
        log("检测到 .7z，尝试用 7z 解压")
        subprocess.run(["7z", "x", f"-o{out_dir}", "-y", str(archive)], check=True)
    else:
        # 未知格式，尝试 unpack_archive 兜底
        shutil.unpack_archive(str(archive), str(out_dir))
    log("解压完成")
    return out_dir


# ------------------------------------------------------------- 启发式推断 ----
def infer_dimension(name: str) -> dict:
    """从单个目录/文件名字符串推断维度。返回 {key: value}，value 为 (取值, 置信度)。"""
    result: dict = {}
    low = name.lower()

    # illumination 档位
    if re.search(r"(illu|illum|light|bright|exp)", low):
        if re.search(r"(low|_l\b|l1)", low):
            result["illumination"] = ("low", "high")
        elif re.search(r"(mid|med|medium|_m\b|m1)", low):
            result["illumination"] = ("mid", "high")
        elif re.search(r"(high|_h\b|h1)", low):
            result["illumination"] = ("high", "high")

    # resolution：形如 150x150 / 150 / 300 等
    m = re.search(r"(\d{3,4})\s*[xX*×]\s*(\d{3,4})", name)
    if m:
        result["resolution"] = (f"{m.group(1)}x{m.group(2)}", "high")
    else:
        m = re.search(r"(?:res|r)(\d{3,4})\b", low)
        if m:
            result["resolution"] = (f"{m.group(1)}x{m.group(1)}", "medium")

    return result


def is_image(p: Path) -> bool:
    return p.suffix.lower() in IMAGE_EXTS


def is_mask(p: Path) -> bool:
    low = p.name.lower()
    return any(h in low for h in MASK_HINTS)


# split / 保留目录的识别集合（小写）
SPLIT_DIRS = {"train", "training", "test", "testing", "val", "validation", "eval"}
NORMAL_DIRS = {"good", "normal", "ok", "okay"}
ANOMALY_DIRS = {"bad", "defect", "anomaly", "ng", "abnormal"}
MASK_DIRS = {"groundtruth", "ground_truth", "gt", "mask", "masks", "label", "labels", "annotations"}
ILLUM_DIR_HINTS = ("illu", "illum", "light", "bright", "exp")


def classify_path(rel_parts: list[str]) -> dict:
    """根据相对路径各部分，推断 (surface, defect, split, scene)。

    约定：root 下第一层 = surface；train/test/ground_truth 下直接子目录 = defect 类别。
    """
    info = {"surface": "unknown", "defect": None, "split": "unknown", "scene": "unknown"}

    parts = list(rel_parts)
    if parts:
        info["surface"] = parts[0]

    # split 判定
    for part in parts:
        pl = part.lower()
        if pl in SPLIT_DIRS:
            info["split"] = pl if pl in {"train", "test"} else "test"
            break

    # defect 类别：取 train/test/ground_truth 之后紧跟的目录名
    # 遍历，找 split 目录或 mask 目录，取其后第一个非图片段作为 defect 名
    for i, part in enumerate(parts):
        pl = part.lower()
        if pl in SPLIT_DIRS or pl in MASK_DIRS:
            # 找其后第一个非 split/mask/illum 段
            for j in range(i + 1, len(parts)):
                cand = parts[j]
                cl = cand.lower()
                if cl in SPLIT_DIRS or cl in MASK_DIRS:
                    break
                if re.search("|".join(ILLUM_DIR_HINTS), cl) or re.fullmatch(r"\d{3,4}([xX*×]\d{3,4})?", cand):
                    break
                info["defect"] = cand
                break
            break

    # normal/anomaly 语义归并
    if info["defect"]:
        dl = info["defect"].lower()
        if dl in NORMAL_DIRS:
            info["defect"] = "good"
        elif dl in ANOMALY_DIRS:
            info["defect"] = "defect"

    # scene：从路径中推断 illumination+resolution 组合
    dims = {}
    for part in parts:
        d = infer_dimension(part)
        dims.update({k: v for k, v in d.items()})
    if dims:
        segs = []
        if "illumination" in dims:
            segs.append(f"illum_{dims['illumination'][0]}")
        if "resolution" in dims:
            segs.append(f"res_{dims['resolution'][0]}")
        info["scene"] = "_".join(segs) if segs else "unknown"

    return info


# ------------------------------------------------------------- 扫描 ----
def scan_tree(root: Path) -> dict:
    log(f"扫描目录：{root}")
    surfaces: dict = {}
    image_files: list[Path] = []
    mask_files: list[Path] = []
    other_files: list[Path] = []

    # 先收集所有图片/掩码/其他文件（mask 优先于 image 判定）
    all_files = [p for p in root.rglob("*") if p.is_file()]
    for p in all_files:
        if is_mask(p):
            mask_files.append(p)
        elif is_image(p):
            image_files.append(p)
        else:
            other_files.append(p)

    log(f"共 {len(all_files)} 个文件：图片 {len(image_files)}，mask {len(mask_files)}，其他 {len(other_files)}")

    # 按图片路径推断 surface 与维度
    for p in image_files:
        rel = p.relative_to(root)
        parts = list(rel.parts[:-1])  # 目录部分
        info = classify_path(parts)
        surf = info.get("surface", "unknown")
        # surface 推断：取图片顶层目录（root 下第一层）作为 surface 候选
        top = parts[0] if parts else "unknown"
        if surf == "unknown":
            surf = top

        s = surfaces.setdefault(surf, {
            "defects": {},
            "scenes": {},
            "image_count": 0,
            "mask_count": 0,
            "sample_images": [],
        })
        s["image_count"] += 1
        if len(s["sample_images"]) < 5:
            s["sample_images"].append(str(rel))

        d = info.get("defect") or "unknown"
        s["defects"][d] = s["defects"].get(d, 0) + 1
        sc = info.get("scene") or "unknown"
        s["scenes"][sc] = s["scenes"].get(sc, 0) + 1

    # mask 归属
    for m in mask_files:
        rel = m.relative_to(root)
        parts = list(rel.parts[:-1])
        top = parts[0] if parts else "unknown"
        surfaces.setdefault(top, {
            "defects": {}, "scenes": {}, "image_count": 0, "mask_count": 0, "sample_images": [],
        })["mask_count"] += 1

    return {
        "root": str(root),
        "total_files": len(all_files),
        "total_images": len(image_files),
        "total_masks": len(mask_files),
        "total_other": len(other_files),
        "surfaces": surfaces,
        "other_file_exts": sorted({p.suffix.lower() or "(none)" for p in other_files}),
    }


# ------------------------------------------------------------- 主流程 ----
def main() -> int:
    ap = argparse.ArgumentParser(description="MSC-AD 下载/解压/生成 inventory")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--archive", type=str, help="本地压缩包路径")
    src.add_argument("--dir", type=str, help="已解压的数据目录路径")
    src.add_argument("--url", type=str, help="下载 URL")
    ap.add_argument("--out-dir", type=str, default=str(DATA_ROOT / "msc_ad"),
                    help="解压/输出目录（默认 data/msc_ad）")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    target_dir: Path
    archive_info: dict = {}

    if args.url:
        archive_path = download(args.url, out_dir / "downloads")
        archive_info = {"url": args.url, "path": str(archive_path),
                        "sha256": file_sha256(archive_path)}
        target_dir = extract(archive_path, out_dir / "raw")
    elif args.archive:
        archive_path = Path(args.archive).resolve()
        if not archive_path.exists():
            log(f"错误：压缩包不存在 {archive_path}")
            return 1
        archive_info = {"path": str(archive_path), "sha256": file_sha256(archive_path)}
        target_dir = extract(archive_path, out_dir / "raw")
    else:
        target_dir = Path(args.dir).resolve()
        if not target_dir.exists():
            log(f"错误：目录不存在 {target_dir}")
            return 1

    inv = scan_tree(target_dir)
    inv["archive"] = archive_info
    inv["generated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    inv["note"] = ("本文档为数据侦察结果，不涉及模型训练。dimension 推断为启发式，"
                   "confidence 非 high 的字段需人工复核。")

    # 写 JSON
    json_path = out_dir / "inventory.json"
    json_path.write_text(json.dumps(inv, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"inventory JSON -> {json_path}")

    # 写 Markdown 报告
    md_path = out_dir / "inventory_report.md"
    md_path.write_text(render_report(inv), encoding="utf-8")
    log(f"inventory 报告 -> {md_path}")

    # 打印摘要
    log(f"总文件 {inv['total_files']}，图片 {inv['total_images']}，mask {inv['total_masks']}")
    for surf, s in inv["surfaces"].items():
        log(f"  surface={surf}: {s['image_count']} 图, {s['mask_count']} mask, "
            f"defects={s['defects']}, scenes={s['scenes']}")

    return 0


def render_report(inv: dict) -> str:
    lines = ["# MSC-AD Dataset Inventory", ""]
    lines.append(f"- 生成时间：{inv['generated_at']}")
    lines.append(f"- 扫描根目录：`{inv['root']}`")
    lines.append(f"- 总文件数：{inv['total_files']}（图片 {inv['total_images']}，mask {inv['total_masks']}，其他 {inv['total_other']}）")
    if inv.get("archive"):
        a = inv["archive"]
        lines.append(f"- 来源压缩包：`{a.get('path', a.get('url'))}`")
        if a.get("sha256"):
            lines.append(f"- SHA256：`{a['sha256']}`")
    if inv.get("other_file_exts"):
        lines.append(f"- 其他文件扩展名：{', '.join(inv['other_file_exts'])}")
    lines.append("")
    lines.append("> 注意：dimension 推断为启发式，需下载后人工复核具体 surface/defect 名。")
    lines.append("")

    for surf, s in inv["surfaces"].items():
        lines.append(f"## Surface: `{surf}`")
        lines.append(f"- 图片数：{s['image_count']}，mask 数：{s['mask_count']}")
        lines.append(f"- 缺陷分布：{s['defects']}")
        lines.append(f"- 场景分布：{s['scenes']}")
        if s["sample_images"]:
            lines.append("- 样例图片：")
            for sp in s["sample_images"]:
                lines.append(f"  - `{sp}`")
        lines.append("")

    lines.append("## 待人工确认项")
    lines.append("")
    lines.append("- [ ] 6 种铸件表面的具体名称")
    lines.append("- [ ] 5 种缺陷类型的官方命名")
    lines.append("- [ ] 3 档光照的 low/mid/high 具体数值")
    lines.append("- [ ] 4 档分辨率的中间两档数值")
    lines.append("- [ ] train/test 划分是否官方给定，还是需自行切分")
    lines.append("- [ ] normal 样本在各光照档下是否成组（决定能否做跨光照 robustness）")
    lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    sys.exit(main())
