#!/usr/bin/env python
"""E4-D0 §20 — visual sanity contact sheet (CPU only).

Shows, for ONE normal specimen and ONE defect specimen, the SAME view across
illuminations 01..10, so a human can confirm that the illumination id really
corresponds to visibly different imaging conditions.

Usage:
  python scripts/e4d0_contact_sheet.py \
      --extract-root data/m2ad/extracted --category Bird \
      --out docs/figures/e4_m2ad_pairing_sanity.png
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--extract-root", required=True)
    ap.add_argument("--metadata", default=None)
    ap.add_argument("--category", default="Bird")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    root = Path(a.extract_root) / a.category
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    goods = sorted(p.name for p in (root / "Good").iterdir() if p.is_dir())
    ngs = sorted(p.name for p in (root / "NG").iterdir() if p.is_dir())

    import random
    rnd = random.Random(a.seed)
    good, ng = rnd.choice(goods), rnd.choice(ngs)

    # pick the first illumination-complete view shared by both (deterministic)
    illums = [f"{i:02d}" for i in range(1, 11)]
    views = sorted(p.stem[1:4] for p in (root / "Good" / good).glob("A*_I01.png"))
    view = views[0]

    def img_path(spec, kind, il):
        return root / kind / spec / f"A{view}_I{il}.png"

    # defect mask (shared across illuminations, one per (specimen, view))
    mask_p = root / "GT" / ng / f"A{view}_mask.png"
    seg_p = root / "GT" / ng / f"A{view}_seg.png"

    fig, axes = plt.subplots(3, 10, figsize=(26, 9.5))
    for j, il in enumerate(illums):
        for i, (spec, kind, title) in enumerate(
                [(good, "Good", "NORMAL"), (ng, "NG", "DEFECT")]):
            p = img_path(spec, kind, il)
            ax = axes[i, j]
            if p.exists():
                ax.imshow(Image.open(p).convert("RGB"))
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_title(f"I{il}", fontsize=9)
            if j == 0:
                ax.set_ylabel(f"{title}\n{spec}\nview {view}", fontsize=9)
    # row 3: mask + seg preview for the defect specimen
    for j, il in enumerate(illums):
        ax = axes[2, j]
        if j == 0:
            ax.set_ylabel("MASK / SEG\n(shared per view)", fontsize=9)
        if j == 1 and mask_p.exists():
            ax.imshow(Image.open(mask_p).convert("L"), cmap="gray", vmin=0, vmax=255)
            ax.set_title("mask", fontsize=9)
        elif j == 2 and seg_p.exists():
            ax.imshow(Image.open(seg_p).convert("L"), cmap="gray", vmin=0, vmax=255)
            ax.set_title("seg", fontsize=9)
        else:
            ax.axis("off")
        ax.set_xticks([]); ax.set_yticks([])

    fig.suptitle(
        f"E4-D0 M2AD illumination sanity — category={a.category}  "
        f"normal={good}  defect={ng}  view={view}  (rows: normal / defect / GT, cols: I01..I10)",
        fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out, dpi=110)
    print(f"[contact sheet] {out}")

    info = {"category": a.category, "normal_specimen": good, "defect_specimen": ng,
            "view": view, "illuminations": illums,
            "mask_exists": mask_p.exists(), "seg_exists": seg_p.exists()}
    print(json.dumps(info, ensure_ascii=False))
    (out.parent / "e4_m2ad_pairing_sanity.json").write_text(json.dumps(info, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
