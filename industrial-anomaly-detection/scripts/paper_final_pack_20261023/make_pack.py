#!/usr/bin/env python
"""把投稿前收尾核查打成 ZIP（不含权重/原图/大缓存）。

仓库内（results/paper_final_pack_20261023/，可入库）：叙述文档、结果表、统计 JSON。
ZIP 内额外包含：per_image/（已跟踪 cell 的字节副本）与 code/（本轮脚本副本），
以及只存在于 ZIP 里的 MANIFEST_ZIP.txt（对全部 zip 条目的 SHA-256）。

不写入仓库的：per_image/ 与 code/（分别可从 results/e2x/cells 与 scripts/ 重建），
以及 ZIP 本体。因此仓库目录保持小而自洽，MANIFEST_SHA256.txt 覆盖仓库内文件。

用法：
    python scripts/paper_final_pack_20261023/make_pack.py
    PACK_ZIP=/tmp/x.zip python scripts/paper_final_pack_20261023/make_pack.py
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import shutil
import subprocess
import tempfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PACK = ROOT / "results" / "paper_final_pack_20261023"
SCRIPTS = ROOT / "scripts" / "paper_final_pack_20261023"
ZIP = Path(os.environ.get("PACK_ZIP", str(ROOT / "results" / "paper_final_pack_20261023.zip")))

INPUTS = [
    "results/e22/cells", "results/e25/cells", "results/e17/summary.csv",
    "results/e17/final_method.json", "results/e13a/e13a_partial.json",
    "results/e18/main_table.md", "results/e18/protocol.md",
    "results/e18/illum_breakdown.csv", "results/e21/analysis_cells.csv",
    "results/e22/main_table_10cat_3seed.csv", "results/e22/illumination_summary.csv",
    "results/e22/illumination_breakdown.csv", "results/e22/illumination_spread.csv",
    "results/e23/doll_illumination_breakdown.csv", "results/e23/doll_seed_summary.csv",
    "results/paper_ready/EVIDENCE_MAP.md",
]


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True,
                          text=True).stdout.strip()


def versions() -> dict:
    v = {"python": platform.python_version(), "platform": platform.platform()}
    for mod in ("numpy", "sklearn", "torch", "scipy"):
        try:
            m = __import__(mod)
            v[mod] = getattr(m, "__version__", "unknown")
        except Exception as exc:  # noqa: BLE001
            v[mod] = f"import-failed: {type(exc).__name__}"
    return v


def main() -> int:
    PACK.mkdir(parents=True, exist_ok=True)

    # 1. 来源与版本（只写 PACK 内，随仓库走）
    inputs = {}
    for rel in INPUTS:
        p = ROOT / rel
        if p.is_dir():
            for q in sorted(p.glob("*.json")):
                inputs[str(q.relative_to(ROOT))] = sha256(q)
        else:
            inputs[rel] = sha256(p)
    prov = {
        "created_utc": subprocess.run(["date", "-u", "+%Y-%m-%dT%H:%M:%SZ"],
                                      capture_output=True, text=True).stdout.strip(),
        "git_head_at_pack_time": git("rev-parse", "HEAD"),
        "git_branch": git("rev-parse", "--abbrev-ref", "HEAD"),
        "git_status_porcelain": git("status", "--porcelain").splitlines(),
        "versions": versions(),
        "input_sha256": inputs,
        "excluded": ["model weights (~0.42 GB)", "M2AD/RobustAD original images",
                     "l0_448 feature cache", "figures/PDFs", "per_image copies (in ZIP only)"],
        "note": "cell JSON 内含 scores_f0/scores_r1/labels/detectable_raw/specimen/"
                "illumination/kind，足以在无原图条件下复算 AUROC/AP 与本文全部统计。",
    }
    (PACK / "provenance.json").write_text(json.dumps(prov, indent=1, ensure_ascii=False) + "\n")

    # 2. 仓库内清单（排除自身）
    man = PACK / "MANIFEST_SHA256.txt"
    if man.exists():
        man.unlink()
    lines = []
    for p in sorted(PACK.rglob("*")):
        if p.is_file() and p != man and p.suffix != ".zip":
            lines.append(f"{sha256(p)}  {p.relative_to(PACK).as_posix()}")
    man.write_text("\n".join(lines) + "\n")

    # 3. 暂存 ZIP 专有内容（不进仓库）
    stage = Path(tempfile.mkdtemp(prefix="pack_stage_"))
    try:
        for name, src in (("e22_cells", ROOT / "results" / "e22" / "cells"),
                          ("e25_cells", ROOT / "results" / "e25" / "cells")):
            dst = stage / "per_image" / name
            dst.mkdir(parents=True, exist_ok=True)
            for p in sorted(src.glob("*.json")):
                shutil.copy2(p, dst / p.name)
        dst = stage / "code"
        dst.mkdir(parents=True, exist_ok=True)
        for p in sorted(SCRIPTS.glob("*.py")):
            shutil.copy2(p, dst / p.name)

        entries = [(p.relative_to(PACK).as_posix(), p) for p in sorted(PACK.rglob("*"))
                   if p.is_file() and p.suffix != ".zip"]
        entries += [(p.relative_to(stage).as_posix(), p) for p in sorted(stage.rglob("*"))
                    if p.is_file()]
        zip_manifest = stage / "MANIFEST_ZIP.txt"
        zip_manifest.write_text("\n".join(f"{sha256(p)}  {name}" for name, p in sorted(entries)) + "\n")
        entries.append(("MANIFEST_ZIP.txt", zip_manifest))

        if ZIP.exists():
            ZIP.unlink()
        ZIP.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(ZIP, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as z:
            for name, p in entries:
                z.write(p, name)
    finally:
        shutil.rmtree(stage, ignore_errors=True)

    size = ZIP.stat().st_size
    print(f"[pack] dir  = {PACK}  ({len(lines)} repo files)")
    print(f"[zip ] path = {ZIP}")
    print(f"[zip ] entries = {len(entries)}")
    print(f"[zip ] size = {size} bytes ({size/2**20:.2f} MiB)")
    print(f"[zip ] sha256 = {sha256(ZIP)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
