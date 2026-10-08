"""Experiment 12 — 批处理编排器（薄封装 experiment10_batch；Exp12 目录 + Exp12 runner）。

并发保证：experiment10_batch 已修为 `per_seed = max(1, workers // n_seeds)` →
总进程数 <= --workers（Exp11 曾因未限制出现 8 进程 OOM，此处沿用修复版）。
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import experiment10_batch as _b  # noqa: E402
import experiment10_progress as _p  # noqa: E402
import experiment12_progress as _p12  # noqa: E402

EXP12 = ROOT / "results" / "experiment_12"
_b.EXP = EXP12
_p.EXP = EXP12
_p.PLAN = EXP12 / "plan.json"


def launch(group: list, backend: str, mode: str, stage: str, wid: int) -> subprocess.Popen:
    log = EXP12 / "logs" / f"{stage}_{backend}_w{wid}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, "-u", str(ROOT / "scripts" / "experiment12_runner.py"),
           "--round", stage, "--mode", mode, "--worker-tag", f"w{wid}", "--units", ",".join(group)]
    fh = open(log, "ab")
    _b._HANDLES.append(fh)
    return subprocess.Popen(cmd, stdout=fh, stderr=subprocess.STDOUT, cwd=str(ROOT),
                            start_new_session=True)


_b.launch = launch

if __name__ == "__main__":
    _b.main()
