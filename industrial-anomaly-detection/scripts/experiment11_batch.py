"""Experiment 11 — 批处理编排器（薄封装 experiment10_batch；改为 Exp11 目录与 Exp11 runner）。"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import experiment10_batch as _b  # noqa: E402
import experiment10_progress as _p  # noqa: E402
import experiment11_progress as _p11  # noqa: E402  (设置 EXP/PLAN)

EXP11 = ROOT / "results" / "experiment_11_c2_adaptive"
_b.EXP = EXP11
_p.EXP = EXP11
_p.PLAN = EXP11 / "plan.json"


def launch(group: list, backend: str, mode: str, stage: str, wid: int) -> subprocess.Popen:
    log = EXP11 / "logs" / f"{stage}_{backend}_w{wid}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, "-u", str(ROOT / "scripts" / "experiment11_runner.py"),
           "--backend", backend, "--round", stage, "--mode", mode,
           "--worker-tag", f"w{wid}", "--units", ",".join(group)]
    fh = open(log, "ab")
    _b._HANDLES.append(fh)
    return subprocess.Popen(cmd, stdout=fh, stderr=subprocess.STDOUT, cwd=str(ROOT),
                            start_new_session=True)


_b.launch = launch

if __name__ == "__main__":
    _b.main()
