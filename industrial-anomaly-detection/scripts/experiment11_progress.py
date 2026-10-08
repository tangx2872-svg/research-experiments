"""Experiment 11 — 进度/ETA 监视器（薄封装 experiment10_progress，输出根与 plan 改为 Exp11）。"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import experiment10_progress as _p  # noqa: E402

_p.EXP = ROOT / "results" / "experiment_11_c2_adaptive"
_p.PLAN = _p.EXP / "plan.json"


def render(write: bool = True) -> str:
    return _p.render(write=write)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--watch", type=int, default=0)
    a = ap.parse_args()
    if a.watch:
        while True:
            print(render(), flush=True)
            time.sleep(a.watch)
    else:
        print(render())
