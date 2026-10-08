"""Experiment 16 runner —— 薄驱动：复用 Exp14 runner 的全部协议层（corrected-189 + strict-V2），
只把输出根重定向到 results/experiment_16_final_validation/，并修正 Exp14 硬编码的输出隔离断言。"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import experiment10_runner as r10  # noqa: E402
import experiment14_runner as r14  # noqa: E402

EXP16 = ROOT / "results" / "experiment_16_final_validation"
r14.EXP = EXP16

_orig_sanity = r10._sanity_override


def _sanity_override_16(module, key, ok, detail):
    root = str(getattr(module, "OUT_ROOT", "") or getattr(module, "EXP_ROOT", ""))
    real = ("experiment_16_final_validation" in root) and ("experiment_14" not in root)
    return _orig_sanity(module, key, real, f"Exp16 output root = {root}")


r10._sanity_override = _sanity_override_16

if __name__ == "__main__":
    r14.main()
