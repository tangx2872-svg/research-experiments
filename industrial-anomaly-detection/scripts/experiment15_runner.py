"""Experiment 15 runner —— 薄驱动：复用 Exp14 runner 的全部协议层，只加 X4a/X4b spec 与输出根重定向。"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import experiment14_candidates as c14  # noqa: E402
import experiment14_runner as r14  # noqa: E402
import experiment15_candidates as c15  # noqa: E402

EXP15 = ROOT / "results" / "experiment_15"

c15.register_into_env14()
_orig_verify = c14.verify_frozen


def _verify_with_x4():
    ok, det = _orig_verify()
    ok2, det2 = c15.verify()
    return bool(ok and ok2), f"{det} || {det2}"


c14.verify_frozen = _verify_with_x4
r14.EXP = EXP15

# Exp14 的 R4 输出隔离断言硬编码 "experiment_14"；Exp15 用真实目录断言替换（不算放宽：仍强制隔离）
import experiment10_runner as r10  # noqa: E402

_orig_sanity = r10._sanity_override


def _sanity_override_15(module, key, ok, detail):
    root = str(getattr(module, "OUT_ROOT", "") or getattr(module, "EXP_ROOT", ""))
    real = ("experiment_15" in root) and ("experiment_14" not in root)
    return _orig_sanity(module, key, real, f"Exp15 output root = {root}")


r10._sanity_override = _sanity_override_15

if __name__ == "__main__":
    r14.main()
