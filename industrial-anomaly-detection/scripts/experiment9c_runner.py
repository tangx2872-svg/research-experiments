"""Experiment 9C runner — Matched-RNG Calibration。

**不改动 9B / 5A-H 的任何代码**：本文件是薄驱动，导入 `experiment9b_runner`，
只做三件事：
  1. 把输出根切到 results/experiment_9c_rng_calibration/
  2. 增加 3 个校准单元 C0/C1/C2（复用 9B 已冻结的 config 定义）
  3. 注入 Protocol V2 的 matched-RNG wrapper（见 experiment9c_rng.py）

用法：
  python -u scripts/experiment9c_runner.py --round v2 --protocol v2 --worker-tag w0 \
      --units "bottle:0:C1_uniform_a025,bottle:0:C2_gate_const025"
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "experiments" / "2026-09-30_illumination_sensitivity_exploration" / "falpha_patchcore"))

import experiment5a_h_runner as h  # noqa: E402
import experiment9b_runner as r9b  # noqa: E402
import experiment9c_rng as r9c  # noqa: E402
import experiment_progress as ep  # noqa: E402

EXP_ROOT = ROOT / "results" / "experiment_9c_rng_calibration"
NINE_B_CACHE = ROOT / "results" / "experiment_9b_screening" / "cache"

# C0/C1/C2 直接复用 9B 已冻结的 config 定义（不新增数学）
C9C = {
    "C0_orig_a000": dict(r9b.CONFIGS["SMOKE_orig_a000"]),
    "C1_uniform_a025": dict(r9b.CONFIGS["SMOKE_uniform_a025"]),
    "C2_gate_const025": dict(r9b.CONFIGS["SMOKE_gate_const025"]),
}
for _c in C9C.values():
    _c["smoke"] = True          # 标记：校准单元，不是候选方法

_CFG = C9C["C2_gate_const025"]
assert _CFG["kind"] == "gate" and _CFG["gate_l2"][0] == "const" and _CFG["gate_l2"][1]["value"] == 0.25
assert C9C["C1_uniform_a025"]["kind"] == "alpha" and C9C["C1_uniform_a025"]["alpha_l2"] == 0.25

# ---- 输出路径重定向（9B 的其它行为不变）----
r9b.EXP_ROOT = EXP_ROOT
r9b.LOG_DIR = EXP_ROOT / "logs"
r9b.REF_DIR = EXP_ROOT / "reference"
if NINE_B_CACHE.exists():
    # 只读复用 9B 的 train-only s_c 缓存（文件名含内容哈希，(category, seed) 相同 -> 同一文件）
    r9b.CACHE_DIR = NINE_B_CACHE
r9b.CONFIGS.update(C9C)
r9b.CANDIDATES = list(r9b.CANDIDATES) + list(C9C)

_UNIT_CTX: dict = {}


def _unit_dir(round_name: str, unit: str) -> Path:
    cat, seed, name = unit.split(":")
    return EXP_ROOT / "raw" / round_name / cat / f"seed_{seed}" / f"config_{name}"


# 每个 unit 开始时把上下文交给 matched-RNG wrapper（用于证据落盘）
_orig_unit_start = ep.Progress.unit_start


def _unit_start(self, uid):  # noqa: ANN001
    cat, seed, name = uid.split(":")
    d = _unit_dir(getattr(self, "name", "round"), uid)
    r9c.clear_evidence()
    r9c.set_context(unit=uid, category=cat, seed=int(seed), out_dir=str(d))
    _UNIT_CTX.update({"round": getattr(self, "name", "round"), "unit": uid, "dir": d})
    return _orig_unit_start(self, uid)


ep.Progress.unit_start = _unit_start

# 9B 的 S2「输出目录隔离」按 experiment_9b_screening 硬编码；9C 重定向后需替换为 9C 断言
_orig_sanity = r9b.pre_run_sanity


def _sanity_9c(units, round_name):
    import csv as _csv
    res = _orig_sanity(units, round_name)
    ok = "experiment_9c_rng_calibration" in str(r9b.EXP_ROOT)
    res["S2_output_isolated"] = (ok, f"out root = {Path(r9b.EXP_ROOT).relative_to(ROOT)}")
    p_csv = r9b.LOG_DIR / f"sanity_pre_{round_name}.csv"
    try:
        r9b.LOG_DIR.mkdir(parents=True, exist_ok=True)
        with open(p_csv, "w", newline="") as f:
            w = _csv.writer(f)
            w.writerow(["check", "status", "detail"])
            for k, (okk, d) in res.items():
                w.writerow([k, "PASS" if okk else "FAIL", d])
    except Exception as exc:
        print(f"[warn] sanity csv rewrite failed: {exc}", flush=True)
    return res


r9b.pre_run_sanity = _sanity_9c

# fit 之后补记 memory bank / tau 证据（独立于 coreset wrapper）
_orig_fit = r9b.patched_fit_dual_model


def _fit_with_bank_evidence(alpha_l2, alpha_l3, category, seed, train_ids, logs_dir):
    model, tm = _orig_fit(alpha_l2, alpha_l3, category, seed, train_ids, logs_dir)
    try:
        bank = tm.memory_bank
        info = {}
        ip = Path(logs_dir).parent / "info.json"
        if ip.exists():
            info = json.loads(ip.read_text())
        ev = {"unit": r9c.get_context().get("unit"), "protocol": r9c.get_context().get("protocol"),
              "memory_bank_sha256": r9c.tensor_sha256(bank),
              "memory_bank_shape": list(bank.shape),
              "memory_bank_stats": r9c.tensor_stats(bank),
              "tau_val": info.get("tau_val"), "coreset_size": info.get("coreset_size"),
              "train_ids_hash": info.get("train_ids_hash"),
              "embedding_dim": int(bank.shape[1]) if bank.ndim == 2 else None,
              "ts": time.time()}
        out = Path(r9c.get_context().get("out_dir") or ".")
        out.mkdir(parents=True, exist_ok=True)
        (out / "bank_evidence.json").write_text(json.dumps(ev, indent=2, ensure_ascii=False))
    except Exception as exc:      # 证据失败不得影响实验
        print(f"[warn] bank evidence failed: {type(exc).__name__}: {exc}", flush=True)
    return model, tm


h.fit_dual_model = _fit_with_bank_evidence

# 同时为 V1 协议保留原始 fit（不加 wrapper 的证据），由 --protocol 决定 wrapper 是否恢复 RNG
r9c.install_matched_rng(seed=0, out_dir=str(EXP_ROOT / "raw"), protocol="v2")
r9c.install_fit_replay()   # V2 的第二个 replay 点：固定 train DataLoader shuffle 行序


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--units", required=True)
    ap.add_argument("--round", required=True, dest="round_name")
    ap.add_argument("--worker-tag", default="w0")
    ap.add_argument("--protocol", default="v2", choices=["v1", "v2"])
    args = ap.parse_args()

    for tok in args.units.split(","):
        _cat, _seed, name = tok.strip().split(":")
        if name not in r9b.CONFIGS:
            raise SystemExit(f"unknown config {name}")
    seeds = {int(tok.split(":")[1]) for tok in args.units.split(",")}
    if len(seeds) != 1:
        raise SystemExit("single seed per worker required")
    r9c.install_matched_rng(seed=seeds.pop(), out_dir=str(EXP_ROOT / "raw"), protocol=args.protocol)
    r9c.install_fit_replay()

    print(f"[9C] protocol={args.protocol} round={args.round_name} worker={args.worker_tag} "
          f"units={args.units}", flush=True)
    sys.argv = ["experiment9c_runner.py", "--units", args.units,
                "--round", args.round_name, "--worker-tag", args.worker_tag]
    r9b.main()


if __name__ == "__main__":
    main()
