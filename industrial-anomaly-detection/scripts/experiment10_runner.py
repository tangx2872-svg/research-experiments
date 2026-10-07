"""Experiment 10 runner — Preservation Recovery Tournament。

复用（零重写）：
  * Matched-RNG Protocol V2 : scripts/experiment9c_rng.py（R1 Engine.fit 入口 + R2 coreset 入口）
  * 两个 backend 的冻结实现  : experiment9b_runner（per-layer alpha / channel gate）
                              experiment7ao_runner（spec step；本实验把模型换成 RecoveryPatchcore）
  * corrected-189 协议       : scripts/experiment10_protocol.py（修 9B-R §11 的 dm.setup 重置问题）

用法：
  python -u scripts/experiment10_runner.py --backend 9b --round s2a --mode corrected189 \
      --worker-tag w0 --units "bottle:0:P10_ORIG_a000,bottle:0:P10_T1_L2a000_L3a025"
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
import experiment9c_rng as r9c  # noqa: E402
import experiment10_candidates as c10  # noqa: E402
import experiment10_protocol as p10  # noqa: E402

EXP = ROOT / "results" / "experiment_10_preservation_recovery"
_PROTO = "v2"


def _write(out_dir, fname: str, rec: dict) -> None:
    try:
        d = Path(out_dir or ".")
        d.mkdir(parents=True, exist_ok=True)
        (d / fname).write_text(json.dumps(rec, indent=2, ensure_ascii=False))
    except Exception as exc:
        print(f"[warn] evidence write failed ({fname}): {exc}", flush=True)


def _install_evidence(round_name: str) -> None:
    """证据层：RNG / embedding(逐位) / memory bank / train-filter 实际图像数。"""
    from anomalib.models.components.sampling.k_center_greedy import KCenterGreedy

    orig_run_config = h.run_config

    def run_config_cfg(cfg, category, seed, val_ids, train_ids, good_paths,
                       defect_paths, device, cfg_dir):
        r9c.clear_evidence()
        r9c.set_context(unit=f"{category}:{seed}:{cfg['name']}", category=category,
                        seed=int(seed), out_dir=str(cfg_dir), protocol=_PROTO, round=round_name)
        p10.set_context(category=category, seed=int(seed), out_dir=str(cfg_dir),
                        train_ids=list(train_ids), mode=p10.get_context().get("mode", "corrected189"))
        return orig_run_config(cfg, category, seed, val_ids, train_ids, good_paths,
                               defect_paths, device, cfg_dir)

    h.run_config = run_config_cfg

    prev_select = KCenterGreedy.select_coreset_idxs

    def select_with_evidence(self):  # noqa: ANN001
        ctx = r9c.get_context()
        unit = str(ctx.get("unit") or "unit").replace(":", "_")
        _write(ctx.get("out_dir"), f"embedding_evidence_{unit}.json",
               {"unit": ctx.get("unit"), "protocol": ctx.get("protocol"),
                "n_observations": int(self.n_observations),
                "embedding_shape": list(self.embedding.shape),
                "embedding_dtype": str(self.embedding.dtype),
                "embedding_sha256": r9c.tensor_sha256(self.embedding),
                "embedding_stats": r9c.tensor_stats(self.embedding), "ts": time.time()})
        return prev_select(self)

    KCenterGreedy.select_coreset_idxs = select_with_evidence

    prev_fit = h.fit_dual_model

    def fit_with_bank_evidence(alpha_l2, alpha_l3, category, seed, train_ids, logs_dir):
        model, tm = prev_fit(alpha_l2, alpha_l3, category, seed, train_ids, logs_dir)
        ctx = r9c.get_context()
        bank = tm.memory_bank
        unit = str(ctx.get("unit") or "unit").replace(":", "_")
        _write(ctx.get("out_dir"), f"bank_evidence_{unit}.json",
               {"unit": ctx.get("unit"), "protocol": ctx.get("protocol"),
                "memory_bank_sha256": r9c.tensor_sha256(bank),
                "memory_bank_shape": list(bank.shape),
                "memory_bank_stats": r9c.tensor_stats(bank),
                "embedding_dim": int(bank.shape[1]) if bank.ndim == 2 else None,
                "ts": time.time()})
        return model, tm

    h.fit_dual_model = fit_with_bank_evidence


def _sanity_override(module, key: str, ok: bool, detail: str) -> None:
    import csv as _csv

    orig = module.pre_run_sanity

    def wrapped(*a, **kw):
        res = orig(*a, **kw)
        res[key] = (ok, detail)
        try:
            with open(Path(module.LOGS_DIR) / "sanity_pre_exp10.csv", "w", newline="") as f:
                w = _csv.writer(f)
                w.writerow(["check", "status", "detail"])
                for k, (okk, d) in res.items():
                    w.writerow([k, "PASS" if okk else "FAIL", d])
        except Exception as exc:
            print(f"[warn] sanity csv rewrite failed: {exc}", flush=True)
        return res

    module.pre_run_sanity = wrapped


def _common(mode: str, seed: int, round_name: str) -> None:
    r9c.install_matched_rng(seed=seed, out_dir=str(EXP / "raw"), protocol=_PROTO)
    r9c.install_fit_replay()
    p10.set_context(mode=mode)
    p10.install_corrected_train_filter()
    _install_evidence(round_name)


def _run_9b(args, seed: int) -> None:
    import experiment9b_runner as be

    be.CONFIGS.update(c10.NINE_B_CONFIGS)
    be.EXP_ROOT = EXP
    be.LOG_DIR = EXP / "logs"
    be.REF_DIR = EXP / "reference"
    be.EXP_ROOT.mkdir(parents=True, exist_ok=True)
    for name in (t.split(":")[2] for t in args.units.split(",")):
        if name not in be.CONFIGS:
            raise SystemExit(f"unknown 9B config {name}")
    be.CANDIDATES = list(be.CANDIDATES) + list(c10.NINE_B_CONFIGS)
    _common(args.mode, seed, args.round_name)
    _sanity_override(be, "S2_output_isolated", "experiment_10_preservation_recovery" in str(be.EXP_ROOT),
                     f"out root = {be.EXP_ROOT.relative_to(ROOT)}")
    sys.argv = ["experiment10_runner.py", "--units", args.units,
                "--round", args.round_name, "--worker-tag", args.worker_tag]
    print(f"[Exp10] backend=9b mode={args.mode} round={args.round_name} worker={args.worker_tag} "
          f"units={args.units}", flush=True)
    be.main()


def _run_7ao(args, seed: int) -> None:
    import experiment7ao_config as c7
    import experiment7ao_runner as r7
    from experiment10_model import RecoveryPatchcore

    r7.ModulePatchcore = RecoveryPatchcore                      # 模型换成含新 step 的子类
    r7.ALL_SPECS.update(c10.SEVEN_AO_SPECS)
    out_root = EXP / "raw" / args.round_name
    r7.LOGS_DIR = EXP / "logs"
    r7.OUT_ROOT = out_root
    h.OUT_ROOT = out_root
    h.LOGS_DIR = r7.LOGS_DIR
    Path(r7.LOGS_DIR).mkdir(parents=True, exist_ok=True)
    for name in (t.split(":")[2] for t in args.units.split(",")):
        if name not in r7.ALL_SPECS:
            raise SystemExit(f"unknown 7A-O spec {name}")
    _common(args.mode, seed, args.round_name)
    _sanity_override(r7, "R4_output_isolated", "experiment_10_preservation_recovery" in str(out_root),
                     f"output root = {out_root.relative_to(ROOT)}")
    sys.argv = ["experiment10_runner.py", "--units", args.units,
                "--worker-tag", args.worker_tag, "--out-root", str(out_root)]
    print(f"[Exp10] backend=7ao mode={args.mode} round={args.round_name} worker={args.worker_tag} "
          f"units={args.units}", flush=True)
    r7.main()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", required=True, choices=["9b", "7ao"])
    ap.add_argument("--units", required=True)
    ap.add_argument("--round", required=True, dest="round_name")
    ap.add_argument("--worker-tag", default="w0")
    ap.add_argument("--mode", default="corrected189", choices=["corrected189", "historical209"])
    ap.add_argument("--protocol", default="v2", choices=["v2"])
    args = ap.parse_args()
    seeds = {int(t.split(":")[1]) for t in args.units.split(",")}
    if len(seeds) != 1:
        raise SystemExit("single seed per worker required")
    if args.backend == "9b":
        _run_9b(args, seeds.pop())
    else:
        _run_7ao(args, seeds.pop())


if __name__ == "__main__":
    main()
