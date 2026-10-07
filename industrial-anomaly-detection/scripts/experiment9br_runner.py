"""Experiment 9B-R runner — Strict Replay Re-evaluation（Protocol V2）。

**不重写任何 strict replay 逻辑**：直接复用 9C-v2b 已验证的实现
  scripts/experiment9c_rng.py  →  install_matched_rng(R2) + install_fit_replay(R1)
并按 9C 的「薄驱动」模式重定向输出到 results/experiment_9b_r_strict_replay/。

两个 backend（按候选来源选择，一个 worker 只用一个 backend）：
  --backend 9b  : experiment9b_runner（REF_original / T1_M7_* / T3_M7_*）
  --backend 7ao : experiment7ao_runner（T2_M10_concat_g100 ≡ 7A-O B1_g100）

用法：
  python -u scripts/experiment9br_runner.py --backend 9b --round r1 --protocol v2 \
      --worker-tag w0 --units "bottle:0:REF_original,bottle:0:T1_M7_L2a000_L3a025"
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

EXP = ROOT / "results" / "experiment_9b_r_strict_replay"

NINE_B_ALIASES = {"REF_original": "SMOKE_orig_a000",
                  "REF_original_repeat": "SMOKE_orig_a000",
                  "T1_M7_L2a000_L3a025": "M7_L2a000_L3a025",
                  "T3_M7_L2a025_L3a000": "M7_L2a025_L3a000"}
SEVEN_AO_ALIASES = {"T2_M10_concat_g100": "B1_g100"}
_PROTO = "v2"


def _write(out_dir, fname: str, rec: dict) -> None:
    try:
        d = Path(out_dir or ".")
        d.mkdir(parents=True, exist_ok=True)
        (d / fname).write_text(json.dumps(rec, indent=2, ensure_ascii=False))
    except Exception as exc:
        print(f"[warn] evidence write failed ({fname}): {exc}", flush=True)


def _install_evidence(round_name: str) -> None:
    """证据层：叠加在 9C 的 RNG 控制之上，只增证据、不改数值。"""
    from anomalib.models.components.sampling.k_center_greedy import KCenterGreedy

    orig_run_config = h.run_config

    def run_config_cfg(cfg, category, seed, val_ids, train_ids, good_paths,
                       defect_paths, device, cfg_dir):
        r9c.clear_evidence()
        r9c.set_context(unit=f"{category}:{seed}:{cfg['name']}", category=category,
                        seed=int(seed), out_dir=str(cfg_dir), protocol=_PROTO,
                        round=round_name)
        return orig_run_config(cfg, category, seed, val_ids, train_ids, good_paths,
                               defect_paths, device, cfg_dir)

    h.run_config = run_config_cfg

    prev_select = KCenterGreedy.select_coreset_idxs

    def select_with_embedding_evidence(self):  # noqa: ANN001
        ctx = r9c.get_context()
        unit = str(ctx.get("unit") or "unit").replace(":", "_")
        _write(ctx.get("out_dir"), f"embedding_evidence_{unit}.json",
               {"unit": ctx.get("unit"), "protocol": ctx.get("protocol"),
                "n_observations": int(self.n_observations),
                "embedding_shape": list(self.embedding.shape),
                "embedding_dtype": str(self.embedding.dtype),
                "embedding_device": str(self.embedding.device),
                "embedding_sha256": r9c.tensor_sha256(self.embedding),
                "embedding_stats": r9c.tensor_stats(self.embedding), "ts": time.time()})
        return prev_select(self)

    KCenterGreedy.select_coreset_idxs = select_with_embedding_evidence

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
    """替换 backend sanity 中按历史目录硬编码的输出隔离检查。"""
    import csv as _csv

    orig = module.pre_run_sanity

    def wrapped(*a, **kw):
        res = orig(*a, **kw)
        res[key] = (ok, detail)
        try:
            with open(Path(module.LOGS_DIR) / "sanity_pre_9br.csv", "w", newline="") as f:
                w = _csv.writer(f)
                w.writerow(["check", "status", "detail"])
                for k, (okk, d) in res.items():
                    w.writerow([k, "PASS" if okk else "FAIL", d])
        except Exception as exc:
            print(f"[warn] sanity csv rewrite failed: {exc}", flush=True)
        return res

    module.pre_run_sanity = wrapped


def _run_9b(args, seed: int) -> None:
    import experiment9b_runner as be

    for alias, base in NINE_B_ALIASES.items():
        if base not in be.CONFIGS:
            raise SystemExit(f"missing 9B base config {base}")
        c = dict(be.CONFIGS[base])
        c.pop("smoke", None)
        be.CONFIGS[alias] = c
    be.EXP_ROOT = EXP
    be.LOG_DIR = EXP / "logs"
    be.REF_DIR = EXP / "reference"
    be.EXP_ROOT.mkdir(parents=True, exist_ok=True)
    for name in (t.split(":")[2] for t in args.units.split(",")):
        if name not in be.CONFIGS:
            raise SystemExit(f"unknown 9B config {name}")
    be.CANDIDATES = list(be.CANDIDATES) + list(NINE_B_ALIASES)

    r9c.install_matched_rng(seed=seed, out_dir=str(EXP / "raw"), protocol=_PROTO)
    r9c.install_fit_replay()
    _install_evidence(args.round_name)
    _sanity_override(be, "S2_output_isolated",
                     "experiment_9b_r_strict_replay" in str(be.EXP_ROOT),
                     f"out root = {be.EXP_ROOT.relative_to(ROOT)}")
    sys.argv = ["experiment9br_runner.py", "--units", args.units,
                "--round", args.round_name, "--worker-tag", args.worker_tag]
    print(f"[9B-R] backend=9b protocol={_PROTO} round={args.round_name} "
          f"worker={args.worker_tag} units={args.units}", flush=True)
    be.main()


def _run_7ao(args, seed: int) -> None:
    import experiment7ao_config as c7
    import experiment7ao_runner as r7

    for alias, base in SEVEN_AO_ALIASES.items():
        if base not in c7.MODULE_SPECS:
            raise SystemExit(f"missing 7A-O base spec {base}")
        r7.ALL_SPECS[alias] = dict(c7.MODULE_SPECS[base])
    out_root = EXP / "raw" / args.round_name
    r7.LOGS_DIR = EXP / "logs"
    r7.OUT_ROOT = out_root
    h.OUT_ROOT = out_root
    h.LOGS_DIR = r7.LOGS_DIR
    Path(r7.LOGS_DIR).mkdir(parents=True, exist_ok=True)
    for name in (t.split(":")[2] for t in args.units.split(",")):
        if name not in r7.ALL_SPECS:
            raise SystemExit(f"unknown 7A-O spec {name}")

    r9c.install_matched_rng(seed=seed, out_dir=str(EXP / "raw"), protocol=_PROTO)
    r9c.install_fit_replay()
    _install_evidence(args.round_name)
    _sanity_override(r7, "R4_output_isolated",
                     "experiment_9b_r_strict_replay" in str(out_root),
                     f"output root = {out_root.relative_to(ROOT)}")
    sys.argv = ["experiment9br_runner.py", "--units", args.units,
                "--worker-tag", args.worker_tag, "--out-root", str(out_root)]
    print(f"[9B-R] backend=7ao protocol={_PROTO} round={args.round_name} "
          f"worker={args.worker_tag} units={args.units}", flush=True)
    r7.main()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", required=True, choices=["9b", "7ao"])
    ap.add_argument("--units", required=True)
    ap.add_argument("--round", required=True, dest="round_name")
    ap.add_argument("--worker-tag", default="w0")
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
