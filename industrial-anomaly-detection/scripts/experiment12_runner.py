"""Experiment 12 runner — Category × Channel adaptive compensation。

**零新模型代码**：复用 experiment11_model.AdaptivePatchcore 的 `residual_gated`
（L2 = F + beta * g_j * IN(F)*rms(F)），只换 spec 注册表（Exp12 registry）。
协议/证据层全部复用：experiment9c_rng（strict-V2）+ experiment10_protocol（corrected-189）
+ experiment10_runner 的 `_common` / `_sanity_override` / `_install_evidence`。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "experiments" / "2026-09-30_illumination_sensitivity_exploration" / "falpha_patchcore"))

import experiment5a_h_runner as h  # noqa: E402
import experiment10_runner as r10  # noqa: E402
import experiment12_candidates as c12  # noqa: E402

EXP = ROOT / "results" / "experiment_12"


def _fix_r6(module) -> None:
    """R6：白名单 + 强制核验 beta/g 与冻结注册表一致（Exp12：g ≡1 时须退化为 B2）。"""
    import numpy as np

    orig = module.pre_run_sanity

    def wrapped(*a, **kw):
        res = orig(*a, **kw)
        allowed = ("alpha_in", "residual", "energy", "concat_dual", "altnorm_strength",
                   "residual_gated")
        ok = all(v["l2"]["kind"] in allowed and v["l3"]["kind"] in allowed
                 for v in module.ALL_SPECS.values())
        bt, bad = c12.b2_beta(), []
        for name, sp in c12.SEVEN_AO_SPECS.items():
            cid, cat = name.split("__")
            l2 = sp["l2"]
            good = (abs(float(l2["beta"]) - bt[cat]) < 1e-12 and len(l2["g"]) == 512
                    and abs(float(np.mean(l2["g"])) - float(c12.gate(cid, cat).mean())) < 1e-9)
            if not good:
                bad.append(name)
        res["R6_no_target_dependent_params"] = (
            bool(ok and not bad),
            f"all spec kinds in frozen whitelist (incl. residual_gated); "
            f"beta/g match frozen Exp12 registry: {not bad} {bad[:3]}")
        return res

    module.pre_run_sanity = wrapped


def _run_7ao(args, seed: int) -> None:
    import experiment7ao_runner as r7
    from experiment11_model import AdaptivePatchcore

    r7.ModulePatchcore = AdaptivePatchcore
    r7.ALL_SPECS.update(c12.SEVEN_AO_SPECS)
    out_root = EXP / "raw" / args.round_name
    r7.LOGS_DIR = EXP / "logs"
    r7.OUT_ROOT = out_root
    h.OUT_ROOT = out_root
    h.LOGS_DIR = r7.LOGS_DIR
    Path(r7.LOGS_DIR).mkdir(parents=True, exist_ok=True)
    for name in (t.split(":")[2] for t in args.units.split(",")):
        if name not in r7.ALL_SPECS:
            raise SystemExit(f"unknown 12 spec {name}")
    r10._common(args.mode, seed, args.round_name)
    r10._sanity_override(r7, "R4_output_isolated", "experiment_12" in str(out_root),
                         f"output root = {out_root.relative_to(ROOT)}")
    _fix_r6(r7)
    sys.argv = ["experiment12_runner.py", "--units", args.units,
                "--worker-tag", args.worker_tag, "--out-root", str(out_root)]
    print(f"[Exp12] backend=7ao mode={args.mode} round={args.round_name} "
          f"worker={args.worker_tag} units={args.units}", flush=True)
    r7.main()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--units", required=True)
    ap.add_argument("--round", required=True, dest="round_name")
    ap.add_argument("--worker-tag", default="w0")
    ap.add_argument("--mode", default="corrected189", choices=["corrected189", "historical209"])
    ap.add_argument("--protocol", default="v2", choices=["v2"])
    args = ap.parse_args()
    seeds = {int(t.split(":")[1]) for t in args.units.split(",")}
    if len(seeds) != 1:
        raise SystemExit("single seed per worker required")
    _run_7ao(args, seeds.pop())


if __name__ == "__main__":
    main()
