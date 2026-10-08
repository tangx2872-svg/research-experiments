"""Experiment 11 runner — 复用 Exp10 的协议/证据层，只换 7A-O spec 注册表与模型类。

复用（零重写）：experiment9c_rng（strict-V2）、experiment10_protocol（corrected-189）、
experiment10_runner 的 `_install_evidence` / `_sanity_override` / `_common` 证据层。
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
import experiment11_candidates as c11  # noqa: E402

EXP = ROOT / "results" / "experiment_11_c2_adaptive"


def _fix_r6(module) -> None:
    """Exp11 的 R6：白名单加入 residual_gated，并**强制核验** beta/g 与冻结注册表一致。"""
    orig = module.pre_run_sanity

    def wrapped(*a, **kw):
        res = orig(*a, **kw)
        allowed = ("alpha_in", "residual", "energy", "concat_dual", "altnorm_strength",
                   "residual_gated")
        ok = all(v["l2"]["kind"] in allowed and v["l3"]["kind"] in allowed
                 for v in module.ALL_SPECS.values())
        bt = c11.beta_table()
        ok2, detail = True, []
        for name, sp in c11.SEVEN_AO_SPECS.items():
            cid, cat = name.split("__")
            base = "E11_B2" if cid.startswith("E11_B2") else cid   # ablation 变体沿用 E11_B2 的 beta
            l2 = sp["l2"]
            if l2["kind"] == "residual":
                good = abs(float(l2["lambda"]) - bt[(base, cat)]) < 1e-12
            else:
                good = (abs(float(l2["beta"]) - 0.25) < 1e-12 and len(l2["g"]) == 512)
            if not good:
                ok2 = False
                detail.append(name)
        res["R6_no_target_dependent_params"] = (
            bool(ok and ok2),
            f"all spec kinds in frozen whitelist (incl. residual_gated); "
            f"beta/g match frozen registry: {ok2} {detail[:3]}")
        return res

    module.pre_run_sanity = wrapped


def _run_7ao(args, seed: int) -> None:
    import experiment7ao_runner as r7
    from experiment11_model import AdaptivePatchcore

    r7.ModulePatchcore = AdaptivePatchcore
    r7.ALL_SPECS.update(c11.SEVEN_AO_SPECS)
    out_root = EXP / "raw" / args.round_name
    r7.LOGS_DIR = EXP / "logs"
    r7.OUT_ROOT = out_root
    h.OUT_ROOT = out_root
    h.LOGS_DIR = r7.LOGS_DIR
    Path(r7.LOGS_DIR).mkdir(parents=True, exist_ok=True)
    for name in (t.split(":")[2] for t in args.units.split(",")):
        if name not in r7.ALL_SPECS:
            raise SystemExit(f"unknown 11 spec {name}")
    r10._common(args.mode, seed, args.round_name)
    r10._sanity_override(r7, "R4_output_isolated", "experiment_11_c2_adaptive" in str(out_root),
                         f"output root = {out_root.relative_to(ROOT)}")
    _fix_r6(r7)
    sys.argv = ["experiment11_runner.py", "--units", args.units,
                "--worker-tag", args.worker_tag, "--out-root", str(out_root)]
    print(f"[Exp11] backend=7ao mode={args.mode} round={args.round_name} "
          f"worker={args.worker_tag} units={args.units}", flush=True)
    r7.main()


def _run_9b(args, seed: int) -> None:
    """参照点（Original / C2-fixed 若需要重跑）走 9B/7A-O 的既有 config。"""
    import experiment9b_runner as be
    import experiment10_candidates as c10

    be.CONFIGS.update(c10.NINE_B_CONFIGS)
    be.EXP_ROOT = EXP
    be.LOG_DIR = EXP / "logs"
    be.REF_DIR = EXP / "reference"
    for name in (t.split(":")[2] for t in args.units.split(",")):
        if name not in be.CONFIGS:
            raise SystemExit(f"unknown 9B config {name}")
    be.CANDIDATES = list(be.CANDIDATES) + list(c10.NINE_B_CONFIGS)
    r10._common(args.mode, seed, args.round_name)
    r10._sanity_override(be, "S2_output_isolated", "experiment_11_c2_adaptive" in str(be.EXP_ROOT),
                         f"out root = {be.EXP_ROOT.relative_to(ROOT)}")
    sys.argv = ["experiment11_runner.py", "--units", args.units,
                "--round", args.round_name, "--worker-tag", args.worker_tag]
    print(f"[Exp11] backend=9b mode={args.mode} round={args.round_name} "
          f"worker={args.worker_tag} units={args.units}", flush=True)
    be.main()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", required=True, choices=["7ao", "9b"])
    ap.add_argument("--units", required=True)
    ap.add_argument("--round", required=True, dest="round_name")
    ap.add_argument("--worker-tag", default="w0")
    ap.add_argument("--mode", default="corrected189", choices=["corrected189", "historical209"])
    ap.add_argument("--protocol", default="v2", choices=["v2"])
    args = ap.parse_args()
    seeds = {int(t.split(":")[1]) for t in args.units.split(",")}
    if len(seeds) != 1:
        raise SystemExit("single seed per worker required")
    if args.backend == "7ao":
        _run_7ao(args, seeds.pop())
    else:
        _run_9b(args, seeds.pop())


if __name__ == "__main__":
    main()
