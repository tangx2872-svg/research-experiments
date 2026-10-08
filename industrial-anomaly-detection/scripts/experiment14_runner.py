"""Experiment 14 runner — broad screening（严格复用既有协议，零协议漂移）。

复用：experiment10_protocol/runner 的 `_common`（= strict-V2 RNG + corrected-189 train filter
+ 证据层）、experiment7ao_runner 的主循环、experiment11_runner 的模型换装模式。
输出根：results/experiment_14/raw/<round>/<category>/seed_<s>/config_<spec>/

用法：
  python -u scripts/experiment14_runner.py --units "bottle:0:E14_A2__bottle" --round a1 --worker-tag w0
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
import experiment14_candidates as c14  # noqa: E402

EXP = ROOT / "results" / "experiment_14"


def _fix_r6(module) -> None:
    """R6：白名单加入 inss；Family A 代数等价 + Family B basis 逐位一致强制核验。"""
    orig = module.pre_run_sanity

    def wrapped(*a, **kw):
        res = orig(*a, **kw)
        ok, detail = c14.verify_frozen()
        res["R6_no_target_dependent_params"] = (bool(ok), detail)
        return res

    module.pre_run_sanity = wrapped


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--units", required=True)
    ap.add_argument("--round", required=True, dest="round_name")
    ap.add_argument("--worker-tag", default="w0")
    ap.add_argument("--mode", default="corrected189", choices=["corrected189"])
    ap.add_argument("--protocol", default="v2", choices=["v2"])
    args = ap.parse_args()

    import experiment7ao_runner as r7
    from experiment14_model import InssPatchcore

    seeds = {int(t.split(":")[1]) for t in args.units.split(",")}
    if len(seeds) != 1:
        raise SystemExit("single seed per worker required")
    seed = seeds.pop()

    r7.ModulePatchcore = InssPatchcore                 # inss 的继承链覆盖 A/B 两 family
    r7.ALL_SPECS.update(c14.ALL14_SPECS)
    out_root = EXP / "raw" / args.round_name
    r7.LOGS_DIR = EXP / "logs"
    r7.OUT_ROOT = out_root
    h.OUT_ROOT = out_root
    h.LOGS_DIR = r7.LOGS_DIR
    Path(r7.LOGS_DIR).mkdir(parents=True, exist_ok=True)
    for name in (t.split(":")[2] for t in args.units.split(",")):
        if name not in r7.ALL_SPECS:
            raise SystemExit(f"unknown 14 spec {name}")

    r10.EXP = EXP                                      # 证据/RNG 输出根重定向到 14
    r10._common(args.mode, seed, args.round_name)
    r10._sanity_override(r7, "R4_output_isolated", "experiment_14" in str(out_root),
                         f"output root = {out_root.relative_to(ROOT)}")
    _fix_r6(r7)
    sys.argv = ["experiment14_runner.py", "--units", args.units,
                "--worker-tag", args.worker_tag, "--out-root", str(out_root)]
    print(f"[Exp14] protocol={args.protocol} mode={args.mode} round={args.round_name} "
          f"worker={args.worker_tag} units={args.units}", flush=True)
    r7.main()


if __name__ == "__main__":
    main()
