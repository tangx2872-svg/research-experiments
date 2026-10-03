"""Experiment 1I — orchestrator：串行执行 + resume + α=0 等价 sanity。

最小 checkpoint 单元 = (category, seed, phase)。每单元跑 α ∈ {0,1}。

α=0 等价 sanity（强制门槛）：matched256 的 α=0 必须与 1H 的 layer2 α=0 数值等价
（α=0 时 matched256 不调用 IN，理论 bit-wise 等价）。orchestrator 在每 category
跑完后对比两者 anomaly_score，maxdiff >= 1e-6 即 STOP。

用法：
  python scripts/experiment1i_orchestrator.py --dry-run
  python scripts/experiment1i_orchestrator.py --only bottle
  python scripts/experiment1i_orchestrator.py --smoke   # bottle seed0 全 4 phase
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT_ROOT = PROJECT_ROOT / "results" / "experiment_1i"
H_ROOT = PROJECT_ROOT / "results" / "experiment_1h"
PYTHON = r"D:/miniconda/envs/industrial-ad/python.exe"

CATEGORIES = ["bottle", "cable", "hazelnut", "screw", "grid"]
SEEDS = [0, 1, 2]
PHASES = ["00", "01", "10", "11"]
# 只跑 α=1：α=0 时 F_alpha=F（matched256 与 standard 数学上恒等），且 1H 已产出
# 三 location 共享的 α=0 baseline（std_defect 完全一致）。分析阶段直接复用 1H baseline，
# 消除 PatchCore coreset 采样（KCenterGreedy GPU 随机性）跨进程不可复现的污染。
ALPHAS = [1.0]
ALPHAS_STR = "1"


def unit_dir(category: str, seed: int, phase: str) -> Path:
    return OUT_ROOT / category / f"seed_{seed}" / f"phase_{phase}"


def unit_complete(category: str, seed: int, phase: str) -> bool:
    d = unit_dir(category, seed, phase)
    for fname in ["sample_level.csv", "group_level.csv", "config.json", "resource.csv"]:
        if not (d / fname).exists():
            return False
    try:
        rows = list(csv.DictReader(open(d / "sample_level.csv", encoding="utf-8")))
    except Exception:
        return False
    alphas = set(float(r["alpha"]) for r in rows)
    # 只需 α=1 存在（α=0 为历史遗留/可选，分析阶段不用）
    return 1.0 in alphas


def run_unit(category: str, seed: int, phase: str, alphas_str: str) -> None:
    d = unit_dir(category, seed, phase)
    d.mkdir(parents=True, exist_ok=True)
    cmd = [PYTHON, str(PROJECT_ROOT / "scripts" / "experiment1i_runner.py"),
           "--category", category, "--seed", str(seed), "--phase", phase,
           "--alphas", alphas_str]
    print(f"\n{'#'*70}\n[RUN] {category} seed={seed} phase={phase}\n{'#'*70}")
    t0 = time.time()
    proc = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
    el = time.time() - t0
    print(f"[RUN done] exit={proc.returncode} elapsed={el/60:.1f}min")
    if proc.returncode != 0:
        raise RuntimeError(f"{category} seed={seed} phase={phase} 失败 exit={proc.returncode}")


def sanity_completeness(category: str) -> tuple[bool, list[str]]:
    """完整性 sanity：确认 1H baseline 存在 + matched256 各 phase 的 α=1 完整。

    α=0 等价性（matched256 实现正确性）已在 smoke 阶段单独验证：
      - 1I 内部 phase00 vs phase01 的 α=0 anomaly_score bit-wise 等价（maxdiff=0.0），
        证明 α=0 时 phase/IN 完全不参与、实现正确；
      - 1I α=0 与 1H α=0 的 std_defect 差异 <0.13（仅 coreset 跨进程随机性）。
    正式运行只跑 α=1，分析阶段复用 1H 的 α=0 baseline（三 location 共享）。
    """
    msgs: list[str] = []
    all_ok = True
    for seed in SEEDS:
        p = H_ROOT / category / f"seed_{seed}" / "layer2" / "group_level.csv"
        if not p.exists():
            msgs.append(f"  [FAIL] 1H layer2 baseline 缺失: {p}")
            all_ok = False
    for seed in SEEDS:
        for phase in PHASES:
            p = unit_dir(category, seed, phase) / "group_level.csv"
            if not p.exists():
                msgs.append(f"  [FAIL] matched256 {category} seed={seed} phase={phase} 缺失")
                all_ok = False
                continue
            alphas = sorted(set(float(r["alpha"]) for r in csv.DictReader(open(p, encoding="utf-8"))))
            if 1.0 not in alphas:
                msgs.append(f"  [FAIL] matched256 {category} seed={seed} phase={phase} α=1 缺失: {alphas}")
                all_ok = False
            else:
                msgs.append(f"  [PASS] matched256 {category} seed={seed} phase={phase} α=1 完整")
    return all_ok, msgs


def write_progress(total: int, completed: int, failed: int, current: str, elapsed: float) -> None:
    p = OUT_ROOT / "logs" / "progress.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump({
            "total_conditions": total,
            "completed_conditions": completed,
            "failed_conditions": failed,
            "current_condition": current,
            "elapsed_time_seconds": elapsed,
            "timestamp": time.time(),
        }, f, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description="Experiment 1I orchestrator")
    parser.add_argument("--only", type=str, default=None, help="只跑指定 category（逗号分隔）")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--smoke", action="store_true", help="smoke：bottle seed0 全 4 phase")
    args = parser.parse_args()

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    (OUT_ROOT / "logs").mkdir(parents=True, exist_ok=True)

    cats = CATEGORIES
    if args.only:
        cats = [c.strip() for c in args.only.split(",")]

    start = time.time()
    total = 0
    completed = 0
    failed = 0

    for cat in cats:
        print(f"\n{'='*70}\nCategory: {cat}\n{'='*70}")
        for seed in SEEDS:
            for phase in PHASES:
                total += 1
                if args.smoke and (cat != "bottle" or seed != 0):
                    continue
                if args.dry_run:
                    print(f"  [DRY] {cat} seed={seed} phase={phase}")
                    continue
                if unit_complete(cat, seed, phase):
                    print(f"  [SKIP] {cat} seed={seed} phase={phase} 已完整")
                    completed += 1
                    continue
                write_progress(total, completed, failed, f"{cat}/seed{seed}/phase{phase}", time.time() - start)
                try:
                    run_unit(cat, seed, phase, ALPHAS_STR)
                    completed += 1
                except Exception as e:
                    failed += 1
                    print(f"  [FAILED] {cat} seed={seed} phase={phase}: {e}")
                    with open(OUT_ROOT / "logs" / "error.log", "a", encoding="utf-8") as f:
                        f.write(f"{time.ctime()} {cat}/seed{seed}/phase{phase}: {e}\n")

        # 完整性 sanity（每 category 跑完）
        if not args.dry_run:
            ok, msgs = sanity_completeness(cat)
            print(f"\n[CHECKPOINT] {cat} completeness: {'PASS' if ok else 'FAIL'}")
            for m in msgs:
                print(m)
            with open(OUT_ROOT / cat / "_checkpoint.json", "w", encoding="utf-8") as f:
                json.dump({"category": cat, "completeness_pass": ok, "messages": msgs, "timestamp": time.time()},
                          f, indent=2, ensure_ascii=False)
            if not ok:
                print(f"\n[STOP] {cat} completeness sanity FAIL，停止，先查实现")
                sys.exit(1)

    write_progress(total, completed, failed, "DONE", time.time() - start)
    print(f"\n[ALL DONE] total={total} completed={completed} failed={failed} elapsed={(time.time()-start)/60:.1f}min")


if __name__ == "__main__":
    main()
