"""Experiment 1C：Multi-Seed Stability Validation。

研究问题：
  Experiment 1B 观察到的 defect-specific α-IN response
  （broken_large 分离度大幅下降 / broken_small 稳定 / contamination 反向），
  是否由 PatchCore coreset sampling 的随机性导致？

方法（严格复用 Experiment 1B，一行不改实验逻辑）：
  保持 1B 全部条件不变（dataset/model/α-IN/阈值规则/validation split 规则），
  仅改变 random seed ∈ {0, 1, 2}，重复 1B 的 run_screening。

  复用方式：直接 import `experiment1b_defect_sensitivity.run_screening`，
  每个 seed 传入独立的 results_root（results/experiment_1c/seed_{N}/），
  实现多 seed 结果隔离，不重构、不改模型、不改 1B 逻辑。

输出：
  results/experiment_1c/
    seed_0/  seed_1/  seed_2/     （结构与 1B 一致：raw/summary/figures/heatmaps/config）

运行方式（项目根目录，industrial-ad 环境）：
  python scripts/experiment1c_multiseed.py --smoke       # smoke test（少量样本，3 seed）
  python scripts/experiment1c_multiseed.py               # 全量 3 seed
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

# 复用 1B 的 run_screening（实验逻辑零改动）
from experiment1b_defect_sensitivity import run_screening  # noqa: E402

RESULTS_ROOT = PROJECT_ROOT / "results" / "experiment_1c"
SEEDS = [0, 1, 2]


def main() -> None:
    parser = argparse.ArgumentParser(description="Experiment 1C multi-seed stability validation")
    parser.add_argument("--smoke", action="store_true", help="smoke test：少量样本")
    parser.add_argument("--smoke-limit", type=int, default=5, help="smoke test 每类样本数")
    parser.add_argument("--seeds", type=str, default="0,1,2", help="逗号分隔的 seed 列表")
    args = parser.parse_args()

    seeds = [int(s) for s in args.seeds.split(",")]

    for seed in seeds:
        seed_root = RESULTS_ROOT / f"seed_{seed}"
        print(f"\n{'#'*70}\n# Experiment 1C: seed={seed} -> {seed_root}\n{'#'*70}")
        run_screening(
            smoke_limit=args.smoke_limit if args.smoke else None,
            seed=seed,
            results_root=seed_root,
        )

    print(f"\n[done] Experiment 1C 完成，seeds={seeds}")
    print(f"[done] 输出目录 = {RESULTS_ROOT}")


if __name__ == "__main__":
    main()
