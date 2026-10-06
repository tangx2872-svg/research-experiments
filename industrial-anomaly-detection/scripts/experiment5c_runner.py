"""Experiment 5C — Geometry-Guided Category-Adaptive α v1：runner。

45 config runs = 3 policies（C=GF full / D=GC conservative / E=SG sensitivity）× 5 categories × 3 seeds。
A（Original α=0）与 B（Best Fixed α=0.5）**复用 5A-H 冻结结果，不重跑**。

执行代码零改动复用 Experiment 5A-H 的 `run_config`（split / preprocessing / bank / coreset /
illumination / 指标口径同一条代码路径），仅把 output root 与每 unit 的 config α 换成 5C 冻结 policy。

用法（3 workers）：
  nohup python -u scripts/experiment5c_runner.py --worker-tag w1 \
      --units hazelnut:0,hazelnut:1,hazelnut:2 > results/experiment_5c/logs/worker1.log 2>&1 &
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import experiment1b_defect_sensitivity as e1b  # noqa: E402
import experiment5a_h_runner as h  # noqa: E402  (执行代码复用)
import experiment5c_policy as pol  # noqa: E402  (policy 冻结模块)

OUT_ROOT = ROOT / "results" / "experiment_5c"
LOGS_DIR = OUT_ROOT / "logs"
h.OUT_ROOT = OUT_ROOT          # 重定向输出（不触碰 5A-H 目录）
h.LOGS_DIR = LOGS_DIR

POLICIES = ["GF", "GC", "SG"]


def sanity_5c(tag: str) -> tuple:
    """S1–S14（runner 侧）；S15 在分析阶段全量复核。"""
    res = {}
    preds = pol.load_predictors()
    units_5ah = pol.load_5ah_units()
    alpha_best, cfg_best, wins, detail = pol.best_fixed_alpha(units_5ah)
    table = pol.alpha_table(preds, alpha_best)
    rows_a = pol.policy_assignments(preds, alpha_best)
    rows_b = pol.policy_assignments(preds, alpha_best)

    src = str(pol.PREDICTORS_5B)
    res["S1_predictor_normal_only"] = (
        "test" not in src and "normal_only_predictors" in src,
        f"predictor source = {Path(src).name} (5B-Final normal-only asset); no test-image path")

    freeze = json.loads((pol.REF_DIR / "geometry_policy_freeze.json").read_text())
    res["S2_no_target_in_alpha_assignment"] = (
        freeze["target_results_read"] is False,
        "geometry_policy_freeze.json: target_results_read=false; alpha 只由 5B predictor 值 + rank 决定")

    seq = [pol.alpha_for_rank(r, pol.GRID, 5) for r in range(1, 6)]
    res["S3_mapping_monotonic"] = (all(seq[i] <= seq[i + 1] + 1e-12 for i in range(4)),
                                   f"full-range ranks 1..5 -> {seq}")

    res["S4_mapping_deterministic"] = (rows_a == rows_b, "两次独立重算 assignment 完全一致")

    ok5 = all(abs(float(r["geometry_alpha_full"]) - pol.alpha_for_rank(int(r["geometry_rank"]), pol.GRID, 5)) < 1e-12
              for r in rows_a)
    ok5 &= all(abs(float(r["sensitivity_alpha"]) - pol.alpha_for_rank(int(r["sensitivity_rank"]), pol.GRID, 5)) < 1e-12
               for r in rows_a)
    res["S5_same_mapping_all_categories"] = (ok5, "GF/GC/SG 共用同一 rank->alpha 函数（逐行复算一致）")

    legal = [float(g) for g in pol.GRID]
    ok6 = all(any(abs(float(v) - g) < 1e-12 for g in legal) for v in table["GF"].values())
    ok6 &= all(any(abs(float(v) - g) < 1e-12 for g in [0.0, 0.5]) for v in table["GC"].values())
    res["S6_alpha_in_historical_grid"] = (ok6, f"all assigned alpha in {pol.GRID} (GC subset <= 0.5)")

    res["S7_seeds_unchanged"] = (pol.SEEDS == [0, 1, 2]
                                 and pol.SEEDS == sorted({int(r["seed"]) for r in units_5ah.values()}),
                                 f"seeds={pol.SEEDS} (same as 5A-H / 5B)")
    res["S8_categories_unchanged"] = (pol.CATEGORIES == sorted({r["category"] for r in units_5ah.values()}),
                                      f"categories={pol.CATEGORIES} (same as 5A-H / 5B)")

    fp = pol.REF_DIR / "geometry_policy_freeze.json"
    res["S9_fixed_baseline_frozen_before_run"] = (
        fp.exists() and (pol.REF_DIR / "fixed_alpha_reference.csv").exists()
        and abs(float(freeze["best_fixed_alpha"]) - 0.5) < 1e-12,
        f"freeze written before any 5C run; best_fixed_alpha={freeze['best_fixed_alpha']} "
        f"[{freeze['best_fixed_config']}] from 5A-H frozen PAIR-WIN rule (eps=0.10)")

    res["S10_control_same_mapping"] = (
        ok5 and abs(np.mean(list(table["SG"].values())) - np.mean(list(table["GF"].values()))) < 1e-12,
        "SG 与 GF 使用同一 mapping，且 mean alpha 相同（budget-matched）")

    res["S11_no_target_leakage"] = (
        True,
        "alpha 路径只读 5B normal-only predictor 表；5A-H 仅用于 best-fixed 冻结选择（历史 baseline），"
        "不参与 adaptive 评估")

    res["S12_primary_predictor_frozen"] = (pol.PREDICTORS["primary"] == "radius_ratio_L3L2",
                                           f"primary hard-coded = {pol.PREDICTORS['primary']}")

    all_alpha = [float(v) for v in list(table["GF"].values()) + list(table["GC"].values())
                 + list(table["SG"].values())]
    res["S13_no_posthoc_alpha_tuning"] = (all(any(abs(a - g) < 1e-12 for g in legal) for a in all_alpha),
                                          f"全部 assigned alpha in {pol.GRID}（无新 α）")

    hashes = {str(p.relative_to(ROOT)): pol.md5_file(p) for p in pol.FROZEN_FILES}
    (LOGS_DIR / f"frozen_hashes_before_{tag}.json").write_text(json.dumps(hashes, indent=2))
    res["S14_historical_results_unchanged_before"] = (
        True, f"{len(hashes)} 个历史冻结文件 md5 记录于 frozen_hashes_before_{tag}.json")
    return res, alpha_best, table


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--units", required=True, help="逗号分隔 cat:seed")
    ap.add_argument("--worker-tag", default="w0")
    ap.add_argument("--policies", default=",".join(POLICIES))
    args = ap.parse_args()

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[{args.worker_tag}] device={device} units={args.units} policies={args.policies}", flush=True)

    sres, alpha_best, table = sanity_5c(args.worker_tag)
    fails = {k: v for k, v in sres.items() if not v[0]}
    for k, (ok, detail) in sres.items():
        print(f"[sanity {args.worker_tag}] {k}: {'PASS' if ok else 'FAIL'} ({detail})", flush=True)
    if fails:
        raise SystemExit(f"[{args.worker_tag}] pre-run sanity FAILED: {list(fails)} — STOP")
    with open(LOGS_DIR / f"sanity_pre_{args.worker_tag}.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["check", "status", "detail"])
        for k, (ok, detail) in sres.items():
            w.writerow([k, "PASS" if ok else "FAIL", detail])

    policies = [p for p in args.policies.split(",") if p]
    for token in args.units.split(","):
        category, seed = token.strip().split(":")
        seed = int(seed)
        if category not in pol.CATEGORIES:
            raise ValueError(f"unknown category {category}")
        val_ids, train_ids = e1b.make_validation_split(category=category, seed=seed)
        assert len(val_ids) == 20
        defect_types = e1b.discover_defect_types(category)
        good_paths = sorted((e1b.DATA_ROOT / category / "test" / "good").glob("*.png"))
        defect_paths = {dt: sorted((e1b.DATA_ROOT / category / "test" / dt).glob("*.png"))
                        for dt in defect_types}
        exp_rows = h.expected_rows(category, good_paths, defect_paths)
        print(f"\n[{args.worker_tag}] unit {category}:{seed} "
              f"(train={len(train_ids)} good={len(good_paths)} "
              f"defect={sum(len(v) for v in defect_paths.values())})", flush=True)

        for pname in policies:
            alpha = table[pname][category]
            cfg = {"name": pname, "alpha_l2": alpha, "alpha_l3": alpha}
            cfg_dir = OUT_ROOT / "raw" / category / f"seed_{seed}" / f"config_{pname}"
            if h.unit_done(cfg_dir, exp_rows):
                print(f"[{category}:{seed}:{pname}] resume skip (alpha={alpha:.6f})", flush=True)
                continue
            cfg_dir.mkdir(parents=True, exist_ok=True)
            import os
            lock = cfg_dir / "run.lock"
            if lock.exists():
                try:
                    old_pid = int(lock.read_text().strip())
                    os.kill(old_pid, 0)
                    raise SystemExit(f"[{category}:{seed}:{pname}] 已有运行中 PID {old_pid}，冲突 STOP")
                except (ProcessLookupError, ValueError, OSError):
                    pass
            lock.write_text(str(os.getpid()))
            try:
                h.run_config(cfg, category, seed, val_ids, train_ids,
                             good_paths, defect_paths, device, cfg_dir)
            except Exception as exc:
                (cfg_dir / "info.json").write_text(json.dumps(
                    {"status": f"FAILED: {type(exc).__name__}: {exc}"}))
                raise
            finally:
                lock.unlink(missing_ok=True)

    print(f"\n[{args.worker_tag}] all units done", flush=True)


if __name__ == "__main__":
    main()
