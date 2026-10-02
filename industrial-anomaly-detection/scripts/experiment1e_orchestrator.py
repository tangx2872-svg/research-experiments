"""Experiment 1E Phase 3-7：正式 Pilot orchestrator。

串行执行 grid -> cable -> screw -> hazelnut，每 category 3 seeds x 5 alphas。
- Phase 5：每 category 完成后做 sanity check，全部 PASS 才继续下一个。
- Phase 6：自动 continue / stop（sanity FAIL 时停止）。
- Phase 7：resume/skip completed conditions（结果完整则 SKIP）。
- Phase 8：资源记录由 experiment1e_runner 在每 alpha 写入 resource.csv。

用法：
  python scripts/experiment1e_orchestrator.py            # 全部 4 category
  python scripts/experiment1e_orchestrator.py --only grid  # 只跑 grid
  python scripts/experiment1e_orchestrator.py --dry-run   # 只打印计划不跑
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT_ROOT = PROJECT_ROOT / "results" / "experiment_1e"
PYTHON = r"D:/miniconda/envs/industrial-ad/python.exe"

CATEGORIES = ["grid", "cable", "screw", "hazelnut"]
SEEDS = [0, 1, 2]
ALPHAS = [0.0, 0.25, 0.5, 0.75, 1.0]

SAMPLE_FIELDS = ["category", "defect_type", "image_path", "seed", "alpha",
                 "is_good", "gt_label", "anomaly_score", "good_mean", "good_std",
                 "z_score", "mask_path", "area_pixels", "area_ratio", "log_area_ratio"]


def condition_dir(category: str, seed: int) -> Path:
    return OUT_ROOT / category / f"seed_{seed}"


def expected_sample_count(category: str) -> int:
    """返回该 category 每个 (seed, alpha) 应有的 test 样本数 = test_good + all defects。"""
    inv_csv = OUT_ROOT / "dataset_inventory.csv"
    with open(inv_csv, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["category"] == category:
                return int(r["test_good"]) + int(r["total_defect_samples"])
    raise ValueError(f"category {category} 不在 inventory 中")


def condition_complete(category: str, seed: int) -> bool:
    d = condition_dir(category, seed)
    sample_csv = d / "sample_level.csv"
    group_csv = d / "group_level.csv"
    config_json = d / "config.json"
    if not (sample_csv.exists() and group_csv.exists() and config_json.exists()):
        return False
    # 校验 sample-level 是否覆盖全部 5 alpha，且每 alpha 样本数正确
    try:
        rows = list(csv.DictReader(open(sample_csv, encoding="utf-8")))
    except Exception:
        return False
    exp = expected_sample_count(category)
    from collections import Counter
    c = Counter((r["alpha"]) for r in rows)
    alphas_present = sorted(set(float(a) for a in c.keys()))
    if alphas_present != ALPHAS:
        return False
    # 每 alpha 样本数 = test_good + defects
    for a in ALPHAS:
        if c[str(a)] != exp:
            return False
    return True


def run_condition(category: str, seed: int) -> None:
    d = condition_dir(category, seed)
    d.mkdir(parents=True, exist_ok=True)
    cmd = [PYTHON, str(PROJECT_ROOT / "scripts" / "experiment1e_runner.py"),
           "--category", category, "--seed", str(seed),
           "--alphas", "0 0.25 0.5 0.75 1"]
    print(f"\n{'#'*70}\n[RUN] {category} seed={seed}\n{'#'*70}")
    t0 = time.time()
    proc = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
    el = time.time() - t0
    print(f"[RUN done] {category} seed={seed} exit={proc.returncode} elapsed={el/60:.1f}min")
    if proc.returncode != 0:
        raise RuntimeError(f"{category} seed={seed} 运行失败 exit={proc.returncode}")


def sanity_check(category: str) -> tuple[bool, list[str]]:
    """Phase 5 sanity checks。返回 (all_pass, messages)。"""
    msgs: list[str] = []
    inv_csv = OUT_ROOT / "dataset_inventory.csv"
    with open(inv_csv, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["category"] == category:
                test_good = int(r["test_good"])
                defect_types = r["defect_types"].split("; ")
                defect_types = [d.split("(")[0] for d in defect_types]
                break
    exp_sample = expected_sample_count(category)

    all_ok = True
    for seed in SEEDS:
        d = condition_dir(category, seed)
        sample_csv = d / "sample_level.csv"
        if not sample_csv.exists():
            msgs.append(f"  [FAIL] {category} seed={seed} sample_level.csv 缺失")
            all_ok = False
            continue
        rows = list(csv.DictReader(open(sample_csv, encoding="utf-8")))
        # 1. completeness: 3 seeds x 5 alphas 全部存在
        alphas = sorted(set(float(r["alpha"]) for r in rows))
        if alphas != ALPHAS:
            msgs.append(f"  [FAIL] {category} seed={seed} alpha 不完整: {alphas}")
            all_ok = False
        # 2. sample count
        from collections import Counter
        c = Counter((r["alpha"], r["is_good"]) for r in rows)
        for a in ALPHAS:
            n_good = c[(str(a), "1")]
            n_def = c[(str(a), "0")]
            if n_good != test_good:
                msgs.append(f"  [FAIL] {category} seed={seed} α={a} test_good={n_good} != {test_good}")
                all_ok = False
            if n_good + n_def != exp_sample:
                msgs.append(f"  [FAIL] {category} seed={seed} α={a} 总样本={n_good+n_def} != {exp_sample}")
                all_ok = False
        # 3. finite values
        for r in rows:
            for k in ["anomaly_score", "good_mean", "good_std", "z_score"]:
                v = r[k]
                if v == "" or v is None:
                    msgs.append(f"  [FAIL] {category} seed={seed} 空值 {k}")
                    all_ok = False
                    break
                try:
                    fv = float(v)
                    if not math.isfinite(fv):
                        msgs.append(f"  [FAIL] {category} seed={seed} 非有限值 {k}={v}")
                        all_ok = False
                        break
                except ValueError:
                    msgs.append(f"  [FAIL] {category} seed={seed} 非数值 {k}={v}")
                    all_ok = False
                    break
        # 4. good normalization: mean(z_good)≈0, std(z_good)≈1（每 seed x alpha）
        for a in ALPHAS:
            z_good = [float(r["z_score"]) for r in rows if float(r["alpha"]) == a and r["is_good"] == "1"]
            if z_good:
                mz = sum(z_good) / len(z_good)
                sz = (sum((x - mz) ** 2 for x in z_good) / (len(z_good) - 1)) ** 0.5 if len(z_good) > 1 else 0
                if abs(mz) > 1e-6 or abs(sz - 1.0) > 1e-6:
                    msgs.append(f"  [FAIL] {category} seed={seed} α={a} good z mean={mz:.6f} std={sz:.6f}")
                    all_ok = False
        # 5. defect discovery: 所有 defect types 存在
        present_dt = sorted(set(r["defect_type"] for r in rows if r["is_good"] == "0"))
        if present_dt != sorted(defect_types):
            msgs.append(f"  [FAIL] {category} seed={seed} defect types {present_dt} != {sorted(defect_types)}")
            all_ok = False
        # 6. area: 0 < area_ratio <= 1
        for r in rows:
            if r["is_good"] == "0":
                ar = r["area_ratio"]
                if ar == "" or ar is None:
                    msgs.append(f"  [FAIL] {category} seed={seed} 空 area_ratio")
                    all_ok = False
                    break
                try:
                    fv = float(ar)
                    if not (0 < fv <= 1):
                        msgs.append(f"  [FAIL] {category} seed={seed} area_ratio={fv} 越界")
                        all_ok = False
                        break
                except ValueError:
                    msgs.append(f"  [FAIL] {category} seed={seed} 非数值 area_ratio={ar}")
                    all_ok = False
                    break

    # 7. 重复行 / 错位检测（seed/alpha/category 一致性）
    all_rows = []
    for seed in SEEDS:
        sample_csv = condition_dir(category, seed) / "sample_level.csv"
        if sample_csv.exists():
            for r in csv.DictReader(open(sample_csv, encoding="utf-8")):
                if r["category"] != category:
                    msgs.append(f"  [FAIL] category 错位: {r['category']} != {category}")
                    all_ok = False
                if int(r["seed"]) != seed:
                    msgs.append(f"  [FAIL] seed 错位: {r['seed']} != {seed}")
                    all_ok = False
                all_rows.append((seed, r["alpha"], r["image_path"]))

    if len(all_rows) != len(set(all_rows)):
        msgs.append(f"  [FAIL] {category} 存在重复行")
        all_ok = False

    return all_ok, msgs


def main() -> None:
    parser = argparse.ArgumentParser(description="Experiment 1E formal pilot orchestrator")
    parser.add_argument("--only", type=str, default=None, help="只运行指定 category（逗号分隔）")
    parser.add_argument("--dry-run", action="store_true", help="只打印计划")
    args = parser.parse_args()

    cats = CATEGORIES
    if args.only:
        cats = [c.strip() for c in args.only.split(",")]

    # 冻结配置必须存在
    if not (OUT_ROOT / "config.json").exists():
        print("[ERROR] 缺少冻结配置 results/experiment_1e/config.json，先完成 Phase 1")
        sys.exit(1)

    for cat in cats:
        print(f"\n{'='*70}\nCategory: {cat}\n{'='*70}")
        for seed in SEEDS:
            if condition_complete(cat, seed):
                print(f"  [SKIP] {cat} seed={seed} 已完整")
                continue
            if args.dry_run:
                print(f"  [DRY] {cat} seed={seed} 将运行")
                continue
            run_condition(cat, seed)
        # Phase 5 sanity check
        if args.dry_run:
            print(f"  [DRY] sanity check {cat}（跳过）")
            continue
        ok, msgs = sanity_check(cat)
        print(f"\n[CHECKPOINT] {cat} sanity check: {'PASS' if ok else 'FAIL'}")
        for m in msgs:
            print(m)
        # 写 checkpoint 记录
        with open(OUT_ROOT / cat / "_checkpoint.json", "w", encoding="utf-8") as f:
            json.dump({"category": cat, "sanity_pass": ok, "messages": msgs,
                       "timestamp": time.time()}, f, indent=2, ensure_ascii=False)
        if not ok:
            print(f"\n[STOP] {cat} sanity check FAIL，停止后续 categories，保留日志，等待人工处理。")
            sys.exit(1)
        print(f"\n[CONTINUE] {cat} PASS，继续下一个 category。")

    print("\n[ALL DONE] 全部 category 运行完成且 sanity check PASS")


if __name__ == "__main__":
    main()
