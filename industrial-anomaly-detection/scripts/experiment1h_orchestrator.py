"""Experiment 1H — orchestrator：串行执行 + baseline 缓存 + sanity + resume + progress。

最小 checkpoint 单元 = (category, seed, location)。每个单元跑完整 5 档 α。

baseline 缓存：α=0 在三个 location 下理论等价。为节省计算，先跑 layer2 的 α=0，
layer3/post_concat 的 α=0 结果直接从 layer2 复制（它们 bit-wise 等价）。runner 仍
对每个 location 独立产出完整 sample_level.csv，orchestrator 在「复制 baseline」时
把 layer2 的 α=0 行写入 layer3/post_concat 的 CSV（defect_type/good 结构完全一致，
仅 location 字段改写）。

用法：
  python scripts/experiment1h_orchestrator.py --dry-run
  python scripts/experiment1h_orchestrator.py --only bottle           # 只跑 bottle
  python scripts/experiment1h_orchestrator.py --smoke                  # smoke：bottle seed0 全 location α=0/1
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
OUT_ROOT = PROJECT_ROOT / "results" / "experiment_1h"
PYTHON = r"D:/miniconda/envs/industrial-ad/python.exe"

CATEGORIES = ["bottle", "cable", "hazelnut", "screw", "grid"]
SEEDS = [0, 1, 2]
LOCATIONS = ["layer2", "layer3", "post_concat"]
ALPHAS = [0.0, 0.25, 0.5, 0.75, 1.0]
ALPHAS_STR = "0 0.25 0.5 0.75 1"


def unit_dir(category: str, seed: int, location: str) -> Path:
    return OUT_ROOT / category / f"seed_{seed}" / location


def unit_complete(category: str, seed: int, location: str) -> bool:
    d = unit_dir(category, seed, location)
    for fname in ["sample_level.csv", "group_level.csv", "config.json", "resource.csv"]:
        if not (d / fname).exists():
            return False
    # 校验 sample_level 覆盖 5 档 α
    try:
        rows = list(csv.DictReader(open(d / "sample_level.csv", encoding="utf-8")))
    except Exception:
        return False
    from collections import Counter
    alphas = sorted(set(float(r["alpha"]) for r in rows))
    return alphas == ALPHAS


def run_unit(category: str, seed: int, location: str, alphas_str: str) -> None:
    d = unit_dir(category, seed, location)
    d.mkdir(parents=True, exist_ok=True)
    cmd = [PYTHON, str(PROJECT_ROOT / "scripts" / "experiment1h_runner.py"),
           "--category", category, "--seed", str(seed), "--location", location,
           "--alphas", alphas_str]
    print(f"\n{'#'*70}\n[RUN] {category} seed={seed} loc={location}\n{'#'*70}")
    t0 = time.time()
    proc = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
    el = time.time() - t0
    print(f"[RUN done] exit={proc.returncode} elapsed={el/60:.1f}min")
    if proc.returncode != 0:
        raise RuntimeError(f"{category} seed={seed} loc={location} 失败 exit={proc.returncode}")


def copy_baseline_from_layer2(category: str, seed: int, target_loc: str) -> None:
    """把 layer2 的 α=0 行复制到 target_loc 的 CSV（location 字段改写）。"""
    src = unit_dir(category, seed, "layer2")
    dst = unit_dir(category, seed, target_loc)
    if not (src / "sample_level.csv").exists():
        raise RuntimeError(f"baseline 源 layer2 缺失: {src}")

    # sample_level：复制 α=0 行，改 location 字段
    src_rows = list(csv.DictReader(open(src / "sample_level.csv", encoding="utf-8")))
    a0_rows = [r for r in src_rows if float(r["alpha"]) == 0.0]
    for r in a0_rows:
        r["location"] = target_loc
    # 若 dst 已有其他 α 行，保留；否则从零开始
    existing = []
    if (dst / "sample_level.csv").exists():
        existing = [r for r in csv.DictReader(open(dst / "sample_level.csv", encoding="utf-8"))]
    merged = [r for r in existing if float(r["alpha"]) != 0.0] + a0_rows
    sample_fields = list(a0_rows[0].keys())
    with open(dst / "sample_level.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=sample_fields)
        w.writeheader()
        w.writerows(merged)

    # group_level：同样复制 α=0 行
    if (src / "group_level.csv").exists():
        src_g = list(csv.DictReader(open(src / "group_level.csv", encoding="utf-8")))
        g0_rows = [r for r in src_g if float(r["alpha"]) == 0.0]
        for r in g0_rows:
            r["location"] = target_loc
        g_existing = []
        if (dst / "group_level.csv").exists():
            g_existing = [r for r in csv.DictReader(open(dst / "group_level.csv", encoding="utf-8"))]
        g_merged = [r for r in g_existing if float(r["alpha"]) != 0.0] + g0_rows
        g_fields = list(g0_rows[0].keys())
        with open(dst / "group_level.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=g_fields)
            w.writeheader()
            w.writerows(g_merged)

    print(f"  [baseline-cache] {category} seed={seed} {target_loc} 的 α=0 从 layer2 复制")


def sanity_check(category: str) -> tuple[bool, list[str]]:
    """α=0 三 location 数值等价 + 完整性。"""
    msgs: list[str] = []
    all_ok = True
    for seed in SEEDS:
        # α=0 等价性：比较三个 location 的 α=0 anomaly_score
        a0_by_loc = {}
        for loc in LOCATIONS:
            p = unit_dir(category, seed, loc) / "sample_level.csv"
            if not p.exists():
                msgs.append(f"  [FAIL] {category} seed={seed} loc={loc} sample_level.csv 缺失")
                all_ok = False
                continue
            rows = list(csv.DictReader(open(p, encoding="utf-8")))
            a0 = {}
            for r in rows:
                if float(r["alpha"]) == 0.0:
                    a0[r["image_path"]] = float(r["anomaly_score"])
            a0_by_loc[loc] = a0
        if len(a0_by_loc) < 3:
            continue
        ref = a0_by_loc["layer2"]
        for loc in ["layer3", "post_concat"]:
            d = a0_by_loc[loc]
            if set(d.keys()) != set(ref.keys()):
                msgs.append(f"  [FAIL] {category} seed={seed} loc={loc} α=0 键集合不一致")
                all_ok = False
                continue
            maxdiff = max(abs(ref[k] - d[k]) for k in ref)
            if maxdiff >= 1e-6:
                msgs.append(f"  [FAIL] {category} seed={seed} loc={loc} α=0 maxdiff={maxdiff:.3e} >= 1e-6")
                all_ok = False
            else:
                msgs.append(f"  [PASS] {category} seed={seed} loc={loc} α=0 等价 (maxdiff={maxdiff:.3e})")
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
    parser = argparse.ArgumentParser(description="Experiment 1H orchestrator")
    parser.add_argument("--only", type=str, default=None, help="只跑指定 category（逗号分隔）")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--smoke", action="store_true", help="smoke：bottle seed0 全 location α=0/1")
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
            for loc in LOCATIONS:
                total += 1
                if args.smoke and (cat != "bottle" or seed != 0):
                    continue
                if args.dry_run:
                    print(f"  [DRY] {cat} seed={seed} loc={loc}")
                    continue
                if unit_complete(cat, seed, loc):
                    print(f"  [SKIP] {cat} seed={seed} loc={loc} 已完整")
                    completed += 1
                    continue
                # smoke 模式：只跑 α=0/1
                alphas_str = "0 1" if args.smoke else ALPHAS_STR
                write_progress(total, completed, failed, f"{cat}/seed{seed}/{loc}", time.time() - start)
                try:
                    run_unit(cat, seed, loc, alphas_str)
                    completed += 1
                except Exception as e:
                    failed += 1
                    print(f"  [FAILED] {cat} seed={seed} loc={loc}: {e}")
                    with open(OUT_ROOT / "logs" / "error.log", "a", encoding="utf-8") as f:
                        f.write(f"{time.ctime()} {cat}/seed{seed}/{loc}: {e}\n")

        # baseline 缓存（非 smoke，α=0 完整）
        if not args.smoke and not args.dry_run:
            for seed in SEEDS:
                if unit_complete(cat, seed, "layer2"):
                    for tloc in ["layer3", "post_concat"]:
                        # 若 target 已完整则跳过；否则复制 α=0 行（run_unit 已跑完非 α=0 部分时补 baseline）
                        if not unit_complete(cat, seed, tloc):
                            copy_baseline_from_layer2(cat, seed, tloc)

        # sanity（非 smoke）
        if not args.smoke and not args.dry_run:
            ok, msgs = sanity_check(cat)
            print(f"\n[CHECKPOINT] {cat} sanity: {'PASS' if ok else 'FAIL'}")
            for m in msgs:
                print(m)
            with open(OUT_ROOT / cat / "_checkpoint.json", "w", encoding="utf-8") as f:
                json.dump({"category": cat, "sanity_pass": ok, "messages": msgs, "timestamp": time.time()},
                          f, indent=2, ensure_ascii=False)
            if not ok:
                print(f"\n[STOP] {cat} sanity FAIL，保留日志等待处理")
                sys.exit(1)

    write_progress(total, completed, failed, "DONE", time.time() - start)
    print(f"\n[ALL DONE] total={total} completed={completed} failed={failed} elapsed={(time.time()-start)/60:.1f}min")


if __name__ == "__main__":
    main()
