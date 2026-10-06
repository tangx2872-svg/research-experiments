"""Experiment 6A — Geometry-Guided Soft Gating: runner（只跑 fragile 端的新 alpha）。

unit = (category, seed, alpha_F)，category in {bottle, grid}（fragile 端），
alpha_F 只能取自 results/experiment_6a/reference/policy_freeze.json 冻结的
candidate_grid + pre_registered_conditional_extension（防止任何新 alpha 被悄悄引入）。

执行代码零改动复用 Experiment 5A-H 的 run_config（split / preprocessing / bank / coreset /
illumination / 指标口径同一条代码路径），只把 output root 换成 results/experiment_6a/raw_new。

用法（tmux，3 workers）：
  python -u scripts/experiment6a_runner.py --units "bottle:1:0.25,grid:0:0.25" --worker-tag w1
  python -u scripts/experiment6a_runner.py --units "bottle:0:0.25" --worker-tag smoke --smoke
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import experiment1b_defect_sensitivity as e1b  # noqa: E402
import experiment5a_h_runner as h  # noqa: E402  (执行代码复用)

OUT_ROOT = ROOT / "results" / "experiment_6a" / "raw_new"
LOGS_DIR = ROOT / "results" / "experiment_6a" / "logs"
REF_DIR = ROOT / "results" / "experiment_6a" / "reference"
FREEZE = REF_DIR / "policy_freeze.json"
SCORES_5A = ROOT / "results" / "experiment_5a" / "raw" / "per_image_scores.csv"

h.OUT_ROOT = OUT_ROOT                 # 重定向输出（不触碰任何历史目录）
h.LOGS_DIR = LOGS_DIR

FRAGILE = ["bottle", "grid"]
TOLERANT = ["cable", "hazelnut", "screw"]


def md5_file(p: Path) -> str:
    import hashlib
    return hashlib.md5(p.read_bytes()).hexdigest()


def legal_alphas() -> list:
    fr = json.loads(FREEZE.read_text())
    a = [float(x) for x in fr["candidate_grid"]] + \
        [float(x) for x in fr["pre_registered_conditional_extension"]["values"]]
    return sorted(set(a))


def tag_of(alpha: float) -> str:
    return "aF" + f"{alpha:.8f}".replace(".", "")


def sanity_pre(tag: str, units: list) -> dict:
    res = {}
    fr = json.loads(FREEZE.read_text())
    res["R1_freeze_exists"] = (FREEZE.exists() and (REF_DIR / "asset_audit.csv").exists(),
                               f"freeze={FREEZE.relative_to(ROOT)}")
    legal = legal_alphas()
    res["R2_alpha_in_frozen_set"] = (all(any(abs(a - L) < 1e-12 for L in legal) for _, _, a in units),
                                     f"legal alpha set = {legal}")
    res["R3_fragile_only"] = (all(c in FRAGILE for c, _, _ in units),
                              f"fragile end only = {FRAGILE} (tolerant end stays at 0.5, reused from 5A-H)")
    res["R4_cuda_available"] = (torch.cuda.is_available(), f"cuda={torch.cuda.is_available()}")
    bad = [k for k, v in fr["input_md5"].items() if md5_file(ROOT / k) != v]
    res["R5_frozen_assets_unchanged"] = (not bad, f"{len(fr['input_md5'])} tracked assets md5 ok; bad={bad}")
    res["R6_no_new_alpha"] = (all(abs(a) > 1e-12 and abs(a - 0.5) > 1e-12 for _, _, a in units),
                              "runner refuses to run alpha_F in {0, 0.5} (those are reused from 5A-H)")
    res["R7_no_history_overwrite"] = (OUT_ROOT.name == "raw_new" and "experiment_6a" in str(OUT_ROOT),
                                      f"output root = {OUT_ROOT.relative_to(ROOT)} (new dir, no overwrite)")
    return res


def smoke_equivalence(alpha: float = 0.25) -> dict:
    """bottle:seed0 @ alpha=0.25 是 5A B1 的既有条件 -> 重跑必须逐位复现。"""
    p = OUT_ROOT / "bottle" / "seed_0" / f"config_{tag_of(alpha)}" / "per_image.csv"
    if not p.exists():
        return {"status": "PENDING"}
    sb = {(r["image_path"], r["subset"], r["shift"]): float(r["score"])
          for r in csv.DictReader(open(p, newline=""))}
    sa = {(r["image_path"], r["subset"], r["shift"]): float(r["score"])
          for r in csv.DictReader(open(SCORES_5A, newline="")) if r["config"] == "B1"}
    keys = set(sa) & set(sb)
    d = max(abs(sa[k] - sb[k]) for k in keys) if keys else float("nan")
    return {"status": "OK", "n_keys": len(keys), "max_abs_score_diff": d,
            "note": "5A B1 (bottle/seed0, alpha=0.25) vs 6A rerun"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--units", required=True, help="逗号分隔 category:seed:alpha")
    ap.add_argument("--worker-tag", default="w0")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    units = []
    for tok in args.units.split(","):
        c, s, a = tok.strip().split(":")
        units.append((c, int(s), float(a)))
    print(f"[{args.worker_tag}] PID={os.getpid()} units={units}", flush=True)

    sres = sanity_pre(args.worker_tag, units)
    for k, (ok, detail) in sres.items():
        print(f"[sanity {args.worker_tag}] {k}: {'PASS' if ok else 'FAIL'} ({detail})", flush=True)
    with open(LOGS_DIR / f"sanity_pre_{args.worker_tag}.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["check", "status", "detail"])
        for k, (ok, detail) in sres.items():
            w.writerow([k, "PASS" if ok else "FAIL", detail])
    fails = [k for k, v in sres.items() if not v[0]]
    if fails:
        raise SystemExit(f"[{args.worker_tag}] pre-run sanity FAILED: {fails} — STOP")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    (LOGS_DIR / f"runner_pid_{args.worker_tag}.txt").write_text(str(os.getpid()))

    for category, seed, alpha in units:
        assert category in FRAGILE, f"runner 只允许 fragile 端: {category}"
        assert any(abs(alpha - L) < 1e-12 for L in legal_alphas()), f"alpha {alpha} not in frozen set"
        val_ids, train_ids = e1b.make_validation_split(category=category, seed=seed)
        assert len(val_ids) == 20
        defect_types = e1b.discover_defect_types(category)
        good_paths = sorted((e1b.DATA_ROOT / category / "test" / "good").glob("*.png"))
        defect_paths = {dt: sorted((e1b.DATA_ROOT / category / "test" / dt).glob("*.png"))
                        for dt in defect_types}
        exp_rows = h.expected_rows(category, good_paths, defect_paths)
        tag = tag_of(alpha)
        cfg = {"name": tag, "alpha_l2": alpha, "alpha_l3": alpha}
        cfg_dir = OUT_ROOT / category / f"seed_{seed}" / f"config_{tag}"
        print(f"\n[{args.worker_tag}] unit {category}:{seed}:aF={alpha:.8f} "
              f"(train={len(train_ids)} good={len(good_paths)} "
              f"defect={sum(len(v) for v in defect_paths.values())})", flush=True)
        if h.unit_done(cfg_dir, exp_rows):
            print(f"[{category}:{seed}:{tag}] resume skip", flush=True)
            continue
        cfg_dir.mkdir(parents=True, exist_ok=True)
        lock = cfg_dir / "run.lock"
        if lock.exists():
            try:
                old_pid = int(lock.read_text().strip())
                os.kill(old_pid, 0)
                raise SystemExit(f"[{category}:{seed}:{tag}] 已有运行中 PID {old_pid}，冲突 STOP")
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

    if args.smoke:
        eq = smoke_equivalence(0.25)
        print(f"\n[{args.worker_tag}] SMOKE equivalence (bottle/seed0 @0.25 vs 5A B1): {eq}", flush=True)
        (LOGS_DIR / "smoke_equivalence.json").write_text(json.dumps(eq, indent=2))

    print(f"\n[{args.worker_tag}] all units done", flush=True)


if __name__ == "__main__":
    main()
