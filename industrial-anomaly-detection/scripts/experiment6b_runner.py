"""Experiment 6B — Confirmatory validation runner（只跑审计后确认缺失的 unit）。

unit = (category, seed, alpha)。**白名单来自 6B freeze 的必需 key 集合**：
  GOAL1 fragile 端 confirm sampling（bottle/grid × 3 seeds × {0.125,0.20,0.30}）
  GOAL2 mean-α matched Uniform baseline（cable/hazelnut/screw × 3 seeds × alpha=0.40091275）
任何不在冻结必需集合中的 unit 都会被拒绝，防止悄悄引入新 α 或新类别组合。

执行代码零改动复用 Experiment 5A-H 的 run_config；输出写到 results/experiment_6b/raw_new/。

用法（screen，3 workers）：
  python -u scripts/experiment6b_runner.py --units "bottle:0:0.125" --worker-tag w1
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
import experiment5a_h_runner as h  # noqa: E402
import experiment6b_confirm as c6  # noqa: E402  (freeze + 必需 key 白名单)

OUT_ROOT = ROOT / "results" / "experiment_6b" / "raw_new"
LOGS_DIR = ROOT / "results" / "experiment_6b" / "logs"
REF_DIR = ROOT / "results" / "experiment_6b" / "reference"
FREEZE = REF_DIR / "policy_freeze.json"
SCORES_5A = ROOT / "results" / "experiment_5a" / "raw" / "per_image_scores.csv"

h.OUT_ROOT = OUT_ROOT
h.LOGS_DIR = LOGS_DIR


def tag_of(alpha: float) -> str:
    return "a" + f"{alpha:.8f}".replace(".", "")


def required_keys() -> set:
    fr = json.loads(FREEZE.read_text())
    return {(c, s, k) for (c, s, k) in c6.all_required_keys(fr)}


def sanity_pre(units: list) -> dict:
    fr = json.loads(FREEZE.read_text())
    req = required_keys()
    res = {}
    res["Q1_freeze_exists"] = (FREEZE.exists() and (REF_DIR / "asset_audit.csv").exists(),
                               f"freeze={FREEZE.relative_to(ROOT)}")
    res["Q2_units_in_frozen_required_set"] = (
        all((c, s, c6.e6a.akey(a)) in req for c, s, a in units),
        "every unit must be required by the frozen 6B protocol")
    res["Q3_identity_unchanged"] = (
        fr["inherits"]["identity"]["fragile"] == c6.FRAGILE
        and fr["inherits"]["identity"]["tolerant"] == c6.TOLERANT,
        f"fragile={c6.FRAGILE}, tolerant={c6.TOLERANT} @ {c6.ALPHA_T}")
    res["Q4_cuda_available"] = (torch.cuda.is_available(), f"cuda={torch.cuda.is_available()}")
    bad = [k for k, v in fr["input_md5"].items() if c6.e6a.md5_file(ROOT / k) != v]
    res["Q5_frozen_assets_unchanged"] = (not bad, f"{len(fr['input_md5'])} assets md5 ok; bad={bad}")
    res["Q6_no_history_overwrite"] = (OUT_ROOT.name == "raw_new" and "experiment_6b" in str(OUT_ROOT),
                                      f"output root = {OUT_ROOT.relative_to(ROOT)}")
    return res


def smoke_equivalence() -> dict:
    """bottle:seed0 @ alpha=0.40091275 是 5A C1 的既有条件 -> 重跑必须逐位复现。"""
    p = OUT_ROOT / "bottle" / "seed_0" / f"config_{tag_of(0.40091275)}" / "per_image.csv"
    if not p.exists():
        return {"status": "PENDING"}
    sb = {(r["image_path"], r["subset"], r["shift"]): float(r["score"])
          for r in csv.DictReader(open(p, newline=""))}
    sa = {(r["image_path"], r["subset"], r["shift"]): float(r["score"])
          for r in csv.DictReader(open(SCORES_5A, newline="")) if r["config"] == "C1"}
    keys = set(sa) & set(sb)
    return {"status": "OK", "n_keys": len(keys),
            "max_abs_score_diff": max(abs(sa[k] - sb[k]) for k in keys) if keys else float("nan"),
            "note": "5A C1 (bottle/seed0, alpha=0.40091275) vs 6B rerun"}


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

    sres = sanity_pre(units)
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
        assert (category, seed, c6.e6a.akey(alpha)) in required_keys(), \
            f"unit {category}:{seed}:{alpha} is NOT required by the frozen 6B protocol"
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
        print(f"\n[{args.worker_tag}] unit {category}:{seed}:a={alpha:.8f} "
              f"(train={len(train_ids)} good={len(good_paths)} "
              f"defect={sum(len(v) for v in defect_paths.values())})", flush=True)
        if h.unit_done(cfg_dir, exp_rows):
            print(f"[{category}:{seed}:{tag}] resume skip", flush=True)
            continue
        cfg_dir.mkdir(parents=True, exist_ok=True)
        lock = cfg_dir / "run.lock"
        if lock.exists():
            try:
                old = int(lock.read_text().strip())
                os.kill(old, 0)
                raise SystemExit(f"[{category}:{seed}:{tag}] 已有运行中 PID {old}，冲突 STOP")
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
        eq = smoke_equivalence()
        print(f"\n[{args.worker_tag}] SMOKE equivalence (bottle/seed0 @0.40091275 vs 5A C1): {eq}", flush=True)
        (LOGS_DIR / "smoke_equivalence.json").write_text(json.dumps(eq, indent=2))
    print(f"\n[{args.worker_tag}] all units done", flush=True)


if __name__ == "__main__":
    main()
