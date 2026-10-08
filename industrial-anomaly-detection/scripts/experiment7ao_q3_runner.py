"""Overnight Queue Q3 — illumination stress-test runner。

复用 Q0 的 fit 代码路径（experiment7ao_runner.patched_fit_dual_model）与 5A-H 的 scoring 原语
（e1b.predict_one / e1.apply_photometric），只把 shift 集合扩展为 Q3 冻结的 12 个 condition。
不计算 pixel AUROC / AUPRO（Q3 不需要，且不影响任何 shift 分数）。

用法：
  python -u scripts/experiment7ao_q3_runner.py --units "M0_original:bottle:0,M1_uniform:bottle:0" --worker-tag q3w1
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import experiment7ao_config as c7  # noqa: E402
import experiment7ao_q3 as q3  # noqa: E402
import experiment7ao_runner as r7  # noqa: E402
import experiment1b_defect_sensitivity as e1b  # noqa: E402
import experiment1_illumination_tradeoff as e1  # noqa: E402

OUT_ROOT = q3.Q3_RAW
LOGS_DIR = q3.Q3_LOG


def method_spec(name: str) -> dict:
    if name in q3.METHOD_SPECS:
        return q3.METHOD_SPECS[name]
    if name in r7.ALL_SPECS:            # 允许把 Q2 winner 的 spec 直接注入
        return r7.ALL_SPECS[name]
    raise SystemExit(f"[Q3 FATAL] unknown method {name}")


def run_unit(method: str, category: str, seed: int, device: torch.device) -> None:
    spec = method_spec(method)
    cfg_dir = OUT_ROOT / category / f"seed_{seed}" / f"method_{method}"
    val_ids, train_ids = e1b.make_validation_split(category=category, seed=seed)
    assert len(val_ids) == 20
    good_paths = sorted((e1b.DATA_ROOT / category / "test" / "good").glob("*.png"))
    defect_types = e1b.discover_defect_types(category)
    defect_paths = {dt: sorted((e1b.DATA_ROOT / category / "test" / dt).glob("*.png")) for dt in defect_types}
    n_def = sum(len(v) for v in defect_paths.values())
    expected = 20 + len(good_paths) + n_def + len(q3.CONDITIONS) * len(good_paths)
    if (cfg_dir / "info.json").exists():
        try:
            d = json.loads((cfg_dir / "info.json").read_text())
            if d.get("status") == "OK" and d.get("n_rows") == expected:
                print(f"[{method}:{category}:{seed}] resume skip", flush=True)
                return
        except Exception:
            pass
    cfg_dir.mkdir(parents=True, exist_ok=True)
    lock = cfg_dir / "run.lock"
    if lock.exists():
        try:
            old = int(lock.read_text().strip())
            os.kill(old, 0)
            raise SystemExit(f"[{method}:{category}:{seed}] running PID {old} — conflict STOP")
        except (ProcessLookupError, ValueError, OSError):
            pass
    lock.write_text(str(os.getpid()))
    t0 = time.time()
    rows: list = []

    def add(subset, cond, defect_type, path, score):
        rows.append({"method": method, "category": category, "seed": seed, "subset": subset,
                     "condition": cond, "severity": "", "family": "", "defect_type": defect_type,
                     "image_path": str(path), "score": score})

    try:
        r7.CAPTURED.clear()
        r7.CAPTURED.update({"config": method, "spec": spec, "embed_dim": -1})
        lm, tm = r7.patched_fit_dual_model(None, None, category, seed, train_ids, LOGS_DIR)
        e1b._move_model_to_device(tm, device)
        val_scores = []
        for vid in val_ids:
            s, _ = e1b.predict_one(tm, e1b.load_image_as_tensor(e1b.DATA_ROOT / category / "train" / "good" / vid), device)
            val_scores.append(float(s))
            add("val", "none", "good", vid, float(s))
        tau_val = float(max(val_scores))
        for p in good_paths:
            s, _ = e1b.predict_one(tm, e1b.load_image_as_tensor(p), device)
            add("clean_good", "none", "good", p, float(s))
        for dt, paths in defect_paths.items():
            for p in paths:
                s, _ = e1b.predict_one(tm, e1b.load_image_as_tensor(p), device)
                add("clean_defect", "none", dt, p, float(s))
        for cname, fam, lv, sev in q3.CONDITIONS:
            for p in good_paths:
                img_s = e1.apply_photometric(e1.load_image_as_tensor(p), fam, lv)
                s, _ = e1b.predict_one(tm, img_s, device)
                r = {"method": method, "category": category, "seed": seed, "subset": "shift_good",
                     "condition": cname, "severity": sev, "family": fam, "defect_type": "good",
                     "image_path": str(p), "score": float(s)}
                rows.append(r)
    except Exception as exc:
        (cfg_dir / "info.json").write_text(json.dumps({"status": f"FAILED: {type(exc).__name__}: {exc}"}))
        lock.unlink(missing_ok=True)
        raise
    peak = "not_measured"
    if torch.cuda.is_available():
        try:
            peak = round(torch.cuda.max_memory_allocated() / (1024 * 1024), 2)
        except Exception:
            pass
    with open(cfg_dir / "per_image.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    info = {"status": "OK", "queue": "Q3", "method": method, "category": category, "seed": seed,
            "spec": spec, "tau_val": tau_val, "n_rows": len(rows), "expected_rows": expected,
            "n_conditions": len(q3.CONDITIONS), "runtime_seconds": round(time.time() - t0, 2),
            "peak_gpu_memory_allocated_mb": peak,
            "embedding_dim": r7.CAPTURED.get("embed_dim", -1),
            "freeze_sha256": c7.sha256_file(q3.Q3_REF / "q3_protocol_freeze.json")}
    (cfg_dir / "info.json").write_text(json.dumps(info, indent=2))
    lock.unlink(missing_ok=True)
    print(f"[{method}:{category}:{seed}] OK rows={len(rows)} runtime={info['runtime_seconds']}s "
          f"tau={tau_val:.4f}", flush=True)
    del lm, tm, rows
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--units", required=True, help="逗号分隔 method:category:seed")
    ap.add_argument("--worker-tag", default="q3w0")
    args = ap.parse_args()
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    units = []
    for tok in args.units.split(","):
        m, c, s = tok.strip().split(":")
        units.append((m, c, int(s)))
    print(f"[{args.worker_tag}] PID={os.getpid()} units={units}", flush=True)
    (LOGS_DIR / f"runner_pid_{args.worker_tag}.txt").write_text(str(os.getpid()))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for m, c, s in units:
        print(f"\n[{args.worker_tag}] unit {m}:{c}:{s}", flush=True)
        run_unit(m, c, s, device)
    print(f"\n[{args.worker_tag}] all units done", flush=True)


if __name__ == "__main__":
    main()
