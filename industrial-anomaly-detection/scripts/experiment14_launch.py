"""Experiment 14 — 并发调度器（<=N GPU processes）+ 断点续跑 + 动态 ETA。

  python scripts/experiment14_launch.py --stage a1 --workers 4
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "results" / "experiment_14"
CATS = ["bottle", "cable", "hazelnut", "screw", "grid"]


def unit_dir(round_name: str, unit: str) -> Path:
    cat, seed, name = unit.split(":")
    return EXP / "raw" / round_name / cat / f"seed_{seed}" / f"config_{name}"


def done(round_name: str, unit: str) -> bool:
    p = unit_dir(round_name, unit) / "info.json"
    if not p.exists():
        return False
    try:
        return str(json.loads(p.read_text()).get("status", "")).startswith("OK")
    except Exception:
        return False


def runtime(round_name: str, unit: str):
    p = unit_dir(round_name, unit) / "info.json"
    try:
        return float(json.loads(p.read_text()).get("runtime_seconds") or 0.0)
    except Exception:
        return 0.0


def run_stage(stage: str, workers: int) -> int:
    plan = json.loads((EXP / "plan.json").read_text())[stage]
    round_name, units = plan["round"], list(plan["units"])
    pending = [u for u in units if not done(round_name, u)]
    total = len(units)
    print(f"[Exp14/{stage}] round={round_name} total={total} pending={len(pending)} workers={workers}", flush=True)
    (EXP / "logs").mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    running: dict = {}
    failed: list = []
    spent: list = []
    while pending or running:
        while pending and len(running) < workers:
            u = pending.pop(0)
            lg = open(EXP / "logs" / f"{stage}_{u.replace(':','_')}.log", "w")
            p = subprocess.Popen([sys.executable, "-u", str(ROOT / "scripts" / "experiment14_runner.py"),
                                  "--units", u, "--round", round_name,
                                  "--worker-tag", u.replace(":", "_")],
                                 stdout=lg, stderr=subprocess.STDOUT, cwd=str(ROOT),
                                 env={**__import__("os").environ, "HF_ENDPOINT": "https://hf-mirror.com"})
            running[p.pid] = (u, p, time.time(), lg)
        time.sleep(5)
        for pid in list(running):
            u, p, ts, lg = running[pid]
            if p.poll() is None:
                continue
            lg.close()
            del running[pid]
            el = time.time() - ts
            if done(round_name, u):
                spent.append(runtime(round_name, u) or el)
            else:
                failed.append(u)
            nd = sum(1 for x in units if done(round_name, x))
            avg = (sum(spent) / len(spent)) if spent else None
            wk = max(1, len(running))
            eta = (avg * (len(pending) + len(running)) / wk) if avg else 0.0
            fmt = lambda s: "%02d:%02d:%02d" % (s // 3600, (s % 3600) // 60, s % 60)
            print(f"[Exp14/{stage}] {nd}/{total} | {100.0*nd/total:5.1f}% | elapsed {fmt(time.time()-t0)} "
                  f"| avg {('%.1fs' % avg) if avg else '--'}/unit | ETA {fmt(eta)} | GPU {wk} | failed {len(failed)}"
                  f"{' | current=' + u if running else ''}", flush=True)
            (EXP / "progress.json").write_text(json.dumps(
                {"stage": stage, "round": round_name, "completed": nd, "total": total,
                 "percent": round(100.0 * nd / total, 2), "elapsed_seconds": round(time.time() - t0, 1),
                 "avg_seconds_per_unit": (round(avg, 2) if avg else None),
                 "eta_seconds": round(eta, 1), "gpu_workers": wk, "failed_units": failed,
                 "timestamp": time.time()}, indent=2, ensure_ascii=False))
    print(f"[Exp14/{stage}] DONE: {sum(1 for x in units if done(round_name,x))}/{total} ok, failed={failed}", flush=True)
    return len(failed)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True)
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    sys.exit(run_stage(a.stage, min(a.workers, 5)))
