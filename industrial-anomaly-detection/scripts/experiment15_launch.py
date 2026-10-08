"""Experiment 15 — 并发调度器（<=4 GPU processes）+ 断点续跑 + 动态 ETA + CURRENT LEADER。"""
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "results" / "experiment_15"


def unit_dir(rnd: str, unit: str) -> Path:
    cat, seed, name = unit.split(":")
    return EXP / "raw" / rnd / cat / f"seed_{seed}" / f"config_{name}"


def done(rnd: str, unit: str) -> bool:
    p = unit_dir(rnd, unit) / "info.json"
    try:
        return p.exists() and str(json.loads(p.read_text()).get("status", "")).startswith("OK")
    except Exception:
        return False


def secs(rnd: str, unit: str) -> float:
    try:
        return float(json.loads((unit_dir(rnd, unit) / "info.json").read_text()).get("runtime_seconds") or 0)
    except Exception:
        return 0.0


def leader() -> str:
    try:
        lb = list(csv.DictReader(open(EXP / "analysis" / "leaderboard.csv")))
        if lb:
            return f" | CURRENT LEADER: {lb[0]['method']} | evidence={lb[0].get('evidence','E1')}"
    except Exception:
        pass
    return ""


def run_stage(stage: str, workers: int) -> int:
    plan = json.loads((EXP / "plan.json").read_text())[stage]
    rnd, units = plan["round"], list(plan["units"])
    pending = [u for u in units if not done(rnd, u)]
    print(f"[Exp15/{stage}] round={rnd} total={len(units)} pending={len(pending)} workers={workers}", flush=True)
    (EXP / "logs").mkdir(parents=True, exist_ok=True)
    t0, running, failed, spent = time.time(), {}, [], []
    while pending or running:
        while pending and len(running) < workers:
            u = pending.pop(0)
            lg = open(EXP / "logs" / f"{stage}_{u.replace(':','_')}.log", "w")
            p = subprocess.Popen([sys.executable, "-u", str(ROOT / "scripts" / "experiment15_runner.py"),
                                  "--units", u, "--round", rnd, "--worker-tag", u.replace(":", "_")],
                                 stdout=lg, stderr=subprocess.STDOUT, cwd=str(ROOT),
                                 env={**os.environ, "HF_ENDPOINT": "https://hf-mirror.com"})
            running[p.pid] = (u, p, time.time(), lg)
        time.sleep(5)
        for pid in list(running):
            u, p, ts, lg = running[pid]
            if p.poll() is None:
                continue
            lg.close(); del running[pid]
            (spent.append(secs(rnd, u) or (time.time() - ts)) if done(rnd, u) else failed.append(u))
            nd = sum(1 for x in units if done(rnd, x))
            avg = (sum(spent) / len(spent)) if spent else None
            wk = max(1, len(running))
            eta = (avg * (len(pending) + len(running)) / wk) if avg else 0.0
            f = lambda s: "%02d:%02d:%02d" % (s // 3600, (s % 3600) // 60, s % 60)
            print(f"[Exp15 TOTAL] {nd}/{len(units)} | {100.0*nd/len(units):5.1f}% | elapsed {f(time.time()-t0)}"
                  f" | avg {('%.1fs' % avg) if avg else '--'}/unit | ETA {f(eta)} | workers {wk}"
                  f" | failed {len(failed)}{leader()}", flush=True)
            (EXP / "progress.json").write_text(json.dumps(
                {"stage": stage, "completed": nd, "total": len(units),
                 "percent": round(100.0 * nd / len(units), 2), "elapsed_seconds": round(time.time() - t0, 1),
                 "avg_seconds_per_unit": (round(avg, 2) if avg else None), "eta_seconds": round(eta, 1),
                 "workers": wk, "failed_units": failed, "timestamp": time.time()}, indent=2, ensure_ascii=False))
    print(f"[Exp15/{stage}] DONE: {sum(1 for x in units if done(rnd,x))}/{len(units)} ok, failed={failed}", flush=True)
    return len(failed)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True)
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    sys.exit(run_stage(a.stage, min(a.workers, 4)))
