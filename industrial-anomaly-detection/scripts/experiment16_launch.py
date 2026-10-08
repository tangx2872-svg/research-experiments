"""Experiment 16 — 并发调度器（<=3 GPU processes，保留显存余量）+ 实时 ETA + 预计完成时间。"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "results" / "experiment_16_final_validation"


def unit_dir(rnd, unit):
    cat, seed, name = unit.split(":")
    return EXP / "raw" / rnd / cat / f"seed_{seed}" / f"config_{name}"


def done(rnd, unit):
    p = unit_dir(rnd, unit) / "info.json"
    try:
        return p.exists() and str(json.loads(p.read_text()).get("status", "")).startswith("OK")
    except Exception:
        return False


def secs(rnd, unit):
    try:
        return float(json.loads((unit_dir(rnd, unit) / "info.json").read_text()).get("runtime_seconds") or 0)
    except Exception:
        return 0.0


def gpu_mem():
    try:
        o = subprocess.check_output(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                                    timeout=5).decode().strip()
        return float(o)
    except Exception:
        return 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True)
    ap.add_argument("--workers", type=int, default=3)
    a = ap.parse_args()
    plan = json.loads((EXP / "plan.json").read_text())[a.stage]
    rnd, units = plan["round"], list(plan["units"])
    pending = [u for u in units if not done(rnd, u)]
    t0 = time.time()
    print(f"[Exp16 P4 GPU] stage={a.stage} round={rnd} total={len(units)} pending={len(pending)} workers={a.workers}",
          flush=True)
    (EXP / "logs").mkdir(parents=True, exist_ok=True)
    running, failed, spent = {}, [], []
    while pending or running:
        while pending and len(running) < a.workers:
            u = pending.pop(0)
            lg = open(EXP / "logs" / f"{a.stage}_{u.replace(':','_')}.log", "w")
            p = subprocess.Popen([sys.executable, "-u", str(ROOT / "scripts" / "experiment16_runner.py"),
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
            fin = (datetime.now() + timedelta(seconds=eta)).strftime("%H:%M")
            f = lambda s: "%02d:%02d:%02d" % (s // 3600, (s % 3600) // 60, s % 60)
            print(f"[Exp16 GPU] {nd}/{len(units)} | {100.0*nd/len(units):5.1f}% | elapsed {f(time.time()-t0)}"
                  f" | {('%.0fs' % avg) if avg else '--'}/unit | ETA {f(eta)} | finish ~{fin}"
                  f" | GPU {gpu_mem()/1024:.1f}/24GB | failed {len(failed)}", flush=True)
            (EXP / "progress.json").write_text(json.dumps(
                {"stage": a.stage, "completed": nd, "total": len(units),
                 "percent": round(100.0 * nd / len(units), 2), "elapsed_seconds": round(time.time() - t0, 1),
                 "avg_seconds_per_unit": (round(avg, 2) if avg else None), "eta_seconds": round(eta, 1),
                 "estimated_finish": fin, "gpu_used_mb": gpu_mem(), "workers": wk,
                 "failed_units": failed, "timestamp": time.time()}, indent=2, ensure_ascii=False))
    print(f"[Exp16/{a.stage}] DONE: {sum(1 for x in units if done(rnd,x))}/{len(units)} ok, failed={failed}",
          flush=True)
    return len(failed)


if __name__ == "__main__":
    sys.exit(main())
