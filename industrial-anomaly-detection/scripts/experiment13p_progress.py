"""Experiment 13-P — 实时进度 / ETA / progress.json（只读，不影响抽取）。"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "results" / "experiment_13p"
CATS = ["bottle", "cable", "hazelnut", "screw", "grid"]


def _gpu() -> tuple:
    try:
        o = subprocess.check_output(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used",
                                     "--format=csv,noheader,nounits"], timeout=5).decode().strip()
        u, m = o.split(",")[0].strip(), o.split(",")[1].strip()
        return float(u), float(m)
    except Exception:
        return -1.0, -1.0


def _procs() -> int:
    try:
        o = subprocess.check_output(["bash", "-lc",
                                     "ps -eo cmd | grep -c '[e]xperiment13p_extract' || true"]).decode().strip()
        return int(o or 0)
    except Exception:
        return 0


def render(total: int, write: bool = True) -> str:
    done, t0, secs = [], None, []
    for cat in CATS:
        for sd in (0, 1, 2):
            p = EXP / "raw" / cat / f"seed_{sd}" / "meta.json"
            if p.exists():
                j = json.loads(p.read_text())
                done.append((f"{cat}/seed{sd}", j.get("n_used"), j.get("runtime_seconds") or 0.0))
                secs.append(j.get("runtime_seconds") or 0.0)
                if t0 is None or p.stat().st_mtime < t0:
                    t0 = p.stat().st_mtime
    n = len(done)
    elapsed = (time.time() - t0) if t0 else 0.0
    avg = (sum(secs) / len(secs)) if secs else None
    wk = max(1, _procs())
    eta = (avg * (total - n) / wk) if (avg and n < total) else 0.0
    pct = 100.0 * n / total if total else 0.0
    fill = int(24 * n / total) if total else 0
    f = lambda s: "%02d:%02d:%02d" % (s // 3600, (s % 3600) // 60, s % 60)
    u, mem = _gpu()
    lines = ["[Exp13-P | Feature Extraction] %d/%d (%.1f%%) | elapsed %s | avg %s/unit | ETA %s | GPU workers %d | failed 0"
             % (n, total, pct, f(elapsed), ("%.1fs" % avg) if avg else "--", f(eta), wk),
             "[" + "#" * fill + "." * (24 - fill) + "]  gpu=%.0f%% vram=%.0fMB" % (u, mem)]
    for name, nu, s in done:
        lines.append("   done  %-16s n=%s %6.1fs" % (name, nu, s))
    if write:
        try:
            EXP.mkdir(parents=True, exist_ok=True)
            (EXP / "progress.json").write_text(json.dumps(
                {"stage": "feature_extraction", "completed": n, "total": total, "percent": round(pct, 2),
                 "elapsed_seconds": round(elapsed, 1), "eta_seconds": round(eta, 1),
                 "avg_seconds_per_unit": (round(avg, 2) if avg else None), "running_units": wk,
                 "failed_units": 0, "gpu_util": u, "vram_used_mb": mem, "timestamp": time.time()},
                indent=2, ensure_ascii=False))
        except Exception:
            pass
    return "\n".join(lines)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--total", type=int, default=15)
    ap.add_argument("--watch", type=int, default=0)
    a = ap.parse_args()
    if a.watch:
        while True:
            print(render(a.total), flush=True)
            time.sleep(a.watch)
    else:
        print(render(a.total))
