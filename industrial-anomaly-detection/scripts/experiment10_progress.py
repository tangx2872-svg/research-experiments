"""Experiment 10 — 实时进度 / ETA / progress.json（只读监视，不影响 runner）。

  python scripts/experiment10_progress.py            # 打印一次
  python scripts/experiment10_progress.py --watch 20 # 循环
计划来自 results/experiment_10_preservation_recovery/plan.json: {stage: [cat:seed:CONFIG, ...]}
"""
from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "results" / "experiment_10_preservation_recovery"
PLAN = EXP / "plan.json"


def _gpu_util() -> float:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
            timeout=5).decode().strip().splitlines()[0]
        return float(out)
    except Exception:
        return -1.0


def scan():
    done, failed, running = {}, {}, []
    for ip in sorted(EXP.glob("raw/*/*/seed_*/config_*/info.json")):
        d = ip.parent
        try:
            info = json.loads(ip.read_text())
        except Exception:
            continue
        cat = d.parent.parent.name
        seed = int(d.parent.name.split("_")[1])
        cfg = d.name.replace("config_", "")
        uid = f"{cat}:{seed}:{cfg}"
        st = str(info.get("status"))
        if st.startswith("FAILED"):
            failed[uid] = st[:80]
        elif st == "OK":
            done[uid] = float(info.get("runtime_seconds") or 0.0)
        if (d / "run.lock").exists():
            running.append(uid)
    return done, failed, running


def render(write: bool = True) -> str:
    plan = json.loads(PLAN.read_text()) if PLAN.exists() else {}
    done, failed, running = scan()
    t0 = min([p.stat().st_mtime for p in EXP.glob("logs/*.log")] or [time.time()])
    elapsed = max(0.0, time.time() - t0)
    all_units = [u for st, us in plan.items() for u in us]
    n, tot = len([u for u in all_units if u in done]), len(all_units)
    avg = (sum(done.values()) / len(done)) if done else None
    eta = (avg * (tot - n) / max(1, len(running))) if (avg and n < tot) else 0.0
    pct = 100.0 * n / tot if tot else 0.0
    fill = int(24 * n / tot) if tot else 0
    fmt = lambda s: "%02d:%02d:%02d" % (s // 3600, (s % 3600) // 60, s % 60)
    cur = [st for st, us in plan.items() if any(u not in done and u not in failed for u in us)]
    lines = ["Experiment10 | %s | %d/%d | %.1f%% | elapsed %s | ETA %s | %s"
             % (",".join(cur) if cur else "DONE", n, tot, pct, fmt(elapsed), fmt(eta),
                ("%.1fs/unit" % avg) if avg else "--"),
             "[" + "#" * fill + "." * (24 - fill) + "]  running=%d failed=%d gpu=%.0f%%"
             % (len(running), len(failed), _gpu_util())]
    for st, us in plan.items():
        dd = [u for u in us if u in done]
        lines.append("   %-12s %2d/%-2d" % (st, len(dd), len(us)))
    for u in running:
        lines.append("   run  %s" % u)
    for u, e in failed.items():
        lines.append("   FAIL %-40s %s" % (u, e))
    txt = "\n".join(lines)
    if write:
        payload = {"current_stage": ",".join(cur) if cur else "DONE",
                   "completed": n, "total": tot, "percent": round(pct, 2),
                   "elapsed_seconds": round(elapsed, 1), "eta_seconds": round(eta, 1),
                   "avg_seconds_per_unit": (round(avg, 1) if avg else None),
                   "running_units": running, "failed_units": failed,
                   "per_stage": {st: {"done": len([u for u in us if u in done]), "total": len(us)}
                                 for st, us in plan.items()},
                   "gpu_util": _gpu_util(), "timestamp": time.time()}
        try:
            EXP.mkdir(parents=True, exist_ok=True)
            (EXP / "progress.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False))
        except Exception:
            pass
    return txt


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--watch", type=int, default=0)
    a = ap.parse_args()
    if a.watch:
        while True:
            print(render(), flush=True)
            time.sleep(a.watch)
    else:
        print(render())
