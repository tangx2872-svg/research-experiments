"""Experiment 9B-R 实时进度 / ETA 监视器（只读，不影响 runner）。

  python scripts/experiment9br_progress.py --total 5 [--watch 15]
显示：completed/total、%、elapsed、ETA（按真实完成 unit 平均耗时动态更新）、s/unit、running、failed。
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "results" / "experiment_9b_r_strict_replay"


def scan():
    done, failed, running, t0 = [], [], [], None
    for ip in sorted(EXP.glob("raw/*/*/seed_*/config_*/info.json")):
        d = ip.parent
        try:
            info = json.loads(ip.read_text())
        except Exception:
            continue
        st = str(info.get("status"))
        unit = "%s/%s" % (d.parent.parent.parent.name, d.name.replace("config_", ""))
        lock = d / "run.lock"
        if st.startswith("FAILED"):
            failed.append((unit, st[:60]))
            continue
        if st == "OK":
            done.append((unit, float(info.get("runtime_seconds") or 0.0)))
        if lock.exists():
            running.append(unit)
    shards = sorted(EXP.glob("progress/*.json"))
    for s in shards:
        try:
            j = json.loads(s.read_text())
        except Exception:
            continue
        for u in j.get("units", []):
            if u.get("started_at") and (t0 is None or u["started_at"] < t0):
                t0 = u["started_at"]
        if j.get("started_at") and (t0 is None or j["started_at"] < t0):
            t0 = j["started_at"]
    return done, failed, running, t0


def render(total: int) -> str:
    done, failed, running, t0 = scan()
    n = len(done)
    elapsed = (time.time() - t0) if t0 else 0.0
    avg = (sum(d[1] for d in done) / n) if n else None
    pct = 100.0 * n / total if total else 0.0
    fill = int(22 * n / total) if total else 0
    bar = "[" + "#" * fill + "." * (22 - fill) + "]"
    if avg and n < total:
        eta = avg * (total - n) / max(1, len(running) or 1)
    else:
        eta = 0.0
    fmt = lambda s: "%02d:%02d:%02d" % (s // 3600, (s % 3600) // 60, s % 60)
    lines = ["Experiment 9B-R · Strict Replay Re-evaluation",
             "%s %d/%d | %.1f%%" % (bar, n, total, pct),
             "Elapsed: %s" % fmt(elapsed),
             "ETA: %s%s" % (fmt(eta), " (done)" if n >= total else ""),
             "Avg/unit: %s | workers: %d | Failed: %d"
             % (("%.1fs" % avg) if avg else "--", max(1, len(running)), len(failed))]
    for u, s in done:
        lines.append("   done  %-38s %6.1fs" % (u, s))
    for u in running:
        lines.append("   run   %s" % u)
    for u, st in failed:
        lines.append("   FAIL  %-38s %s" % (u, st))
    return "\n".join(lines)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--total", type=int, required=True)
    ap.add_argument("--watch", type=int, default=0)
    a = ap.parse_args()
    if a.watch:
        while True:
            print(render(a.total), flush=True)
            time.sleep(a.watch)
    else:
        print(render(a.total))
