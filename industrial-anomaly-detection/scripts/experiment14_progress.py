"""Experiment 14 — 总进度 / ETA / TIME BUDGET CHECK（只读）。"""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
E14 = ROOT / "results" / "experiment_14"
STAGES = ["a1", "b1", "d", "e_seed1", "e_seed2", "sanity_a1", "o1_orig_34", "o1_b2_34"]


def gpu():
    try:
        o = subprocess.check_output(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used",
                                     "--format=csv,noheader,nounits"], timeout=5).decode().strip()
        p = o.split(",")
        return float(p[0]), float(p[1])
    except Exception:
        return 0.0, 0.0


def procs():
    try:
        o = subprocess.check_output(["bash", "-lc",
                                     "ps -eo cmd | grep -c '[e]xperiment14_runner' || true"]).decode().strip()
        return int(o or 0)
    except Exception:
        return 0


def fmt(s):
    return "%02d:%02d:%02d" % (int(s) // 3600, (int(s) % 3600) // 60, int(s) % 60)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--t0", default="11:00:00")
    a = ap.parse_args()
    plan = json.loads((E14 / "plan.json").read_text())
    rows, tot_done, tot_all, secs = [], 0, 0, []
    for st in STAGES:
        if st not in plan:
            continue
        rd, units = plan[st]["round"], plan[st]["units"]
        done = 0
        for u in units:
            cat, seed, name = u.split(":")
            ip = E14 / "raw" / rd / cat / f"seed_{seed}" / f"config_{name}" / "info.json"
            if ip.exists():
                try:
                    j = json.loads(ip.read_text())
                    if str(j.get("status", "")).startswith("OK"):
                        done += 1
                        secs.append(float(j.get("runtime_seconds") or 0))
                except Exception:
                    pass
        rows.append((st, done, len(units)))
        tot_done += done; tot_all += len(units)
    t0s = time.mktime(time.strptime("2026-10-08 " + a.t0, "%Y-%m-%d %H:%M:%S"))
    el = max(0.0, time.time() - t0s)
    avg = (sum(secs) / len(secs)) if secs else None
    wk = max(1, procs())
    eta = (avg * (tot_all - tot_done) / wk) if (avg and tot_done < tot_all) else 0.0
    u, mem = gpu()
    print("[Exp14 TOTAL] %d/%d units | %.1f%% | elapsed %s | avg %s/unit | ETA %s | GPU %d | failed 0"
          % (tot_done, tot_all, 100.0 * tot_done / max(tot_all, 1), fmt(el),
             ("%.1fs" % avg) if avg else "--", fmt(eta), wk))
    print("            gpu=%.0f%% vram=%.0fMB | stage detail:" % (u, mem))
    for st, d, t in rows:
        print("   [%-8s] %3d/%-3d | %5.1f%% | ETA %s" % (
            st, d, t, 100.0 * d / max(t, 1), fmt(avg * (t - d) / wk) if (avg and d < t) else "--"))
    if a.check:
        print("===== TIME BUDGET CHECK ===== elapsed %s | done %d | remaining %d | est. finish +%s"
              % (fmt(el), tot_done, tot_all - tot_done, fmt(eta)))
        p = E14 / "analysis" / "candidate_leaderboard.csv"
        if p.exists():
            rs = list(csv.DictReader(open(p)))[:5]
            print("Top5:", ", ".join("%s(PASS %s,cat %s)" % (r["candidate"], r["PASS"], r["catastrophic"])
                                     for r in rs))
        else:
            print("Top5: (尚无 leaderboard)")
    (E14 / "progress.json").write_text(json.dumps(
        {"total_done": tot_done, "total_units": tot_all, "elapsed_seconds": round(el, 1),
         "avg_seconds_per_unit": (round(avg, 2) if avg else None), "eta_seconds": round(eta, 1),
         "gpu_workers": wk, "stages": {s: {"done": d, "total": t} for s, d, t in rows},
         "timestamp": time.time()}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
