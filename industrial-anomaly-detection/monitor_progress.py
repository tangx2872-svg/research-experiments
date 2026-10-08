#!/usr/bin/env python
"""独立进度监控（只读，不干扰正在运行的实验）。

用法：
  python monitor_progress.py                      # 自动发现最新 progress 分片
  python monitor_progress.py --root results/experiment_9a_screening
  python monitor_progress.py --root results/experiment_9a_screening --name round0
  python monitor_progress.py --scan results/experiment_7a_o/raw   # 无 progress 分片时的旧目录扫描
  python monitor_progress.py --once               # 只输出一次，不循环
  python monitor_progress.py --interval 20        # 刷新间隔（默认 25s）

ETA 口径（多 worker 感知）：
  * throughput：最近 N 个**真实完成** unit 的 wall-clock 速率（自动含并行度）——优先；
  * duration/worker：最近 N 个 unit 平均耗时 ÷ 活跃 worker 数——兜底。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "scripts"))

import experiment_progress as ep  # noqa: E402


def discover_latest_progress(results_root: Path):
    """找最新被写入的 progress 分片所在 (root, name)。"""
    best = None
    for p in results_root.glob("*/progress/*__*.json"):
        try:
            j = json.loads(p.read_text())
        except Exception:
            continue
        m = p.stat().st_mtime
        if best is None or m > best[0]:
            best = (m, p.parent.parent, j.get("name"))
    return (best[1], best[2]) if best else (None, None)


def latest_results_dir(results_root: Path):
    cands = [d for d in results_root.glob("*") if d.is_dir()]
    if not cands:
        return None
    return max(cands, key=lambda d: d.stat().st_mtime)


def scan_raw(raw_root: Path, manifest: Path | None = None):
    """旧实验兜底：从 raw/**/info.json + run.lock 统计。"""
    done, running, durations, fail = [], [], [], []
    pats = ["*/*/*/info.json", "*/*/info.json",
            "*/*/*/*/info.json", "*/*/*/*/*/info.json"]  # [cat/seed/cfg] 与 [round/cat/seed/cfg]
    units_dirs = sorted({p.parent for pat in pats for p in raw_root.glob(pat)})
    for d in units_dirs:
        info_p = d / "info.json"
        try:
            info = json.loads(info_p.read_text())
        except Exception:
            continue
        uid = f"{d.parent.parent.name}:{d.name}"
        if (d / "run.lock").exists():
            running.append(uid)
        elif info.get("status") == "OK":
            done.append(uid)
            if info.get("runtime_seconds"):
                durations.append(float(info["runtime_seconds"]))
        else:
            fail.append(uid)
    total = None
    if manifest and manifest.exists():
        try:
            m = json.loads(manifest.read_text())
            total = len(m.get("units", m)) if isinstance(m, (dict, list)) else None
        except Exception:
            total = None
    if total is None:
        total = len(done) + len(running) + len(fail) or None
    starts = []
    for d in units_dirs:
        info_p = d / "info.json"
        if info_p.exists() and (d / "run.lock").exists():
            starts.append(info_p.stat().st_mtime)
    return {"total": total or 0, "completed": len(done) + len(fail), "ok": len(done),
            "failed": len(fail), "durations": durations, "running": running,
            "started_at": min([d.stat().st_mtime for d in units_dirs], default=None)}



def write_progress_json(root: Path, name, est, shards):
    """把跨 worker 聚合快照写成 <root>/progress.json（协议 §10 要求，含断点信息）。"""
    if root is None:
        return
    out = {
        "round": name, "completed": est["completed"], "total": est["total"],
        "remaining": est["remaining"], "failed": est["failed"],
        "percentage": round(100.0 * est["completed"] / est["total"], 1) if est["total"] else 0.0,
        "current_unit": est["last_unit"],
        "current_candidate": (est["last_unit"] or "").split(":")[-1] or None,
        "current_category": (est["last_unit"] or "").split(":")[0] or None,
        "workers_active": est["workers_active"],
        "elapsed_seconds": round(est["elapsed"], 1),
        "avg_unit_seconds": est["avg_unit_seconds"],
        "throughput_units_per_min": est["throughput_units_per_min"],
        "eta_seconds": est["eta_seconds"], "eta_basis": est["eta_basis"],
        "estimated_finish_time": (time.strftime("%Y-%m-%d %H:%M:%S",
                                 time.localtime(est["estimated_finish"]))
                                  if est["estimated_finish"] else None),
        "last_update": time.strftime("%Y-%m-%d %H:%M:%S"),
        "per_worker": [{"worker": j.get("worker"), "pid": j.get("pid"),
                        "done": len(j.get("units", [])), "of": j.get("units_total"),
                        "current": j.get("current"),
                        "unit_ids": j.get("unit_ids")} for j in shards],
    }
    (root / "progress.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))


def print_progress_mode(root: Path, name, results_root: Path):
    shards, alive = ep.load_shards(root, name)
    if not shards:
        print(f"[monitor] no progress shards under {root}/progress (name={name})")
        return None
    est = ep.estimate(shards, alive_workers=alive)
    print(ep.render_bar(est))
    for j in sorted(shards, key=lambda s: s.get("worker", "")):
        units = j.get("units", [])
        cur = j.get("current") or "-"
        print(f"  worker {j.get('worker'):<6} pid={j.get('pid')} "
              f"{len(units)}/{j.get('units_total')} done | current: {cur} | "
              f"updated {time.strftime('%H:%M:%S', time.localtime(j.get('updated_at', 0)))}")
    tail = est.get("_tail") or []
    recent = ep.aggregate(shards)["units"][-3:]
    for u in recent:
        print(f"  last done: {u['id']:<24} {u.get('seconds')}s  ({u.get('worker')})")
    return est


def print_scan_mode(raw_root: Path, manifest):
    sc = scan_raw(raw_root, manifest)
    total, done = sc["total"], sc["completed"]
    durations = sc["durations"]
    recent = durations[-8:]
    avg = (sum(recent) / len(recent)) if recent else None
    remaining = max(total - done, 0)
    workers = max(1, len(sc["running"]))
    eta = (remaining * avg / workers) if avg else None
    est = {"total": total, "completed": done, "elapsed": 0.0,
           "avg_unit_seconds": avg, "eta_seconds": eta,
           "eta_basis": "duration/worker", "estimated_finish": (time.time() + eta) if eta else None,
           "workers_active": workers, "throughput_units_per_min": None}
    print(ep.render_bar(est))
    print(f"  running: {len(sc['running'])} {sc['running'][:3]}")
    print(f"  failed : {sc['failed']}")
    return est


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="")
    ap.add_argument("--name", default="")
    ap.add_argument("--scan", default="")
    ap.add_argument("--manifest", default="")
    ap.add_argument("--interval", type=float, default=25.0)
    ap.add_argument("--once", action="store_true")
    args = ap.parse_args()

    results_root = ROOT / "results"
    root = Path(args.root) if args.root else None
    name = args.name or None
    if args.scan:
        root, name = None, None
    elif root is None:
        root, name = discover_latest_progress(results_root)
        if root is None:
            d = latest_results_dir(results_root)
            print(f"[monitor] no progress shards found; newest results dir = {d}")
            print("[monitor] 提示：--scan <results/xxx/raw> 可扫描旧实验目录")
            return
    while True:
        print("=" * 72)
        print(f"[monitor] {time.strftime('%Y-%m-%d %H:%M:%S')} | "
              f"root={root if root else args.scan} name={name}")
        if args.scan:
            print_scan_mode(Path(args.scan), Path(args.manifest) if args.manifest else None)
        else:
            est = print_progress_mode(root, name, results_root)
            if est is not None:
                shards, _alive = ep.load_shards(root, name)
                write_progress_json(root, name, est, shards)
        print("=" * 72, flush=True)
        if args.once:
            return
        time.sleep(args.interval)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[monitor] stopped")
