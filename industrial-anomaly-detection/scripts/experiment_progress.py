"""共享进度 / ETA 工具（Experiment 9A 起，所有新 runner 默认使用）。

要点：
  * 每个 worker 独立写分片 `<root>/progress/<name>__<worker>.json`，unit 结束立刻刷新；
  * 任意进程（含独立 monitor_progress.py）只读聚合，动态算真实 ETA；
  * ETA 双口径：throughput（最近 N 次真实完成 wall-clock 速率，含并行度）优先，
    duration/worker（最近 N 个 unit 平均耗时 ÷ 活跃 worker 数）兜底。
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

PROGRESS_DIRNAME = "progress"
BAR_WIDTH = 22
RECENT_WINDOW = 8
# throughput 基需要足够长的时间跨度，否则「多个 unit 几乎同时完成」会算出荒谬的速率
MIN_RATE_SPAN = 20.0   # seconds


def progress_dir(root: Path) -> Path:
    return Path(root) / PROGRESS_DIRNAME


def shard_path(root: Path, name: str, worker: str) -> Path:
    return progress_dir(root) / f"{name}__{worker}.json"


def load_shards(root: Path, name: str | None = None, alive_seconds: float = 300.0):
    """读取 <root>/progress/*.json -> (shards, n_alive)。只读，不创建目录。"""
    d = progress_dir(root)
    shards = []
    if not d.exists():
        return shards, 0
    for p in sorted(d.glob("*.json")):
        try:
            j = json.loads(p.read_text())
        except Exception:
            continue
        if name and j.get("name") != name:
            continue
        j["_path"] = str(p)
        shards.append(j)
    now = time.time()
    n_alive = sum(1 for j in shards if now - float(j.get("updated_at", 0)) <= alive_seconds)
    return shards, n_alive


def aggregate(shards) -> dict:
    # 每个 worker 一个分片；全局 total = 各活跃 worker 的 unit 数之和
    # （同一 worker tag 若有多个分片，只取 updated_at 最新的，避免重复计数）
    latest = {}
    for j in shards:
        w = j.get("worker") or j.get("_path")
        if w not in latest or float(j.get("updated_at", 0)) > float(latest[w].get("updated_at", 0)):
            latest[w] = j
    total = sum(int(j.get("units_total", 0)) for j in latest.values())
    units = []
    for j in shards:
        for u in j.get("units", []):
            u = dict(u)
            u["worker"] = j.get("worker")
            units.append(u)
    units.sort(key=lambda u: float(u.get("ts", 0.0)))
    ok = [u for u in units if u.get("status") == "ok"]
    failed = [u for u in units if u.get("status") == "failed"]
    durations = [float(u["seconds"]) for u in ok if u.get("seconds") is not None]
    starts = [float(j["started_at"]) for j in shards if j.get("started_at")]
    return {"total": total, "completed": len(ok) + len(failed), "ok": len(ok),
            "failed": len(failed), "units": units, "durations": durations,
            "started_at": min(starts) if starts else None}


def fmt_hms(seconds):
    if seconds is None:
        return "--:--"
    seconds = int(max(seconds, 0))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


def fmt_clock(epoch):
    if not epoch:
        return "--:--"
    return time.strftime("%H:%M", time.localtime(epoch))


def estimate(shards, now=None, window=RECENT_WINDOW, alive_workers=None) -> dict:
    """多 worker 感知的 ETA。"""
    now = time.time() if now is None else now
    agg = aggregate(shards)
    total, done = agg["total"], agg["completed"]
    remaining = max(total - done, 0)
    elapsed = (now - agg["started_at"]) if agg["started_at"] else 0.0
    durations = agg["durations"]
    recent = durations[-window:]
    avg_unit = (sum(recent) / len(recent)) if recent else None

    if alive_workers is None:
        alive_workers = max(1, sum(
            1 for j in shards
            if now - float(j.get("updated_at", 0)) <= 300.0 and not j.get("closed")))
    workers = max(1, int(alive_workers))

    ts = [float(u["ts"]) for u in agg["units"] if u.get("status") == "ok" and u.get("ts")]
    rate = None
    if len(ts) >= 3:
        win = ts[-window:]
        span = win[-1] - win[0]
        if span >= MIN_RATE_SPAN:
            rate = (len(win) - 1) / span
        # span 过短时 rate=None -> 自动退回 duration/worker 基（更稳健）
    eta_rate = (remaining / rate) if (rate and rate > 0) else None
    eta_dur = (remaining * avg_unit / workers) if (avg_unit is not None) else None

    if remaining == 0:
        eta, basis = 0.0, "done"
    elif eta_rate is not None:
        eta, basis = eta_rate, "throughput"
    elif eta_dur is not None:
        eta, basis = eta_dur, "duration/worker"
    else:
        eta, basis = None, "insufficient-data"

    return {"total": total, "completed": done, "ok": agg["ok"], "failed": agg["failed"],
            "remaining": remaining, "elapsed": elapsed, "avg_unit_seconds": avg_unit,
            "last_unit_seconds": durations[-1] if durations else None,
            "throughput_units_per_min": (rate * 60.0) if rate else None,
            "workers_active": workers, "eta_seconds": eta, "eta_basis": basis,
            "eta_duration_basis_seconds": eta_dur, "eta_throughput_basis_seconds": eta_rate,
            "estimated_finish": (now + eta) if eta is not None else None,
            "last_unit": agg["units"][-1]["id"] if agg["units"] else None,
            "last_unit_worker": agg["units"][-1].get("worker") if agg["units"] else None,
            "now": now}


def render_bar(est: dict, width: int = BAR_WIDTH) -> str:
    total, done = est["total"], est["completed"]
    frac = (done / total) if total else 0.0
    filled = int(round(frac * width))
    bar = "#" * filled + "." * (width - filled)
    avg = est["avg_unit_seconds"]
    thr = est["throughput_units_per_min"]
    return (
        f"[{bar}] {done}/{total} ({100.0 * frac:.1f}%)\n"
        f"Elapsed: {fmt_hms(est['elapsed'])} | Avg/unit: "
        f"{(f'{avg:.0f}s' if avg else '--')} | workers: {est['workers_active']}"
        f" | throughput: {(f'{thr:.2f} unit/min' if thr else '--')}\n"
        f"ETA: {fmt_hms(est['eta_seconds'])} ({est['eta_basis']})"
        f" | Expected finish: {fmt_clock(est['estimated_finish'])}"
    )


def render_line(est: dict) -> str:
    """单行实时进度（runner 每完成一个 unit 打印一次）。"""
    total, done = est["total"], est["completed"]
    pct = 100.0 * done / total if total else 0.0
    avg = est["avg_unit_seconds"]
    return (
        f"[progress] {done}/{total} | {pct:.1f}% | elapsed {fmt_hms(est['elapsed'])} | "
        f"avg/unit {(f'{avg:.0f}s' if avg else '--')} | ETA {fmt_hms(est['eta_seconds'])} "
        f"({est['eta_basis']}) | finish ~{fmt_clock(est['estimated_finish'])} | "
        f"last {est['last_unit']}"
    )


class Progress:
    """单个 worker 进程的进度写入器。"""

    def __init__(self, root, name, unit_ids, worker="w0"):
        self.root = Path(root)
        self.name = name
        self.worker = worker
        self.path = shard_path(self.root, name, worker)
        self.unit_ids = [str(u) for u in unit_ids]
        self.units = []
        self.started_at = time.time()
        self.updated_at = self.started_at
        self.current = None
        self.closed = False
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._write()

    def _payload(self) -> dict:
        return {"name": self.name, "worker": self.worker, "pid": os.getpid(),
                "units_total": len(self.unit_ids), "unit_ids": self.unit_ids,
                "started_at": self.started_at, "updated_at": self.updated_at,
                "current": self.current, "closed": self.closed, "units": self.units}

    def _write(self) -> None:
        tmp = self.path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(self._payload(), indent=2))
        tmp.replace(self.path)

    def unit_start(self, unit_id) -> None:
        self.current = str(unit_id)
        self.updated_at = time.time()
        self._write()

    def heartbeat(self) -> None:
        self.updated_at = time.time()
        self._write()

    def unit_end(self, unit_id, seconds, status="ok", detail="") -> dict:
        self.units.append({"id": str(unit_id), "status": status,
                           "seconds": round(float(seconds), 2),
                           "ts": time.time(), "detail": detail})
        self.current = None
        self.updated_at = time.time()
        self._write()
        shards, alive = load_shards(self.root, self.name)
        est = estimate(shards, alive_workers=alive)
        print(render_line(est), flush=True)
        return est

    def close(self) -> None:
        self.closed = True
        self.updated_at = time.time()
        self._write()
        shards, _ = load_shards(self.root, self.name)
        est = estimate(shards, alive_workers=0)
        print("\n" + render_bar({**est, "workers_active": 0}), flush=True)


def snapshot(root, name=None) -> dict:
    """只读快照（monitor 用）。"""
    shards, alive = load_shards(root, name)
    return {"shards": shards, "n_alive": alive,
            "estimate": estimate(shards, alive_workers=alive)}
