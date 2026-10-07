#!/usr/bin/env python3
"""只读实验监控器（不占用 GPU、不写入实验结果目录、不干预实验进程）。

行为：
  * 每 POLL 秒做一次纯只读检查（ps + 读 progress 分片 / finisher 日志 + 文件 mtime）；
  * 自动发现正在运行的实验进程（scripts/experiment*.py），无需手动指定；
  * 只在三类事件发生时写 event 并退出（由 CodeBuddy 收到通知后转发微信）：
      COMPLETE  实验进程结束且 done >= total（或日志出现 DONE 标记）
      FAILED    实验进程结束但存在 Traceback / done < total
      STALL     进程仍在但连续 STALL_SEC 无任何进度
      ENDED     进程结束但无法判定成功/失败（如实报告）

用法：
  python -u monitor/exp_monitor.py                 # 默认 300s 轮询
  python -u monitor/exp_monitor.py --poll 180 --stall 900
  python -u monitor/exp_monitor.py --wait-launch 21600   # 无实验时最多等待 6h

停止监控：
  kill <watcher_pid>          （PID 见 monitor/state/watcher.pid 与启动日志首行）
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path("/root/autodl-tmp/research-experiments/industrial-anomaly-detection")
MON = ROOT / "monitor"
STATE = MON / "state"
LOGS = MON / "logs"

# 需要排除的常驻 / 非实验进程
EXCLUDE = ("jupyter", "tensorboard", "monitor_progress", "exp_monitor", "exp_monitor.py")


# ---------------------------------------------------------------- 进程发现
def ps_rows() -> list[dict]:
    out = subprocess.run(["ps", "-eo", "pid,etimes,args"], capture_output=True, text=True)
    rows = []
    for line in out.stdout.splitlines()[1:]:
        parts = line.split(None, 2)
        if len(parts) < 3:
            continue
        pid, etimes, args = parts
        if not args.strip():
            continue
        rows.append({"pid": int(pid), "etimes": int(etimes), "args": args})
    return rows


def find_experiments() -> list[dict]:
    """返回正在运行的实验进程组：按脚本名聚合。"""
    groups: dict[str, dict] = {}
    for r in ps_rows():
        a = r["args"]
        if "python" not in a.split()[0]:
            continue
        if any(x in a for x in EXCLUDE):
            continue
        m = re.search(r"(scripts/[A-Za-z0-9_\-]+\.py)", a)
        if not m:
            continue
        name = m.group(1)
        g = groups.setdefault(name, {"script": name, "pids": [], "args": a, "max_etimes": 0})
        g["pids"].append(r["pid"])
        g["max_etimes"] = max(g["max_etimes"], r["etimes"])
    return list(groups.values())


def infer_root(args: str) -> Path | None:
    """从 cmdline 猜实验结果目录；失败则用最近活跃的 results 子目录。"""
    m = re.search(r"--(?:root|results|results-dir|out|output|output-dir|outdir)[=\s]+(\S+)", args)
    if m:
        p = Path(m.group(1))
        if not p.is_absolute():
            p = ROOT / p
        if p.exists():
            return p
    best, best_t = None, 0.0
    for d in (ROOT / "results").iterdir():
        if not d.is_dir() or d.name.startswith("_"):
            continue
        try:
            t = max((f.stat().st_mtime for f in d.rglob("*") if f.is_file()), default=0.0)
        except OSError:
            continue
        if t > best_t:
            best, best_t = d, t
    if best and time.time() - best_t < 3600:
        return best
    return None


# ---------------------------------------------------------------- 进度读取
def newest_mtime(root: Path) -> float:
    try:
        return max((f.stat().st_mtime for f in root.rglob("*") if f.is_file()), default=0.0)
    except OSError:
        return 0.0


def read_progress(root: Path) -> dict:
    """优先 progress 分片（9A 风格）→ progress.json → finisher.log（8B 风格）。"""
    info = {"source": "none", "done": None, "total": None, "failed": 0}
    if root is None:
        return info

    pdir = root / "progress"
    if pdir.is_dir():
        total = done = failed = 0
        n_shards = 0
        for p in sorted(pdir.glob("*.json")):
            try:
                j = json.loads(p.read_text())
            except Exception:
                continue
            n_shards += 1
            total += int(j.get("units_total", 0) or 0)
            units = j.get("units", []) or []
            done += len(units)
            failed += sum(1 for u in units if u.get("status") == "failed")
        if n_shards:
            info.update(source="progress-shards", done=done, total=total, failed=failed)
            return info

    pj = root / "progress.json"
    if pj.exists():
        try:
            j = json.loads(pj.read_text())
            n = len(j.get("per_worker", []) or []) or 1
            info.update(source="progress.json", done=int(j.get("completed", 0)) * n,
                        total=int(j.get("total", 0)) * n, failed=int(j.get("failed", 0)))
            return info
        except Exception:
            pass

    logs = sorted((root / "logs").glob("*.log")) if (root / "logs").is_dir() else []
    fin = [p for p in logs if "finisher" in p.name]
    done = None
    if fin:
        try:
            txt = fin[0].read_text(errors="ignore")
            hits = re.findall(r"done=(\d+)", txt)
            if hits:
                done = int(hits[-1])
        except OSError:
            pass
    total = None
    up = root / "probe" / "units_plan.json"
    if not up.exists():
        cand = list(root.rglob("units_plan.json"))
        up = cand[0] if cand else None
    if up and up.exists():
        try:
            total = int(json.loads(up.read_text()).get("n_units"))
        except Exception:
            total = None
    if done is not None:
        info.update(source="finisher.log", done=done, total=total)
    return info


def scan_errors(root: Path, tail_bytes: int = 400_000) -> list[str]:
    errs = []
    if root is None:
        return errs
    for p in sorted((root / "logs").glob("*.log")) if (root / "logs").is_dir() else []:
        try:
            data = p.read_bytes()[-tail_bytes:].decode("utf-8", "ignore")
        except OSError:
            continue
        for m in re.finditer(r"(Traceback \(most recent call last\):[\s\S]{0,600})", data):
            errs.append(f"{p.name}: " + m.group(1).strip().splitlines()[-1][:300])
        for m in re.finditer(r"^\s*(?:\[ERROR\]|Error:|ERROR:)\s*(.{0,200})$", data, re.M):
            errs.append(f"{p.name}: {m.group(1).strip()[:200]}")
    return errs[-10:]


def finished_rounds() -> dict:
    """扫描所有 results/*/ 下已完成的 round。

    目的：有些 round 总时长 < 轮询间隔（如 9A round1 只跑 3 分钟），
    进程可能在两次 tick 之间就退出，光靠 ps 会漏报。这里用落盘的
    progress.json / raw/<round>/ 结果目录补齐检测。
    """
    out: dict[str, dict] = {}
    base = ROOT / "results"
    if not base.is_dir():
        return out
    for root in sorted(base.iterdir()):
        if not root.is_dir() or root.name.startswith("_"):
            continue
        rounds: dict[str, dict] = {}
        pj = root / "progress.json"
        if pj.exists():
            try:
                j = json.loads(pj.read_text())
            except Exception:
                j = {}
            if j.get("round"):
                rounds[j["round"]] = {
                    "done": j.get("completed"), "total": j.get("total"),
                    "remaining": j.get("remaining"), "failed": j.get("failed", 0),
                    "elapsed": j.get("elapsed_seconds"), "last": j.get("last_update"),
                }
        rawd = root / "raw"
        if rawd.is_dir():
            for rd in sorted(rawd.iterdir()):
                if not rd.is_dir():
                    continue
                st = rounds.setdefault(rd.name, {})
                st["n_units"] = len(list(rd.rglob("info.json")))
        for r, st in rounds.items():
            fin = (st.get("remaining") == 0) or (
                st.get("done") is not None and st.get("total") and st["done"] >= st["total"])
            st.update(root=str(root), round=r, finished=bool(fin))
            out[f"{root.name}::{r}"] = st
    return out


def gpu_line() -> str:
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total",
                              "--format=csv,noheader"], capture_output=True, text=True, timeout=20)
        return out.stdout.strip().replace("\n", " | ")
    except Exception as e:
        return f"nvidia-smi failed: {e}"


# ---------------------------------------------------------------- 主循环
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--poll", type=int, default=300)
    ap.add_argument("--stall", type=int, default=900)          # 15 分钟无进度
    ap.add_argument("--wait-launch", type=int, default=21600)  # 无实验时最多等 6h
    ap.add_argument("--max-total", type=int, default=172800)   # 监控总时长上限 48h
    args = ap.parse_args()

    STATE.mkdir(parents=True, exist_ok=True)
    LOGS.mkdir(parents=True, exist_ok=True)
    (STATE / "watcher.pid").write_text(str(os.getpid()))
    logf = LOGS / f"watcher_{time.strftime('%Y%m%d_%H%M%S')}.log"
    evf = MON / "events.jsonl"
    print(f"[watcher] pid={os.getpid()} poll={args.poll}s stall={args.stall}s log={logf}", flush=True)

    def emit(event: str, payload: dict) -> None:
        rec = {"ts": time.strftime("%Y-%m-%d %H:%M:%S"), "epoch": time.time(),
               "event": event, **payload}
        with open(evf, "a") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print("[EVENT] " + json.dumps(rec, ensure_ascii=False), flush=True)

    t_start = time.time()
    cur = None          # 当前跟踪的实验
    last_done = None
    last_change_ts = None
    idle_since = time.time()
    # 启动前就已完成的 round 不告警，只关心之后新完成的
    seen_rounds = {k for k, v in finished_rounds().items() if v["finished"]}
    print(f"[watcher] pre-finished rounds ignored: {sorted(seen_rounds)}", flush=True)

    while True:
        now = time.time()
        if now - t_start > args.max_total:
            emit("TIMEOUT", {"note": "watcher reached max-total, stopped"})
            return 0

        # 1) 先扫已完成 round（补齐短任务漏报）
        for key, st in finished_rounds().items():
            if st["finished"] and key not in seen_rounds:
                seen_rounds.add(key)
                if cur is not None and st["round"] == cur.get("round"):
                    cur["emitted"] = True
                emit("COMPLETE", {
                    "script": "round-scan", "root": st["root"], "round": st["round"],
                    "done": st.get("total") if st.get("done") is None else st.get("done"),
                    "total": st.get("total"), "failed": st.get("failed", 0),
                    "elapsed_s": st.get("elapsed"), "last_update": st.get("last"),
                    "progress_source": "progress.json/raw", "gpu": gpu_line()})

        exps = find_experiments()

        if exps and cur is None:
            e = exps[0]
            root = infer_root(e["args"])
            cur = {"script": e["script"], "root": root, "start_wall": time.time(),
                   "pids_first": e["pids"][:]}
            pr = read_progress(root)
            last_done = pr["done"]
            last_change_ts = now
            print(f"[watcher] attached: {e['script']} pids={e['pids']} root={root} "
                  f"progress={pr}", flush=True)

        if cur is None:
            if now - idle_since > args.wait_launch:
                emit("NO_EXPERIMENT", {"note": f"no experiment process within {args.wait_launch}s"})
                return 0
            print(f"[watcher] waiting for experiment... ({int(now - idle_since)}s)", flush=True)
            time.sleep(args.poll)
            continue

        alive = [p for p in ps_rows() if p["pid"] in cur["pids_first"]]
        # 允许后续新增的同脚本 worker
        if exps:
            for e in exps:
                if e["script"] == cur["script"]:
                    for pid in e["pids"]:
                        if pid not in cur["pids_first"]:
                            cur["pids_first"].append(pid)
            alive = [p for p in ps_rows() if p["pid"] in cur["pids_first"]]

        root = cur["root"] or infer_root(cur["script"])
        pr = read_progress(root)
        try:
            cur["round"] = json.loads((root / "progress.json").read_text()).get("round")
        except Exception:
            pass
        act = newest_mtime(root) if root else 0.0
        done, total = pr["done"], pr["total"]

        if done is not None and done != last_done:
            last_done = done
            last_change_ts = now
        if act and act > cur.get("last_act", 0):
            cur["last_act"] = act
            if done is None:
                last_change_ts = now

        snap = {"script": cur["script"], "root": str(root) if root else None,
                "alive": len(alive), "pids": cur["pids_first"], "done": done,
                "total": total, "failed": pr.get("failed", 0),
                "progress_source": pr["source"],
                "elapsed_s": int(now - cur["start_wall"]),
                "since_change_s": int(now - (last_change_ts or now)),
                "gpu": gpu_line()}
        print("[tick] " + json.dumps(snap, ensure_ascii=False), flush=True)

        if not alive:
            if cur.get("emitted"):
                print("[watcher] process exited; round already reported, stop.", flush=True)
                return 0
            errs = scan_errors(root)
            ok = (total is not None and done is not None and done >= total) or bool(
                re.search(r"DONE|finished", " ".join(
                    p.name for p in ((root / "logs").glob("*.log") if root and (root / "logs").is_dir() else [])))
            )
            if errs and not (total is not None and done is not None and done >= total):
                emit("FAILED", {**snap, "errors": errs})
            elif ok:
                emit("COMPLETE", snap)
            else:
                emit("ENDED", {**snap, "errors": errs})
            return 0

        if now - (last_change_ts or now) >= args.stall:
            emit("STALL", {**snap, "stall_s": int(now - (last_change_ts or now))})
            return 0

        time.sleep(args.poll)


if __name__ == "__main__":
    sys.exit(main())
