"""Experiment 10 — 批处理编排器（无人值守）。

职责：
  * 把一批 unit 轮询分配到 N 个 worker 进程（各自独立进程 = 与 9B-R 相同的并行方式）
  * 已完成 unit 自动跳过（h.unit_done + 计划表双重检查）
  * 单 unit 失败 -> 整个 worker 组的未完成 unit 重排一次重试（最多 1 次）
  * 连续 3 个 unit 失败 -> 停止新任务 + 写 ERROR_REPORT.md（不无限失败循环）
  * 持续更新 progress.json / progress.log

  python -u scripts/experiment10_batch.py --stage s0s1 --backend 9b --mode corrected189 \
      --workers 5 --units "bottle:0:P10_ORIG_a000,..."
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "results" / "experiment_10_preservation_recovery"
sys.path.insert(0, str(ROOT / "scripts"))
import experiment10_progress as prog  # noqa: E402


def unit_dir(backend: str, stage: str, unit: str) -> Path:
    cat, seed, cfg = unit.split(":")
    if backend == "7ao":
        return EXP / "raw" / stage / cat / f"seed_{seed}" / f"config_{cfg}"
    return EXP / "raw" / stage / cat / f"seed_{seed}" / f"config_{cfg}"


def is_done(backend: str, stage: str, unit: str) -> bool:
    ip = unit_dir(backend, stage, unit) / "info.json"
    if not ip.exists():
        return False
    try:
        return str(json.loads(ip.read_text()).get("status")) == "OK"
    except Exception:
        return False


_HANDLES: list = []   # 必须持有引用：否则 file 对象被 GC 关闭，子进程日志丢失（Exp10 bug-1）


def launch(group: list, backend: str, mode: str, stage: str, wid: int) -> subprocess.Popen:
    log = EXP / "logs" / f"{stage}_{backend}_w{wid}.log"   # 含 backend，避免同 stage 不同 backend 撞车（Exp10 bug-2）
    log.parent.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, "-u", str(ROOT / "scripts" / "experiment10_runner.py"),
           "--backend", backend, "--round", stage, "--mode", mode,
           "--worker-tag", f"w{wid}", "--units", ",".join(group)]
    fh = open(log, "ab")
    _HANDLES.append(fh)
    return subprocess.Popen(cmd, stdout=fh, stderr=subprocess.STDOUT, cwd=str(ROOT),
                            start_new_session=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True)
    ap.add_argument("--backend", required=True, choices=["9b", "7ao"])
    ap.add_argument("--mode", default="corrected189")
    ap.add_argument("--workers", type=int, default=5)
    ap.add_argument("--units", required=True)
    a = ap.parse_args()

    units = [u.strip() for u in a.units.split(",") if u.strip()]
    pending = [u for u in units if not is_done(a.backend, a.stage, u)]
    print(f"[batch:{a.stage}] backend={a.backend} mode={a.mode} workers={a.workers} "
          f"units={len(units)} pending={len(pending)}", flush=True)
    (EXP / "logs" / f"{a.stage}_batch_start.txt").write_text(time.strftime("%F %T"))
    fails_in_row, total_fails = 0, 0
    attempt = 0
    while pending and attempt < 2 and fails_in_row < 3:
        attempt += 1
        # 按 seed 分组再轮询：runner 要求单 worker 单 seed（Exp10 bug-3）
        groups = []
        for _seed in sorted({u.split(":")[1] for u in pending}):
            _us = [u for u in pending if u.split(":")[1] == _seed]
            groups += [_us[i::a.workers] for i in range(min(a.workers, len(_us)))]
        groups = [g for g in groups if g]
        print(f"[batch:{a.stage}] attempt {attempt}: launching {len(groups)} workers "
              f"for {len(pending)} units", flush=True)
        procs = [launch(g, a.backend, a.mode, a.stage, i) for i, g in enumerate(groups)]
        while any(p.poll() is None for p in procs):
            print(prog.render(), flush=True)
            time.sleep(20)
        failed = []
        for u in pending:
            ip = unit_dir(a.backend, a.stage, u) / "info.json"
            ok = False
            if ip.exists():
                try:
                    ok = str(json.loads(ip.read_text()).get("status")) == "OK"
                except Exception:
                    ok = False
            if not ok:
                failed.append(u)
        newly_failed = len(failed)
        total_fails += newly_failed
        fails_in_row = newly_failed if newly_failed else 0
        print(f"[batch:{a.stage}] attempt {attempt} done: failed={failed}", flush=True)
        pending = failed
    if pending and fails_in_row >= 3:
        (EXP / "ERROR_REPORT.md").write_text(
            f"# ERROR_REPORT.md — Experiment 10\n\nstage `{a.stage}` 出现连续 ≥3 个 unit 失败，"
            f"已停止新 GPU 任务。\n\n未完成 unit:\n" + "\n".join(f"- {u}" for u in pending) + "\n")
        print(f"[batch:{a.stage}] HARD STOP: consecutive failures >=3 -> ERROR_REPORT.md", flush=True)
    print(prog.render(), flush=True)
    print(f"[batch:{a.stage}] FINISHED pending={len(pending)} total_fail_events={total_fails}",
          flush=True)


if __name__ == "__main__":
    main()
