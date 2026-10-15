#!/usr/bin/env python
"""E22 overnight watchdog (lightweight, read-mostly).

Checks every 120 s:
  - runner PID (results/e22/status.json:pid) alive
  - status heartbeat age (results/e22/status.json:last_heartbeat)
  - driver log mtime growth
  - disk free (> 8 GB required)
  - GPU utilisation / memory

Auto-recovery policy (protocol section 5):
  - if the runner is gone, the matrix is incomplete, disk is safe and no
    ENGINEERING_BLOCKED marker exists -> relaunch the driver (resume), at most 2 times
    per watchdog lifetime.
  - if a single cell failed twice -> the runner itself writes stage=ENGINEERING_BLOCKED;
    the watchdog then does NOT restart anything.
  - if disk free < 8 GB -> write results/e22/STOP_disk.txt and do not restart.
  - never kills other experiments; never retries on scientific grounds.

State: results/e22/watchdog_state.json ; log: results/e22/logs/watchdog.log
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "e22"
STATUS = OUT / "status.json"
STATE = OUT / "watchdog_state.json"
LATEST_RUN = OUT / "logs" / "latest_run.txt"
STOP_DISK = OUT / "STOP_disk.txt"
TOTAL_CELLS = 21
MAX_RECOVERIES = 2
POLL_S = 120
HEARTBEAT_STALE_S = 900      # 15 min without heartbeat = suspicious
DISK_MIN_GB = 8.0

sys.path.insert(0, str(ROOT / "scripts"))
import e8_common as E8  # noqa: E402


def now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def log(msg: str) -> None:
    print(f"[E22 watchdog {now()}] {msg}", flush=True)


def pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except (ProcessLookupError, PermissionError):
        return False
    except Exception:  # noqa: BLE001
        return False


def read_json(p: Path):
    try:
        return json.loads(p.read_text())
    except Exception:  # noqa: BLE001
        return None


def any_e22_alive() -> bool:
    """Robust liveness: runner/driver process patterns + pid files + status pid + log mtime."""
    for pat in ("e22_full_category_validation.py", "e22_overnight_run.sh"):
        pg = subprocess.run(["pgrep", "-f", pat], capture_output=True, text=True)
        if pg.returncode == 0 and pg.stdout.strip():
            return True
    for f in (OUT / "logs" / "driver.pid",):
        if f.exists():
            try:
                if pid_alive(int(f.read_text().strip())):
                    return True
            except Exception:  # noqa: BLE001
                pass
    st = read_json(STATUS) or {}
    pid = st.get("pid")
    if pid:
        try:
            if pid_alive(int(pid)):
                return True
        except Exception:  # noqa: BLE001
            pass
    logs = sorted((OUT / "logs").glob("*.log"), key=lambda p: p.stat().st_mtime, reverse=True)
    if logs and (time.time() - logs[0].stat().st_mtime) < 300:
        return True
    return False


def n_new_cells() -> int:
    d = OUT / "cells"
    if not d.is_dir():
        return 0
    n = 0
    for p in sorted(d.glob("*.json")):
        c = read_json(p)
        if c and c.get("source") == "NEW" and c.get("status") == "OK":
            n += 1
    return n


def launch_driver(tag: str) -> int:
    cmd = ["setsid", "nohup", "bash", "scripts/e22_overnight_run.sh", tag]
    with open(OUT / "logs" / "driver_stdout.log", "ab") as fh:
        p = subprocess.Popen(cmd, stdout=fh, stderr=fh, cwd=str(ROOT), start_new_session=True)
    log(f"launched driver pid={p.pid} tag={tag}")
    return p.pid


def main() -> int:
    state = read_json(STATE) or {"recoveries": 0, "events": []}
    log(f"watchdog start pid={os.getpid()} | recoveries so far={state['recoveries']}")
    grace = int(os.environ.get("E22_WATCHDOG_GRACE_S", "90"))
    log(f"startup grace {grace}s before first liveness check")
    time.sleep(grace)
    while True:
        st = read_json(STATUS) or {}
        done = n_new_cells()
        if done >= TOTAL_CELLS:
            log(f"matrix complete ({done}/{TOTAL_CELLS}) -> watchdog exits")
            state["events"].append({"ts": now(), "event": "complete", "done": done})
            STATE.write_text(json.dumps(state, indent=1))
            return 0

        # never restart past an explicit stop marker
        for marker in (OUT / "STOP_repro_fail.txt",):
            if marker.exists():
                log(f"{marker.name} present -> watchdog stops (no auto restart)")
                state["events"].append({"ts": now(), "event": "stop_marker",
                                        "marker": marker.name})
                STATE.write_text(json.dumps(state, indent=1))
                return 7
        rg = read_json(OUT / "repro_gate.json")
        if rg and not rg.get("all_pass", False):
            log("repro_gate.json all_pass=False -> watchdog stops (no auto restart)")
            state["events"].append({"ts": now(), "event": "repro_gate_failed"})
            STATE.write_text(json.dumps(state, indent=1))
            return 8

        disk = E8.free_gb()
        if disk < DISK_MIN_GB:
            STOP_DISK.write_text(f"E22 STOP: disk free {disk:.2f} GB < {DISK_MIN_GB} GB at {now()}\n")
            log(f"!! disk {disk:.2f} GB < {DISK_MIN_GB} GB -> STOP, no restart")
            state["events"].append({"ts": now(), "event": "disk_stop", "disk_gb": disk})
            STATE.write_text(json.dumps(state, indent=1))
            return 4

        stage = st.get("stage")
        pid = st.get("pid")
        alive = pid_alive(int(pid)) if pid else False
        hb_age = None
        if st.get("last_heartbeat"):
            try:
                hb_age = time.time() - datetime.fromisoformat(st["last_heartbeat"]).timestamp()
            except Exception:  # noqa: BLE001
                hb_age = None

        # is ANY e22 process alive? (repro gate, matrix, or the driver shell itself)
        matrix_running = any_e22_alive()

        if stage == "ENGINEERING_BLOCKED":
            log("stage=ENGINEERING_BLOCKED -> watchdog stops (no auto restart)")
            state["events"].append({"ts": now(), "event": "engineering_blocked"})
            STATE.write_text(json.dumps(state, indent=1))
            return 5

        if not matrix_running and not alive:
            if state["recoveries"] >= MAX_RECOVERIES:
                log(f"runner gone and recoveries={state['recoveries']} >= {MAX_RECOVERIES} "
                    f"-> STOP (ENGINEERING BLOCKED)")
                state["events"].append({"ts": now(), "event": "max_recoveries", "done": done})
                STATE.write_text(json.dumps(state, indent=1))
                return 6
            state["recoveries"] += 1
            state["events"].append({"ts": now(), "event": "recovery", "n": state["recoveries"],
                                    "done_cells": done, "stage": stage, "heartbeat_age_s": hb_age})
            STATE.write_text(json.dumps(state, indent=1))
            launch_driver(f"recovery{state['recoveries']}")
        elif matrix_running and hb_age is not None and hb_age > HEARTBEAT_STALE_S:
            log(f"runner alive but heartbeat stale ({hb_age:.0f}s) - monitoring only")
        else:
            log(f"ok: stage={stage} done={done}/{TOTAL_CELLS} alive={matrix_running} "
                f"hb_age={None if hb_age is None else round(hb_age)}s disk={disk:.1f}GB")
        time.sleep(POLL_S)


if __name__ == "__main__":
    raise SystemExit(main())
