#!/usr/bin/env bash
# E25: wait for the downloader (results/e25/download.pid) to exit, then run the
# integrity check and leave a marker. Read-only w.r.t. the dataset itself.
set -u
cd "$(dirname "$0")/.." || exit 1

PID=$(cat results/e25/download.pid)
LOG=results/e25/logs/verify.log
: > "$LOG"

while kill -0 "$PID" 2>/dev/null; do sleep 30; done

echo "[watch] downloader pid=$PID exited at $(date -Iseconds)" >> "$LOG"
python scripts/e25_verify_download.py >> "$LOG" 2>&1
rc=$?
echo "[watch] verify_exit=$rc at $(date -Iseconds)" >> "$LOG"
echo "$(date -Iseconds) rc=$rc" > results/e25/download_done.flag
