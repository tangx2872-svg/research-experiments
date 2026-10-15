#!/usr/bin/env bash
# E22 overnight driver (unattended): preflight -> reproduction gate -> 21-cell matrix.
# Frozen protocol: results/e22/protocol.md
# Resume-safe: the matrix skips cells already written as status=OK, source=NEW.
# The reproduction gate runs ONCE per driver invocation and blocks the matrix on failure.
set -u
cd /root/autodl-tmp/research-experiments/industrial-anomaly-detection

PY=/root/miniconda3/bin/python
OUT=results/e22
mkdir -p "$OUT/logs"
RUN_TAG="${1:-manual}"
TS=$(date +%Y%m%d_%H%M%S)
LOG="$OUT/logs/overnight_${TS}_${RUN_TAG}.log"
echo "$LOG" > "$OUT/logs/latest_run.txt"
echo "$$" > "$OUT/logs/driver.pid"

{
  echo "[E22 DRIVER] start $(date '+%F %T %Z') tag=${RUN_TAG} pid=$$ log=${LOG}"
  echo "[E22 DRIVER] protocol: results/e22/protocol.md | method frozen R1-ILA BETA=0.1"
  echo "[E22 DRIVER] disk: $(df -h /root/autodl-tmp | tail -1)"

  echo "[E22 DRIVER] --- stage: preflight ---"
  $PY -u scripts/e22_full_category_validation.py --preflight || {
    echo "[E22 DRIVER] PREFLIGHT FAILED -> stop"; exit 3; }

  if [ ! -f "$OUT/repro_gate.json" ]; then
    echo "[E22 DRIVER] --- stage: reproduction gate ---"
    $PY -u scripts/e22_full_category_validation.py --repro
    RC=$?
    if [ $RC -ne 0 ]; then
      echo "[E22 DRIVER] REPRODUCTION GATE FAILED rc=$RC -> matrix NOT started"
      exit $RC
    fi
  else
    echo "[E22 DRIVER] reproduction gate already recorded ($OUT/repro_gate.json) -> skip"
  fi

  echo "[E22 DRIVER] --- stage: matrix (7 new categories x 3 seeds = 21 cells) ---"
  $PY -u scripts/e22_full_category_validation.py --matrix
  RC=$?
  echo "[E22 DRIVER] matrix rc=$RC"
  echo "[E22 DRIVER] end $(date '+%F %T %Z')"
  exit $RC
} 2>&1 | tee -a "$LOG"
