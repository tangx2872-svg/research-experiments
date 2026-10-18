#!/usr/bin/env bash
# E25 driver: run PCB then MetalParts, each in its own process, resumable, logged.
# Usage: setsid nohup bash scripts/e25_run_all.sh >> results/e25/logs/driver.log 2>&1 &
set -u
ROOT=/root/autodl-tmp/research-experiments/industrial-anomaly-detection
cd "$ROOT" || exit 1
LOGD=results/e25/logs
mkdir -p "$LOGD"
echo $$ > results/e25/run_all.pid
echo "[driver] start $(date -Is) pid=$$"

dom_done() {
  python - "$1" <<'PY'
import json, os, sys
p = "results/e25/partial.json"
dom = sys.argv[1]
if not os.path.exists(p):
    print(0); raise SystemExit
try:
    d = json.load(open(p))
except Exception:
    print(0); raise SystemExit
print(1 if dom in d.get("domains", {}) else 0)
PY
}

for dom in PCB MetalParts; do
  if [ "$(dom_done "$dom")" = "1" ]; then
    echo "[driver] SKIP $dom (already in partial.json)"
    continue
  fi
  for attempt in 1 2 3; do
    echo "[driver] $(date -Is) RUN $dom attempt $attempt -> $LOGD/e25_${dom}.log"
    python scripts/e25_robustad_eval.py --domain "$dom" >> "$LOGD/e25_${dom}.log" 2>&1
    rc=$?
    echo "[driver] $(date -Is) $dom attempt $attempt exit=$rc"
    if [ "$rc" -eq 0 ] && [ "$(dom_done "$dom")" = "1" ]; then
      break
    fi
    sleep 20
  done
done
echo "[driver] done $(date -Is)"
