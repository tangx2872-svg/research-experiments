#!/usr/bin/env bash
# Overnight driver 2026-10-18: wait for E25 multiseed -> E25 multiseed analysis
# -> E24 strength round 1 (seed 0) -> E24 analysis. Resumable (cell-level skip).
set -u
ROOT=/root/autodl-tmp/research-experiments/industrial-anomaly-detection
cd "$ROOT" || exit 1
LOGD=results/overnight_20261018
mkdir -p "$LOGD"
echo $$ > "$LOGD/night.pid"
echo "[night] start $(date -Is) pid=$$"

# 1) wait for the E25 multi-seed process (started separately) to finish
while pgrep -f "e25_robustad_eval.py --domain PCB,MetalParts" > /dev/null; do
  sleep 30
done
echo "[night] E25 finished $(date -Is)"

# 2) E25 multi-seed summary (CPU)
python scripts/e25_multiseed_analysis.py >> "$LOGD/e25_multiseed_analysis.log" 2>&1
echo "[night] e25 multiseed analysis exit=$? $(date -Is)"

# 3) E24 strength round 1 (18 cells, seed 0), max 2 retries
for attempt in 1 2; do
  python scripts/e24_runner.py --stage strength \
    --strengths S05,S10,S15,S20,S25,S30 --categories Bird,Car,Cube --seeds 0 \
    >> "$LOGD/e24_strength.log" 2>&1
  rc=$?
  echo "[night] e24 strength attempt $attempt exit=$rc $(date -Is)"
  [ "$rc" -eq 0 ] && break
  sleep 30
done

# 4) E24 analysis (round 1 ranking + top2 selection; CPU)
python scripts/e24_analysis.py >> "$LOGD/e24_analysis.log" 2>&1
echo "[night] e24 analysis exit=$? $(date -Is)"
echo "[night] stage-1 done $(date -Is)"
