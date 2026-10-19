#!/usr/bin/env bash
# Overnight stage 2: wait for stage 1 (strength round 1 + analysis), then
#   (a) multi-seed stability of the top-2 strengths (seeds 1,2; seed0 reused)
#   (b) perturbation-type ablation at S15 (seed 0)
#   (c) re-run the E24 analysis with all stages present
set -u
ROOT=/root/autodl-tmp/research-experiments/industrial-anomaly-detection
cd "$ROOT" || exit 1
LOGD=results/overnight_20261018
mkdir -p "$LOGD"
echo $$ > "$LOGD/night_stage2.pid"
echo "[stage2] start $(date -Is) pid=$$"

# wait for stage 1 driver
while pgrep -f "scripts/night_run_20261018.sh" > /dev/null; do
  sleep 30
done
echo "[stage2] stage1 finished $(date -Is)"

TOP2=$(python - <<'PY'
import json, sys
p = "results/e24/final_strength_analysis.json"
try:
    d = json.load(open(p))
    t = d.get("top2_by_stability") or []
    print(",".join(t))
except Exception as e:
    print("")
PY
)
if [ -z "$TOP2" ]; then
  echo "[stage2] top2 unavailable -> STOP"
  exit 1
fi
echo "[stage2] top2 = $TOP2"

for attempt in 1 2; do
  python scripts/e24_runner.py --stage stability --strengths "$TOP2" \
    --categories Bird,Car,Cube --seeds 1,2 >> "$LOGD/e24_stability.log" 2>&1
  rc=$?
  echo "[stage2] stability attempt $attempt exit=$rc $(date -Is)"
  [ "$rc" -eq 0 ] && break
  sleep 30
done

for attempt in 1 2; do
  python scripts/e24_runner.py --stage type --single-strength S15 \
    --types A0,A1,A2,A3,A4,A5,A6 --categories Bird,Car,Cube --seeds 0 \
    >> "$LOGD/e24_type.log" 2>&1
  rc=$?
  echo "[stage2] type attempt $attempt exit=$rc $(date -Is)"
  [ "$rc" -eq 0 ] && break
  sleep 30
done

python scripts/e24_analysis.py >> "$LOGD/e24_analysis.log" 2>&1
echo "[stage2] analysis exit=$? $(date -Is)"
echo "[stage2] done $(date -Is)"
