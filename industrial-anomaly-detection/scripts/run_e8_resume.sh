#!/bin/bash
# E8 resume: Stage A (Car/Cube) -> gate all -> Stage C core matrix -> analysis.
# Every step is idempotent / resume-aware (already-built caches and DONE units are skipped).
set -u
cd /root/autodl-tmp/research-experiments/industrial-anomaly-detection
export HF_ENDPOINT=https://hf-mirror.com
mkdir -p results/e8/logs results/e8/raw
echo "[E8 RESUME] start $(date '+%F %T')"

echo "===== 1/4 Stage A resume: Car + Cube (streaming selective extraction) ====="
python -u scripts/e8_fetch.py --categories Car,Cube >> results/e8/logs/stageA_prep.log 2>&1
echo "[E8 RESUME] stageA exit=$?"

echo "===== 2/4 gate all categories (Bird/Car/Cube) -> prep.json ====="
python -u scripts/e8_fetch.py --gate-only --categories Bird,Car,Cube >> results/e8/logs/stageA_prep.log 2>&1
echo "[E8 RESUME] gate exit=$?"

echo "===== 3/4 Stage C core matrix (3 cats x 3 methods x 3 seeds; resume-aware) ====="
python -u scripts/e8_runner.py --stage c --categories Bird,Car,Cube >> results/e8/logs/stageC.log 2>&1
echo "[E8 RESUME] stageC exit=$?"

echo "===== 4/4 analysis + figures + verdict ====="
python -u scripts/e8_analysis.py >> results/e8/logs/analysis.log 2>&1
echo "[E8 RESUME] analysis exit=$?"

echo "[E8 RESUME] done $(date '+%F %T')"
