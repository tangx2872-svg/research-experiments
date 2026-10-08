#!/bin/bash
# 8B Phase B 收尾：等待 420 units 完成（或 worker 全部退出）→ probe 分析 → 报告
# 用途：无人值守（interactive 会话可能断开），不改变任何科学协议。
cd /root/autodl-tmp/research-experiments/industrial-anomaly-detection
LOG=results/experiment8b/logs/finisher.log
echo "[finisher] start $(date)" >> $LOG
while true; do
  n=$(python - <<'PY'
from pathlib import Path
import json
ok=0
for p in Path("results/experiment8b/probe/raw").rglob("info.json"):
    try:
        if json.loads(p.read_text()).get("status")=="OK": ok+=1
    except Exception: pass
print(ok)
PY
)
  alive=$(pgrep -fc experiment8b_probe_runner.py || true)
  echo "[finisher] $(date) done=$n alive=$alive" >> $LOG
  if [ "$n" -ge 420 ] || [ "$alive" -eq 0 ]; then break; fi
  sleep 60
done
echo "[finisher] probe finished $(date) done=$n" >> $LOG
python -u scripts/experiment8b_probe_analyze.py >> $LOG 2>&1
python -u scripts/experiment8b_report.py >> $LOG 2>&1
echo "[finisher] DONE $(date)" >> $LOG
