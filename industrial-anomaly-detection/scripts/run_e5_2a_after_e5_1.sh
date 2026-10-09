#!/usr/bin/env bash
# run_e5_2a_after_e5_1.sh — sequential follow-up dispatcher for E5-1 -> E5-2A.
#
# Implements the E5-2A protocol sections 1, 2, 13, 14 and 21:
#   wait for E5-1 completion -> safety gate (7 checks) -> mechanically read E5-1 decisions
#   -> dispatch on the number of GO survivors.
#
#   CASE A (>=2 GO) : E5-2A multi-view survivor validation (top-2 survivors)
#   CASE B (1 GO)   : E5-2A single-survivor validation
#   CASE C (0 GO)   : GPU METHOD WORK IS STOPPED. No E5-2A is launched. Exit BLOCKED.
#
# SAFETY: this dispatcher will NOT start any GPU work unless --allow-gpu is passed explicitly.
#         That is deliberate: a machine reading of E5-1 must not silently authorise GPU compute.
#
# NOTE (2026-10-09): this script was authored AFTER E5-1 had already completed. It was therefore
#         never launched as a watcher. It is retained as the reusable, auditable dispatcher for
#         future rounds. E5-1 finished with 0 GO -> CASE C -> E5-FAILURE-AUDIT (CPU-only).
#
# Usage:
#   scripts/run_e5_2a_after_e5_1.sh                 # gate + dispatch decision only (dry run)
#   scripts/run_e5_2a_after_e5_1.sh --allow-gpu     # also launch E5-2A when CASE A/B
#   scripts/run_e5_2a_after_e5_1.sh --wait          # wait for E5-1 to finish first (60 s poll)

set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PROGRESS="results/e5_1/progress.json"
LOGDIR="results/e5_2a/logs"
ALLOW_GPU=0
WAIT=0
for a in "$@"; do
  case "$a" in
    --allow-gpu) ALLOW_GPU=1 ;;
    --wait)      WAIT=1 ;;
    *) echo "unknown arg: $a" >&2; exit 2 ;;
  esac
done
mkdir -p "$LOGDIR"
LOG="$LOGDIR/dispatcher.log"
say() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

# ---------------------------------------------------------------- section 21: wait for E5-1
if [ "$WAIT" -eq 1 ]; then
  say "waiting for E5-1 to complete (polling every 60 s; low frequency on purpose)"
  while pgrep -f "e5_1_runner.py" >/dev/null 2>&1; do sleep 60; done
  say "no e5_1_runner process remains"
fi

# ---------------------------------------------------------------- section 14: safety gate
GATE_FAIL=0
check() { # name, command -> pass/fail
  if eval "$2" >/dev/null 2>&1; then say "GATE PASS  $1"; else say "GATE FAIL  $1"; GATE_FAIL=1; fi
}
check "1 E5-1 result complete (progress.json)"      "[ -s '$PROGRESS' ]"
check "2 E5-1 README exists"                        "[ -s results/e5_1/README.md ]"
check "3 E5-1 decisions exist (>=4 raw json)"       "[ \$(ls results/e5_1/raw/*.json 2>/dev/null | wc -l) -ge 4 ]"
check "4 no active E5-1 GPU process"                "! pgrep -f 'e5_1_runner.py|e5_1_retinex_prep.py'"
check "5 git state known (clean tree except known files)" "git rev-parse HEAD"
check "6 no leakage flagged"                        "python -c \"import json,sys; d=json.load(open('$PROGRESS')); sys.exit(0 if all(d.get('sanity',{}).values()) else 1)\""

if [ "$GATE_FAIL" -ne 0 ]; then
  say "RESULT = BLOCKED (safety gate failed). No E5-2A started. Human action required."
  exit 3
fi

# ---------------------------------------------------------------- section 1/2: mechanical branch
BRANCH=$(python - <<'PY'
import json, pathlib
raw = pathlib.Path("results/e5_1/raw")
dec = {}
for p in sorted(raw.glob("*.json")):
    j = json.loads(p.read_text())
    dec[j["candidate"]] = j["decision"]["decision"]
go = sorted([k for k, v in dec.items() if v == "GO"])
print(f"{len(go)}|{','.join(go)}|{json.dumps(dec)}")
PY
)
NGO="${BRANCH%%|*}"; REST="${BRANCH#*|}"; GOSET="${REST%%|*}"; DECMAP="${REST#*|}"
say "E5-1 mechanical decisions: $DECMAP"
say "SURVIVORS (GO) = [${GOSET}]  count=$NGO"

case "$NGO" in
  0)
    # ------------------------------------------------------- section 2 CASE C
    say "BRANCH = CASE C (0 GO)"
    say "STOP GPU METHOD WORK. E5-2A is NOT executable (no survivor)."
    say "Routing is CPU-only: E5-FAILURE-AUDIT (docs/E5_FAILURE_AUDIT.md, scripts/e5_failure_audit.py)."
    say "Forbidden: rescue, tuning, threshold change, C01++, C08 tuned, B2/X6c recall, C10 top-up."
    say "RESULT = BLOCKED_BY_CASE_C (GPU intentionally idle)"
    exit 0
    ;;
  1) MODE="single-survivor" ;;
  *) MODE="top-2-multi-view" ;;
esac

say "BRANCH = CASE $([ "$NGO" -eq 1 ] && echo B || echo A) — mode=$MODE"
if [ "$ALLOW_GPU" -ne 1 ]; then
  say "RESULT = READY_TO_RUN (dry run). Re-run with --allow-gpu to launch E5-2A."
  exit 0
fi

# ---------------------------------------------------------------- section 3/4/6: launch (A/B only)
# NOTE: `docs/E5_2A_FROZEN_PROTOCOL.md` must exist (frozen BEFORE reading any new-view result).
if [ ! -s docs/E5_2A_FROZEN_PROTOCOL.md ]; then
  say "RESULT = BLOCKED (docs/E5_2A_FROZEN_PROTOCOL.md missing — freeze first, section 3)."
  exit 4
fi
say "launching E5-2A ($MODE) in background"
HF_ENDPOINT=https://hf-mirror.com setsid nohup python -u scripts/e5_2a_runner.py \
  --survivors "$GOSET" --mode "$MODE" >> "$LOGDIR/e5_2a_run.log" 2>&1 < /dev/null &
say "PID=$!  log=$LOGDIR/e5_2a_run.log  progress=results/e5_2a/progress.json"
exit 0
