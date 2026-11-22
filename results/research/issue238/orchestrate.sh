#!/bin/bash
# Serial driver: wait until the cards are free, then run H, then J.
cd /fs04/ar57/wenyu/GeneRec || exit 1
PY=/home/wlia0047/ar57_scratch/wenyu/genrec_env_v2/bin/python3.9
LOG=results/research/issue238/orchestrate.log
for i in $(seq 60); do
  BUSY=$(pgrep -fc "run_stage3_param_" || true)
  [ "${BUSY:-0}" -eq 0 ] && break
  sleep 20
done
echo "=== cards free $(date -Iseconds)" >> "$LOG"
for ARM in TIGER_kappa_cost_rq TIGER_kappa_sk_only_rq; do
  echo "=== $ARM start $(date -Iseconds)" >> "$LOG"
  "$PY" results/research/issue238/run_arm.py "$ARM" >> "$LOG" 2>&1
  RC=$?
  echo "=== $ARM exit=$RC $(date -Iseconds)" >> "$LOG"
  [ $RC -ne 0 ] && break
done
echo "=== ORCHESTRATE COMPLETE ===" >> "$LOG"
