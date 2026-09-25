#!/usr/bin/env bash
set -euo pipefail
ROOT=/home/wlia0047/ar57/wenyu/GeneRec/results/stage3_T5Train/curvature_RQ-VAE_iter14
FLAG=/home/wlia0047/ar57/wenyu/GeneRec/stage2_RQ-VAE/curvature_RQ-VAE_iter14/logs/pipeline_status.flag
LOG=/home/wlia0047/ar57/wenyu/GeneRec/stage2_RQ-VAE/curvature_RQ-VAE_iter14/logs/wait_stage3_test_final.log
log() { echo "$(date '+%F %T') $*" | tee -a "$LOG"; }
log "waiting for test_final.json under $ROOT"
while true; do
  tf=$(find "$ROOT" -name test_final.json 2>/dev/null | head -1)
  if [ -n "$tf" ]; then
    log "found $tf"
    cat "$tf" | tee -a "$LOG"
    echo "STAGE3_DONE $tf" > "$FLAG"
    exit 0
  fi
  if ! pgrep -f 'run_stage3_iter14.py' >/dev/null 2>&1; then
    launcher="$ROOT/logs/_stage3_launcher.log"
    if [ -f "$launcher" ] && grep -q '\[RecBole-aligned\] test=' "$launcher" 2>/dev/null; then
      log "Stage3 exited; test line in launcher"
      grep '\[RecBole-aligned\] test=' "$launcher" | tail -1 | tee -a "$LOG"
      echo "STAGE3_DONE launcher" > "$FLAG"
      exit 0
    fi
    log "ERROR: Stage3 not running and no test_final.json"
    tail -30 "$launcher" 2>/dev/null | tee -a "$LOG" || true
    echo "STAGE3_FAILED" > "$FLAG"
    exit 1
  fi
  ep=$(grep -oP '\[train\] epoch=\K[0-9]+' "$ROOT/logs/_stage3_launcher.log" 2>/dev/null | tail -1 || true)
  log "still training epoch=${ep:-?}"
  sleep 120
done
