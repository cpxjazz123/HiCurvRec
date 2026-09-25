#!/usr/bin/env bash
set -euo pipefail
PY=/home/wlia0047/ar57_scratch/wenyu/genrec_env/bin/python3.10
ITER14=/home/wlia0047/ar57/wenyu/GeneRec/stage2_RQ-VAE/curvature_RQ-VAE_iter14
STAGE2_LOG="$ITER14/logs/iter14_stage2_train.log"
MON_LOG="$ITER14/logs/monitor_iter14_pipeline.log"
STAGE3_ROOT=/home/wlia0047/ar57/wenyu/GeneRec/results/stage3_T5Train/curvature_RQ-VAE_iter14
STAGE3_LOG_DIR="$STAGE3_ROOT/logs"

log() { echo "$(date '+%F %T') $*" | tee -a "$MON_LOG"; }

log "monitor: waiting for Stage2 completion"
while true; do
  if grep -q '\[train\] done at global_step=100000' "$STAGE2_LOG" 2>/dev/null; then
    log "monitor: Stage2 finished"
    break
  fi
  if ! pgrep -f 'curvature_RQ-VAE_iter14/curvature_RQ-VAE.py' >/dev/null 2>&1; then
    if grep -q '\[train\] done' "$STAGE2_LOG" 2>/dev/null; then
      log "monitor: Stage2 process exited with done marker"
      break
    fi
    log "monitor: ERROR Stage2 died without done marker"
    tail -20 "$STAGE2_LOG" | tee -a "$MON_LOG"
    exit 1
  fi
  sleep 60
done

log "monitor: running export_sids_for_stage3.py"
cd "$ITER14"
"$PY" scripts/export_sids_for_stage3.py 2>&1 | tee -a "$MON_LOG"

log "monitor: launching Stage3"
mkdir -p "$STAGE3_LOG_DIR"
nohup "$PY" -u scripts/run_stage3_iter14.py >> "$ITER14/logs/stage3_iter14_background.log" 2>&1 &
log "monitor: Stage3 launcher pid=$!"

while true; do
  tf=$(find "$STAGE3_ROOT" -name test_final.json 2>/dev/null | head -1)
  if [ -n "$tf" ]; then
    log "monitor: Stage3 test_final.json written at $tf"
    cat "$tf" | tee -a "$MON_LOG"
    break
  fi
  launcher=$(find "$STAGE3_ROOT" -name '_stage3_launcher.log' 2>/dev/null | head -1)
  if [ -n "$launcher" ] && grep -q '\[RecBole-aligned\] test=' "$launcher" 2>/dev/null; then
    log "monitor: Stage3 test line found in launcher log"
    grep '\[RecBole-aligned\] test=' "$launcher" | tail -1 | tee -a "$MON_LOG"
    break
  fi
  if ! pgrep -f 'run_stage3_iter14.py' >/dev/null 2>&1 && \
     ! pgrep -f 'train_HG-Rec.py' >/dev/null 2>&1; then
    if [ -n "$launcher" ]; then
      if grep -q 'Traceback' "$launcher"; then
        log "monitor: ERROR Stage3 traceback"
        tail -30 "$launcher" | tee -a "$MON_LOG"
        exit 1
      fi
      log "monitor: Stage3 processes ended without test_final.json"
      tail -10 "$launcher" | tee -a "$MON_LOG"
      break
    fi
  fi
  sleep 120
done
log "monitor: pipeline watch finished"
