#!/usr/bin/env bash
set -euo pipefail
PY=/home/wlia0047/ar57_scratch/wenyu/genrec_env_v2/bin/python3.9
ITER14=/home/wlia0047/ar57/wenyu/GeneRec/stage2_RQ-VAE/curvature_RQ-VAE_iter14
STAGE2_LOG="$ITER14/logs/iter14_stage2_train.log"
MON_LOG="$ITER14/logs/monitor_iter14_pipeline.log"
STAGE3_ROOT=/home/wlia0047/ar57/wenyu/GeneRec/results/stage3_T5Train/curvature_RQ-VAE_iter14
FLAG="$ITER14/logs/pipeline_status.flag"
log() { echo "$(date '+%F %T') watcher: $*" | tee -a "$MON_LOG" >> "$FLAG"; }

# Stage2
while true; do
  if grep -q '\[train\] done at global_step=100000' "$STAGE2_LOG" 2>/dev/null; then
    log "STAGE2_DONE marker in log"
    break
  fi
  if ! pgrep -f 'curvature_RQ-VAE_iter14/curvature_RQ-VAE.py' >/dev/null 2>&1; then
    if grep -q '\[train\] done' "$STAGE2_LOG" 2>/dev/null; then break; fi
    log "PIPELINE_ERROR Stage2 died"
    exit 1
  fi
  sleep 45
done

# Ensure export + stage3 (monitor may also do this; idempotent check)
if [ ! -f /home/wlia0047/ar57/wenyu/GeneRec/results/stage2_RQ-VAE/curvature_RQ-VAE_iter14/item_sids.json ]; then
  log "running export_sids_for_stage3.py (watcher)"
  cd "$ITER14" && "$PY" scripts/export_sids_for_stage3.py >> "$MON_LOG" 2>&1
fi
if ! pgrep -f 'run_stage3_iter14.py' >/dev/null && ! pgrep -f 'train_HG-Rec.py' >/dev/null; then
  if ! find "$STAGE3_ROOT" -name test_final.json 2>/dev/null | grep -q .; then
    log "launching Stage3 (watcher backup)"
    mkdir -p "$STAGE3_ROOT/logs"
    nohup "$PY" -u "$ITER14/scripts/run_stage3_iter14.py" >> "$ITER14/logs/stage3_iter14_background.log" 2>&1 &
  fi
fi

# Stage3 - find test_final.json anywhere under stage3 iter14
while true; do
  tf=$(find "$STAGE3_ROOT" -name test_final.json 2>/dev/null | head -1)
  if [ -n "$tf" ]; then
    log "PIPELINE_COMPLETE"
    cat "$tf" >> "$FLAG"
    exit 0
  fi
  if ! pgrep -f 'run_stage3_iter14.py' >/dev/null && ! pgrep -f 'train_HG-Rec.py' >/dev/null; then
    launcher=$(find "$STAGE3_ROOT" -name '_stage3_launcher.log' 2>/dev/null | head -1)
    if [ -n "$launcher" ] && grep -q 'Traceback' "$launcher"; then
      log "PIPELINE_ERROR Stage3 traceback"
      exit 1
    fi
    if [ -n "$launcher" ] && grep -q '\[RecBole-aligned\] test=' "$launcher"; then
      log "PIPELINE_COMPLETE from launcher log"
      exit 0
    fi
  fi
  sleep 90
done
