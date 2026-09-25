#!/usr/bin/env bash
set -euo pipefail
ROOT=/home/wlia0047/ar57/wenyu/GeneRec/results/stage3_T5Train/curvature_RQ-VAE_iter14
OUT=/home/wlia0047/ar57/wenyu/GeneRec/stage2_RQ-VAE/curvature_RQ-VAE_iter14/logs/iter14_test_final_snapshot.json
METRICS="$ROOT/logs/Amazon_2023_Instruments/Sep-25-2026_21-07-42/training_metrics.jsonl"
LOG="$ROOT/logs/_stage3_launcher.log"
for i in $(seq 1 90); do
  tf=$(find "$ROOT" -name test_final.json 2>/dev/null | head -1)
  if [ -n "$tf" ]; then cp -f "$tf" "$OUT"; echo "$(date -Is) PIPELINE_COMPLETE" >> "${OUT%.json}.log"; exit 0; fi
  if [ -f "$LOG" ] && grep -q '\[RecBole-aligned\] test=' "$LOG" 2>/dev/null; then
    grep '\[RecBole-aligned\] test=' "$LOG" | tail -1 > "$OUT.line"
    echo "$(date -Is) TEST_LINE" >> "${OUT%.json}.log"; exit 0
  fi
  if ! pgrep -f 'torchrun.*run_stage3_iter14' >/dev/null 2>&1; then
    echo "$(date -Is) TORCHRUN_ENDED" >> "${OUT%.json}.log"; tail -20 "$LOG" >> "${OUT%.json}.log" 2>/dev/null || true; exit 1
  fi
  ep=$(python3 -c "import json;e='?'
for l in open('$METRICS'):
 o=json.loads(l)
 if o.get('event')=='train': e=o['epoch']
print(e)" 2>/dev/null || echo ?)
  echo "$(date -Is) poll=$i epoch=$ep" >> "${OUT%.json}.log"
  sleep 60
done
exit 2
