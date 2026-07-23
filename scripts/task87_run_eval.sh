#!/bin/bash
# Task #87 Stage 4 evaluation
set -euo pipefail
source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh
conda activate /home/wlia0047/ar57_scratch/wenyu/grid_toys
cd /home/wlia0047/ar57/wenyu/GeneRec

PRED_TENSOR=/home/wlia0047/ar57/wenyu/GeneRec/logs/inference/runs/2026-07-19/09-33-15/pickle/merged_predictions_tensor.pt
SID_PATH=/home/wlia0047/ar57/wenyu/GeneRec/products/task87_tiger_baseline/stage2_rqvae_infer/sid_dedup.pt
CKPT=/home/wlia0047/ar57/wenyu/GeneRec/products/task87_tiger_baseline/stage3_train/best.ckpt
OUT_JSON=/home/wlia0047/ar57/wenyu/GeneRec/verdicts/task87_tiger_baseline_eval.json

# Use the proven task388v4 item-level evaluator (modified to point at GeneRec configs)
python /home/wlia0047/ar57/wenyu/GeneRec/scripts/task87_s4_item_eval.py \
    --variant task87_tiger_baseline \
    --constrained_pt "$PRED_TENSOR" \
    --sid "$SID_PATH" \
    --ckpt "$CKPT" \
    --out_json "$OUT_JSON" \
    2>&1 | tee /home/wlia0047/ar57/wenyu/GeneRec/logs/task87_s4_eval.log
