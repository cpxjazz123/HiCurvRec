#!/bin/bash
# Stage 4 inference + item-level eval for v5 (5 道防线版)
# Usage: bash launch_v5_s4_inference_eval.sh <variant> <gpu>
#   variant: H_E_E_E | H_H_E_E | H_H_H_H
set -euo pipefail
source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh
conda activate /home/wlia0047/ar57_scratch/wenyu/grid_toys

cd /fs04/ar57/wenyu/GeneRec/GRID/.claude/worktrees/task9-decoder-only

VARIANT=$1
GPU=${2:-0}

case $VARIANT in
    H_E_E_E) SHORT=hee ;;
    H_H_E_E) SHORT=hhee ;;
    H_H_H_H) SHORT=hhhh ;;
esac

DATA_DIR=/fs04/ar57/wenyu/GeneRec/GRID/data/amazon_data/toys
SID_PATH=/fs04/ar57/wenyu/GeneRec/GRID/logs/inference/runs/task388v4_${SHORT}_s22/pickle/merged_predictions_tensor.pt

# Pick BEST ckpt (top-1 from save_top_k) — 不再 fallback 到 last
S3_CKPT_DIR=/fs04/ar57/wenyu/GeneRec/GRID/logs/train/runs/task388v5_${VARIANT,,}_s3/checkpoints
BEST_CKPT=$(ls -d ${S3_CKPT_DIR}/checkpoint_epoch*.ckpt 2>/dev/null | sort -t= -k3 -n | tail -1)
if [ -z "$BEST_CKPT" ]; then
    echo "ERROR: no best ckpt (checkpoint_epoch*.ckpt) found in $S3_CKPT_DIR"
    exit 1
fi

SAFE_CKPT=/home/wlia0047/ar57/wenyu/GeneRec/GRID/task_artifacts/results/exp388v5/task388v5_${SHORT}_tiger.ckpt
mkdir -p /home/wlia0047/ar57/wenyu/GeneRec/GRID/task_artifacts/results/exp388v5
cp -f "$BEST_CKPT" "$SAFE_CKPT"
echo "[Stage4 v5 $VARIANT] BEST ckpt (top-1): $BEST_CKPT"
echo "[Stage4 v5 $VARIANT] SID_PATH: $SID_PATH"
echo "[Stage4 v5 $VARIANT] GPU: $GPU"

# Step 1: Stage 4 inference (generate top-10 candidates per user)
CUDA_VISIBLE_DEVICES=${GPU} python -m src.inference \
    experiment=tiger_inference_flat \
    data_dir=${DATA_DIR} \
    semantic_id_path=${SID_PATH} \
    num_hierarchies=4 \
    ckpt_path=${SAFE_CKPT} \
    ++should_skip_retry=true \
    paths.root_dir=/fs04/ar57/wenyu/GeneRec/GRID \
    id=task388v5_${VARIANT,,}_s4 \
    2>&1 | tee /home/wlia0047/ar57/wenyu/GeneRec/GRID/task_artifacts/scripts/logs/v5_${VARIANT,,}_s4.log

# Step 2: Item-level R@5/R@10/NDCG@5/NDCG@10 eval
S4_PRED=/fs04/ar57/wenyu/GeneRec/GRID/logs/inference/runs/task388v5_${VARIANT,,}_s4/pickle/merged_predictions_tensor.pt
EVAL_OUT=/home/wlia0047/ar57/wenyu/GeneRec/GRID/task_artifacts/results/exp388v5/task388v5_${VARIANT,,}_eval.json

python task_artifacts/scripts/task388v4_s4_item_eval.py \
    --variant ${VARIANT} \
    --constrained_pt ${S4_PRED} \
    --sid ${SID_PATH} \
    --ckpt ${SAFE_CKPT} \
    --out_json ${EVAL_OUT} \
    2>&1 | tee /home/wlia0047/ar57/wenyu/GeneRec/GRID/task_artifacts/scripts/logs/v5_${VARIANT,,}_s4_eval.log