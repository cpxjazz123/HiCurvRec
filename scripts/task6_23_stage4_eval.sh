#!/bin/bash
# task18_23_stage4_eval.sh — Stage 4 inference + SIDRetrievalEvaluator for HRQ/AQ
# Usage: ./task18_23_stage4_eval.sh <algo>  (algo = HRQ or AQ)
# Run AFTER TIGER val ckpt @ step 1600 is ready

set -euo pipefail
source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh
conda activate /home/wlia0047/ar57_scratch/wenyu/grid_toys
cd /fs04/ar57/wenyu/GeneRec/GRID

ALGO=$1

case $ALGO in
  HRQ)
    SEM_ID=/home/wlia0047/ar57/wenyu/GeneRec/GRID/logs/inference/runs/task18_hrq_s22/pickle/merged_predictions_tensor.pt
    S3_RUN=task18_hrq_s3
    OUT_DIR=/home/wlia0047/ar57/wenyu/GeneRec/GRID/logs/inference/runs/task18_hrq_s4
    SAFE_CKPT=/home/wlia0047/ar57/wenyu/GeneRec/GRID/task_artifacts/results/exp20/task18_hrq_tiger.ckpt
    EVAL_OUT=/home/wlia0047/ar57/wenyu/GeneRec/GRID/result/task7/task18_eval_HRQ.json
    GPU=0
    ;;
  AQ)
    SEM_ID=/home/wlia0047/ar57/wenyu/GeneRec/GRID/logs/inference/runs/task19_aq_s22/pickle/merged_predictions_tensor.pt
    S3_RUN=task19_aq_s3
    OUT_DIR=/home/wlia0047/ar57/wenyu/GeneRec/GRID/logs/inference/runs/task19_aq_s4
    SAFE_CKPT=/home/wlia0047/ar57/wenyu/GeneRec/GRID/task_artifacts/results/exp21/task19_aq_tiger.ckpt
    EVAL_OUT=/home/wlia0047/ar57/wenyu/GeneRec/GRID/result/task8/task19_eval_AQ.json
    GPU=1
    ;;
  *)
    echo "Usage: $0 HRQ|AQ"
    exit 1
    ;;
esac

# Pick best_*.ckpt or step=1600 ckpt
S3_CKPT=$(ls -d /home/wlia0047/ar57/wenyu/GeneRec/GRID/logs/train/runs/${S3_RUN}/checkpoints/best*.ckpt 2>/dev/null | head -1)
if [ -z "$S3_CKPT" ] || [ ! -f "$S3_CKPT" ]; then
    S3_CKPT=$(ls -d /home/wlia0047/ar57/wenyu/GeneRec/GRID/logs/train/runs/${S3_RUN}/checkpoints/checkpoint_epoch=000_step=001600.ckpt 2>/dev/null | head -1)
fi
if [ -z "$S3_CKPT" ] || [ ! -f "$S3_CKPT" ]; then
    echo "ERROR: Stage 3 ckpt not found for $ALGO"
    exit 1
fi
echo "[info] $ALGO Stage 3 ckpt: $S3_CKPT"

mkdir -p /home/wlia0047/ar57/wenyu/GeneRec/GRID/task_artifacts/results/exp${ALGO,,}
cp -f "$S3_CKPT" "$SAFE_CKPT"
echo "[info] Safe copy: $SAFE_CKPT"

# Stage 4 inference (use a free GPU; 2/3 if available)
mkdir -p $OUT_DIR
FREE_GPU=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | awk -F',' '$2 < 1000 {print $1; exit}')
if [ -z "$FREE_GPU" ]; then FREE_GPU=$GPU; fi
echo "[info] Using GPU $FREE_GPU for inference"

CUDA_VISIBLE_DEVICES=$FREE_GPU python -m src.inference \
    experiment=tiger_inference_flat \
    data_dir=/fs04/ar57/wenyu/GeneRec/GRID/data/amazon_data/toys \
    semantic_id_path=$SEM_ID \
    num_hierarchies=4 \
    ckpt_path=$SAFE_CKPT \
    ++should_skip_retry=true \
    paths.root_dir=/fs04/ar57/wenyu/GeneRec/GRID \
    id=${S3_RUN}_s4 \
    2>&1 | tee /home/wlia0047/ar57/wenyu/GeneRec/GRID/task_artifacts/scripts/logs/${S3_RUN}_s4.log

# Eval (run with same GPU)
cd /home/wlia0047/ar57/wenyu/GeneRec/GRID
CUDA_VISIBLE_DEVICES=$FREE_GPU python task_artifacts/scripts/hrq_aq_eval.py \
    --algo $ALGO \
    --sid $SEM_ID \
    --stage4 $OUT_DIR/pickle/merged_predictions_tensor.pt \
    --ckpt $SAFE_CKPT \
    --num_hier 4 \
    --out_json $EVAL_OUT

echo "=== $ALGO Eval done → $EVAL_OUT ==="