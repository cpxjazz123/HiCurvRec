#!/bin/bash
# launch_task59_s4.sh — HHHH Stage 4 inference
# 依赖: stage 3 best ckpt = logs/train/runs/task59_hhhh_s3/checkpoints/checkpoint_epoch=000_step=000700.ckpt
set -euo pipefail
source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh
conda activate /home/wlia0047/ar57_scratch/wenyu/grid_toys

cd /home/wlia0047/ar57/wenyu/GeneRec/GRID

SID_PATH=/home/wlia0047/ar57/wenyu/GeneRec/GRID/task_artifacts/results/exp59/task59_hhhh_l4_sid_tensor.pt
CKPT_PATH=/home/wlia0047/ar57/wenyu/GeneRec/GRID/logs/train/runs/task59_hhhh_s3/checkpoints/checkpoint_epoch=000_step=000700.ckpt

LOG_DIR=/home/wlia0047/ar57/wenyu/GeneRec/GRID/task_artifacts/scripts/logs
PID_DIR=/home/wlia0047/ar57/wenyu/GeneRec/GRID/task_artifacts/scripts/pids
mkdir -p $LOG_DIR $PID_DIR

echo "[task59 HHHH s4] using ckpt: $CKPT_PATH"
echo "[task59 HHHH s4] using SID:  $SID_PATH"

CUDA_VISIBLE_DEVICES=0 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
nohup python -m src.inference \
    experiment=tiger_inference_flat \
    data_dir=/home/wlia0047/ar57/wenyu/GeneRec/GRID/data/amazon_data/toys \
    semantic_id_path=${SID_PATH} \
    num_hierarchies=4 \
    ckpt_path=${CKPT_PATH} \
    ++should_skip_retry=true \
    paths.root_dir=/home/wlia0047/ar57/wenyu/GeneRec/GRID \
    id=task59_hhhh_s4 \
    > $LOG_DIR/task6_hhhh_s4_v1.log 2>&1 &

PID=$!
echo "$PID" > $PID_DIR/task59_hhhh_s4_pid
echo "[task59 HHHH s4] PID=$PID"
