#!/bin/bash
# Task #175 Stage 2 — RQ-VAE codebook inference (κ LOCKED at ORC)
# 输入: products/task175/orc_locked/best_loss_model.pth
# 输出: HG-Rec/dataset/Instruments/Instruments_t5_rqvae_orc_locked.npy

set -e

source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh
conda activate /home/wlia0047/ar57_scratch/wenyu/grid_toys

cd /home/wlia0047/ar57/wenyu/GeneRec/HG-Rec

export PYTHONPATH=/home/wlia0047/ar57/wenyu/GeneRec/HG-Rec:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export HF_HOME=/home/wlia0047/ar57_scratch/wenyu/hf_models
export TRITON_CACHE_DIR=/home/wlia0047/.triton/cache_task175_s2

LOG_DIR=/home/wlia0047/ar57/wenyu/GeneRec/logs/task175
TS=$(date +%b-%d-%Y_%H-%M-%S | sed 's/^.../\L&/')
LOG_FILE=$LOG_DIR/stage2_orc_locked_${TS}.log

mkdir -p $LOG_DIR

CKPT_PATH=/home/wlia0047/ar57/wenyu/GeneRec/products/task175/orc_locked/best_loss_model.pth
OUTPUT_PATH=/home/wlia0047/ar57/wenyu/GeneRec/HG-Rec/dataset/Instruments/Instruments_t5_rqvae_orc_locked.npy

echo "===== [Task #175 Stage 2] ORC LOCKED codebook inference launched at $(date) =====" | tee $LOG_FILE
echo "  ckpt: $CKPT_PATH" | tee -a $LOG_FILE
echo "  output: $OUTPUT_PATH" | tee -a $LOG_FILE

python3 -u /home/wlia0047/ar57/wenyu/GeneRec/scripts/task175_orc_locked_stage2_codebook.py \
    --ckpt_path $CKPT_PATH \
    --output_path $OUTPUT_PATH \
    --device cuda:0 \
    --max_sinkhorn_iters 30 \
    2>&1 | tee -a $LOG_FILE

EXIT_CODE=${PIPESTATUS[0]}
echo "===== [Task #175 Stage 2] exit code: $EXIT_CODE at $(date) =====" | tee -a $LOG_FILE
[ $EXIT_CODE -ne 0 ] && exit 1

ls -la $OUTPUT_PATH | tee -a $LOG_FILE
echo "✅ Task #175 Stage 2 完成" | tee -a $LOG_FILE