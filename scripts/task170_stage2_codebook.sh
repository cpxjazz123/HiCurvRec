#!/bin/bash
# Task #170 Stage 2 — κ-Stereo + Sinkhorn(ALL 3 layers) codebook inference

set -e

source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh
conda activate /home/wlia0047/ar57_scratch/wenyu/grid_toys

cd /home/wlia0047/ar57/wenyu/GeneRec/HG-Rec

export PYTHONPATH=/home/wlia0047/ar57/wenyu/GeneRec/HG-Rec:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=0
export HF_HOME=/home/wlia0047/ar57_scratch/wenyu/hf_models

LOG_DIR=/home/wlia0047/ar57/wenyu/GeneRec/logs/task170
mkdir -p $LOG_DIR
TS=$(date +%b-%d-%Y_%H-%M-%S | sed 's/^.../\L&/')
LOG_FILE=$LOG_DIR/stage2_codebook_${TS}.log

BEST_CKPT=$(ls -t /home/wlia0047/ar57/wenyu/GeneRec/products/task170/phase_b_kappa_sinkhorn_all3/*/best_loss_model.pth 2>/dev/null | head -1)
OUTPUT_PATH=/home/wlia0047/ar57/wenyu/GeneRec/HG-Rec/dataset/Instruments/Instruments_t5_rqvae_phase_b_kappa_sinkhorn_all3.npy

if [ -z "$BEST_CKPT" ] || [ ! -f "$BEST_CKPT" ]; then
    echo "❌ Stage 1 best ckpt MISSING for #170" | tee "$LOG_FILE"
    exit 1
fi

echo "===== [Task #170 Stage 2] κ-Stereo + Sinkhorn(ALL) → SID codebook launched at $(date) =====" | tee "$LOG_FILE"
echo "Best ckpt: $BEST_CKPT" | tee -a $LOG_FILE
echo "Output: $OUTPUT_PATH" | tee -a $LOG_FILE

python3 /home/wlia0047/ar57/wenyu/GeneRec/scripts/task164_stage2_codebook.py \
    --ckpt_path $BEST_CKPT \
    --output_path $OUTPUT_PATH \
    --device cuda:0 \
    --max_sinkhorn_iters 30 \
    2>&1 | tee -a $LOG_FILE

STAGE2_EXIT=${PIPESTATUS[0]}
echo "===== [Task #170 Stage 2] exit code: $STAGE2_EXIT at $(date) =====" | tee -a $LOG_FILE

if [ $STAGE2_EXIT -ne 0 ] || [ ! -f "$OUTPUT_PATH" ]; then
    echo "❌ Stage 2 FAILED" | tee -a $LOG_FILE
    exit 1
fi

echo "✅ Task #170 Stage 2 SID codebook 生成成功" | tee -a $LOG_FILE
ls -la $OUTPUT_PATH | tee -a $LOG_FILE