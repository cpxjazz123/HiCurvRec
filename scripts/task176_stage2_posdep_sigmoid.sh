#!/bin/bash
# Task #176 Stage 2 — PosDepBetaHRQVAE (sigmoid) codebook inference
# GPU 0 (GPU 1/2 已被 #176/#177 Stage 3 占用)

set -e

source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh
conda activate /home/wlia0047/ar57_scratch/wenyu/grid_toys

cd /home/wlia0047/ar57/wenyu/GeneRec/HG-Rec

export PYTHONPATH=/home/wlia0047/ar57/wenyu/GeneRec/HG-Rec:$PYTHONPATH
export HF_HOME=/home/wlia0047/ar57_scratch/wenyu/hf_models
export TRITON_CACHE_DIR=/home/wlia0047/.triton/cache_task176_s2

LOG_DIR=/home/wlia0047/ar57/wenyu/GeneRec/logs/task176
TS=$(date +%b-%d-%Y_%H-%M-%S | sed 's/^.../\L&/')
LOG_FILE=$LOG_DIR/stage2_sigmoid_${TS}.log

CKPT=/home/wlia0047/ar57/wenyu/GeneRec/products/task176/posdep_sigmoid/best_loss_model.pth
OUT_DIR=/home/wlia0047/ar57/wenyu/GeneRec/products/task176/t5mini_posdep_sigmoid
mkdir -p $OUT_DIR

echo "===== [Task #176 Stage 2] sigmoid β(x) launched at $(date) =====" | tee $LOG_FILE

python3 -u /home/wlia0047/ar57/wenyu/GeneRec/scripts/task176_177_posdep_beta_stage2_codebook.py \
    --ckpt_path $CKPT \
    --output_path $OUT_DIR/_t5_rqvae_posdep_sigmoid.npy \
    --beta_mode sigmoid \
    --beta_base 1.0 --beta_scale 10.0 --beta_max 5.0 \
    --device cuda:0 \
    --max_sinkhorn_iters 30 \
    2>&1 | tee -a $LOG_FILE

EXIT_CODE=${PIPESTATUS[0]}
echo "===== [Task #176 Stage 2] exit code: $EXIT_CODE at $(date) =====" | tee -a $LOG_FILE
[ $EXIT_CODE -ne 0 ] && exit 1

echo "✅ Task #176 Stage 2 完成 (output: $OUT_DIR/_t5_rqvae_posdep_sigmoid.npy)" | tee -a $LOG_FILE