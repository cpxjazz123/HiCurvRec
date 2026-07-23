#!/bin/bash
# Task #73 RQ-VAE training for TIGER-SAS (SASRec 128d embedding) on Instrument
# Input: ETEGRec/dataset/Musical_Instruments/Musical_Instruments_emb_128.npy (24588, 128)
# Output: products/task73/rqvae_tiger_sas/
# Run on GPU 1 (after TIGER-text RQ-VAE finishes); ETEGRec uses GPU 0

set -e

source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh
conda activate /home/wlia0047/ar57_scratch/wenyu/genrec_env

cd /home/wlia0047/ar57/wenyu/GeneRec/ETEGRec/RQVAE

DATA_PATH="/home/wlia0047/ar57/wenyu/GeneRec/ETEGRec/dataset/Musical_Instruments/Musical_Instruments_emb_128.npy"
CKPT_DIR="/home/wlia0047/ar57/wenyu/GeneRec/products/task73/rqvae_tiger_sas"
LOG_DIR="/home/wlia0047/ar57/wenyu/GeneRec/logs"

mkdir -p "$CKPT_DIR" "$LOG_DIR"

export CUDA_VISIBLE_DEVICES=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

LOG_FILE="$LOG_DIR/task73_rqvae_tiger_sas.log"
echo "===== [Task #73] RQ-VAE (TIGER-SAS) started at $(date) =====" | tee "$LOG_FILE"

# For 128-dim SASRec embedding → 128-dim RQ-VAE, use smaller layers
python -u main.py \
    --lr 1e-3 \
    --epochs 5000 \
    --batch_size 1024 \
    --weight_decay 1e-4 \
    --lr_scheduler_type cosine \
    --e_dim 128 \
    --quant_loss_weight 1.0 \
    --beta 0.25 \
    --num_emb_list 256 256 256 \
    --sk_epsilons 0.0 0.0 0.0 \
    --layers 512 256 128 \
    --vq_type vq \
    --loss_type mse \
    --data_path "$DATA_PATH" \
    --ckpt_dir "$CKPT_DIR" \
    --eval_step 50 \
    --device cuda:0 \
    2>&1 | tee -a "$LOG_FILE"

echo "===== [Task #73] RQ-VAE (TIGER-SAS) completed at $(date) =====" | tee -a "$LOG_FILE"
