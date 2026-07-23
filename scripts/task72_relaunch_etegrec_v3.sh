#!/bin/bash
# Task #72 Phase 4: ETEGRec RELAUNCH v3
# Fixes: (1) safe_load RQVAE nested state_dict (2) batch_size=256 + grad_acc=2 for OOM fix
# (3) PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True for fragmentation
# Usage: bash scripts/task72_relaunch_etegrec_v3.sh cuda:1

set -e

source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh
conda activate /home/wlia0047/ar57_scratch/wenyu/genrec_env

cd /home/wlia0047/ar57/wenyu/GeneRec/ETEGRec

GPU="${1:-cuda:0}"
export CUDA_VISIBLE_DEVICES=$(echo $GPU | grep -oP '\d+')
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

LOG_FILE="/home/wlia0047/ar57/wenyu/GeneRec/logs/task72_phase4_etegrec_train_v3.log"

echo "===== ETEGRec v3 on $GPU starting at $(date) =====" | tee -a "$LOG_FILE"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES" | tee -a "$LOG_FILE"
echo "PYTORCH_CUDA_ALLOC_CONF=$PYTORCH_CUDA_ALLOC_CONF" | tee -a "$LOG_FILE"

# Verify RQVAE checkpoint
RQVAE_CKPT="./dataset/Musical_Instruments/256-256-256-128.rqvae.pth"
if [ ! -f "$RQVAE_CKPT" ]; then
    echo "ERROR: RQVAE checkpoint not found at $RQVAE_CKPT"
    exit 1
fi

# Run on specified GPU with reduced batch size for OOM fix
python main.py \
    --config ./config/musical_instruments.yaml \
    --lr_rec=0.005 \
    --lr_id=0.0001 \
    --cycle=2 \
    --eval_step=2 \
    --rec_kl_loss=0.0001 \
    --rec_dec_cl_loss=0.0003 \
    --id_kl_loss=0.0001 \
    --id_dec_cl_loss=0.0003 \
    --batch_size=256 \
    --gradient_accumulation_steps=2 \
    >> "$LOG_FILE" 2>&1

echo "===== ETEGRec v3 completed at $(date) =====" | tee -a "$LOG_FILE"
