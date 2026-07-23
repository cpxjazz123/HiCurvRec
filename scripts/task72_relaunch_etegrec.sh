#!/bin/bash
# Task #72 Phase 4: ETEGRec RELAUNCH after safe_load fix
# Fix: utils.py safe_load now handles RQVAE nested state_dict

set -e

source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh
conda activate /home/wlia0047/ar57_scratch/wenyu/genrec_env

cd /home/wlia0047/ar57/wenyu/GeneRec/ETEGRec

GPU="${1:-cuda:0}"
LOG_FILE="/home/wlia0047/ar57/wenyu/GeneRec/logs/task72_phase4_etegrec_train_v2.log"

echo "===== ETEGRec v2 on $GPU starting at $(date) =====" | tee -a "$LOG_FILE"

# Verify RQVAE checkpoint and fix
RQVAE_CKPT="./dataset/Musical_Instruments/256-256-256-128.rqvae.pth"
if [ ! -f "$RQVAE_CKPT" ]; then
    echo "ERROR: RQVAE checkpoint not found at $RQVAE_CKPT"
    exit 1
fi

# Run on specified GPU
export CUDA_VISIBLE_DEVICES=$(echo $GPU | grep -oP '\d+')
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
    >> "$LOG_FILE" 2>&1

echo "===== ETEGRec v2 completed at $(date) =====" | tee -a "$LOG_FILE"
