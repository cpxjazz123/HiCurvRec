#!/bin/bash
# Task #72 Phase 4: ETEGRec full training for Musical_Instruments
# Uses 2 GPUs. Requires RQVAE pretrained checkpoint first.

set -e

source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh
conda activate /home/wlia0047/ar57_scratch/wenyu/genrec_env

cd /home/wlia0047/ar57/wenyu/GeneRec/ETEGRec

# Verify RQVAE checkpoint exists
RQVAE_CKPT="./dataset/Musical_Instruments/256-256-256-128.rqvae.pth"
if [ ! -f "$RQVAE_CKPT" ]; then
    echo "ERROR: RQVAE checkpoint not found at $RQVAE_CKPT"
    echo "Run RQVAE pretraining first: task72_phase4_etegrec_pretrain_rqvae.sh"
    exit 1
fi

echo "Starting ETEGRec training at $(date)"
echo "Using config: ./config/musical_instruments.yaml"

export CUDA_VISIBLE_DEVICES=1,2
accelerate launch --config_file accelerate_config_ddp_2gpu.yaml main.py \
    --config ./config/musical_instruments.yaml \
    --lr_rec=0.005 \
    --lr_id=0.0001 \
    --cycle=2 \
    --eval_step=2 \
    --rec_kl_loss=0.0001 \
    --rec_dec_cl_loss=0.0003 \
    --id_kl_loss=0.0001 \
    --id_dec_cl_loss=0.0003

echo "ETEGRec training completed at $(date)"