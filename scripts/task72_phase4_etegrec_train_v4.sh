#!/bin/bash
# Task #72 Phase 4: ETEGRec v4 (single-GPU, batch_size=128, grad_acc=4)
# 根因: v3 用 2-GPU DDP 但 GPU 2 被 SASRec 占据,导致 fallback 单卡 OOM (41.75 GB > 44.39 GB)
# 修复: 单 GPU (CUDA_VISIBLE_DEVICES=1), batch_size 从 512 降到 128, grad_acc=4 (effective batch 512)

set -e

source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh
conda activate /home/wlia0047/ar57_scratch/wenyu/genrec_env

cd /home/wlia0047/ar57/wenyu/GeneRec/ETEGRec

# Verify RQVAE checkpoint exists
RQVAE_CKPT="./dataset/Musical_Instruments/256-256-256-128.rqvae.pth"
if [ ! -f "$RQVAE_CKPT" ]; then
    echo "ERROR: RQVAE checkpoint not found at $RQVAE_CKPT"
    exit 1
fi

# GPU selection: single GPU. Defaults to GPU 1 (currently used by Phase 6 BERT4Rec).
# Override with CUDA_VISIBLE_DEVICES env var before launch.
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Use Python directly (no accelerate), single-process
# Override config defaults: batch_size 512 -> 128, gradient_accumulation_steps 1 -> 4
python main.py \
    --config ./config/musical_instruments.yaml \
    --batch_size=128 \
    --gradient_accumulation_steps=4 \
    --lr_rec=0.005 \
    --lr_id=0.0001 \
    --cycle=2 \
    --eval_step=2 \
    --rec_kl_loss=0.0001 \
    --rec_dec_cl_loss=0.0003 \
    --id_kl_loss=0.0001 \
    --id_dec_cl_loss=0.0003 \
    >> /home/wlia0047/ar57/wenyu/GeneRec/logs/task72_phase4_etegrec_train_v4.log 2>&1

echo "ETEGRec v4 training completed at $(date)"
