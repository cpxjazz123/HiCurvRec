#!/bin/bash
# Task #253 Stage 2 — 用 Issue #13 Gate 2 best_collision ep34 ckpt 推断 SID
set -e
REPO=/home/wlia0047/ar57/wenyu/GeneRec
source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh
conda activate /home/wlia0047/ar57_scratch/wenyu/grid_toys
cd $REPO/HG-Rec

export PYTHONPATH=$REPO/HG-Rec:$PYTHONPATH
export HF_HOME=/home/wlia0047/ar57_scratch/wenyu/hf_models

CKPT=$REPO/products/task253/hrqvae_mobius_residual/Jul-29-2026_03-29-51_beta_0.500_codebook_[64,128,256]_sk_0.000/best_collision_model.pth
OUT=$REPO/HG-Rec/dataset/Instruments/Instruments_t5_rqvae_mobius_residual_issue13_gate2.npy
LOG=$REPO/logs/task253/stage2_codebook.out
mkdir -p $REPO/logs/task253 $REPO/products/task253

CUDA_VISIBLE_DEVICES=3 TRITON_CACHE_DIR=/home/wlia0047/.triton/cache_task253_stage2 \
python3 -u $REPO/scripts/task223_stage2_codebook.py \
    --ckpt_path "$CKPT" \
    --output_path "$OUT" \
    --device cuda:0 \
    --max_sinkhorn_iters 30 \
    > $LOG 2>&1

echo "[$(date)] Task #253 Stage 2 done"
tail -30 $LOG