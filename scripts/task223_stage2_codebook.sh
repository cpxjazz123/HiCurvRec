#!/bin/bash
# Task #223 — Stage 2 SID 推断 (用 Task #222 best_collision ep29 ckpt)
set -e
REPO=/home/wlia0047/ar57/wenyu/GeneRec
source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh
conda activate /home/wlia0047/ar57_scratch/wenyu/grid_toys
cd $REPO/HG-Rec

export PYTHONPATH=$REPO/HG-Rec:$PYTHONPATH
export HF_HOME=/home/wlia0047/ar57_scratch/wenyu/hf_models

mkdir -p $REPO/products/task223 $REPO/logs/task223

CKPT=$REPO/products/task222/hrqvae_pck_replay/Jul-26-2026_23-03-27_beta_0.500_codebook_[64,128,256]_sk_0.000/best_collision_model.pth
OUT=$REPO/HG-Rec/dataset/Instruments/Instruments_t5_rqvae_task223_pck.npy
LOG=$REPO/logs/task223/stage2_codebook.out

CUDA_VISIBLE_DEVICES=3 TRITON_CACHE_DIR=/home/wlia0047/.triton/cache_task223_stage2 \
python3 -u $REPO/scripts/task223_stage2_codebook.py \
    --ckpt_path "$CKPT" \
    --output_path "$OUT" \
    --device cuda:0 \
    --max_sinkhorn_iters 30 \
    > $LOG 2>&1

echo "[$(date)] Task #223 Stage 2 done"
tail -30 $LOG