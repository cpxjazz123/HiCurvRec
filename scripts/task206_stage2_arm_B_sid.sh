#!/bin/bash
# Task #206 Stage 2 — 欧式 Arm B SID 推理 (Sinkhorn 解码 + 4-digit dedup)
#
# 目的: 跟 #84 HG-Rec baseline 对比, 走完整 Stage 2 -> 3 -> 4 流水线.
# 跟 task188_stage2_codebook.py 完全相同 pattern, 只是 ckpt_path 和 output_path 不同.
# GPU 1 (Arm B 1000ep 已用 GPU 0, 让出来; sanity iii 在 GPU 2 跑到 11+ 分钟还没完)

set -e
REPO=/home/wlia0047/ar57/wenyu/GeneRec
source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh
conda activate /home/wlia0047/ar57_scratch/wenyu/grid_toys
export PYTHONPATH=$REPO/HG-Rec:$PYTHONPATH
export HF_HOME=/home/wlia0047/ar57_scratch/wenyu/hf_models

mkdir -p $REPO/logs/task206 $REPO/HG-Rec/dataset/Instruments

CKPT=$(ls -t $REPO/products/task206/stage1_arm_B_euclidean/*/best_collision_model.pth | head -1)
OUT=$REPO/HG-Rec/dataset/Instruments/Instruments_t5_rqvae_euclidean_v1.npy
LOG=$REPO/logs/task206/stage2_arm_B_euclidean.out
GPU=1

echo "[$(date)] === #206 Stage 2 Arm B 1000ep: ckpt=$CKPT, output=$OUT, GPU=$GPU ===" | tee $LOG

CUDA_VISIBLE_DEVICES=$GPU TRITON_CACHE_DIR=/home/wlia0047/.triton/cache_task206_stage2_arm_B \
python3 -u $REPO/scripts/task188_stage2_codebook.py \
    --ckpt_path $CKPT \
    --output_path $OUT \
    --device cuda:0 \
    > $LOG 2>&1 &
PID=$!
echo $PID > $REPO/products/task206/_STAGE2_PID_arm_B
echo "[$(date)] Stage 2 launched PID=$PID" | tee -a $LOG