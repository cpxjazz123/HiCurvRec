#!/bin/bash
# Task #402 / Stage 4 R@K eval wait helper
# 等 task402 Stage 3 (PID 1077186) 完成, 然后 launch Stage 4 R@K eval
# 跟 task398 wait helper 模式一致 (R12 ckpt 强制落盘)
set -e

# Step 1: wait Stage 3
TRAIN_PID=1077186
echo "[$(date)] Wait for task402 Stage 3 PID $TRAIN_PID to complete..."
while kill -0 $TRAIN_PID 2>/dev/null; do
    sleep 60
done
echo "[$(date)] task402 Stage 3 PID $TRAIN_PID finished"

# Step 2: find best ckpt
CKPT_DIR=/home/wlia0047/ar57/wenyu/GeneRec/products/task402_ckpt_downstream/ckpt
echo "[$(date)] Check ckpt in $CKPT_DIR"
ls -la $CKPT_DIR 2>&1

# Step 3: launch Stage 4 R@K eval
source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh
conda activate /home/wlia0047/ar57_scratch/wenyu/genrec_env
cd /home/wlia0047/ar57/wenyu/GeneRec
export PYTHONPATH=/home/wlia0047/ar57/wenyu/GeneRec/HG-Rec:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=2
export HF_HOME=/home/wlia0047/ar57_scratch/wenyu/hf_models
export TRITON_CACHE_DIR=/home/wlia0047/.triton/cache_task402_stage4
mkdir -p $TRITON_CACHE_DIR

mkdir -p /home/wlia0047/ar57/wenyu/GeneRec/products/task402_ckpt_downstream/stage4
cd /home/wlia0047/ar57/wenyu/GeneRec/products/task402_ckpt_downstream/stage4
nohup bash /home/wlia0047/ar57/wenyu/GeneRec/scripts/task398_stage4_rk_eval.sh \
    _t5_rqvae_task402.npy \
    /home/wlia0047/ar57/wenyu/GeneRec/products/task402_ckpt_downstream/ckpt/ \
    /home/wlia0047/ar57/wenyu/GeneRec/products/task402_ckpt_downstream/stage4 \
    > stage4_nohup.out 2>&1 &
STAGE4_PID=$!
disown
echo $STAGE4_PID > _STAGE4_PID
echo "[$(date)] Launched task402 Stage 4 PID: $STAGE4_PID"