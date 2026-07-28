#!/bin/bash
# Task #204 Stage 1 臂 A — c=1 fixed + quant_loss_weight=2.2 (用户 2026-07-26 决定性实验)
#
# 目的: 让 c=1 的 quant_loss magnitude ≈ c=10 的 magnitude (2.2×)。
# 如果 collision ≈ #199 D c=10 (8.26%) → c=10 的优势是 loss 量级伪影, 几何本身无害也无益.
# 如果 collision 退化 (>> 8.26%) → c=10 的优势是真实几何增益, c=1 是真实最优.
#
# 配置 (跟 #199 D c=1 对齐):
#   epochs=1000, batch_size=256, beta=0.5, sk_epsilons=[0,0,0], e_dim=32, num_emb_list=[64,128,256]
#   lr=1e-3, kmeans_iters=1000, loss_type=poincare, curvatures=[1,1,1], quant_loss_weight=2.2
# GPU 1

set -e
REPO=/home/wlia0047/ar57/wenyu/GeneRec
source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh
conda activate /home/wlia0047/ar57_scratch/wenyu/grid_toys
cd $REPO/HG-Rec
export PYTHONPATH=$REPO/HG-Rec:$PYTHONPATH
export HF_HOME=/home/wlia0047/ar57_scratch/wenyu/hf_models

mkdir -p $REPO/logs/task204 $REPO/products/task204

DATA=$REPO/HG-Rec/dataset/Instruments/item_emb.parquet
GPU=1
SAVE_DIR=$REPO/products/task204/stage1_arm_A_c1_w22
LOG_FILE=$REPO/logs/task204/stage1_arm_A.log
rm -f $LOG_FILE

echo "[$(date)] === #204 臂 A: c=[1,1,1] + quant_loss_weight=2.2, GPU=$GPU, epochs=1000 ===" | tee -a $LOG_FILE

CUDA_VISIBLE_DEVICES=$GPU TRITON_CACHE_DIR=/home/wlia0047/.triton/cache_task204_arm_A \
python3 -u $REPO/HG-Rec/train_hrqvae.py \
    --data_path $DATA \
    --lr 1e-3 --epochs 1000 --batch_size 256 \
    --loss_type poincare --kmeans_init True --kmeans_iters 1000 \
    --sk_epsilons 0.0 0.0 0.0 --sk_iters 50 \
    --num_emb_list 64 128 256 --e_dim 32 \
    --beta 0.5 --layers 512 256 128 64 \
    --curvatures 1.0,1.0,1.0 \
    --quant_loss_weight 2.2 \
    --save_limit 5 \
    --device cuda:0 \
    --ckpt_dir $SAVE_DIR \
    > $LOG_FILE 2>&1 &
TRAIN_PID=$!
echo $TRAIN_PID > $REPO/products/task204/_TRAINING_PID_arm_A
echo "[$(date)] Arm A launched PID=$TRAIN_PID" | tee -a $LOG_FILE
echo "[$(date)] 监控: tail -f $LOG_FILE" | tee -a $LOG_FILE