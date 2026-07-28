#!/bin/bash
# Task #199 Stage 1 臂 B — 单个可学习 c, exp(θ) 参数化 (全局共享)
#
# 用户 2026-07-26 根因诊断: κ_max 小 50 倍 (tanh(θ) ∈ [-2,2], 健康 λ∈[5,75] 需要 c∈[10,100]).
# 改 c = exp(θ).clamp(max=(5/r_median)²), 覆盖 [1, 400+], 梯度不饱和.
#
# 配置 (跟 #181 baseline 对齐):
#   epochs=1000, batch_size=256, beta=0.5, sk_epsilons=[0,0,0], e_dim=32, num_emb_list=[64,128,256]
#   lr=1e-3, kmeans_iters=1000, loss_type=poincare
# 变量: kappa_mode=exp_global (单 θ 共享 c, 三层都用同一个 c)
# GPU 1

set -e
REPO=/home/wlia0047/ar57/wenyu/GeneRec
source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh
conda activate /home/wlia0047/ar57_scratch/wenyu/grid_toys
cd $REPO/HG-Rec
export PYTHONPATH=$REPO/HG-Rec:$PYTHONPATH
export HF_HOME=/home/wlia0047/ar57_scratch/wenyu/hf_models

mkdir -p $REPO/logs/task199 $REPO/products/task199

DATA=$REPO/HG-Rec/dataset/Instruments/item_emb.parquet
GPU=1
SAVE_DIR=$REPO/products/task199/stage1_arm_B_exp_global
LOG_FILE=$REPO/logs/task199/stage1_arm_B.log
rm -f $LOG_FILE

echo "[$(date)] === #199 臂 B: exp_global, GPU=$GPU, epochs=1000 ===" | tee -a $LOG_FILE

CUDA_VISIBLE_DEVICES=$GPU TRITON_CACHE_DIR=/home/wlia0047/.triton/cache_task199_arm_B \
python3 -u $REPO/HG-Rec/train_hrqvae.py \
    --data_path $DATA \
    --lr 1e-3 --epochs 1000 --batch_size 256 \
    --loss_type poincare --kmeans_init True --kmeans_iters 1000 \
    --sk_epsilons 0.0 0.0 0.0 --sk_iters 50 \
    --num_emb_list 64 128 256 --e_dim 32 \
    --beta 0.5 --layers 512 256 128 64 \
    --kappa_mode exp_global --r_target_list 2.0,2.7,3.4 \
    --save_limit 5 \
    --device cuda:0 \
    --ckpt_dir $SAVE_DIR \
    > $LOG_FILE 2>&1 &
TRAIN_PID=$!
echo $TRAIN_PID > $REPO/products/task199/_TRAINING_PID_arm_B
echo "[$(date)] Arm B launched PID=$TRAIN_PID" | tee -a $LOG_FILE
echo "[$(date)] 监控: tail -f $LOG_FILE" | tee -a $LOG_FILE