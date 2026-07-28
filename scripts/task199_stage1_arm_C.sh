#!/bin/bash
# Task #199 Stage 1 臂 C — 逐层可学习 c_ℓ, exp(θ_ℓ) 参数化
#
# 用户 2026-07-26: 三层要达到 λ≈10 需要的 c 差 18 倍 (L0≈30, L1≈250, L2≈550).
# c_max = (5/r)² per layer: L0=6.25 (r=2), L1=3.43 (r=2.7), L2=2.16 (r=3.4).
# 训练时 θ 自由学, 自动 clamp 到 c_max.
# GPU 2

set -e
REPO=/home/wlia0047/ar57/wenyu/GeneRec
source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh
conda activate /home/wlia0047/ar57_scratch/wenyu/grid_toys
cd $REPO/HG-Rec
export PYTHONPATH=$REPO/HG-Rec:$PYTHONPATH
export HF_HOME=/home/wlia0047/ar57_scratch/wenyu/hf_models

mkdir -p $REPO/logs/task199 $REPO/products/task199

DATA=$REPO/HG-Rec/dataset/Instruments/item_emb.parquet
GPU=2
SAVE_DIR=$REPO/products/task199/stage1_arm_C_exp_per_layer
LOG_FILE=$REPO/logs/task199/stage1_arm_C.log
rm -f $LOG_FILE

echo "[$(date)] === #199 臂 C: exp_per_layer, GPU=$GPU, epochs=1000 ===" | tee -a $LOG_FILE

CUDA_VISIBLE_DEVICES=$GPU TRITON_CACHE_DIR=/home/wlia0047/.triton/cache_task199_arm_C \
python3 -u $REPO/HG-Rec/train_hrqvae.py \
    --data_path $DATA \
    --lr 1e-3 --epochs 1000 --batch_size 256 \
    --loss_type poincare --kmeans_init True --kmeans_iters 1000 \
    --sk_epsilons 0.0 0.0 0.0 --sk_iters 50 \
    --num_emb_list 64 128 256 --e_dim 32 \
    --beta 0.5 --layers 512 256 128 64 \
    --kappa_mode exp_per_layer --r_target_list 2.0,2.7,3.4 \
    --save_limit 5 \
    --device cuda:0 \
    --ckpt_dir $SAVE_DIR \
    > $LOG_FILE 2>&1 &
TRAIN_PID=$!
echo $TRAIN_PID > $REPO/products/task199/_TRAINING_PID_arm_C
echo "[$(date)] Arm C launched PID=$TRAIN_PID" | tee -a $LOG_FILE
echo "[$(date)] 监控: tail -f $LOG_FILE" | tee -a $LOG_FILE