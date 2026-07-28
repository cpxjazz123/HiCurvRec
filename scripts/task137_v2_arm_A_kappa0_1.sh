#!/bin/bash
# Task #137 v2 A-arm retrain — κ_max=0.1 (vs 0.5 v1 坍缩)
# R137 fix + κ_max 更小 → 减少 sph 流形对 codebook 强制约束
# 短训练 200 epoch 因仅验证代码路径 (与 Task #137 main A 臂 1000 epoch 区分)
# GPU 0 (R7 — B/C 臂占 GPU 1, ETEGRec 占 GPU 2)

set -e

source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh
conda activate /home/wlia0047/ar57_scratch/wenyu/grid_toys

cd /home/wlia0047/ar57/wenyu/GeneRec

LOG_DIR=/home/wlia0047/ar57/wenyu/GeneRec/logs
PROD_DIR=/home/wlia0047/ar57/wenyu/GeneRec/products/task137_v2/train/arm_A_M1_kappa0.1
mkdir -p $LOG_DIR $PROD_DIR

TS=$(date +%b-%d-%Y_%H-%M-%S | sed 's/^.../\L&/')
LOG_FILE=$LOG_DIR/task137_v2_arm_A_kappa0.1_${TS}.log
PID_FILE=$PROD_DIR/_TRAINING_PID

echo "===== [Task #137 v2 A κ=0.1] launch at $(date) =====" | tee $LOG_FILE
echo "R137 fix + κ_max=0.1 (vs v1 κ_max=0.5 SID 坍缩)" | tee -a $LOG_FILE
echo "GPU 0 (R7 — B/C 臂 GPU 1, ETEGRec GPU 2)" | tee -a $LOG_FILE
echo "Output: $PROD_DIR" | tee -a $LOG_FILE
echo "短训练 200 epoch (仅验证 v2 codebook 不坍缩)" | tee -a $LOG_FILE

# R11.3 自决: κ_max=0.1 (vs v1 0.5) — 减小 sph 流形对 codebook 强制约束, 期望 SID 不再坍缩到 12 unique
# R11.3 自决: lr_theta=1e-3 (与 v1 一致)
# R11.3 自决: 200 epoch (vs v1 1000) — 短训练, 因 κ_max 更小, 200 epoch 已足够 convergence

CUDA_VISIBLE_DEVICES=0 python3 /home/wlia0047/ar57/wenyu/GeneRec/scripts/task89_stage1_train_rqvae.py \
    --M 1 \
    --epochs 200 \
    --batch_size 256 \
    --lr 1e-3 \
    --lr_theta 1e-3 \
    --theta_init 0.01 \
    --kappa_max 0.1 \
    --seed 42 \
    --num_emb_list 64 128 256 \
    --e_dim 32 \
    --layers 512 256 128 \
    --loss_type poincare \
    --beta 1.0 \
    --quant_loss_weight 1.0 \
    --sk_epsilons 0.0 0.0 0.0 \
    --sk_iters 50 \
    --kmeans_init \
    --kmeans_iters 1000 \
    --data_path /home/wlia0047/ar57/wenyu/GeneRec/HG-Rec/dataset/Instruments/item_emb.parquet \
    --device cuda:0 \
    --ckpt_dir $PROD_DIR \
    --kappa_log_path $PROD_DIR/kappa_history.json \
    --log_interval 10 \
    --save_every 50 \
    2>&1 | tee -a $LOG_FILE

echo "===== [Task #137 v2 A κ=0.1] done at $(date) =====" | tee -a $LOG_FILE
rm -f $PID_FILE