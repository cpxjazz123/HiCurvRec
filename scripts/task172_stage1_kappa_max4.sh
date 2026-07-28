#!/bin/bash
# Task #172 Stage 1 — κ-Stereographic + κ_max=4.0 (比 #164/169/170/171 的 2.0 翻倍)
# 假设: 给 κ 训练空间更大, 让模型能探索更深双曲 (κ → -4 接近半个负空间 ball 半径)
# GPU 3 (R7: 完全空闲)

set -e

source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh
conda activate /home/wlia0047/ar57_scratch/wenyu/grid_toys

cd /home/wlia0047/ar57/wenyu/GeneRec

LOG_DIR=/home/wlia0047/ar57/wenyu/GeneRec/logs/task172
mkdir -p $LOG_DIR
TS=$(date +%b-%d-%Y_%H-%M-%S | sed 's/^.../\L&/')
LOG_FILE=$LOG_DIR/stage1_phase_ab_${TS}.log

PROD_DIR=/home/wlia0047/ar57/wenyu/GeneRec/products/task172/phase_b_kappa_max4/${TS}
mkdir -p $PROD_DIR

export TRITON_CACHE_DIR=/home/wlia0047/.triton/cache_task172_s1
mkdir -p "$TRITON_CACHE_DIR"

echo "===== [Task #172 Stage 1] κ-Stereo + κ_max=4.0 Phase A/B launched at $(date) =====" | tee $LOG_FILE
echo "Recipe: κ-Stereographic + θ_init=[0,0,0] + Phase A 100ep frozen + Phase B 100ep lr_theta=1e-5 + κ_max=4.0" | tee -a $LOG_FILE
echo "GPU 3 (CUDA_VISIBLE_DEVICES=3, completely idle)" | tee -a $LOG_FILE
echo "vs #164/169/170/171 (κ_max=2.0): util 100%, κ_m ≈ -0.09 to -0.10" | tee -a $LOG_FILE
echo "Target: test R@10 > 0.1058 (Stop hook 条件)" | tee -a $LOG_FILE
echo "Output: $PROD_DIR" | tee -a $LOG_FILE

# 关键改动: --kappa_max 4.0 (vs 默认 2.0)
CUDA_VISIBLE_DEVICES=3 TRITON_CACHE_DIR="$TRITON_CACHE_DIR" python3 /home/wlia0047/ar57/wenyu/GeneRec/scripts/task89_stage1_train_rqvae.py \
    --M 1 \
    --epochs 200 \
    --batch_size 256 \
    --lr 1e-3 \
    --lr_theta 1e-5 \
    --theta_init_list 0.0 0.0 0.0 \
    --kappa_max 4.0 \
    --kappa_freeze_epochs 100 \
    --seed 42 \
    --num_emb_list 32 64 256 \
    --e_dim 32 \
    --layers 512 256 128 \
    --loss_type poincare \
    --beta 0.5 \
    --quant_loss_weight 1.0 \
    --sk_epsilons 0.0 0.0 0.0 \
    --sk_iters 50 \
    --dead_code_reset_every 0 \
    --kmeans_init \
    --kmeans_iters 1000 \
    --data_path /home/wlia0047/ar57/wenyu/GeneRec/HG-Rec/dataset/Instruments/item_emb.parquet \
    --device cuda:0 \
    --ckpt_dir $PROD_DIR \
    --kappa_log_path $PROD_DIR/kappa_history.json \
    --log_interval 20 \
    --save_every 50 \
    2>&1 | tee -a $LOG_FILE

echo "===== [Task #172 Stage 1] exit code: ${PIPESTATUS[0]} at $(date) =====" | tee -a $LOG_FILE
echo "Stage 1 complete. Best ckpt: $PROD_DIR/best_loss_model.pth" | tee -a $LOG_FILE
ls -la $PROD_DIR/ | tee -a $LOG_FILE