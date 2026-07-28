#!/bin/bash
# Task #164 Phase A/B κ-Stereographic 修复 — Stage 1 训练
# Phase A: kappa_freeze_epochs=100 (κ frozen at 0 for first 100 epochs)
# Phase B: 100 ep unfrozen with lr_theta=1e-5 (κ slowly learnable, codebook ahead)
# 同一 P0 Recipe (θ_init=[0,0,0], sk=0, β=0.5) + Phase A/B intervention
# 目标: util L0/L1/L2 ≥ 80% + κ_m 缓慢微调不坍塌

set -e

source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh
conda activate /home/wlia0047/ar57_scratch/wenyu/grid_toys

cd /home/wlia0047/ar57/wenyu/GeneRec

LOG_DIR=/home/wlia0047/ar57/wenyu/GeneRec/logs/task164
mkdir -p $LOG_DIR
TS=$(date +%b-%d-%Y_%H-%M-%S | sed 's/^.../\L&/')
LOG_FILE=$LOG_DIR/phase_b_train_${TS}.log

PROD_DIR=/home/wlia0047/ar57/wenyu/GeneRec/products/task164/phase_b_kappa_decouple/${TS}
mkdir -p $PROD_DIR
PID_FILE=$PROD_DIR/_TRAINING_PID

echo "===== [Task #164 Phase B κ-decouple] launched at $(date) =====" | tee $LOG_FILE
echo "Recipe: κ-Stereographic + θ_init=[0,0,0] + Phase A 100ep frozen + Phase B 100ep lr_theta=1e-5" | tee -a $LOG_FILE
echo "GPU 0 (CUDA_VISIBLE_DEVICES=0, completely idle)" | tee -a $LOG_FILE
echo "vs P0 (无 Phase A/B, θ_init=[0,0,0]): κ 全跑到 -1.99, util 0.39%~3.12%" | tee -a $LOG_FILE
echo "vs #144 (Stage 1 util 100%, R@10=0.0973): 同 Phase A/B 机制, 旧 3-branch 公式" | tee -a $LOG_FILE
echo "Target (vs #88 baseline 0.1051): R@10 > 0.1051 (或 > 0.1058 phonism)" | tee -a $LOG_FILE
echo "Output: $PROD_DIR" | tee -a $LOG_FILE

CUDA_VISIBLE_DEVICES=0 python3 /home/wlia0047/ar57/wenyu/GeneRec/scripts/task89_stage1_train_rqvae.py \
    --M 1 \
    --epochs 200 \
    --batch_size 256 \
    --lr 1e-3 \
    --lr_theta 1e-5 \
    --theta_init_list 0.0 0.0 0.0 \
    --kappa_max 2.0 \
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

echo "===== [Task #164 Phase B κ-decouple] completed at $(date) =====" | tee -a $LOG_FILE
rm -f $PID_FILE
