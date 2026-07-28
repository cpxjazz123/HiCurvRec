#!/bin/bash
# Task #162 Stage 1 — paper-faithful HG-Rec free-curv per-layer κ
# 用户 2026-07-24 22:50 提议: 论文版本 HG-Rec + 三层 κ 随机 init + 可学习
# 关键参数: β=0.5 (paper Table 6), codebook=[32,64,256] (paper Table 6),
#           sk_epsilons=[0,0,0] (paper Table 6), θ_init per-layer random
# 跟 Task #89 A 臂差异: β 0.25 → 0.5, codebook [64,128,256] → [32,64,256], sk 0.5 → 0
# 跟 Task #137 v2 κ=0.1 差异: paper-faithful β=0.5 + per-layer random θ_init (vs κ=0.1)
# 输出: products/task162/rqvae_free_curv/<ts>/best_ckpt.pth + κ_history.json

set -e

source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh
conda activate /home/wlia0047/ar57_scratch/wenyu/grid_toys

cd /home/wlia0047/ar57/wenyu/GeneRec

LOG_DIR=/home/wlia0047/ar57/wenyu/GeneRec/logs/task162
TS=$(date +%b-%d-%Y_%H-%M-%S | sed 's/^.../\L&/')
LOG_FILE=$LOG_DIR/stage1_train_${TS}.log

PROD_DIR=/home/wlia0047/ar57/wenyu/GeneRec/products/task162/rqvae_free_curv/${TS}
mkdir -p $LOG_DIR $PROD_DIR

PID_FILE=$PROD_DIR/_TRAINING_PID

echo "===== [Task #162 Stage 1] paper-faithful HG-Rec free-curv launched at $(date) =====" | tee $LOG_FILE
echo "Recipe: paper-faithful β=0.5 + [32,64,256] + sk=0 + per-layer θ random init + learnable κ" | tee -a $LOG_FILE
echo "GPU 3 (CUDA_VISIBLE_DEVICES=3)" | tee -a $LOG_FILE
echo "θ_init per-layer (seed=42 uniform[-1,1]): L0=0.0837, L1=-0.285, L2=-0.135 → κ_init [0.167, -0.555, -0.268] (κ_max=2.0, θ small random)" | tee -a $LOG_FILE
echo "Output: $PROD_DIR" | tee -a $LOG_FILE
echo "Stage 1 fork: scripts/task89_stage1_train_rqvae.py" | tee -a $LOG_FILE

CUDA_VISIBLE_DEVICES=3 python3 /home/wlia0047/ar57/wenyu/GeneRec/scripts/task89_stage1_train_rqvae.py \
    --M 1 \
    --epochs 1000 \
    --batch_size 256 \
    --lr 1e-3 \
    --lr_theta 5e-3 \
    --theta_init_list 0.0837 -0.285 -0.135 \
    --kappa_max 2.0 \
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
    --save_every 100 \
    2>&1 | tee -a $LOG_FILE

echo "===== [Task #162 Stage 1] paper-faithful HG-Rec free-curv completed at $(date) =====" | tee -a $LOG_FILE
rm -f $PID_FILE