#!/bin/bash
# Task #176 Stage 1 — κ-Stereographic + β(x) sigmoid MCKG-style 位置依赖
# β(x) = β_base · sigmoid(-sign(κ)·||x||²·scale)
# 单 Phase 200 epoch, R12 best_loss_model.pth 强制存
# GPU 1 (Stage 3 #175 占 GPU 0)

set -e

source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh
conda activate /home/wlia0047/ar57_scratch/wenyu/grid_toys

cd /home/wlia0047/ar57/wenyu/GeneRec/HG-Rec

export PYTHONPATH=/home/wlia0047/ar57/wenyu/GeneRec/HG-Rec:$PYTHONPATH
export HF_HOME=/home/wlia0047/ar57_scratch/wenyu/hf_models
export TRITON_CACHE_DIR=/home/wlia0047/.triton/cache_task176_s1

LOG_DIR=/home/wlia0047/ar57/wenyu/GeneRec/logs/task176
TS=$(date +%b-%d-%Y_%H-%M-%S | sed 's/^.../\L&/')
LOG_FILE=$LOG_DIR/stage1_sigmoid_${TS}.log

CKPT_DIR=/home/wlia0047/ar57/wenyu/GeneRec/products/task176/posdep_sigmoid
KAPPA_LOG=/home/wlia0047/ar57/wenyu/GeneRec/products/task176/posdep_sigmoid/kappa_history.json
mkdir -p $LOG_DIR $CKPT_DIR

echo "===== [Task #176 Stage 1] sigmoid β(x) MCKG-style launched at $(date) =====" | tee $LOG_FILE

python3 -u /home/wlia0047/ar57/wenyu/GeneRec/scripts/task176_177_posdep_beta_stage1_train.py \
    --beta_mode sigmoid \
    --beta_base 1.0 --beta_scale 10.0 --beta_max 5.0 \
    --M 3 --kappa_max 2.0 \
    --epochs 200 --batch_size 256 --lr 1e-3 \
    --num_emb_list 32 64 256 1 --e_dim 32 --layers 512 256 128 \
    --loss_type poincare --beta 1.0 --quant_loss_weight 1.0 \
    --sk_epsilons 0.0 0.0 0.0 0.0 --sk_iters 50 \
    --kmeans_init --kmeans_iters 1000 \
    --dead_code_reset_every 20 --dead_code_reset_threshold 0.0 \
    --dead_code_replace_ratio 0.1 --seed 42 \
    --data_path /home/wlia0047/ar57/wenyu/GeneRec/HG-Rec/dataset/Instruments/item_emb.parquet \
    --device cuda:0 \
    --ckpt_dir $CKPT_DIR \
    --kappa_log_path $KAPPA_LOG \
    --log_interval 10 --save_every 50 \
    2>&1 | tee -a $LOG_FILE

EXIT_CODE=${PIPESTATUS[0]}
echo "===== [Task #176 Stage 1] exit code: $EXIT_CODE at $(date) =====" | tee -a $LOG_FILE
[ $EXIT_CODE -ne 0 ] && exit 1

echo "✅ Task #176 Stage 1 完成 (best ckpt: $CKPT_DIR/best_loss_model.pth)" | tee -a $LOG_FILE