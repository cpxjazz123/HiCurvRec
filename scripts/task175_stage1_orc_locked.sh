#!/bin/bash
# Task #175 Stage 1 — κ-Stereographic + κ LOCKED at Ollivier ORC 实测值
# Per-layer per-component κ = [-0.653, -0.829, -0.840] (G0 attr / G1 interact / G2 cooccur)
# M=3, e_dim=32 → 11+11+10
# 200 epoch single Phase, R12 best_loss_model.pth force-save

set -e

source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh
conda activate /home/wlia0047/ar57_scratch/wenyu/grid_toys

cd /home/wlia0047/ar57/wenyu/GeneRec/HG-Rec

export PYTHONPATH=/home/wlia0047/ar57/wenyu/GeneRec/HG-Rec:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export HF_HOME=/home/wlia0047/ar57_scratch/wenyu/hf_models
export TRITON_CACHE_DIR=/home/wlia0047/.triton/cache_task175_s1

LOG_DIR=/home/wlia0047/ar57/wenyu/GeneRec/logs/task175
TS=$(date +%b-%d-%Y_%H-%M-%S | sed 's/^.../\L&/')
LOG_FILE=$LOG_DIR/stage1_orc_locked_${TS}.log

CKPT_DIR=/home/wlia0047/ar57/wenyu/GeneRec/products/task175/orc_locked
KAPPA_LOG=/home/wlia0047/ar57/wenyu/GeneRec/products/task175/orc_locked/kappa_history.json
mkdir -p $LOG_DIR $CKPT_DIR

echo "===== [Task #175 Stage 1] ORC LOCKED κ launched at $(date) =====" | tee $LOG_FILE

python3 -u /home/wlia0047/ar57/wenyu/GeneRec/scripts/task175_orc_locked_stage1_train.py \
    --kappa_orc_values -0.653 -0.829 -0.840 \
    --M 3 \
    --epochs 200 \
    --batch_size 256 \
    --lr 1e-3 \
    --num_emb_list 32 64 256 1 \
    --e_dim 32 \
    --layers 512 256 128 \
    --loss_type poincare \
    --beta 1.0 \
    --quant_loss_weight 1.0 \
    --sk_epsilons 0.0 0.0 0.0 0.0 \
    --sk_iters 50 \
    --kmeans_init \
    --kmeans_iters 1000 \
    --dead_code_reset_every 20 \
    --dead_code_reset_threshold 0.0 \
    --dead_code_replace_ratio 0.1 \
    --seed 42 \
    --data_path /home/wlia0047/ar57/wenyu/GeneRec/HG-Rec/dataset/Instruments/item_emb.parquet \
    --device cuda:0 \
    --ckpt_dir $CKPT_DIR \
    --kappa_log_path $KAPPA_LOG \
    --log_interval 10 \
    --save_every 50 \
    2>&1 | tee -a $LOG_FILE

EXIT_CODE=${PIPESTATUS[0]}
echo "===== [Task #175 Stage 1] exit code: $EXIT_CODE at $(date) =====" | tee -a $LOG_FILE
[ $EXIT_CODE -ne 0 ] && exit 1

echo "✅ Task #175 Stage 1 完成 (best ckpt: $CKPT_DIR/best_loss_model.pth)" | tee -a $LOG_FILE