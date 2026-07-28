#!/usr/bin/env bash
# Task #234 E2-b Stage 1 — NormCap + c_k spread + (rho, theta), alpha=0.6 (高分化对照)
# GPU: 3 (per user 2026-07-26 编排)
set -euo pipefail
source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh
conda activate /home/wlia0047/ar57_scratch/wenyu/grid_toys
cd /home/wlia0047/ar57/wenyu/GeneRec/HG-Rec

TS=$(date +%Y%m%d_%H%M%S)
SAVE_DIR=/home/wlia0047/ar57/wenyu/GeneRec/products/task234/e2b_spread_rhotheta_a06
LOG_DIR=/home/wlia0047/ar57/wenyu/GeneRec/logs/task234
LOG_FILE=$LOG_DIR/e2b_stage1_${TS}.log
PID_FILE=$SAVE_DIR/_TRAINING_PID

mkdir -p "$SAVE_DIR" "$LOG_DIR"
echo "[task234] launching E2-b Stage 1 (alpha=0.6) on cuda:3 ..."

python3 -u train_hrqvae.py \
    --data_path /home/wlia0047/ar57/wenyu/GeneRec/HG-Rec/dataset/Instruments/item_emb.parquet \
    --lr 1e-3 --epochs 200 --batch_size 1024 \
    --loss_type poincare --kmeans_init True --kmeans_iters 1000 \
    --sk_epsilons 0.0 0.0 0.0 --sk_iters 50 \
    --num_emb_list 64 128 256 --e_dim 36 \
    --beta 0.5 --layers 512 256 128 36 \
    --product_manifold --angular_dim 4 --radial_dim 32 \
    --use_normcap --normcap_target 0.4 \
    --assignment_mode per_codeword_kappa \
    --c_k_min 0.5 --c_k_max 5.0 --c_k_seed 42 \
    --c_k_update_mode spread --c_k_alpha 0.6 \
    --c_k_clamp_min 0.5 --c_k_clamp_max 2.0 --c_k_dead_thr 20 \
    --distance_mode rho_theta \
    --save_limit 5 --device cuda:3 \
    --ckpt_dir "$SAVE_DIR" > "$LOG_FILE" 2>&1 &

TRAIN_PID=$!
echo "$TRAIN_PID" > "$PID_FILE"
echo "[task234] E2-b PID=$TRAIN_PID, log=$LOG_FILE"
