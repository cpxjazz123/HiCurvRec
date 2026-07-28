#!/usr/bin/env bash
# Task #218 Stage 1 — Per-Codeword κ + spread + NormCap + simvq (完整 200 epochs)
# J-plan J0 配置跑满 (jplan J0 在 50 epoch 时被 kill, 需要 200 epochs 完整 ckpt)
# 配置: --assignment_mode per_codeword_kappa --c_k_min 0.5 --c_k_max 5.0 --c_k_update_mode spread
#        + --use_normcap + --anti_collapse simvq + --distance_mode rho_theta
# GPU: 0 (空闲)
set -e
REPO=/home/wlia0047/ar57/wenyu/GeneRec
source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh
conda activate /home/wlia0047/ar57_scratch/wenyu/grid_toys
cd $REPO/HG-Rec

export PYTHONPATH=$REPO/HG-Rec:$PYTHONPATH
export HF_HOME=/home/wlia0047/ar57_scratch/wenyu/hf_models

DATA=$REPO/HG-Rec/dataset/Instruments/item_emb.parquet
DEPTH=$REPO/HG-Rec/dataset/Instruments/category_depth.npy
GPU=0
SAVE_DIR=$REPO/products/task218/stage1_pck_spread
LOG_DIR=$REPO/logs/task218
LOG_FILE=$LOG_DIR/stage1_pck_spread_$(date +%Y%m%d_%H%M%S).log
mkdir -p "$SAVE_DIR" "$LOG_DIR"

echo "[$(date)] === Task #218 Stage 1: per_codeword_kappa c_k[0.5,5] spread + NormCap + simvq, GPU=$GPU ==="
echo "log: $LOG_FILE"

CUDA_VISIBLE_DEVICES=$GPU TRITON_CACHE_DIR=/home/wlia0047/.triton/cache_task218 \
python3 -u $REPO/HG-Rec/train_hrqvae.py \
    --data_path $DATA \
    --depth_npy_path $DEPTH \
    --lr 1e-3 --epochs 200 --batch_size 256 \
    --num_emb_list 64 128 256 --e_dim 32 --layers 4 3 2 \
    --product_manifold --angular_dim 4 --radial_dim 28 \
    --assignment_mode per_codeword_kappa \
    --c_k_min 0.5 --c_k_max 5.0 --c_k_seed 42 \
    --c_k_update_mode spread --c_k_alpha 0.3 \
    --c_k_clamp_min 0.5 --c_k_clamp_max 2.0 --c_k_dead_thr 20 \
    --distance_mode rho_theta \
    --beta 0.5 --quant_loss_weight 1.0 \
    --sk_epsilons 0.003 0.003 0.003 \
    --use_normcap --normcap_target 0.4 \
    --eval_step 5 --num_workers 0 \
    --loss_type poincare \
    --anti_collapse simvq \
    --w_anchor 0.0 \
    --device cuda:0 \
    --ckpt_dir "$SAVE_DIR" \
    > "$LOG_FILE" 2>&1 &
TRAIN_PID=$!
echo $TRAIN_PID > $SAVE_DIR/_TRAINING_PID
echo "[$(date)] Task #218 Stage 1 PID=$TRAIN_PID" 
