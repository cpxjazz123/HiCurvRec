#!/usr/bin/env bash
# Phase 4 — 细粒度 w_angular 扫描: {1, 2, 3, 5} 找 cos_std ∈ [0.30, 0.40] Goldilocks
# v6 recipe (β=0.5, bs=256, w_div=100, ang_dim=2, 50 epoch), 仅 w_angular 变化
# Usage: bash m_arm_step3_wangular_sweep.sh <w_angular> <gpu_id>

set -e
REPO=/home/wlia0047/ar57/wenyu/GeneRec
W_ANG=$1
GPU=$2

if [ -z "$W_ANG" ] || [ -z "$GPU" ]; then
    echo "Usage: $0 <w_angular> <gpu_id>"
    echo "  w_angular ∈ {1, 2, 3, 5}"
    exit 1
fi

source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh
conda activate /home/wlia0047/ar57_scratch/wenyu/grid_toys

DATA=$REPO/HG-Rec/dataset/Instruments/item_emb.parquet
OUTDIR=$REPO/products/m_arm/m_radius_spread_step3_50ep_wdiv100_wangular_sweep_w${W_ANG}
LOG="${OUTDIR}/train.log"
mkdir -p "$OUTDIR"

COMMON="\
  --data_path $DATA \
  --lr 1e-3 --batch_size 256 --epochs 50 \
  --num_emb_list 64 128 256 --e_dim 34 \
  --loss_type poincare --kmeans_init True --kmeans_iters 1000 \
  --sk_epsilons 0.0 0.0 0.0 --sk_iters 50 \
  --beta 0.5 \
  --product_manifold --angular_dim 2 --radial_dim 32 \
  --r_target_list 2.0,2.7,3.4 --norm_target 1.0 1.35 1.70 --gamma_norm 5.0 \
  --alpha 1.0 --beta_radial 1.0 \
  --use_normcap --normcap_target 0.3 --normcap_euc_only \
  --w_ent 0.1 --w_div 100.0 --w_angular ${W_ANG} --target_angular_std 0.35 \
  --anti_collapse dead_revive \
  --r_spread_list 0.3 0.3 0.3 \
  --eval_step 2 --num_workers 0 \
  --device cuda:0 \
  --ckpt_dir $OUTDIR"

echo "==== Sweep: w_angular=${W_ANG} (GPU=${GPU}) — v6 recipe, 50 epochs ===="
echo "log → $LOG"
cd "$REPO"
LR_LOG_R=10.0 \
DISABLE_USAGE_KILL=1 \
CUDA_VISIBLE_DEVICES=$GPU \
TRITON_CACHE_DIR=/home/wlia0047/.triton/cache_wangular_sweep_w${W_ANG} \
mkdir -p /home/wlia0047/.triton/cache_wangular_sweep_w${W_ANG}
LR_LOG_R=10.0 \
DISABLE_USAGE_KILL=1 \
CUDA_VISIBLE_DEVICES=$GPU \
TRITON_CACHE_DIR=/home/wlia0047/.triton/cache_wangular_sweep_w${W_ANG} \
PYTHONUNBUFFERED=1 \
  python3 -u HG-Rec/train_hrqvae.py $COMMON \
    > "$LOG" 2>&1
echo "exit=$?"
