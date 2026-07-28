#!/usr/bin/env bash
# 用户 2026-07-27 第二步: per-codeword radius spread 跑满 50 epoch.
# 配置 = 第一步 (M2 + --r_spread_list 0.3 0.3 0.3 + LR_LOG_R=10.0).
# 每 5 epoch (跟 --eval_step 一致) 报告 r_std/min/max/utilization/collision_rate/NaN.
# 提前中止: (a) NaN → kill, (b) epoch 30 utilization < 20% → kill.

set -e
REPO=/home/wlia0047/ar57/wenyu/GeneRec
source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh
conda activate /home/wlia0047/ar57_scratch/wenyu/grid_toys

GPU="${1:-0}"
DATA=$REPO/HG-Rec/dataset/Instruments/item_emb.parquet
OUTDIR=$REPO/products/m_arm/m_radius_spread_step2_50ep
LOG="${OUTDIR}/train.log"
mkdir -p "$OUTDIR"

# 配置一行不改 (跟第一步完全一致, 只改 --epochs 50).
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
  --w_ent 0.1 --anti_collapse dead_revive \
  --r_spread_list 0.3 0.3 0.3 \
  --eval_step 5 --num_workers 0 \
  --device cuda:0 \
  --ckpt_dir $OUTDIR"

echo "==== Phase 0.5 第二步: 50 epoch radius spread 训练 (GPU=$GPU) ===="
echo "log → $LOG"
cd "$REPO"
LR_LOG_R=10.0 \
CUDA_VISIBLE_DEVICES=$GPU \
TRITON_CACHE_DIR=/home/wlia0047/.triton/cache_radius_spread_step2 \
PYTHONUNBUFFERED=1 \
  python3 -u HG-Rec/train_hrqvae.py $COMMON \
    > "$LOG" 2>&1
echo "exit=$?"