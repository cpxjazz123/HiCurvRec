#!/usr/bin/env bash
# Step 3 v3: 50 epoch EMA + radius spread + w_div=10.0 + kill 禁用.
# 配置 = Step 2 + EMA (utils.py) + 新增 --w_div 10.0 (强推方向多样)
# + div_loss 用 EMA 中心化后的 z_c 算 (避免 1e6 raw 精度丢失).
# DISABLE_USAGE_KILL=1 让 50 epoch 跑完 (即使 util 短期低也不 kill).

set -e
REPO=/home/wlia0047/ar57/wenyu/GeneRec
source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh
conda activate /home/wlia0047/ar57_scratch/wenyu/grid_toys

GPU="${1:-0}"
DATA=$REPO/HG-Rec/dataset/Instruments/item_emb.parquet
OUTDIR=$REPO/products/m_arm/m_radius_spread_step3_50ep_wdiv10
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
  --w_ent 0.1 --w_div 10.0 --anti_collapse dead_revive \
  --r_spread_list 0.3 0.3 0.3 \
  --eval_step 5 --num_workers 0 \
  --device cuda:0 \
  --ckpt_dir $OUTDIR"

echo "==== Step 3 v3: w_div=10.0 + kill 禁用 (GPU=$GPU) ===="
echo "log → $LOG"
cd "$REPO"
LR_LOG_R=10.0 \
DISABLE_USAGE_KILL=1 \
CUDA_VISIBLE_DEVICES=$GPU \
TRITON_CACHE_DIR=/home/wlia0047/.triton/cache_radius_spread_step3_wdiv10 \
PYTHONUNBUFFERED=1 \
  python3 -u HG-Rec/train_hrqvae.py $COMMON \
    > "$LOG" 2>&1
echo "exit=$?"