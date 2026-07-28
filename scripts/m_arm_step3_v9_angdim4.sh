#!/usr/bin/env bash
# Step 3 v9: angular_dim 2D→4D, radial_dim 32D→30D. e_dim 维持 34 (encoder 不动).
# 其他 v8 配置: β=1.0 + bs=1024 + 100 epoch (比 200 早出结果).
# 5-cond knobs 全保 (LR_LOG_R, anti_collapse, w_angular, normcap, r_target, r_spread, γ_norm).
# 假设: 4D hyp 角带宽 √(4/64)≈0.25 (vs 2D 0.18 vs baseline 32D 0.71),
#       给 collision 一些呼吸空间, 看早期 trajectory 是否下降.

set -e
REPO=/home/wlia0047/ar57/wenyu/GeneRec
source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh
conda activate /home/wlia0047/ar57_scratch/wenyu/grid_toys

GPU="${1:-0}"
DATA=$REPO/HG-Rec/dataset/Instruments/item_emb.parquet
OUTDIR=$REPO/products/m_arm/m_radius_spread_step3_v9_angdim4
LOG="${OUTDIR}/train.log"
mkdir -p "$OUTDIR"

COMMON="\
  --data_path $DATA \
  --lr 1e-3 --batch_size 1024 --epochs 100 \
  --num_emb_list 64 128 256 --e_dim 34 \
  --loss_type poincare --kmeans_init True --kmeans_iters 1000 \
  --sk_epsilons 0.0 0.0 0.0 --sk_iters 50 \
  --beta 1.0 \
  --product_manifold --angular_dim 4 --radial_dim 30 \
  --r_target_list 2.0,2.7,3.4 --norm_target 1.0 1.35 1.70 --gamma_norm 5.0 \
  --alpha 1.0 --beta_radial 1.0 \
  --use_normcap --normcap_target 0.3 --normcap_euc_only \
  --w_ent 0.1 --w_angular 10.0 --target_angular_std 0.35 \
  --anti_collapse dead_revive \
  --r_spread_list 0.3 0.3 0.3 \
  --eval_step 5 --num_workers 0 \
  --device cuda:0 \
  --ckpt_dir $OUTDIR"

echo "==== Step 3 v9: angular_dim=4 + radial_dim=30, β=1.0, bs=1024, 100 epoch (GPU=$GPU) ===="
echo "log → $LOG"
cd "$REPO"
LR_LOG_R=10.0 \
DISABLE_USAGE_KILL=1 \
CUDA_VISIBLE_DEVICES=$GPU \
TRITON_CACHE_DIR=/home/wlia0047/.triton/cache_radius_spread_step3_v9 \
PYTHONUNBUFFERED=1 \
  python3 -u HG-Rec/train_hrqvae.py $COMMON \
    > "$LOG" 2>&1
echo "exit=$?"