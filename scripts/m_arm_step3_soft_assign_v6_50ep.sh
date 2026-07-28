#!/usr/bin/env bash
# 方向 J1: v6 recipe MINUS w_angular/target_angular_std + 软分配 wrapper.
# 目的: 不强行 cos_std 推送, 看软分配训练动态能否天然长出 cos_std>0.3 (避免 w_angular 强拉的副作用).
# 风险: 码字互相拉近 → utilization 崩溃.

set -e
REPO=/home/wlia0047/ar57/wenyu/GeneRec
source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh
conda activate /home/wlia0047/ar57_scratch/wenyu/grid_toys

GPU="${1:-0}"
DATA=$REPO/HG-Rec/dataset/Instruments/item_emb.parquet
TS=$(date +%b-%d-%Y-%H-%M-%S | tr '[:upper:]' '[:lower:]')
OUTDIR=$REPO/products/m_arm/m_soft_assign_v6_50ep_${TS}
LOG="${OUTDIR}/train.log"
mkdir -p "$OUTDIR"

# v6 recipe - w_angular/target_angular_std (测试软分配动态本身)
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
  --w_ent 0.1 --w_div 100.0 \
  --anti_collapse dead_revive \
  --r_spread_list 0.3 0.3 0.3 \
  --eval_step 2 --num_workers 0 \
  --device cuda:0 \
  --ckpt_dir $OUTDIR"

echo "==== 方向 J1 v6 50 epoch 软分配 (τ=1.0→0.05) ===="
echo "log → $LOG"
echo "outdir → $OUTDIR"
cd "$REPO"
LR_LOG_R=10.0 \
DISABLE_USAGE_KILL=1 \
CUDA_VISIBLE_DEVICES=$GPU \
TRITON_CACHE_DIR=/home/wlia0047/.triton/cache_soft_assign_v6 \
PYTHONUNBUFFERED=1 \
  python3 -u scripts/m_arm_step3_soft_assign.py $COMMON \
    > "$LOG" 2>&1
echo "exit=$?"
