#!/usr/bin/env bash
# 方向 I: 放大 κ 弯曲效应 (v6 recipe + c=10 或 c=100).
# 用户 2026-07-27 假设: √c·ρ≈0.54 离"visible curvature effect"还远, 把 c 调大让
# 同样幅度的 radius 差异被放大成更大的 argmin 分歧. Task #201 theta_init=log(c) 已支持.

set -e
REPO=/home/wlia0047/ar57/wenyu/GeneRec
source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh
conda activate /home/wlia0047/ar57_scratch/wenyu/grid_toys

GPU="${1:-0}"
C_VAL="${2:-10}"
DATA=$REPO/HG-Rec/dataset/Instruments/item_emb.parquet
TS=$(date +%b-%d-%Y_%H-%M-%S | sed 's/^.../\L&/')
OUTDIR=$REPO/products/m_arm/m_radius_spread_step3_50ep_wdiv100_c${C_VAL}_${TS}
LOG="${OUTDIR}/train.log"
mkdir -p "$OUTDIR"

# theta_init = log(C_VAL). Task #199 exp(θ) κ 参数化 → c = exp(theta_init).
THETA_INIT=$(python3 -c "import math; print(math.log(${C_VAL}))")

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
  --w_ent 0.1 --w_div 100.0 --w_angular 10.0 --target_angular_std 0.35 \
  --anti_collapse dead_revive \
  --r_spread_list 0.3 0.3 0.3 \
  --eval_step 2 --num_workers 0 \
  --device cuda:0 \
  --ckpt_dir $OUTDIR \
  --kappa_mode exp_global --theta_init ${THETA_INIT}"

echo "==== 方向 I: c=${C_VAL} (theta_init=${THETA_INIT}) v6 recipe (GPU=$GPU) ===="
echo "outdir → $OUTDIR"
echo "log → $LOG"
cd "$REPO"
LR_LOG_R=10.0 \
DISABLE_USAGE_KILL=1 \
CUDA_VISIBLE_DEVICES=$GPU \
TRITON_CACHE_DIR=/home/wlia0047/.triton/cache_c${C_VAL}_${TS} \
PYTHONUNBUFFERED=1 \
  python3 -u HG-Rec/train_hrqvae.py $COMMON \
    > "$LOG" 2>&1
echo "exit=$?"