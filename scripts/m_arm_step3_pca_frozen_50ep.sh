#!/usr/bin/env bash
# 方向 H: PCA 初始化 hyp codeword 方向 + 冻结, 只 log_r 可学. v6 recipe + w_angular=10 (cos_std 强制 pen).
# 用户 2026-07-27 假设: collision 爆炸是 w_angular 把码字推到残差向量密度低的地方, 钉死方向可缓解.

set -e
REPO=/home/wlia0047/ar57/wenyu/GeneRec
source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh
conda activate /home/wlia0047/ar57_scratch/wenyu/grid_toys

GPU="${1:-0}"
DATA=$REPO/HG-Rec/dataset/Instruments/item_emb.parquet
TS=$(date +%b-%d-%Y_%H-%M-%S | sed 's/^.../\L&/')
OUTDIR=$REPO/products/m_arm/m_radius_spread_step3_50ep_wdiv100_pca_frozen_${TS}
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
  --w_ent 0.1 --w_div 100.0 --w_angular 10.0 --target_angular_std 0.35 \
  --anti_collapse dead_revive \
  --r_spread_list 0.3 0.3 0.3 \
  --eval_step 2 --num_workers 0 \
  --device cuda:0 \
  --ckpt_dir $OUTDIR"

echo "==== 方向 H: PCA-init hyp codewords + freeze (GPU=$GPU) ===="
echo "outdir → $OUTDIR"
echo "log → $LOG"
cd "$REPO"
LR_LOG_R=10.0 \
DISABLE_USAGE_KILL=1 \
CUDA_VISIBLE_DEVICES=$GPU \
TRITON_CACHE_DIR=/home/wlia0047/.triton/cache_pca_frozen_${TS} \
PYTHONUNBUFFERED=1 \
  python3 -u scripts/m_arm_step3_pca_frozen.py $COMMON \
    > "$LOG" 2>&1
echo "exit=$?"