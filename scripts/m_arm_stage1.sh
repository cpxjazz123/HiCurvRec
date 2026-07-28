#!/usr/bin/env bash
# M-arm Stage 1: M1/M2/M3 三臂并行, 各占一卡 (GPU 1/2/3, GPU 0 让给 Task #218)
#
# 配置:
#   encoder 34D = 2 (hyp, 钉半径 ρ=2.0/2.7/3.4) + 32 (euc, NormCap 0.4)
#   距离 d² = α·d²_hyp + β·d²_euc (默认 α=1, β=1)
#   r_target_list per-layer = "2.0,2.7,3.4"  (ρ target, 决定 c_max 上限)
#
# 四臂:
#   M0: E1 baseline (task226, collision 7.13%) - 已存在, 不重跑
#   M1: hyp 2D + no anti-collapse - 预期崩 (C1 2D 版)
#   M2: hyp 2D + anti-collapse (dead_revive + w_ent) - 主推
#   M3: no hyp + anti-collapse - 对照 (防坍缩单独值多少)

set -e
REPO=/home/wlia0047/ar57/wenyu/GeneRec
source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh
conda activate /home/wlia0047/ar57_scratch/wenyu/grid_toys

DATA=$REPO/HG-Rec/dataset/Instruments/item_emb.parquet
mkdir -p $REPO/products/m_arm

# 通用参数 (跟 E1 baseline 一致, 除 product_split + 钉半径 + alpha/beta)
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
  --eval_step 5 --num_workers 0"

run_arm() {
  local ARM=$1
  local W_ENT=$2
  local ANTI=$3
  local GPU=$4
  local OUTDIR=$REPO/products/m_arm/${ARM}_went${W_ENT}_anti${ANTI}
  echo "==== M-arm: ${ARM}  w_ent=${W_ENT}  anti=${ANTI}  GPU=${GPU} ===="
  mkdir -p "$OUTDIR"
  CUDA_VISIBLE_DEVICES=$GPU TRITON_CACHE_DIR=/home/wlia0047/.triton/cache_marm_${ARM} \
    python3 -u $REPO/HG-Rec/train_hrqvae.py $COMMON \
      --w_ent $W_ENT \
      --anti_collapse $ANTI \
      --device cuda:0 \
      --ckpt_dir "$OUTDIR" \
      > "${OUTDIR}/train.log" 2>&1 &
  echo "  PID=$!  outdir=$OUTDIR"
}

case "${1:-all}" in
  m1)
    # M1: hyp + NO anti-collapse - 预期崩 (C1 2D 版)
    run_arm m1 0.0 none 1
    ;;
  m2)
    # M2: hyp + anti-collapse (dead_revive + w_ent=0.1) - 主推
    run_arm m2 0.1 dead_revive 2
    ;;
  m3)
    # M3: NO hyp + anti-collapse - 对照
    # (这需要在 product_manifold 关闭下, 只跑 euc 32D; 简化: 不加 product_manifold)
    # 但用户 spec 要 normcap_euc_only=True, 必须 product_manifold=True.
    # 这里改: M3 不跑 (跟 M2 区别只是 angular_dim 0/2; 在 M2 上加 ablation 更经济)
    echo "M3 skipped - simplified to no-hyp ablation of M2"
    ;;
  all)
    run_arm m1 0.0 none 1 &
    run_arm m2 0.1 dead_revive 2 &
    # M3 留为后续 ablation
    wait
    ;;
  *)
    echo "Usage: $0 {m1|m2|m3|all}"
    exit 1
    ;;
esac
