#!/usr/bin/env bash
# Phase 1 (J-plan): 类别锚定 × 双曲 4 臂 + w_anchor sweep.
# Stage 1 only (50 epoch). 全部带 --anti_collapse simvq + NormCap + per_codeword_kappa.

set -e
cd /home/wlia0047/ar57/wenyu/GeneRec
source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh
conda activate /home/wlia0047/ar57_scratch/wenyu/grid_toys

DATA="/home/wlia0047/ar57/wenyu/GeneRec/HG-Rec/dataset/Instruments/item_emb.parquet"
DEPTH="/home/wlia0047/ar57/wenyu/GeneRec/HG-Rec/dataset/Instruments/category_depth.npy"

PROD="/home/wlia0047/ar57/wenyu/GeneRec/products/jplan"
mkdir -p $PROD

# 通用 recipe (E1 + simvq + per_codeword_kappa)
COMMON="\
  --data_path $DATA \
  --depth_npy_path $DEPTH \
  --lr 1e-3 --batch_size 256 --epochs 50 \
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
  --anti_collapse simvq"

run_arm() {
  local ARM=$1        # 0, 1, 2, 3
  local WANCHOR=$2    # 0.0, 0.1, 1.0
  local GPU=$3        # 0..3
  local OUTDIR="${PROD}/j${ARM}_w${WANCHOR}_cuda${GPU}"
  echo "==== J-plan arm: J${ARM} w_anchor=${WANCHOR} on cuda:${GPU} ===="
  echo "  output: $OUTDIR"
  echo "  start:  $(date)"
  mkdir -p "$OUTDIR"
  python HG-Rec/train_hrqvae.py $COMMON \
    --w_anchor $WANCHOR \
    --rho_min 1.5 --rho_max 2.9 \
    --device cuda:${GPU} \
    --ckpt_dir "$OUTDIR" 2>&1 | tee "${OUTDIR}/train.log"
  echo "  end:    $(date)"
}

case "${1:-phase1a}" in
  j0)
    run_arm 0 0.0 0
    ;;
  j1_01)
    run_arm 1 0.1 1
    ;;
  j1_10)
    run_arm 1 1.0 1
    ;;
  j2_01)
    run_arm 2 0.1 2
    ;;
  j2_10)
    run_arm 2 1.0 2
    ;;
  j3_01)
    run_arm 3 0.1 3
    ;;
  j3_10)
    run_arm 3 1.0 3
    ;;
  phase1a)  # J0 baseline + J1/J2/J3 w=0.1, 4 卡并行
    bash $0 j0 &
    bash $0 j1_01 &
    bash $0 j2_01 &
    bash $0 j3_01 &
    wait
    ;;
  phase1b)  # w=1.0 扫, 3 卡并行 (GPU 0 在 phase1a 后释放)
    bash $0 j1_10 &
    bash $0 j2_10 &
    bash $0 j3_10 &
    wait
    ;;
  all)
    bash $0 phase1a
    bash $0 phase1b
    ;;
  *)
    echo "Usage: $0 {j0|j1_01|j1_10|j2_01|j2_10|j3_01|j3_10|phase1a|phase1b|all}"
    exit 1
    ;;
esac
