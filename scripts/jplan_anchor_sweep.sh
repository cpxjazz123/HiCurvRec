#!/usr/bin/env bash
# R11.3 自主决策: anchor 0.1 压垮学习, 扫小档 {0.001, 0.01, 0.05}.
# 4 GPU 并行 (J0 重跑做对照, J1/J2/J3 anchor 微调).
# 仍然全 simvq + NormCap + per_codeword_kappa.

set -e
cd /home/wlia0047/ar57/wenyu/GeneRec
source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh
conda activate /home/wlia0047/ar57_scratch/wenyu/grid_toys

DATA="/home/wlia0047/ar57/wenyu/GeneRec/HG-Rec/dataset/Instruments/item_emb.parquet"
DEPTH="/home/wlia0047/ar57/wenyu/GeneRec/HG-Rec/dataset/Instruments/category_depth.npy"

PROD="/home/wlia0047/ar57/wenyu/GeneRec/products/jplan"
mkdir -p $PROD

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
  local WANCHOR=$1
  local GPU=$2
  local OUTDIR="${PROD}/jsweep_w${WANCHOR}_cuda${GPU}"
  echo "==== anchor sweep: w_anchor=${WANCHOR} on cuda:${GPU} ===="
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

case "${1:-all}" in
  w001)
    run_arm 0.001 0
    ;;
  w01)
    run_arm 0.01 1
    ;;
  w05)
    run_arm 0.05 2
    ;;
  all)
    bash $0 w001 &
    bash $0 w01 &
    bash $0 w05 &
    wait
    ;;
  *)
    echo "Usage: $0 {w001|w01|w05|all}"
    exit 1
    ;;
esac
