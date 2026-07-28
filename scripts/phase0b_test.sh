#!/usr/bin/env bash
# Phase 0B: 防坍缩独立测试 (用户 2026-07-27 Phase 0B 方案).
#
# 判据: 利用率从 ~23% 回到 ≥ 80% ✅ / 仍 < 50% ❌ (停止 Phase 1).
#
# 跑 3 个独立试验 (同 D1-like 已知崩配置, 只换 anti_collapse):
#   baseline        → anti_collapse=none     (D1 对照)
#   + dead_revive   → anti_collapse=dead_revive
#   + simvq         → anti_collapse=simvq
#
# 每个 50 epoch, 单卡 GPU 0 (~15-20min/试验).
# 数据集: Musical_Instruments (R5 硬规则).

set -e
cd /home/wlia0047/ar57/wenyu/GeneRec
source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh
conda activate /home/wlia0047/ar57_scratch/wenyu/grid_toys

mkdir -p /home/wlia0047/ar57/wenyu/GeneRec/products/phase0b

DATA_PATH="/home/wlia0047/ar57/wenyu/GeneRec/HG-Rec/dataset/Instruments/item_emb.parquet"

# D1-like 已知崩配方: spread 模式 + 大码本 + 大 c_k 范围 + 无 NormCap.
# 这是历史上 A 系列触发坍缩的真实配方 (参 memory phase0-mode-collapse).
COMMON_FLAGS="\
  --data_path ${DATA_PATH} \
  --lr 1e-3 --batch_size 256 --epochs 50 \
  --num_emb_list 64 128 256 \
  --e_dim 32 --layers 4 3 2 \
  --product_manifold --angular_dim 4 --radial_dim 28 \
  --assignment_mode per_codeword_kappa \
  --c_k_min 0.5 --c_k_max 5.0 --c_k_seed 42 \
  --c_k_update_mode spread --c_k_alpha 0.3 \
  --c_k_clamp_min 0.5 --c_k_clamp_max 2.0 --c_k_dead_thr 20 \
  --distance_mode rho_theta \
  --beta 0.5 --quant_loss_weight 1.0 \
  --loss_mult_codebook 1.0 \
  --sk_epsilons 0.003 0.003 0.003 \
  --eval_step 5 --num_workers 0 \
  --device cuda:0 \
  --loss_type poincare"

# Anti-collapse 三种模式, 通过 --anti_collapse 切换.
run_trial() {
  local ANTI=$1
  local OUTDIR="/home/wlia0047/ar57/wenyu/GeneRec/products/phase0b/anti_${ANTI}"
  echo "============================================================"
  echo "Phase 0B  trial: anti_collapse=${ANTI}"
  echo "  output: ${OUTDIR}"
  echo "  start: $(date)"
  echo "============================================================"
  mkdir -p "${OUTDIR}"
  python HG-Rec/train_hrqvae.py ${COMMON_FLAGS} \
    --anti_collapse ${ANTI} \
    --ckpt_dir ${OUTDIR} 2>&1 | tee "${OUTDIR}/train.log"
  echo "  end:   $(date)"
}

case "${1:-all}" in
  none|baseline)
    run_trial none
    ;;
  dead|dead_revive)
    run_trial dead_revive
    ;;
  simvq)
    run_trial simvq
    ;;
  all)
    run_trial none
    run_trial dead_revive
    run_trial simvq
    ;;
  *)
    echo "Usage: $0 {none|dead_revive|simvq|all}"
    exit 1
    ;;
esac
