#!/bin/bash
# Task #206-rev2 Step 3: Stage 2 SID 推理, 对 4 个 ckpt (2 对 collision-aligned)
#
# Pair A1 (低 collision ~44%):
#   - 双曲 baseline epoch 19 (collision 42.81%)
#   - 欧式 ×4 β=1 epoch 4 (collision 44.47%)
# Pair A2 (高 collision ~67%):
#   - 双曲 baseline epoch 14 (collision 69.40%)
#   - 欧式 ×4 β=1 epoch 9 (collision 63.61%)
#
# GPU 3 给所有 (空闲), GPU 0/2 可能还有尾单

set -e
REPO=/home/wlia0047/ar57/wenyu/GeneRec
source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh
conda activate /home/wlia0047/ar57_scratch/wenyu/grid_toys
export PYTHONPATH=$REPO/HG-Rec:$PYTHONPATH

mkdir -p $REPO/logs/task206 $REPO/HG-Rec/dataset/Instruments

run_stage2() {
    local NAME=$1
    local CKPT=$2
    local OUT=$3
    local GPU=$4
    local LOG=$REPO/logs/task206/stage2_${NAME}.out

    echo "[$(date)] === Stage 2 $NAME: $CKPT → $OUT, GPU=$GPU ===" | tee $LOG
    CUDA_VISIBLE_DEVICES=$GPU TRITON_CACHE_DIR=/home/wlia0047/.triton/cache_task206d_${NAME} \
    python3 -u $REPO/scripts/task188_stage2_codebook.py \
        --ckpt_path $CKPT \
        --output_path $OUT \
        --device cuda:0 \
        > $LOG 2>&1
    echo "[$(date)] $NAME done."
}

run_stage2 "hyp_e19" \
    $REPO/products/task206/stage1_baseline_retrain/Jul-26-2026_14-47-13_beta_1.000_codebook_\[64,128,256\]_sk_0.000/epoch_19_collision_0.4281_model.pth \
    $REPO/HG-Rec/dataset/Instruments/Instruments_t5_rqvae_hyp_e19.npy \
    3

run_stage2 "euc_e4" \
    $REPO/products/task206/stage1_euclidean_fix_x4_beta1/Jul-26-2026_14-49-17_beta_1.000_codebook_\[64,128,256\]_sk_0.000/epoch_4_collision_0.4447_model.pth \
    $REPO/HG-Rec/dataset/Instruments/Instruments_t5_rqvae_euc_e4.npy \
    3

run_stage2 "hyp_e14" \
    $REPO/products/task206/stage1_baseline_retrain/Jul-26-2026_14-47-13_beta_1.000_codebook_\[64,128,256\]_sk_0.000/epoch_14_collision_0.6940_model.pth \
    $REPO/HG-Rec/dataset/Instruments/Instruments_t5_rqvae_hyp_e14.npy \
    3

run_stage2 "euc_e9" \
    $REPO/products/task206/stage1_euclidean_fix_x4_beta1/Jul-26-2026_14-49-17_beta_1.000_codebook_\[64,128,256\]_sk_0.000/epoch_9_collision_0.6361_model.pth \
    $REPO/HG-Rec/dataset/Instruments/Instruments_t5_rqvae_euc_e9.npy \
    3

echo "[$(date)] === ALL 4 Stage 2 done ==="
ls -la $REPO/HG-Rec/dataset/Instruments/Instruments_t5_rqvae_{hyp_e19,euc_e4,hyp_e14,euc_e9}.npy