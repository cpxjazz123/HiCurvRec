#!/usr/bin/env bash
# Task #233 — 方向 I c=10 ep1 Stage 2 SID 推断 (GPU 0)
# ckpt: products/m_arm/m_radius_spread_step3_50ep_wdiv100_c10_jul-27-2026_*/best_collision_model.pth
# metric: ep1 collision=51.41% (4558 collision groups stuck — Sinkhorn 不收敛, 短期早退即可).

set -e
REPO=/home/wlia0047/ar57/wenyu/GeneRec
source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh
conda activate /home/wlia0047/ar57_scratch/wenyu/grid_toys
cd $REPO/HG-Rec

export PYTHONPATH=$REPO/HG-Rec:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=0
export HF_HOME=/home/wlia0047/ar57_scratch/wenyu/hf_models
export TRITON_CACHE_DIR=/home/wlia0047/.triton/cache_task233_c10_s2
mkdir -p $TRITON_CACHE_DIR
mkdir -p $REPO/logs/task233

# 上轮 glob bug: jul-27-2026-* (-) 实际是 jul-27-2026_16-18-09 (_), glob 不匹配. 改 hardcode.
BEST_CKPT="/home/wlia0047/ar57/wenyu/GeneRec/products/m_arm/m_radius_spread_step3_50ep_wdiv100_c10_jul-27-2026_16-18-09/Jul-27-2026_16-18-16_beta_0.500_codebook_[64,128,256]_sk_0.000/best_collision_model.pth"
OUTPUT_PATH=$REPO/HG-Rec/dataset/Instruments/Instruments_t5_rqvae_task233_c10_ep1.npy

if [ ! -f "$BEST_CKPT" ]; then
    echo "❌ c=10 ep1 best_collision_model.pth MISSING"
    exit 1
fi
echo "✅ ckpt: $BEST_CKPT"
echo "✅ output: $OUTPUT_PATH"

LOG_FILE=$REPO/logs/task233/c10_ep1_stage2_$(date +%b-%d-%Y_%H-%M-%S | sed 's/^.../\L&/').log

echo "===== [Task #233 c=10 ep1 Stage 2] launched at $(date) =====" | tee $LOG_FILE

# max_sinkhorn_iters=5 (上轮 30 iter stuck at 1658 groups, Sinkhorn 不收敛, 5 已足)
python3 -u $REPO/scripts/task210_B1_stage2_codebook.py \
    --ckpt_path $BEST_CKPT \
    --output_path $OUTPUT_PATH \
    --device cuda:0 \
    --max_sinkhorn_iters 5 \
    2>&1 | tee -a $LOG_FILE

EXIT=${PIPESTATUS[0]}
echo "===== Task #233 c=10 ep1 Stage 2 exit=$EXIT at $(date) =====" | tee -a $LOG_FILE

if [ $EXIT -ne 0 ] || [ ! -f "$OUTPUT_PATH" ]; then
    echo "❌ c=10 ep1 Stage 2 FAILED" | tee -a $LOG_FILE
    exit 1
fi

echo "✅ Task #233 c=10 ep1 Stage 2 SID saved" | tee -a $LOG_FILE
ls -la $OUTPUT_PATH ${OUTPUT_PATH%.npy}_diagnostic.json | tee -a $LOG_FILE