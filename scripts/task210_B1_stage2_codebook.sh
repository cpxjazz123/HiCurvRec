#!/bin/bash
# Task #210 Phase B B1 Stage 2 — SID inference (Sinkhorn + dedup, 30 轮)
# Input: products/task210/hrqvae_B1/Jul-26-2026_19-43-26_*/best_collision_model.pth (collision 0.0963)
# Output: HG-Rec/dataset/Instruments/Instruments_t5_rqvae_task210_B1.npy
# GPU 1 (R7: A3 eval 在 GPU 0, 1/2/3 空闲)
set -e

REPO=/home/wlia0047/ar57/wenyu/GeneRec
source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh
conda activate /home/wlia0047/ar57_scratch/wenyu/grid_toys
cd $REPO/HG-Rec

export PYTHONPATH=$REPO/HG-Rec:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=1
export HF_HOME=/home/wlia0047/ar57_scratch/wenyu/hf_models
export TRITON_CACHE_DIR=/home/wlia0047/.triton/cache_task210_B1_s2
mkdir -p $TRITON_CACHE_DIR

mkdir -p $REPO/logs/task210

BEST_CKPT=$REPO/products/task210/hrqvae_B1/Jul-26-2026_19-43-26_beta_1.000_codebook_[64,128,256]_sk_0.000/best_collision_model.pth
OUTPUT_PATH=$REPO/HG-Rec/dataset/Instruments/Instruments_t5_rqvae_task210_B1.npy

if [ ! -f "$BEST_CKPT" ]; then
    echo "❌ B1 best_collision_model.pth MISSING: $BEST_CKPT"
    exit 1
fi
echo "✅ B1 best ckpt: $BEST_CKPT"

LOG_FILE=$REPO/logs/task210/B1_stage2_codebook_$(date +%b-%d-%Y_%H-%M-%S | sed 's/^.../\L&/').log

echo "===== [Task #210 B1 Stage 2] SID inference launched at $(date) =====" | tee $LOG_FILE
echo "Best ckpt: $BEST_CKPT" | tee -a $LOG_FILE
echo "Output: $OUTPUT_PATH" | tee -a $LOG_FILE

python3 -u $REPO/scripts/task210_B1_stage2_codebook.py \
    --ckpt_path $BEST_CKPT \
    --output_path $OUTPUT_PATH \
    --device cuda:0 \
    --max_sinkhorn_iters 30 \
    2>&1 | tee -a $LOG_FILE

EXIT_CODE=${PIPESTATUS[0]}
echo "===== [Task #210 B1 Stage 2] exit code: $EXIT_CODE at $(date) =====" | tee -a $LOG_FILE

if [ $EXIT_CODE -ne 0 ] || [ ! -f "$OUTPUT_PATH" ]; then
    echo "❌ B1 Stage 2 FAILED" | tee -a $LOG_FILE
    exit 1
fi

echo "✅ Task #210 B1 Stage 2 SID codebook 生成成功" | tee -a $LOG_FILE
ls -la $OUTPUT_PATH | tee -a $LOG_FILE
ls -la ${OUTPUT_PATH%.npy}_diagnostic.json | tee -a $LOG_FILE
cat ${OUTPUT_PATH%.npy}_diagnostic.json | tee -a $LOG_FILE