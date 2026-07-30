#!/bin/bash
# Task #194 / Issue #40 Gate 1 — Stage 2 ONLY (continuation)
# Stage 1 done. Stage 2 = 5 min, light inference. Run immediately on GPU 1.
# Stage 3 + Stage 4 will be launched separately when GPU 1 frees (after task327 Stage 3 finishes ~16:35 AEST).

set -uo pipefail

REPO=/home/wlia0047/ar57/wenyu/GeneRec
cd $REPO

LOG_DIR=$REPO/logs/task194_gate1
PRODUCTS_DIR=$REPO/products/task194/protocol_match_k064

# Stage 1 ckpt (confirmed exists after Stage 1 silent "failure" was actually just launcher glob path bug)
STAGE1_BEST=$(ls -t $PRODUCTS_DIR/*/best_collision_model.pth 2>/dev/null | head -1)
if [ -z "$STAGE1_BEST" ]; then
    echo "❌ Stage 1 best_collision_model.pth NOT FOUND in $PRODUCTS_DIR/*/" | tee -a $LOG_DIR/main.log
    exit 1
fi
echo "✅ Stage 1 best ckpt: $STAGE1_BEST" | tee -a $LOG_DIR/main.log

mkdir -p $LOG_DIR

export PYTHONPATH=$REPO/HG-Rec:$REPO/scripts:${PYTHONPATH:-}
export TRITON_CACHE_DIR=/home/wlia0047/.triton/cache_task194_gate1_k064
mkdir -p $TRITON_CACHE_DIR

source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh
conda activate /home/wlia0047/ar57_scratch/wenyu/genrec_env

# GPU 1: task327 Stage 3 Ep121/200 already, ~2.7h remaining. Stage 2 (5 min) shared OK (low impact, near completion).
# R7 spirit: this is continuation of Gate 1 which started on GPU 2 originally; switching to GPU 1 because task328 α=1.0 occupied GPU 2 at 12:58.
export CUDA_VISIBLE_DEVICES=1

# ---------------------------------------------------------------------------
# Stage 2 — Codebook inference (NO Sinkhorn, argmin only)
# ---------------------------------------------------------------------------
STAGE2_TS=$(date +%b-%d-%Y_%H-%M-%S | sed 's/^.../\L&/')
STAGE2_LOG=$LOG_DIR/stage2_${STAGE2_TS}.log
OUTPUT_NPY=$REPO/HG-Rec/dataset/Instruments/Instruments_t5_hrqvae_protocol_match_k064.npy

echo "===== [Issue #40 Gate 1 / K=64 stage2_only] launched at $(date) =====" | tee "$STAGE2_LOG"
echo "Driver: task194_stage2 fork with --sk_eps_override 0.0 (= #84 baseline argmin only)" | tee -a "$STAGE2_LOG"
echo "Best ckpt: $STAGE1_BEST" | tee -a "$STAGE2_LOG"
echo "Output: $OUTPUT_NPY" | tee -a "$STAGE2_LOG"

python3 $REPO/scripts/task194_stage2_codebook.py \
    --ckpt_path "$STAGE1_BEST" \
    --output_path "$OUTPUT_NPY" \
    --device cuda:0 \
    --max_sinkhorn_iters 30 \
    --sk_eps_override 0.0 \
    >> $STAGE2_LOG 2>&1

STAGE2_EXIT=$?
echo "===== Stage 2 exit: $STAGE2_EXIT at $(date) =====" | tee -a "$STAGE2_LOG"
if [ $STAGE2_EXIT -ne 0 ]; then
    echo "❌ Stage 2 failed" | tee -a "$STAGE2_LOG"
    exit 1
fi

if [ ! -f "$OUTPUT_NPY" ]; then
    echo "❌ Stage 2 output $OUTPUT_NPY NOT FOUND" | tee -a "$STAGE2_LOG"
    exit 1
fi
echo "✅ Stage 2 output: $OUTPUT_NPY ($(stat -c%s $OUTPUT_NPY) bytes)" | tee -a "$STAGE2_LOG"
echo "✅ Stage 2 DONE — ready for Stage 3 (waiting for GPU 1 to free)" | tee -a "$STAGE2_LOG"