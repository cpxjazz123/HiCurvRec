#!/bin/bash
# Task #276 Stage 2 — A2 curriculum SID inference (Sinkhorn + dedup)
# 2026-07-29
#
# 输入: A2_extend_ep50 best_collision_model.pth (L0=89.1%, collision=0.1532)
# 输出: products/task276/stage2/A2_t5_hrqvae_poincare.npy (9922, 4) int array

set -euo pipefail

REPO=/home/wlia0047/ar57/wenyu/GeneRec
source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh
conda activate /home/wlia0047/ar57_scratch/wenyu/grid_toys

export PYTHONPATH=$REPO/HG-Rec:${PYTHONPATH:-}
export HF_HOME=${HF_HOME:-/home/wlia0047/ar57_scratch/wenyu/hf_models}

A2_CKPT=$(ls -t $REPO/products/task275/A2_extend_ep50/*/best_collision_model.pth 2>/dev/null | head -1)
OUT_DIR=$REPO/products/task276/stage2
mkdir -p $OUT_DIR
LOG_FILE=$REPO/logs/task276/stage2_inference_$(date +%Y-%m-%d_%H-%M-%S).log
mkdir -p $REPO/logs/task276

GPU=${GPU:-0}

echo "[$(date)] A2 Stage 2 inference 启动"
echo "[$(date)] ckpt=$A2_CKPT"
echo "[$(date)] out_dir=$OUT_DIR"
echo "[$(date)] GPU=$GPU"

TRITON_CACHE_DIR=/home/wlia0047/.triton/cache_task276_stage2 \
CUDA_VISIBLE_DEVICES=$GPU \
timeout 1200 python3 -u $REPO/scripts/task276_stage2_inference.py \
    --ckpt_path "$A2_CKPT" \
    --output_path "$OUT_DIR/A2_t5_hrqvae_poincare.npy" \
    --data_path "$REPO/HG-Rec/dataset/Instruments/item_emb.parquet" \
    --device cuda:0 \
    --sk_max_iters 30 \
    > $LOG_FILE 2>&1

EXIT=$?
echo "[$(date)] exit code=$EXIT"
if [ -f "$OUT_DIR/A2_t5_hrqvae_poincare.npy" ]; then
    echo "[$(date)] ✅ SID .npy 落盘"
    python3 -c "
import numpy as np
a = np.load('$OUT_DIR/A2_t5_hrqvae_poincare.npy')
print(f'shape={a.shape}, dtype={a.dtype}')
print(f'L0 unique={len(set(a[:,0].tolist()))}/64')
print(f'L1 unique={len(set([tuple(r[:2]) for r in a]))}/4096 (L0+L1 joint)')
print(f'4-digit unique={len(set([tuple(r) for r in a]))}/9922')
print(f'first 3 codes: {a[:3].tolist()}')
"
else
    echo "[$(date)] ❌ SID .npy 未生成"
fi
