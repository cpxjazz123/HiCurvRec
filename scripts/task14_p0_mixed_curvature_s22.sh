#!/bin/bash
# task4_p0_mixed_curvature_s22.sh — Phase 2 P0.3 Stage 2.2: Mixed-Curvature SID inference
# 输入: P0.3 ckpt + concat embedding
# 输出: merged_predictions_tensor.pt
# 注意: P0.3 用自定义 Mixed-Curvature trainer, ckpt 不是标准 rqvae 格式
# 需要写一个 inference wrapper (后面 step 完成时写)
set -euo pipefail
source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh
conda activate /home/wlia0047/ar57_scratch/wenyu/grid_toys
cd /home/wlia0047/ar57/wenyu/GeneRec/GRID

# P0.3 ckpt 路径: 自定义 trainer, 不是标准 rqvae 输出
S21_CKPT=$(ls -d /home/wlia0047/ar57/wenyu/GeneRec/GRID/logs/train/runs/task58_p0_mixed_curvature_*/checkpoints/ckpt_mixed_curvature.ckpt 2>/dev/null | head -1)
if [ -z "$S21_CKPT" ] || [ ! -f "$S21_CKPT" ]; then
    echo "ERROR: P0.3 ckpt not found"
    exit 1
fi
echo "[task58 P0.3 s22] ckpt: $S21_CKPT"
echo "[task58 P0.3 s22] WARN: P0.3 uses custom Mixed-Curvature trainer, sid_inference needs custom wrapper"
echo "         Falling back to extract codes via custom script task4_p0_mixed_sid_inference.py"

mkdir -p /home/wlia0047/ar57/wenyu/GeneRec/GRID/logs/inference/runs/task58_p0_mixed_curvature_s22/pickle

CUDA_VISIBLE_DEVICES=0 /home/wlia0047/ar57_scratch/wenyu/grid_toys/bin/python \
    task_artifacts/scripts/mixed_curvature/task4_p0_mixed_sid_inference.py \
    --ckpt-path "$S21_CKPT" \
    --out-path /home/wlia0047/ar57/wenyu/GeneRec/GRID/logs/inference/runs/task58_p0_mixed_curvature_s22/pickle/merged_predictions_tensor.pt \
    2>&1 | tee /home/wlia0047/ar57/wenyu/GeneRec/GRID/task_artifacts/scripts/logs/task4_p0_mixed_curvature_s22.log

echo "[done] SID at /home/wlia0047/ar57/wenyu/GeneRec/GRID/logs/inference/runs/task58_p0_mixed_curvature_s22/pickle/merged_predictions_tensor.pt"