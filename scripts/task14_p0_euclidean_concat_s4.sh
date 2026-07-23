#!/bin/bash
# task4_p0_euclidean_concat_s4.sh — Phase 2 P0.2 Stage 4: TIGER inference
# 输入: P0.2 Stage 3 best_*.ckpt + Stage 2.2 SID
# 输出: merged_predictions_tensor.pt (测试集预测)
set -euo pipefail
source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh
conda activate /home/wlia0047/ar57_scratch/wenyu/grid_toys
cd /home/wlia0047/ar57/wenyu/GeneRec/GRID

DATA_DIR=/home/wlia0047/ar57/wenyu/GeneRec/GRID/data/amazon_data/toys
SID_PATH=/home/wlia0047/ar57/wenyu/GeneRec/GRID/logs/inference/runs/task58_p0_euclidean_concat_s22/pickle/merged_predictions_tensor.pt
S3_CKPT=$(ls -d /home/wlia0047/ar57/wenyu/GeneRec/GRID/logs/train/runs/task58_p0_euclidean_concat_s3/checkpoints/*.ckpt 2>/dev/null | head -1)
if [ -z "$S3_CKPT" ] || [ ! -f "$S3_CKPT" ]; then
    echo "ERROR: P0.2 Stage 3 ckpt not found"
    exit 1
fi
echo "[task58 P0.2 s4] Stage 3 ckpt: $S3_CKPT"

# 复制 ckpt 为不含 = 的文件名以避免 Hydra parser 报错
SAFE_CKPT=/home/wlia0047/ar57/wenyu/GeneRec/GRID/task_artifacts/results/exp58/task4_p0_euclidean_concat_tiger.ckpt
mkdir -p /home/wlia0047/ar57/wenyu/GeneRec/GRID/task_artifacts/results/exp58
cp -f "$S3_CKPT" "$SAFE_CKPT"

CUDA_VISIBLE_DEVICES=0 python -m src.inference \
    experiment=tiger_inference_flat \
    data_dir=${DATA_DIR} \
    semantic_id_path=${SID_PATH} \
    num_hierarchies=4 \
    ckpt_path=${SAFE_CKPT} \
    ++should_skip_retry=true \
    paths.root_dir=/home/wlia0047/ar57/wenyu/GeneRec/GRID \
    id=task58_p0_euclidean_concat_s4 \
    2>&1 | tee /home/wlia0047/ar57/wenyu/GeneRec/GRID/task_artifacts/scripts/logs/task4_p0_euclidean_concat_s4.log

echo "[done] predictions at /home/wlia0047/ar57/wenyu/GeneRec/GRID/logs/inference/runs/task58_p0_euclidean_concat_s4/pickle/"