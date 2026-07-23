#!/bin/bash
# task3_group_c_s4.sh — Group C Stage 4: TIGER inference
# 输入: Group C Stage 3 best_*.ckpt + Stage 2.2 SID
# 输出: merged_predictions_tensor.pt (shape (19412, 10, 4))
set -euo pipefail
source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh
conda activate /home/wlia0047/ar57_scratch/wenyu/grid_toys
cd /fs04/ar57/wenyu/GeneRec/GRID

DATA_DIR=/fs04/ar57/wenyu/GeneRec/GRID/data/amazon_data/toys
SID_PATH=/home/wlia0047/ar57/wenyu/GeneRec/GRID/logs/inference/runs/task13_group_c_s22/pickle/merged_predictions_tensor.pt
S3_CKPT_DIR=/home/wlia0047/ar57/wenyu/GeneRec/GRID/logs/train/runs/task13_group_c_s3/checkpoints
# Group C s3 早停在 step 2400 (best_ndcg@10=0.0310 @ step 2399); save_top_k 未产出 best.ckpt
# 回退: 优先 best*.ckpt, 否则 latest checkpoint_*.ckpt
S3_BEST_CKPT=$(ls -1t ${S3_CKPT_DIR}/best*.ckpt 2>/dev/null | head -1 || true)
if [ -z "$S3_BEST_CKPT" ] || [ ! -f "$S3_BEST_CKPT" ]; then
    S3_BEST_CKPT=$(ls -1t ${S3_CKPT_DIR}/checkpoint_*.ckpt 2>/dev/null | head -1 || true)
    if [ -z "$S3_BEST_CKPT" ] || [ ! -f "$S3_BEST_CKPT" ]; then
        echo "ERROR: Group C Stage 3 ckpt not found in $S3_CKPT_DIR" >&2
        exit 1
    fi
    echo "[fallback] best*.ckpt 缺失; 使用最新 ckpt: $S3_BEST_CKPT"
else
    echo "[info] Stage 3 best ckpt: $S3_BEST_CKPT"
fi

# 安全拷贝 ckpt 为不含 = 的文件名以避免 Hydra parser 报错
SAFE_CKPT=/home/wlia0047/ar57/wenyu/GeneRec/GRID/task_artifacts/results/exp15/task3_group_c_tiger.ckpt
mkdir -p /home/wlia0047/ar57/wenyu/GeneRec/GRID/task_artifacts/results/exp13
cp -f "$S3_BEST_CKPT" "$SAFE_CKPT"

CUDA_VISIBLE_DEVICES=${1:-0} python -m src.inference \
    experiment=tiger_inference_flat \
    data_dir=${DATA_DIR} \
    semantic_id_path=${SID_PATH} \
    num_hierarchies=4 \
    ckpt_path=${SAFE_CKPT} \
    ++should_skip_retry=true \
    paths.root_dir=/fs04/ar57/wenyu/GeneRec/GRID \
    id=task13_group_c_s4 \
    2>&1 | tee /home/wlia0047/ar57/wenyu/GeneRec/GRID/task_artifacts/scripts/logs/task3_group_c_s4.log
