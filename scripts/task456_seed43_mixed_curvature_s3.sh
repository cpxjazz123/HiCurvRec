#!/bin/bash
# task456_seed43_mixed_curvature_s3.sh — Phase 2 Multi-Seed validation
# seed=43, 复用 P0.3 Stage 2.2 SID tensor (deterministic from Stage 2.1 ckpt)
# 只变 TIGER training seed, 验证 +39% R@10 是否 systematic
set -euo pipefail
source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh
conda activate /home/wlia0047/ar57_scratch/wenyu/grid_toys
cd /home/wlia0047/ar57/wenyu/GeneRec/GRID

DATA_DIR=/home/wlia0047/ar57/wenyu/GeneRec/GRID/data/amazon_data/toys
SID_PATH=/home/wlia0047/ar57/wenyu/GeneRec/GRID/logs/inference/runs/task58_p0_mixed_curvature_s22/pickle/merged_predictions_tensor.pt
if [ ! -f "$SID_PATH" ]; then
    echo "ERROR: P0.3 SID not found at $SID_PATH"
    exit 1
fi
echo "[task456 seed=43 s3] SID: $SID_PATH"

CUDA_VISIBLE_DEVICES=0 python -m src.train \
    experiment=tiger_train_flat \
    data_dir=${DATA_DIR} \
    semantic_id_path=${SID_PATH} \
    num_hierarchies=4 \
    sequence_length=120 \
    seed=43 \
    trainer.accelerator=gpu \
    trainer.devices=1 \
    trainer.strategy=auto \
    ++should_skip_retry=true \
    paths.root_dir=/home/wlia0047/ar57/wenyu/GeneRec/GRID \
    id=task456_seed43_mixed_curvature_s3 \
    2>&1 | tee /home/wlia0047/ar57/wenyu/GeneRec/GRID/task_artifacts/scripts/logs/task456_seed43_mixed_curvature_s3.log

echo "[done] ckpt at /home/wlia0047/ar57/wenyu/GeneRec/GRID/logs/train/runs/task456_seed43_mixed_curvature_s3/checkpoints/"