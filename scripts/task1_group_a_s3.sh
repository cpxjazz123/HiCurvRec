#!/bin/bash
# task3_group_a_s3.sh — Group A Stage 3: TIGER training
# 输入：Group A Stage 2.2 SID tensor
# 输出：best_tiger_*.ckpt (早停)
set -euo pipefail
source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh
conda activate /home/wlia0047/ar57_scratch/wenyu/grid_toys
cd /fs04/ar57/wenyu/GeneRec/GRID

DATA_DIR=/fs04/ar57/wenyu/GeneRec/GRID/data/amazon_data/toys
SID_PATH=$(ls -d /home/wlia0047/ar57/wenyu/GeneRec/GRID/logs/inference/runs/task13_group_a_s22/pickle/merged_predictions_tensor.pt 2>/dev/null | head -1)
if [ -z "$SID_PATH" ] || [ ! -f "$SID_PATH" ]; then
    echo "ERROR: Group A Stage 2.2 SID tensor not found"
    exit 1
fi
echo "[info] SID: $SID_PATH"

CUDA_VISIBLE_DEVICES=0 python -m src.train \
    experiment=tiger_train_flat \
    data_dir=${DATA_DIR} \
    semantic_id_path=${SID_PATH} \
    num_hierarchies=4 \
    sequence_length=120 \
    trainer.accelerator=gpu \
    trainer.devices=1 \
    trainer.strategy=auto \
    ++should_skip_retry=true \
    paths.root_dir=/fs04/ar57/wenyu/GeneRec/GRID \
    id=task13_group_a_s3 \
    2>&1 | tee /home/wlia0047/ar57/wenyu/GeneRec/GRID/task_artifacts/scripts/logs/task3_group_a_s3.log
