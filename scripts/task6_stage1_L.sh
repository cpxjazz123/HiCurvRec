#!/bin/bash
# task8_stage1_L.sh — task10 P5 Stage 1a Flan-T5-Large (780M) Toys embedding
#
# GPU 分配: GPU 0 (cuda:0) — 当前空闲
# 输入: data/amazon_data/toys
# 输出: <run_ts>/pickle/merged_predictions_tensor.pt  shape (11924, 1024)
#
# 启动命令:
#   cd /fs04/ar57/wenyu/GeneRec/GRID
#   bash /home/wlia0047/ar57/wenyu/GeneRec/GRID/task_artifacts/scripts/task8_stage1_L.sh

set -euo pipefail
source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh
conda activate /home/wlia0047/ar57_scratch/wenyu/grid_toys
cd /fs04/ar57/wenyu/GeneRec/GRID

DATA_DIR=/fs04/ar57/wenyu/GeneRec/GRID/data/amazon_data/toys

mkdir -p /home/wlia0047/ar57/wenyu/GeneRec/GRID/task_artifacts/scripts/logs

CUDA_VISIBLE_DEVICES=0 python -m src.inference \
    experiment=sem_embeds_inference_flat \
    data_dir=${DATA_DIR} \
    embedding_model=google/flan-t5-large \
    trainer.accelerator=gpu \
    trainer.devices=1 \
    paths.root_dir=/fs04/ar57/wenyu/GeneRec/GRID \
    id=task8_s1_L \
    2>&1 | tee /home/wlia0047/ar57/wenyu/GeneRec/GRID/task_artifacts/scripts/logs/task8_s1_L.log