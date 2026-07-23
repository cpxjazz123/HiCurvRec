#!/bin/bash
# task9_stage21_l5w256.sh — task11 L=5, W=256 Stage 2.1 RKMeans 训练
#
# GPU 分配: GPU 3 (cuda:3)
# 训练量: 5000 steps（L × 1000）
# 输出: <run_ts>/checkpoints/checkpoint_step=005000.ckpt
#
# 启动命令:
#   cd /fs04/ar57/wenyu/GeneRec/GRID
#   bash /home/wlia0047/ar57/wenyu/GeneRec/GRID/task_artifacts/scripts/task9_stage21_l5w256.sh

set -euo pipefail
source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh
conda activate /home/wlia0047/ar57_scratch/wenyu/grid_toys
cd /fs04/ar57/wenyu/GeneRec/GRID

EMB_PATH=/home/wlia0047/ar57/wenyu/GeneRec/GRID/logs/inference/runs/2026-07-07/18-51-46/pickle/merged_predictions_tensor.pt
DATA_DIR=/fs04/ar57/wenyu/GeneRec/GRID/data/amazon_data/toys

mkdir -p /home/wlia0047/ar57/wenyu/GeneRec/GRID/task_artifacts/scripts/logs

CUDA_VISIBLE_DEVICES=3 python -m src.train \
    experiment=rkmeans_train_flat \
    data_dir=${DATA_DIR} \
    embedding_path=${EMB_PATH} \
    embedding_dim=2048 \
    num_hierarchies=5 \
    codebook_width=256 \
    trainer.max_steps=5000 \
    trainer.accelerator=gpu \
    trainer.devices=1 \
    trainer.strategy=auto \
    paths.root_dir=/fs04/ar57/wenyu/GeneRec/GRID \
    2>&1 | tee /home/wlia0047/ar57/wenyu/GeneRec/GRID/task_artifacts/scripts/logs/task9_s21_l5w256.log