#!/bin/bash
# task9_stage21_l3w512.sh — task11 L=3, W=512 Stage 2.1 RKMeans 训练
#
# GPU 分配: GPU 1 (cuda:1)
# 训练量: 3000 steps（论文 Section 3.1: "RK-Means layer-wise 1k steps per layer"）
# 输出: <run_ts>/checkpoints/checkpoint_step=003000.ckpt
#
# 启动命令:
#   cd /fs04/ar57/wenyu/GeneRec/GRID
#   bash /home/wlia0047/ar57/wenyu/GeneRec/GRID/task_artifacts/scripts/task9_stage21_l3w512.sh

set -euo pipefail
source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh
conda activate /home/wlia0047/ar57_scratch/wenyu/grid_toys
cd /fs04/ar57/wenyu/GeneRec/GRID

EMB_PATH=/home/wlia0047/ar57/wenyu/GeneRec/GRID/logs/inference/runs/2026-07-07/18-51-46/pickle/merged_predictions_tensor.pt
DATA_DIR=/fs04/ar57/wenyu/GeneRec/GRID/data/amazon_data/toys

mkdir -p /home/wlia0047/ar57/wenyu/GeneRec/GRID/task_artifacts/scripts/logs

CUDA_VISIBLE_DEVICES=1 python -m src.train \
    experiment=rkmeans_train_flat \
    data_dir=${DATA_DIR} \
    embedding_path=${EMB_PATH} \
    embedding_dim=2048 \
    num_hierarchies=3 \
    codebook_width=512 \
    trainer.max_steps=3000 \
    trainer.accelerator=gpu \
    trainer.devices=1 \
    trainer.strategy=auto \
    paths.root_dir=/fs04/ar57/wenyu/GeneRec/GRID \
    2>&1 | tee /home/wlia0047/ar57/wenyu/GeneRec/GRID/task_artifacts/scripts/logs/task9_s21_l3w512.log