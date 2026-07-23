#!/bin/bash
# task9_stage3_l3w512.sh — task11 L=3, W=512 Stage 3 TIGER 训练
#
# GPU 分配: GPU 1 (cuda:1) — Stage 2.2 inference 完成后腾出
# 输入: Stage 2.2 dedup SID tensor (shape: (4, 11924))
# num_hierarchies=4 (L=3 + 1 dedup column)
# 输出: <run_ts>/checkpoints/...
#
# 启动命令:
#   cd /fs04/ar57/wenyu/GeneRec/GRID
#   bash /home/wlia0047/ar57/wenyu/GeneRec/GRID/task_artifacts/scripts/task9_stage3_l3w512.sh

set -euo pipefail
source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh
conda activate /home/wlia0047/ar57_scratch/wenyu/grid_toys
cd /fs04/ar57/wenyu/GeneRec/GRID

SID_PATH=/fs04/ar57/wenyu/GeneRec/GRID/logs/inference/runs/2026-07-10/13-21-02/pickle/merged_predictions_tensor_dedup.pt
DATA_DIR=/fs04/ar57/wenyu/GeneRec/GRID/data/amazon_data/toys

mkdir -p /home/wlia0047/ar57/wenyu/GeneRec/GRID/task_artifacts/scripts/logs

CUDA_VISIBLE_DEVICES=1 python -m src.train \
    experiment=tiger_train_flat \
    data_dir=${DATA_DIR} \
    semantic_id_path=${SID_PATH} \
    num_hierarchies=4 \
    sequence_length=120 \
    seed=42 \
    optim.optimizer.lr=5e-4 \
    trainer.max_steps=50000 \
    trainer.val_check_interval=2000 \
    trainer.accelerator=gpu \
    trainer.devices=1 \
    paths.root_dir=/fs04/ar57/wenyu/GeneRec/GRID \
    2>&1 | tee /home/wlia0047/ar57/wenyu/GeneRec/GRID/task_artifacts/scripts/logs/task9_s3_l3w512.log