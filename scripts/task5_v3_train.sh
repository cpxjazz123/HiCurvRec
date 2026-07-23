#!/bin/bash
# task7_v3_train.sh — 修复 task9 Decoder-Only mode collapse（v3：仅降低 LR，去掉 LinearLR）
#
# 问题: task9 v2 用 LR=0.001 + 无 warmup，导致 model mode collapse (81% top-1 = [123,142,222,0])
# 修复: 用 paper LR=5e-4 + 50k steps（早停），无自定义 scheduler（避免 Hydra LinearLR 配置错误）
#
# 训练配置: T5 decoder-only with GPT-2 backbone (decoder-only via causal mask)
# 数据: Amazon Toys 5-core (19412 users, 11924 items)
# 训练量: 50k steps（远小于 paper 320k，但有 early_stopping 早停）
#
# 启动命令:
#   cd /fs04/ar57/wenyu/GeneRec/GRID
#   bash /home/wlia0047/ar57/wenyu/GeneRec/GRID/task_artifacts/scripts/task7_v3_train.sh

set -euo pipefail
source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh
conda activate /home/wlia0047/ar57_scratch/wenyu/grid_toys
cd /fs04/ar57/wenyu/GeneRec/GRID

SID_PATH=/fs04/ar57/wenyu/GeneRec/GRID/logs/inference/runs/2026-07-09/09-28-56/pickle/merged_predictions_tensor_dedup.pt
DATA_DIR=/fs04/ar57/wenyu/GeneRec/GRID/data/amazon_data/toys

mkdir -p /home/wlia0047/ar57/wenyu/GeneRec/GRID/task_artifacts/scripts/logs

python -m src.train \
    experiment=tiger_decoder_only_train_flat \
    data_dir=${DATA_DIR} \
    semantic_id_path=${SID_PATH} \
    num_hierarchies=4 \
    sequence_length=120 \
    seed=42 \
    optim.optimizer.lr=5e-4 \
    optim.optimizer.weight_decay=1e-6 \
    trainer.max_steps=50000 \
    trainer.val_check_interval=2000 \
    trainer.accelerator=gpu \
    trainer.devices=1 \
    trainer.strategy=auto \
    paths.root_dir=/fs04/ar57/wenyu/GeneRec/GRID \
    2>&1 | tee /home/wlia0047/ar57/wenyu/GeneRec/GRID/task_artifacts/scripts/logs/task7_v3_train.log