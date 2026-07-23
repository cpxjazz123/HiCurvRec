#!/bin/bash
# task8_stage1_XXL.sh — task10 P5 Stage 1b Flan-T5-XXL (11B) Toys embedding
#
# GPU 分配: GPU 1 (cuda:1) — 当前空闲
# ⚠️ XXL 11B fp16 ~22GB 显存，需降 batch_size_per_device + bf16-mixed 防 OOM
# 输入: data/amazon_data/toys
# 输出: <run_ts>/pickle/merged_predictions_tensor.pt  shape (11924, 4096)
#
# 启动命令:
#   cd /fs04/ar57/wenyu/GeneRec/GRID
#   bash /home/wlia0047/ar57/wenyu/GeneRec/GRID/task_artifacts/scripts/task8_stage1_XXL.sh

set -euo pipefail
source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh
conda activate /home/wlia0047/ar57_scratch/wenyu/grid_toys
cd /fs04/ar57/wenyu/GeneRec/GRID

DATA_DIR=/fs04/ar57/wenyu/GeneRec/GRID/data/amazon_data/toys

mkdir -p /home/wlia0047/ar57/wenyu/GeneRec/GRID/task_artifacts/scripts/logs

CUDA_VISIBLE_DEVICES=1 python -m src.inference \
    experiment=sem_embeds_inference_flat \
    data_dir=${DATA_DIR} \
    embedding_model=google/flan-t5-xxl \
    trainer.accelerator=gpu \
    trainer.devices=1 \
    trainer.precision=bf16-mixed \
    +data_loading.predict_dataloader_config.dataloader.batch_size_per_device=4 \
    paths.root_dir=/fs04/ar57/wenyu/GeneRec/GRID \
    id=task8_s1_XXL \
    2>&1 | tee /home/wlia0047/ar57/wenyu/GeneRec/GRID/task_artifacts/scripts/logs/task8_s1_XXL.log