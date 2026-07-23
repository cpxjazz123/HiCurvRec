#!/usr/bin/env bash
# P4 seed43 patience-test: 验证 seed43 是否会逃出 plateau
# 配置完全沿用 seed43 原始 (seed=43, 数据集, 模型, 优化器), 仅:
#   - patience: 3 -> 8 (允许更长平台期)
#   - max_steps: 10000 -> 600 (只需跑到 ~600 步观察逃逸)
#   - 不做 test (只观察 val_R@10 轨迹)

set -e

DATA_DIR=/home/wlia0047/ar57/wenyu/GeneRec/GRID/data/amazon_data/toys
SID_PATH=/home/wlia0047/ar57/wenyu/GeneRec/GRID/logs/inference/runs/task19_aq_s22/pickle/merged_predictions_tensor.pt
GRID=/home/wlia0047/ar57/wenyu/GeneRec/GRID
SEED_DIR=/home/wlia0047/ar57_scratch/wenyu/p4_seed43_patience
GPU=${1:-0}

mkdir -p ${SEED_DIR}

# 完全沿用 seed43 原始 config, 仅 patience + max_steps 改
OVERRIDES=(
    "experiment=tiger_train_flat"
    "data_dir=${DATA_DIR}"
    "semantic_id_path=${SID_PATH}"
    "num_hierarchies=4"
    "sequence_length=120"
    "ckpt_path=null"
    "seed=43"
    "id=p4_seed43_patience8"
    "data_loading.train_dataloader_config.dataloader.data_folder=${DATA_DIR}/diag_train"
    "data_loading.val_dataloader_config.dataloader.data_folder=${DATA_DIR}/diag_val"
    "callbacks.model_checkpoint.monitor=val/recall@10"
    "callbacks.model_checkpoint.save_top_k=1"
    "callbacks.early_stopping.monitor=val/recall@10"
    "callbacks.early_stopping.patience=8"
    "callbacks.early_stopping.min_delta=0.001"
    "callbacks.early_stopping.mode=max"
    "trainer.val_check_interval=500"
    "trainer.max_steps=600"
    "trainer.accelerator=gpu"
    "trainer.devices=1"
    "data_loading.train_dataloader_config.dataloader.num_workers=0"
    "data_loading.train_dataloader_config.dataloader.persistent_workers=false"
    "data_loading.train_dataloader_config.dataloader.timeout=0"
    "data_loading.val_dataloader_config.dataloader.num_workers=0"
    "data_loading.val_dataloader_config.dataloader.persistent_workers=false"
    "data_loading.val_dataloader_config.dataloader.timeout=0"
    "paths.root_dir=${GRID}"
)

echo "=== Seed 43 patience=8 max_steps=600 on GPU ${GPU} ==="
date

source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh
conda activate /home/wlia0047/ar57_scratch/wenyu/grid_toys
cd ${GRID}
CUDA_VISIBLE_DEVICES=${GPU} PYTHONPATH=${GRID} python -m src.train \
    "${OVERRIDES[@]}" \
    "++should_skip_retry=True" \
    2>&1 | tee "${SEED_DIR}/train.log"

echo "=== Patience test done ==="
date

# 复制 metrics csv 到 SEED_DIR 便于查看
TRAIN_RUN_DIR=${GRID}/logs/train/runs/p4_seed43_patience8
if [ -d "${TRAIN_RUN_DIR}/csv" ]; then
    cp -r "${TRAIN_RUN_DIR}/csv" ${SEED_DIR}/
fi
if [ -d "${TRAIN_RUN_DIR}/checkpoints" ]; then
    cp -r "${TRAIN_RUN_DIR}/checkpoints" ${SEED_DIR}/ 2>/dev/null || true
fi