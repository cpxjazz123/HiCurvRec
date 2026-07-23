#!/usr/bin/env bash
# P4 launcher: 为单个 seed 跑训练, 用 GPU i
# 用法: bash p4_run_one_seed.sh <seed> <gpu_id>

set -e

SEED=${1:?"usage: $0 <seed> <gpu_id>"}
GPU=${2:?"usage: $0 <seed> <gpu_id>"}

DATA_DIR=/home/wlia0047/ar57/wenyu/GeneRec/GRID/data/amazon_data/toys
SID_PATH=/home/wlia0047/ar57/wenyu/GeneRec/GRID/logs/inference/runs/task19_aq_s22/pickle/merged_predictions_tensor.pt
GRID=/home/wlia0047/ar57/wenyu/GeneRec/GRID
SEED_DIR=/home/wlia0047/ar57_scratch/wenyu/p4_3seed/seed_${SEED}

mkdir -p ${SEED_DIR}

# 复制 task21 already-trained SID (semantic_id_path 是 Stage 2 的产物，所有 seed 共享)
# 不需要重新训 Stage 1/2

# 训练 overrides (写死, 见 选点规则.md)
OVERRIDES=(
    "experiment=tiger_train_flat"
    "data_dir=${DATA_DIR}"
    "semantic_id_path=${SID_PATH}"
    "num_hierarchies=4"
    "sequence_length=120"
    "ckpt_path=null"
    "seed=${SEED}"
    "id=p4_seed${SEED}"
    "data_loading.train_dataloader_config.dataloader.data_folder=${DATA_DIR}/diag_train"
    "data_loading.val_dataloader_config.dataloader.data_folder=${DATA_DIR}/diag_val"
    "callbacks.model_checkpoint.monitor=val/recall@10"
    "callbacks.model_checkpoint.save_top_k=1"
    "callbacks.early_stopping.monitor=val/recall@10"
    "callbacks.early_stopping.patience=3"
    "callbacks.early_stopping.min_delta=0.001"
    "callbacks.early_stopping.mode=max"
    "trainer.val_check_interval=500"
    "trainer.max_steps=10000"
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

echo "=== Seed ${SEED} on GPU ${GPU}: starting ==="
date

source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh
conda activate /home/wlia0047/ar57_scratch/wenyu/grid_toys
cd ${GRID}
CUDA_VISIBLE_DEVICES=${GPU} PYTHONPATH=${GRID} python -m src.train \
    "${OVERRIDES[@]}" \
    2>&1 | tee "${SEED_DIR}/train.log"

echo "=== Seed ${SEED} on GPU ${GPU}: done ==="
date

# Train finished, now eval on diag_test (only once)
TRAIN_RUN_DIR=${GRID}/logs/train/runs/p4_seed${SEED}
BEST_CKPT=$(ls ${TRAIN_RUN_DIR}/checkpoints/checkpoint_*_val_recall@10*.ckpt 2>/dev/null | sort -t= -k4 -n | tail -1)
if [ -z "${BEST_CKPT}" ]; then
    # Try without "val_recall@10" suffix
    BEST_CKPT=$(ls ${TRAIN_RUN_DIR}/checkpoints/*.ckpt 2>/dev/null | head -1)
fi
if [ -z "${BEST_CKPT}" ]; then
    echo "Seed ${SEED}: no ckpt found in ${TRAIN_RUN_DIR}/checkpoints"
    exit 1
fi

# Symlink ckpts to SEED_DIR for visibility
mkdir -p ${SEED_DIR}/checkpoints
cp -al ${TRAIN_RUN_DIR}/checkpoints ${SEED_DIR}/ 2>/dev/null || cp -r ${TRAIN_RUN_DIR}/checkpoints ${SEED_DIR}/ 2>/dev/null || true
cp -r ${TRAIN_RUN_DIR}/csv ${SEED_DIR}/ 2>/dev/null || true

echo "=== Seed ${SEED}: test eval on diag_test (only ONCE) ==="
date

cd ${GRID}
CUDA_VISIBLE_DEVICES=${GPU} PYTHONPATH=${GRID} python -m src.inference \
    experiment=tiger_inference_flat \
    data_dir=${DATA_DIR} \
    semantic_id_path=${SID_PATH} \
    ckpt_path=${BEST_CKPT} \
    num_hierarchies=4 \
    sequence_length=120 \
    "paths.root_dir=${GRID}" \
    "paths.output_path=${SEED_DIR}/test_eval" \
    "data_loading.predict_dataloader_config.dataloader.data_folder=${DATA_DIR}/diag_test" \
    2>&1 | tee "${SEED_DIR}/test_eval.log"

echo "=== Seed ${SEED} on GPU ${GPU}: full done ==="
date
