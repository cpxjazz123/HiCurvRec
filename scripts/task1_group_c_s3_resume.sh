#!/bin/bash
# task3_group_c_s3_resume.sh — 从 step 2400 ckpt 恢复 Group C s3 训练
# 原 run 被 SIGTERM 杀死，恢复后等早停
set -euo pipefail

source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh
conda activate /home/wlia0047/ar57_scratch/wenyu/grid_toys

cd /fs04/ar57/wenyu/GeneRec/GRID

DATA_DIR=/fs04/ar57/wenyu/GeneRec/GRID/data/amazon_data/toys
SID_PATH=/home/wlia0047/ar57/wenyu/GeneRec/GRID/logs/inference/runs/task13_group_c_s22/pickle/merged_predictions_tensor.pt
CKPT_PATH=/home/wlia0047/ar57/wenyu/GeneRec/GRID/logs/train/runs/task3_group_c_s3/checkpoints/last.ckpt

if [ ! -f "$SID_PATH" ]; then
    echo "ERROR: Group C SID tensor not found: $SID_PATH" >&2
    exit 1
fi
if [ ! -f "$CKPT_PATH" ]; then
    echo "ERROR: last.ckpt not found: $CKPT_PATH" >&2
    exit 1
fi
echo "[info] Resume from ckpt: $CKPT_PATH"
echo "[info] Group C SID tensor: $SID_PATH"

# 用 setsid + nohup 让 python 进程独立于 bash shell
setsid nohup bash -c "
    source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh
    conda activate /home/wlia0047/ar57_scratch/wenyu/grid_toys
    cd /fs04/ar57/wenyu/GeneRec/GRID
    CUDA_VISIBLE_DEVICES=${1:-0} python -u -m src.train \
        experiment=tiger_train_flat \
        data_dir=${DATA_DIR} \
        semantic_id_path=${SID_PATH} \
        num_hierarchies=4 \
        sequence_length=120 \
        trainer.accelerator=gpu \
        trainer.devices=1 \
        trainer.strategy=auto \
        ++should_skip_retry=true \
        +ckpt_path=${CKPT_PATH} \
        paths.root_dir=/fs04/ar57/wenyu/GeneRec/GRID \
        id=task13_group_c_s3_resume
" > /home/wlia0047/ar57/wenyu/GeneRec/GRID/task_artifacts/scripts/logs/task3_group_c_s3_resume.log 2>&1 &
disown
echo "Launched Group C s3 resume as PID $!"