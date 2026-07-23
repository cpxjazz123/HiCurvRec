#!/bin/bash
# Task #90 — FMLP-Rec 复现 (paper Table 2 #3, paper R@10=0.0657)
# 官方 repo: https://github.com/RUCAIBox/FMLP-Rec (已找到)
# 单独 PyTorch 训练, 不基于 RecBole
# 数据: Amazon Musical Instruments 5-core (复用 data/amazon_data/musical_instruments/)
# GPU 推荐: 释放后 GPU (task80/#82/#83 完成后)
set -e

GENE_REC=/home/wlia0047/ar57/wenyu/GeneRec
EXTERNAL=$GENE_REC/external
LOGS=$GENE_REC/logs

# Step 1: 克隆 FMLP-Rec 仓库 (if not exists)
if [ ! -d "$EXTERNAL/FMLP-Rec" ]; then
    echo "[$(date '+%H:%M:%S')] Cloning FMLP-Rec repo..." | tee -a $LOGS/task90_fmlp_setup.log
    git clone https://github.com/RUCAIBox/FMLP-Rec.git $EXTERNAL/FMLP-Rec
fi

# Step 2: 编辑 data 路径到 Musical Instruments 数据集
DATA_DIR=$GENE_REC/data/amazon_data/musical_instruments

GPU=${CUDA_VISIBLE_DEVICES:-0}
LOG_FILE=$LOGS/task90_fmlp_rec_jul-23-2026_13-58-00.log

echo "===== Starting FMLP-Rec (paper Table 2 #3) on GPU $GPU at $(date) =====" | tee -a $LOG_FILE
cd $EXTERNAL/FMLP-Rec
source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh
conda activate /home/wlia0047/ar57_scratch/wenyu/recbole_env

# FMLP-Rec 用自有 config + main.py (需根据 README 调整参数)
python3 main.py --data_path $DATA_DIR --dataset_name Instruments --gpu_id $GPU \
    >> $LOG_FILE 2>&1 || echo "[WARN] FMLP-Rec main.py signature may differ — see README"

echo "===== FMLP-Rec completed at $(date) =====" | tee -a $LOG_FILE
