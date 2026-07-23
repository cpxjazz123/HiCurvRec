#!/bin/bash
# Task #91 — DuoRec 复现 (paper Table 2 #4, paper R@10=0.0588)
# 官方 repo: https://github.com/RuihongQiu/DuoRec (WSDM 2022, RecBole-based)
# 数据: Amazon Musical_Instruments 5-core (RecBole .inter 格式已 ready in data/recbole/)
# Fix 2026-07-23: seq.yaml 实际在仓库根 (不在 configs/), 同时已创建 symlink dataset/Musical_Instruments → /home/wlia0047/.../data/recbole/Musical_Instruments
set -e

GENE_REC=/home/wlia0047/ar57/wenyu/GeneRec
EXTERNAL=$GENE_REC/external
LOGS=$GENE_REC/logs

# Step 1: 克隆 DuoRec repo
if [ ! -d "$EXTERNAL/DuoRec" ]; then
    echo "[$(date '+%H:%M:%S')] Cloning DuoRec repo..." | tee -a $LOGS/task91_duorec_setup.log
    git clone https://github.com/RuihongQiu/DuoRec.git $EXTERNAL/DuoRec
fi

# Step 2: Symlink 我们的 RecBole Musical_Instruments 数据集到 DuoRec/dataset/
mkdir -p $EXTERNAL/DuoRec/dataset
if [ ! -e "$EXTERNAL/DuoRec/dataset/Musical_Instruments" ]; then
    ln -sf $GENE_REC/data/recbole/Musical_Instruments $EXTERNAL/DuoRec/dataset/Musical_Instruments
fi

GPU=${CUDA_VISIBLE_DEVICES:-0}
LOG_FILE=$LOGS/task91_duorec_jul-23-2026_15-22-00.log

echo "===== Starting DuoRec (paper Table 2 #4) on GPU $GPU at $(date) =====" | tee -a $LOG_FILE
cd $EXTERNAL/DuoRec
source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh
conda activate /home/wlia0047/ar57_scratch/wenyu/recbole_env

# R12 FIX (2026-07-23): 把 _TRAINING_PID 写到 products/task91/
TRAINING_PID_FILE=$GENE_REC/products/task91/_TRAINING_PID
mkdir -p $GENE_REC/products/task91
rm -f $TRAINING_PID_FILE

# DuoRec 默认 seq.yaml 在仓库根 (不是 configs/), dataset 已 symlink
python3 run_seq.py --model DuoRec --dataset Musical_Instruments --config_files seq.yaml \
    --gpu_id $GPU \
    >> $LOG_FILE 2>&1 &
TRAIN_PID=$!
echo $TRAIN_PID > $TRAINING_PID_FILE
echo "[$(date '+%H:%M:%S')] task91 training PID: $TRAIN_PID" | tee -a $LOG_FILE

wait $TRAIN_PID
echo "===== DuoRec completed at $(date) =====" | tee -a $LOG_FILE
rm -f $TRAINING_PID_FILE
