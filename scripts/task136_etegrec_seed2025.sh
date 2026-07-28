#!/bin/bash
# Task #136 Phase 4 — ETEGRec Musical_Instruments (seed=2025, paper-aligned cycle=2)
# R11.3 自决: 复用 /ETEGRec/config/musical_instruments.yaml (cycle=2, RQVAE 256-256-256-128 128d),
#              accelerate launch DDP if multi-GPU else single GPU, seed=2025 (DECOR paper default)
# 数据已就绪: /ETEGRec/dataset/Musical_Instruments/ (128d + 896d emb + RQVAE pth + train/valid/test jsonl)
# GPU 2 (R7 — TIGER GPU 0, ETEGRec GPU 2, CoST/LFTER GPU 1/3)
# R12 兼容: ETEGRec 内置 checkpoint callback (save every 2 epochs + early_stop 30)

set -e

source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh
conda activate /home/wlia0047/ar57_scratch/wenyu/grid_toys

cd /home/wlia0047/ar57/wenyu/GeneRec/ETEGRec

LOG_DIR=/home/wlia0047/ar57/wenyu/GeneRec/logs
TS=$(date +%b-%d-%Y_%H-%M-%S | sed 's/^.../\L&/')
LOG_FILE=$LOG_DIR/task136_etegrec_seed2025_${TS}.log
PID_FILE=/home/wlia0047/ar57/wenyu/GeneRec/products/task136/_ETEGREC_PID

mkdir -p $LOG_DIR /home/wlia0047/ar57/wenyu/GeneRec/products/task136

echo "===== [Task #136 Phase 4 ETEGRec] launch at $(date) =====" | tee $LOG_FILE
echo "Dataset=Musical_Instruments Config=musical_instruments.yaml seed=2025 cycle=2" | tee -a $LOG_FILE
echo "GPU 2 (R7 — TIGER GPU 0, CoST GPU 1, ETEGRec GPU 2)" | tee -a $LOG_FILE

# Verify datasets ready
DATASET_DIR=/home/wlia0047/ar57/wenyu/GeneRec/ETEGRec/dataset/Musical_Instruments
for f in 256-256-256-128.rqvae.pth Musical_Instruments_emb_128.npy \
         Musical_Instruments.train.jsonl Musical_Instruments.valid.jsonl Musical_Instruments.test.jsonl; do
    if [ ! -f "$DATASET_DIR/$f" ]; then
        echo "❌ $DATASET_DIR/$f NOT FOUND" | tee -a $LOG_FILE
        exit 1
    fi
done
echo "✅ All dataset files present" | tee -a $LOG_FILE

# Single GPU accelerate (R7 — TIGER GPU 0 busy, we use GPU 2)
# Note: musical_instruments.yaml has cycle=2 + warmup_steps=0 already; config matches paper default
# R11.3 自决: batch_size 512→128 (OOM fix, T5-small+RQ-VAE 在 batch=512 用了 43 GB)
# + PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True (减少碎片)
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
CUDA_VISIBLE_DEVICES=2 accelerate launch --config_file accelerate_config_ddp_1gpu.yaml main.py \
    --config ./config/musical_instruments.yaml \
    --batch_size=128 \
    --eval_batch_size=16 \
    2>&1 | tee -a "$LOG_FILE" &

TRAIN_PID=$!
echo "$TRAIN_PID" > "$PID_FILE"
echo "===== [Task #136 ETEGRec] Training PID: $TRAIN_PID =====" | tee -a $LOG_FILE

wait $TRAIN_PID
EXIT_CODE=$?
echo "===== [Task #136 ETEGRec] Training completed at $(date), exit code: $EXIT_CODE =====" | tee -a $LOG_FILE

# Clean up PID file
rm -f "$PID_FILE"
