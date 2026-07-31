#!/bin/bash
# Task #396b / Issue #99 - Stage 3 T5-mini training (post #97 patch + #99 Stage 2 PASS)
# Uses task396a 4-digit SID (Instruments_t5_rqvae_task396.npy, unique 9922/9922=100%)
# Per HG-Rec baseline recipe: 200 epoch, early_stop=20, beam_size=20, seed=42

set -e

source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh
conda activate /home/wlia0047/ar57_scratch/wenyu/genrec_env

cd /home/wlia0047/ar57/wenyu/GeneRec/HG-Rec

export PYTHONPATH=/home/wlia0047/ar57/wenyu/GeneRec/HG-Rec:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=0
export HF_HOME=/home/wlia0047/ar57_scratch/wenyu/hf_models

LOG_DIR=/home/wlia0047/ar57/wenyu/GeneRec/logs
TS=$(date +%b-%d-%Y_%H-%M-%S | sed 's/^.../\L&/')
LOG_FILE=$LOG_DIR/task396b_t5mini_stage3_${TS}.log
PID_FILE=/home/wlia0047/ar57/wenyu/GeneRec/products/task396_issue99_stage3_t5_train/_STAGE3_TRAINING_PID

mkdir -p $LOG_DIR
mkdir -p /home/wlia0047/ar57/wenyu/GeneRec/products/task396_issue99_stage3_t5_train/ckpt/

echo "===== [Task #396b Stage 3] HG-Rec T5-mini training (post #97 patch) =====" | tee $LOG_FILE
echo "Started at $(date)" | tee -a $LOG_FILE
echo "Dataset=Instruments code_path=_t5_rqvae_task396.npy (task396a 4-digit SID, unique 9922/9922)" | tee -a $LOG_FILE
echo "GPU 0 (R7 空闲 — task394 已释放)" | tee -a $LOG_FILE
echo "Recipe: 200 epoch + early_stop=20, beam=20, seed=42 (HG-Rec baseline)" | tee -a $LOG_FILE

# Verify Stage 2 SID exists
SID_FILE=/home/wlia0047/ar57/wenyu/GeneRec/HG-Rec/dataset/Instruments/Instruments_t5_rqvae_task396.npy
if [ ! -f "$SID_FILE" ]; then
    echo "❌ $SID_FILE NOT FOUND — task396a must run first" | tee -a $LOG_FILE
    exit 1
fi

# Run training (use task84 baseline Stage 3 script as fork)
python3 /home/wlia0047/ar57/wenyu/GeneRec/scripts/task84_hgrec_stage3_train.py \
    --dataset_name Instruments \
    --dataset_path /home/wlia0047/ar57/wenyu/GeneRec/HG-Rec/dataset/ \
    --code_path _t5_rqvae_task396.npy \
    --codebook_size 64 128 256 1 \
    --num_epochs 200 \
    --batch_size 256 \
    --lr 1e-4 \
    --num_layers 6 \
    --num_decoder_layers 4 \
    --d_model 128 \
    --d_ff 1024 \
    --num_heads 6 \
    --d_kv 64 \
    --vocab_size 1025 \
    --max_len 20 \
    --pad_token_id 0 \
    --eos_token_id 0 \
    --device cuda:0 \
    --mode train \
    --save_path /home/wlia0047/ar57/wenyu/GeneRec/products/task396_issue99_stage3_t5_train/ckpt/ \
    --log_path /home/wlia0047/ar57/wenyu/GeneRec/logs/ \
    --seed 42 \
    --early_stop 20 \
    --beam_size 20 \
    --infer_size 96 \
    2>&1 | tee -a "$LOG_FILE" &

TRAIN_PID=$!
echo "$TRAIN_PID" > "$PID_FILE"
echo "===== [Task #396b Stage 3] Training PID: $TRAIN_PID =====" | tee -a $LOG_FILE

wait $TRAIN_PID
EXIT_CODE=$?
echo "===== [Task #396b Stage 3] Training completed at $(date), exit code: $EXIT_CODE =====" | tee -a $LOG_FILE

# R12 验证 ckpt 落盘
BEST_CKPT=$(find /home/wlia0047/ar57/wenyu/GeneRec/products/task396_issue99_stage3_t5_train/ckpt/ -name "HG_Rec_best.pth" 2>/dev/null | head -1)
if [ -n "$BEST_CKPT" ]; then
    echo "✅ R12 best ckpt 落盘: $BEST_CKPT" | tee -a $LOG_FILE
else
    echo "❌ R12 best ckpt MISSING" | tee -a $LOG_FILE
    exit 1
fi

# 清理 PID file
rm -f "$PID_FILE"