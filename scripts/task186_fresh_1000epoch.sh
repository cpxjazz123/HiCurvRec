#!/bin/bash
# Task #186 — Fresh 1000 epoch training from epoch 0, no resume, no early stop.
# User 2026-07-25 18:00 指令: 完整从 0 开始训练 1000 epoch, 不加早停.
# Recipe: Phase 0.6 paper-aligned (Instruments + T5-small 5.5M + paper_fix codebook).

set -e

source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh
conda activate /home/wlia0047/ar57_scratch/wenyu/grid_toys

cd /home/wlia0047/ar57/wenyu/GeneRec/HG-Rec

export PYTHONPATH=/home/wlia0047/ar57/wenyu/GeneRec/HG-Rec:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=0
export HF_HOME=/home/wlia0047/ar57_scratch/wenyu/hf_models
export TRITON_CACHE_DIR=/home/wlia0047/.triton/cache_task186
mkdir -p $TRITON_CACHE_DIR

LOG_DIR=/home/wlia0047/ar57/wenyu/GeneRec/logs/task186
TS=$(date +%b-%d-%Y_%H-%M-%S | sed 's/^.../\L&/')
LOG_FILE=$LOG_DIR/train_${TS}.log
mkdir -p $LOG_DIR

SAVE_PATH=/home/wlia0047/ar57/wenyu/GeneRec/products/task186/t5small_fresh_1000epoch

echo "===== [Task #186] Fresh 1000 epoch training at $(date) =====" | tee $LOG_FILE
echo "GPU: $CUDA_VISIBLE_DEVICES" | tee -a $LOG_FILE
echo "Save path: $SAVE_PATH" | tee -a $LOG_FILE

python3 -u /home/wlia0047/ar57/wenyu/GeneRec/scripts/task84_hgrec_stage3_train.py \
    --batch_size 1024 \
    --infer_size 96 \
    --num_epochs 1000 \
    --lr 1e-4 \
    --num_layers 6 \
    --num_decoder_layers 4 \
    --d_model 128 \
    --d_ff 1024 \
    --num_heads 6 \
    --d_kv 64 \
    --dropout_rate 0.1 \
    --vocab_size 1025 \
    --max_len 20 \
    --pad_token_id 0 \
    --eos_token_id 0 \
    --dataset_name Instruments \
    --dataset_path /home/wlia0047/ar57/wenyu/GeneRec/HG-Rec/dataset/ \
    --codebook_size 64 128 256 1 \
    --code_path _t5_rqvae_paper_fix.npy \
    --mode train \
    --save_path $SAVE_PATH \
    --log_path $LOG_DIR \
    --seed 42 \
    --early_stop 99999 \
    --disable_early_stop \
    --topk_list 5 10 20 \
    --beam_size 20 \
    2>&1 | tee -a $LOG_FILE

EXIT_CODE=${PIPESTATUS[0]}
echo "===== [Task #186] exit code: $EXIT_CODE at $(date) =====" | tee -a $LOG_FILE
[ $EXIT_CODE -ne 0 ] && exit 1

# R12: 训练完成清理 PID 文件
rm -f /home/wlia0047/ar57/wenyu/GeneRec/products/task186/_TRAINING_PID

echo "✅ Task #186 1000 epoch training complete" | tee -a $LOG_FILE
