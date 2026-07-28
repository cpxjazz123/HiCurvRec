#!/bin/bash
# Task #156 Stage 3 — HG-Rec T5-small training (code-default recipe fork)
# 2026-07-24 (用户提议: 跑 HG-Rec code-default recipe β=0.25 [32,64,256] sk=0.5)
# 镜像 task84_hgrec_stage3_train.sh, 但:
#   - code_path = _t5_rqvae_code_default.npy (Stage 2 输出)
#   - codebook_size = [32, 64, 256, 1] (vs #84 [64, 128, 256, 1])
#   - ckpt_dir = products/task156/ckpt_hgrec (vs #84 products/task84)
#   - 复用 scripts/task84_hgrec_stage3_train.py (R12 save_limit=1 已加)
# R11.3: seed=42 (R5 跨 run 固定), 200 epoch + early_stop=20 (跟 Task #84 baseline 一致)
# R7: GPU 3 (R7 强制空闲 — Stage 1+2 占用 GPU 3)

set -e

source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh
conda activate /home/wlia0047/ar57_scratch/wenyu/genrec_env

cd /home/wlia0047/ar57/wenyu/GeneRec/HG-Rec

# PYTHONPATH for `from model.HG_Rec`
export PYTHONPATH=/home/wlia0047/ar57/wenyu/GeneRec/HG-Rec:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=3
export HF_HOME=/home/wlia0047/ar57_scratch/wenyu/hf_models
export TRITON_CACHE_DIR=/home/wlia0047/.triton/cache_task156

LOG_DIR=/home/wlia0047/ar57/wenyu/GeneRec/logs/task156
TS=$(date +%b-%d-%Y_%H-%M-%S | sed 's/^.../\L&/')
LOG_FILE=$LOG_DIR/stage3_train_${TS}.log
PID_FILE=/home/wlia0047/ar57/wenyu/GeneRec/products/task156/_STAGE3_TRAINING_PID

mkdir -p $LOG_DIR
mkdir -p /home/wlia0047/ar57/wenyu/GeneRec/products/task156/ckpt_hgrec/Instruments

echo "===== [Task #156 Stage 3] HG-Rec T5-small training launched at $(date) =====" | tee $LOG_FILE
echo "Recipe: code-default (β=0.25, [32,64,256], sk=0.5, 1869 epoch)" | tee -a $LOG_FILE
echo "code_path = _t5_rqvae_code_default.npy (Stage 2 output)" | tee -a $LOG_FILE
echo "GPU 3" | tee -a $LOG_FILE
echo "Stage 3 fork: scripts/task84_hgrec_stage3_train.py (R12 save_limit=1)" | tee -a $LOG_FILE

# Verify Stage 2 codebook exists (will exist ~3 min after Stage 1 completes)
CODE_FILE=/home/wlia0047/ar57/wenyu/GeneRec/HG-Rec/dataset/Instruments/Instruments_t5_rqvae_code_default.npy
if [ ! -f "$CODE_FILE" ]; then
    echo "❌ $CODE_FILE NOT FOUND — Stage 2 not complete, exit 1" | tee -a $LOG_FILE
    exit 1
fi

# Same config as Task #84 baseline (single variable principle: only codebook differs)
python3 /home/wlia0047/ar57/wenyu/GeneRec/scripts/task84_hgrec_stage3_train.py \
    --dataset_name Instruments \
    --dataset_path /home/wlia0047/ar57/wenyu/GeneRec/HG-Rec/dataset/ \
    --code_path _t5_rqvae_code_default.npy \
    --codebook_size 32 64 256 1 \
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
    --save_path /home/wlia0047/ar57/wenyu/GeneRec/products/task156/ckpt_hgrec/ \
    --log_path /home/wlia0047/ar57/wenyu/GeneRec/logs/task156/ \
    --seed 42 \
    --early_stop 20 \
    --beam_size 20 \
    --infer_size 96 \
    2>&1 | tee -a "$LOG_FILE" &

TRAIN_PID=$!
echo "$TRAIN_PID" > "$PID_FILE"
echo "===== [Task #156 Stage 3] Training PID: $TRAIN_PID =====" | tee -a $LOG_FILE

wait $TRAIN_PID
EXIT_CODE=$?
echo "===== [Task #156 Stage 3] Training completed at $(date), exit code: $EXIT_CODE =====" | tee -a $LOG_FILE

# R12 验证 ckpt
BEST_CKPT=/home/wlia0047/ar57/wenyu/GeneRec/products/task156/ckpt_hgrec/Instruments/*/HG_Rec_best.pth
if ls $BEST_CKPT 2>/dev/null; then
    echo "✅ R12 best ckpt 落盘: $(ls $BEST_CKPT | head -1)" | tee -a $LOG_FILE
else
    echo "❌ R12 best ckpt MISSING" | tee -a $LOG_FILE
    exit 1
fi

rm -f "$PID_FILE"
