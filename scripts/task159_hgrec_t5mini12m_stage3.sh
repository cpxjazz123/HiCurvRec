#!/bin/bash
# Task #159 Stage 3 — T5-mini 12M (d_model=256, 4+4 layers)
# 2026-07-24 (用户 T5 容量 ladder request)
# R11.3 决策:
#   - 复用 Task #156 code_default codebook (β=0.25, [32,64,256], sk=0.5)
#   - 复用 scripts/task84_hgrec_stage3_train.py fork
#   - T5 config: d_model=256, 4+4 layers, d_ff=1024, 6 heads × 64 d_kv (~12M params)
#   - GPU 1 (R7: 空闲)
# R12: save_limit=1 (fork 已加)

set -e

source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh
conda activate /home/wlia0047/ar57_scratch/wenyu/genrec_env

cd /home/wlia0047/ar57/wenyu/GeneRec/HG-Rec

export PYTHONPATH=/home/wlia0047/ar57/wenyu/GeneRec/HG-Rec:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=1
export HF_HOME=/home/wlia0047/ar57_scratch/wenyu/hf_models
export TRITON_CACHE_DIR=/home/wlia0047/.triton/cache_task159

LOG_DIR=/home/wlia0047/ar57/wenyu/GeneRec/logs/task159
TS=$(date +%b-%d-%Y_%H-%M-%S | sed 's/^.../\L&/')
LOG_FILE=$LOG_DIR/stage3_train_${TS}.log
PID_FILE=/home/wlia0047/ar57/wenyu/GeneRec/products/task159/_STAGE3_TRAINING_PID

mkdir -p $LOG_DIR
mkdir -p /home/wlia0047/ar57/wenyu/GeneRec/products/task159/ckpt_hgrec/Instruments

echo "===== [Task #159 Stage 3] T5-mini 12M training launched at $(date) =====" | tee $LOG_FILE
echo "Recipe: T5-mini 12M + Task #156 code_default codebook (β=0.25, [32,64,256], sk=0.5)" | tee -a $LOG_FILE
echo "GPU 1 (R7 并行 Task #156 #157 #122)" | tee -a $LOG_FILE
echo "code_path = _t5_rqvae_code_default.npy" | tee -a $LOG_FILE
echo "T5 config: 4 enc + 4 dec layers, d_model=256, d_ff=1024, 6 heads (T5-mini ~12M)" | tee -a $LOG_FILE
echo "Stage 3 fork: scripts/task84_hgrec_stage3_train.py (R12 save_limit=1)" | tee -a $LOG_FILE

# Verify codebook exists
CODE_FILE=/home/wlia0047/ar57/wenyu/GeneRec/HG-Rec/dataset/Instruments/Instruments_t5_rqvae_code_default.npy
if [ ! -f "$CODE_FILE" ]; then
    echo "❌ $CODE_FILE NOT FOUND" | tee -a $LOG_FILE
    exit 1
fi

# T5-mini config
python3 /home/wlia0047/ar57/wenyu/GeneRec/scripts/task84_hgrec_stage3_train.py \
    --dataset_name Instruments \
    --dataset_path /home/wlia0047/ar57/wenyu/GeneRec/HG-Rec/dataset/ \
    --code_path _t5_rqvae_code_default.npy \
    --codebook_size 32 64 256 1 \
    --num_epochs 200 \
    --batch_size 256 \
    --lr 1e-4 \
    --num_layers 4 \
    --num_decoder_layers 4 \
    --d_model 256 \
    --d_ff 1024 \
    --num_heads 6 \
    --d_kv 64 \
    --vocab_size 1025 \
    --max_len 20 \
    --pad_token_id 0 \
    --eos_token_id 0 \
    --device cuda:0 \
    --mode train \
    --save_path /home/wlia0047/ar57/wenyu/GeneRec/products/task159/ckpt_hgrec/ \
    --log_path /home/wlia0047/ar57/wenyu/GeneRec/logs/task159/ \
    --seed 42 \
    --early_stop 20 \
    --beam_size 20 \
    --infer_size 96 \
    2>&1 | tee -a "$LOG_FILE" &

TRAIN_PID=$!
echo "$TRAIN_PID" > "$PID_FILE"
echo "===== [Task #159 Stage 3] Training PID: $TRAIN_PID =====" | tee -a $LOG_FILE

wait $TRAIN_PID
EXIT_CODE=$?
echo "===== [Task #159 Stage 3] Training completed at $(date), exit code: $EXIT_CODE =====" | tee -a $LOG_FILE

# R12 verify ckpt
BEST_CKPT=/home/wlia0047/ar57/wenyu/GeneRec/products/task159/ckpt_hgrec/Instruments/*/HG_Rec_best.pth
if ls $BEST_CKPT 2>/dev/null; then
    echo "✅ R12 best ckpt 落盘: $(ls $BEST_CKPT | head -1)" | tee -a $LOG_FILE
else
    echo "❌ R12 best ckpt MISSING" | tee -a $LOG_FILE
    exit 1
fi

rm -f "$PID_FILE"
