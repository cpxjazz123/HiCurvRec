#!/bin/bash
# Task #157 Stage 3 — T5-base 220M 训练 (跟 Task #84 baseline fork, 但 T5 config 不同)
# 2026-07-24 (用户选 C: 并行 Task #156 GPU 3 + Task #157 GPU 0)
# R11.3 决策:
#   - 复用 Task #156 Stage 2 codebook (code-default β=0.25, [32,64,256], sk=0.5)
#   - 复用 scripts/task84_hgrec_stage3_train.py fork (T5Config 接受 num_layers/decoder_layers/d_model/...)
#   - 唯一改动: T5 配置 → T5-base (12+12 layers, d_model=768, d_ff=3072, 12 heads)
#   - GPU 0 (并行 Task #156 在 GPU 3)
# R7: GPU 0 (R7 强制空闲 — 用户已批准并行)
# R12: save_limit=1 (在 task84 fork 已加)

set -e

source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh
conda activate /home/wlia0047/ar57_scratch/wenyu/genrec_env

cd /home/wlia0047/ar57/wenyu/GeneRec/HG-Rec

export PYTHONPATH=/home/wlia0047/ar57/wenyu/GeneRec/HG-Rec:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=0
export HF_HOME=/home/wlia0047/ar57_scratch/wenyu/hf_models
export TRITON_CACHE_DIR=/home/wlia0047/.triton/cache_task157

LOG_DIR=/home/wlia0047/ar57/wenyu/GeneRec/logs/task157
TS=$(date +%b-%d-%Y_%H-%M-%S | sed 's/^.../\L&/')
LOG_FILE=$LOG_DIR/stage3_train_${TS}.log
PID_FILE=/home/wlia0047/ar57/wenyu/GeneRec/products/task157/_STAGE3_TRAINING_PID

mkdir -p $LOG_DIR
mkdir -p /home/wlia0047/ar57/wenyu/GeneRec/products/task157/ckpt_hgrec/Instruments

echo "===== [Task #157 Stage 3] T5-base 220M training launched at $(date) =====" | tee $LOG_FILE
echo "Recipe: T5-base 220M + Task #156 code-default SID (β=0.25, [32,64,256], sk=0.5)" | tee -a $LOG_FILE
echo "GPU 0 (R7 并行 Task #156 在 GPU 3)" | tee -a $LOG_FILE
echo "code_path = _t5_rqvae_code_default.npy" | tee -a $LOG_FILE
echo "T5 config: 12 enc + 12 dec layers, d_model=768, d_ff=3072, 12 heads (T5-base)" | tee -a $LOG_FILE
echo "Stage 3 fork: scripts/task84_hgrec_stage3_train.py (R12 save_limit=1)" | tee -a $LOG_FILE

# Verify Stage 2 codebook exists (Task #156 Stage 2 must complete first)
CODE_FILE=/home/wlia0047/ar57/wenyu/GeneRec/HG-Rec/dataset/Instruments/Instruments_t5_rqvae_code_default.npy
if [ ! -f "$CODE_FILE" ]; then
    echo "❌ $CODE_FILE NOT FOUND — Task #156 Stage 2 not complete" | tee -a $LOG_FILE
    exit 1
fi

# T5-base config (跟 HG-Rec 官方 T5-base 标准)
python3 /home/wlia0047/ar57/wenyu/GeneRec/scripts/task84_hgrec_stage3_train.py \
    --dataset_name Instruments \
    --dataset_path /home/wlia0047/ar57/wenyu/GeneRec/HG-Rec/dataset/ \
    --code_path _t5_rqvae_code_default.npy \
    --codebook_size 32 64 256 1 \
    --num_epochs 200 \
    --batch_size 256 \
    --lr 1e-4 \
    --num_layers 12 \
    --num_decoder_layers 12 \
    --d_model 768 \
    --d_ff 3072 \
    --num_heads 12 \
    --d_kv 64 \
    --vocab_size 1025 \
    --max_len 20 \
    --pad_token_id 0 \
    --eos_token_id 0 \
    --device cuda:0 \
    --mode train \
    --save_path /home/wlia0047/ar57/wenyu/GeneRec/products/task157/ckpt_hgrec/ \
    --log_path /home/wlia0047/ar57/wenyu/GeneRec/logs/task157/ \
    --seed 42 \
    --early_stop 20 \
    --beam_size 20 \
    --infer_size 96 \
    2>&1 | tee -a "$LOG_FILE" &

TRAIN_PID=$!
echo "$TRAIN_PID" > "$PID_FILE"
echo "===== [Task #157 Stage 3] Training PID: $TRAIN_PID =====" | tee -a $LOG_FILE

wait $TRAIN_PID
EXIT_CODE=$?
echo "===== [Task #157 Stage 3] Training completed at $(date), exit code: $EXIT_CODE =====" | tee -a $LOG_FILE

# R12 验证 ckpt
BEST_CKPT=/home/wlia0047/ar57/wenyu/GeneRec/products/task157/ckpt_hgrec/Instruments/*/HG_Rec_best.pth
if ls $BEST_CKPT 2>/dev/null; then
    echo "✅ R12 best ckpt 落盘: $(ls $BEST_CKPT | head -1)" | tee -a $LOG_FILE
else
    echo "❌ R12 best ckpt MISSING" | tee -a $LOG_FILE
    exit 1
fi

rm -f "$PID_FILE"
