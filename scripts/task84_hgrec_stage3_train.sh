#!/bin/bash
# Task #84 Stage 3 — HG-Rec T5-small training (fork train_HG-Rec.py with R12 + Stage 2 codebook)
# R11.3 自决: 200 epoch + early_stop=20 patience, num_beams=20, beam_size=20, T5-small (6 enc + 4 dec layers)
# GPU 3 (R7 空闲 — S3Rec GPU 0, P5-SID GPU 2, Stage 1+2 GPU 1)
# R12: train_HG-Rec fork 已加 save_limit=1 (save new ckpt 时删旧) — 训练完成后清理 stage 3 ckpt 目录

set -e

source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh
conda activate /home/wlia0047/ar57_scratch/wenyu/genrec_env

cd /home/wlia0047/ar57/wenyu/GeneRec/HG-Rec

# Use Stage 3 fork (script not in HG-Rec cwd; need PYTHONPATH for `from model.hg_rec`)
export PYTHONPATH=/home/wlia0047/ar57/wenyu/GeneRec/HG-Rec:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=3
export HF_HOME=/home/wlia0047/ar57_scratch/wenyu/hf_models

LOG_DIR=/home/wlia0047/ar57/wenyu/GeneRec/logs
TS=$(date +%b-%d-%Y_%H-%M-%S | sed 's/^.../\L&/')
LOG_FILE=$LOG_DIR/task84_hgrec_stage3_train_${TS}.log
PID_FILE=/home/wlia0047/ar57/wenyu/GeneRec/products/task84/_STAGE3_TRAINING_PID

mkdir -p $LOG_DIR
mkdir -p /home/wlia0047/ar57/wenyu/GeneRec/products/task84/ckpt_hgrec/Instruments

echo "===== [Task #84 Stage 3] HG-Rec T5-small training launched at $(date) =====" | tee $LOG_FILE
echo "Dataset=Instruments code_path=_t5_hrqvae_poincare.npy (Stage 2 output)" | tee -a $LOG_FILE
echo "GPU 3 (R7 空闲)" | tee -a $LOG_FILE
echo "Stage 3 fork: scripts/task84_hgrec_stage3_train.py (R12 save_limit=1, code_path default fixed)" | tee -a $LOG_FILE

# Verify Stage 2 codebook exists
CODE_FILE=/home/wlia0047/ar57/wenyu/GeneRec/HG-Rec/dataset/Instruments/Instruments_t5_hrqvae_poincare.npy
if [ ! -f "$CODE_FILE" ]; then
    echo "❌ $CODE_FILE NOT FOUND — Stage 2 not complete" | tee -a $LOG_FILE
    exit 1
fi

# R11.3: 默认 200 epoch + early_stop=20, beam=20 (paper default)
python3 /home/wlia0047/ar57/wenyu/GeneRec/scripts/task84_hgrec_stage3_train.py \
    --dataset_name Instruments \
    --dataset_path /home/wlia0047/ar57/wenyu/GeneRec/HG-Rec/dataset/ \
    --code_path _t5_hrqvae_poincare.npy \
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
    --save_path /home/wlia0047/ar57/wenyu/GeneRec/products/task84/ckpt_hgrec/ \
    --log_path /home/wlia0047/ar57/wenyu/GeneRec/logs/ \
    --seed 42 \
    --early_stop 20 \
    --beam_size 20 \
    --infer_size 96 \
    2>&1 | tee -a "$LOG_FILE" &

TRAIN_PID=$!
echo "$TRAIN_PID" > "$PID_FILE"
echo "===== [Task #84 Stage 3] Training PID: $TRAIN_PID =====" | tee -a $LOG_FILE

wait $TRAIN_PID
EXIT_CODE=$?
echo "===== [Task #84 Stage 3] Training completed at $(date), exit code: $EXIT_CODE =====" | tee -a $LOG_FILE

# R12 验证 ckpt 是否落盘 (R12 fork: 删除旧, 只保留 best)
BEST_CKPT=/home/wlia0047/ar57/wenyu/GeneRec/products/task84/ckpt_hgrec/Instruments/*/HG_Rec_best.pth
if ls $BEST_CKPT 2>/dev/null; then
    echo "✅ R12 best ckpt 落盘 (单一保留): $(ls $BEST_CKPT | head -1)" | tee -a $LOG_FILE
else
    echo "❌ R12 best ckpt MISSING" | tee -a $LOG_FILE
    exit 1
fi

# 清理 PID file
rm -f "$PID_FILE"
