#!/bin/bash
# Task #297 / Issue #25 Gate 3 — Stage 3 T5-mini 200 epoch + Stage 4 eval
# 2026-07-29
#
# 背景: Issue #25 §Gate 0/1/2 全 PASS
#       Gate 3 = T5-mini 200 epoch + R@10 > 0.1020
#       code_path = _t5_hrqvae_issue25_gate2_k0128.npy
#       save_path = products/task297/t5mini_issue25_gate3/
#
# GATE_DECLARATION:
#   gate_3_issue_25: R@10 > 0.1020
#   auto_proceed_after_stage_2: false
#   precedent_override:       forbidden
#
# R7: GPU 1 空闲
# R12: save_limit=1 + best ckpt
# R137: NaN guard (task89 内置)

set -uo pipefail

REPO=/home/wlia0047/ar57/wenyu/GeneRec
cd $REPO

LOG_DIR=$REPO/logs/task297
mkdir -p "$LOG_DIR"

# /tmp/genrec_env 替代 grid_toys
export PYTHONPATH=/tmp/genrec_env:$REPO/HG-Rec:${PYTHONPATH:-}
echo "PYTHONPATH=$PYTHONPATH"
python3 -c "import torch; print('torch=', torch.__version__, 'cuda=', torch.cuda.is_available())" 2>&1 | head -3

CODE_FILE=$REPO/HG-Rec/dataset/Instruments/Instruments_t5_hrqvae_issue25_gate2_k0128.npy
if [ ! -f "$CODE_FILE" ]; then
    echo "❌ $CODE_FILE NOT FOUND — Gate 2 not complete"
    exit 1
fi

SAVE_PATH=$REPO/products/task297/t5mini_issue25_gate3
mkdir -p $SAVE_PATH
TS=$(date +%Y%m%d_%H%M%S)
LOG_FILE=$LOG_DIR/stage3_gate3_${TS}.log

echo "[$(date)] Launching Task #297 Gate 3: T5-mini 200 epoch + Stage 4 eval"
echo "  save_path: $SAVE_PATH"
echo "  code_path: _t5_hrqvae_issue25_gate2_k0128.npy"
echo "  GPU: 1 (R7 空闲)"

TRITON_CACHE_DIR=/home/wlia0047/.triton/cache_task297_gate3
mkdir -p "$TRITON_CACHE_DIR"

CUDA_VISIBLE_DEVICES=1 TRITON_CACHE_DIR="$TRITON_CACHE_DIR" \
  nohup python3 $REPO/scripts/task84_hgrec_stage3_train.py \
    --dataset_name Instruments \
    --dataset_path $REPO/HG-Rec/dataset/ \
    --code_path _t5_hrqvae_issue25_gate2_k0128.npy \
    --codebook_size 128 128 256 1 \
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
    --save_path $SAVE_PATH/ \
    --log_path $REPO/logs/ \
    --seed 42 \
    --early_stop 20 \
    --beam_size 20 \
    --infer_size 96 \
    > "$LOG_FILE" 2>&1 &

TRAIN_PID=$!
echo "$TRAIN_PID" > $REPO/products/task297/_TRAINING_PID_GATE3
echo "[$(date)] Gate 3 Stage 3 training PID=$TRAIN_PID, GPU=1"
echo ""
echo "===== Task #297 Gate 3 Launched ====="
echo "PID=$TRAIN_PID (GPU 1, T5-mini 200 epoch, K=128, Gate 2 codebook)"
echo "Log: $LOG_FILE"
echo "Expected runtime: ~1-1.5h (T5-mini 9.18M × 200 epoch)"
echo "Gate 3 通过条件: R@10 > 0.1020"
