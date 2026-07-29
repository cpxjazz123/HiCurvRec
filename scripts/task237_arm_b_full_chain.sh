#!/bin/bash
# ⚠️  [Issue #19 Gate 2 annotation 2026-07-29]
# 本脚本不含闸门求值点 (Stage 2 后 → Stage 3 前无任何 evaluate_stage_2_gate 调用).
# 按 Issue #19 §Gate 2 规则, **不得用于任何声明了 Stage 2 级闸门的 issue** 执行.
# 若新 issue 需要本脚本的链式逻辑, 必须先在 Stage 2 推断**结束后**插入
# `source scripts/issue19_gate_template.sh && evaluate_stage_2_gate <diag.json>` 调用,
# 不通过则不启动 Stage 3. See verdicts/task281_issue19_gate0_replay.md 完整清单.
#
# GATE_DECLARATION:
#   section_6_7_4_stop_loss_i: bound
#   stage_2_threshold:        15
#   stage_3_threshold:        NA
#   auto_proceed_after_stage_2: false
#   precedent_override:       forbidden
# (Issue #19 + Issue #21 双闸门: Stage 2 collision 差 ≥ 15pp + §6.7.4 stop-loss (i) L0 ≥ 90%)
#
# Chain: partial Sinkhorn max_iters=10 → T5-mini 200 epoch → eval → slice
# Wall-clock: ~50min (Stage 3 = ~44min is bottleneck)
set -e
REPO=/home/wlia0047/ar57/wenyu/GeneRec
source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh
conda activate /home/wlia0047/ar57_scratch/wenyu/grid_toys
cd $REPO/HG-Rec
export PYTHONPATH=$REPO/HG-Rec:$PYTHONPATH
export HF_HOME=/home/wlia0047/ar57_scratch/wenyu/hf_models
export CUDA_VISIBLE_DEVICES=0
export TRITON_CACHE_DIR=/home/wlia0047/.triton/cache_task237

mkdir -p $REPO/products/task237 $REPO/logs/task237

# === Stage 2: partial Sinkhorn (max_iters=10) ===
CKPT=$REPO/products/task84/ckpt/Instruments/Jul-23-2026_20-08-06_beta_1.000_codebook_[64,128,256]_sk_0.000/best_loss_model.pth
SID_OUT=$REPO/HG-Rec/dataset/Instruments/Instruments_t5_rqvae_task237_armB.npy
LOG_STAGE2=$REPO/logs/task237/stage2_codebook.out

echo "[$(date +%Y-%m-%d_%H:%M:%S)] === Task #237 Arm B Stage 2: partial Sinkhorn max_iters=10 ===" | tee $LOG_STAGE2

CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/home/wlia0047/.triton/cache_task237_stage2 \
python3 -u $REPO/scripts/task223_stage2_codebook.py \
    --ckpt_path "$CKPT" \
    --output_path "$SID_OUT" \
    --device cuda:0 \
    --max_sinkhorn_iters 10 \
    >> $LOG_STAGE2 2>&1

# Copy SID file to expected T5-mini Stage 3 path
SID_FILE_BASE="_t5_rqvae_task237_armB.npy"
SID_FILE_PATH=$REPO/HG-Rec/dataset/Instruments/$SID_FILE_BASE
cp "$SID_OUT" "$SID_FILE_PATH"

# Extract Stage 2 collision_rate
COLLISION_LINE=$(grep -E "Final.*collision_rate|Final\s*collision" $LOG_STAGE2 | tail -1)
COLLISION_RATE=$(echo "$COLLISION_LINE" | grep -oE "[0-9]+\.[0-9]+" | tail -1)
echo "[$(date +%Y-%m-%d_%H:%M:%S)] Stage 2 done. collision_rate = $COLLISION_RATE (vs Arm A=0.99, vs Arm C=0.05)"
echo "$COLLISION_RATE" > $REPO/products/task237/_STAGE2_COLLISION

# === Stage 3: T5-mini training ===
DATA_DIR=$REPO/HG-Rec/dataset
SAVE_DIR=$REPO/products/task237/t5mini_armB
LOG_STAGE3=$REPO/logs/task237/stage3_armB.out
HEARTBEAT_FILE=$REPO/products/task237/_HEARTBEAT
PID_FILE=$REPO/products/task237/_TRAINING_PID

rm -f $LOG_STAGE3 $HEARTBEAT_FILE $PID_FILE
mkdir -p $SAVE_DIR

echo "[$(date +%Y-%m-%d_%H:%M:%S)] === Stage 3 T5-mini training (R12 + heartbeat) ===" | tee $LOG_STAGE3

CUDA_VISIBLE_DEVICES=0 python3 -u $REPO/scripts/task84_hgrec_stage3_train.py \
    --dataset_name Instruments --dataset_path $DATA_DIR/ \
    --code_path $SID_FILE_BASE \
    --codebook_size 64 128 256 1 \
    --num_epochs 200 --batch_size 256 --lr 1e-4 \
    --num_layers 6 --num_decoder_layers 4 --d_model 128 --d_ff 1024 \
    --num_heads 6 --d_kv 64 --vocab_size 1025 --max_len 20 \
    --pad_token_id 0 --eos_token_id 0 \
    --device cuda:0 --mode train \
    --save_path $SAVE_DIR \
    --log_path $REPO/logs/task237/ \
    --seed 42 --early_stop 20 --beam_size 20 --infer_size 96 \
    > $LOG_STAGE3 2>&1 &
TRAIN_PID=$!
echo $TRAIN_PID > $PID_FILE

echo "[$(date +%Y-%m-%d_%H:%M:%S)] Stage 3 launched PID=$TRAIN_PID" | tee -a $LOG_STAGE3

# Heartbeat
START_TS=$(date +%s)
DEADLINE_TS=$((START_TS + 14400))

while true; do
    NOW_TS=$(date +%s)
    if [ $NOW_TS -gt $DEADLINE_TS ]; then
        echo "[heartbeat] 4h timeout" >> $HEARTBEAT_FILE
        break
    fi
    if ! kill -0 $TRAIN_PID 2>/dev/null; then
        echo "[heartbeat $(date +%Y-%m-%d_%H:%M:%S)] PID $TRAIN_PID exited" > $HEARTBEAT_FILE
        break
    fi
    GPU_LINE=$(nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader 2>&1 | head -1)
    CKPT_LINE=$(ls -la $SAVE_DIR/Instruments/*/HG_Rec_best.pth 2>/dev/null | tail -1 || echo "no_ckpt_yet")
    cat > $HEARTBEAT_FILE <<EOF
heartbeat_ts=$(date +%Y-%m-%d_%H:%M:%S)
pid=$TRAIN_PID alive=true
gpu=$GPU_LINE
ckpt=$CKPT_LINE
elapsed_sec=$((NOW_TS - START_TS))
EOF
    sleep 60
done

echo "[$(date +%Y-%m-%d_%H:%M:%S)] === Stage 3 done ===" | tee -a $LOG_STAGE3
find $SAVE_DIR -name "HG_Rec_best.pth" -exec ls -la {} \; 2>&1 | tee -a $LOG_STAGE3

echo "[$(date +%Y-%m-%d_%H:%M:%S)] === Task #237 Arm B full chain end (next tick 跑 Stage 4 + slice) ==="
