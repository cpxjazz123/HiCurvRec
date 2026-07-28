#!/bin/bash
# Task #233 Stage 3 — dual_v5 Stage 3 RERUN with heartbeat + epoch-level logging
# Issue #8 (Task #233): 验证 dual_v5 -10.3% R@10 drop 是否为 truncation artifact
# 配置 跟 task200 完全一致 (T5-mini 9.18M, 200 epoch, early_stop=20, dual_v5 SID)
# 改进:
#   1. Heartbeat (每 30s 写 products/task233/_HEARTBEAT)
#   2. 训练脚本开头打印 GPU 状态 + PID + epoch start time
#   3. early_stop counter 每个 epoch 都打印 (task84_hgrec_stage3_train.py 已有 line 255, 确认)

set -e
REPO=/home/wlia0047/wenyu/GeneRec 2>/dev/null || REPO=/home/wlia0047/wenyu/GeneRec
REPO=/home/wlia0047/ar57/wenyu/GeneRec
cd $REPO
source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh
conda activate /home/wlia0047/ar57_scratch/wenyu/grid_toys
cd $REPO/HG-Rec
export PYTHONPATH=$REPO/HG-Rec:$PYTHONPATH
export HF_HOME=/home/wlia0047_scratch/wenyu/hf_models 2>/dev/null
export HF_HOME=/home/wlia0047/ar57_scratch/wenyu/hf_models
export CUDA_VISIBLE_DEVICES=0

# === 关键: R12 + heartbeat 用专属 TRITON cache ===
export TRITON_CACHE_DIR=/home/wlia0047/.triton/cache_task233
mkdir -p $TRITON_CACHE_DIR

mkdir -p $REPO/logs/task233 $REPO/products/task233

# dual_v5 SID 已存在 (task200 推断产物, Jul 26 01:59 落盘)
SID_ACTUAL="Instruments_t5_rqvae_dual_v5.npy"
if [ ! -f "$REPO/HG-Rec/dataset/Instruments/$SID_ACTUAL" ]; then
    echo "❌ dual_v5 SID file MISSING: $REPO/HG-Rec/dataset/Instruments/$SID_ACTUAL"
    echo "   需先跑 task200_stage2_codebook.py --ckpt_path .../best_collision_model.pth --output_suffix dual_v5"
    exit 1
fi
echo "✅ dual_v5 SID file present: $(ls -la $REPO/HG-Rec/dataset/Instruments/$SID_ACTUAL | awk '{print $5, $9}')"

DATA_DIR=$REPO/HG-Rec/dataset
SID_FILE="_t5_rqvae_dual_v5.npy"
GPU=0
SAVE_DIR=$REPO/products/task233/t5mini_dual_v5_rerun
LOG_FILE=$REPO/logs/task233/stage3_dual_v5_rerun.out
HEARTBEAT_FILE=$REPO/products/task233/_HEARTBEAT
PID_FILE=$REPO/products/task233/_TRAINING_PID

# 清理 (避免干扰)
rm -f $LOG_FILE $HEARTBEAT_FILE $PID_FILE
mkdir -p $SAVE_DIR

echo "[$(date +%Y-%m-%d_%H:%M:%S)] === Task #233 Stage 3 dual_v5 RERUN (Issue #8) ===" | tee -a $LOG_FILE
echo "[$(date +%Y-%m-%d_%H:%M:%S)] GPU=$GPU  Save dir=$SAVE_DIR  Sid=$SID_FILE" | tee -a $LOG_FILE
echo "[$(date +%Y-%m-%d_%H:%M:%S)] Pre-launch GPU state:" | tee -a $LOG_FILE
nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader 2>&1 | tee -a $LOG_FILE

# === 启动 Stage 3 训练 (后台) ===
CUDA_VISIBLE_DEVICES=$GPU python3 -u $REPO/scripts/task84_hgrec_stage3_train.py \
    --dataset_name Instruments --dataset_path $DATA_DIR/ \
    --code_path $SID_FILE \
    --codebook_size 64 128 256 1 \
    --num_epochs 200 --batch_size 256 --lr 1e-4 \
    --num_layers 6 --num_decoder_layers 4 --d_model 128 --d_ff 1024 \
    --num_heads 6 --d_kv 64 --vocab_size 1025 --max_len 20 \
    --pad_token_id 0 --eos_token_id 0 \
    --device cuda:0 --mode train \
    --save_path $SAVE_DIR \
    --log_path $REPO/logs/task233/ \
    --seed 42 --early_stop 20 --beam_size 20 --infer_size 96 \
    > $LOG_FILE 2>&1 &
TRAIN_PID=$!
echo $TRAIN_PID > $PID_FILE

echo "[$(date +%Y-%m-%d_%H:%M:%S)] Stage 3 launched PID=$TRAIN_PID  heartbeat=$HEARTBEAT_FILE" | tee -a $LOG_FILE

# === Heartbeat: 每 30s 写时间戳 + GPU util + ckpt state ===
# 持续 4 小时 (480 个 30s = 14400s, 设上限 5400 = 90 min 保险), 在 PID alive 期间循环
START_TS=$(date +%s)
DEADLINE_TS=$((START_TS + 14400))   # 4h 上限

while true; do
    NOW_TS=$(date +%s)
    if [ $NOW_TS -gt $DEADLINE_TS ]; then
        echo "[heartbeat] 4h 超时, 自动停掉监控 (训练可能仍在跑, 独立 PID)"
        break
    fi
    if ! kill -0 $TRAIN_PID 2>/dev/null; then
        # 进程已退出
        echo "[heartbeat $(date +%Y-%m-%d_%H:%M:%S)] PID $TRAIN_PID 已退出. 监控结束." > $HEARTBEAT_FILE
        echo "[$(date +%Y-%m-%d_%H:%M:%S)] Stage 3 PID $TRAIN_PID 退出 (完成或崩溃)" | tee -a $LOG_FILE
        break
    fi
    # 写 heartbeat (含 GPU 状态 + ckpt 状态)
    GPU_LINE=$(nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader 2>&1 | head -1)
    CKPT_LINE=$(ls -la $SAVE_DIR/Instruments/*/HG_Rec_best.pth 2>/dev/null | tail -1 || echo "no_ckpt_yet")
    EPOCH_PROGRESS=$(grep -E "^[0-9]+,[0-9]+,.*[0-9]+,[0-9]+$" $LOG_FILE 2>/dev/null | tail -1 || echo "")
    cat > $HEARTBEAT_FILE <<EOF
heartbeat_ts=$(date +%Y-%m-%d_%H:%M:%S)
pid=$TRAIN_PID alive=true
gpu=$GPU_LINE
ckpt=$CKPT_LINE
elapsed_sec=$((NOW_TS - START_TS))
EOF
    sleep 30
done

echo "[$(date +%Y-%m-%d_%H:%M:%S)] === Heartbeat 监控结束 ===" | tee -a $LOG_FILE

# 最终 ckpt 状态
echo "[$(date +%Y-%m-%d_%H:%M:%S)] Final ckpt:" | tee -a $LOG_FILE
find $SAVE_DIR -name "HG_Rec_best.pth" -exec ls -la {} \; 2>&1 | tee -a $LOG_FILE
echo "[$(date +%Y-%m-%d_%H:%M:%S)] === Task #233 Stage 3 RERUN 监控完结 ===" | tee -a $LOG_FILE
