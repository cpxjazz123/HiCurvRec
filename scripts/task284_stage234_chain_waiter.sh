#!/bin/bash
# Task #284 Stage 2/3/4 chain waiter — Stage 1 完成后自动 fire Stage 2 Sinkhorn → Stage 3 T5-mini → Stage 4 eval
# 2026-07-29
#
# 监控 Arm A + Arm B Stage 1 主进程 (PIDs from _TRAINING_PID_A/B)
# 完成后 fire:
#   Stage 2: task144_stage2_codebook.py 改 arm_dir 路径 (跑 task284 armA/armB)
#   Stage 3: task84_hgrec_stage3_train.py (T5-mini 200 ep)
#   Stage 4: task278_batch_stage4_eval.py (v3 CLI args)
#
# R12: Stage 3 训练 ckpt 在 R12 强制保存
# R7: 2 臂并行 GPU 2 (Arm A) + GPU 3 (Arm B)

set -uo pipefail
REPO=/home/wlia0047/ar57/wenyu/GeneRec
export PYTHONPATH="$REPO/HG-Rec:${PYTHONPATH:-}"
export HF_HOME="${HF_HOME:-/home/wlia0047/ar57_scratch/wenyu/hf_models}"
REPO=/home/wlia0047/ar57/wenyu/GeneRec
cd $REPO

LOG_DIR=$REPO/logs/task284
mkdir -p "$LOG_DIR"

LOG=$LOG_DIR/stage234_waiter.log
echo "[$(date)] Task #284 Stage 2/3/4 waiter 启动, 监控 Stage 1 PIDs..." > $LOG

source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh
conda activate /home/wlia0047/ar57_scratch/wenyu/grid_toys

export PYTHONPATH=$REPO/HG-Rec:$PYTHONPATH
export HF_HOME=/home/wlia0047/ar57_scratch/wenyu/hf_models

PID_A_FILE=$REPO/products/task284/_TRAINING_PID_A
PID_B_FILE=$REPO/products/task284/_TRAINING_PID_B
PID_A=$(cat $PID_A_FILE 2>/dev/null)
PID_B=$(cat $PID_B_FILE 2>/dev/null)

echo "[$(date)] PID_A=$PID_A (armA), PID_B=$PID_B (armB)" >> $LOG

# === 阶段 1: 等 Stage 1 两个主进程都完成 ===
while true; do
    A_ALIVE=$(kill -0 $PID_A 2>/dev/null && echo 1 || echo 0)
    B_ALIVE=$(kill -0 $PID_B 2>/dev/null && echo 1 || echo 0)
    echo "[$(date)] Arm A alive=$A_ALIVE, Arm B alive=$B_ALIVE" >> $LOG
    if [ "$A_ALIVE" = "0" ] && [ "$B_ALIVE" = "0" ]; then
        echo "[$(date)] Stage 1 双臂完成, ckpt 落盘 ≥ 60s 后 fire Stage 2" >> $LOG
        sleep 60
        break
    fi
    sleep 120  # check every 2 min
done

# === 阶段 2: Stage 2 Sinkhorn codebook (2 臂并行) ===
echo "[$(date)] === Fire Stage 2 Sinkhorn (2 臂并行) ===" >> $LOG

# Arm A: K=256 num_emb_list
SID_A=$REPO/HG-Rec/dataset/Instruments/Instruments_t5_hrqvae_kappa_decouple_armA_k0256.npy
LOG_2A=$LOG_DIR/stage2_armA.log
CKPT_A=$(find $REPO/products/task284/hrqvae_k0256_armA_phaseA_only -name "best_loss_model.pth" | head -1)
if [ -z "$CKPT_A" ]; then
    echo "[$(date)] ❌ Arm A: best_loss_model.pth 不存在, skip Stage 2/3/4" >> $LOG
else
    echo "[$(date)] Arm A Stage 2: $CKPT_A → $SID_A" >> $LOG
    CUDA_VISIBLE_DEVICES=2 TRITON_CACHE_DIR=/home/wlia0047/.triton/cache_task284_armA_stage2 \
    python3 -u $REPO/scripts/task284_stage2_free_curv.py \
        --ckpt_path "$CKPT_A" \
        --output_path "$SID_A" \
        --device cuda:0 \
        --max_sinkhorn_iters 30 \
        > $LOG_2A 2>&1 &
    PID_2A=$!
    echo "[$(date)] Arm A Stage 2 PID=$PID_2A" >> $LOG
fi

# Arm B: K=256 num_emb_list
SID_B=$REPO/HG-Rec/dataset/Instruments/Instruments_t5_hrqvae_kappa_decouple_armB_k0256.npy
LOG_2B=$LOG_DIR/stage2_armB.log
CKPT_B=$(find $REPO/products/task284/hrqvae_k0256_armB_decouple -name "best_loss_model.pth" | head -1)
if [ -z "$CKPT_B" ]; then
    echo "[$(date)] ❌ Arm B: best_loss_model.pth 不存在, skip Stage 2/3/4" >> $LOG
else
    echo "[$(date)] Arm B Stage 2: $CKPT_B → $SID_B" >> $LOG
    CUDA_VISIBLE_DEVICES=3 TRITON_CACHE_DIR=/home/wlia0047/.triton/cache_task284_armB_stage2 \
    python3 -u $REPO/scripts/task284_stage2_free_curv.py \
        --ckpt_path "$CKPT_B" \
        --output_path "$SID_B" \
        --device cuda:0 \
        --max_sinkhorn_iters 30 \
        > $LOG_2B 2>&1 &
    PID_2B=$!
    echo "[$(date)] Arm B Stage 2 PID=$PID_2B" >> $LOG
fi

# 等 Stage 2 完成 (R11 fix: 用 ${VAR:-} 防 unbound)
PIDS_2=""
[ -n "${PID_2A:-}" ] && PIDS_2="$PIDS_2 $PID_2A"
[ -n "${PID_2B:-}" ] && PIDS_2="$PIDS_2 $PID_2B"
if [ -n "$PIDS_2" ]; then
    wait $PIDS_2 2>/dev/null
fi
echo "[$(date)] === Stage 2 完成 ===" >> $LOG

# === 阶段 3: Stage 3 T5-mini training (2 臂并行 GPU 2 + GPU 3) ===
echo "[$(date)] === Fire Stage 3 T5-mini (2 臂并行 GPU 2 + GPU 3) ===" >> $LOG

# Arm A: codebook_size=[256,128,256,1] d_model=128
if [ -f "$SID_A" ]; then
    LOG_3A=$LOG_DIR/stage3_armA.log
    rm -f $LOG_3A
    echo "[$(date)] Arm A Stage 3: T5-mini training on cuda:2" >> $LOG
    CUDA_VISIBLE_DEVICES=2 TRITON_CACHE_DIR=/home/wlia0047/.triton/cache_task284_armA_stage3 \
    python3 -u $REPO/scripts/task84_hgrec_stage3_train.py \
        --dataset_name Instruments \
        --dataset_path $REPO/HG-Rec/dataset/ \
        --code_path _t5_hrqvae_kappa_decouple_armA_k0256.npy \
        --codebook_size 256 128 256 1 \
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
        --save_path $REPO/products/task284/t5mini_armA \
        --log_path $LOG_DIR/ \
        --seed 42 \
        --early_stop 20 \
        --beam_size 20 \
        --infer_size 96 \
        > $LOG_3A 2>&1 &
    PID_3A=$!
    echo "[$(date)] Arm A Stage 3 PID=$PID_3A" >> $LOG
fi

# Arm B: codebook_size=[256,128,256,1] d_model=128
if [ -f "$SID_B" ]; then
    LOG_3B=$LOG_DIR/stage3_armB.log
    rm -f $LOG_3B
    echo "[$(date)] Arm B Stage 3: T5-mini training on cuda:3" >> $LOG
    CUDA_VISIBLE_DEVICES=3 TRITON_CACHE_DIR=/home/wlia0047/.triton/cache_task284_armB_stage3 \
    python3 -u $REPO/scripts/task84_hgrec_stage3_train.py \
        --dataset_name Instruments \
        --dataset_path $REPO/HG-Rec/dataset/ \
        --code_path _t5_hrqvae_kappa_decouple_armB_k0256.npy \
        --codebook_size 256 128 256 1 \
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
        --save_path $REPO/products/task284/t5mini_armB \
        --log_path $LOG_DIR/ \
        --seed 42 \
        --early_stop 20 \
        --beam_size 20 \
        --infer_size 96 \
        > $LOG_3B 2>&1 &
    PID_3B=$!
    echo "[$(date)] Arm B Stage 3 PID=$PID_3B" >> $LOG
fi

# 等 Stage 3 完成 (R11 fix: 用 ${VAR:-} 防 unbound, 单臂失败不阻塞另一臂)
PIDS_3=""
[ -n "${PID_3A:-}" ] && PIDS_3="$PIDS_3 $PID_3A"
[ -n "${PID_3B:-}" ] && PIDS_3="$PIDS_3 $PID_3B"
if [ -n "$PIDS_3" ]; then
    wait $PIDS_3 2>/dev/null
fi
echo "[$(date)] === Stage 3 完成 ===" >> $LOG

# === 阶段 4: Stage 4 eval (复用 task278_batch_stage4_eval.py v3) ===
echo "[$(date)] === Fire Stage 4 eval (2 臂) ===" >> $LOG

# Arm A Stage 4
CKPT_A_HG=$(find $REPO/products/task284/t5mini_armA -name "HG_Rec_best.pth" | head -1)
if [ -n "$CKPT_A_HG" ]; then
    LOG_4A=$LOG_DIR/stage4_armA_eval.out
    OUTPUT_A=$REPO/verdicts/task284_armA_test_metrics.json
    echo "[$(date)] Arm A Stage 4: $CKPT_A_HG" >> $LOG
    CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/home/wlia0047/.triton/cache_task284_armA_stage4 \
    python3 -u $REPO/scripts/task278_batch_stage4_eval.py \
        --ckpt_path "$CKPT_A_HG" \
        --code_path _t5_hrqvae_kappa_decouple_armA_k0256.npy \
        --codebook_size 256 128 256 1 \
        --vocab_size 1025 \
        --d_model 128 \
        --d_ff 1024 \
        --num_layers 6 \
        --num_decoder_layers 4 \
        --num_heads 6 \
        --d_kv 64 \
        --max_len 20 \
        --device cuda:0 \
        --output_path "$OUTPUT_A" \
        > $LOG_4A 2>&1
    echo "[$(date)] Arm A Stage 4 done, R@10=$(python3 -c "import json; print(json.load(open('$OUTPUT_A'))['Recall@10'])" 2>/dev/null)" >> $LOG
fi

# Arm B Stage 4
CKPT_B_HG=$(find $REPO/products/task284/t5mini_armB -name "HG_Rec_best.pth" | head -1)
if [ -n "$CKPT_B_HG" ]; then
    LOG_4B=$LOG_DIR/stage4_armB_eval.out
    OUTPUT_B=$REPO/verdicts/task284_armB_test_metrics.json
    echo "[$(date)] Arm B Stage 4: $CKPT_B_HG" >> $LOG
    CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/home/wlia0047/.triton/cache_task284_armB_stage4 \
    python3 -u $REPO/scripts/task278_batch_stage4_eval.py \
        --ckpt_path "$CKPT_B_HG" \
        --code_path _t5_hrqvae_kappa_decouple_armB_k0256.npy \
        --codebook_size 256 128 256 1 \
        --vocab_size 1025 \
        --d_model 128 \
        --d_ff 1024 \
        --num_layers 6 \
        --num_decoder_layers 4 \
        --num_heads 6 \
        --d_kv 64 \
        --max_len 20 \
        --device cuda:0 \
        --output_path "$OUTPUT_B" \
        > $LOG_4B 2>&1
    echo "[$(date)] Arm B Stage 4 done, R@10=$(python3 -c "import json; print(json.load(open('$OUTPUT_B'))['Recall@10'])" 2>/dev/null)" >> $LOG
fi

echo "[$(date)] === Task #284 Stage 2/3/4 全套完成 ===" >> $LOG
