#!/bin/bash
# Task #279 — K-sweep 扩展 K=512 + K=1024
# 2026-07-29
#
# 复用 task194 框架 (Stage 1 RQ-VAE + Stage 2 Sinkhorn + Stage 3 T5-mini + Stage 4 eval)
# 跑 K ∈ {512, 1024} 2 臂, GPU 0/1 并行.
# 期望: 验证 task194 K=256 (R@10=0.1053) 趋势在更大 K 下是否继续上升.

set -e
REPO=/home/wlia0047/ar57/wenyu/GeneRec
source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh
conda activate /home/wlia0047/ar57_scratch/wenyu/grid_toys
cd $REPO/HG-Rec
export PYTHONPATH=$REPO/HG-Rec:$PYTHONPATH
export HF_HOME=/home/wlia0047/ar57_scratch/wenyu/hf_models

LOG=$REPO/logs/task279/dispatch.log
mkdir -p $REPO/logs/task279 $REPO/products/task279 $REPO/verdicts

echo "[$(date)] Task #279 K-sweep K=512 + K=1024 dispatch 启动" | tee $LOG

# ========================================================================
# Stage 1 — RQ-VAE training (2 臂并行, GPU 0/1)
# ========================================================================
stage1() {
    local K=$1
    local GPU=$2
    local OUT_DIR=$REPO/products/task279/hrqvae_k${K}
    local LOG_FILE=$REPO/logs/task279/stage1_k${K}.log
    local TRITON_CACHE_DIR=/home/wlia0047/.triton/cache_task279_k${K}
    mkdir -p $TRITON_CACHE_DIR $OUT_DIR

    echo "[$(date)] Stage 1: K=$K → GPU $GPU, OUT=$OUT_DIR" | tee -a $LOG
    CUDA_VISIBLE_DEVICES=$GPU TRITON_CACHE_DIR=$TRITON_CACHE_DIR DISABLE_USAGE_KILL=1 \
    python3 -u train_hrqvae.py \
        --lr 1e-3 \
        --epochs 500 \
        --batch_size 1024 \
        --num_workers 4 \
        --eval_step 5 \
        --learner AdamW \
        --lr_scheduler_type linear \
        --warmup_epochs 20 \
        --data_path ./dataset/Instruments/item_emb.parquet \
        --weight_decay 0.0 \
        --dropout_prob 0.0 \
        --loss_type poincare \
        --kmeans_init True \
        --kmeans_iters 1000 \
        --sk_epsilons 0.0 0.0 0.0 \
        --sk_iters 50 \
        --num_emb_list ${K} 128 256 \
        --e_dim 32 \
        --quant_loss_weight 1.0 \
        --beta 0.5 \
        --layers 512 256 128 64 \
        --save_limit 50 \
        --device cuda:0 \
        --ckpt_dir $OUT_DIR \
        > $LOG_FILE 2>&1
    echo "[$(date)] Stage 1 K=$K DONE (exit=$?)" | tee -a $LOG
}

# R12 PID file
PID_FILE=$REPO/products/task279/_STAGE1_PIDS

stage1 512 0 &
PID512=$!
echo "$PID512" > $PID_FILE
echo "[$(date)] Stage 1 K=512 PID=$PID512" | tee -a $LOG

stage1 1024 1 &
PID1024=$!
echo "$PID1024" >> $PID_FILE
echo "[$(date)] Stage 1 K=1024 PID=$PID1024" | tee -a $LOG

# Wait for both Stage 1 main processes to finish
echo "[$(date)] Waiting for 2 Stage 1 main processes to finish..." | tee -a $LOG
while true; do
    CNT=$(ps -eo args | grep "train_hrqvae.py" | grep "/task279/hrqvae_k" | grep -v grep | wc -l)
    echo "[$(date)] train_hrqvae(task279) main processes alive: $CNT" | tee -a $LOG
    if [ "$CNT" -le 0 ]; then
        echo "[$(date)] Stage 1 DONE — proceeding to Stage 2" | tee -a $LOG
        break
    fi
    sleep 120  # check every 2 min
done

# ========================================================================
# Stage 2 — Sinkhorn codebook generation (2 臂, GPU 0/1)
# ========================================================================
echo "[$(date)] Fire Stage 2 — 2 臂 Sinkhorn codebook" | tee -a $LOG
stage2() {
    local K=$1
    local GPU=$2
    local OUT_DIR=$REPO/products/task279/hrqvae_k${K}
    local BEST_CKPT=$(find $OUT_DIR -name "best_collision_model.pth" 2>/dev/null | head -1)
    if [ -z "$BEST_CKPT" ] || [ ! -f "$BEST_CKPT" ]; then
        echo "⚠️ K=$K: no best_collision_model.pth, try best_loss_model.pth" | tee -a $LOG
        BEST_CKPT=$(find $OUT_DIR -name "best_loss_model.pth" 2>/dev/null | head -1)
    fi
    if [ -z "$BEST_CKPT" ] || [ ! -f "$BEST_CKPT" ]; then
        echo "❌ K=$K: no usable ckpt, skip stage 2" | tee -a $LOG
        return 1
    fi
    local SID_OUT=$REPO/products/task279/Instruments_t5_rqvae_k0${K}.npy
    local LOG_FILE=$REPO/logs/task279/stage2_k0${K}.log
    echo "[$(date)] K=$K: Sinkhorn SID from $BEST_CKPT → $SID_OUT" | tee -a $LOG
    CUDA_VISIBLE_DEVICES=$GPU python3 -u $REPO/scripts/task194_stage2_codebook.py \
        --ckpt_path $BEST_CKPT \
        --output_path $SID_OUT \
        --device cuda:0 \
        --max_sinkhorn_iters 30 \
        > $LOG_FILE 2>&1
    echo "[$(date)] Stage 2 K=$K DONE (exit=$?)" | tee -a $LOG
}

stage2 512 0 &
stage2 1024 1 &
wait
echo "[$(date)] Stage 2 all done" | tee -a $LOG

# Copy SID to dataset dir
for K in 512 1024; do
    SID_SRC=$REPO/products/task279/Instruments_t5_rqvae_k0${K}.npy
    if [ -f "$SID_SRC" ]; then
        cp $SID_SRC $REPO/HG-Rec/dataset/Instruments/Instruments_t5_rqvae_k0${K}.npy
        echo "[$(date)] Copied SID to dataset dir: K=$K" | tee -a $LOG
    else
        echo "❌ K=$K: SID missing, abort Stage 3" | tee -a $LOG
        exit 1
    fi
done

# ========================================================================
# Stage 3 — T5-mini training (2 臂并行, GPU 0/1)
# ========================================================================
echo "[$(date)] Fire Stage 3 — 2 臂 T5-mini training" | tee -a $LOG
stage3() {
    local K=$1
    local GPU=$2
    local OUT=$REPO/products/task279/t5mini_k0${K}
    local LOG_FILE=$REPO/logs/task279/stage3_k0${K}.log
    mkdir -p $OUT
    echo "[$(date)] Stage 3: K=$K → GPU $GPU" | tee -a $LOG
    CUDA_VISIBLE_DEVICES=$GPU python3 -u $REPO/scripts/task84_hgrec_stage3_train.py \
        --dataset_name Instruments \
        --code_path _t5_rqvae_k0${K}.npy \
        --codebook_size ${K} 128 256 1 \
        --num_layers 6 \
        --num_decoder_layers 4 \
        --d_model 128 \
        --d_ff 1024 \
        --num_heads 6 \
        --d_kv 64 \
        --vocab_size $((K + 128 + 256 + 1 + 1)) \
        --max_len 20 \
        --num_epochs 200 \
        --batch_size 256 \
        --lr 1e-4 \
        --early_stop 20 \
        --seed 42 \
        --log_path $REPO/logs/task279/ \
        --save_path $OUT \
        --device cuda:0 \
        --mode train \
        > $LOG_FILE 2>&1
    echo "[$(date)] Stage 3 K=$K DONE (exit=$?)" | tee -a $LOG
}

stage3 512 0 &
stage3 1024 1 &
wait
echo "[$(date)] Stage 3 all done" | tee -a $LOG

# ========================================================================
# Stage 4 — R@10 eval (2 臂, GPU 0/1)
# ========================================================================
echo "[$(date)] Fire Stage 4 — 2 臂 R@10 eval" | tee -a $LOG
stage4() {
    local K=$1
    local GPU=$2
    local CKPT=$(find $REPO/products/task279/t5mini_k0${K} -name "HG_Rec_best.pth" 2>/dev/null | head -1)
    local OUTPUT=$REPO/verdicts/task279_k0${K}_test_metrics.json
    local LOG_FILE=$REPO/logs/task279/stage4_k0${K}_eval.out
    if [ -z "$CKPT" ] || [ ! -f "$CKPT" ]; then
        echo "❌ K=$K: no HG_Rec_best.pth, skip Stage 4" | tee -a $LOG
        return 1
    fi
    echo "[$(date)] Stage 4: K=$K → GPU $GPU, ckpt=$CKPT" | tee -a $LOG
    CUDA_VISIBLE_DEVICES=$GPU TRITON_CACHE_DIR=/home/wlia0047/.triton/cache_task279_eval_k${K} \
    python3 -u $REPO/scripts/task278_batch_stage4_eval.py \
        --ckpt_path "$CKPT" \
        --code_path "_t5_rqvae_k0${K}.npy" \
        --codebook_size "${K},128,256,1" \
        --d_model 128 \
        --d_ff 1024 \
        --num_layers 6 \
        --num_decoder_layers 4 \
        --num_heads 6 \
        --d_kv 64 \
        --vocab_size $((K + 128 + 256 + 1 + 1)) \
        --max_len 20 \
        --device cuda:0 \
        --output_path "$OUTPUT" \
        > $LOG_FILE 2>&1
    if [ -f "$OUTPUT" ]; then
        R10=$(python3 -c "import json; print(json.load(open('$OUTPUT')).get('Recall@10', -1))")
        echo "[$(date)] Stage 4 K=$K R@10=$R10" | tee -a $LOG
    fi
}

stage4 512 0 &
stage4 1024 1 &
wait
echo "[$(date)] Stage 4 all done — Task #279 闭环" | tee -a $LOG

# Cleanup PID file
rm -f $REPO/products/task279/_STAGE1_PIDS