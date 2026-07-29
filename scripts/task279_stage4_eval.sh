#!/bin/bash
# Task #279 — Stage 4 eval after K=512 + K=1024 Stage 3 complete
# 2026-07-29
#
# Wait for both HG_Rec_best.pth to appear (Stage 3 done), then run Stage 4 eval.

set -e
REPO=/home/wlia0047/ar57/wenyu/GeneRec
source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh
conda activate /home/wlia0047/ar57_scratch/wenyu/grid_toys
cd $REPO
export PYTHONPATH=$REPO/HG-Rec:$PYTHONPATH

mkdir -p $REPO/verdicts

K0512_CKPT="$REPO/products/task279/t5mini_k0512/Instruments/Jul-29-2026_12-53-18/HG_Rec_best.pth"
K1024_CKPT="$REPO/products/task279/t5mini_k01024/Instruments/Jul-29-2026_12-58-17/HG_Rec_best.pth"

echo "[$(date)] Task #279 Stage 4 eval 启动, 等待 K=512 + K=1024 Stage 3 ckpts..."

# Wait for both ckpts to be saved
while true; do
    K0512_AGE=""
    K1024_AGE=""
    [ -f "$K0512_CKPT" ] && K0512_AGE=$(($(date +%s) - $(stat -c %Y "$K0512_CKPT")))
    [ -f "$K1024_CKPT" ] && K1024_AGE=$(($(date +%s) - $(stat -c %Y "$K1024_CKPT")))
    echo "[$(date)] K=512 ckpt age=${K0512_AGE}s, K=1024 ckpt age=${K1024_AGE}s"

    # Wait until both Stage 3 done (no python3 process for task279 t5mini)
    ALIVE=$(ps -eo args | grep "task84_hgrec_stage3_train.py" | grep "/task279/t5mini" | grep -v grep | wc -l)
    echo "[$(date)] Stage 3 train processes alive: $ALIVE"

    if [ "$ALIVE" -le 0 ] && [ -f "$K0512_CKPT" ] && [ -f "$K1024_CKPT" ]; then
        # Both ckpts exist and no training alive
        if [ -n "$K0512_AGE" ] && [ "$K0512_AGE" -gt 60 ] && [ -n "$K1024_AGE" ] && [ "$K1024_AGE" -gt 60 ]; then
            echo "[$(date)] Both Stage 3 done, both ckpts saved > 60s ago. Proceed Stage 4."
            break
        fi
    fi
    sleep 120
done

# Stage 4 eval — 2 arms in parallel
stage4() {
    local K=$1
    local GPU=$2
    local CKPT=$3
    local VOCAB=$4
    local OUTPUT="$REPO/verdicts/task279_k0${K}_test_metrics.json"
    local LOG="$REPO/logs/task279/stage4_k0${K}_eval.out"
    echo "[$(date)] Stage 4 K=$K → GPU $GPU"
    CUDA_VISIBLE_DEVICES=$GPU TRITON_CACHE_DIR=/home/wlia0047/.triton/cache_task279_eval_k${K} \
    python3 -u $REPO/scripts/task278_batch_stage4_eval.py \
        --ckpt_path "$CKPT" \
        --code_path "_t5_rqvae_k0${K}.npy" \
        --codebook_size "${K},128,256,1" \
        --vocab_size $VOCAB \
        --d_model 128 \
        --d_ff 1024 \
        --num_layers 6 \
        --num_decoder_layers 4 \
        --num_heads 6 \
        --d_kv 64 \
        --max_len 20 \
        --device cuda:0 \
        --output_path "$OUTPUT" \
        > $LOG 2>&1
    if [ -f "$OUTPUT" ]; then
        R10=$(python3 -c "import json; print(json.load(open('$OUTPUT')).get('Recall@10', -1))")
        echo "[$(date)] Stage 4 K=$K R@10=$R10"
    fi
}

stage4 512 0 "$K0512_CKPT" 1025 &
stage4 1024 1 "$K1024_CKPT" 1410 &
wait
echo "[$(date)] Task #279 Stage 4 全部完成"