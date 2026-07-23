#!/usr/bin/env bash
# task472_multiseed_train.sh — 串行跑 P0.3_v3 / A_v3 / HHHH 三 variant × seed=43/44/45
# 串行避免并行 KMeans CPU 争抢
set -uo pipefail
GRID=/home/wlia0047/ar57/wenyu/GeneRec/GRID
SCRIPT_DIR=$GRID/task_artifacts/scripts/mixed_curvature
LOG_DIR=$GRID/task_artifacts/logs/task472
mkdir -p "$LOG_DIR"

run_train () {
    local name=$1; shift
    local h1=$1; shift
    local h2=$1; shift
    local h3=$1; shift
    local o1=$1; shift
    local seed=$1; shift

    local tag="task472_${name}_s${seed}"
    local log="$LOG_DIR/${tag}.log"

    if [ -f "$GRID/logs/train/runs/${tag}/final.pt" ]; then
        echo "[$tag] final.pt exists, skip"
        return 0
    fi

    echo "[$tag] start seed=$seed (h=($h1,$h2,$h3)) o=$o1"
    python3 "$SCRIPT_DIR/task4_per_layer_train.py" \
        --l1_h_offset=$o1 --l1_h_dim=$h1 \
        --l2_h_offset=$o1 --l2_h_dim=$h2 \
        --l3_h_offset=$o1 --l3_h_dim=$h3 \
        --seed=$seed \
        --task-tag=$tag \
        > "$log" 2>&1
    echo "[$tag] done exit=$?"
}

# P0.3_v3 (text [0:32]): h=(32,0,0), o=0
# A_v3 (no H):       h=(0,0,0),  o=768
# HHHH (brand all):  h=(32,32,32), o=768

# seed=43 先跑
for seed in 43; do
    run_train p0_3 32 0 0 0   $seed
    run_train a     0 0 0 768 $seed
    run_train hhhh 32 32 32 768 $seed
done
echo "task472 done."
