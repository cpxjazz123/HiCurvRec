#!/usr/bin/env bash
# task472_one.sh — single-variant task472 launcher
# Usage: task472_one.sh <name> <h1> <h2> <h3> <o1> <seed>
set -uo pipefail
GRID=/home/wlia0047/ar57/wenyu/GeneRec/GRID
SCRIPT_DIR=$GRID/task_artifacts/scripts/mixed_curvature

name=$1; h1=$2; h2=$3; h3=$4; o1=$5; seed=$6
tag="task472_${name}_s${seed}"
log_dir=$GRID/logs/train/runs/${tag}
log_file=/tmp/${tag}.log

mkdir -p "$log_dir"
nohup python3 "$SCRIPT_DIR/task4_per_layer_train.py" \
    --l1_h_offset=$o1 --l1_h_dim=$h1 \
    --l2_h_offset=$o1 --l2_h_dim=$h2 \
    --l3_h_offset=$o1 --l3_h_dim=$h3 \
    --seed=$seed \
    --task-tag=$tag \
    > "$log_file" 2>&1 &
PID=$!
disown
echo "$PID $tag"
