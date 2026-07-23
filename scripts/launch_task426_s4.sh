#!/bin/bash
# Task 426 / 427 — Stage 4 inference + item-level eval for both ablation runs.

set -euo pipefail
source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh
conda activate /home/wlia0047/ar57_scratch/wenyu/grid_toys

cd /fs04/ar57/wenyu/GeneRec/GRID/.claude/worktrees/task9-decoder-only

GRID=/fs04/ar57/wenyu/GeneRec/GRID
LOG_DIR=$GRID/task_artifacts/scripts/logs
DATA_DIR=$GRID/data/amazon_data/toys
OUT_RESULTS=$GRID/task_artifacts/results/exp388v5/task426_dual_ablation
mkdir -p $OUT_RESULTS

run_s4_eval () {
    local ID=$1
    local SID=$2
    local GPU=$3
    local SAFE_CKPT=$OUT_RESULTS/${ID}_tiger.ckpt
    local S4_PRED=$GRID/logs/inference/runs/${ID}_s4/pickle/merged_predictions_tensor.pt
    local EVAL_OUT=$OUT_RESULTS/${ID}_eval.json

    # Pick BEST ckpt
    local S3_CKPT_DIR=$GRID/logs/train/runs/${ID}/checkpoints
    local BEST_CKPT=$(ls -d ${S3_CKPT_DIR}/checkpoint_epoch*.ckpt 2>/dev/null | sort -t= -k3 -n | tail -1)
    if [ -z "$BEST_CKPT" ]; then
        echo "ERROR: no checkpoint found in $S3_CKPT_DIR"
        return 1
    fi

    cp -f "$BEST_CKPT" "$SAFE_CKPT"
    echo "[$ID] BEST=$BEST_CKPT -> $SAFE_CKPT"

    # Stage 4 inference
    CUDA_VISIBLE_DEVICES=${GPU} python -m src.inference \
        experiment=tiger_inference_flat \
        data_dir=${DATA_DIR} \
        semantic_id_path=${SID} \
        num_hierarchies=4 \
        ckpt_path=${SAFE_CKPT} \
        ++should_skip_retry=true \
        paths.root_dir=${GRID} \
        id=${ID}_s4 \
        2>&1 | tee $LOG_DIR/${ID}_s4.log

    # Item-level R@5/10/NDCG eval
    python $GRID/.claude/worktrees/task9-decoder-only/task_artifacts/scripts/task388v4_s4_item_eval.py \
        --variant X_X_X_${ID} \
        --constrained_pt ${S4_PRED} \
        --sid ${SID} \
        --ckpt ${SAFE_CKPT} \
        --out_json ${EVAL_OUT} \
        2>&1 | tee $LOG_DIR/${ID}_s4_eval.log
}

# A: H_E_E_E L1 flattened
SID_A=$GRID/task_artifacts/results/exp388v5/task426_dual_ablation/sid/sid_H_EEE_L1_flatten.pt
run_s4_eval task426_A_s3 $SID_A 0

# B: baseline L1 dispersed
SID_B=$GRID/task_artifacts/results/exp388v5/task426_dual_ablation/sid/sid_baseline_L1_disperse.pt
run_s4_eval task427_B_s3_rerun $SID_B 1
