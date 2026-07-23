#!/bin/bash
# P4 launcher — 3 methods × 3 seeds = 9 runs on 4 GPUs
# Each method uses a different semantic_id_path
# All use tiger_decoder_only_train_flat_p4.yaml (patience=8, user-disjoint split, val_check=100)
#
# Run matrix:
#   GPU0: baseline s42 (during slot 1), then HRQ s43 (slot 2), then AQ s44 (slot 3)
#   GPU1: baseline s43 → HRQ s44 → AQ s42
#   GPU2: baseline s44 → HRQ s42 → AQ s43
#   GPU3: spillover if any GPU finishes early
#
# Each method uses 1 GPU per run.

set -e
cd /home/wlia0047/ar57/wenyu/GeneRec/GRID
source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh
conda activate /home/wlia0047/ar57_scratch/wenyu/grid_toys

DATA_DIR=/home/wlia0047/ar57/wenyu/GeneRec/GRID/data/amazon_data/toys
RUN_BASE=/home/wlia0047/ar57/wenyu/GeneRec/GRID/logs/train/runs
P4_OUT=/home/wlia0047/ar57/wenyu/GeneRec/GRID/result/p4_3seed_v2

# Method -> SID path (Stage 2 merged_predictions_tensor.pt format: (4, 11924))
declare -A SID_PATHS=(
    ["baseline"]="logs/inference/runs/task13_group_a_s22/pickle/merged_predictions_tensor.pt"
    ["HRQ"]="logs/inference/runs/task18_hrq_s22/pickle/merged_predictions_tensor.pt"
    ["AQ"]="logs/inference/runs/task19_aq_s22/pickle/merged_predictions_tensor.pt"
)

# Build all 9 (method, seed) jobs
JOBS=()
for METHOD in baseline HRQ AQ; do
    for SEED in 42 43 44; do
        JOBS+=("$METHOD:$SEED")
    done
done

# Shuffle to balance GPU load (assign jobs round-robin)
echo "=== P4 9-run launcher ==="
echo "Total jobs: ${#JOBS[@]}"
echo "Methods: baseline / HRQ / AQ"
echo "Seeds: 42 / 43 / 44"
echo "GPUs: 0-3 (4 in parallel)"
echo ""

mkdir -p "$P4_OUT"

# Function to run one training
run_one() {
    local METHOD=$1
    local SEED=$2
    local GPU_ID=$3
    local SID_PATH=${SID_PATHS[$METHOD]}
    local TAG="p4_${METHOD}_s${SEED}"
    local EXPERIMENT=tiger_decoder_only_train_flat_p4
    local OUTDIR="$P4_OUT/${TAG}"

    echo "[$(date +%H:%M:%S)] GPU$GPU_ID start: $TAG (sid=$SID_PATH, seed=$SEED)"

    mkdir -p "$OUTDIR"
    CUDA_VISIBLE_DEVICES=$GPU_ID python -m src.train \
        experiment=$EXPERIMENT \
        data_dir=$DATA_DIR \
        semantic_id_path=$SID_PATH \
        num_hierarchies=4 \
        seed=$SEED \
        paths.output_dir=$OUTDIR \
        "++trainer.devices=1" \
        "++trainer.strategy=auto" \
        2>&1 | tee "$OUTDIR/train.log"

    echo "[$(date +%H:%M:%S)] GPU$GPU_ID done: $TAG"
}

# Launch all 9 jobs in parallel, max 4 at a time
GPU_ID=0
PIDS=()
JOB_IDX=0

for METHOD in baseline HRQ AQ; do
    for SEED in 42 43 44; do
        # Wait if 4 jobs already running
        while [ ${#PIDS[@]} -ge 4 ]; do
            NEW_PIDS=()
            for PID in "${PIDS[@]}"; do
                if kill -0 $PID 2>/dev/null; then
                    NEW_PIDS+=($PID)
                fi
            done
            PIDS=("${NEW_PIDS[@]}")
            if [ ${#PIDS[@]} -ge 4 ]; then
                sleep 30
            fi
        done
        # Launch this job
        run_one $METHOD $SEED $GPU_ID &
        PIDS+=($!)
        # Rotate GPU assignment
        GPU_ID=$(( (GPU_ID + 1) % 4 ))
        JOB_IDX=$((JOB_IDX + 1))
    done
done

# Wait for all to finish
echo ""
echo "=== Waiting for all 9 jobs to finish ==="
wait
echo "=== All 9 jobs done ==="