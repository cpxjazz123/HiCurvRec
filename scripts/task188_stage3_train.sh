#!/bin/bash
# Task #188 Phase 3: Stage 3 T5-small training × 4 codebook × 3 seed = 12 runs
# Use 4 GPU parallel (cuda:0/1/2/3) - one run per GPU at a time.
# Each run ~ 4-7 min. 3 rounds, ~15-20 min total parallel.
set -e
source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh
conda activate /home/wlia0047/ar57_scratch/wenyu/grid_toys

REPO=/home/wlia0047/ar57/wenyu/GeneRec
export PYTHONPATH=$REPO/HG-Rec:$PYTHONPATH
export HF_HOME=/home/wlia0047/ar57_scratch/wenyu/hf_models
export TRITON_CACHE_DIR=/home/wlia0047/.triton/cache_task188_stage3

# Codebook paths (4 tiers)
declare -A CB_PATHS
CB_PATHS[t1_8pct]=$REPO/HG-Rec/dataset/Instruments/Instruments_t5_rqvae_t1_8pct.npy
CB_PATHS[t2_10pct]=$REPO/HG-Rec/dataset/Instruments/Instruments_t5_rqvae_t2_10pct.npy
CB_PATHS[t3_12pct]=$REPO/HG-Rec/dataset/Instruments/Instruments_t5_rqvae_t3_12pct.npy
CB_PATHS[t4_13pct]=$REPO/HG-Rec/dataset/Instruments/Instruments_t5_rqvae_t4_13pct.npy

# 3 seeds
SEEDS=(42 123 2024)

# 4 GPUs
GPUS=(0 1 2 3)
# Initial: first batch with 4 codebooks parallel on 4 GPUs (1 seed at a time)
# Each round = 4 codebooks trained. 3 rounds total.

run_stage3() {
    local tier=$1
    local gpu=$2
    local seed=$3
    local log_dir=$4
    local save_dir=$5

    mkdir -p "$log_dir" "$save_dir"

    local code_path=$(echo "${CB_PATHS[$tier]}" | grep -oE "_t5_rqvae_[a-z0-9_]+.npy")
    if [ -z "$code_path" ]; then
        echo "❌ No code_path resolved for tier $tier"
        return 1
    fi
    echo "[$(date)] Start tier=$tier gpu=$gpu seed=$seed code_path=$code_path" | tee -a "$log_dir/launcher.log"

    cd $REPO/HG-Rec
    python3 -u $REPO/scripts/task84_hgrec_stage3_train.py \
        --dataset_name Instruments \
        --dataset_path $REPO/HG-Rec/dataset/ \
        --code_path "${code_path}" \
        --codebook_size 64 128 256 1 \
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
        --device cuda:$gpu \
        --mode train \
        --save_path "$save_dir" \
        --log_path "$log_dir" \
        --seed $seed \
        --early_stop 20 \
        --beam_size 20 \
        --infer_size 96 \
        > "$log_dir/train.out" 2>&1
}

# Round 1: seed=42, all 4 tiers in parallel
echo "===== Round 1: seed=42 =====" | tee -a logs/task188/phase3_launcher.out
declare -A PIDS_R1
for i in 0 1 2 3; do
    tier=$(echo "t$((i+1))_$(case $i in 0) echo '8pct';; 1) echo '10pct';; 2) echo '12pct';; 3) echo '13pct'; esac)")
    gpu=$i
    save_dir=$REPO/products/task188/t5small_${tier}/seed42
    log_dir=$REPO/logs/task188/stage3_${tier}_seed42
    mkdir -p $save_dir $log_dir
    (
        run_stage3 $tier $gpu 42 "$log_dir" "$save_dir"
        echo "[$(date)] DONE tier=$tier gpu=$gpu seed=42" | tee -a logs/task188/phase3_launcher.out
    ) &
    PIDS_R1[$tier]=$!
    disown
done
echo "Round 1 PIDs: $(for k in "${!PIDS_R1[@]}"; do echo -n "$k=${PIDS_R1[$k]} "; done)"
wait "${PIDS_R1[@]}"

# Round 2: seed=123
echo "===== Round 2: seed=123 =====" | tee -a logs/task188/phase3_launcher.out
declare -A PIDS_R2
for i in 0 1 2 3; do
    tier=$(echo "t$((i+1))_$(case $i in 0) echo '8pct';; 1) echo '10pct';; 2) echo '12pct';; 3) echo '13pct'; esac)")
    gpu=$i
    save_dir=$REPO/products/task188/t5small_${tier}/seed123
    log_dir=$REPO/logs/task188/stage3_${tier}_seed123
    mkdir -p $save_dir $log_dir
    (
        run_stage3 $tier $gpu 123 "$log_dir" "$save_dir"
        echo "[$(date)] DONE tier=$tier gpu=$gpu seed=123" | tee -a logs/task188/phase3_launcher.out
    ) &
    PIDS_R2[$tier]=$!
    disown
done
echo "Round 2 PIDs: $(for k in "${!PIDS_R2[@]}"; do echo -n "$k=${PIDS_R2[$k]} "; done)"
wait "${PIDS_R2[@]}"

# Round 3: seed=2024
echo "===== Round 3: seed=2024 =====" | tee -a logs/task188/phase3_launcher.out
declare -A PIDS_R3
for i in 0 1 2 3; do
    tier=$(echo "t$((i+1))_$(case $i in 0) echo '8pct';; 1) echo '10pct';; 2) echo '12pct';; 3) echo '13pct'; esac)")
    gpu=$i
    save_dir=$REPO/products/task188/t5small_${tier}/seed2024
    log_dir=$REPO/logs/task188/stage3_${tier}_seed2024
    mkdir -p $save_dir $log_dir
    (
        run_stage3 $tier $gpu 2024 "$log_dir" "$save_dir"
        echo "[$(date)] DONE tier=$tier gpu=$gpu seed=2024" | tee -a logs/task188/phase3_launcher.out
    ) &
    PIDS_R3[$tier]=$!
    disown
done
echo "Round 3 PIDs: $(for k in "${!PIDS_R3[@]}"; do echo -n "$k=${PIDS_R3[$k]} "; done)"
wait "${PIDS_R3[@]}"

echo "===== ALL 12 Stage 3 runs COMPLETE at $(date) =====" | tee -a logs/task188/phase3_launcher.out
