#!/bin/bash
# Task #194 — K0 码本容量扫描 (4 臂并发, 4 GPU)
# 目的: K0={32,64,128,256} 是否单因素控制 collision
# 复用 task188/192 recipe, 只改 num_emb_list[0] (K0)
# 4 臂并发 ~50 min, Stage 2/3/4 后续手动 fire

set -e
source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh
conda activate /home/wlia0047/ar57_scratch/wenyu/grid_toys

REPO=/home/wlia0047/ar57/wenyu/GeneRec
export PYTHONPATH=$REPO/HG-Rec:$PYTHONPATH
export HF_HOME=/home/wlia0047/ar57_scratch/wenyu/hf_models
mkdir -p $REPO/logs/task194 $REPO/products/task194

cd $REPO/HG-Rec

run_k0_arm() {
    local K0=$1
    local GPU=$2
    local OUT_DIR=$REPO/products/task194/hrqvae_k0${K0}
    local CACHE_DIR=/home/wlia0047/.triton/cache_task194_k0${K0}
    local LOG_FILE=$REPO/logs/task194/k0_arm_${K0}.log
    mkdir -p $OUT_DIR $CACHE_DIR

    echo "[$(date)] Start K0=$K0 on cuda:$GPU"
    export CUDA_VISIBLE_DEVICES=$GPU
    export TRITON_CACHE_DIR=$CACHE_DIR
    mkdir -p $TRITON_CACHE_DIR

    python3 -u train_hrqvae.py \
        --lr 1e-3 \
        --epochs 500 \
        --batch_size 256 \
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
        --num_emb_list ${K0} 128 256 \
        --e_dim 32 \
        --quant_loss_weight 1.0 \
        --beta 0.5 \
        --layers 512 256 128 64 \
        --save_limit 50 \
        --device cuda:0 \
        --ckpt_dir $OUT_DIR \
        > $LOG_FILE 2>&1

    echo "[$(date)] DONE K0=$K0 (exit=$?); ckpts in $OUT_DIR"
}

# 4 臂并发, GPU 0-3 各 1 臂
declare -A K0_GPU
K0_GPU[32]=0
K0_GPU[64]=1
K0_GPU[128]=2
K0_GPU[256]=3

for K0 in 32 64 128 256; do
    GPU=${K0_GPU[$K0]}
    run_k0_arm $K0 $GPU &
    PIDS[$K0]=$!
    disown
done

echo "==== Task #194 K0 scan PIDs: $(for k in "${!PIDS[@]}"; do echo -n "K0=$k:PID=${PIDS[$k]} "; done) ===="
wait "${PIDS[@]}"

echo "==== Task #194 4 臂完成 at $(date) ====" | tee -a $REPO/logs/task194/k0_scan_done.log