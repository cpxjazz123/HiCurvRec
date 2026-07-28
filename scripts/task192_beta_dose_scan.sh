#!/bin/bash
# Task #192 — β 剂量扫描 (Instruments, 4 臂并发)
# 目的: 通过改变 β 看 collision 剂量-反应, 验证 Task #191 (L0_err ~ collision, r=0.861) 是否机制因果.
#
# 4 臂: β=0.25 (cuda:0) / β=0.5 (cuda:1) / β=1.0 (cuda:2) / β=2.0 (cuda:3)
# 每臂 1000 epoch, --save_limit 50 (让每 5 epoch 留一个 snapshot, 跟 task188 一致)
#
# 使用时机: Task #188 Phase 4 (12 Stage 4 eval) 完成 → GPU 释放 → fire 本脚本.
#
# 输出:
#   products/task192/hrqvae_beta0.25/{每 5 epoch 一个 ckpt + best_* + log}
#   products/task192/hrqvae_beta0.5/...
#   products/task192/hrqvae_beta1.0/...   (β=1.0 是 task188 已经训过的对照组, 这里重训一次看种子内方差)
#   products/task192/hrqvae_beta2.0/...
#   logs/task192/beta_arm_*.log
#
# 后续诊断: task192_beta_dose_diagnose.py 跟 task191_l0_quant_error_per_epoch.py 同模板,
#          对 4 个 β × 每个 epoch 各算 L0_err / collision / ρ / ‖z‖.

set -e
source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh
conda activate /home/wlia0047/ar57_scratch/wenyu/grid_toys

REPO=/home/wlia0047/ar57/wenyu/GeneRec
export PYTHONPATH=$REPO/HG-Rec:$PYTHONPATH
export HF_HOME=/home/wlia0047/ar57_scratch/wenyu/hf_models
mkdir -p $REPO/logs/task192 $REPO/products/task192

# 4 臂训练参数 (跟 task188 phase 1 全部一致, 只 β 不同)
COMMON_ARGS=(
    --lr 1e-3
    --epochs 1000
    --batch_size 256
    --num_workers 4
    --eval_step 5
    --learner AdamW
    --lr_scheduler_type linear
    --warmup_epochs 20
    --data_path ./dataset/Instruments/item_emb.parquet
    --weight_decay 0.0
    --dropout_prob 0.0
    --loss_type poincare
    --kmeans_init True
    --kmeans_iters 1000
    --sk_epsilons 0.0 0.0 0.0
    --sk_iters 50
    --num_emb_list 64 128 256
    --e_dim 32
    --quant_loss_weight 1.0
    --layers 512 256 128 64
    --save_limit 50
)

declare -A BETA_GPU
BETA_GPU[0.25]=0
BETA_GPU[0.5]=1
BETA_GPU[1.0]=2
BETA_GPU[2.0]=3

run_beta_arm() {
    local beta=$1
    local gpu=$2
    local out_dir=$REPO/products/task192/hrqvae_beta${beta}
    local cache_dir=/home/wlia0047/.triton/cache_task192_beta${beta}
    local log_file=$REPO/logs/task192/beta_arm_${beta}.log
    mkdir -p $out_dir $cache_dir

    echo "[$(date)] Start β=$beta on cuda:$gpu"
    cd $REPO/HG-Rec
    export CUDA_VISIBLE_DEVICES=$gpu
    export TRITON_CACHE_DIR=$cache_dir
    mkdir -p $TRITON_CACHE_DIR

    python3 -u train_hrqvae.py \
        "${COMMON_ARGS[@]}" \
        --beta $beta \
        --device cuda:0 \
        --ckpt_dir $out_dir \
        > $log_file 2>&1

    echo "[$(date)] DONE β=$beta (exit=$?); ckpts in $out_dir"
}

# 4 臂并发 — 4 GPU 各跑 1 臂
for beta in 0.25 0.5 1.0 2.0; do
    gpu=${BETA_GPU[$beta]}
    (
        run_beta_arm $beta $gpu
    ) &
    PIDS_R[$beta]=$!
    disown
done

echo "==== Task #192 β scan PIDs: $(for k in "${!PIDS_R[@]}"; do echo -n "β$k=${PIDS_R[$k]} "; done) ===="
wait "${PIDS_R[@]}"

echo "==== Task #192 4 臂完成 at $(date) ====" | tee -a $REPO/logs/task192/beta_scan_done.log
