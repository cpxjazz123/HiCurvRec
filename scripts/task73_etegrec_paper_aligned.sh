#!/bin/bash
# Task #73 ETEGRec paper-aligned rerun (R@10 target ≥ 0.0586, paper 0.0624)
# Hyperparam: lr_rec=0.003 (paper center), μ=1e-4, λ=1e-4 (paper center, was 3e-4 in prior)
# Prior run (Jul-21-2026 ~ Jul-22-2026) R@10=0.025993 (-58%), failed.
# Single GPU (paper used single-GPU for d_model=128), early_stop=15

set -e

source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh
conda activate /home/wlia0047/ar57_scratch/wenyu/genrec_env

cd /home/wlia0047/ar57/wenyu/GeneRec/ETEGRec

LOG_DIR="/home/wlia0047/ar57/wenyu/GeneRec/logs"
TS=$(date +%b-%d-%Y_%H-%M-%S | sed 's/^.../\L&/')
RUN_NAME="task73_pa_${TS}"
LOG_FILE="$LOG_DIR/task73_etegrec_pa_${TS}.log"

mkdir -p "$LOG_DIR"

echo "===== [Task #73] ETEGRec paper-aligned launched at $(date) =====" | tee "$LOG_FILE"
echo "RUN_NAME: $RUN_NAME" | tee -a "$LOG_FILE"
echo "Hyperparam: lr_rec=0.003, lr_id=0.0001, μ=1e-4, λ=1e-4, cycle=2" | tee -a "$LOG_FILE"
echo "Single GPU (use_ddp=False)" | tee -a "$LOG_FILE"

# 显式 export GPU id (用 GPU 0, 其他 3 张卡留给后续 baselines 并行)
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# 不使用 accelerate DDP (单卡)
# 注意: 默认 yaml 的 batch_size=512 OOM (44GB 显存不够), 改用 batch_size=128 + grad_accum=4 (effective batch=512, 与 prior run 一致)
python3 main.py \
    --config ./config/musical_instruments.yaml \
    --lr_rec=0.003 \
    --lr_id=0.0001 \
    --rec_kl_loss=0.0001 \
    --rec_dec_cl_loss=0.0001 \
    --id_kl_loss=0.0001 \
    --id_dec_cl_loss=0.0001 \
    --cycle=2 \
    --eval_step=2 \
    --early_stop=15 \
    --batch_size=128 \
    --gradient_accumulation_steps=4 \
    2>&1 | tee -a "$LOG_FILE"

echo "===== [Task #73] ETEGRec paper-aligned completed at $(date) =====" | tee -a "$LOG_FILE"
echo "===== [Task #73] ETEGRec paper-aligned done at $(date) ====="
