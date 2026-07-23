#!/bin/bash
# Task #74 — ETEGRec 128d paper-cycle (warmup_steps=0, warm_epoch=1, early_stop=30)
# vs Task #73 paper_exact: 去掉 step-level warmup + cycle 内 ID epoch=1 + patience 30
# 验证 R3 假设: R@10 是否从 0.0253 提升到 paper 0.0624
# 目标 (vs paper 0.0624):
#   R@10 ≥ 0.0624 ✅ paper-EXACT 复现成功
#   0.05-0.0624 🟡 partial (-20%)
#   0.03-0.05 ⚠️ partial improvement → cycle=4 试一遍
#   <0.03 ❌ R3 否证 → 切换 R4 (上游 code diff)

set -e

source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh
conda activate /home/wlia0047/ar57_scratch/wenyu/genrec_env

cd /home/wlia0047/ar57/wenyu/GeneRec/ETEGRec

LOG_DIR="/home/wlia0047/ar57/wenyu/GeneRec/logs"
TS=$(date +%b-%d-%Y_%H-%M-%S | sed 's/^.../\L&/')
RUN_NAME="task74_pc_${TS}"
LOG_FILE="$LOG_DIR/task74_etegrec_pc_${TS}.log"

mkdir -p "$LOG_DIR"

echo "===== [Task #74] ETEGRec 128d paper-cycle launched at $(date) =====" | tee "$LOG_FILE"
echo "RUN_NAME: $RUN_NAME" | tee -a "$LOG_FILE"
echo "paper-cycle config: warmup_steps=0, warm_epoch=1, early_stop=30, 128d SASRec-only emb, code_length=4" | tee -a "$LOG_FILE"
echo "Datasets: Musical_Instruments (57439 users / 24587 items / 511837 interactions)" | tee -a "$LOG_FILE"
echo "Target: R@10 ≥ 0.0624 (paper Musical_Instruments ETEGRec)" | tee -a "$LOG_FILE"

export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# batch_size 512 OOMs on 44GB L40S — use bs=128 + grad_accum=4 (effective 512)
# paper-cycle 关键修改:
#   --warmup_steps=0 (no step-level lr warmup)
#   --warm_epoch=1 (cycle 内 ID epoch, paper exact)
#   --early_stop=30 (2x patience for cycle V-shape)
#   --code_length=4 (paper default, was 3 in some prior runs)
python3 main.py \
    --config ./config/musical_instruments.yaml \
    --lr_rec=0.005 \
    --lr_id=0.0001 \
    --rec_kl_loss=0.0001 \
    --rec_dec_cl_loss=0.0003 \
    --id_kl_loss=0.0001 \
    --id_dec_cl_loss=0.0003 \
    --cycle=2 \
    --eval_step=2 \
    --early_stop=30 \
    --warmup_steps=0 \
    --lr_scheduler_type=cosine \
    --code_length=4 \
    --batch_size=128 \
    --gradient_accumulation_steps=4 \
    2>&1 | tee -a "$LOG_FILE"

echo "===== [Task #74] ETEGRec 128d paper-cycle completed at $(date) =====" | tee -a "$LOG_FILE"
echo "===== [Task #74] ETEGRec 128d paper-cycle done at $(date) ====="