#!/bin/bash
# Task #185 — 取消早停 + 从 Task #181 best ckpt 继续训练到 1000 epoch
# 启动 GPU 0 (Task #184 Stage 1 训练已完成, GPU 0 空闲)

set -e

source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh
conda activate /home/wlia0047/ar57_scratch/wenyu/grid_toys
cd /home/wlia0047/ar57/wenyu/GeneRec/HG-Rec

export PYTHONPATH=/home/wlia0047/ar57/wenyu/GeneRec/HG-Rec:$PYTHONPATH
export HF_HOME=/home/wlia0047/ar57_scratch/wenyu/hf_models
export TRITON_CACHE_DIR=/home/wlia0047/.triton/cache_task185

mkdir -p /home/wlia0047/ar57/wenyu/GeneRec/products/task185
mkdir -p /fs04/ar57/wenyu/GeneRec/logs/task185

# Task #181 best ckpt (val NDCG@20=0.1003, val R@10=0.1262, test R@10=0.1057)
RESUME_CKPT=/home/wlia0047/ar57/wenyu/GeneRec/products/task181/t5small_phase0.6/Instruments/Jul-25-2026_16-29-42/HG_Rec_best.pth
if [ ! -f "$RESUME_CKPT" ]; then
    echo "❌ Resume ckpt MISSING: $RESUME_CKPT"
    exit 1
fi
echo "✅ Resume ckpt: $RESUME_CKPT"

echo "===== [Task #185] Resume from Task #181 + disable early stop + 1000 epoch at $(date) ====="

python3 /home/wlia0047/ar57/wenyu/GeneRec/scripts/task84_hgrec_stage3_train.py \
    --dataset_name Instruments \
    --batch_size 1024 \
    --infer_size 96 \
    --lr 1e-4 \
    --num_epochs 1000 \
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
    --device cuda:0 \
    --mode train \
    --save_path /home/wlia0047/ar57/wenyu/GeneRec/products/task185/t5small_resume/ \
    --log_path /home/wlia0047/ar57/wenyu/GeneRec/logs/ \
    --seed 42 \
    --early_stop 99999 \
    --disable_early_stop \
    --resume_from $RESUME_CKPT \
    --beam_size 20 \
    2>&1 | tee /fs04/ar57/wenyu/GeneRec/logs/task185/stage3_resume.out

echo "===== [Task #185] Stage 3 completed at $(date) ====="
