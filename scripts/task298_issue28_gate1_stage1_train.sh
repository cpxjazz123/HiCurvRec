#!/bin/bash
# Task #298 / Issue #28 Gate 1 — Stage 1 端到端 100 epoch 训练 (Gumbel-Softmax)
# 2026-07-29
#
# 背景: Issue #28 §Gate 0 PASS (scripts/task298_issue28_gate0_gumbel_softmax.py
#       + scripts/task298_train_hrqvae_gumbel.py)
#       Gate 1 = Stage 1 端到端 100 epoch 训练 (不预训练 frozen codebook)
#       ckpt_dir = products/task298/hrqvae_issue28_gate1/
#
# GATE_DECLARATION:
#   gate_1_issue_28: L0/L1/L2 util ≥ 90% + collision ≤ 0.20 at ep ≥ 50
#   auto_proceed_after_stage_0:    true
#   precedent_override:            forbidden
#
# R7: GPU 0 空闲
# R12: save_limit=1 + 每个 epoch 强制保存 best_loss_model.pth
# R137: NaN guard (任务 #89 内置)

set -uo pipefail

REPO=/home/wlia0047/ar57/wenyu/GeneRec
cd $REPO

LOG_DIR=$REPO/logs/task298
mkdir -p "$LOG_DIR"

# /tmp/genrec_env 替代 grid_toys (节点重置后 grid_toys env 不可用)
export PYTHONPATH=/tmp/genrec_env:$REPO/HG-Rec:${PYTHONPATH:-}
echo "PYTHONPATH=$PYTHONPATH"
python3 -c "import torch; print('torch=', torch.__version__, 'cuda=', torch.cuda.is_available())" 2>&1 | head -3

SAVE_PATH=$REPO/products/task298/hrqvae_issue28_gate1
mkdir -p $SAVE_PATH
TS=$(date +%Y%m%d_%H%M%S)
LOG_FILE=$LOG_DIR/stage1_gate1_${TS}.log

echo "[$(date)] Launching Task #298 Gate 1: Stage 1 100 epoch Gumbel-Softmax 训练"
echo "  save_path: $SAVE_PATH"
echo "  τ_l = [1.0, 0.5, 0.1] (per-layer 异构温度)"
echo "  c_k_range = [(1,5), (0.5,20), (0.5,20)] (per-layer 异构 metric)"
echo "  GPU: 0 (R7 空闲)"

TRITON_CACHE_DIR=/home/wlia0047/.triton/cache_task298_gate1
mkdir -p "$TRITON_CACHE_DIR"

CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR="$TRITON_CACHE_DIR" \
  nohup python3 $REPO/scripts/task298_train_hrqvae_gumbel.py \
    --lr 1e-3 \
    --epochs 100 \
    --batch_size 256 \
    --num_workers 4 \
    --eval_step 5 \
    --learner AdamW \
    --lr_scheduler_type linear \
    --warmup_epochs 20 \
    --data_path $REPO/HG-Rec/dataset/Instruments/item_emb.parquet \
    --weight_decay 0 \
    --dropout_prob 0.0 \
    --bn False \
    --loss_type mse \
    --kmeans_init True \
    --kmeans_iters 100 \
    --sk_epsilons 0.0 0.0 0.0 \
    --sk_iters 50 \
    --device cuda:0 \
    --num_emb_list 64 128 256 \
    --e_dim 32 \
    --quant_loss_weight 1.0 \
    --beta 0.5 \
    --loss_mult_codebook 1.0 \
    --layers 512 256 128 64 \
    --save_limit 1 \
    --ckpt_dir $SAVE_PATH/ \
    --seed 42 \
    --tau_list 1.0 0.5 0.1 \
    --c_k_range_list 1.0:5.0,0.5:20.0,0.5:20.0 \
    --c_k_seed 42 \
    > "$LOG_FILE" 2>&1 &

TRAIN_PID=$!
echo "$TRAIN_PID" > $REPO/products/task298/_TRAINING_PID_GATE1
echo "[$(date)] Gate 1 Stage 1 training PID=$TRAIN_PID, GPU=0"
echo ""
echo "===== Task #298 Gate 1 Launched ====="
echo "PID=$TRAIN_PID (GPU 0, Gumbel-Softmax 100 epoch, K=[64,128,256])"
echo "Log: $LOG_FILE"
echo "Expected runtime: ~1-2h (T5-mini 9.18M × 100 epoch ≈ baseline task89 比例)"
echo "Gate 1 通过条件: L0/L1/L2 util ≥ 90% + collision ≤ 0.20 at ep ≥ 50"