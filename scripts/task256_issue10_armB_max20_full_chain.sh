#!/bin/bash
# ⚠️  [Issue #19 Gate 2 annotation 2026-07-29]
# 本脚本不含闸门求值点 (Stage 2 后 → Stage 3 前无任何 evaluate_stage_2_gate 调用).
# 按 Issue #19 §Gate 2 规则, **不得用于任何声明了 Stage 2 级闸门的 issue** 执行.
# 若新 issue 需要本脚本的链式逻辑, 必须先在 Stage 2 推断**结束后**插入
# `source scripts/issue19_gate_template.sh && evaluate_stage_2_gate <diag.json>` 调用,
# 不通过则不启动 Stage 3. See verdicts/task281_issue19_gate0_replay.md 完整清单.
# 同时本脚本 Issue #10 关闭时明确记为 "不建议启动" (Chain 仍未拆 Stage 2/3 闸门).
# 跟 Task #237 (max_iters=10) 平行, 跟 Arm C (max_iters=30) 平行.
# 3-arm 因果曲线 max_iters ∈ {0, 10, 20, 30}, 找 collision→R@10 因果曲线.
#
# Gate 1 pass bar:
#   - collision 跟 Arm A (no-Sinkhorn, ~0.99) + Arm C (full-Sinkhorn ~0.05) 各差 ≥ 15pp
#   - utilization 变动幅度 ≤ collision 变动幅度 (否则 Sinkhorn confounded)
#
# Gate 2 pass bar:
#   - 单调下降 (C 0.1058 ≥ B20 ≥ B10 ≥ A 0.1020, 序与 collision 反向) → collision 是杠杆
#   - B20 < 0.1020 → U 形 / 非单调, collision 不是可用代理
#   - B20 > 0.1058 → 仅靠 Stage 2 后处理即可越过现有最优
#
# R11.4 不可逆决策: 本脚本 pre-prepared, 等用户决策 Issue #10 方向 A 后启动.
# R12 强制 ckpt + heartbeat wrapper (R88 daemon 检测).
# GPU 0 (R7 / Task #243 已关闭, 4 卡空闲).

set -e
REPO=/home/wlia0047/ar57/wenyu/GeneRec
source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh
conda activate /home/wlia0047/ar57_scratch/wenyu/grid_toys

mkdir -p $REPO/products/task256 $REPO/logs/task256
cd $REPO/HG-Rec
export PYTHONPATH=$REPO/HG-Rec:$PYTHONPATH
export HF_HOME=/home/wlia0047/ar57_scratch/wenyu/hf_models

# Same Stage 1 vanilla codebook as #84 baseline (no re-train)
# Task #237 路径: $REPO/products/task84/ckpt/Instruments/Jul-23-2026_20-08-06_beta_1.000_codebook_[64,128,256]_sk_0.000/best_loss_model.pth
CKPT=$REPO/products/task84/ckpt/Instruments/Jul-23-2026_20-08-06_beta_1.000_codebook_[64,128,256]_sk_0.000/best_loss_model.pth
SID_OUT=$REPO/HG-Rec/dataset/Instruments/Instruments_t5_rqvae_task256_armB_max20.npy
LOG_STAGE2=$REPO/logs/task256/stage2_codebook.out

echo "[$(date)] === Task #256 Arm B (max_iters=20) Stage 2: mid Sinkhorn ===" | tee $LOG_STAGE2

# Stage 2: 推断 SID, max_sinkhorn_iters=20
CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/home/wlia0047/.triton/cache_task256_stage2 \
python3 -u $REPO/scripts/task223_stage2_codebook.py \
    --ckpt_path "$CKPT" \
    --output_path "$SID_OUT" \
    --device cuda:0 \
    --max_sinkhorn_iters 20 \
    >> $LOG_STAGE2 2>&1

echo "[$(date)] Stage 2 done, expected collision in (0.05, 0.10)" | tee -a $LOG_STAGE2
tail -30 $LOG_STAGE2

# Stage 3: T5-mini 50 epoch (R12 ckpt + heartbeat + task233 R12 fix)
PROD_DIR=$REPO/products/task256/t5mini_armB_max20
LOG_DIR=$REPO/logs/task256
mkdir -p $PROD_DIR $LOG_DIR

LOG_FILE=$LOG_DIR/stage3_train.out
echo "[$(date)] === Task #256 Stage 3: T5-mini 50 epoch on max_iters=20 SID, GPU=0 ===" | tee $LOG_FILE
echo "[$(date)] R12: save_strategy=epoch + save_total_limit=1 (latest ckpt only)" | tee -a $LOG_FILE

CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/home/wlia0047/.triton/cache_task256_stage3 \
python3 -u $REPO/scripts/task84_hgrec_stage3_train.py \
    --dataset_name Instruments \
    --dataset_path $REPO/HG-Rec/dataset/ \
    --code_path _t5_rqvae_task256_armB_max20.npy \
    --codebook_size 64 128 256 1 \
    --num_epochs 50 \
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
    --device cuda:0 \
    --mode train \
    --save_path $PROD_DIR \
    --log_path $LOG_DIR \
    --seed 42 \
    --early_stop 10 \
    --beam_size 20 \
    --infer_size 96 \
    > $LOG_FILE 2>&1 &
TRAIN_PID=$!
echo $TRAIN_PID > $PROD_DIR/_TRAINING_PID
echo "[$(date)] Task #256 Stage 3 launched PID=$TRAIN_PID" | tee -a $LOG_FILE

# Stage 4 跑完后会自动跑, 详见 task253_stage4_eval.py 已验证可用
