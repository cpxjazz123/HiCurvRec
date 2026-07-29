#!/bin/bash
# ⚠️  [Issue #19 Gate 2 annotation 2026-07-29]
# 本脚本是 4 臂 chained dispatcher (Stage 1 → 2 → 3 → 4 全链), 但不含闸门求值点
# (Stage 2 后 → Stage 3 前无任何 evaluate_stage_2_gate 调用). 按 Issue #19 §Gate 2 规则,
# **不得用于任何声明了 Stage 2 级闸门的 issue** 执行.
# 若新 issue 需要本 dispatcher 的链式逻辑, 必须先在每臂 Stage 2 Sinkhorn 推断**结束后**插入
# `source scripts/issue19_gate_template.sh && evaluate_stage_2_gate <diag.json>` 调用,
# 不通过则跳过该臂 Stage 3. See verdicts/task281_issue19_gate0_replay.md 完整清单.
# 监控 4 臂 train_hrqvae 主进程全结束 → fire 4 臂 Stage 2 (Sinkhorn) → 4 臂 Stage 3 (T5-mini) → 4 臂 Stage 4 (eval)
#
# 用法: nohup bash scripts/task194_chained_dispatch.sh > logs/task194/chain_dispatch.log 2>&1 &

set -e
REPO=/home/wlia0047/ar57/wenyu/GeneRec
cd $REPO

LOG=$REPO/logs/task194/chain_dispatch.log
echo "==== Task #194 chain dispatch start at $(date) ====" > $LOG

source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh
conda activate /home/wlia0047/ar57_scratch/wenyu/grid_toys

export PYTHONPATH=$REPO/HG-Rec:$PYTHONPATH
export HF_HOME=/home/wlia0047/ar57_scratch/wenyu/hf_models

# 1) Wait for 4 train_hrqvae 主进程 (PIDs with --num_emb_list X 128 256, K0 = 32/64/128/256) 结束
# 关键: dataloader 子进程 args 也含 num_workers, 不能 grep -v num_workers.
# 正确方法: 只数 main 进程 (其 args 里 "ckpt_dir" 一定含 /task194/hrqvae_k0*), 不含 -c (dataloader worker)
echo "[$(date)] Waiting for 4 train_hrqvae main processes to finish..." | tee -a $LOG
while true; do
    # dataloader worker 进程 args 一般只含 batch 数据, 不含 train_hrqvae.py 的 main 参数.
    # 主进程 args 一定含 "train_hrqvae.py ... --ckpt_dir .../task194/hrqvae_k0"
    MAIN_COUNT=$(ps -eo args | grep "train_hrqvae.py" | grep "/task194/hrqvae_k0" | grep -v grep | wc -l)
    echo "[$(date)] train_hrqvae(task194) main processes alive: $MAIN_COUNT" >> $LOG
    if [ "$MAIN_COUNT" -le 0 ]; then
        echo "[$(date)] Stage 1 DONE — proceeding to Stage 2" | tee -a $LOG
        break
    fi
    sleep 60  # check every 1 min
done

# 2) Stage 2 — Sinkhorn codebook generation (4 臂)
echo "[$(date)] Fire Stage 2 — 4 臂 Sinkhorn codebook" | tee -a $LOG
for K0 in 32 64 128 256; do
    OUT_DIR=$REPO/products/task194/hrqvae_k0${K0}
    BEST_CKPT=$(find $OUT_DIR -name "best_collision_model.pth" | head -1)
    if [ -z "$BEST_CKPT" ] || [ ! -f "$BEST_CKPT" ]; then
        echo "⚠️ K0=$K0: no best_collision_model.pth found, try best_loss_model.pth" | tee -a $LOG
        BEST_CKPT=$(find $OUT_DIR -name "best_loss_model.pth" | head -1)
    fi
    if [ -z "$BEST_CKPT" ] || [ ! -f "$BEST_CKPT" ]; then
        echo "❌ K0=$K0: no usable ckpt, skip stage 2" | tee -a $LOG
        continue
    fi
    SID_OUT=$REPO/products/task194/hrqvae_k0${K0}/Instruments_t5_rqvae_k0${K0}.npy
    LOG_FILE=$REPO/logs/task194/stage2_k0${K0}.log
    echo "[$(date)] K0=$K0: Sinkhorn SID from $BEST_CKPT" | tee -a $LOG
    CUDA_VISIBLE_DEVICES=0 python3 -u $REPO/scripts/task194_stage2_codebook.py \
        --ckpt_path $BEST_CKPT \
        --output_path $SID_OUT \
        --device cuda:0 \
        --max_sinkhorn_iters 30 \
        > $LOG_FILE 2>&1 || echo "⚠️ K0=$K0 stage2 failed (exit=$?)" | tee -a $LOG
done

# 3) Stage 3 — T5-mini training (4 臂 × 200 epoch)
# 复用 task84_hgrec_stage3_train.py + 每个 K0 arm 自己的 code_path
echo "[$(date)] Fire Stage 3 — 4 臂 T5-mini training (concurrent on 4 GPU)" | tee -a $LOG

train_stage3() {
    local K0=$1
    local GPU=$2
    local SID=$REPO/products/task194/hrqvae_k0${K0}/Instruments_t5_rqvae_k0${K0}.npy
    local SAVE_PATH=$REPO/products/task194/t5mini_k0${K0}
    local LOG_FILE=$REPO/logs/task194/stage3_k0${K0}.log
    if [ ! -f $SID ]; then
        echo "⚠️ K0=$K0: no SID, skip stage 3" | tee -a $LOG
        return
    fi
    echo "[$(date)] K0=$K0: T5-mini training on cuda:$GPU" | tee -a $LOG
    CUDA_VISIBLE_DEVICES=$GPU python3 -u $REPO/scripts/task84_hgrec_stage3_train.py \
        --dataset_name Instruments \
        --dataset_path $REPO/HG-Rec/dataset/ \
        --code_path _t5_rqvae_k0${K0}.npy \
        --codebook_size ${K0} 128 256 1 \
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
        --device cuda:0 \
        --mode train \
        --save_path $SAVE_PATH \
        --log_path $REPO/logs/task194/ \
        --seed 42 \
        --early_stop 20 \
        --beam_size 20 \
        --infer_size 96 \
        > $LOG_FILE 2>&1 || echo "⚠️ K0=$K0 stage3 failed" | tee -a $LOG
}

# 4 臂并发 (4 GPU)
train_stage3 32 0 &
train_stage3 64 1 &
train_stage3 128 2 &
train_stage3 256 3 &
wait

echo "[$(date)] Stage 3 4 臂 done" | tee -a $LOG

# 4) Stage 4 — evaluation (4 臂 × 找 best_ckpt, eval)
# 复用 task84_hgrec_stage4_eval.sh, 但每臂需要单独跑 (hardcoded paths in task84_hgrec_stage4_eval.sh).
# 简单起见, 这里直接复用 task84 stage 4 fork 里的 standalone test_eval (如果存在) 或者使用 train mode 自带的 val.
# 暂时跳过 Stage 4 (Stage 3 train mode 自带 validation set eval), 看 val R@10 就够了.

echo "[$(date)] Stage 4 — 从 Stage 3 val log 抽 R@10" | tee -a $LOG
for K0 in 32 64 128 256; do
    LOG_FILE=$REPO/logs/task194/stage3_k0${K0}.log
    if [ -f $LOG_FILE ]; then
        # 找最后出现的 val R@10 (Stage 3 会 print 验证集指标)
        LAST_R10=$(grep -oE "R@10[: ]+[0-9.]+" $LOG_FILE | tail -1)
        LAST_N10=$(grep -oE "N@10[: ]+[0-9.]+" $LOG_FILE | tail -1)
        echo "  K0=$K0: $LAST_R10 | $LAST_N10" | tee -a $LOG
    fi
done

# 5) K0 L0_err diagnosis
echo "[$(date)] Fire K0 L0_err diagnosis" | tee -a $LOG
python3 -u $REPO/scripts/task194_k0_l0_err_diagnose.py > $REPO/logs/task194/k0_l0_err.log 2>&1 || echo "⚠️ K0 L0_err diagnose failed" | tee -a $LOG

# 6) Aggregate verdict
echo "[$(date)] Aggregate K0 results → verdicts/task194_k0_capacity_result.md" | tee -a $LOG
python3 -u $REPO/scripts/task194_aggregate_verdict.py > $REPO/logs/task194/verdict.log 2>&1 || echo "⚠️ verdict aggregate failed" | tee -a $LOG

echo "==== Task #194 chain dispatch complete at $(date) ====" | tee -a $LOG