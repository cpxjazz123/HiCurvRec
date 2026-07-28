#!/bin/bash
# Task #237 — Issue #10 Gate 1 Arm B: partial Sinkhorn (max_iters=10)
#
# 同 vanilla #84 Stage 1 codebook (不用 PC κ/Gromov, 跟 Arm A/C 一致),
# 只改 Stage 2 Sinkhorn 强度 (max_sinkhorn_iters=10 vs Arm A=0 vs Arm C=30),
# 验证 collision→R@10 因果曲线在 vanilla 族内是否单调.
#
# Gate 1 pass bar:
#   - collision 必须与 Arm A (no-Sinkhorn, ~0.99) + Arm C (full-Sinkhorn, ~0.05)
#     各差 ≥ 15pp (3 个不同水平)
#   - utilization 变动幅度需 ≤ collision 变动幅度 (否则 Sinkhorn 同时动了俩变量)
#
# Gate 2 pass bar:
#   - 单调下降 (C 0.1058 ≥ B ≥ A 0.1020, 序与 collision 反向) → collision 在 vanilla
#     族内是有效杠杆, <=12% bar (按 task236 统一口径重述后) 保留
#   - B < 0.1020 (U 形 / 非单调) → collision 不是可用代理, Issue #9 Gate 1 的
#     collision <= 0.3706 必须从门槛降级为诊断量
#   - B > 0.1058 → 仅靠 Stage 2 后处理即可越过现有最优
set -e
REPO=/home/wlia0047/ar57/wenyu/GeneRec
source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh
conda activate /home/wlia0047/ar57_scratch/wenyu/grid_toys

mkdir -p $REPO/products/task237 $REPO/logs/task237
cd $REPO/HG-Rec
export PYTHONPATH=$REPO/HG-Rec:$PYTHONPATH
export HF_HOME=/home/wlia0047/ar57_scratch/wyu/hf_models 2>/dev/null
export HF_HOME=/home/wlia0047/ar57_scratch/wenyu/hf_models

# Same Stage 1 vanilla codebook as #84 baseline (no re-train)
CKPT=$REPO/products/task84/ckpt/Instruments/Jul-23-2026_20-08-06_beta_1.000_codebook_[64,128,256]_sk_0.000/best_loss_model.pth
SID_OUT=$REPO/HG-Rec/dataset/Instruments/Instruments_t5_rqvae_task237_armB.npy
LOG_STAGE2=$REPO/logs/task237/stage2_codebook.out

echo "[$(date)] === Task #237 Arm B Stage 2: partial Sinkhorn max_iters=10 ===" | tee $LOG_STAGE2

CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/home/wlia0047/.triton/cache_task237_stage2 \
python3 -u $REPO/scripts/task223_stage2_codebook.py \
    --ckpt_path "$CKPT" \
    --output_path "$SID_OUT" \
    --device cuda:0 \
    --max_sinkhorn_iters 10 \
    >> $LOG_STAGE2 2>&1

echo "[$(date)] Stage 2 done, collision should be in (0.05, 0.99)" | tee -a $LOG_STAGE2
tail -30 $LOG_STAGE2

# Stage 3 + 4 will follow in next tick (or launch script can chain)
echo "[$(date)] === Stage 3 T5-mini training (R12 ckpt + heartbeat) ==="
# Stage 3 launcher: task237_stage3.sh will call task84_hgrec_stage3_train.py with task233 R12 fix
