#!/bin/bash
# Task #196 — 4 臂 γ 扫描 Stage 1 (500 epoch × 4 GPU 并发)
#
# 变量: γ ∈ {0, 0.1, 0.5, 2.0}
# 不变: β=0.5, K0=64, num_emb_list=[64,128,256], e_dim=32, loss_type=poincare
#       c=1.0 (HG-Rec baseline), sk_eps=0 (paper 默认, 关闭 Sinkhorn)
# 新增 CLI: --norm_target (nargs='+', per-layer r_target_norm) + --gamma_norm
#
# Patch 位置:
#   1. HG-Rec/model/utils.py: HVectorQuantization.__init__ 加 r_target_norm, gamma_norm
#   2. HG-Rec/model/utils.py: HVectorQuantization.forward (line 691+) 加 norm_loss
#   3. HG-Rec/model/utils.py: HResidualVectorQuantization.__init__ 加 r_target_norm_list, gamma_norm
#   4. HG-Rec/model/hrqvae.py: HRQVAE.__init__ 透传 r_target_norm_list, gamma_norm
#   5. HG-Rec/train_hrqvae.py: 加 --norm_target (nargs='+'), --gamma_norm
#
# 判据:
#   - ρ_p50 三层 ≥ 1.8/2.4/3.0 (per-layer r_target_norm = ρ/2 = 0.9/1.2/1.5)
#   - λ_p50 三层 ≥ 4.0/7.0/12.0
#   - collision ≤ 13% (基线 1.5×)
#   - recon_loss 不发散
#
# 预算: 4-6 h (500 epoch × 4 GPU 并发)

set -e
REPO=/home/wlia0047/ar57/wenyu/GeneRec
source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh
conda activate /home/wlia0047/ar57_scratch/wenyu/grid_toys
cd $REPO/HG-Rec
export PYTHONPATH=$REPO/HG-Rec:$PYTHONPATH
export HF_HOME=/home/wlia0047/ar57_scratch/wenyu/hf_models

mkdir -p $REPO/logs/task196 $REPO/products/task196

DATA=$REPO/HG-Rec/dataset/Instruments/item_emb.parquet

# norm_target = ρ/2 (切空间目标). ρ_目标 = 2.0/2.7/3.4 (来自 #206m + #199 r_target_list 默认值)
# 所以 r_target_norm = 1.0/1.35/1.70 (per-layer 切空间目标)
NORM_TARGETS="1.0 1.35 1.70"

# 4 个 γ 臂 (分配到 4 张卡, 0/1/2/3)
GAMMAS=(0.0 0.1 0.5 2.0)
GPUS=(0 1 2 3)

# 启动 4 个臂
PIDS=()
for i in 0 1 2 3; do
    GAMMA=${GAMMAS[$i]}
    GPU=${GPUS[$i]}
    LOG=$REPO/logs/task196/arm${i}_gamma${GAMMA}_stage1.log
    SAVE_DIR=$REPO/products/task196/arm${i}_gamma${GAMMA}

    echo "[$(date)] === Task #196 臂 ${i}: γ=${GAMMA}, GPU=${GPU} ==="
    echo "[$(date)] norm_target=${NORM_TARGETS}, beta=0.5, c=1.0 (paper baseline)"
    echo "[$(date)] epochs=500, batch_size=1024, save_limit=50, eval_step=1"
    echo "[$(date)] log=$LOG"

    CUDA_VISIBLE_DEVICES=$GPU TRITON_CACHE_DIR=/home/wlia0047/.triton/cache_task196_arm${i} \
    nohup python3 -u $REPO/HG-Rec/train_hrqvae.py \
        --data_path $DATA \
        --lr 1e-3 --epochs 500 --batch_size 1024 \
        --loss_type poincare --kmeans_init True --kmeans_iters 1000 \
        --sk_epsilons 0.0 0.0 0.0 --sk_iters 50 \
        --num_emb_list 64 128 256 --e_dim 32 \
        --beta 0.5 --layers 512 256 128 64 \
        --curvatures 1.0,1.0,1.0 \
        --quant_loss_weight 1.0 \
        --norm_target $NORM_TARGETS \
        --gamma_norm $GAMMA \
        --save_limit 50 \
        --device cuda:0 \
        --ckpt_dir $SAVE_DIR \
        > $LOG 2>&1 &

    PID=$!
    PIDS+=($PID)
    echo "[$(date)] PID=$PID"
done

# 写 _TRAINING_PID 给每个臂 (R12/R88 daemon 用)
for i in 0 1 2 3; do
    echo "${PIDS[$i]}" > $REPO/products/task196/arm${i}_gamma${GAMMAS[$i]}/_TRAINING_PID
done

echo ""
echo "=== 4 臂已启动 ==="
echo "PID 列表: ${PIDS[*]}"
echo "监控: tail -f $REPO/logs/task196/arm{0,1,2,3}_gamma*_stage1.log"
echo "GPU 利用: watch -n 5 nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader"
echo ""
echo "=== 判据 ==="
echo "GO (norm 推上去 + collision ≤ 13%):"
echo "  看 logs/task196/arm*_stage1.log 里的 [hypnorm] 行, 检查 ρ_p50 / λ_κ"
echo "NO-GO (collision > 13% 或 recon 发散):"
echo "  看 arm0 vs arm3, 如果 γ=2.0 崩 → norm_loss 过强, 调小 γ"
echo "NO-GO (γ 完全拉不动码字):"
echo "  看 arm3 (γ=2.0) ρ 是否跟 arm0 (γ=0) 一样 → 加大 γ"