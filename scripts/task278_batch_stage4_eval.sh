#!/bin/bash
# Task #278 — 批量 Stage 4 R@10 eval for untested ckpts
# 2026-07-29
#
# 遍历 14 个未测试 ckpt, 跑 test eval (~2 min/ckpt, ~28 min total).
# 复用 task278_batch_stage4_eval.py 通用 driver (跟 task243 v2 exclude-start-token 一致).

set -e
REPO=/home/wlia0047/ar57/wenyu/GeneRec
source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh
conda activate /home/wlia0047/ar57_scratch/wenyu/grid_toys
cd $REPO

mkdir -p $REPO/logs/task278 $REPO/verdicts

# ckpt_path|code_path|output_name 列表
CKPTS=(
    "products/task144/ckpt_arm_A/Instruments/Jul-24-2026_17-30-43/HG_Rec_best.pth|_t5_hrqvae_kappa_decouple_arm_A.npy|task144_armA"
    "products/task144/ckpt_arm_B/Instruments/Jul-24-2026_17-30-43/HG_Rec_best.pth|_t5_hrqvae_kappa_decouple_arm_B.npy|task144_armB"
    "products/task156/ckpt_hgrec/Instruments/Jul-24-2026_21-10-08/HG_Rec_best.pth|_t5_rqvae_code_default.npy|task156"
    "products/task160/ckpt_hgrec/Instruments/Jul-24-2026_21-42-45/HG_Rec_best.pth|_t5_rqvae_code_default.npy|task160"
    "products/task161/ckpt_hgrec/Instruments/Jul-24-2026_22-34-31/HG_Rec_best.pth|_t5_rqvae_code_default.npy|task161"
    "products/task194/t5mini_k032/Instruments/Jul-25-2026_23-00-08/HG_Rec_best.pth|_t5_rqvae_k032.npy|task194_k032"
    "products/task194/t5mini_k064/Instruments/Jul-25-2026_23-00-08/HG_Rec_best.pth|_t5_rqvae_k064.npy|task194_k064"
    "products/task194/t5mini_k0128/Instruments/Jul-25-2026_23-00-08/HG_Rec_best.pth|_t5_rqvae_k0128.npy|task194_k0128"
    "products/task194/t5mini_k0256/Instruments/Jul-25-2026_23-00-08/HG_Rec_best.pth|_t5_rqvae_k0256.npy|task194_k0256"
    "products/task206/t5small_hyp_e19/Instruments/Jul-26-2026_15-03-46/HG_Rec_best.pth|_t5_rqvae_hyp_e19.npy|task206_hyp_e19"
    "products/task206/t5small_euc_2000ep/Instruments/Jul-26-2026_15-03-46/HG_Rec_best.pth|_t5_rqvae_euclidean_2000ep.npy|task206_euc_2000ep"
    "products/task218/ckpt_hgrec_pck_spread/Instruments/Jul-27-2026_01-33-37/HG_Rec_best.pth|_t5_hrqvae_pck_spread.npy|task218_pck_spread"
    "products/task89/ckpt_hgrec_curv_free_M1/Instruments/Jul-24-2026_02-11-11/HG_Rec_best.pth|_t5_hrqvae_curv_free_M1.npy|task89_curv_free_M1"
)

GPU=${GPU:-0}
echo "[$(date)] Task #278 批量 Stage 4 eval 启动 GPU=$GPU, ${#CKPTS[@]} ckpts"

for i in "${!CKPTS[@]}"; do
    IFS='|' read -ra parts <<< "${CKPTS[$i]}"
    CKPT_REL="${parts[0]}"
    CODE_PATH="${parts[1]}"
    OUTPUT_NAME="${parts[2]}"

    CKPT_ABS="$REPO/$CKPT_REL"
    OUTPUT="$REPO/verdicts/${OUTPUT_NAME}_test_metrics.json"
    LOG="$REPO/logs/task278/stage4_${OUTPUT_NAME}_eval.out"

    echo "[$(date +%Y-%m-%d_%H:%M:%S)] [$((i+1))/${#CKPTS[@]}] === Eval $OUTPUT_NAME ==="
    echo "ckpt: $CKPT_ABS"
    echo "code_path: $CODE_PATH"

    CUDA_VISIBLE_DEVICES=$GPU TRITON_CACHE_DIR=/home/wlia0047/.triton/cache_task278_$OUTPUT_NAME \
    timeout 600 python3 -u $REPO/scripts/task278_batch_stage4_eval.py \
        --ckpt_path "$CKPT_ABS" \
        --code_path "$CODE_PATH" \
        --device cuda:0 \
        --output_path "$OUTPUT" \
        > $LOG 2>&1 || echo "FAILED ${OUTPUT_NAME}"

    if [ -f "$OUTPUT" ]; then
        R10=$(python3 -c "import json; print(json.load(open('$OUTPUT'))['Recall@10'])")
        echo "[$(date +%Y-%m-%d_%H:%M:%S)] $OUTPUT_NAME R@10=$R10"
    else
        echo "[$(date +%Y-%m-%d_%H:%M:%S)] $OUTPUT_NAME: NO OUTPUT FILE"
    fi
done

echo "[$(date)] Task #278 批量 eval 完成"