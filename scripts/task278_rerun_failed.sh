#!/bin/bash
# Task #278 (rerun) — 修过的 driver 重跑 5 个失败 ckpt
# 2026-07-29
#
# 失败原因: driver 硬编码 codebook_size=[64,128,256,1] + 默认 d_model=128, 但每个 ckpt 训练配置不同
# 修法: scripts/task278_batch_stage4_eval.py 加 --codebook_size CLI 参数

set -e
REPO=/home/wlia0047/ar57/wenyu/GeneRec
source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh
conda activate /home/wlia0047/ar57_scratch/wenyu/grid_toys
cd $REPO

# ckpt_path|code_path|codebook_size|d_model|d_ff|num_layers|num_decoder_layers|num_heads|output_name
RUNS=(
    "products/task156/ckpt_hgrec/Instruments/Jul-24-2026_21-10-08/HG_Rec_best.pth|_t5_rqvae_code_default.npy|32,64,256,1|128|1024|6|4|6|task156"
    "products/task160/ckpt_hgrec/Instruments/Jul-24-2026_21-42-45/HG_Rec_best.pth|_t5_rqvae_code_default.npy|32,64,256,1|512|2048|6|6|8|task160"
    "products/task161/ckpt_hgrec/Instruments/Jul-24-2026_22-34-31/HG_Rec_best.pth|_t5_rqvae_code_default.npy|32,64,256,1|256|1024|4|4|4|task161"
    "products/task194/t5mini_k032/Instruments/Jul-25-2026_23-00-08/HG_Rec_best.pth|_t5_rqvae_k032.npy|32,128,256,1|128|1024|6|4|6|task194_k032"
    "products/task194/t5mini_k0128/Instruments/Jul-25-2026_23-00-08/HG_Rec_best.pth|_t5_rqvae_k0128.npy|128,128,256,1|128|1024|6|4|6|task194_k0128"
    "products/task194/t5mini_k0256/Instruments/Jul-25-2026_23-00-08/HG_Rec_best.pth|_t5_rqvae_k0256.npy|256,128,256,1|128|1024|6|4|6|task194_k0256"
)

GPU=${GPU:-0}
echo "[$(date)] Task #278 rerun 启动 GPU=$GPU, ${#RUNS[@]} ckpts"

for i in "${!RUNS[@]}"; do
    IFS='|' read -ra parts <<< "${RUNS[$i]}"
    CKPT_REL="${parts[0]}"
    CODE_PATH="${parts[1]}"
    CB_SIZE="${parts[2]}"
    D_MODEL="${parts[3]}"
    D_FF="${parts[4]}"
    N_LAYERS="${parts[5]}"
    N_DEC_LAYERS="${parts[6]}"
    N_HEADS="${parts[7]}"
    OUTPUT_NAME="${parts[8]}"

    CKPT_ABS="$REPO/$CKPT_REL"
    OUTPUT="$REPO/verdicts/${OUTPUT_NAME}_test_metrics.json"
    LOG="$REPO/logs/task278/stage4_${OUTPUT_NAME}_eval_rerun.out"

    echo "[$(date +%Y-%m-%d_%H:%M:%S)] [$((i+1))/${#RUNS[@]}] === Rerun $OUTPUT_NAME (cb=$CB_SIZE d_model=$D_MODEL) ==="

    CUDA_VISIBLE_DEVICES=$GPU TRITON_CACHE_DIR=/home/wlia0047/.triton/cache_task278_$OUTPUT_NAME \
    timeout 600 python3 -u $REPO/scripts/task278_batch_stage4_eval.py \
        --ckpt_path "$CKPT_ABS" \
        --code_path "$CODE_PATH" \
        --codebook_size "$CB_SIZE" \
        --d_model "$D_MODEL" \
        --d_ff "$D_FF" \
        --num_layers "$N_LAYERS" \
        --num_decoder_layers "$N_DEC_LAYERS" \
        --num_heads "$N_HEADS" \
        --device cuda:0 \
        --output_path "$OUTPUT" \
        > $LOG 2>&1 || echo "FAILED ${OUTPUT_NAME}"

    if [ -f "$OUTPUT" ]; then
        R10=$(python3 -c "import json; d=json.load(open('$OUTPUT')); print(d.get('Recall@10', -1))")
        echo "[$(date +%Y-%m-%d_%H:%M:%S)] $OUTPUT_NAME R@10=$R10"
    else
        echo "[$(date +%Y-%m-%d_%H:%M:%S)] $OUTPUT_NAME: NO OUTPUT FILE"
    fi
done

echo "[$(date)] Task #278 rerun 完成"