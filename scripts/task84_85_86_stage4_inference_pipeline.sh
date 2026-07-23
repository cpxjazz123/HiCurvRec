#!/bin/bash
# Stage 4 inference pipeline for Task #84 (TIGER T5) / #85 (FDSA+S3Rec) / #86 (P5-CID+SID)
# Designed to run AFTER all training tasks (Task #78/#80/#81/#82/#83) complete.
#
# Usage:
#   bash task84_85_86_stage4_inference_pipeline.sh [tiger|fdsa|s3rec|p5_cid|p5_sid|all]
set -e

GENE_REC=/home/wlia0047/ar57/wenyu/GeneRec
LOGS=$GENE_REC/logs
PRODUCTS=$GENE_REC/products
TS=$(date '+%j-%H%M%S')
RESULT_DIR=$GENE_REC/results/stage4_${TS}
mkdir -p $RESULT_DIR

run_tiger() {
    # Task #78 (TIGER T5) inference
    # Best checkpoint: /home/wlia0047/ar57/wenyu/GeneRec/external/LETTER/LETTER-TIGER/ckpt/Instruments_tiger/checkpoint-{BEST_STEP}
    # Reference: task61_letter_tiger_inference.md - R@10=0.0997
    CKPT_DIR=$GENE_REC/external/LETTER/LETTER-TIGER/ckpt/Instruments_tiger
    BEST_CKPT=$(ls -d $CKPT_DIR/checkpoint-* 2>/dev/null | sort -V | tail -1)
    if [ -z "$BEST_CKPT" ]; then
        echo "[FAIL] No checkpoint found in $CKPT_DIR (task78 not yet complete)"
        return 1
    fi
    echo "[RUN] TIGER T5 inference: $BEST_CKPT"
    source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh
    conda activate /home/wlia0047/ar57_scratch/wenyu/grid_toys
    
    cd $GENE_REC/external/LETTER/LETTER-TIGER
    python3 test.py \
        --gpu_id 0 \
        --ckpt_path $BEST_CKPT \
        --dataset Instruments \
        --data_path /home/wlia0047/ar57/wenyu/GeneRec/external/LETTER/data \
        --test_batch_size 32 \
        --num_beams 20 \
        --test_prompt_ids 0 \
        --index_file /home/wlia0047/ar57/wenyu/GeneRec/external/LETTER/LETTER-TIGER/Instruments_tiger.index.json \
        > $LOGS/task84_tiger_t5_inference_${TS}.log 2>&1
    echo "[DONE] task84 → $LOGS/task84_tiger_t5_inference_${TS}.log"
}

run_fdsa() {
    # Task #80 (FDSA) RecBole: evaluate test set from best checkpoint
    # Best model auto-saved by RecBole early stopping
    CKPT_FILE=$GENE_REC/RecBole/saved/FDSA-*.pth
    if ! ls $CKPT_FILE > /dev/null 2>&1; then
        echo "[FAIL] No FDSA checkpoint found"
        return 1
    fi
    source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh
    conda activate /home/wlia0047/ar57_scratch/wenyu/recbole_env
    
    cd $GENE_REC/RecBole
    python3 run_recbole_baseline.py --model=FDSA --gpu_id=0 \
        --config=musical_instruments.yaml \
        > $LOGS/task85_fdsa_inference_${TS}.log 2>&1
    echo "[DONE] task85 FDSA → $LOGS/task85_fdsa_inference_${TS}.log"
}

run_s3rec() {
    CKPT_FILE=$GENE_REC/RecBole/saved/S3Rec-*.pth
    if ! ls $CKPT_FILE > /dev/null 2>&1; then
        echo "[FAIL] No S3Rec checkpoint found"
        return 1
    fi
    source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh
    conda activate /home/wlia0047/ar57_scratch/wenyu/recbole_env
    
    cd $GENE_REC/RecBole
    python3 run_recbole_baseline.py --model=S3Rec --gpu_id=0 \
        --config=musical_instruments_sequential_paper.yaml \
        > $LOGS/task85_s3rec_inference_${TS}.log 2>&1
    echo "[DONE] task85 S3Rec → $LOGS/task85_s3rec_inference_${TS}.log"
}

run_p5_cid() {
    CKPT=$PRODUCTS/task82/p5_cid_instruments.pt
    if [ ! -f $CKPT ]; then
        echo "[FAIL] No P5-CID checkpoint found at $CKPT"
        return 1
    fi
    source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh
    conda activate /home/wlia0047/ar57_scratch/wenyu/llm_recsys_id_env
    
    cd $GENE_REC/external/LLM-RecSys-ID
    python3 main.py --seed 42 \
        --task instruments \
        --eval_only \
        --model_dir $CKPT \
        --item_representation CF \
        --number_of_items 24588 \
        --data_order remapped_sequential \
        --max_history 20 \
        --logging_dir $LOGS/task86_p5_cid_inference_${TS}.log \
        > $LOGS/task86_p5_cid_inference_${TS}.log 2>&1
    echo "[DONE] task86 P5-CID → $LOGS/task86_p5_cid_inference_${TS}.log"
}

run_p5_sid() {
    CKPT=$PRODUCTS/task83/p5_sid_instruments.pt
    if [ ! -f $CKPT ]; then
        echo "[FAIL] No P5-SID checkpoint found at $CKPT"
        return 1
    fi
    source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh
    conda activate /home/wlia0047/ar57_scratch/wenyu/llm_recsys_id_env
    
    cd $GENE_REC/external/LLM-RecSys-ID
    python3 main.py --seed 42 \
        --task instruments \
        --eval_only \
        --model_dir $CKPT \
        --item_representation remapped_sequential \
        --number_of_items 24588 \
        --data_order remapped_sequential \
        --max_history 20 \
        --logging_dir $LOGS/task86_p5_sid_inference_${TS}.log \
        > $LOGS/task86_p5_sid_inference_${TS}.log 2>&1
    echo "[DONE] task86 P5-SID → $LOGS/task86_p5_sid_inference_${TS}.log"
}

case "${1:-all}" in
    tiger)    run_tiger ;;
    fdsa)     run_fdsa ;;
    s3rec)    run_s3rec ;;
    p5_cid)   run_p5_cid ;;
    p5_sid)   run_p5_sid ;;
    all)
        run_tiger
        run_fdsa
        run_s3rec
        run_p5_cid
        run_p5_sid
        ;;
    *) echo "Usage: $0 [tiger|fdsa|s3rec|p5_cid|p5_sid|all]"; exit 1 ;;
esac

echo "[COMPLETE] Stage 4 inference finished. Results: $RESULT_DIR"
