#!/bin/bash
source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh
conda activate /home/wlia0047/ar57_scratch/wenyu/genrec_env
cd /home/wlia0047/ar57/wenyu/GeneRec/LETTER/LETTER-TIGER

export CUDA_VISIBLE_DEVICES=1,2
torchrun --nproc_per_node=2 --master_port=2315 ./finetune_flexible_gpu.py \
    --output_dir ./ckpt/Instruments/ \
    --dataset Instruments \
    --per_device_batch_size 256 \
    --learning_rate 5e-4 \
    --epochs 200 \
    --index_file .index.json \
    --temperature 1.0
