#!/bin/bash
# Task #33 launcher — phonism TIGER 200 epochs × seed=123 on cuda:1
set -e
source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh
conda activate /home/wlia0047/ar57_scratch/wenyu/genrec_env
cd /home/wlia0047/ar57/wenyu/genrec
export PYTHONPATH="/home/wlia0047/ar57/wenyu/genrec:${PYTHONPATH}"
export CUDA_VISIBLE_DEVICES=1
mkdir -p /home/wlia0047/ar57/wenyu/GeneRec/logs

python -m genrec.trainers.tiger_trainer \
    config/tiger/amazon/tiger.gin \
    --split toys \
    --gin "train.epochs=200" \
    --gin "train.wandb_logging=False" \
    --gin "train.eval_test_every_epoch=2" \
    --gin "train.save_dir_root='out/tiger/amazon/toys/seed123/'" \
    --gin "train.pretrained_rqvae_path='out/tiger/amazon/toys/rqvae/checkpoint_19999.pt'" \
    --gin "train.seed=123" \
    --gin "AmazonSeqDataset.encoder_model_name='/home/wlia0047/ar57/wenyu/genrec/models_hub/sentence-t5-base'" \
    > /home/wlia0047/ar57/wenyu/GeneRec/logs/task33_phonism_seed123.log 2>&1
