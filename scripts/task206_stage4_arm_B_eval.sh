#!/bin/bash
# Task #206 Stage 4 — Arm B (欧式) Recall eval vs HG-Rec baseline
#
# 跟 task188_stage4_eval.sh 同 pattern, 但只跑一次 (single-arm 对照).
# 比较 Arm A (双曲, #84 R@10=0.1020) vs Arm B (欧式, 本次).
# 这是论文基调决定实验.

set -e
source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh
conda activate /home/wlia0047/ar57_scratch/wenyu/grid_toys

REPO=/home/wlia0047/ar57/wenyu/GeneRec
export PYTHONPATH=$REPO/HG-Rec:$PYTHONPATH
export HF_HOME=/home/wlia0047/ar57_scratch/wenyu/hf_models
mkdir -p $REPO/logs/task206 $REPO/verdicts

GPU=0
CKPT_DIR=$REPO/products/task206/t5small_euclidean/Instruments
BEST_CKPT=$(ls -t $CKPT_DIR/*/HG_Rec_best.pth 2>/dev/null | head -1)
RESULT_JSON=$REPO/verdicts/task206_arm_B_euclidean_metrics.json
LOG_DIR=$REPO/logs/task206/stage4_arm_B_eval
mkdir -p $LOG_DIR

if [ -z "$BEST_CKPT" ] || [ ! -f "$BEST_CKPT" ]; then
    echo "❌ Best ckpt MISSING for Arm B. 检查 $CKPT_DIR/"
    exit 1
fi

CODE_PATH=_t5_rqvae_euclidean_v1.npy
COLLISION=0.5866   # 1000ep Arm B 最佳 collision rate (从 Stage 1 log 读)

echo "[$(date)] === #206 Stage 4 Arm B: ckpt=$BEST_CKPT, GPU=$GPU ===" | tee $LOG_DIR/stage4_eval.log

cd $REPO/HG-Rec

CUDA_VISIBLE_DEVICES=$GPU TRITON_CACHE_DIR=/home/wlia0047/.triton/cache_task206_stage4_arm_B \
python3 -u <<PYTHON_EOF 2>&1 | tee -a $LOG_DIR/stage4_eval.log
import sys, os, json
sys.path.insert(0, '/home/wlia0047/ar57/wenyu/GeneRec/HG-Rec')
sys.path.insert(0, '/home/wlia0047/ar57/wenyu/GeneRec/scripts')

import torch
from model.HG_Rec import HG_Rec
from data.dataset import GenRecDataset
from data.dataloader import GenRecDataLoader

import importlib.util as _ilu
_s4_spec = _ilu.spec_from_file_location(
    'task84_s3_train_fork',
    '/home/wlia0047/ar57/wenyu/GeneRec/scripts/task84_hgrec_stage3_train.py',
)
_s4_mod = _ilu.module_from_spec(_s4_spec)
_s4_spec.loader.exec_module(_s4_mod)
evaluate = _s4_mod.evaluate

config = {
    'batch_size': 256,
    'infer_size': 96,
    'lr': 1e-4,
    'device': 'cuda:0',
    'num_layers': 6,
    'num_decoder_layers': 4,
    'd_model': 128,
    'd_ff': 1024,
    'num_heads': 6,
    'd_kv': 64,
    'dropout_rate': 0.1,
    'vocab_size': 1025,
    'pad_token_id': 0,
    'eos_token_id': 0,
    'feed_forward_proj': 'relu',
    'max_len': 20,
    'dataset_name': 'Instruments',
    'dataset_path': '/home/wlia0047/ar57/wenyu/GeneRec/HG-Rec/dataset/',
    'codebook_size': [64, 128, 256, 1],
    'code_path': '$CODE_PATH',
    'topk_list': [5, 10, 20],
    'beam_size': 20,
}

device = torch.device('cuda:0')
model = HG_Rec(config)
print(f'[Task #206 Stage 4 Arm B] Loading best ckpt...', flush=True)
model.load_state_dict(torch.load('$BEST_CKPT', map_location='cpu'))
model.to(device)

test_dataset = GenRecDataset(
    dataset_path=os.path.join(config['dataset_path'], config['dataset_name'], 'test.parquet'),
    code_path=os.path.join(config['dataset_path'], config['dataset_name'], config['dataset_name'] + config['code_path']),
    mode='evaluation',
    codebook_size=config['codebook_size'],
    max_len=config['max_len']
)
test_dataloader = GenRecDataLoader(test_dataset, batch_size=config['infer_size'], shuffle=False)
print(f'[Task #206 Stage 4 Arm B] Test dataset size: {len(test_dataset)}', flush=True)

avg_recalls, avg_ndcgs = evaluate(model, test_dataloader, config['topk_list'], config['beam_size'], device)

result = {
    'task': 'task206_euclidean_vs_hyperbolic',
    'arm': 'B_euclidean',
    'collision_rate': $COLLISION,
    'recipe': 'T5-small 5.5M + Euclidean argmin + L2 commit/code',
    'best_ckpt': '$BEST_CKPT',
    'selected_by': 'best_val_NDCG@20',
    'beam_size': 20,
    'test_recalls': avg_recalls,
    'test_ndcgs': avg_ndcgs,
    'hgrec_baseline_arm_A_R@10': 0.1020,
}
os.makedirs(os.path.dirname('$RESULT_JSON'), exist_ok=True)
with open('$RESULT_JSON', 'w') as f:
    json.dump(result, f, indent=2)
print(f'[Task #206 Stage 4 Arm B] Saved: $RESULT_JSON', flush=True)
print(f'[Task #206 Stage 4 Arm B] Test recalls: {avg_recalls}', flush=True)
print(f'[Task #206 Stage 4 Arm B] Test NDCGs: {avg_ndcgs}', flush=True)
print(f'[Task #206 Stage 4 Arm B] HG-Rec baseline R@10 = 0.1020', flush=True)
PYTHON_EOF
echo "[$(date)] === Stage 4 Arm B DONE ===" | tee -a $LOG_DIR/stage4_eval.log