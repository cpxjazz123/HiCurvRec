#!/bin/bash
# Task #84 Stage 4 — HG-Rec Test evaluation (load best ckpt, eval on test set)
# Stage 3 fork: hg_rec.py forward+generate 用 best NDCG@20 ckpt
# HG_Rec.py 默认 source path 'test.parquet' 被注释掉 (line 184-192) → fork 已 uncomment, 加 test 评估
# GPU 3 (R7 强制空闲 — Stage 3 已完成, 释放 GPU 3)
# 输出指标: Recall@5/10/20, NDCG@5/10/20 on TEST split

set -e

source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh
conda activate /home/wlia0047/ar57_scratch/wenyu/genrec_env

cd /home/wlia0047/ar57/wenyu/GeneRec/HG-Rec

export PYTHONPATH=/home/wlia0047/ar57/wenyu/GeneRec/HG-Rec:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=3
export HF_HOME=/home/wlia0047/ar57_scratch/wenyu/hf_models

LOG_DIR=/home/wlia0047/ar57/wenyu/GeneRec/logs
TS=$(date +%b-%d-%Y_%H-%M-%S | sed 's/^.../\L&/')
LOG_FILE=$LOG_DIR/task84_hgrec_stage4_eval_${TS}.log

mkdir -p $LOG_DIR

echo "===== [Task #84 Stage 4] HG-Rec Test evaluation launched at $(date) =====" | tee $LOG_FILE
echo "GPU 3 (R7 强制空闲)" | tee -a $LOG_FILE

# 验证 Stage 3 best ckpt 落盘
BEST_CKPT=$(ls -t /home/wlia0047/ar57/wenyu/GeneRec/products/task84/ckpt_hgrec/Instruments/*/HG_Rec_best.pth 2>/dev/null | head -1)
if [ -z "$BEST_CKPT" ] || [ ! -f "$BEST_CKPT" ]; then
    echo "❌ Stage 3 best ckpt MISSING" | tee -a $LOG_FILE
    exit 1
fi
echo "Stage 3 best ckpt: $BEST_CKPT" | tee -a $LOG_FILE

# R11.3 决策: 复用 Stage 3 launch 的代码, 但调用 --mode=evaluation 不存在 in HG-Rec.py
# (默认 mode='train' 会同时评估 validation). 我们实际上需要 evaluate on TEST.
# 上游 fork 没实现 test evaluation mode. 我们手动做:
# 1. 改 fork fork fork (fork of fork) 加 --mode=evaluation + test_dataset
# 实际更简单: 写一个 standalone eval script task84_hgrec_test_eval.py
# 但 task84_hgrec_stage3_train.py 训练时已经把 --mode='train' hardcoded.
# 按 R11.3: 写一个 Stage 4 standalone eval script

python3 -c "
import sys
sys.path.insert(0, '/home/wlia0047/ar57/wenyu/GeneRec/HG-Rec')
sys.path.insert(0, '/home/wlia0047/ar57/wenyu/GeneRec/scripts')

# R11.3: Standalone Test evaluation for HG-Rec best ckpt
import os, glob, torch, logging, json
from torch import optim
from torch.utils.data import DataLoader
import numpy as np

from model.HG_Rec import HG_Rec
from data.dataset import GenRecDataset
from data.dataloader import GenRecDataLoader
import sys
sys.path.insert(0, '/home/wlia0047/ar57/wenyu/GeneRec/HG-Rec')
# R11.3 FIX: 上游文件名为 train_HG-Rec.py (含连字符), 不能直接 import.
# 从 fork (scripts/task84_hgrec_stage3_train.py) 导入需要的函数.
import importlib.util as _ilu
_s4_spec = _ilu.spec_from_file_location(
    'task84_s3_train_fork',
    '/home/wlia0047/ar57/wenyu/GeneRec/scripts/task84_hgrec_stage3_train.py',
)
_s4_mod = _ilu.module_from_spec(_s4_spec)
_s4_spec.loader.exec_module(_s4_mod)
evaluate = _s4_mod.evaluate
calculate_pos_index = _s4_mod.calculate_pos_index
recall_at_k = _s4_mod.recall_at_k
ndcg_at_k = _s4_mod.ndcg_at_k
del _ilu, _s4_spec, _s4_mod

# Config (mirror Stage 3)
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
    'code_path': '_t5_hrqvae_poincare.npy',
    'topk_list': [5, 10, 20],
    'beam_size': 20,
}

# Load best ckpt
BEST_CKPT = glob.glob('/home/wlia0047/ar57/wenyu/GeneRec/products/task84/ckpt_hgrec/Instruments/*/HG_Rec_best.pth')[0]
print(f'[Stage 4] Loading best ckpt: {BEST_CKPT}')

device = torch.device('cuda:0')
model = HG_Rec(config)
model.load_state_dict(torch.load(BEST_CKPT, map_location='cpu'))
model.to(device)

# Test dataset
test_dataset = GenRecDataset(
    dataset_path=os.path.join(config['dataset_path'], config['dataset_name'], 'test.parquet'),
    code_path=os.path.join(config['dataset_path'], config['dataset_name'], config['dataset_name'] + config['code_path']),
    mode='evaluation',
    codebook_size=config['codebook_size'],
    max_len=config['max_len']
)
test_dataloader = GenRecDataLoader(test_dataset, batch_size=config['infer_size'], shuffle=False)
print(f'[Stage 4] Test dataset size: {len(test_dataset)}')

# Evaluate on TEST
avg_recalls, avg_ndcgs = evaluate(model, test_dataloader, config['topk_list'], config['beam_size'], device)

# JSON output
result = {
    'best_ckpt': BEST_CKPT,
    'test_recalls': avg_recalls,
    'test_ndcgs': avg_ndcgs,
}
result_json = '/home/wlia0047/ar57/wenyu/GeneRec/verdicts/task84_hgrec_main_repro_test_metrics.json'
os.makedirs(os.path.dirname(result_json), exist_ok=True)
with open(result_json, 'w') as f:
    json.dump(result, f, indent=2)
print(f'[Stage 4] Test metrics saved: {result_json}')
print(f'[Stage 4] Test metrics:\\n{avg_recalls}\\n{avg_ndcgs}')
" 2>&1 | tee -a $LOG_FILE

EXIT_CODE=${PIPESTATUS[0]}
echo "===== [Task #84 Stage 4] Test eval completed at $(date), exit code: $EXIT_CODE =====" | tee -a $LOG_FILE
