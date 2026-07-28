#!/bin/bash
# Task #155 — HG-Rec num_beams=50 re-eval (filter_items=False 替代方案)
# 2026-07-24 (用户 2026-07-24 提议 filter_items=False, HG-Rec 无此概念 → 用 num_beams=50 作为最接近替代)
#
# 唯一改动 vs Task #84 baseline: beam_size=20 → 50 (增大搜索空间)
# 其他完全一致 (复用 Task #84 best ckpt)
#
# R7 GPU: GPU 3 (空闲)
# 预计: Stage 4 re-eval ~5-12 min (beam 50 比 20 慢 ~2.5x)

set -eo pipefail

export CUDA_VISIBLE_DEVICES=3
export PYTHONPATH=/home/wlia0047/ar57/wenyu/GeneRec/HG-Rec:${PYTHONPATH:-}

source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh
conda activate /home/wlia0047/ar57_scratch/wenyu/genrec_env

LOG_DIR=/home/wlia0047/ar57/wenyu/GeneRec/logs/task155
mkdir -p "$LOG_DIR"

BEST_CKPT=/home/wlia0047/ar57/wenyu/GeneRec/products/task84/ckpt_hgrec/Instruments/Jul-23-2026_20-24-44/HG_Rec_best.pth
RESULT_JSON=/home/wlia0047/ar57/wenyu/GeneRec/verdicts/task155_hgrec_beams50_metrics.json

if [ ! -f "$BEST_CKPT" ]; then
    echo "❌ Task #84 best ckpt NOT FOUND: $BEST_CKPT"
    exit 1
fi

cd /home/wlia0047/ar57/wenyu/GeneRec/HG-Rec

TS=$(date +%Y%m%d_%H%M%S)
LOG_FILE="$LOG_DIR/beams50_eval_${TS}.log"

echo "===== [Task #155] HG-Rec num_beams=50 eval launched at $(date) =====" | tee "$LOG_FILE"
echo "GPU: 3 (R7 空闲)" | tee -a "$LOG_FILE"
echo "Best ckpt: $BEST_CKPT (Task #84 baseline)" | tee -a "$LOG_FILE"
echo "唯一改动: beam_size 20 → 50" | tee -a "$LOG_FILE"
echo "Output JSON: $RESULT_JSON" | tee -a "$LOG_FILE"

python3 -c "
import sys, os, glob, torch, json
sys.path.insert(0, '/home/wlia0047/ar57/wenyu/GeneRec/HG-Rec')
sys.path.insert(0, '/home/wlia0047/ar57/wenyu/GeneRec/scripts')
from torch.utils.data import DataLoader
import numpy as np
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

# 配置 (mirror Stage 3, 但 beam_size=50)
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
    'beam_size': 50,  # *** Task #155 唯一改动 ***
}

device = torch.device('cuda:0')
model = HG_Rec(config)
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
print(f'[Task #155] Test dataset: {len(test_dataset)}, beam_size={config[\"beam_size\"]}')

avg_recalls, avg_ndcgs = evaluate(model, test_dataloader, config['topk_list'], config['beam_size'], device)

result = {
    'task': 'task155_hgrec_beams50',
    'best_ckpt': '$BEST_CKPT',
    'beam_size': 50,
    'test_recalls': avg_recalls,
    'test_ndcgs': avg_ndcgs,
}
os.makedirs(os.path.dirname('$RESULT_JSON'), exist_ok=True)
with open('$RESULT_JSON', 'w') as f:
    json.dump(result, f, indent=2)
print(f'[Task #155] Saved: $RESULT_JSON')
print(f'[Task #155] Recalls: {avg_recalls}')
print(f'[Task #155] NDCGs: {avg_ndcgs}')
" 2>&1 | tee -a "$LOG_FILE"

EXIT_CODE=${PIPESTATUS[0]}
echo "===== [Task #155] eval done at $(date), exit=$EXIT_CODE =====" | tee -a "$LOG_FILE"