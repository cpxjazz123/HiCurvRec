#!/bin/bash
# Task #209 A3 Stage 4 — Test evaluation (load best ckpt @ epoch 58)
# A3 = Task #209 Phase 1 完整方法 arm (--norm_target + --gamma_norm + --scale_norm + --w_path 1.0 hyp + rho_targets)
# A3 T5-mini: num_layers=6, num_decoder_layers=4, d_model=128, num_heads=6, vocab_size=11000, codebook=[64,128,256,1]
# GPU 0 (R7: 4 卡空闲)
set -e

REPO=/home/wlia0047/ar57/wenyu/GeneRec
source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh
conda activate /home/wlia0047/ar57_scratch/wenyu/grid_toys
cd $REPO/HG-Rec

export PYTHONPATH=$REPO/HG-Rec:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=0
export HF_HOME=/home/wlia0047/ar57_scratch/wenyu/hf_models
export TRITON_CACHE_DIR=/home/wlia0047/.triton/cache_task209_A3_s4
mkdir -p $TRITON_CACHE_DIR

mkdir -p $REPO/logs/task209 $REPO/verdicts

BEST_CKPT=$REPO/products/task209/t5small_A3/Instruments/Jul-26-2026_18-40-36/HG_Rec_best.pth
if [ ! -f "$BEST_CKPT" ]; then
    echo "❌ A3 Stage 3 best ckpt MISSING: $BEST_CKPT"
    exit 1
fi
echo "✅ A3 Stage 3 best ckpt: $BEST_CKPT"

LOG_FILE=$REPO/logs/task209/A3_stage4_eval_$(date +%b-%d-%Y_%H-%M-%S | sed 's/^.../\L&/').log
RESULT_JSON=$REPO/verdicts/task209_A3_test_metrics.json

echo "===== [Task #209 A3 Stage 4] Test eval launched at $(date) =====" | tee $LOG_FILE

python3 -u <<PYTHON_EOF 2>&1 | tee -a "$LOG_FILE"
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

# A3 训练时实际参数 (从 task209_phase2b_A3_stage3_train.sh)
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
    'vocab_size': 11000,
    'pad_token_id': 0,
    'eos_token_id': 0,
    'feed_forward_proj': 'relu',
    'max_len': 20,
    'dataset_name': 'Instruments',
    'dataset_path': '/home/wlia0047/ar57/wenyu/GeneRec/HG-Rec/dataset/',
    'codebook_size': [64, 128, 256, 1],
    'code_path': '_t5_rqvae_task209_A3.npy',
    'topk_list': [5, 10, 20],
    'beam_size': 20,
}

device = torch.device('cuda:0')
model = HG_Rec(config)
print(f'[Task #209 A3 Stage 4] Loading best ckpt...', flush=True)
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
print(f'[Task #209 A3 Stage 4] Test dataset size: {len(test_dataset)}', flush=True)

avg_recalls, avg_ndcgs = evaluate(model, test_dataloader, config['topk_list'], config['beam_size'], device)

result = {
    'task': 'task209_A3_stage4_test',
    'arm': 'A3',
    'recipe': 'Task #209 Phase 1 A3 (--norm_target + --gamma_norm + --scale_norm poincare + --w_path 1.0 hyp + rho_targets) + T5-mini 9.18M',
    'best_ckpt': '$BEST_CKPT',
    'val_best_epoch': 58,
    'val_best_NDCG@20': 0.0882,
    'val_R@10_best': 0.1050,
    'beam_size': 20,
    'test_recalls': avg_recalls,
    'test_ndcgs': avg_ndcgs,
}
os.makedirs(os.path.dirname('$RESULT_JSON'), exist_ok=True)
with open('$RESULT_JSON', 'w') as f:
    json.dump(result, f, indent=2)
print(f'[Task #209 A3 Stage 4] Saved: $RESULT_JSON', flush=True)
print(f'[Task #209 A3 Stage 4] Test recalls: {avg_recalls}', flush=True)
print(f'[Task #209 A3 Stage 4] Test NDCGs: {avg_ndcgs}', flush=True)
PYTHON_EOF

EXIT_CODE=${PIPESTATUS[0]}
echo "===== [Task #209 A3 Stage 4] exit code: $EXIT_CODE at $(date) =====" | tee -a $LOG_FILE
[ $EXIT_CODE -ne 0 ] && exit 1

echo "✅ Task #209 A3 Stage 4 完成" | tee -a $LOG_FILE
cat $RESULT_JSON | tee -a $LOG_FILE