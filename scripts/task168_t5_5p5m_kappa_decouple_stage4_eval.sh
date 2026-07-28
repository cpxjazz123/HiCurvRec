#!/bin/bash
# Task #168 Stage 4 — T5-5.5M + κ-Stereographic SID test eval
# R88 daemon 自动触发 (training PID 退出后)

set -e

source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh
conda activate /home/wlia0047/ar57_scratch/wenyu/grid_toys

cd /home/wlia0047/ar57/wenyu/GeneRec/HG-Rec

export PYTHONPATH=/home/wlia0047/ar57/wenyu/GeneRec/HG-Rec:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=3
export HF_HOME=/home/wlia0047/ar57_scratch/wenyu/hf_models
export TRITON_CACHE_DIR=/home/wlia0047/.triton/cache_task168_s4

LOG_DIR=/home/wlia0047/ar57/wenyu/GeneRec/logs/task168
TS=$(date +%b-%d-%Y_%H-%M-%S | sed 's/^.../\L&/')
LOG_FILE=$LOG_DIR/stage4_eval_${TS}.log
RESULT_JSON=/home/wlia0047/ar57/wenyu/GeneRec/verdicts/task168_t5_5p5m_kappa_metrics.json

mkdir -p $LOG_DIR

echo "===== [Task #168 Stage 4] T5-5.5M + κ-Stereographic SID test eval launched at $(date) =====" | tee $LOG_FILE

# FIX: 实际 ckpt 路径是 t5_5p5m_kappa_decouple/*/Instruments/*/HG_Rec_best.pth (跟 task166/167 修过的 glob 模式一致)
BEST_CKPT=$(ls -t /home/wlia0047/ar57/wenyu/GeneRec/products/task168/t5_5p5m_kappa_decouple/*/Instruments/*/HG_Rec_best.pth 2>/dev/null | head -1)
if [ -z "$BEST_CKPT" ] || [ ! -f "$BEST_CKPT" ]; then
    echo "❌ Stage 3 best ckpt MISSING (products/task168/t5_5p5m_kappa_decouple/*/Instruments/*/HG_Rec_best.pth)" | tee -a $LOG_FILE
    exit 1
fi
echo "Stage 3 best ckpt: $BEST_CKPT" | tee -a $LOG_FILE

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

# T5-5.5M config (Task #168 R11.3 决策: 5+5 layers, d_model=192, d_ff=1024, 3 heads × d_kv=64)
config = {
    'batch_size': 256,
    'infer_size': 96,
    'lr': 1e-4,
    'device': 'cuda:0',
    'num_layers': 5,
    'num_decoder_layers': 5,
    'd_model': 192,
    'd_ff': 1024,
    'num_heads': 3,
    'd_kv': 64,
    'dropout_rate': 0.1,
    'vocab_size': 1025,
    'pad_token_id': 0,
    'eos_token_id': 0,
    'feed_forward_proj': 'relu',
    'max_len': 20,
    'dataset_name': 'Instruments',
    'dataset_path': '/home/wlia0047/ar57/wenyu/GeneRec/HG-Rec/dataset/',
    'codebook_size': [32, 64, 256, 1],
    'code_path': '_t5_rqvae_phase_b_kappa_decouple.npy',
    'topk_list': [5, 10, 20],
    'beam_size': 20,
}

device = torch.device('cuda:0')
model = HG_Rec(config)
print(f'[Task #168 Stage 4] Loading best ckpt...', flush=True)
model.load_state_dict(torch.load('$BEST_CKPT', map_location='cpu'))
model.to(device)
print(f'[Task #168 Stage 4] Model loaded on {device}', flush=True)
print(f'[Task #168 Stage 4] Total params: {sum(p.numel() for p in model.parameters()):,}', flush=True)

test_dataset = GenRecDataset(
    dataset_path=os.path.join(config['dataset_path'], config['dataset_name'], 'test.parquet'),
    code_path=os.path.join(config['dataset_path'], config['dataset_name'], config['dataset_name'] + config['code_path']),
    mode='evaluation',
    codebook_size=config['codebook_size'],
    max_len=config['max_len']
)
test_dataloader = GenRecDataLoader(test_dataset, batch_size=config['infer_size'], shuffle=False)
print(f'[Task #168 Stage 4] Test dataset size: {len(test_dataset)}', flush=True)

avg_recalls, avg_ndcgs = evaluate(model, test_dataloader, config['topk_list'], config['beam_size'], device)

result = {
    'task': 'task168_t5_5p5m_kappa_decouple',
    'recipe': 'T5-5.5M (5+5 layers, d_model=192, d_ff=1024, 3 heads × d_kv=64) + κ-Stereographic Phase B SID + 200 epoch',
    'best_ckpt': '$BEST_CKPT',
    'beam_size': 20,
    'test_recalls': avg_recalls,
    'test_ndcgs': avg_ndcgs,
}
os.makedirs(os.path.dirname('$RESULT_JSON'), exist_ok=True)
with open('$RESULT_JSON', 'w') as f:
    json.dump(result, f, indent=2)
print(f'[Task #168 Stage 4] Saved: $RESULT_JSON', flush=True)
print(f'[Task #168 Stage 4] Test recalls: {avg_recalls}', flush=True)
print(f'[Task #168 Stage 4] Test NDCGs: {avg_ndcgs}', flush=True)
PYTHON_EOF

EXIT_CODE=${PIPESTATUS[0]}
echo "===== [Task #168 Stage 4] exit: $EXIT_CODE at $(date) =====" | tee -a $LOG_FILE
[ $EXIT_CODE -ne 0 ] && exit 1

python3 -c "
import json
d = json.load(open('$RESULT_JSON'))
print('=== Task #168 Test Results (T5-5.5M + κ-Stereographic) ===')
r5, r10, r20 = d['test_recalls'][0], d['test_recalls'][1], d['test_recalls'][2]
n5, n10, n20 = d['test_ndcgs'][0], d['test_ndcgs'][1], d['test_ndcgs'][2]
print(f'R@5:  {r5:.4f}')
print(f'R@10: {r10:.4f}')
print(f'R@20: {r20:.4f}')
print(f'NDCG@5:  {n5:.4f}')
print(f'NDCG@10: {n10:.4f}')
print(f'NDCG@20: {n20:.4f}')
print()
print('Ablation across capacities (κ-Stereographic Phase B SID, 4 档):')
print(f'  T5-5.5M   (#168):   R@10 = {r10:.4f}  ⭐ (capacity ladder 新增最小档)')
print(f'  T5-mini 9.18M (#166): R@10 = [待 Stage 4 完成]')
print(f'  T5-small 60M (#165 v3): R@10 = [待 Stage 4 完成]')
print(f'  T5-base 220M (#167):   R@10 = [待 Stage 4 完成]')
print()
print('vs capacity-matched control (普通 SID):')
print(f'  无 T5-5.5M 普通 SID baseline (新档位, 估 ~0.10)')
print(f'  vs baseline 0.1058: ', '+' if r10 > 0.1058 else '-', f'{abs((r10 - 0.1058)/0.1058*100):.2f}%')
" | tee -a $LOG_FILE