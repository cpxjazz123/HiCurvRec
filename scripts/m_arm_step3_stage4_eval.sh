#!/bin/bash
# 方向 F Stage 4: v11/v12 SID-trained T5-mini test R@10 eval
# Usage: bash m_arm_step3_stage4_eval.sh <arm>
#   arm ∈ {v11_angdim8, v12_angdim16}

set -e

ARM=${1:-v11_angdim8}

source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh
conda activate /home/wlia0047/ar57_scratch/wenyu/grid_toys

cd /home/wlia0047/ar57/wenyu/GeneRec/HG-Rec

export PYTHONPATH=/home/wlia0047/ar57/wenyu/GeneRec/HG-Rec:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export HF_HOME=/home/wlia0047/ar57_scratch/wenyu/hf_models
export TRITON_CACHE_DIR=/home/wlia0047/.triton/cache_marm_${ARM}_s4

LOG_DIR=/home/wlia0047/ar57/wenyu/GeneRec/logs/m_arm_step3
mkdir -p $LOG_DIR

# Map arm to (SID suffix, products subdir, codebook_size, hgrec_config)
case "$ARM" in
    v11_angdim8)
        SID_SUFFIX="_t5_hrqvae_m_arm_v11_angdim8_ep4.npy"
        PROD_SUBDIR="step3_v11_angdim8_ep4_sid"
        CODEBOOK="64 128 256 1"
        ;;
    v12_angdim16)
        SID_SUFFIX="_t5_hrqvae_m_arm_v12_angdim16_ep4.npy"
        PROD_SUBDIR="step3_v12_angdim16_ep4_sid"
        CODEBOOK="64 128 256 1"
        ;;
    *)
        echo "Unknown arm: $ARM (must be v11_angdim8 or v12_angdim16)"
        exit 1
        ;;
esac

TS=$(date +%b-%d-%Y_%H-%M-%S | sed 's/^.../\L&/')
LOG_FILE=$LOG_DIR/stage4_eval_${ARM}_${TS}.log
RESULT_JSON=/home/wlia0047/ar57/wenyu/GeneRec/verdicts/task112_m_arm_${ARM}_metrics.json

echo "===== [Task #112 Stage 4] M-arm $ARM T5-mini 9.18M test eval launched at $(date) =====" | tee $LOG_FILE

BEST_CKPT=$(ls -t /home/wlia0047/ar57/wenyu/GeneRec/products/m_arm/${PROD_SUBDIR}/*/Instruments/*/HG_Rec_best.pth 2>/dev/null | head -1)
if [ -z "$BEST_CKPT" ] || [ ! -f "$BEST_CKPT" ]; then
    echo "❌ Stage 3 best ckpt MISSING for $ARM (look in products/m_arm/$PROD_SUBDIR/...)" | tee -a $LOG_FILE
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
    'codebook_size': [int(x) for x in '$CODEBOOK'.split()],
    'code_path': '$SID_SUFFIX',
    'topk_list': [5, 10, 20],
    'beam_size': 20,
}

device = torch.device('cuda:0')
model = HG_Rec(config)
print(f'[Task #112 Stage 4 arm=$ARM] Loading best ckpt...', flush=True)
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
print(f'[Task #112 Stage 4 arm=$ARM] Test dataset size: {len(test_dataset)}', flush=True)

avg_recalls, avg_ndcgs = evaluate(model, test_dataloader, config['topk_list'], config['beam_size'], device)

result = {
    'task': f'task112_m_arm_$ARM',
    'arm': '$ARM',
    'recipe': 'T5-mini 9.18M (6L/4D, d_model=128) + M-arm v11/v12 ep4 SID, 200 epoch early_stop=20',
    'best_ckpt': '$BEST_CKPT',
    'beam_size': 20,
    'test_recalls': avg_recalls,
    'test_ndcgs': avg_ndcgs,
    'baseline_r10': 0.1020,
}
os.makedirs(os.path.dirname('$RESULT_JSON'), exist_ok=True)
with open('$RESULT_JSON', 'w') as f:
    json.dump(result, f, indent=2)
print(f'[Task #112 Stage 4 arm=$ARM] Saved: $RESULT_JSON', flush=True)
print(f'[Task #112 Stage 4 arm=$ARM] Test recalls: {avg_recalls}', flush=True)
print(f'[Task #112 Stage 4 arm=$ARM] Test NDCGs: {avg_ndcgs}', flush=True)
print(f'[Task #112 Stage 4 arm=$ARM] vs HG-Rec baseline R@10={result["baseline_r10"]}', flush=True)
PYTHON_EOF

EXIT_CODE=${PIPESTATUS[0]}
echo "===== [Task #112 Stage 4 arm=$ARM] exit code: $EXIT_CODE at $(date) =====" | tee -a $LOG_FILE
[ $EXIT_CODE -ne 0 ] && exit 1

echo "✅ Task #112 Stage 4 $ARM 完成" | tee -a $LOG_FILE
cat $RESULT_JSON | tee -a $LOG_FILE
