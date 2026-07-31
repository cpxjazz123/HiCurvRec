#!/bin/bash
# Task #350 / Issue #62 — Gate 4 Stage 4 eval (Arm C #30+#43 联合 SID)
# 2026-07-31
#
# 背景: Issue #62 §Gate 1+2+3 PASS (Stage 1 L0/L1/L2 100% util + Stage 2 4-digit unique +
#       Stage 3 T5-mini 200 epoch training 完成). Gate 4 = Stage 4 评估 Test R@10 > 0.1042 (#43 单点).
# R11.3: 沿用 task301 Issue #30 Stage 4 eval recipe (mirror Task #84 pattern)
# R7: GPU 1 (R7 空闲)
# R12: 复用 Stage 3 best ckpt
# R14: Issue #62 hard-stop at Gate 4 fail

set -o pipefail

source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh
conda activate /home/wlia0047/ar57_scratch/wenyu/genrec_env

REPO=/home/wlia0047/ar57/wenyu/GeneRec
cd $REPO/HG-Rec

export PYTHONPATH=$REPO/HG-Rec:${PYTHONPATH:-}
export CUDA_VISIBLE_DEVICES=1
export HF_HOME=/home/wlia0047/scratch/wenyu/hf_models
export TRITON_CACHE_DIR=/home/wlia0047/.triton/cache_task350_gate4

LOG_DIR=$REPO/logs/task350
TS=$(date +%b-%d-%Y_%H-%M-%S | sed 's/^.../\L&/')
LOG_FILE=$LOG_DIR/stage4_eval_${TS}.log
RESULT_JSON=$REPO/verdicts/task350_issue62_armc_stage4_metrics.json

mkdir -p $LOG_DIR

echo "===== [Task #350 Issue #62 Gate 4 Stage 4] #30+#43 Arm C eval launched at $(date) =====" | tee $LOG_FILE
echo "code_path = _t5_hrqvae_issue62_gate1_armc.npy (Arm C #30+#43)" | tee -a $LOG_FILE
echo "GPU 1 (R7 空闲)" | tee -a $LOG_FILE
echo "Output JSON: $RESULT_JSON" | tee -a $LOG_FILE

BEST_CKPT=$(ls -t $REPO/products/task350/ckpt_hgrec_issue62_armc/Instruments/*/HG_Rec_best.pth 2>/dev/null | head -1)
if [ -z "$BEST_CKPT" ] || [ ! -f "$BEST_CKPT" ]; then
    echo "❌ Stage 3 best ckpt MISSING" | tee -a $LOG_FILE
    exit 1
fi
echo "Stage 3 best ckpt: $BEST_CKPT" | tee -a $LOG_FILE

PYTHON_BIN=/home/wlia0047/ar57_scratch/wenyu/genrec_env/bin/python

$PYTHON_BIN -c "
import sys, os, glob, torch, json
sys.path.insert(0, '$REPO/HG-Rec')
sys.path.insert(0, '$REPO/scripts')

from model.HG_Rec import HG_Rec
from data.dataset import GenRecDataset
from data.dataloader import GenRecDataLoader

import importlib.util as _ilu
_s4_spec = _ilu.spec_from_file_location(
    'task84_s3_train_fork',
    '$REPO/scripts/task84_hgrec_stage3_train.py',
)
_s4_mod = _ilu.module_from_spec(_s4_spec)
_s4_spec.loader.exec_module(_s4_mod)
evaluate = _s4_mod.evaluate

# Config (mirror Task #84 + Issue #30 except codebook_size + code_path per Issue #62 Arm C)
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
    'dataset_path': '$REPO/HG-Rec/dataset/',
    'codebook_size': [64, 128, 256, 1],   # *** Issue #62 Arm C baseline K + 4th dedup ***
    'code_path': '_t5_hrqvae_issue62_gate1_armc.npy',   # *** Issue #62 Arm C ***
    'topk_list': [5, 10, 20],
    'beam_size': 20,
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
print(f'[Stage 4 Issue #62 Arm C] Test dataset size: {len(test_dataset)}, codebook={config[\"codebook_size\"]}')

avg_recalls, avg_ndcgs = evaluate(model, test_dataloader, config['topk_list'], config['beam_size'], device)

result = {
    'task': 'task350_issue62_armc_30_43_joint',
    'recipe': '#30 per-layer r_l=[0.1,1,10]+s_l=[2,2,2] + #43 HypPreEncoder (Issue #62 Arm C, K=[64,128,256] + Sinkhorn ON)',
    'best_ckpt': '$BEST_CKPT',
    'beam_size': 20,
    'test_recalls': avg_recalls,
    'test_ndcgs': avg_ndcgs,
}
os.makedirs(os.path.dirname('$RESULT_JSON'), exist_ok=True)
with open('$RESULT_JSON', 'w') as f:
    json.dump(result, f, indent=2)
print(f'[Stage 4 Issue #62 Arm C] Saved: $RESULT_JSON')
print(f'[Stage 4 Issue #62 Arm C] Test recalls: {avg_recalls}')
print(f'[Stage 4 Issue #62 Arm C] Test NDCGs: {avg_ndcgs}')
" 2>&1 | tee -a $LOG_FILE