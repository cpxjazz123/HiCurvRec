#!/bin/bash
# Task #162 Stage 4 — T5-mini 5.5M + paper-free-curv per-layer κ init [0,0,0] test eval
# Fork of task161 stage4, target = products/task162/...

set -e

source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh
conda activate /home/wlia0047/ar57_scratch/wenyu/genrec_env

cd /home/wlia0047/ar57/wenyu/GeneRec/HG-Rec

export PYTHONPATH=/home/wlia0047/ar57/wenyu/GeneRec/HG-Rec:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=1
export HF_HOME=/home/wlia0047/wenyu/scratch/hf_models
export TRITON_CACHE_DIR=/home/wlia0047/.triton/cache_task162

LOG_DIR=/home/wlia0047/ar57/wenyu/GeneRec/logs/task162
TS=$(date +%b-%d-%Y_%H-%M-%S | sed 's/^.../\L&/')
LOG_FILE=$LOG_DIR/stage4_eval_${TS}.log
RESULT_JSON=/home/wlia0047/ar57/wenyu/GeneRec/verdicts/task162_paper_free_curv_metrics.json

mkdir -p $LOG_DIR

echo "===== [Task #162 Stage 4] T5-mini 5.5M + paper-free-curv test eval launched at $(date) =====" | tee $LOG_FILE
echo "Recipe: T5-mini 5.5M (d_model=128, 4+4 layers, 4 heads × d_kv=32) + paper-free-curv SID (κ_init=[0,0,0])" | tee -a $LOG_FILE
echo "GPU 1" | tee -a $LOG_FILE
echo "Output JSON: $RESULT_JSON" | tee -a $LOG_FILE

export RESULT_JSON

BEST_CKPT=$(ls -t /home/wlia0047/ar57/wenyu/GeneRec/products/task162/ckpt_hgrec/Instruments/*/HG_Rec_best.pth 2>/dev/null | head -1)
if [ -z "$BEST_CKPT" ] || [ ! -f "$BEST_CKPT" ]; then
    echo "❌ Stage 3 best ckpt MISSING" | tee -a $LOG_FILE
    exit 1
fi
echo "Stage 3 best ckpt: $BEST_CKPT" | tee -a $LOG_FILE
export BEST_CKPT

python3 -u <<'PYTHON_EOF' 2>&1 | tee -a "$LOG_FILE"
import sys, os, json
sys.path.insert(0, '/home/wlia0047/ar57/wenyu/GeneRec/HG-Rec')
sys.path.insert(0, '/home/wlia0047/ar57/wenyu/GeneRec/scripts')

from model.HG_Rec import HG_Rec
from data.dataset import GenRecDataset
from data.dataloader import GenRecDataLoader

import importlib.util as _ilu
_s4_spec = _ilu.spec_from_file_location('task84_s3_train_fork', '/home/wlia0047/ar57/wenyu/GeneRec/scripts/task84_hgrec_stage3_train.py')
_s4_mod = _ilu.module_from_spec(_s4_spec)
_s4_spec.loader.exec_module(_s4_mod)
evaluate = _s4_mod.evaluate

config = {
    'batch_size': 256, 'infer_size': 96, 'lr': 1e-4, 'device': 'cuda:0',
    'num_layers': 4, 'num_decoder_layers': 4,
    'd_model': 128, 'd_ff': 512, 'num_heads': 4, 'd_kv': 32,
    'dropout_rate': 0.1, 'vocab_size': 4500,
    'pad_token_id': 0, 'eos_token_id': 0, 'feed_forward_proj': 'relu',
    'max_len': 20,
    'dataset_name': 'Instruments',
    'dataset_path': '/home/wlia0047/ar57/wenyu/GeneRec/HG-Rec/dataset/',
    'codebook_size': [32, 64, 256, 1],
    'code_path': '_t5_rqvae_paper_free_curv.npy',
    'topk_list': [5, 10, 20], 'beam_size': 20,
}

import torch
device = torch.device('cuda:0')
model = HG_Rec(config)
print(f'[Stage 4] Loading best ckpt...', flush=True)
model.load_state_dict(torch.load(os.environ['BEST_CKPT'], map_location='cpu'))
model.to(device)
print(f'[Stage 4] Model loaded on {device}', flush=True)

test_dataset = GenRecDataset(
    dataset_path=os.path.join(config['dataset_path'], config['dataset_name'], 'test.parquet'),
    code_path=os.path.join(config['dataset_path'], config['dataset_name'], config['dataset_name'] + config['code_path']),
    mode='evaluation', codebook_size=config['codebook_size'], max_len=config['max_len']
)
test_dataloader = GenRecDataLoader(test_dataset, batch_size=config['infer_size'], shuffle=False)
print(f'[Stage 4] Test dataset size: {len(test_dataset)}', flush=True)

avg_recalls, avg_ndcgs = evaluate(model, test_dataloader, config['topk_list'], config['beam_size'], device)

result = {
    'task': 'task162_paper_free_curv',
    'recipe': 'T5-mini 5.5M (d_model=128) + paper-free-curv SID (Stage 1 θ_init=[0,0,0], β=0.5, [32,64,256], sk=0)',
    't5_config': '4 enc + 4 dec, d_model=128, d_ff=512, 4 heads × d_kv=32 (4×32=128 strict)',
    'best_ckpt': os.environ['BEST_CKPT'],
    'beam_size': 20, 'test_recalls': avg_recalls, 'test_ndcgs': avg_ndcgs,
}
os.makedirs(os.path.dirname(os.environ['RESULT_JSON']), exist_ok=True)
with open(os.environ['RESULT_JSON'], 'w') as f:
    json.dump(result, f, indent=2)
print(f'[Stage 4] Saved: {os.environ["RESULT_JSON"]}', flush=True)
print(f'[Stage 4] Test recalls: {avg_recalls}', flush=True)
print(f'[Stage 4] Test NDCGs: {avg_ndcgs}', flush=True)
PYTHON_EOF

EXIT_CODE=${PIPESTATUS[0]}
echo "===== [Task #162 Stage 4] Test eval completed at $(date), exit code: $EXIT_CODE =====" | tee -a $LOG_FILE