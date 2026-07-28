#!/bin/bash
# Task #156 Stage 4 — HG-Rec Test evaluation (code-default recipe fork)
# 2026-07-24 (镜像 task84 stage4_eval, 但 codebook [32,64,256,1] + code_path _t5_rqvae_code_default.npy)
# 复用 scripts/task84_hgrec_stage3_train.py 的 evaluate() 函数 (跟 task84 同样做法)
# 输出: verdicts/task156_hgrec_code_default_metrics.json (Test Recall@5/10/20 + NDCG@5/10/20)

set -e

source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh
conda activate /home/wlia0047/ar57_scratch/wenyu/genrec_env

cd /home/wlia0047/ar57/wenyu/GeneRec/HG-Rec

export PYTHONPATH=/home/wlia0047/ar57/wenyu/GeneRec/HG-Rec:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=3
export HF_HOME=/home/wlia0047/ar57_scratch/wenyu/hf_models
export TRITON_CACHE_DIR=/home/wlia0047/.triton/cache_task156

LOG_DIR=/home/wlia0047/ar57/wenyu/GeneRec/logs/task156
TS=$(date +%b-%d-%Y_%H-%M-%S | sed 's/^.../\L&/')
LOG_FILE=$LOG_DIR/stage4_eval_${TS}.log
RESULT_JSON=/home/wlia0047/ar57/wenyu/GeneRec/verdicts/task156_hgrec_code_default_metrics.json

mkdir -p $LOG_DIR

echo "===== [Task #156 Stage 4] HG-Rec Test evaluation launched at $(date) =====" | tee $LOG_FILE
echo "Recipe: code-default (β=0.25, [32,64,256], sk=0.5, 1869 epoch)" | tee -a $LOG_FILE
echo "GPU 3" | tee -a $LOG_FILE
echo "Output JSON: $RESULT_JSON" | tee -a $LOG_FILE

# 验证 Stage 3 best ckpt 落盘
BEST_CKPT=$(ls -t /home/wlia0047/ar57/wenyu/GeneRec/products/task156/ckpt_hgrec/Instruments/*/HG_Rec_best.pth 2>/dev/null | head -1)
if [ -z "$BEST_CKPT" ] || [ ! -f "$BEST_CKPT" ]; then
    echo "❌ Stage 3 best ckpt MISSING" | tee -a $LOG_FILE
    exit 1
fi
echo "Stage 3 best ckpt: $BEST_CKPT" | tee -a $LOG_FILE

# Standalone Test evaluation (mirror Task #84 pattern: import evaluate from stage3 train fork)
python3 -c "
import sys, os, glob, torch, json
sys.path.insert(0, '/home/wlia0047/ar57/wenyu/GeneRec/HG-Rec')
sys.path.insert(0, '/home/wlia0047/ar57/wenyu/GeneRec/scripts')

from model.HG_Rec import HG_Rec
from data.dataset import GenRecDataset
from data.dataloader import GenRecDataLoader

# R11.3 FIX: 上游文件 task84_hgrec_stage3_train.py 包含 evaluate/recall_at_k/ndcg_at_k
import importlib.util as _ilu
_s4_spec = _ilu.spec_from_file_location(
    'task84_s3_train_fork',
    '/home/wlia0047/ar57/wenyu/GeneRec/scripts/task84_hgrec_stage3_train.py',
)
_s4_mod = _ilu.module_from_spec(_s4_spec)
_s4_spec.loader.exec_module(_s4_mod)
evaluate = _s4_mod.evaluate

# Config (mirror Task #84 except codebook_size + code_path)
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
    'codebook_size': [32, 64, 256, 1],   # *** Task #156 code-default ***
    'code_path': '_t5_rqvae_code_default.npy',   # *** Task #156 code-default ***
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
print(f'[Stage 4] Test dataset size: {len(test_dataset)}, codebook={config[\"codebook_size\"]}')

avg_recalls, avg_ndcgs = evaluate(model, test_dataloader, config['topk_list'], config['beam_size'], device)

result = {
    'task': 'task156_hgrec_code_default',
    'recipe': 'code-default (β=0.25, [32,64,256], sk=0.5, 1869 ep)',
    'best_ckpt': '$BEST_CKPT',
    'beam_size': 20,
    'test_recalls': avg_recalls,
    'test_ndcgs': avg_ndcgs,
}
os.makedirs(os.path.dirname('$RESULT_JSON'), exist_ok=True)
with open('$RESULT_JSON', 'w') as f:
    json.dump(result, f, indent=2)
print(f'[Stage 4] Saved: $RESULT_JSON')
print(f'[Stage 4] Test recalls: {avg_recalls}')
print(f'[Stage 4] Test NDCGs: {avg_ndcgs}')
" 2>&1 | tee -a $LOG_FILE

EXIT_CODE=${PIPESTATUS[0]}
echo "===== [Task #156 Stage 4] Test eval completed at $(date), exit code: $EXIT_CODE =====" | tee -a $LOG_FILE
