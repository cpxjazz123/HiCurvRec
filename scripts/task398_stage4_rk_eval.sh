#!/bin/bash
# Task #398 / Stage 4 R@K eval post-task396 (Issue #104 闭环)
# v1: 基于 task174 stage4 v3 模式 + task84 baseline recipe
# Per HG-Rec baseline R@10=0.1020 (Musical_Instruments, Task #84) — GO/NO-GO 决策阈值

set -e

source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh
conda activate /home/wlia0047/ar57_scratch/wenyu/genrec_env

cd /home/wlia0047/ar57/wenyu/GeneRec/HG-Rec

export PYTHONPATH=/home/wlia0047/ar57/wenyu/GeneRec/HG-Rec:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1}
export HF_HOME=/home/wlia0047/ar57_scratch/wenyu/hf_models
export TRITON_CACHE_DIR=/home/wlia0047/.triton/cache_task398_s4
mkdir -p $TRITON_CACHE_DIR

LOG_DIR=/home/wlia0047/ar57/wenyu/GeneRec/logs
TS=$(date +%b-%d-%Y_%H-%M-%S | sed 's/^.../\L&/')
LOG_FILE=$LOG_DIR/task398_stage4_eval_${TS}.log
RESULT_JSON=/home/wlia0047/ar57/wenyu/GeneRec/verdicts/task398_stage4_rk_eval_metrics.json

mkdir -p $LOG_DIR

echo "===== [Task #398 Stage 4 v1] T5-mini 5.5M + #97 patch + 200 epoch eval launched at $(date) =====" | tee $LOG_FILE

BEST_CKPT=$(ls -t /home/wlia0047/ar57/wenyu/GeneRec/products/task396_issue99_stage3_t5_train/ckpt/Instruments/*/HG_Rec_best.pth 2>/dev/null | head -1)
if [ -z "$BEST_CKPT" ] || [ ! -f "$BEST_CKPT" ]; then
    echo "❌ Stage 3 best ckpt MISSING for task396" | tee -a $LOG_FILE
    exit 1
fi
echo "Stage 3 best ckpt: $BEST_CKPT" | tee -a $LOG_FILE

# Verify task396a SID file exists
SID_FILE=/home/wlia0047/ar57/wenyu/GeneRec/HG-Rec/dataset/Instruments/Instruments_t5_rqvae_task396.npy
if [ ! -f "$SID_FILE" ]; then
    echo "❌ $SID_FILE NOT FOUND" | tee -a $LOG_FILE
    exit 1
fi
echo "SID file: $SID_FILE" | tee -a $LOG_FILE

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

# Per task396b Stage 3 训练 config (HG-Rec baseline recipe):
# num_layers=6, num_decoder_layers=4, d_model=128, d_ff=1024, num_heads=6, d_kv=64
# vocab_size=1025, max_len=20, pad_token_id=0, eos_token_id=0
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
    'code_path': '_t5_rqvae_task396.npy',
    'topk_list': [5, 10, 20],
    'beam_size': 20,
}

device = torch.device('cuda:0')
model = HG_Rec(config)
print(f'[Task #398 Stage 4 v1] Loading best ckpt from {os.environ.get("BEST_CKPT_PATH", "(unset)")}...', flush=True)

ckpt_path = '$BEST_CKPT'
print(f'[Task #398 Stage 4 v1] ckpt_path = {ckpt_path}', flush=True)
state = torch.load(ckpt_path, map_location='cpu')
model.load_state_dict(state)
model.to(device)
print(f'[Task #398 Stage 4 v1] Loaded ckpt OK', flush=True)

test_dataset = GenRecDataset(
    dataset_path=os.path.join(config['dataset_path'], config['dataset_name'], 'test.parquet'),
    code_path=os.path.join(config['dataset_path'], config['dataset_name'], config['dataset_name'] + config['code_path']),
    mode='evaluation',
    codebook_size=config['codebook_size'],
    max_len=config['max_len']
)
test_dataloader = GenRecDataLoader(test_dataset, batch_size=config['infer_size'], shuffle=False)
print(f'[Task #398 Stage 4 v1] Test dataset size: {len(test_dataset)}', flush=True)

avg_recalls, avg_ndcgs = evaluate(model, test_dataloader, config['topk_list'], config['beam_size'], device)
print(f'[Task #398 Stage 4 v1] Test recalls: {avg_recalls}', flush=True)
print(f'[Task #398 Stage 4 v1] Test NDCGs: {avg_ndcgs}', flush=True)

# Recall@5/10/20, NDCG@5/10/20 — extract per topk_list (cast to float for json serialization)
recalls_dict = {f'Recall@{k}': float(v) for k, v in zip(config['topk_list'], avg_recalls)}
ndcgs_dict = {f'NDCG@{k}': float(v) for k, v in zip(config['topk_list'], avg_ndcgs)}

# HG-Rec baseline (Task #84) for GO/NO-GO comparison
hgrec_baseline = {
    'Recall@5': 0.0816,
    'Recall@10': 0.1020,
    'Recall@20': 0.1279,
    'NDCG@5': 0.0690,
    'NDCG@10': 0.0755,
    'NDCG@20': 0.0821,
}

# GO/NO-GO decision per Issue #104 spec
r10 = recalls_dict.get('Recall@10', 0.0)
go_nogo = 'GO' if r10 > 0.1020 else 'NO-GO'

result = {
    'task': 'task398_stage4_rk_eval',
    'recipe': 'T5-mini 5.5M + #97 patch poincare_distance fix + Stage 1+2+3 (#97+task395+task396) + 200 epoch',
    'best_ckpt': '$BEST_CKPT',
    'beam_size': 20,
    'topk_list': config['topk_list'],
    'recalls': recalls_dict,
    'ndcgs': ndcgs_dict,
    'hgrec_baseline': hgrec_baseline,
    'r10_delta_vs_baseline': r10 - 0.1020,
    'go_nogo_decision': go_nogo,
    'issue_104_target': 'R@10 > 0.1020 (HG-Rec baseline threshold)',
}
os.makedirs(os.path.dirname('$RESULT_JSON'), exist_ok=True)
with open('$RESULT_JSON', 'w') as f:
    json.dump(result, f, indent=2)
print(f'[Task #398 Stage 4 v1] Saved: $RESULT_JSON', flush=True)
print(f'[Task #398 Stage 4 v1] GO/NO-GO: {go_nogo} (R@10={r10:.4f} vs baseline 0.1020)', flush=True)
PYTHON_EOF

EXIT_CODE=${PIPESTATUS[0]}
echo "===== [Task #398 Stage 4 v1] exit code: $EXIT_CODE at $(date) =====" | tee -a $LOG_FILE
[ $EXIT_CODE -ne 0 ] && exit 1

echo "✅ Task #398 Stage 4 完成" | tee -a $LOG_FILE
cat $RESULT_JSON | tee -a $LOG_FILE