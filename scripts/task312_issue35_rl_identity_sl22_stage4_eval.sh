#!/usr/bin/env bash
# Task #312 — Stage 4 eval on r_l identity + s_l=[2,2,2] ckpt + beam=50
# 目的: 验证 s_l 是不是真杠杆 (vs Issue #30 GO)
# K14 最优: beam_size = 50 (Stage 4 inference protocol 层唯一杠杆)
# R11.5: 等 Stage 3 完成后启动 (Stage 3 ETA ~85min, started 05:09)

set -uo pipefail

REPO=/home/wlia0047/ar57/wenyu/GeneRec
cd $REPO

export PYTHONPATH=$REPO/HG-Rec:$REPO/scripts:${PYTHONPATH:-}
PYTHON_BIN=/home/wlia0047/ar57_scratch/wenyu/genrec_env/bin/python

CKPT_PATH=$(ls -td $REPO/products/task312/ckpt_hgrec_issue35/Instruments/*/HG_Rec_best.pth 2>/dev/null | head -1)
if [ -z "$CKPT_PATH" ]; then
    echo "ERROR: no ckpt found at $REPO/products/task312/ckpt_hgrec_issue35/Instruments/*/HG_Rec_best.pth"
    echo "Please ensure Stage 3 training has completed and saved HG_Rec_best.pth"
    exit 1
fi
echo "Using ckpt: $CKPT_PATH"

LOG_DIR=$REPO/logs/task312
mkdir -p $LOG_DIR

TS=$(date +%Y%m%d_%H%M%S)
LOG="$LOG_DIR/stage4_eval_${TS}.log"
echo "===== [Task #312 / Issue #35 Gate 4 Stage 4 eval] beam=50 launched at $TS =====" | tee "$LOG"
echo "code_path = _t5_hrqvae_issue35_rl_identity_sl22.npy" | tee -a "$LOG"
echo "ckpt = $CKPT_PATH" | tee -a "$LOG"
echo "GPU 2 (R7 空闲, task309 GPU 1 / task313 GPU 3 / task312 Stage 3 GPU 0)" | tee -a "$LOG"

CUDA_VISIBLE_DEVICES=2 "$PYTHON_BIN" - <<PYEOF 2>&1 | tee -a "$LOG"
import sys, os, json, time
sys.path.insert(0, '$REPO/HG-Rec')
sys.path.insert(0, '$REPO/src')
sys.path.insert(0, '$REPO/scripts')

import torch
from model.HG_Rec import HG_Rec
from data.dataset import GenRecDataset
from data.dataloader import GenRecDataLoader

import importlib.util as _ilu
_s4_spec = _ilu.spec_from_file_location('task84_s3_train_fork', '$REPO/scripts/task84_hgrec_stage3_train.py')
_s4_mod = _ilu.module_from_spec(_s4_spec)
_s4_spec.loader.exec_module(_s4_mod)
evaluate = _s4_mod.evaluate

config = {
    'batch_size': 256, 'infer_size': 96, 'lr': 1e-4, 'device': 'cuda:0',
    'num_layers': 6, 'num_decoder_layers': 4,
    'd_model': 128, 'd_ff': 1024, 'num_heads': 6, 'd_kv': 64,
    'dropout_rate': 0.1,
    'vocab_size': 1025, 'pad_token_id': 0, 'eos_token_id': 0,
    'feed_forward_proj': 'relu', 'max_len': 20,
    'dataset_name': 'Instruments',
    'dataset_path': '$REPO/HG-Rec/dataset/',
    'codebook_size': [64, 128, 256, 1],
    'code_path': '_t5_hrqvae_issue35_rl_identity_sl22.npy',
    'topk_list': [5, 10, 20],
    'beam_size': 50,
}

device = torch.device('cuda:0')
model = HG_Rec(config)
ckpt = torch.load(r'''${CKPT_PATH}''', map_location='cpu')
missing, unexpected = model.load_state_dict(ckpt, strict=False)
print(f'[TASK312] missing={len(missing)}, unexpected={len(unexpected)}', flush=True)
model.to(device); model.eval()

test_ds = GenRecDataset(
    dataset_path=os.path.join(config['dataset_path'], config['dataset_name'], 'test.parquet'),
    code_path=os.path.join(config['dataset_path'], config['dataset_name'], config['dataset_name'] + config['code_path']),
    mode='evaluation',
    codebook_size=config['codebook_size'],
    max_len=config['max_len'],
)

from transformers import T5Config
tokenizer = T5Config(
    vocab_size=config['vocab_size'],
    pad_token_id=config['pad_token_id'],
    eos_token_id=config['eos_token_id'],
    feed_forward_proj=config['feed_forward_proj'],
    d_model=config['d_model'],
    d_ff=config['d_ff'],
    d_kv=config['d_kv'],
    num_heads=config['num_heads'],
    num_layers=config['num_layers'],
    num_decoder_layers=config['num_decoder_layers'],
    dropout_rate=config['dropout_rate'],
)
test_ds.tokenizer = tokenizer

test_dl = GenRecDataLoader(test_ds, batch_size=config['batch_size'], shuffle=False)

t0 = time.time()
avg_recalls, avg_ndcgs = evaluate(model, test_dl, config['topk_list'], config['beam_size'], device)
elapsed = time.time() - t0
out = {
    'task': 'task312_issue35_rl_identity_sl22_stage4_eval',
    'ckpt_path': '${CKPT_PATH}',
    'code_path': '_t5_hrqvae_issue35_rl_identity_sl22.npy',
    'beam_size': 50,
    'config': config,
    'elapsed_sec': elapsed,
    **{f'test_{k}': v for k, v in avg_recalls.items()},
    **{f'test_{k}': v for k, v in avg_ndcgs.items()},
}
metrics_dir = '$REPO/verdicts'
os.makedirs(metrics_dir, exist_ok=True)
out_path = os.path.join(metrics_dir, 'task312_rl_identity_sl22_metrics.json')
with open(out_path, 'w') as f:
    json.dump(out, f, indent=2, default=float)
print(f'[TASK312] metrics written to {out_path}')
print(f'[TASK312] elapsed={elapsed:.1f}s')
print(f'[TASK312] Recall@5/10/20 = {[avg_recalls["Recall@5"], avg_recalls["Recall@10"], avg_recalls["Recall@20"]]}')
print(f'[TASK312] NDCG@5/10/20 = {[avg_ndcgs["NDCG@5"], avg_ndcgs["NDCG@10"], avg_ndcgs["NDCG@20"]]}')
PYEOF
echo "===== [Task #312 Stage 4 eval] completed at $(date) =====" | tee -a "$LOG"