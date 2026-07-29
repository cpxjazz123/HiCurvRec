#!/usr/bin/env bash
# Task #301 Stage 4 eval @ beam=150 — test if K=100 is ceiling
# K=50 R@10=0.1041, K=100 R@10=0.1045. Test K=150 to see if curve plateaus.
# GPU 0 (R7 空闲, 全部 4 GPU idle)

set -uo pipefail

REPO=/home/wlia0047/ar57/wenyu/GeneRec
cd $REPO

export PYTHONPATH=$REPO/HG-Rec:$REPO/scripts:${PYTHONPATH:-}
PYTHON_BIN=/home/wlia0047/ar57_scratch/wenyu/genrec_env/bin/python

CKPT_PATH=/home/wlia0047/ar57/wenyu/GeneRec/products/task301/ckpt_hgrec_issue30/Instruments/Jul-30-2026_00-17-08/HG_Rec_best.pth
if [ ! -f "$CKPT_PATH" ]; then
    echo "ERROR: no ckpt found at $CKPT_PATH"
    exit 1
fi
echo "Using ckpt: $CKPT_PATH (size: $(stat -c %s $CKPT_PATH))"

LOG_DIR=$REPO/logs/task301
mkdir -p $LOG_DIR

TS=$(date +%Y%m%d_%H%M%S)
LOG="$LOG_DIR/stage4_eval_beam120_${TS}.log"
echo "===== [Task #301 / Issue #30 ckpt @ beam=120 ceiling test] launched at $TS =====" | tee "$LOG"
echo "code_path = _t5_hrqvae_issue30_per_layer_transforms.npy" | tee -a "$LOG"
echo "ckpt = $CKPT_PATH" | tee -a "$LOG"
echo "GPU 0 (R7 空闲)" | tee -a "$LOG"

CUDA_VISIBLE_DEVICES=0 "$PYTHON_BIN" - <<PYEOF 2>&1 | tee -a "$LOG"
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

# T5-mini config
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
    'code_path': '_t5_hrqvae_issue30_per_layer_transforms.npy',
    'topk_list': [5, 10, 20],
    'beam_size': 120,  # ceiling test
}

device = torch.device('cuda:0')
model = HG_Rec(config)
ckpt = torch.load(r'''${CKPT_PATH}''', map_location='cpu')
missing, unexpected = model.load_state_dict(ckpt, strict=False)
print(f'[TASK301 Issue #30 beam=120] missing={len(missing)}, unexpected={len(unexpected)}', flush=True)
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
    'task': 'task301_issue30_ckpt_beam120_ceiling_test',
    'ckpt_path': '${CKPT_PATH}',
    'code_path': '_t5_hrqvae_issue30_per_layer_transforms.npy',
    'beam_size': 120,
    'config': config,
    'elapsed_sec': elapsed,
    **{f'test_{k}': v for k, v in avg_recalls.items()},
    **{f'test_{k}': v for k, v in avg_ndcgs.items()},
}
metrics_dir = '$REPO/verdicts'
os.makedirs(metrics_dir, exist_ok=True)
out_path = os.path.join(metrics_dir, 'task301_issue30_beam120_metrics.json')
with open(out_path, 'w') as f:
    json.dump(out, f, indent=2, default=float)
print(f'[TASK301 Issue #30 beam=120] metrics written to {out_path}')
print(f'[TASK301 Issue #30 beam=120] elapsed={elapsed:.1f}s')
print(f'[TASK301 Issue #30 beam=120] Recall@5/10/20 = {[avg_recalls["Recall@5"], avg_recalls["Recall@10"], avg_recalls["Recall@20"]]}')
print(f'[TASK301 Issue #30 beam=120] NDCG@5/10/20 = {[avg_ndcgs["NDCG@5"], avg_ndcgs["NDCG@10"], avg_ndcgs["NDCG@20"]]}')
PYEOF
echo "===== [Task #301 Issue #30 beam=120] completed at $(date) =====" | tee -a "$LOG"
