#!/usr/bin/env bash
# Task #482 — Issue #57 Stage 4 eval launcher (proper)
#
# Pattern: follow task307_stage4_beam_ablation.sh (inline Python heredoc)
# Issue #57 model is HG_Rec_Issue57 wrapper, not vanilla HG_Rec
# SID file used by #158/#159 training: _A2_t5_hrqvae_poincare.npy
#
# 用法:
#   bash scripts/task482_issue57_stage4_eval.sh
#
# 默认 cuda:0 (GPU 0/3 当前空闲)
#
set -uo pipefail

REPO=/home/wlia0047/ar57/wenyu/GeneRec
cd $REPO

export PYTHONPATH=$REPO/HG-Rec:$REPO/scripts:${PYTHONPATH:-}
PYTHON_BIN=/home/wlia0047/ar57_scratch/wenyu/genrec_env/bin/python
DEVICE=cuda:0

CKPT_158=$(ls -td $REPO/products/task158/ckpt/Instruments/*/HG_Rec_best.pth 2>/dev/null | head -1)
CKPT_159=$(ls -td $REPO/products/task159/ckpt/Instruments/*/HG_Rec_best.pth 2>/dev/null | head -1)

if [ -z "$CKPT_158" ] || [ -z "$CKPT_159" ]; then
    echo "ERROR: ckpt not found"
    echo "  #158: $CKPT_158"
    echo "  #159: $CKPT_159"
    exit 1
fi

echo "Using ckpts:"
echo "  #158 random init: $CKPT_158"
echo "  #159 hyp init (sphere): $CKPT_159"
echo "  device: $DEVICE"
echo ""

LOG_DIR=$REPO/logs/task482
mkdir -p $LOG_DIR

run_one_arm() {
    local ARM=$1        # "random" or "hyp"
    local CKPT=$2
    local SID_INIT=$3   # "random" or "hyperbolic"
    local TS=$(date +%Y%m%d_%H%M%S)
    local LOG="$LOG_DIR/${ARM}_${TS}.log"
    echo "===== [ARM=$ARM sid_init=$SID_INIT] launching at $TS =====" | tee "$LOG"

    CUDA_VISIBLE_DEVICES=0 "$PYTHON_BIN" - <<PYEOF 2>&1 | tee -a "$LOG"
import sys, os, json, time
sys.path.insert(0, '$REPO/HG-Rec')
sys.path.insert(0, '$REPO/scripts')

import torch
from model.HG_Rec_issue57 import HG_Rec_Issue57
from data.dataset import GenRecDataset
from data.dataloader import GenRecDataLoader

import importlib.util as _ilu
_s4_spec = _ilu.spec_from_file_location('task158_s3_train_fork', '$REPO/scripts/task158_issue57_stage3_train.py')
_s4_mod = _ilu.module_from_spec(_s4_spec)
_s4_spec.loader.exec_module(_s4_mod)
evaluate = _s4_mod.evaluate

config = {
    'batch_size': 256, 'infer_size': 96, 'lr': 5e-4, 'device': 'cuda:0',
    'num_layers': 6, 'num_decoder_layers': 4,
    'd_model': 128, 'd_ff': 1024, 'num_heads': 6, 'd_kv': 64,
    'dropout_rate': 0.1,
    'vocab_size': 1025, 'pad_token_id': 0, 'eos_token_id': 1024,
    'feed_forward_proj': 'relu', 'max_len': 20,
    'dataset_name': 'Instruments',
    'dataset_path': '$REPO/HG-Rec/dataset/',
    'codebook_size': [64, 128, 256, 1],
    'code_path': '_A2_t5_hrqvae_poincare.npy',
    'topk_list': [5, 10, 20],
    'beam_size': 20,
    'sid_embedding_init': '${SID_INIT}',
    'hyp_c': 0.74,
}

device = torch.device('cuda:0')
model = HG_Rec_Issue57(config)
ckpt = torch.load(r'''${CKPT}''', map_location='cpu')
missing, unexpected = model.load_state_dict(ckpt, strict=False)
print(f'[ARM=$ARM] missing={len(missing)}, unexpected={len(unexpected)}', flush=True)
if missing:
    print(f'  missing keys (first 5): {missing[:5]}', flush=True)
if unexpected:
    print(f'  unexpected keys (first 5): {unexpected[:5]}', flush=True)
model.to(device); model.eval()

test_ds = GenRecDataset(
    dataset_path=os.path.join(config['dataset_path'], config['dataset_name'], 'test.parquet'),
    code_path=os.path.join(config['dataset_path'], config['dataset_name'], config['dataset_name'] + config['code_path']),
    mode='evaluation',
    codebook_size=config['codebook_size'],
    max_len=config['max_len'],
)
test_dl = GenRecDataLoader(test_ds, batch_size=config['infer_size'], shuffle=False)
print(f'[ARM=$ARM] Test dataset size: {len(test_ds)}', flush=True)

t0 = time.time()
avg_recalls, avg_ndcgs = evaluate(model, test_dl, config['topk_list'], config['beam_size'], device)
elapsed = time.time() - t0

result = {
    'task': 'task482_issue57_stage4_eval',
    'arm': '$ARM',
    'sid_embedding_init': '$SID_INIT',
    'hyp_c': 0.74,
    'ckpt_path': r'''${CKPT}''',
    'note': 'Issue #57 Stage 4 eval: random vs hyp (sphere) init baseline',
    'test_recalls': avg_recalls,
    'test_ndcgs': avg_ndcgs,
    'elapsed_sec': elapsed,
}
out_json = '$REPO/verdicts/task482_${ARM}_stage4_metrics.json'
os.makedirs(os.path.dirname(out_json), exist_ok=True)
with open(out_json, 'w') as f:
    json.dump(result, f, indent=2)
print(f'[ARM=$ARM] Saved {out_json}', flush=True)
print(f'[ARM=$ARM] R@5={avg_recalls["Recall@5"]:.4f}, R@10={avg_recalls["Recall@10"]:.4f}, R@20={avg_recalls["Recall@20"]:.4f}', flush=True)
print(f'[ARM=$ARM] NDCG@5={avg_ndcgs["NDCG@5"]:.4f}, NDCG@10={avg_ndcgs["NDCG@10"]:.4f}, NDCG@20={avg_ndcgs["NDCG@20"]:.4f}', flush=True)
print(f'[ARM=$ARM] elapsed={elapsed:.1f}s', flush=True)
PYEOF
}

# Sequential runs (avoid GPU contention + same GPU)
run_one_arm "random" "$CKPT_158" "random"
run_one_arm "hyp"    "$CKPT_159" "hyperbolic"

echo ""
echo "===== [SUMMARY] Issue #57 Stage 4 eval ====="
echo ""
echo "baseline (HG-Rec Task #84): R@5=0.0816, R@10=0.1020, R@20=0.1279"
echo "Issue #30 GO (Task #301):    R@5=0.0820, R@10=0.1022, R@20=0.1234"
echo ""
for ARM in random hyp; do
    JSON="$REPO/verdicts/task482_${ARM}_stage4_metrics.json"
    if [ -f "$JSON" ]; then
        echo "arm=$ARM (sid_embedding_init):"
        python3 -c "import json;d=json.load(open('$JSON'));r=d['test_recalls'];n=d['test_ndcgs'];print(f'  R@5={r[\"Recall@5\"]:.4f}, R@10={r[\"Recall@10\"]:.4f}, R@20={r[\"Recall@20\"]:.4f}, NDCG@10={n[\"NDCG@10\"]:.4f}, elapsed={d[\"elapsed_sec\"]:.1f}s')"
    fi
done
