#!/usr/bin/env bash
# Task #310 — Stage 4 repetition_penalty ablation on #30 GO ckpt + beam=50
# 目的: 验证 repetition_penalty ∈ {1.0, 1.2, 1.5} 是否 R@10 杠杆
# K15 推断: repetition_penalty 跟 length_penalty 同类 (scoring 偏置) 应 NO-GO

set -uo pipefail

REPO=/home/wlia0047/ar57/wenyu/GeneRec
cd $REPO

export PYTHONPATH=$REPO/HG-Rec:$REPO/scripts:${PYTHONPATH:-}
PYTHON_BIN=/home/wlia0047/ar57_scratch/wenyu/genrec_env/bin/python

CKPT_PATH=$(ls -td $REPO/products/task301/ckpt_hgrec_issue30/Instruments/*/HG_Rec_best.pth 2>/dev/null | head -1)
if [ -z "$CKPT_PATH" ]; then
    echo "ERROR: no ckpt found"
    exit 1
fi
echo "Using ckpt: $CKPT_PATH"

LOG_DIR=$REPO/logs/task310
mkdir -p $LOG_DIR

run_one_rp() {
    local RP=$1
    local TS=$(date +%Y%m%d_%H%M%S)
    local LOG="$LOG_DIR/rp${RP}_${TS}.log"
    echo "===== [RP=$RP BEAM=50] launching at $TS =====" | tee "$LOG"

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

# Override evaluate to pass repetition_penalty to generate


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
    'beam_size': 50,
}

device = torch.device('cuda:0')
model = HG_Rec(config)
ckpt = torch.load(r'''${CKPT_PATH}''', map_location='cpu')
missing, unexpected = model.load_state_dict(ckpt, strict=False)
print(f'[RP=${RP}] missing={len(missing)}, unexpected={len(unexpected)}', flush=True)
model.to(device); model.eval()

# R11.5: Monkey-patch model.generate to inject repetition_penalty
_orig_generate = model.generate
def _patched_generate(input_ids, attention_mask=None, num_beams=20, **kwargs):
    return _orig_generate(input_ids=input_ids, attention_mask=attention_mask, num_beams=num_beams, repetition_penalty=${RP}, **kwargs)
model.generate = _patched_generate

test_ds = GenRecDataset(
    dataset_path=os.path.join(config['dataset_path'], config['dataset_name'], 'test.parquet'),
    code_path=os.path.join(config['dataset_path'], config['dataset_name'], config['dataset_name'] + config['code_path']),
    mode='evaluation',
    codebook_size=config['codebook_size'],
    max_len=config['max_len'],
)
test_dl = GenRecDataLoader(test_ds, batch_size=config['infer_size'], shuffle=False)
print(f'[RP=${RP}] Test dataset size: {len(test_ds)}', flush=True)

t0 = time.time()
avg_recalls, avg_ndcgs = evaluate(model, test_dl, config['topk_list'], config['beam_size'], device)
elapsed = time.time() - t0

result = {
    'task': 'task310_stage4_repetition_penalty_ablation',
    'repetition_penalty': ${RP},
    'beam_size': 50,
    'ckpt_path': r'''${CKPT_PATH}''',
    'note': '#30 GO ckpt + beam=50 + repetition_penalty ablation',
    'test_recalls': avg_recalls,
    'test_ndcgs': avg_ndcgs,
    'elapsed_sec': elapsed,
}
out_json = '$REPO/verdicts/task310_rp${RP}_metrics.json'
os.makedirs(os.path.dirname(out_json), exist_ok=True)
with open(out_json, 'w') as f:
    json.dump(result, f, indent=2)
print(f'[RP=${RP}] Saved {out_json}', flush=True)
print(f'[RP=${RP}] R@5={avg_recalls["Recall@5"]:.4f}, R@10={avg_recalls["Recall@10"]:.4f}, R@20={avg_recalls["Recall@20"]:.4f}', flush=True)
print(f'[RP=${RP}] NDCG@10={avg_ndcgs["NDCG@10"]:.4f}', flush=True)
print(f'[RP=${RP}] elapsed={elapsed:.1f}s', flush=True)
PYEOF
}

# Sequential runs (avoid GPU contention with task309 Stage 3 on GPU 1)
for RP in 1.0 1.2 1.5; do
    run_one_rp "$RP"
done

echo ""
echo "===== [SUMMARY] repetition_penalty ablation on #30 GO ckpt + beam=50 ====="
echo ""
echo "baseline (beam=50, rp=1.0 default): R@10=0.1045 (task307)"
echo ""
for RP in 1.0 1.2 1.5; do
    JSON="$REPO/verdicts/task310_rp${RP}_metrics.json"
    if [ -f "$JSON" ]; then
        echo "rp=$RP:"
        python3 -c "import json;d=json.load(open('$JSON'));print(f'  R@5={d[\"test_recalls\"][\"Recall@5\"]:.4f}, R@10={d[\"test_recalls\"][\"Recall@10\"]:.4f}, R@20={d[\"test_recalls\"][\"Recall@20\"]:.4f}, NDCG@10={d[\"test_ndcgs\"][\"NDCG@10\"]:.4f}, elapsed={d[\"elapsed_sec\"]:.1f}s')"
    fi
done
