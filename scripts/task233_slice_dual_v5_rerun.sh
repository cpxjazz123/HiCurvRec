#!/bin/bash
# Task #233 Phase 2 — dual_v5 RERUN Head/Body/Tail slice (Task #209 protocol)
# Issue #8 procedure step 4: compare to A0 baseline (Head R@10=0.1794)

set -e
REPO=/home/wlia0047/ar57/wenyu/GeneRec
source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh
conda activate /home/wlia0047/ar57_scratch/wenyu/grid_toys
cd $REPO/HG-Rec

export PYTHONPATH=$REPO/HG-Rec:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=0
export HF_HOME=/home/wlia0047/ar57_scratch/wenyu/hf_models
export TRITON_CACHE_DIR=/home/wlia0047/.triton/cache_task233_slice
mkdir -p $TRITON_CACHE_DIR

mkdir -p $REPO/logs/task233 $REPO/products/task233

# === 找 dual_v5 RERUN best ckpt ===
DUAL_CKPTS=$(find $REPO/products/task233/t5mini_dual_v5_rerun/Instruments -name "HG_Rec_best.pth" 2>/dev/null | sort)
if [ -z "$DUAL_CKPTS" ]; then
    echo "❌ No dual_v5 RERUN HG_Rec_best.pth found"
    echo "   先跑 task233_stage3_dual_v5_rerun.sh + task233_stage4_dual_v5_rerun.sh"
    exit 1
fi
DUAL_CKPT=$(echo "$DUAL_CKPTS" | tail -1)
echo "✅ dual_v5 RERUN ckpt: $DUAL_CKPT"

# A0 baseline (task181) ckpt
A0_CKPT=/home/wlia0047/ar57/wenyu/GeneRec/products/task181/t5small_phase0.6/Instruments/Jul-25-2026_16-29-42/HG_Rec_best.pth
if [ ! -f "$A0_CKPT" ]; then
    echo "⚠️ A0 baseline ckpt not at expected path; trying glob"
    A0_CKPT=$(ls /home/wlia0047/ar57/wenyu/GeneRec/products/task181/t5small_phase0.6/Instruments/*/HG_Rec_best.pth 2>/dev/null | tail -1)
fi
echo "✅ A0 baseline ckpt: $A0_CKPT"

LOG_FILE=$REPO/logs/task233/slice_dual_v5_rerun_$(date +%b-%d-%Y_%H-%M-%S | sed 's/^.../\L&/').log

echo "===== [Task #233 Slice] Head/Body/Tail evaluation launched at $(date) =====" | tee $LOG_FILE

python3 -u <<PYTHON_EOF 2>&1 | tee -a "$LOG_FILE"
import sys, os, json, time
sys.path.insert(0, '/home/wlia0047/ar57/wenyu/GeneRec/HG-Rec')
sys.path.insert(0, '/home/wlia0047/ar57/wenyu/GeneRec/scripts')

import numpy as np
import pandas as pd
import torch
from collections import Counter

# === 切片划分 (跟 task209_phase3_slice_eval.py 一致) ===
print("=== Slice 划分: Head/Body/Tail ===", flush=True)
df = pd.read_parquet('/home/wlia0047/ar57/wenyu/GeneRec/HG-Rec/dataset/Instruments/train.parquet')
cnt = Counter()
for h in df['history']:
    for it in h:
        cnt[int(it)] += 1
for t in df['target']:
    cnt[int(t)] += 1

n = len(cnt)
sorted_items = sorted(cnt.items(), key=lambda x: -x[1])
head_cutoff = int(n * 0.2)
tail_cutoff = int(n * 0.8)
head_items = set([it for it, _ in sorted_items[:head_cutoff]])
tail_items = set([it for it, _ in sorted_items[tail_cutoff:]])
body_items = set([it for it, _ in sorted_items[head_cutoff:tail_cutoff]])
print(f"Total: {n}, Head: {len(head_items)}, Body: {len(body_items)}, Tail: {len(tail_items)}", flush=True)

slice_info = {
    'total': n,
    'head': sorted(head_items),
    'body': sorted(body_items),
    'tail': sorted(tail_items),
}
slice_path = '/home/wlia0047/ar57/wenyu/GeneRec/products/task233/phase3_slices.json'
with open(slice_path, 'w') as f:
    json.dump(slice_info, f)
print(f"✅ Slice saved: {slice_path}", flush=True)

# === 评估 ===
from model.HG_Rec import HG_Rec
from data.dataset import GenRecDataset
from data.dataloader import GenRecDataLoader
import importlib.util as _ilu
_s4_spec = _ilu.spec_from_file_location('s3_fork', '/home/wlia0047/ar57/wenyu/GeneRec/scripts/task84_hgrec_stage3_train.py')
_s3 = _ilu.module_from_spec(_s4_spec); _s4_spec.loader.exec_module(_s3)
evaluate = _s3.evaluate

ARMS = {
    'A0_baseline_#181': {
        'ckpt': '$A0_CKPT',
        'sid': '/home/wlia0047/ar57/wenyu/GeneRec/HG-Rec/dataset/Instruments/Instruments_t5_rqvae_paper_fix.npy',
        'codebook_size': [64, 128, 256, 1],
        'vocab_size': 1025,
    },
    'dual_v5_RERUN_#233': {
        'ckpt': '$DUAL_CKPT',
        'sid': '/home/wlia0047/ar57/wenyu/GeneRec/HG-Rec/dataset/Instruments/Instruments_t5_rqvae_dual_v5.npy',
        'codebook_size': [64, 128, 256, 1],
        'vocab_size': 1025,
    },
}

test = pd.read_parquet('/home/wlia0047/ar57/wenyu/GeneRec/HG-Rec/dataset/Instruments/test.parquet')

results = {}
for arm_name, arm in ARMS.items():
    if arm['ckpt'] is None or not os.path.exists(arm['ckpt']):
        print(f"⚠️ {arm_name}: ckpt MISSING, skip", flush=True)
        continue
    print(f"\n=== {arm_name} ===", flush=True)
    print(f"  ckpt: {arm['ckpt']}", flush=True)
    print(f"  sid:  {arm['sid']}", flush=True)
    config = {
        'codebook_size': arm['codebook_size'],
        'num_layers': 6,
        'num_decoder_layers': 4,
        'd_model': 128,
        'd_ff': 1024,
        'num_heads': 6,
        'd_kv': 64,
        'vocab_size': arm['vocab_size'],
        'pad_token_id': 0,
        'eos_token_id': 0,
        'dropout_rate': 0.0,
        'feed_forward_proj': 'relu',
    }
    model = HG_Rec(config).to('cuda:0')
    sd = torch.load(arm['ckpt'], map_location='cuda:0', weights_only=False)
    model.load_state_dict(sd)
    model.eval()
    print(f"  ✅ Loaded ckpt", flush=True)

    arm_results = {}
    for slice_name, slice_items in [('head', head_items), ('body', body_items), ('tail', tail_items)]:
        test_slice = test[test['target'].isin(slice_items)].reset_index(drop=True)
        if len(test_slice) == 0:
            arm_results[slice_name] = {'n_samples': 0}
            continue
        tmp_path = f'/home/wlia0047/ar57/wenyu/GeneRec/products/task233/phase3_test_{slice_name}_{arm_name}.parquet'
        test_slice.to_parquet(tmp_path)
        ds = GenRecDataset(
            dataset_path=tmp_path,
            code_path=arm['sid'],
            mode='evaluation',
            codebook_size=arm['codebook_size'],
            max_len=20,
        )
        dl = GenRecDataLoader(ds, batch_size=128, shuffle=False)
        recalls, ndcgs = evaluate(model, dl, [10], 20, 'cuda:0')
        n_samp = len(test_slice)
        arm_results[slice_name] = {
            'n_samples': int(n_samp),
            'R@10': float(recalls['Recall@10']),
            'N@10': float(ndcgs['NDCG@10']),
        }
        print(f"  {slice_name} (n={n_samp}): R@10={recalls['Recall@10']:.4f}  N@10={ndcgs['NDCG@10']:.4f}", flush=True)
        os.remove(tmp_path)

    # All (full test set)
    ds = GenRecDataset(
        dataset_path='/home/wlia0047/ar57/wenyu/GeneRec/HG-Rec/dataset/Instruments/test.parquet',
        code_path=arm['sid'],
        mode='evaluation',
        codebook_size=arm['codebook_size'],
        max_len=20,
    )
    dl = GenRecDataLoader(ds, batch_size=128, shuffle=False)
    recalls, ndcgs = evaluate(model, dl, [10], 20, 'cuda:0')
    arm_results['all'] = {
        'R@10': float(recalls['Recall@10']),
        'N@10': float(ndcgs['NDCG@10']),
    }
    print(f"  ALL: R@10={recalls['Recall@10']:.4f}  N@10={ndcgs['NDCG@10']:.4f}", flush=True)
    results[arm_name] = arm_results

slice_eval_path = '/home/wlia0047/ar57/wenyu/GeneRec/products/task233/phase3_slice_eval.json'
with open(slice_eval_path, 'w') as f:
    json.dump(results, f, indent=2)
print(f"\n✅ Slice eval saved: {slice_eval_path}", flush=True)
PYTHON_EOF

echo "===== [Task #233 Slice] DONE at $(date) =====" | tee -a $LOG_FILE
