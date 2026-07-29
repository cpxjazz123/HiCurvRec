#!/usr/bin/env bash
# Task #316 — Issue #38 Arm δ1: Stage 4 frequency prior rerank on Issue #30 GO ckpt
# v2: K=50 (Issue #30 ceiling 0.1041), 不带 output_scores (省 memory)
# GPU 0 (R7 空闲, 4 GPU 全 idle)

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

LOG_DIR=$REPO/logs/task316
mkdir -p $LOG_DIR

TS=$(date +%Y%m%d_%H%M%S)
LOG="$LOG_DIR/stage4_freq_rerank_${TS}.log"
echo "===== [Task #316 / Issue #38 Arm δ1 freq prior rerank v2] launched at $TS =====" | tee "$LOG"
echo "code_path = _t5_hrqvae_issue30_per_layer_transforms.npy" | tee -a "$LOG"
echo "ckpt = $CKPT_PATH" | tee -a "$LOG"
echo "GPU 0 (R7 空闲)" | tee -a "$LOG"
echo "K=50 (Issue #30 ceiling 0.1041)" | tee -a "$LOG"
echo "alpha sweep: [0.0, 0.1, 0.3, 0.5, 1.0]" | tee -a "$LOG"

CUDA_VISIBLE_DEVICES=0 "$PYTHON_BIN" - <<PYEOF 2>&1 | tee -a "$LOG"
import sys, os, json, time
sys.path.insert(0, '$REPO/HG-Rec')
sys.path.insert(0, '$REPO/src')
sys.path.insert(0, '$REPO/scripts')

import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from model.HG_Rec import HG_Rec
from data.dataset import GenRecDataset
from data.dataloader import GenRecDataLoader

import importlib.util as _ilu
_s4_spec = _ilu.spec_from_file_location('task84_s3_train_fork', '$REPO/scripts/task84_hgrec_stage3_train.py')
_s4_mod = _ilu.module_from_spec(_s4_spec)
_s4_spec.loader.exec_module(_s4_mod)
calculate_pos_index = _s4_mod.calculate_pos_index
recall_at_k = _s4_mod.recall_at_k
ndcg_at_k = _s4_mod.ndcg_at_k

# Load freq prior
FREQ_PATH = '$REPO/HG-Rec/dataset/Instruments/Instruments_item_log_freq.npy'
log_freq = torch.from_numpy(np.load(FREQ_PATH)).float()  # (9923,)
print(f'[TASK316] log_freq shape: {log_freq.shape}, range: [{log_freq.min():.3f}, {log_freq.max():.3f}]')

# T5-mini Issue #30 config
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
    'beam_size': 50,  # Issue #30 K=50 ceiling
}

device = torch.device('cuda:0')
model = HG_Rec(config)
ckpt = torch.load(r'''${CKPT_PATH}''', map_location='cpu')
missing, unexpected = model.load_state_dict(ckpt, strict=False)
print(f'[TASK316] missing={len(missing)}, unexpected={len(unexpected)}', flush=True)
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

# Arm δ1 rerank hook (no output_scores, just post-hoc freq rerank)
def freq_rerank_eval(model, eval_loader, topk_list, beam_size, device, freq_prior, alpha=0.5):
    """Evaluate with frequency prior rerank.

    For each batch: generate (batch, K*seq_len), reshape to (batch, K, seq_len),
    compute freq_score per beam = sum log_freq over non-pad tokens,
    rerank beams: combined_score = beam_index * (1-alpha) - freq_score * alpha (lower better),
    then call calculate_pos_index on reranked seqs.
    """
    model.eval()
    recalls = {'Recall@' + str(k): [] for k in topk_list}
    ndcgs = {'NDCG@' + str(k): [] for k in topk_list}

    from tqdm import tqdm
    def set_color(s, c): return s
    with torch.no_grad():
        for batch in tqdm(eval_loader, ncols=100, desc=set_color(f"FreqRerank alpha={alpha:.2f}", "pink")):
            input_ids = batch['history'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['target'].to(device)

            preds = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                num_beams=beam_size,
            )
            preds = preds[:, 1:]  # Exclude start token
            batch_size = input_ids.shape[0]
            seqs = preds.view(batch_size, beam_size, -1)  # (B, K, S)
            seq_len = seqs.shape[-1]

            # Compute freq_score per beam
            pad_mask = (seqs == 0).float()
            freq_per_token = freq_prior.to(device)[seqs]  # (B, K, S)
            freq_per_token = freq_per_token * (1 - pad_mask)
            freq_score = freq_per_token.sum(dim=-1)  # (B, K)

            # Combined score (lower is better)
            beam_idx = torch.arange(beam_size, device=device, dtype=torch.float32)
            beam_idx = beam_idx.unsqueeze(0).expand(batch_size, -1)
            combined_score = beam_idx * (1 - alpha) - freq_score * alpha
            rerank_order = combined_score.argsort(dim=-1)  # (B, K)

            # Gather reranked sequences
            rerank_idx = rerank_order.unsqueeze(-1).expand(-1, -1, seq_len)
            reranked_seqs = torch.gather(seqs, 1, rerank_idx)  # (B, K, S)

            pos_index = calculate_pos_index(reranked_seqs, labels, maxk=beam_size)
            for k in topk_list:
                recall = recall_at_k(pos_index, k).mean().item()
                ndcg = ndcg_at_k(pos_index, k).mean().item()
                recalls['Recall@' + str(k)].append(recall)
                ndcgs['NDCG@' + str(k)].append(ndcg)

    avg_recalls = {k: sum(v) / len(v) for k, v in recalls.items()}
    avg_ndcgs = {k: sum(v) / len(v) for k, v in ndcgs.items()}
    return avg_recalls, avg_ndcgs


# Run with alpha sweep
results = {}
for alpha in [0.0, 0.1, 0.3, 0.5, 1.0]:
    t0 = time.time()
    avg_recalls, avg_ndcgs = freq_rerank_eval(
        model, test_dl, config['topk_list'], config['beam_size'], device,
        freq_prior=log_freq, alpha=alpha,
    )
    elapsed = time.time() - t0
    print(f'\n[TASK316 alpha={alpha}] elapsed={elapsed:.1f}s', flush=True)
    print(f'  Recall@5/10/20 = [{avg_recalls["Recall@5"]:.4f}, {avg_recalls["Recall@10"]:.4f}, {avg_recalls["Recall@20"]:.4f}]')
    print(f'  NDCG@5/10/20 = [{avg_ndcgs["NDCG@5"]:.4f}, {avg_ndcgs["NDCG@10"]:.4f}, {avg_ndcgs["NDCG@20"]:.4f}]')
    results[f'alpha_{alpha}'] = {
        'alpha': alpha,
        'elapsed_sec': elapsed,
        **{f'test_{k}': v for k, v in avg_recalls.items()},
        **{f'test_{k}': v for k, v in avg_ndcgs.items()},
    }

# Save results
out = {
    'task': 'task316_issue38_arm_delta1_freq_prior_rerank_v2',
    'ckpt_path': '${CKPT_PATH}',
    'code_path': '_t5_hrqvae_issue30_per_layer_transforms.npy',
    'beam_size': 50,
    'config': config,
    'freq_alpha_sweep': results,
}
metrics_dir = '$REPO/verdicts'
os.makedirs(metrics_dir, exist_ok=True)
out_path = os.path.join(metrics_dir, 'task316_arm_delta1_freq_rerank_metrics.json')
with open(out_path, 'w') as f:
    json.dump(out, f, indent=2, default=float)
print(f'\n[TASK316] metrics written to {out_path}')
PYEOF
echo "===== [Task #316 / Arm δ1 v2] completed at $(date) =====" | tee -a "$LOG"