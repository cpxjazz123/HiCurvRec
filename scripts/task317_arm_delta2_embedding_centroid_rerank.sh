#!/usr/bin/env bash
# Task #317 — Issue #38 Arm δ2: Stage 4 embedding centroid similarity rerank on Issue #30 GO ckpt
# v2: Pre-compute centroids from test.parquet directly (avoid item_to_code complexity)
# GPU 0 (R7 空闲, 4 GPU 全 idle)

set -uo pipefail

REPO=/home/wlia0047/ar57/wenyu/GeneRec
cd $REPO

export PYTHONPATH=$REPO/HG-Rec:$REPO/scripts:${PYTHONPATH:-}
PYTHON_BIN=/home/wlia0047/ar57_scratch/wenyu/genrec_env/bin/python

CKPT_PATH=$REPO/products/task301/ckpt_hgrec_issue30/Instruments/Jul-30-2026_00-17-08/HG_Rec_best.pth
if [ ! -f "$CKPT_PATH" ]; then
    echo "ERROR: no ckpt found at $CKPT_PATH"
    exit 1
fi
echo "Using ckpt: $CKPT_PATH (size: $(stat -c %s $CKPT_PATH))"

LOG_DIR=$REPO/logs/task317
mkdir -p $LOG_DIR

TS=$(date +%Y%m%d_%H%M%S)
LOG="$LOG_DIR/stage4_emb_centroid_rerank_${TS}.log"
echo "===== [Task #317 / Issue #38 Arm δ2 embedding centroid rerank v2] launched at $TS =====" | tee "$LOG"
echo "code_path = _t5_hrqvae_issue30_per_layer_transforms.npy" | tee -a "$LOG"
echo "ckpt = $CKPT_PATH" | tee -a "$LOG"
echo "GPU 0 (R7 空闲)" | tee -a "$LOG"
echo "K=50 (Issue #30 ceiling 0.1041)" | tee -a "$LOG"
echo "alpha sweep: [0.0, 0.1, 0.3, 0.5, 1.0]" | tee -a "$LOG"

CUDA_VISIBLE_DEVICES=0 "$PYTHON_BIN" - <<PYEOF 2>&1 | tee -a "$LOG"
import sys, os, json, time
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, '$REPO/HG-Rec')
sys.path.insert(0, '$REPO/src')
sys.path.insert(0, '$REPO/scripts')

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

# Load item embeddings (Stage 1 T5-base output)
print('[TASK317] Loading item_emb.parquet ...', flush=True)
emb_df = pd.read_parquet('$REPO/HG-Rec/dataset/Instruments/item_emb.parquet')
emb_df = emb_df.sort_values('ItemID').reset_index(drop=True)
n_items = len(emb_df)
e_dim = len(emb_df.iloc[0]['embedding'])
print(f'[TASK317] n_items={n_items}, e_dim={e_dim}', flush=True)
item_emb = np.zeros((n_items, e_dim), dtype=np.float32)
for i in range(n_items):
    item_emb[i] = emb_df.iloc[i]['embedding'].astype(np.float32)
item_emb_t = torch.from_numpy(item_emb)
item_emb_norm = torch.nn.functional.normalize(item_emb_t, dim=-1)  # (9922, 768) unit-norm
print(f'[TASK317] item_emb_norm shape: {item_emb_norm.shape}', flush=True)

# Pre-compute per-sequence history centroid from test.parquet
print('[TASK317] Pre-computing per-sequence history centroids from test.parquet ...', flush=True)
test_raw = pd.read_parquet('$REPO/HG-Rec/dataset/Instruments/test.parquet')
n_test = len(test_raw)
print(f'[TASK317] n_test rows: {n_test}', flush=True)

# GenRecDataset uses sequences with target appended and history = sequence[:-1]
# For each row, history is a sequence of item IDs (variable length, may contain 0 for pad)
# Build centroid: per-row mean of item_emb[history] (excluding 0 pad)
centroids = np.zeros((n_test, e_dim), dtype=np.float32)
for i in range(n_test):
    hist = np.array(test_raw.iloc[i]['history'], dtype=np.int64)
    # Filter out 0 (PAD) and out-of-range
    valid_mask = (hist > 0) & (hist < n_items)
    valid_items = hist[valid_mask]
    if len(valid_items) > 0:
        centroids[i] = item_emb[valid_items].mean(axis=0)
    else:
        centroids[i] = 0.0
centroids_t = torch.from_numpy(centroids)
centroids_norm = torch.nn.functional.normalize(centroids_t, dim=-1)  # (n_test, 768) unit-norm
print(f'[TASK317] centroids_norm shape: {centroids_norm.shape}', flush=True)

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
    'beam_size': 50,
}

device = torch.device('cuda:0')
model = HG_Rec(config)
ckpt = torch.load(r'''${CKPT_PATH}''', map_location='cpu')
missing, unexpected = model.load_state_dict(ckpt, strict=False)
print(f'[TASK317] missing={len(missing)}, unexpected={len(unexpected)}', flush=True)
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

# Verify dataset size matches n_test
print(f'[TASK317] test_ds size: {len(test_ds)}', flush=True)
assert len(test_ds) == n_test, f"dataset size mismatch: {len(test_ds)} vs {n_test}"

test_dl = GenRecDataLoader(test_ds, batch_size=config['batch_size'], shuffle=False)

# Arm δ2 rerank hook (embedding centroid similarity, pre-computed centroids)
def centroid_rerank_eval(model, eval_loader, topk_list, beam_size, device, item_emb_norm, centroids_norm, alpha=0.5):
    """Evaluate with embedding centroid similarity rerank.

    centroids_norm: (N, 768) pre-computed per-row history centroids (unit-norm).
    For each batch:
      - beams → beam item embeddings (B, K, S) → beam_centroid (B, K, 768) → unit-norm
      - cosine_sim = centroid @ beam_centroid.T (B, K)  [higher = more relevant]
      - combined_score = beam_index * (1-alpha) - cosine_sim * alpha  [lower = better]
    """
    model.eval()
    recalls = {'Recall@' + str(k): [] for k in topk_list}
    ndcgs = {'NDCG@' + str(k): [] for k in topk_list}
    item_emb_norm_dev = item_emb_norm.to(device)
    centroids_norm_dev = centroids_norm.to(device)

    from tqdm import tqdm
    def set_color(s, c): return s
    sample_offset = 0
    with torch.no_grad():
        for batch in tqdm(eval_loader, ncols=100, desc=set_color(f"CentroidRerank alpha={alpha:.2f}", "cyan")):
            input_ids = batch['history'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['target'].to(device)
            batch_size = input_ids.shape[0]

            preds = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                num_beams=beam_size,
            )
            preds = preds[:, 1:]  # Exclude start token
            seqs = preds.view(batch_size, beam_size, -1)  # (B, K, S)
            seq_len = seqs.shape[-1]

            # Beam centroid: seqs (B, K, S) → item_emb_norm → (B, K, S, 768) → mean (B, K, 768)
            beam_emb = item_emb_norm_dev[seqs]  # (B, K, S, 768)
            beam_pad_mask = (seqs != 0).float().unsqueeze(-1)  # (B, K, S, 1)
            beam_emb = beam_emb * beam_pad_mask
            n_valid_beam = beam_pad_mask.sum(dim=2).clamp(min=1)  # (B, K, 1)
            beam_centroid = beam_emb.sum(dim=2) / n_valid_beam  # (B, K, 768)
            beam_centroid = torch.nn.functional.normalize(beam_centroid, dim=-1)  # (B, K, 768) unit-norm

            # Slice pre-computed centroids for this batch (eval order = sequential)
            centroids_batch = centroids_norm_dev[sample_offset:sample_offset + batch_size]  # (B, 768)
            sample_offset += batch_size

            # Cosine sim (B, K)
            cosine_sim = (centroids_batch.unsqueeze(1) * beam_centroid).sum(dim=-1)  # (B, K)

            # Combined score (lower is better)
            beam_idx = torch.arange(beam_size, device=device, dtype=torch.float32)
            beam_idx = beam_idx.unsqueeze(0).expand(batch_size, -1)
            combined_score = beam_idx * (1 - alpha) - cosine_sim * alpha
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
    avg_recalls, avg_ndcgs = centroid_rerank_eval(
        model, test_dl, config['topk_list'], config['beam_size'], device,
        item_emb_norm=item_emb_norm, centroids_norm=centroids_norm, alpha=alpha,
    )
    elapsed = time.time() - t0
    print(f'\n[TASK317 alpha={alpha}] elapsed={elapsed:.1f}s', flush=True)
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
    'task': 'task317_issue38_arm_delta2_embedding_centroid_rerank',
    'ckpt_path': '${CKPT_PATH}',
    'code_path': '_t5_hrqvae_issue30_per_layer_transforms.npy',
    'beam_size': 50,
    'config': config,
    'centroid_alpha_sweep': results,
}
metrics_dir = '$REPO/verdicts'
os.makedirs(metrics_dir, exist_ok=True)
out_path = os.path.join(metrics_dir, 'task317_arm_delta2_embedding_centroid_metrics.json')
with open(out_path, 'w') as f:
    json.dump(out, f, indent=2, default=float)
print(f'[TASK317] Saved metrics to {out_path}', flush=True)

# Final summary
print()
print('=' * 70)
print('TASK #317 Arm δ2 centroid rerank — alpha sweep summary')
print('=' * 70)
for ak, r in results.items():
    print(f"  {ak}: R@10={r['test_Recall@10']:.4f}, NDCG@10={r['test_NDCG@10']:.4f}")
print()
print('baseline (alpha=0):  R@10 = ', results['alpha_0.0']['test_Recall@10'])
print('Issue #30 K=50 ceiling: 0.1041')
print('Issue #30 K=100 ceiling: 0.1045')
print('task194_k0256 anchor: 0.1053')
PYEOF
