#!/usr/bin/env bash
# Task #319 — Issue #38 Arm δ3: Stage 4 sequence-level diversity reward rerank (vectorized v2)
# GPU 1 (R7 空闲, 4 GPU 全 idle)

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

LOG_DIR=$REPO/logs/task319
mkdir -p $LOG_DIR

TS=$(date +%Y%m%d_%H%M%S)
LOG="$LOG_DIR/stage4_diversity_rerank_v2_${TS}.log"
echo "===== [Task #319 / Issue #38 Arm δ3 diversity rerank v2] launched at $TS =====" | tee "$LOG"
echo "code_path = _t5_hrqvae_issue30_per_layer_transforms.npy" | tee -a "$LOG"
echo "ckpt = $CKPT_PATH" | tee -a "$LOG"
echo "GPU 1 (R7 空闲)" | tee -a "$LOG"
echo "K=50 (Issue #30 ceiling 0.1041)" | tee -a "$LOG"
echo "alpha sweep: [0.0, 0.3, 0.5, 0.8, 1.0]" | tee -a "$LOG"

CUDA_VISIBLE_DEVICES=1 "$PYTHON_BIN" - <<PYEOF 2>&1 | tee -a "$LOG"
import sys, os, json, time
import numpy as np
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

device = torch.device('cuda:0')  # Inside CUDA_VISIBLE_DEVICES=1, cuda:0 == GPU 1
model = HG_Rec(config)
ckpt = torch.load(r'''${CKPT_PATH}''', map_location='cpu')
missing, unexpected = model.load_state_dict(ckpt, strict=False)
print(f'[TASK319] missing={len(missing)}, unexpected={len(unexpected)}', flush=True)
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

# Arm δ3 vectorized diversity rerank (v2)
def diversity_rerank_eval(model, eval_loader, topk_list, beam_size, device, alpha=0.5):
    """Evaluate with vectorized diversity reward rerank.

    Diversity = average over j != i of (1 - shared_count / max_count)
      where shared_count = |intersect of i and j (ignoring pad)|
      and max_count = max(|items_i|, |items_j|)

    Vectorized: for each batch,
      - seqs (B, K, S) → one-hot or hash per beam
      - For speed: use bag-of-ids up to vocab_size=1025
      - bag (B, K, V) where V=1025
      - intersect_count[i, j] = sum_v min(bag[i, v], bag[j, v])  (B, K, K)
      - count[i] = sum_v bag[i, v]  (B, K)
      - shared_fraction[i, j] = intersect_count[i, j] / max(count[i], count[j])
      - diversity_score[i] = mean over j != i of (1 - shared_fraction)
    """
    model.eval()
    recalls = {'Recall@' + str(k): [] for k in topk_list}
    ndcgs = {'NDCG@' + str(k): [] for k in topk_list}
    vocab_size = 1025  # covers all item IDs in vocab (1..9922 mapped to 1..9921, PAD=0)

    from tqdm import tqdm
    def set_color(s, c): return s
    with torch.no_grad():
        for batch in tqdm(eval_loader, ncols=100, desc=set_color(f"DiversityRerank v2 alpha={alpha:.2f}", "yellow")):
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

            # Build bag-of-ids (B*K, V) — vectorized via index_put_
            pad_mask = (seqs != 0)  # (B, K, S) bool
            bag = torch.zeros(batch_size * beam_size, vocab_size, device=device, dtype=torch.float32)
            flat_seqs = seqs.view(-1, seq_len).long()  # (B*K, S)
            flat_mask = pad_mask.view(-1, seq_len)  # (B*K, S) bool
            bk_idx = torch.arange(batch_size * beam_size, device=device).unsqueeze(-1).expand(-1, seq_len)  # (B*K, S)
            valid = flat_mask
            bag.index_put_(
                indices=(bk_idx[valid], flat_seqs[valid]),
                values=torch.ones(int(valid.sum().item()), device=device, dtype=torch.float32),
                accumulate=True,
            )
            bag = bag.view(batch_size, beam_size, vocab_size)  # (B, K, V)

            # Count per beam
            count = bag.sum(dim=-1)  # (B, K)
            count_safe = count.clamp(min=1)  # avoid div by 0

            # Intersect count: (B, K1, V) x (B, K2, V) → sum over V of min
            # For efficiency: only compute pairwise min, but (K, K) is 50x50=2500 per sample
            # Vectorized: bag.unsqueeze(2) * bag.unsqueeze(1) → product (B, K1, K2, V) — too big
            # Better: pairwise dot product via min trick
            # intersect[i, j] = sum_v min(bag[i, v], bag[j, v])
            # = sum_v (bag[i, v] + bag[j, v] - |bag[i, v] - bag[j, v]|) / 2
            # = (count[i] + count[j]) / 2 - sum_v |bag[i, v] - bag[j, v]| / 2
            # = (count[i] + count[j]) / 2 - L1_dist[i, j] / 2
            # L1 dist: |bag[i] - bag[j]|_1 = sum_v |bag[i, v] - bag[j, v]|
            # Compute L1 dist via broadcasting: bag.unsqueeze(1) - bag.unsqueeze(2) → (B, K, K, V)
            # That's 50*50*1025 = 2.6M elements per sample × 256 batch = 670M — too big
            # Alternative: use float16 chunked, or compute on CPU
            # R11.5: For speed, use simpler diversity = unique_items / total_items per beam
            # (no pairwise — single beam statistic, cheap)
            # This is "intra-beam diversity" not "inter-beam diversity" — but still a valid rerank signal

            # Intra-beam diversity: unique items per beam
            # unique_per_beam[b, k] = count[b, k] distinct items / total items in beam
            # = number of bins where bag[b, k, v] > 0 / total items
            unique_per_beam = (bag > 0).float().sum(dim=-1)  # (B, K)
            total_items = count  # (B, K)
            intra_diversity = unique_per_beam / total_items.clamp(min=1)  # (B, K), in [0.5, 1.0] for non-degenerate beams

            # Inter-beam diversity approximation: 1 - average pairwise shared_fraction (vectorized)
            # chunked over batch_size to save memory
            inter_diversity = torch.zeros(batch_size, beam_size, device=device)
            chunk_size = 32  # 32 × 50 × 50 × 1025 × 4 bytes = 32 MB
            for cs in range(0, batch_size, chunk_size):
                ce = min(cs + chunk_size, batch_size)
                bag_chunk = bag[cs:ce]  # (c, K, V)
                count_chunk = count[cs:ce]  # (c, K)
                # L1 distance: sum_v |bag[i, v] - bag[j, v]|
                diff = bag_chunk.unsqueeze(1) - bag_chunk.unsqueeze(2)  # (c, K, K, V)
                l1 = diff.abs().sum(dim=-1)  # (c, K, K)
                # intersect = (count[i] + count[j] - L1) / 2
                ci = count_chunk.unsqueeze(1).expand(-1, beam_size, beam_size)  # (c, K_i, K_j)
                cj = count_chunk.unsqueeze(2).expand(-1, beam_size, beam_size)  # (c, K_i, K_j)
                intersect = (ci + cj - l1) / 2  # (c, K, K)
                # shared_fraction = intersect / max(count[i], count[j])
                max_count = torch.maximum(ci, cj).clamp(min=1)  # (c, K, K)
                shared_frac = intersect / max_count  # (c, K, K)
                # diversity[i] = mean over j != i of (1 - shared_frac)
                eye_mask = 1 - torch.eye(beam_size, device=device).unsqueeze(0)  # (1, K, K)
                div = (1 - shared_frac) * eye_mask  # (c, K, K)
                inter_div = div.sum(dim=-1) / (beam_size - 1)  # (c, K)
                inter_diversity[cs:ce] = inter_div

            # Combined diversity: alpha * intra_diversity + (1 - alpha) * inter_diversity (lower better — diversity_score higher)
            # We want high diversity → lower combined score = beam_idx * (1-alpha) - diversity * alpha
            diversity_score = intra_diversity * 0.5 + inter_diversity * 0.5  # (B, K)

            beam_idx = torch.arange(beam_size, device=device, dtype=torch.float32)
            beam_idx = beam_idx.unsqueeze(0).expand(batch_size, -1)
            combined_score = beam_idx * (1 - alpha) - diversity_score * alpha
            rerank_order = combined_score.argsort(dim=-1)  # (B, K)

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
for alpha in [0.0, 0.3, 0.5, 0.8, 1.0]:
    t0 = time.time()
    avg_recalls, avg_ndcgs = diversity_rerank_eval(
        model, test_dl, config['topk_list'], config['beam_size'], device,
        alpha=alpha,
    )
    elapsed = time.time() - t0
    print(f'\n[TASK319 alpha={alpha}] elapsed={elapsed:.1f}s', flush=True)
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
    'task': 'task319_issue38_arm_delta3_sequence_diversity_rerank_v2',
    'ckpt_path': '${CKPT_PATH}',
    'code_path': '_t5_hrqvae_issue30_per_layer_transforms.npy',
    'beam_size': 50,
    'config': config,
    'diversity_alpha_sweep': results,
}
metrics_dir = '$REPO/verdicts'
os.makedirs(metrics_dir, exist_ok=True)
out_path = os.path.join(metrics_dir, 'task319_arm_delta3_sequence_diversity_metrics.json')
with open(out_path, 'w') as f:
    json.dump(out, f, indent=2, default=float)
print(f'[TASK319] Saved metrics to {out_path}', flush=True)

# Final summary
print()
print('=' * 70)
print('TASK #319 Arm δ3 diversity rerank v2 — alpha sweep summary')
print('=' * 70)
for ak, r in results.items():
    print(f"  {ak}: R@10={r['test_Recall@10']:.4f}, NDCG@10={r['test_NDCG@10']:.4f}")
print()
print('baseline (alpha=0):  R@10 = ', results['alpha_0.0']['test_Recall@10'])
print('Issue #30 K=50 ceiling: 0.1041')
print('Issue #30 K=100 ceiling: 0.1045')
print('task194_k0256 anchor: 0.1053')
PYEOF
