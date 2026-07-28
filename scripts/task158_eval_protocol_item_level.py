"""
Task #158 — Eval protocol test: item-level R@K vs strict 4-token match.

Background: User 2026-07-24 asked "why hg-rec still not match paper metrics".
Hypothesis: paper might use item-level R@K (decode beam SID -> item_id -> top-K)
instead of strict 4-token exact match. Task #84 (poincare codebook) has 0% collision,
so this test on Task #84 ckpt should yield identical numbers (null result).
Real protocol-difference test will require Task #156's collision-containing codebook.

This script:
1. Loads Task #84 ckpt + poincare codebook
2. Runs strict 4-token eval (existing) -> R@5/10/20
3. Runs item-level R@K eval (decode SID -> item_id) -> R@5/10/20
4. Compares and writes verdict

Output: verdicts/task158_eval_protocol_item_level_metrics.json
"""
import sys, os, json, collections
import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, '/home/wlia0047/ar57/wenyu/GeneRec/HG-Rec')
sys.path.insert(0, '/home/wlia0047/ar57/wenyu/GeneRec/scripts')

from data.dataset import GenRecDataset
from data.dataloader import GenRecDataLoader
from model.HG_Rec import HG_Rec

# ==== Config: mirror Task #84 ====
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
    'codebook_size': [64, 128, 256, 1],   # Task #84 poincare
    'code_path': '_t5_hrqvae_poincare.npy',
    'topk_list': [5, 10, 20],
    'beam_size': 20,
}

# ==== Load ckpt ====
CKPT = '/home/wlia0047/ar57/wenyu/GeneRec/products/task84/ckpt_hgrec/Instruments/Jul-23-2026_20-24-44/HG_Rec_best.pth'
if not os.path.exists(CKPT):
    raise FileNotFoundError(f"Task #84 ckpt missing: {CKPT}")

# ==== Load codebook + build reverse map (codebook[4 tokens] -> item_id) ====
# Task #84 codebook format: raw 4-token indices from RQ-VAE (0..codebook_size[m]-1)
# But model.generate() emits "offsets" = [c + sum(codebook_size[0:i]) + 1, ...] (per item2code in dataset.py)
# So the model's output tokens are offsets, NOT raw indices.
# To decode predicted tokens back to item_id, we need to:
#   1. Convert offsets back to raw indices: offset[i] = c + sum(codebook_size[0:i]) + 1
#      so raw[c] = offset - sum(codebook_size[0:i]) - 1
#   2. Look up (raw[0], raw[1], raw[2], raw[3]) in codebook -> item_id

CODE_FILE = os.path.join(config['dataset_path'], config['dataset_name'],
                        config['dataset_name'] + config['code_path'])
codebook_raw = np.load(CODE_FILE)  # (N_items, 4) raw indices
print(f'[Task #158] Codebook: {codebook_raw.shape} dtype={codebook_raw.dtype}')

# Build raw-tuple -> list[item_id] reverse map
raw_to_items = collections.defaultdict(list)
for item_id, raw in enumerate(codebook_raw):
    raw_to_items[tuple(raw.tolist())].append(item_id + 1)  # dataset uses 1-based item_id
print(f'[Task #158] Codebook reverse map: {len(raw_to_items)} unique raw SIDs for {len(codebook_raw)} items')
n_collisions = sum(1 for v in raw_to_items.values() if len(v) > 1)
print(f'[Task #158] Collisions: {n_collisions} raw SIDs shared by multiple items (collision_rate={1 - len(raw_to_items)/len(codebook_raw):.4f})')

# Pre-compute offsets->raw conversion
cs = config['codebook_size'][:4]  # [64, 128, 256, 1]
# Only first 3 matter (4th is dedup digit, not used for matching; Task #84 codebook has 4 cols)
sums = [0]
for i in range(3):
    sums.append(sums[-1] + cs[i])

def offsets_to_raw(offsets_4):
    """Convert model's output 4 tokens (offsets) back to raw RQ-VAE indices."""
    out = []
    for i, off in enumerate(offsets_4):
        if i < 3:
            out.append(int(off) - sums[i] - 1)
        else:
            out.append(int(off))  # 4th column (dedup digit), kept as-is
    return tuple(out)

# ==== Build model + load ckpt ====
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f'[Task #158] Loading model on {device}...')
model = HG_Rec(config)
model.load_state_dict(torch.load(CKPT, map_location='cpu'))
model.to(device).eval()

# ==== Load test dataset ====
test_dataset = GenRecDataset(
    dataset_path=os.path.join(config['dataset_path'], config['dataset_name'], 'test.parquet'),
    code_path=os.path.join(config['dataset_path'], config['dataset_name'],
                           config['dataset_name'] + config['code_path']),
    mode='evaluation',
    codebook_size=config['codebook_size'],
    max_len=config['max_len'],
)
test_loader = GenRecDataLoader(test_dataset, batch_size=config['infer_size'], shuffle=False)
print(f'[Task #158] Test dataset: {len(test_dataset)} samples')

# ==== Eval: BOTH strict 4-token and item-level R@K ====
topk_list = config['topk_list']
beam_size = config['beam_size']

# Accumulators
strict_recalls = {'R@' + str(k): [] for k in topk_list}
strict_ndcgs = {'N@' + str(k): [] for k in topk_list}
item_recalls = {'R@' + str(k): [] for k in topk_list}
item_ndcgs = {'N@' + str(k): [] for k in topk_list}

n_batches = 0
import time
t0 = time.time()
with torch.no_grad():
    for batch_idx, batch in enumerate(test_loader):
        input_ids = batch['history'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['target'].to(device)  # (batch, 4) target SID offsets

        preds = model.generate(input_ids=input_ids, attention_mask=attention_mask, num_beams=beam_size)
        preds = preds[:, 1:]  # exclude start token
        preds = preds.reshape(input_ids.shape[0], beam_size, -1)  # (batch, beam, 4)

        # ---- Strict 4-token match (existing) ----
        # pos_index[i, b] = True iff beam b of sample i has same 4 tokens as label
        strict_pos = torch.zeros((preds.shape[0], beam_size), dtype=torch.bool)
        for i in range(preds.shape[0]):
            cur_label = labels[i].tolist()
            for b in range(beam_size):
                if preds[i, b].tolist() == cur_label:
                    strict_pos[i, b] = True
                    break  # only first matching beam counts

        # ---- Item-level R@K ----
        # For each sample:
        #   true_item_ids = set of item_ids whose raw SID matches label
        #   For each beam, decode predicted offsets -> raw SID -> item_ids
        #   pos_index[i, b] = True iff any item_id from beam b's SID is in true_item_ids
        item_pos = torch.zeros((preds.shape[0], beam_size), dtype=torch.bool)
        for i in range(preds.shape[0]):
            cur_label_offsets = labels[i].tolist()
            cur_label_raw = offsets_to_raw(cur_label_offsets)
            true_item_ids = set(raw_to_items.get(cur_label_raw, []))
            for b in range(beam_size):
                cur_pred_offsets = preds[i, b].tolist()
                cur_pred_raw = offsets_to_raw(cur_pred_offsets)
                pred_item_ids = set(raw_to_items.get(cur_pred_raw, []))
                if pred_item_ids & true_item_ids:
                    item_pos[i, b] = True
                    break  # first match

        # Aggregate
        for k in topk_list:
            strict_recalls['R@' + str(k)].append(strict_pos[:, :k].any(dim=1).float().mean().item())
            strict_ndcgs['N@' + str(k)].append((strict_pos[:, :k].float() / torch.log2(torch.arange(k, 0, -1).float().to(device) + 1)).sum(dim=1).mean().item())
            item_recalls['R@' + str(k)].append(item_pos[:, :k].any(dim=1).float().mean().item())
            item_ndcgs['N@' + str(k)].append((item_pos[:, :k].float() / torch.log2(torch.arange(k, 0, -1).float().to(device) + 1)).sum(dim=1).mean().item())

        n_batches += 1
        if batch_idx % 20 == 0:
            elapsed = time.time() - t0
            print(f'  batch {batch_idx}/{len(test_loader)} ({elapsed:.0f}s)')

elapsed = time.time() - t0
print(f'\n[Task #158] Eval done: {n_batches} batches in {elapsed:.1f}s')

# ==== Average ====
def avg(d): return {k: sum(v)/len(v) for k, v in d.items()}
strict_recalls_avg = avg(strict_recalls)
strict_ndcgs_avg = avg(strict_ndcgs)
item_recalls_avg = avg(item_recalls)
item_ndcgs_avg = avg(item_ndcgs)

# ==== Delta ====
def delta(d_item, d_strict):
    return {k: d_item[k] - d_strict[k] for k in d_item}
r_delta = delta(item_recalls_avg, strict_recalls_avg)
n_delta = delta(item_ndcgs_avg, strict_ndcgs_avg)

print('\n=== Strict 4-token (existing) ===')
for k in strict_recalls_avg: print(f'  {k}: {strict_recalls_avg[k]:.4f}')
for k in strict_ndcgs_avg: print(f'  {k}: {strict_ndcgs_avg[k]:.4f}')

print('\n=== Item-level R@K (proposed) ===')
for k in item_recalls_avg: print(f'  {k}: {item_recalls_avg[k]:.4f}')
for k in item_ndcgs_avg: print(f'  {k}: {item_ndcgs_avg[k]:.4f}')

print('\n=== Delta (item - strict) ===')
for k in r_delta: print(f'  {k}: {r_delta[k]:+.4f}')
for k in n_delta: print(f'  {k}: {n_delta[k]:+.4f}')

# ==== Save ====
result = {
    'task': 'task158_eval_protocol_item_level',
    'recipe': 'Task #84 ckpt + poincare codebook (0% collision), 2 evals on same test set',
    'test_samples': len(test_dataset),
    'beam_size': beam_size,
    'strict_protocol': {
        'recalls': strict_recalls_avg,
        'ndcgs': strict_ndcgs_avg,
        'note': 'Existing logic: 4-token exact match, first matching beam counts',
    },
    'item_level_protocol': {
        'recalls': item_recalls_avg,
        'ndcgs': item_ndcgs_avg,
        'note': 'Decode beam SID offsets -> raw indices -> item_id, first item-matching beam counts',
    },
    'delta': {
        'recalls': r_delta,
        'ndcgs': n_delta,
    },
    'codebook_collision_rate': 1 - len(raw_to_items) / len(codebook_raw),
    'interpretation': (
        'NULL RESULT expected because Task #84 poincare codebook has 0% collision. '
        'Strict 4-token match == item-level R@K when no collisions. '
        'Real protocol difference will only matter for Task #156 (5.5% collision).'
    ),
}
out_path = '/home/wlia0047/ar57/wenyu/GeneRec/verdicts/task158_eval_protocol_item_level_metrics.json'
os.makedirs(os.path.dirname(out_path), exist_ok=True)
with open(out_path, 'w') as f:
    json.dump(result, f, indent=2)
print(f'\n[Task #158] Saved: {out_path}')
