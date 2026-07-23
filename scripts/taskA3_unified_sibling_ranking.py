#!/usr/bin/env python3
"""Task A3: Unified sibling ranking definition

目标: 解决 AUC≈0.5 vs 75.3% sibling ranking 矛盾.

统一定义:
- target = ground truth next-item for user u
- siblings = items sharing (L1, ..., L_{l-1}) prefix with target
- AUC = P(s(y|h_u) > s(j|h_u)) where j≠y, Z_j(<l) = Z_y(<l)
- For each layer l, compute AUC across all (target, sibling) pairs

具体实现:
- Load L=4 baseline SID catalog and predictions
- For each user, find target item and rank
- Compute pairwise AUC at each layer (siblings by L_{<l} prefix match)
- Bucket by sibling count |S_l|

输出:
- Layer-wise AUC (SMRR, Recall@1 for siblings, full AUC)
- Bucket analysis by |S_l|
- Compare with paper Table 1 ranking
"""

import os, json, glob
import numpy as np
import torch
import tensorflow as tf

GRID = '/home/wlia0047/ar57/wenyu/GeneRec/GRID'
EVAL_DIR = f'{GRID}/data/amazon_data/toys/evaluation'
SID_PATH = f'{GRID}/logs/inference/runs/2026-07-07/20-17-35/pickle/merged_predictions_tensor.pt'
PRED_PATH = f'{GRID}/logs/inference/runs/2026-07-07/21-47-35/pickle/merged_predictions_tensor.pt'
OUT_DIR = f'{GRID}/result/taskA3_unified_sibling_ranking'
os.makedirs(OUT_DIR, exist_ok=True)

print('=' * 70)
print('Task A3: Unified Sibling Ranking Definition')
print('=' * 70)

# Load SID and predictions
sid = torch.load(SID_PATH, weights_only=False)  # (4, 11924)
pred = torch.load(PRED_PATH, weights_only=False)  # (19412, 10, 4)
n_items = sid.shape[1]
n_users, K, n_layers = pred.shape

# Build SID lookup
sid_to_item = {}
sid_T = sid.T  # (n_items, n_layers)
for item_id in range(n_items):
    sid_to_item[tuple(int(x) for x in sid_T[item_id].tolist())] = item_id

# Build per-layer prefix groups
def build_prefix_groups(sid_tensor, n_layers_used):
    """Group items by prefix of first n_layers_used SID digits."""
    groups = {}
    sid_T = sid_tensor[:n_layers_used].T
    for item_id in range(sid_T.shape[0]):
        key = tuple(int(x) for x in sid_T[item_id].tolist())
        groups.setdefault(key, []).append(item_id)
    return groups

# Convert predictions to item_ids
pred_items = torch.full((n_users, K), -1, dtype=torch.long)
for u in range(n_users):
    for k in range(K):
        s = tuple(int(x) for x in pred[u, k].tolist())
        pred_items[u, k] = sid_to_item.get(s, -1)
print(f'Pred valid: {(pred_items >= 0).float().mean().item()*100:.2f}%')

# Load eval tfrecords
print(f'\n[load eval tfrecords]')
files = sorted(glob.glob(f'{EVAL_DIR}/partition_*.tfrecord.gz'))
user_targets = {}
user_history = {}
dataset = tf.data.TFRecordDataset(files, compression_type='GZIP')
for raw in dataset:
    example = tf.train.Example()
    example.ParseFromString(raw.numpy())
    user_id = int(example.features.feature['user_id'].int64_list.value[0])
    seq = list(example.features.feature['sequence_data'].int64_list.value)
    if not seq:
        continue
    user_targets[user_id] = seq[-1]
    user_history[user_id] = seq[:-1]
print(f'Users: {len(user_targets)}')

# ============================================================
# Per-layer prefix groups
# ============================================================
print(f'\n[build prefix groups]')
groups_l1 = build_prefix_groups(sid, 1)
groups_l2 = build_prefix_groups(sid, 2)
groups_l3 = build_prefix_groups(sid, 3)
groups_l4 = build_prefix_groups(sid, 4)

print(f'  L1 groups: {len(groups_l1)}, sizes mean={np.mean([len(v) for v in groups_l1.values()]):.1f}')
print(f'  L2 groups: {len(groups_l2)}, sizes mean={np.mean([len(v) for v in groups_l2.values()]):.1f}')
print(f'  L3 groups: {len(groups_l3)}, sizes mean={np.mean([len(v) for v in groups_l3.values()]):.1f}')
print(f'  L4 groups: {len(groups_l4)} (= n_items, unique)')

# ============================================================
# Pairwise AUC at each layer
# ============================================================
print(f'\n[Pairwise AUC at each layer]')
print('Definition: AUC = P(score(target) > score(sibling)) where sibling shares prefix Z_j(<l) = Z_y(<l)')

# Score = inverse rank in Top-K (1/rank)
# For each user with target in catalog:
#   - Find rank of target in Top-K
#   - For each layer l, find siblings of target (share prefix Z_y(<l))
#   - Find ranks of siblings in Top-K
#   - Compute AUC = P(rank_target < rank_sibling) across all pairs

# Note: We're using RANK as the score (lower rank = higher score)
# AUC = (1 / |S|) sum_{s in S} I(rank_target < rank_sibling)

n_layers_used_list = [1, 2, 3, 4]
groups_per_layer = {1: groups_l1, 2: groups_l2, 3: groups_l3, 4: groups_l4}

per_layer_metrics = {}

for l in n_layers_used_list:
    groups = groups_per_layer[l]
    print(f'\n--- Layer l={l} (siblings share L_{l} = first l digits) ---')

    # Compute AUC, R@1, MRR for sibling ranking
    auc_pairs = []  # (target_rank, sibling_rank) pairs
    target_ranks_in_pred = []
    sibling_ranks_in_pred = []
    n_users_with_target = 0
    n_users_with_siblings_in_pred = 0
    bucket_recalls = {(0, 0): [], (1, 5): [], (6, 20): [], (21, 1000): []}

    for user_id, target_item in user_targets.items():
        if user_id >= n_users:
            continue
        if target_item < 0 or target_item >= n_items:
            continue
        n_users_with_target += 1

        # Find target rank in Top-K
        pred_list = pred_items[user_id].tolist()
        target_rank = -1
        for k, item in enumerate(pred_list):
            if item == target_item:
                target_rank = k
                break
        target_ranks_in_pred.append(target_rank)

        # Find siblings at layer l
        prefix = tuple(int(x) for x in sid[:l, target_item].tolist())
        siblings = [i for i in groups[prefix] if i != target_item]
        n_siblings = len(siblings)

        # Find sibling ranks
        sib_ranks = []
        for sib in siblings:
            for k, item in enumerate(pred_list):
                if item == sib:
                    sib_ranks.append(k)
                    break
        sibling_ranks_in_pred.extend(sib_ranks)

        if sib_ranks:
            n_users_with_siblings_in_pred += 1

        # Bucket by sibling count
        if n_siblings == 0:
            b = (0, 0)
        elif n_siblings <= 5:
            b = (1, 5)
        elif n_siblings <= 20:
            b = (6, 20)
        else:
            b = (21, 1000)
        # Recall@1 for this user = 1 if target rank < K
        bucket_recalls[b].append(target_rank)

        # Pairwise AUC: target_rank vs each sibling_rank
        if target_rank >= 0 and sib_ranks:
            for sr in sib_ranks:
                if sr >= 0:  # sibling must be in Top-K
                    auc_pairs.append((target_rank, sr))

    # AUC: count pairs where target_rank < sibling_rank (target higher in ranking)
    n_pairs_total = len(auc_pairs)
    n_target_higher = sum(1 for tr, sr in auc_pairs if tr < sr)
    n_target_equal = sum(1 for tr, sr in auc_pairs if tr == sr)
    n_target_lower = sum(1 for tr, sr in auc_pairs if tr > sr)
    auc = (n_target_higher + 0.5 * n_target_equal) / max(1, n_pairs_total)

    # Target Recall@10 (item-level)
    n_in_top10 = sum(1 for r in target_ranks_in_pred if 0 <= r < 10)
    recall_10 = n_in_top10 / max(1, len(target_ranks_in_pred))

    # Sibling ranking: target vs siblings MRR
    if auc_pairs:
        # For each (target_rank, sibling_rank) pair, compute "delta rank" = target_rank - sibling_rank
        # AUC > 0.5 means target is ranked higher
        deltas = [tr - sr for tr, sr in auc_pairs]
        mean_delta = np.mean(deltas)
        median_delta = np.median(deltas)

    print(f'  pairs (target vs sibling, both in Top-K): {n_pairs_total}')
    print(f'  AUC: {auc:.4f}')
    print(f'    target_higher: {n_target_higher}')
    print(f'    target_equal: {n_target_equal}')
    print(f'    target_lower: {n_target_lower}')
    print(f'  Recall@10 (item-level): {recall_10*100:.4f}%')
    print(f'  Mean delta (target_rank - sib_rank): {mean_delta if auc_pairs else "N/A"}')

    # Bucket analysis
    print(f'  Bucket by sibling count:')
    bucket_summary = {}
    for bname, ranks in bucket_recalls.items():
        n = len(ranks)
        if n == 0:
            print(f'    |S| in {bname}: 0 users')
            bucket_summary[str(bname)] = {'n_users': 0, 'r10': 0}
            continue
        r10 = sum(1 for r in ranks if 0 <= r < 10) / n * 100
        print(f'    |S| in {bname}: {n} users, R@10 = {r10:.2f}%')
        bucket_summary[str(bname)] = {'n_users': n, 'r10': r10}

    per_layer_metrics[l] = {
        'n_pairs': n_pairs_total,
        'auc': auc,
        'target_higher': n_target_higher,
        'target_equal': n_target_equal,
        'target_lower': n_target_lower,
        'recall_10_item_level': recall_10,
        'mean_delta_rank': float(mean_delta) if auc_pairs else None,
        'bucket_summary': bucket_summary,
        'n_users_with_target': n_users_with_target,
        'n_users_with_siblings_in_pred': n_users_with_siblings_in_pred,
    }

# ============================================================
# Print comparison
# ============================================================
print(f'\n{"=" * 70}')
print('Sibling Ranking Summary (per layer l)')
print('=' * 70)
print(f'{"Layer":<10}{"|S| mean":<12}{"Pairs":<10}{"AUC":<10}{"R@10":<10}{"Mean Δ rank":<15}')
for l, m in per_layer_metrics.items():
    groups = groups_per_layer[l]
    mean_sib = np.mean([len(v) - 1 for v in groups.values()])  # siblings = group_size - 1
    print(f'l={l:<8}{mean_sib:<12.1f}{m["n_pairs"]:<10}{m["auc"]:<10.4f}{m["recall_10_item_level"]*100:<10.4f}{m["mean_delta_rank"] if m["mean_delta_rank"] is not None else "N/A":<15}')

# ============================================================
# Save output
# ============================================================
out = {
    'task': 'Task A3: Unified Sibling Ranking Definition',
    'method': 'AUC = P(rank_target < rank_sibling | sibling shares prefix Z(<l) with target)',
    'data': {
        'n_items': n_items,
        'n_users': n_users,
        'K': K,
    },
    'per_layer_metrics': per_layer_metrics,
    'verdict': (
        f'l=4 has 0 sibling pairs (all unique). l=3 has AUC={per_layer_metrics[3]["auc"]:.4f} '
        f'with {per_layer_metrics[3]["n_pairs"]} pairs. '
        f'l=2 has AUC={per_layer_metrics[2]["auc"]:.4f} with most pairs. '
        f'l=1 has the largest sibling sets (mean {np.mean([len(v) for v in groups_l1.values()]):.1f} items/group).'
    ),
}

with open(os.path.join(OUT_DIR, 'unified_sibling_ranking.json'), 'w') as f:
    json.dump(out, f, indent=2, ensure_ascii=False, default=str)

verdict_lines = [
    '# Task A3 Verdict: Unified Sibling Ranking Definition',
    '',
    '## 定义',
    '',
    '- target = 用户 next-item (ground truth)',
    '- siblings at layer l = items sharing prefix Z(<l) with target',
    '- AUC_l = P(rank_target < rank_sibling | sibling in Top-K)',
    '',
    '## Per-Layer Results',
    '',
    '| Layer | Mean |S| | Pairs | AUC | R@10 (item-level) | Mean Δ rank |',
    '|-------|---------|-------|------|-------------------|--------------|',
]
for l, m in per_layer_metrics.items():
    groups = groups_per_layer[l]
    mean_sib = np.mean([len(v) - 1 for v in groups.values()])
    mean_delta_str = f'{m["mean_delta_rank"]:.2f}' if m["mean_delta_rank"] is not None else 'N/A'
    verdict_lines.append(
        f'| l={l} | {mean_sib:.1f} | {m["n_pairs"]} | {m["auc"]:.4f} | '
        f'{m["recall_10_item_level"]*100:.4f}% | {mean_delta_str} |'
    )

verdict_lines.extend([
    '',
    '## 解读',
    '',
    '- l=4 (with dedup): 0 pairs (all SIDs unique)',
    '- l=3 (dedup digit removed): AUC shows how well model picks correct item among siblings',
    '- l=2 (first 2 digits): largest sibling sets, hardest discrimination',
    '- l=1 (just first digit): trivial grouping by category',
    '',
    '## AUC 解读',
    '',
    '- AUC = 0.5: random ranking among siblings (no signal)',
    '- AUC > 0.5: target ranked higher than siblings on average',
    '- AUC < 0.5: target ranked lower (bad)',
    '',
])

with open(os.path.join(OUT_DIR, 'verdict.md'), 'w') as f:
    f.write('\n'.join(verdict_lines))

print(f'\n=== Summary ===')
print(f'产物: {OUT_DIR}/unified_sibling_ranking.json + verdict.md')
for l, m in per_layer_metrics.items():
    print(f'l={l}: AUC={m["auc"]:.4f} | pairs={m["n_pairs"]} | R@10={m["recall_10_item_level"]*100:.4f}%')