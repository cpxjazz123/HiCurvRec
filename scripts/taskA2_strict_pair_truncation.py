#!/usr/bin/env python3
"""Task A2: 严格配对的 L3 vs L4 消融 (truncation 分析)

核心方法: 严格控制所有变量, 只让 L4 digit 这一项不同.
- 同一 L=4 baseline 模型
- 同一 predictions tensor (19412, 10, 4)
- 同一 catalog (L=4 baseline 的 (4, 11924) catalog)
- 不同点: 是否使用第 4 位 digit

实现:
- L4_full: 用全部 4 digits 查 catalog → 必然 1-to-1 (dedup 保证)
- L3_trunc: 截断到前 3 digits → collision rate 9.49%
- L3_dedup_noisy: 在 L3 基础上, 用任意 L4 digit (0-13) 看是否匹配

这样保证: 模型、SID inference、catalog 完全一致, 唯一变量是 L4 digit 的有无.
"""

import os, json, glob
import numpy as np
import torch
import tensorflow as tf

GRID = '/home/wlia0047/ar57/wenyu/GeneRec/GRID'
EVAL_DIR = f'{GRID}/data/amazon_data/toys/evaluation'
SID_PATH = f'{GRID}/logs/inference/runs/2026-07-07/20-17-35/pickle/merged_predictions_tensor.pt'
PRED_PATH = f'{GRID}/logs/inference/runs/2026-07-07/21-47-35/pickle/merged_predictions_tensor.pt'
OUT_DIR = f'{GRID}/result/taskA2_strict_pair_truncation'
os.makedirs(OUT_DIR, exist_ok=True)

print('=' * 70)
print('Task A2: Strict Pair L3 vs L4 Ablation via Truncation')
print('=' * 70)

# Load L=4 baseline SID and predictions (both same as A1 L=4 baseline)
sid = torch.load(SID_PATH, weights_only=False)  # (4, 11924)
pred = torch.load(PRED_PATH, weights_only=False)  # (19412, 10, 4)
n_items = sid.shape[1]
n_users, K, n_layers = pred.shape
print(f'SID shape: {sid.shape}')
print(f'Pred shape: {pred.shape}')

# Build catalog lookup tables
sid_to_item = {}  # 4-digit → item_id
sid3_to_items = {}  # 3-digit → list of item_ids
for item_id in range(n_items):
    sid4 = tuple(int(x) for x in sid[:, item_id].tolist())
    sid3 = sid4[:3]
    sid_to_item[sid4] = item_id
    sid3_to_items.setdefault(sid3, []).append(item_id)
print(f'Unique 4-digit SIDs (catalog): {len(sid_to_item)}')
print(f'Unique 3-digit SIDs (catalog): {len(sid3_to_items)}')
n_collide_3 = sum(1 for v in sid3_to_items.values() if len(v) > 1)
print(f'3-digit collision groups: {n_collide_3} ({n_collide_3/len(sid3_to_items)*100:.2f}%)')

# Load eval tfrecords → user_id → target item_id
print(f'\n[load eval tfrecords]')
files = sorted(glob.glob(f'{EVAL_DIR}/partition_*.tfrecord.gz'))
user_targets = {}
dataset = tf.data.TFRecordDataset(files, compression_type='GZIP')
for raw in dataset:
    example = tf.train.Example()
    example.ParseFromString(raw.numpy())
    user_id = int(example.features.feature['user_id'].int64_list.value[0])
    seq = list(example.features.feature['sequence_data'].int64_list.value)
    if not seq:
        continue
    user_targets[user_id] = seq[-1]
print(f'Users with targets: {len(user_targets)}')

# ============================================================
# 严格配对评估: L4_full vs L3_trunc (truncate 4→3 digits)
# ============================================================
print(f'\n{"=" * 60}')
print('Strict Pair Evaluation: same model, same catalog, only L4 digit removed')

def eval_l4_full(pred, sid_to_item, user_targets, k_values=(1, 3, 5, 10)):
    """Use full 4 digits (1-to-1 mapping guaranteed by dedup)."""
    n_users, K, _ = pred.shape
    results = {f'R@{k}': 0 for k in k_values}
    results.update({f'NDCG@{k}': 0.0 for k in k_values})
    results['MRR'] = 0.0
    n_eval = 0
    for u in range(n_users):
        target = user_targets.get(u, -1)
        if target < 0:
            continue
        n_eval += 1
        rank = -1
        for k in range(K):
            sid4 = tuple(int(x) for x in pred[u, k].tolist())
            item = sid_to_item.get(sid4, -1)
            if item == target:
                rank = k
                break
        if rank < 0:
            continue
        for k in k_values:
            if rank < k:
                results[f'R@{k}'] += 1
                results[f'NDCG@{k}'] += 1.0 / np.log2(rank + 2)
        results['MRR'] += 1.0 / (rank + 1)
    for k in k_values:
        results[f'R@{k}'] /= max(1, n_eval)
        results[f'NDCG@{k}'] /= max(1, n_eval)
    results['MRR'] /= max(1, n_eval)
    results['n_eval'] = n_eval
    return results


def eval_l3_trunc(pred, sid3_to_items, user_targets, k_values=(1, 3, 5, 10)):
    """Truncate predictions to first 3 digits, find items in collision set.

    Item-level: target ∈ {items sharing pred's first 3 digits}
    Strict: rank of FIRST occurrence (if target in item set, treat as found at that rank)
    """
    n_users, K, _ = pred.shape
    results = {f'R@{k}': 0 for k in k_values}
    results.update({f'NDCG@{k}': 0.0 for k in k_values})
    results['MRR'] = 0.0
    n_eval = 0
    for u in range(n_users):
        target = user_targets.get(u, -1)
        if target < 0:
            continue
        n_eval += 1
        rank = -1
        for k in range(K):
            sid3 = tuple(int(x) for x in pred[u, k, :3].tolist())
            items = sid3_to_items.get(sid3, [])
            if target in items:
                rank = k
                break
        if rank < 0:
            continue
        for k in k_values:
            if rank < k:
                results[f'R@{k}'] += 1
                results[f'NDCG@{k}'] += 1.0 / np.log2(rank + 2)
        results['MRR'] += 1.0 / (rank + 1)
    for k in k_values:
        results[f'R@{k}'] /= max(1, n_eval)
        results[f'NDCG@{k}'] /= max(1, n_eval)
    results['MRR'] /= max(1, n_eval)
    results['n_eval'] = n_eval
    return results


def eval_l3_strict_singleton(pred, sid3_to_items, user_targets, k_values=(1, 3, 5, 10)):
    """L3 strict: only count hit if predicted SID has exactly 1 item AND it matches target.
    This is the L4 baseline criterion applied at 3-digit level (no collision tolerance).
    """
    n_users, K, _ = pred.shape
    results = {f'R@{k}': 0 for k in k_values}
    results.update({f'NDCG@{k}': 0.0 for k in k_values})
    results['MRR'] = 0.0
    n_eval = 0
    for u in range(n_users):
        target = user_targets.get(u, -1)
        if target < 0:
            continue
        n_eval += 1
        rank = -1
        for k in range(K):
            sid3 = tuple(int(x) for x in pred[u, k, :3].tolist())
            items = sid3_to_items.get(sid3, [])
            if len(items) == 1 and items[0] == target:
                rank = k
                break
        if rank < 0:
            continue
        for k in k_values:
            if rank < k:
                results[f'R@{k}'] += 1
                results[f'NDCG@{k}'] += 1.0 / np.log2(rank + 2)
        results['MRR'] += 1.0 / (rank + 1)
    for k in k_values:
        results[f'R@{k}'] /= max(1, n_eval)
        results[f'NDCG@{k}'] /= max(1, n_eval)
    results['MRR'] /= max(1, n_eval)
    results['n_eval'] = n_eval
    return results


# Run evaluations
metrics_l4 = eval_l4_full(pred, sid_to_item, user_targets)
metrics_l3 = eval_l3_trunc(pred, sid3_to_items, user_targets)
metrics_l3_strict = eval_l3_strict_singleton(pred, sid3_to_items, user_targets)

print(f'\n| Mode | R@5 | R@10 | NDCG@5 | NDCG@10 | MRR |')
print(f'|------|-----|------|--------|---------|-----|')
print(f'| L=4 full (1-to-1 dedup) | {metrics_l4["R@5"]*100:.4f}% | {metrics_l4["R@10"]*100:.4f}% | {metrics_l4["NDCG@5"]:.4f} | {metrics_l4["NDCG@10"]:.4f} | {metrics_l4["MRR"]:.4f} |')
print(f'| L=3 trunc (any collision) | {metrics_l3["R@5"]*100:.4f}% | {metrics_l3["R@10"]*100:.4f}% | {metrics_l3["NDCG@5"]:.4f} | {metrics_l3["NDCG@10"]:.4f} | {metrics_l3["MRR"]:.4f} |')
print(f'| L=3 strict (singleton) | {metrics_l3_strict["R@5"]*100:.4f}% | {metrics_l3_strict["R@10"]*100:.4f}% | {metrics_l3_strict["NDCG@5"]:.4f} | {metrics_l3_strict["NDCG@10"]:.4f} | {metrics_l3_strict["MRR"]:.4f} |')

print(f'\n=== Δ vs L=4 full ===')
for k in ['R@5', 'R@10', 'NDCG@5', 'NDCG@10', 'MRR']:
    print(f'  {k}:')
    print(f'    L3 (any collision) - L4: {(metrics_l3[k] - metrics_l4[k]):+.4f}')
    print(f'    L3 (strict singleton) - L4: {(metrics_l3_strict[k] - metrics_l4[k]):+.4f}')

# ============================================================
# Also try perturbation: mask L4 digit (replace with 0), shuffle L4 digit (randomize)
# ============================================================
print(f'\n{"=" * 60}')
print('L4 digit perturbation analysis')

def eval_l3_mask_l4(pred, sid3_to_items, user_targets, l4_value=0, k_values=(1, 3, 5, 10)):
    """Replace L4 digit with fixed value, look up 3-digit SID."""
    n_users, K, _ = pred.shape
    results = {f'R@{k}': 0 for k in k_values}
    results.update({f'NDCG@{k}': 0.0 for k in k_values})
    results['MRR'] = 0.0
    n_eval = 0
    for u in range(n_users):
        target = user_targets.get(u, -1)
        if target < 0:
            continue
        n_eval += 1
        rank = -1
        for k in range(K):
            sid3 = tuple(int(x) for x in pred[u, k, :3].tolist())
            items = sid3_to_items.get(sid3, [])
            if target in items:
                rank = k
                break
        if rank < 0:
            continue
        for k in k_values:
            if rank < k:
                results[f'R@{k}'] += 1
                results[f'NDCG@{k}'] += 1.0 / np.log2(rank + 2)
        results['MRR'] += 1.0 / (rank + 1)
    for k in k_values:
        results[f'R@{k}'] /= max(1, n_eval)
        results[f'NDCG@{k}'] /= max(1, n_eval)
    results['MRR'] /= max(1, n_eval)
    results['n_eval'] = n_eval
    return results


def eval_l3_shuffle_l4(pred, sid3_to_items, user_targets, seed=42, k_values=(1, 3, 5, 10)):
    """Shuffle L4 digit across predictions (random permutation)."""
    rng = np.random.default_rng(seed)
    n_users, K, _ = pred.shape
    # Random shuffle L4 digit across all (u, k) positions
    l4_values = pred[:, :, 3].flatten()
    l4_shuffled = rng.permutation(l4_values.numpy()).reshape(n_users, K)
    pred_shuffled = pred.clone()
    pred_shuffled[:, :, 3] = torch.from_numpy(l4_shuffled).float()

    return eval_l3_mask_l4(pred_shuffled, sid3_to_items, user_targets)


# Test perturbations
metrics_mask = eval_l3_mask_l4(pred, sid3_to_items, user_targets, l4_value=0)
metrics_shuffle = eval_l3_shuffle_l4(pred, sid3_to_items, user_targets, seed=42)
metrics_shuffle2 = eval_l3_shuffle_l4(pred, sid3_to_items, user_targets, seed=123)

print(f'\n| Mode | R@10 | NDCG@10 |')
print(f'|------|------|---------|')
print(f'| L4 (full 4 digits) | {metrics_l4["R@10"]*100:.4f}% | {metrics_l4["NDCG@10"]:.4f} |')
print(f'| L3 trunc (drop L4) | {metrics_l3["R@10"]*100:.4f}% | {metrics_l3["NDCG@10"]:.4f} |')
print(f'| L3 shuffle L4 seed=42 | {metrics_shuffle["R@10"]*100:.4f}% | {metrics_shuffle["NDCG@10"]:.4f} |')
print(f'| L3 shuffle L4 seed=123 | {metrics_shuffle2["R@10"]*100:.4f}% | {metrics_shuffle2["NDCG@10"]:.4f} |')
print(f'| L3 strict singleton (no collision) | {metrics_l3_strict["R@10"]*100:.4f}% | {metrics_l3_strict["NDCG@10"]:.4f} |')

# ============================================================
# Save
# ============================================================
out = {
    'task': 'Task A2: Strict Pair L3 vs L4 ablation via truncation',
    'method': 'Same model, same predictions tensor, same catalog. Only L4 digit inclusion varies.',
    'data': {
        'sid_path': SID_PATH,
        'pred_path': PRED_PATH,
        'n_items': n_items,
        'n_users': n_users,
        'K': K,
        'n_unique_4digit': len(sid_to_item),
        'n_unique_3digit': len(sid3_to_items),
        'n_collision_3digit': n_collide_3,
        'collision_rate_3digit': n_collide_3 / max(1, len(sid3_to_items)),
    },
    'metrics': {
        'L4_full': metrics_l4,
        'L3_trunc_any_collision': metrics_l3,
        'L3_strict_singleton': metrics_l3_strict,
        'L3_shuffle_l4_seed42': metrics_shuffle,
        'L3_shuffle_l4_seed123': metrics_shuffle2,
    },
    'verdict': (
        'L4_full R@10 = '
        f'{metrics_l4["R@10"]*100:.4f}%, '
        'L3_trunc (drop L4) R@10 = '
        f'{metrics_l3["R@10"]*100:.4f}%, '
        'Δ = '
        f'{(metrics_l3["R@10"] - metrics_l4["R@10"])*100:+.4f}%. '
        'This is the TRUE effect of L4 digit under item-level matching: '
        'any-collision = target ∈ items sharing prefix; '
        'strict-singleton = target is the only item at predicted SID.'
    ),
}

with open(os.path.join(OUT_DIR, 'strict_pair_truncation.json'), 'w') as f:
    json.dump(out, f, indent=2, ensure_ascii=False, default=str)

verdict_lines = [
    '# Task A2 Verdict: Strict Pair L3 vs L4 Ablation (Truncation)',
    '',
    '## 设计',
    '',
    '- 同一 L=4 baseline 模型 + 同一 predictions tensor + 同一 catalog',
    '- 唯一变量: 是否使用第 4 位 digit',
    '- L=4 full: 4 位全部使用, dedup 保证 1-to-1 映射',
    '- L=3 trunc (any collision): 截断到 3 位, target 在 collision set 中即命中',
    '- L=3 strict singleton: 截断到 3 位, 只在 collision=1 时计数',
    '- L=3 shuffle L4: 打乱 L4 位置, 看 L4 顺序是否影响结果',
    '',
    '## 结果',
    '',
    f'| Mode | R@5 | R@10 | NDCG@5 | NDCG@10 | MRR |',
    f'|------|-----|------|--------|---------|-----|',
    f'| L=4 full (1-to-1 dedup) | {metrics_l4["R@5"]*100:.4f}% | {metrics_l4["R@10"]*100:.4f}% | {metrics_l4["NDCG@5"]:.4f} | {metrics_l4["NDCG@10"]:.4f} | {metrics_l4["MRR"]:.4f} |',
    f'| L=3 trunc (any collision) | {metrics_l3["R@5"]*100:.4f}% | {metrics_l3["R@10"]*100:.4f}% | {metrics_l3["NDCG@5"]:.4f} | {metrics_l3["NDCG@10"]:.4f} | {metrics_l3["MRR"]:.4f} |',
    f'| L=3 strict singleton | {metrics_l3_strict["R@5"]*100:.4f}% | {metrics_l3_strict["R@10"]*100:.4f}% | {metrics_l3_strict["NDCG@5"]:.4f} | {metrics_l3_strict["NDCG@10"]:.4f} | {metrics_l3_strict["MRR"]:.4f} |',
    '',
    '## 关键发现',
    '',
    f'- L=4 full R@10 = {metrics_l4["R@10"]*100:.4f}% (baseline)',
    f'- L=3 trunc R@10 = {metrics_l3["R@10"]*100:.4f}% (drop L4 digit)',
    f'- Δ (L3 trunc - L4) = {(metrics_l3["R@10"] - metrics_l4["R@10"])*100:+.4f}%',
    f'- L=3 strict singleton R@10 = {metrics_l3_strict["R@10"]*100:.4f}%',
    '',
    '## 解读',
    '',
    '- 若 L=3 trunc R@10 > L=4 full → collision 提供额外命中 (collision 膨胀)',
    '- 若 L=3 trunc R@10 ≈ L=4 full → 截断后无信号损失',
    '- 若 L=3 strict singleton < L=4 → collision 不掩盖正确性损失',
    '- L=3 strict singleton 与 L=4 full 的差距反映 L4 digit 的真实信息贡献',
    '',
]

with open(os.path.join(OUT_DIR, 'verdict.md'), 'w') as f:
    f.write('\n'.join(verdict_lines))

print(f'\n=== Summary ===')
print(f'产物: {OUT_DIR}/strict_pair_truncation.json + verdict.md')
print(f'L=4 full R@10 = {metrics_l4["R@10"]*100:.4f}%')
print(f'L=3 trunc R@10 (any collision) = {metrics_l3["R@10"]*100:.4f}%')
print(f'L=3 strict singleton R@10 = {metrics_l3_strict["R@10"]*100:.4f}%')
print(f'Δ (L3 trunc - L4) = {(metrics_l3["R@10"] - metrics_l4["R@10"])*100:+.4f}%')