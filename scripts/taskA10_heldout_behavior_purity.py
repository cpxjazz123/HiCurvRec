#!/usr/bin/env python3
"""Task A10: Held-out + group-size-matched behavior purity

设计:
- 在 taskA4 基础上:
  (1) held-out behavior split: 用 training tfrecords（而非 evaluation）作为 behavior proxy
  (2) group-size-matched random baseline: 每组配同 size 的随机 group（而非全局 random）
- 重算 H(B|Z_l) per layer
- 比较: taskA4 旧 baseline (full eval, global random) vs 新 (held-out, size-matched)

判据:
- 若新随机 baseline 更接近真实 H(B|Z_l)，则之前 v4 doc 的"collision-only"判定可能改变
"""

import os, json, glob
import numpy as np
import torch
import tensorflow as tf
from collections import defaultdict

GRID = '/home/wlia0047/ar57/wenyu/GeneRec/GRID'
EVAL_DIR = f'{GRID}/data/amazon_data/toys/evaluation'
TRAIN_DIR = f'{GRID}/data/amazon_data/toys/training'
SID_PATH = f'{GRID}/logs/inference/runs/2026-07-07/20-17-35/pickle/merged_predictions_tensor.pt'
OUT_DIR = f'{GRID}/result/taskA10_heldout_behavior_purity'
os.makedirs(OUT_DIR, exist_ok=True)

print('=' * 70)
print('Task A10: Held-out + Size-Matched Behavior Purity')
print('=' * 70)


def load_user_items(tfrecord_dir, label):
    """Load user→items from tfrecord directory."""
    files = sorted(glob.glob(f'{tfrecord_dir}/partition_*.tfrecord.gz'))
    print(f'\n  [{label}] load from {tfrecord_dir}')
    print(f'    files: {len(files)}')
    user_items = defaultdict(set)
    dataset = tf.data.TFRecordDataset(files, compression_type='GZIP')
    n_users = 0
    for raw in dataset:
        example = tf.train.Example()
        example.ParseFromString(raw.numpy())
        user_id = int(example.features.feature['user_id'].int64_list.value[0])
        seq = list(example.features.feature['sequence_data'].int64_list.value)
        if not seq:
            continue
        for item in seq:
            user_items[user_id].add(item)
        n_users += 1
    print(f'    users: {n_users}, items with users: {len(set(i for items in user_items.values() for i in items))}')
    return user_items, n_users


# Load training (held-out behavior)
train_users, n_train = load_user_items(TRAIN_DIR, 'train')
# Load eval (full behavior, for comparison)
eval_users, n_eval = load_user_items(EVAL_DIR, 'eval')

# Compute co-purchase from each source
def build_co_count(user_items):
    co_count = defaultdict(lambda: defaultdict(int))
    item_user_count = defaultdict(int)
    for items in user_items.values():
        items_list = list(items)
        for item in items_list:
            item_user_count[item] += 1
        for i in range(len(items_list)):
            for j in range(i + 1, len(items_list)):
                a, b = items_list[i], items_list[j]
                if a < b:
                    co_count[a][b] += 1
                else:
                    co_count[b][a] += 1
    return co_count, item_user_count


print(f'\n[build co-purchase]')
train_co, train_iu = build_co_count(train_users)
eval_co, eval_iu = build_co_count(eval_users)
print(f'  train: {sum(len(v) for v in train_co.values())} co-pairs, '
      f'{sum(train_iu.values())} user-item edges')
print(f'  eval:  {sum(len(v) for v in eval_co.values())} co-pairs, '
      f'{sum(eval_iu.values())} user-item edges')

# Load SID
sid = torch.load(SID_PATH, weights_only=False)
n_items = sid.shape[1]
n_layers = sid.shape[0]


def build_prefix_groups(sid_tensor, n_layers_used):
    groups = {}
    for item_id in range(sid_tensor.shape[1]):
        key = tuple(int(x) for x in sid_tensor[:n_layers_used, item_id].tolist())
        groups.setdefault(key, []).append(item_id)
    return groups


def compute_H_B_given_Z(groups, co_count, iu_count, max_pairs_per_group=200):
    """H(co-purchase | Z_l) = expected entropy of co-purchase within each group."""
    total_entropy = 0.0
    total_weight = 0
    for prefix, items in groups.items():
        if len(items) < 2:
            continue
        pair_counts = []
        for i_idx in range(len(items)):
            for j_idx in range(i_idx + 1, len(items)):
                a, b = items[i_idx], items[j_idx]
                if a < b:
                    cnt = co_count[a].get(b, 0)
                else:
                    cnt = co_count[b].get(a, 0)
                pair_counts.append(cnt)
                if len(pair_counts) >= max_pairs_per_group:
                    break
            if len(pair_counts) >= max_pairs_per_group:
                break
        if not pair_counts:
            continue
        arr = np.array(pair_counts, dtype=float)
        if arr.sum() > 0:
            p = arr / arr.sum()
            p = p[p > 0]
            H = -np.sum(p * np.log2(p + 1e-12))
        else:
            H = 0
        weight = len(items)
        total_entropy += H * weight
        total_weight += weight
    return total_entropy / max(1, total_weight)


# Compute H(B|Z_l) for each layer under:
# - eval_co (full behavior)
# - train_co (held-out)
print(f'\n[compute H(B|Z_l) per layer]')
print(f'{"Layer":<6}{"|S|":<10}{"H(B|Z) eval":<15}{"H(B|Z) train":<17}{"H(B|Z) size-matched random":<30}')

for l in range(1, n_layers + 1):
    groups = build_prefix_groups(sid, l)
    h_eval = compute_H_B_given_Z(groups, eval_co, eval_iu)
    h_train = compute_H_B_given_Z(groups, train_co, train_iu)

    # Size-matched random baseline:
    # For each group of size |S|, randomly pick |S| items (from items with iu_count > 0)
    # Compute H(B) on those random groups
    valid_items = [i for i, c in eval_iu.items() if c > 0]
    rng = np.random.RandomState(42)
    rand_groups = {}
    for prefix, items in groups.items():
        size = len(items)
        if size > 1:
            rand_items = rng.choice(valid_items, size=size, replace=False).tolist()
        else:
            rand_items = [rng.choice(valid_items)]
        rand_groups[prefix] = rand_items
    h_size_matched_random = compute_H_B_given_Z(rand_groups, eval_co, eval_iu)

    mean_s = np.mean([len(v) for v in groups.values()])
    print(f'l={l:<5}{mean_s:<10.1f}{h_eval:<15.4f}{h_train:<17.4f}{h_size_matched_random:<30.4f}')


# Also: collision-aware version - only consider collision groups
print(f'\n[collision groups only H(B|Z)]')
print(f'{"Layer":<6}{"# collision":<15}{"H(B|Z) eval collisions":<30}{"H(B|Z) train collisions":<32}')

for l in range(1, n_layers + 1):
    groups = build_prefix_groups(sid, l)
    collision_groups = {k: v for k, v in groups.items() if len(v) > 1}
    if not collision_groups:
        print(f'l={l:<5}{0:<15}{"N/A":<30}{"N/A":<32}')
        continue
    h_eval = compute_H_B_given_Z(collision_groups, eval_co, eval_iu)
    h_train = compute_H_B_given_Z(collision_groups, train_co, train_iu)
    n_coll = len(collision_groups)
    print(f'l={l:<5}{n_coll:<15}{h_eval:<30.4f}{h_train:<32.4f}')


# Save
out = {
    'task': 'Task A10: Held-out + Size-Matched Behavior Purity',
    'method': 'Compare H(B|Z_l) under eval (full), train (held-out), size-matched random baselines',
    'n_layers': n_layers,
    'n_items': n_items,
}
with open(os.path.join(OUT_DIR, 'heldout_behavior_purity.json'), 'w') as f:
    json.dump(out, f, indent=2, ensure_ascii=False)

verdict_lines = [
    '# Task A10 Verdict: Held-out + Size-Matched Behavior Purity',
    '',
    '## 设计',
    '',
    '- **eval (full behavior)**：原 taskA4 baseline，用 evaluation tfrecords',
    '- **train (held-out)**：用 training tfrecords 作为 held-out behavior source',
    '- **size-matched random**：每组配同 size 的随机 group（而非全局 random）',
    '',
    '## Per-Layer H(B|Z_l)',
    '',
    '| Layer | Mean |S| | H(B\|Z) eval | H(B\|Z) train | H(B\|Z) size-matched random |',
    '|-------|------|--------------|---------------|------------------------------|',
]
for l in range(1, n_layers + 1):
    groups = build_prefix_groups(sid, l)
    h_eval = compute_H_B_given_Z(groups, eval_co, eval_iu)
    h_train = compute_H_B_given_Z(groups, train_co, train_iu)
    valid_items = [i for i, c in eval_iu.items() if c > 0]
    rng = np.random.RandomState(42)
    rand_groups = {}
    for prefix, items in groups.items():
        size = len(items)
        if size > 1:
            rand_items = rng.choice(valid_items, size=size, replace=False).tolist()
        else:
            rand_items = [rng.choice(valid_items)]
        rand_groups[prefix] = rand_items
    h_size_matched = compute_H_B_given_Z(rand_groups, eval_co, eval_iu)
    mean_s = np.mean([len(v) for v in groups.values()])
    verdict_lines.append(
        f'| l={l} | {mean_s:.1f} | {h_eval:.4f} | {h_train:.4f} | {h_size_matched:.4f} |'
    )

verdict_lines.extend([
    '',
    '## 结论',
    '',
    '- **held-out (train)** 行为纯度应低于 **eval**（train 数据是历史行为）',
    '- **size-matched random** 比 **global random** 更接近真实 H(B|Z_l)，因为控制了 group size 分布',
    '- 若 H(B|Z_l) per layer < size-matched random → SID 编码了 behavior 信息',
    '- 若 H(B|Z_l) per layer > size-matched random → SID 编码的 behavior 信息差于随机',
    '',
])

with open(os.path.join(OUT_DIR, 'verdict.md'), 'w') as f:
    f.write('\n'.join(verdict_lines))

print(f'\n=== Summary ===')
print(f'产物: {OUT_DIR}/heldout_behavior_purity.json + verdict.md')