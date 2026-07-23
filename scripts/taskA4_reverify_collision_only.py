#!/usr/bin/env python3
"""Task A4: 重新验证 collision-only layer

v4 doc 声称 L3 layer 是 "collision reduction layer" (CR=0.95, ΔH_beh≈0).
本任务验证该 claim:
- CR_l = collision reduction rate at layer l
- ΔH_l^beh = behavior purity increment at layer l (从 L_{l-1} 到 L_l)

判定: collision-only layer 应满足
  CR_l >> 0 (collision rate 大幅下降)
  AND
  ΔH_l^beh ≈ 0 (behavior 信息不增)

具体:
- 计算每层的 collision rate (catalog)
- 计算每层的 behavior purity (用 user co-purchase 作为 proxy for behavior)
- 验证 claim

实现:
1. Catalog-level collision rate per layer
2. Behavior purity: H(co-purchase | Z_l) for each layer
3. ΔH_l^beh = H(B|Z_l) - H(B|Z_{l-1})
4. 输出 CR_l vs ΔH_l^beh 散点
"""

import os, json, glob
import numpy as np
import torch
import tensorflow as tf
from collections import defaultdict
from scipy.special import entr

GRID = '/home/wlia0047/ar57/wenyu/GeneRec/GRID'
EVAL_DIR = f'{GRID}/data/amazon_data/toys/evaluation'
SID_PATH = f'{GRID}/logs/inference/runs/2026-07-07/20-17-35/pickle/merged_predictions_tensor.pt'
PRED_PATH = f'{GRID}/logs/inference/runs/2026-07-07/21-47-35/pickle/merged_predictions_tensor.pt'
OUT_DIR = f'{GRID}/result/taskA4_reverify_collision_only'
os.makedirs(OUT_DIR, exist_ok=True)

print('=' * 70)
print('Task A4: Re-verify "collision-only layer" claim')
print('=' * 70)

# Load SID and predictions
sid = torch.load(SID_PATH, weights_only=False)  # (4, 11924)
pred = torch.load(PRED_PATH, weights_only=False)
n_items = sid.shape[1]
n_users, K, n_layers = pred.shape

# Load eval tfrecords (for user co-purchase behavior)
print(f'\n[load eval tfrecords → user history → item co-purchase]')
files = sorted(glob.glob(f'{EVAL_DIR}/partition_*.tfrecord.gz'))
dataset = tf.data.TFRecordDataset(files, compression_type='GZIP')

# Build item → users dict
item_users = defaultdict(set)
user_items = defaultdict(set)
user_sequences = {}
for raw in dataset:
    example = tf.train.Example()
    example.ParseFromString(raw.numpy())
    user_id = int(example.features.feature['user_id'].int64_list.value[0])
    seq = list(example.features.feature['sequence_data'].int64_list.value)
    if not seq:
        continue
    user_sequences[user_id] = seq
    for item in seq:
        item_users[item].add(user_id)
        user_items[user_id].add(item)

print(f'  items with users: {len(item_users)}')
print(f'  users with items: {len(user_items)}')

# Co-purchase graph: item_i ↔ item_j if shared users
print(f'\n[build co-purchase matrix]')
co_count = defaultdict(lambda: defaultdict(int))
for user_id, items in user_items.items():
    items_list = list(items)
    for i in range(len(items_list)):
        for j in range(i+1, len(items_list)):
            a, b = items_list[i], items_list[j]
            if a < b:
                co_count[a][b] += 1
            else:
                co_count[b][a] += 1
print(f'  co-purchase pairs: {sum(len(v) for v in co_count.values())}')

# ============================================================
# Per-layer analysis: collision rate CR_l and behavior purity ΔH_l^beh
# ============================================================
print(f'\n[per-layer analysis]')

# Layer 0 baseline: no SID grouping → all items in their own group (no collision)
# Layer l: group items by first l digits of SID

def build_prefix_groups(sid_tensor, n_layers_used):
    groups = {}
    for item_id in range(sid_tensor.shape[1]):
        key = tuple(int(x) for x in sid_tensor[:n_layers_used, item_id].tolist())
        groups.setdefault(key, []).append(item_id)
    return groups


def compute_layer_stats(layer, prefix_groups):
    """For layer l, compute:
    - n_groups: number of unique prefixes
    - collision_count: groups with > 1 items
    - collision_rate: fraction of items that are in a collision group
    - n_items_in_collision: items sharing SID with at least one other
    """
    n_items = sum(len(v) for v in prefix_groups.values())
    n_groups = len(prefix_groups)
    n_collision_groups = sum(1 for v in prefix_groups.values() if len(v) > 1)
    n_items_in_collision = sum(len(v) for v in prefix_groups.values() if len(v) > 1)
    collision_rate = n_items_in_collision / max(1, n_items)
    return {
        'n_groups': n_groups,
        'n_collision_groups': n_collision_groups,
        'n_items_in_collision': n_items_in_collision,
        'collision_rate': collision_rate,
    }


layer_stats = {}
for l in range(1, n_layers + 1):
    groups = build_prefix_groups(sid, l)
    stats = compute_layer_stats(l, groups)
    layer_stats[l] = stats
    print(f'  Layer {l}: {stats["n_groups"]} groups, {stats["n_collision_groups"]} collision groups, '
          f'collision_rate = {stats["collision_rate"]*100:.2f}%')

# Compute CR_l = 1 - collision_rate_l / collision_rate_{l-1} (rate of reduction)
print(f'\n[CR_l (collision reduction rate per layer)]')
cr_per_layer = {}
prev_rate = 1.0  # l=0 baseline: all items in different groups
for l in range(1, n_layers + 1):
    cur_rate = layer_stats[l]['collision_rate']
    if prev_rate > 0:
        cr = 1 - cur_rate / prev_rate
    else:
        cr = 0.0
    cr_per_layer[l] = cr
    print(f'  CR_{l} = 1 - {cur_rate:.4f}/{prev_rate:.4f} = {cr:.4f}')
    prev_rate = cur_rate

# ============================================================
# Behavior purity: H(co-purchase | Z_l)
# ============================================================
print(f'\n[behavior purity H(B|Z_l)]')

def compute_behavior_purity(layer, prefix_groups, co_count, max_pairs_per_group=200):
    """Compute H(co-purchase | Z_l) = expected entropy of co-purchase within each group.

    For each group (items sharing prefix Z_l):
    - Sample all (i, j) pairs from group
    - Compute co-purchase frequency distribution P(co_count)
    - H(group) = -Σ p log p
    Average over groups (weighted by group size)
    """
    total_entropy = 0.0
    total_weight = 0
    group_entropies = []
    for prefix, items in prefix_groups.items():
        if len(items) < 2:
            continue
        # Sample pairs from this group
        pair_counts = []
        for i_idx in range(len(items)):
            for j_idx in range(i_idx+1, len(items)):
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
        # Entropy of co-count distribution
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
        group_entropies.append((len(items), H))
    avg_entropy = total_entropy / max(1, total_weight)
    return avg_entropy, group_entropies


bp_per_layer = {}
for l in range(1, n_layers + 1):
    groups = build_prefix_groups(sid, l)
    bp, ge = compute_behavior_purity(l, groups, co_count)
    bp_per_layer[l] = bp
    print(f'  H(B|Z_{l}) = {bp:.4f}')

# ΔH_l^beh = H(B|Z_l) - H(B|Z_{l-1})
print(f'\n[ΔH_l^beh (behavior purity increment per layer)]')
delta_h_per_layer = {}
prev_h = bp_per_layer[1]  # baseline = layer 1
for l in range(2, n_layers + 1):
    cur_h = bp_per_layer[l]
    delta = cur_h - prev_h
    delta_h_per_layer[l] = delta
    print(f'  ΔH_{l}^beh = H(Z_{l}) - H(Z_{l-1}) = {cur_h:.4f} - {prev_h:.4f} = {delta:+.4f}')
    prev_h = cur_h

# Also baseline: H(B|random_group) — random grouping entropy
print(f'\n[baseline: H(B | random groups)]')
np.random.seed(42)
random_sizes = [len(v) for v in build_prefix_groups(sid, 1).values()]
# Random groups of same sizes but shuffled
shuffled_items = np.random.permutation(n_items)
random_groups = {}
idx = 0
for prefix, size in enumerate(random_sizes):
    grp = shuffled_items[idx:idx+size].tolist()
    random_groups[prefix] = grp
    idx += size
bp_random, _ = compute_behavior_purity(0, random_groups, co_count)
print(f'  H(B | random groups of same sizes) = {bp_random:.4f}')

# ============================================================
# Validate claim: CR_l >> 0 AND ΔH_l^beh ≈ 0 for "collision-only layer"
# ============================================================
print(f'\n{"=" * 60}')
print('Validating claim: "L3 is collision-only layer"')

print(f'\n| Layer | CR_l | ΔH_l^beh | Collision-only? |')
print(f'|-------|------|----------|-----------------|')
for l in range(1, n_layers + 1):
    cr = cr_per_layer.get(l, 0)
    if l == 1:
        dh_str = 'N/A (baseline)'
        claim = 'N/A'
    else:
        dh = delta_h_per_layer.get(l, 0)
        dh_str = f'{dh:+.4f}'
        # collision-only iff CR_l >> 0 AND |ΔH_l^beh| small
        is_collision_only = cr > 0.5 and abs(dh) < 0.5 * abs(bp_per_layer[1] - bp_per_layer[n_layers])
        claim = 'YES' if is_collision_only else 'NO'
    print(f'| l={l} | {cr:.4f} | {dh_str} | {claim} |')

# ============================================================
# Save
# ============================================================
out = {
    'task': 'Task A4: Re-verify "collision-only layer" claim',
    'method': 'CR_l = collision reduction rate per layer; ΔH_l^beh = behavior purity increment',
    'data': {
        'n_items': n_items,
        'n_layers': n_layers,
    },
    'layer_stats': layer_stats,
    'collision_reduction_per_layer': cr_per_layer,
    'behavior_purity_per_layer': bp_per_layer,
    'behavior_purity_increment': delta_h_per_layer,
    'random_baseline_entropy': bp_random,
    'verdict': (
        f'Layer 3 has CR_3={cr_per_layer.get(3, 0):.4f} '
        f'and ΔH_3^beh={delta_h_per_layer.get(3, 0):+.4f}. '
        f'Layer 4 has CR_4={cr_per_layer.get(4, 0):.4f} '
        f'and ΔH_4^beh={delta_h_per_layer.get(4, 0):+.4f}. '
        'Collision-only layer = CR_l >> 0 AND |ΔH_l^beh| small. '
        'See verdict.md for per-layer verdict.'
    ),
}

with open(os.path.join(OUT_DIR, 'collision_only_layer.json'), 'w') as f:
    json.dump(out, f, indent=2, ensure_ascii=False, default=str)

verdict_lines = [
    '# Task A4 Verdict: Re-verify "Collision-Only Layer" Claim',
    '',
    '## 定义',
    '',
    '- CR_l = 1 - collision_rate_l / collision_rate_{l-1} (collision reduction rate)',
    '- ΔH_l^beh = H(B|Z_l) - H(B|Z_{l-1}) (behavior purity increment)',
    '- collision-only layer iff CR_l >> 0 AND |ΔH_l^beh| small',
    '',
    '## Per-layer Stats',
    '',
    '| Layer | n_groups | n_collision_groups | collision_rate | CR_l | ΔH_l^beh |',
    '|-------|----------|--------------------|--------------|------|----------|',
]
for l in range(1, n_layers + 1):
    s = layer_stats[l]
    cr = cr_per_layer.get(l, 0)
    if l == 1:
        dh_str = 'N/A'
    else:
        dh = delta_h_per_layer.get(l, 0)
        dh_str = f'{dh:+.4f}'
    verdict_lines.append(
        f'| l={l} | {s["n_groups"]} | {s["n_collision_groups"]} | {s["collision_rate"]*100:.2f}% | '
        f'{cr:.4f} | {dh_str} |'
    )

verdict_lines.extend([
    '',
    '## Behavior Purity H(B|Z_l)',
    '',
    f'- Random baseline: H(B | random groups) = {bp_random:.4f}',
])
for l, bp in bp_per_layer.items():
    verdict_lines.append(f'- H(B|Z_{l}) = {bp:.4f}')

verdict_lines.extend([
    '',
    '## 判定',
    '',
    '- v4 doc 声称 L3 是 collision-only layer (CR=0.95, ΔH≈0)',
    '- 实际数据看 collision_rate reduction 与 behavior purity 是否同时显著',
    '- 若 CR 大但 ΔH 也大 → 该 layer 不仅去 collision 还编码 behavior',
    '- 若 CR 大且 ΔH 小 → collision-only 假设成立',
    '',
])

with open(os.path.join(OUT_DIR, 'verdict.md'), 'w') as f:
    f.write('\n'.join(verdict_lines))

print(f'\n=== Summary ===')
print(f'产物: {OUT_DIR}/collision_only_layer.json + verdict.md')
for l in range(1, n_layers + 1):
    cr = cr_per_layer.get(l, 0)
    if l == 1:
        print(f'l={l}: CR={cr:.4f}')
    else:
        dh = delta_h_per_layer.get(l, 0)
        print(f'l={l}: CR={cr:.4f} | ΔH_beh={dh:+.4f}')