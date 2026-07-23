#!/usr/bin/env python3
"""Task A9: Collision group 统一 exact-item tie-breaking

设计:
- 在 taskA1 collision-aware 框架基础上加三档:
  - random_lower_bound: 命中 collision group 时从 group 里随机挑一个 item
  - oracle_upper_bound: 命中 collision group 时选 ground truth
  - reranker_actual: 当前模型（命中 collision group 时选排序第一个）
- 每档报告 R@5/R@10/NDCG@10/MRR
- 算法: AQ_additive, HRQ_v2, L=4 baseline

判据:
- 若 oracle 上界 = reranker 实际 → collision 内 model 选得对（无 tie-breaking 问题）
- 若 oracle 上界 > reranker 实际 → collision 内有 tie-breaking 损失
- 若 random 下界 ≈ reranker 实际 → model 在 collision 内是随机的
"""

import os, json, glob
import numpy as np
import torch
import tensorflow as tf
from collections import defaultdict

GRID = '/home/wlia0047/ar57/wenyu/GeneRec/GRID'
EVAL_DIR = f'{GRID}/data/amazon_data/toys/evaluation'
OUT_DIR = f'{GRID}/result/taskA9_collision_tiebreaking'
os.makedirs(OUT_DIR, exist_ok=True)

print('=' * 70)
print('Task A9: Collision Group Tie-Breaking (random / oracle / reranker)')
print('=' * 70)

# Load eval tfrecords
print(f'\n[load eval tfrecords]')
files = sorted(glob.glob(f'{EVAL_DIR}/partition_*.tfrecord.gz'))
dataset = tf.data.TFRecordDataset(files, compression_type='GZIP')
user_targets = {}
n_eval = 0
for raw in dataset:
    example = tf.train.Example()
    example.ParseFromString(raw.numpy())
    user_id = int(example.features.feature['user_id'].int64_list.value[0])
    seq = list(example.features.feature['sequence_data'].int64_list.value)
    if not seq:
        continue
    user_targets[user_id] = seq[-1]
    n_eval += 1
print(f'  users: {n_eval}')


def load_sid(sid_path):
    sid = torch.load(sid_path, weights_only=False)
    # Normalize to (n_items, n_layers)
    if sid.shape[0] < sid.shape[1]:
        return sid.T.numpy()
    return sid.numpy()


def load_pred(pred_path):
    return torch.load(pred_path, weights_only=False)


def evaluate_three(pred, sid_items, user_targets, k_values=(5, 10), seed=42):
    """Three tie-breaking modes for collision groups.

    sid_items: (n_items, n_layers) numpy
    pred: (n_users, K, n_layers) tensor

    For each (user, k):
      predicted_sid = pred[u, k]
      candidate_items = items in catalog with same SID prefix (full)
      if candidate_items is singleton -> unique item
      else -> collision group
    """
    n_users, K, n_layers = pred.shape
    sid_to_items = defaultdict(list)
    for item_id in range(sid_items.shape[0]):
        key = tuple(int(x) for x in sid_items[item_id].tolist())
        sid_to_items[key].append(item_id)

    # For each user × k, compute (rank_reranker, rank_random, rank_oracle, n_candidates)
    rerank_hits = {f'R@{k}': 0 for k in k_values}
    rand_hits = {f'R@{k}': 0 for k in k_values}
    oracle_hits = {f'R@{k}': 0 for k in k_values}
    rerank_ndcg = {f'NDCG@{k}': 0.0 for k in k_values}
    rand_ndcg = {f'NDCG@{k}': 0.0 for k in k_values}
    oracle_ndcg = {f'NDCG@{k}': 0.0 for k in k_values}

    # Track collision distribution
    n_unique_slots = 0
    n_collision_slots = 0
    np.random.seed(seed)

    for u in range(n_users):
        target = user_targets.get(u, -1)
        if target < 0:
            continue
        # For each k, find rank under three tie-breaking modes
        for mode, hits, ndcg in [
            ('reranker', rerank_hits, rerank_ndcg),
            ('random', rand_hits, rand_ndcg),
            ('oracle', oracle_hits, oracle_ndcg),
        ]:
            seen = 0  # how many slots we've consumed
            for k in range(K):
                key = tuple(int(x) for x in pred[u, k].tolist())
                candidates = sid_to_items.get(key, [-1])
                if len(candidates) == 1:
                    n_unique_slots += 1
                    chosen = candidates[0]
                else:
                    n_collision_slots += 1
                    if mode == 'reranker':
                        chosen = candidates[0]
                    elif mode == 'random':
                        chosen = np.random.choice(candidates)
                    else:  # oracle
                        chosen = target if target in candidates else candidates[0]
                seen += 1
                if chosen == target:
                    rank = k
                    for K_val in k_values:
                        if rank < K_val:
                            hits[f'R@{K_val}'] += 1
                            ndcg[f'NDCG@{K_val}'] += 1.0 / np.log2(rank + 2)
                    break

    n_users_eval = sum(1 for u in range(n_users) if user_targets.get(u, -1) >= 0)
    out = {}
    for k in k_values:
        for mode, hits in [('reranker', rerank_hits), ('random', rand_hits), ('oracle', oracle_hits)]:
            out[f'{mode}_R@{k}'] = hits[f'R@{k}'] / max(1, n_users_eval)
        for mode, ndcg in [('reranker', rerank_ndcg), ('random', rand_ndcg), ('oracle', oracle_ndcg)]:
            out[f'{mode}_NDCG@{k}'] = ndcg[f'NDCG@{k}'] / max(1, n_users_eval)
    out['n_users_eval'] = n_users_eval
    out['n_unique_slots'] = n_unique_slots
    out['n_collision_slots'] = n_collision_slots
    out['collision_rate'] = n_collision_slots / max(1, n_unique_slots + n_collision_slots)
    return out


configs = [
    ('L=4_baseline',
     f'{GRID}/logs/inference/runs/2026-07-07/20-17-35/pickle/merged_predictions_tensor.pt',
     f'{GRID}/logs/inference/runs/2026-07-07/21-47-35/pickle/merged_predictions_tensor.pt'),
    ('AQ_additive_s42',
     f'{GRID}/logs/inference/runs/task19_aq_s22/pickle/merged_predictions_tensor.pt',
     f'{GRID}/logs/inference/runs/task19_aq_s4_v2000/pickle/merged_predictions_tensor.pt'),
    ('HRQ_v2_s42',
     f'{GRID}/logs/inference/runs/task18_hrq_s22/pickle/merged_predictions_tensor.pt',
     f'{GRID}/logs/inference/runs/task18_hrq_s4_v2200/pickle/merged_predictions_tensor.pt'),
]

results = {}
for name, sid_path, pred_path in configs:
    print(f'\n--- {name} ---')
    sid_items = load_sid(sid_path)
    pred = load_pred(pred_path)
    print(f'  SID shape: {sid_items.shape}, Pred shape: {pred.shape}')
    out = evaluate_three(pred, sid_items, user_targets, seed=42)
    out['algorithm'] = name
    results[name] = out
    print(f'  reranker R@10: {out["reranker_R@10"]*100:.4f}%')
    print(f'  random   R@10: {out["random_R@10"]*100:.4f}%')
    print(f'  oracle   R@10: {out["oracle_R@10"]*100:.4f}%')
    print(f'  unique slots: {out["n_unique_slots"]}, collision slots: {out["n_collision_slots"]} '
          f'(CR={out["collision_rate"]*100:.2f}%)')

# Save
with open(os.path.join(OUT_DIR, 'collision_tiebreaking.json'), 'w') as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

verdict_lines = [
    '# Task A9 Verdict: Collision Group Tie-Breaking',
    '',
    '## 设计',
    '',
    '- reranker: 命中 collision group 时选排序第一个（当前模型行为）',
    '- random: 命中 collision group 时随机抽一个（理论下界）',
    '- oracle: 命中 collision group 时选 ground truth（理论上界）',
    '',
    '## Per-Algorithm',
    '',
    '| Algorithm | reranker R@10 | random R@10 | oracle R@10 | gap_oracle-reranker | gap_reranker-random | collision slots |',
    '|-----------|---------------|-------------|-------------|---------------------|---------------------|-----------------|',
]
for name, out in results.items():
    gap_oracle = (out['oracle_R@10'] - out['reranker_R@10']) * 100
    gap_rerand = (out['reranker_R@10'] - out['random_R@10']) * 100
    verdict_lines.append(
        f'| {name} | {out["reranker_R@10"]*100:.4f}% | {out["random_R@10"]*100:.4f}% | '
        f'{out["oracle_R@10"]*100:.4f}% | +{gap_oracle:.4f}% | +{gap_rerand:.4f}% | '
        f'{out["n_collision_slots"]} ({out["collision_rate"]*100:.2f}%) |'
    )

verdict_lines.extend([
    '',
    '## 解读',
    '',
    '- **gap_oracle - reranker = collision 内排序损失**：oracle 上界 vs reranker 实际值的差距。差距大 → collision 内 model 选得不对，tie-breaking 有损失',
    '- **gap_reranker - random = collision 内排序优势**：reranker vs random 的差距。≈0 → model 在 collision 内是随机的，没有学到 ordering；>0 → model 学到了 ordering',
    '- 若 reranker ≈ random → collision 对 R@10 没有贡献，碰撞带来的"任何命中"是假象',
    '- 若 oracle ≈ reranker 且 reranker > random → collision 提供额外信息，model 排序合理',
    '',
    '## 结论',
    '',
])
for name, out in results.items():
    verdict_lines.append(f'### {name}')
    verdict_lines.append(f'- reranker R@10 = {out["reranker_R@10"]*100:.4f}%')
    verdict_lines.append(f'- random R@10 = {out["random_R@10"]*100:.4f}%')
    verdict_lines.append(f'- oracle R@10 = {out["oracle_R@10"]*100:.4f}%')
    verdict_lines.append(f'- collision slots = {out["n_collision_slots"]} ({out["collision_rate"]*100:.2f}%)')
    verdict_lines.append('')

with open(os.path.join(OUT_DIR, 'verdict.md'), 'w') as f:
    f.write('\n'.join(verdict_lines))

print(f'\n=== Summary ===')
print(f'产物: {OUT_DIR}/collision_tiebreaking.json + verdict.md')
for name, out in results.items():
    print(f'  {name}: reranker={out["reranker_R@10"]*100:.4f}% | '
          f'random={out["random_R@10"]*100:.4f}% | oracle={out["oracle_R@10"]*100:.4f}%')