#!/usr/bin/env python3
"""Task #50 战线一 S6 — item2vec (行为共现, 无 margin)

目标: 从 user-item 行为序列共现训出 embedding, 区别于 MCKG 的 margin ranking.
  - 用 Toys 的训练序列 (从 user_history 抽)
  - SGNS (Skip-Gram with Negative Sampling) 或简化版 pairwise co-occurrence
  - 输出 64d embedding, 无几何约束

数据: data/amazon_data/toys/diag_test/*.tfrecord (user-item 交互)
"""
import glob
import json
import os
import sys
import random
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path('/home/wlia0047/arenyu/GeneRec') if False else Path('/home/wlia0047/ar57/wenyu/GeneRec')
sys.path.insert(0, str(ROOT))
OUT_DIR = ROOT / 'products/task158_s6_item2vec'
OUT_DIR.mkdir(exist_ok=True)
DEVICE = 'cuda:0'
SEED = 42
EMBED_DIM = 64
WINDOW = 5
NEG_K = 5
BATCH = 4096
N_STEPS = 5000
LR = 1e-3

# 从 Toys tfrecord 抽 user-item pairs
TFRECORD_GLOB = str(ROOT / 'data/amazon_data/toys/diag_test/partition_0.tfrecord.gz')


def load_interactions(max_n=50000):
    """从 partition tfrecord 抽 (item_id, item_id) co-occurrence pairs (基于 user history)."""
    import tensorflow as tf
    files = sorted(glob.glob(TFRECORD_GLOB))
    print(f'  partition files: {len(files)}')
    sequences = []
    ds = tf.data.TFRecordDataset(files, compression_type='GZIP')
    for i, raw in enumerate(ds.take(max_n)):
        ex = tf.train.Example()
        ex.ParseFromString(raw.numpy())
        feats = ex.features.feature
        items = None
        for key in ['sequence_data', 'items', 'item_ids', 'sequence']:
            if key in feats:
                items = list(feats[key].int64_list.value)
                break
        if items is None or len(items) < 2:
            continue
        sequences.append(items)
    print(f'  loaded {len(sequences)} user sequences')
    return sequences


def build_pairs(sequences, window=WINDOW, max_pairs=2000000):
    """从序列构造 co-occurrence pairs."""
    pairs = []
    rng = random.Random(SEED)
    for seq in sequences:
        if len(seq) < 2:
            continue
        for i in range(len(seq)):
            lo = max(0, i - window)
            hi = min(len(seq), i + window + 1)
            for j in range(lo, hi):
                if i == j:
                    continue
                pairs.append((seq[i], seq[j]))
                if len(pairs) >= max_pairs:
                    return pairs
    print(f'  total pairs: {len(pairs)}')
    return pairs


class Item2Vec(nn.Module):
    def __init__(self, n_items, embed_dim=64):
        super().__init__()
        self.in_emb = nn.Embedding(n_items, embed_dim)
        self.out_emb = nn.Embedding(n_items, embed_dim)
        nn.init.uniform_(self.in_emb.weight, -0.5/embed_dim, 0.5/embed_dim)
        nn.init.zeros_(self.out_emb.weight)

    def forward(self, center, context, neg):
        v_c = self.in_emb(center)  # (B, d)
        v_o = self.out_emb(context)  # (B, d)
        v_n = self.out_emb(neg)  # (B, K, d)
        pos_score = (v_c * v_o).sum(dim=-1)
        neg_score = (v_n * v_c.unsqueeze(1)).sum(dim=-1)
        # SGNS loss
        loss = -F.logsigmoid(pos_score).mean() - F.logsigmoid(-neg_score).mean()
        return loss


def train_i2v(model, pairs, n_items, n_steps=N_STEPS, batch_size=BATCH, neg_k=NEG_K, lr=LR, tag=''):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    N = len(pairs)
    pairs_arr = np.array(pairs, dtype=np.int64)
    trajectory = {'step': [], 'loss': []}
    for step in range(n_steps):
        idx = np.random.randint(0, N, batch_size)
        center = torch.from_numpy(pairs_arr[idx, 0]).to(DEVICE)
        context = torch.from_numpy(pairs_arr[idx, 1]).to(DEVICE)
        neg = torch.randint(0, n_items, (batch_size, neg_k)).to(DEVICE)
        loss = model(center, context, neg)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step % 500 == 0 or step == n_steps - 1:
            trajectory['step'].append(step)
            trajectory['loss'].append(loss.item())
            print(f'  [{tag}] step {step}: loss={loss.item():.4f}')
    return trajectory


def main():
    print('========== 战线一 S6 item2vec 训练 ==========')
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)

    sequences = load_interactions()
    if not sequences:
        print('No interactions loaded, aborting')
        return

    pairs = build_pairs(sequences)
    if not pairs:
        print('No pairs built, aborting')
        return

    n_items = max(max(p) for p in pairs) + 1
    n_items = max(n_items, 11924)  # 至少 11924 个 items
    print(f'  n_items (max id+1): {n_items}')
    print(f'  n_pairs: {len(pairs)}')

    model = Item2Vec(n_items=n_items, embed_dim=EMBED_DIM).to(DEVICE)
    traj = train_i2v(model, pairs, n_items, tag='S6-i2v')

    # 提取 embedding
    print()
    print('Extracting item embeddings...')
    in_w = model.in_emb.weight.detach().cpu().numpy()[:11924]  # 取 Toys 数量
    out_w = model.out_emb.weight.detach().cpu().numpy()[:11924]
    # SGNS 经典用法: input + output 平均
    z = (in_w + out_w) / 2
    z_norm = z / np.linalg.norm(z, axis=-1, keepdims=True).clip(min=1e-6)

    out = OUT_DIR / 'entity_embedding.pt'
    torch.save({
        'subspace_item': z_norm.reshape(1, z_norm.shape[0], z_norm.shape[1]),
        'fused_item': z_norm,
        'method': 'S6-item2vec',
        'config': {'embed_dim': EMBED_DIM, 'window': WINDOW, 'neg_k': NEG_K, 'n_steps': N_STEPS},
        'trajectory': traj,
    }, out)
    print(f'  saved: {out}')

    n = np.linalg.norm(z_norm, axis=-1)
    print(f'  embedding norm: mean={n.mean():.3f}, max={n.max():.3f}, '
          f'p99={np.percentile(n, 99):.3f}')

    summary = {
        'task': 'Task #50 战线一 S6 item2vec 训练',
        'config': {'embed_dim': EMBED_DIM, 'window': WINDOW, 'neg_k': NEG_K, 'n_steps': N_STEPS,
                   'lr': LR, 'batch': BATCH},
        'data': {
            'n_sequences': len(sequences),
            'n_pairs': len(pairs),
            'n_items': n_items,
        },
        'norm_stats': {
            'mean': float(n.mean()),
            'max': float(n.max()),
            'p99': float(np.percentile(n, 99)),
        },
        'final_loss': traj['loss'][-1],
        'trajectory': traj,
    }
    with open(OUT_DIR / 'task158_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    print(f'  summary: {OUT_DIR / "task158_summary.json"}')


if __name__ == '__main__':
    main()