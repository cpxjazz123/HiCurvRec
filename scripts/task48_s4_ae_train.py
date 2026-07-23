#!/usr/bin/env python3
"""Task #48 战线一 S4 — AE 训练 (纯重构目标)

目标: 训出与 QMP 度量结构匹配的纯重构 embedding
  - Encoder: input_dim → 64
  - Decoder: 64 → input_dim
  - Loss: MSE(x, decoder(encoder(x)))
  - 不使用 margin / contrastive / geometric — 纯度量重建目标

数据: data/amazon_data/toys/items/data_*.tfrecord.gz (text: Title+Brand+Categories+Price)
输出: entity_embedding.pt (S4)  shape (N, 64), L2-normalized
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

ROOT = Path('/home/wlia0047/ar57/wenyu/GeneRec')
sys.path.insert(0, str(ROOT))
OUT_DIR = ROOT / 'products/task156_s4_ae'
OUT_DIR.mkdir(exist_ok=True)
DEVICE = 'cuda:0'
SEED = 42
EMBED_DIM = 64
HIDDEN = 256
BATCH = 512
N_STEPS = 5000
LR = 1e-3
VOCAB_SIZE = 4096

TFRECORD_GLOB = str(ROOT / 'data/amazon_data/toys/items/data_*.tfrecord.gz')


def load_texts_from_tfrecord(max_n=12000):
    """从 tfrecord 抽 item text (Title + Brand + Categories + Price)."""
    import tensorflow as tf
    files = sorted(glob.glob(TFRECORD_GLOB))
    print(f'  tfrecord files: {len(files)}')
    texts = []
    ds = tf.data.TFRecordDataset(files, compression_type='GZIP')
    for raw in ds.take(max_n):
        ex = tf.train.Example()
        ex.ParseFromString(raw.numpy())
        text_bytes = ex.features.feature['text'].bytes_list.value[0]
        texts.append(text_bytes.decode('utf-8', errors='ignore').strip())
    return texts


def text_to_bow(texts, vocab_size=VOCAB_SIZE):
    """简易 bag-of-words."""
    cnt = Counter()
    for t in texts:
        words = t.lower().split()
        cnt.update(words)
    vocab = [w for w, _ in cnt.most_common(vocab_size)]
    word2id = {w: i for i, w in enumerate(vocab)}
    bow = np.zeros((len(texts), vocab_size), dtype=np.float32)
    for i, t in enumerate(texts):
        for w in t.lower().split():
            if w in word2id:
                bow[i, word2id[w]] += 1.0
    norm = np.linalg.norm(bow, axis=-1, keepdims=True).clip(min=1e-6)
    bow = bow / norm
    return bow, vocab


class AE(nn.Module):
    def __init__(self, in_dim, hidden=256, embed_dim=64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, embed_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(embed_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, in_dim),
        )

    def forward(self, x):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return z, x_hat


def train_ae(model, x, n_steps=N_STEPS, batch_size=BATCH, lr=LR, tag=''):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    N = x.shape[0]
    trajectory = {'step': [], 'recon': []}
    for step in range(n_steps):
        idx = torch.randint(0, N, (batch_size,))
        batch = x[idx]
        z, x_hat = model(batch)
        loss = F.mse_loss(x_hat, batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step % 500 == 0 or step == n_steps - 1:
            trajectory['step'].append(step)
            trajectory['recon'].append(loss.item())
            print(f'  [{tag}] step {step}: recon={loss.item():.4f}')
    return trajectory


def main():
    print('========== 战线一 S4 AE 训练 ==========')
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)

    texts = load_texts_from_tfrecord()
    print(f'  loaded {len(texts)} texts')
    if texts:
        print(f'  example: {texts[0][:100]}')

    bow, vocab = text_to_bow(texts)
    print(f'  bow shape: {bow.shape}, mean: {bow.mean():.4f}')

    x = torch.from_numpy(bow).float().to(DEVICE)
    model = AE(in_dim=bow.shape[1], hidden=HIDDEN, embed_dim=EMBED_DIM).to(DEVICE)
    traj = train_ae(model, x, tag='S4-AE')

    print()
    print('Extracting embeddings...')
    model.eval()
    with torch.no_grad():
        z_list = []
        for i in range(0, len(x), 1024):
            z, _ = model(x[i:i+1024])
            z_list.append(z)
        z_all = torch.cat(z_list, dim=0).cpu().numpy()

    z_norm = z_all / np.linalg.norm(z_all, axis=-1, keepdims=True).clip(min=1e-6)

    out = OUT_DIR / 'entity_embedding.pt'
    torch.save({
        'subspace_item': z_norm.reshape(1, z_norm.shape[0], z_norm.shape[1]),
        'fused_item': z_norm,
        'method': 'S4-AE',
        'config': {'embed_dim': EMBED_DIM, 'hidden': HIDDEN, 'n_steps': N_STEPS, 'vocab_size': VOCAB_SIZE},
        'trajectory': traj,
    }, out)
    print(f'  saved: {out}')

    n = np.linalg.norm(z_norm, axis=-1)
    print(f'  embedding norm: mean={n.mean():.3f}, max={n.max():.3f}, '
          f'p99={np.percentile(n, 99):.3f}')

    summary = {
        'task': 'Task #48 战线一 S4 AE 训练',
        'config': {'embed_dim': EMBED_DIM, 'hidden': HIDDEN, 'n_steps': N_STEPS, 'lr': LR, 'batch': BATCH,
                   'vocab_size': VOCAB_SIZE},
        'data': {
            'n_items': len(texts),
            'vocab_size': bow.shape[1],
        },
        'norm_stats': {
            'mean': float(n.mean()),
            'max': float(n.max()),
            'p99': float(np.percentile(n, 99)),
        },
        'final_recon': traj['recon'][-1],
        'trajectory': traj,
    }
    with open(OUT_DIR / 'task156_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    print(f'  summary: {OUT_DIR / "task156_summary.json"}')


if __name__ == '__main__':
    main()