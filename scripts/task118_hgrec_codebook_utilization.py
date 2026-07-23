#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Task #118 — HG-Rec codebook utilization / collision rate.

For both ckpt versions (Task #84 c=[1,1,1] vs Task #88 c=[0.5,0.5,0.5]),
quantize all item embeddings and measure:

  Per-layer utilization:
    U_i = |unique(idx_i)| / |codebook_i|     (i = 0,1,2)

  Collision rate (final SID collision):
    sid = concat(idx_0, idx_1, idx_2, idx_3)  (4-tuple per item)
    collision_rate = (N_items - |unique sid tuples|) / N_items

  Per-layer effective cardinality:
    log2(|unique idx_i|) — info bits preserved by layer

  Full SID effective cardinality:
    log2(|unique sid tuples|)

These are HG-Rec paper-reported metrics (Section 5 / Table 4).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, '/home/wlia0047/ar57/wenyu/GeneRec/HG-Rec')
sys.path.insert(0, '/home/wlia0047/ar57/wenyu/GeneRec/scripts')

from task116_hgrec_delta_per_layer import load_hrqvae, extract_per_layer_residuals


def compute_utilization(assignments, codebook_sizes):
    """For each layer i:
       U_i = unique(idx_i).numel() / codebook_sizes[i]
       Returns per-layer dict + collision rate (4-tuple).
    """
    num_layers = len(assignments)
    per_layer = []
    for i in range(num_layers):
        idx = assignments[i]
        if hasattr(idx, 'cpu'):
            idx = idx.cpu()
        idx_np = idx.numpy() if hasattr(idx, 'numpy') else np.asarray(idx)
        n_unique = int(np.unique(idx_np).size)
        util = n_unique / codebook_sizes[i]
        per_layer.append({
            'layer': i,
            'codebook_size': codebook_sizes[i],
            'unique_used': n_unique,
            'utilization': util,
            'log2_cardinality': math.log2(max(n_unique, 1)),
        })

    # Collision rate via 4-tuple (or N-tuple) sid
    sid = np.stack([a.cpu().numpy() if hasattr(a, 'cpu') else np.asarray(a) for a in assignments], axis=1)
    n_items = sid.shape[0]
    # Use view as 1D structured (each row -> bytes)
    sid_view = np.ascontiguousarray(sid).view(np.dtype((np.void, sid.dtype.itemsize * sid.shape[1])))
    _, counts = np.unique(sid_view, return_counts=True)
    n_unique_tuples = int(counts.size)
    n_collisions = int(n_items - n_unique_tuples)
    collision_rate = n_collisions / n_items

    # Max-collision items (top bucket)
    if counts.size > 0:
        max_bucket = int(counts.max())
        top10_bucket = int(np.sort(counts)[-min(10, counts.size):].sum())
    else:
        max_bucket = 0
        top10_bucket = 0

    return {
        'per_layer': per_layer,
        'sid_collision': {
            'n_items': n_items,
            'n_unique_sids': n_unique_tuples,
            'n_collisions': n_collisions,
            'collision_rate': collision_rate,
            'max_bucket_items': max_bucket,
            'top10_bucket_items': top10_bucket,
            'log2_sid_cardinality': math.log2(max(n_unique_tuples, 1)),
        },
    }


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt', required=True)
    p.add_argument('--emb', required=True)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--label', required=True)
    p.add_argument('--batch_size', type=int, default=512)
    p.add_argument('--output', required=True)
    return p.parse_args()


def main():
    args = parse_args()
    print(f"[Task #118] HG-Rec codebook utilization — {args.label}")
    print(f"  ckpt: {args.ckpt}")
    print(f"  embedding: {args.emb}")

    # Load embedding
    if args.emb.endswith('.parquet'):
        import pandas as pd
        df = pd.read_parquet(args.emb)
        emb_col = None
        for c in ['embedding', 'emb', 'features', 'vec']:
            if c in df.columns:
                emb_col = c
                break
        if emb_col is None:
            emb_col = df.columns[0]
        embeddings_np = np.stack(df[emb_col].values)
        print(f"  loaded {embeddings_np.shape[0]} embeddings from parquet col '{emb_col}'")
    else:
        raise ValueError(f"Unknown embedding format: {args.emb}")

    # Load HRQ-VAE
    model, model_args = load_hrqvae(args.ckpt, args.device)
    codebook_sizes = model_args['num_emb_list']
    print(f"  codebook sizes: {codebook_sizes}")
    print(f"  curvatures: {model_args['curvatures']}")

    # Extract per-layer assignments
    residuals, assignments, codewords = extract_per_layer_residuals(
        model, embeddings_np, args.device, args.batch_size, use_sk=False,
    )
    print(f"  assignments shapes: {[a.shape for a in assignments]}")

    # Compute utilization + collision
    metrics = compute_utilization(assignments, codebook_sizes)

    # Save
    output = {
        'task': 'task118_hgrec_codebook_utilization',
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'label': args.label,
        'ckpt': args.ckpt,
        'embedding_file': args.emb,
        'codebook_sizes': codebook_sizes,
        'curvatures': model_args['curvatures'],
        'num_items': int(embeddings_np.shape[0]),
        'embedding_dim': int(embeddings_np.shape[1]),
        'metrics': metrics,
    }
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\n[Saved] {args.output}")

    # Summary
    print(f"\n=== Summary ({args.label}) ===")
    for pl in metrics['per_layer']:
        print(f"  Layer {pl['layer']}: {pl['unique_used']:>5}/{pl['codebook_size']:<5} "
              f"= {pl['utilization']*100:.2f}% util, "
              f"log2(cardinality)={pl['log2_cardinality']:.2f} bits")
    sc = metrics['sid_collision']
    print(f"\n  Full SID collision:")
    print(f"    N items: {sc['n_items']}")
    print(f"    N unique SID tuples: {sc['n_unique_sids']}")
    print(f"    Collision rate: {sc['collision_rate']*100:.2f}%")
    print(f"    log2(SID cardinality): {sc['log2_sid_cardinality']:.2f} bits")
    print(f"    Max bucket: {sc['max_bucket_items']} items/SID, top10: {sc['top10_bucket_items']}")


if __name__ == '__main__':
    main()