#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Task #89 — Stage 0: Block-wise stress grid search.

Reuses Task #84 HRQ-VAE ckpt. For each of 4 residual layers (L0 raw, L1, L2, L3 final),
splits the 32-dim latent into M blocks:
  M=1 → 32 (whole)
  M=2 → 16+16
  M=3 → 11+11+10

For each block, runs stress grid on Task #82 B refined κ grid [0, -0.05, ..., -2.0]
to check whether sub-blocks learn a different κ than the whole.

If ALL blocks' best κ stay near 0 → strong signal that even sub-block decomposition
cannot find hyperbolic structure (likely NO-GO for Stage 1).
If ANY block's best κ is significantly negative → sub-block structure differs from
whole, supports Stage 1 R2 hypothesis.
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
from task80_stage1c_true_distortion import compute_true_distortion


# Task #82 B refined κ grid
KAPPA_GRID_REFINED = [
    0.0, -0.05, -0.10, -0.15, -0.20, -0.25, -0.30, -0.50, -1.00, -1.50, -2.00,
]


def split_into_blocks(vec: np.ndarray, M: int) -> list:
    """Split vec (..., 32) into M contiguous blocks of size floor(32/M) or floor+1.

    M=1 → [32]
    M=2 → [16, 16]
    M=3 → [11, 11, 10] (32 = 11+11+10)
    """
    n = vec.shape[-1]
    assert n == 32, f"Expected 32-dim, got {n}"
    base = n // M
    rem = n % M
    blocks = []
    offset = 0
    for m in range(M):
        sz = base + (1 if m < rem else 0)
        blocks.append(vec[..., offset:offset + sz])
        offset += sz
    return blocks


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt', required=True)
    p.add_argument('--emb', required=True)
    p.add_argument('--device', default='cpu', help='Stage 0 = CPU only')
    p.add_argument('--M_list', type=int, nargs='+', default=[1, 2, 3])
    p.add_argument('--n_subset', type=int, default=300)
    p.add_argument('--n_iter', type=int, default=200)
    p.add_argument('--lr', type=float, default=0.005)
    p.add_argument('--d_mds', type=int, default=32)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--batch_size', type=int, default=512)
    p.add_argument('--output', required=True)
    return p.parse_args()


def main():
    args = parse_args()
    print(f"[Task #89 Stage 0] Block-wise stress grid (per-layer × M-blocks)")
    print(f"  ckpt: {args.ckpt}")
    print(f"  M_list: {args.M_list}")
    print(f"  κ grid: {KAPPA_GRID_REFINED}")
    print(f"  device: {args.device} (CPU only — Stage 0 no training)")

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
        print(f"  loaded {embeddings_np.shape[0]} embeddings (dim={embeddings_np.shape[1]})")
    else:
        raise ValueError(f"Unknown embedding format: {args.emb}")

    # Load HRQ-VAE
    model, model_args = load_hrqvae(args.ckpt, args.device)
    num_layers = len(model_args['num_emb_list'])
    curvatures = model_args['curvatures']
    print(f"  num codebook layers: {num_layers}")
    print(f"  num_emb_list: {model_args['num_emb_list']}")
    print(f"  per-layer curvatures: {curvatures}")

    # Extract per-layer residuals
    residuals, assignments, codewords = extract_per_layer_residuals(
        model, embeddings_np, args.device, args.batch_size, use_sk=False,
    )
    print(f"  residual shapes: {[r.shape for r in residuals]}")

    # Main loop: per layer × per M × per block × per κ
    layer_results = []
    for layer_idx, r in enumerate(residuals):
        if layer_idx == 0:
            label = 'L0_raw_encoded_z'
        elif layer_idx < num_layers:
            label = f'L{layer_idx}_pre_quant_input'
        else:
            label = f'L{num_layers}_final_residual'
        norm_mean = r.norm(dim=-1).mean().item()
        print(f"\n[Layer {layer_idx}] {label}, ||r||={norm_mean:.4f}, shape={tuple(r.shape)}")

        r_np = r.detach().cpu().numpy().astype(np.float32)
        if len(r_np) > args.n_subset:
            rng = np.random.RandomState(args.seed)
            idx = rng.choice(len(r_np), size=args.n_subset, replace=False)
            r_subset = r_np[idx]
        else:
            r_subset = r_np

        layer_data = {
            'layer': layer_idx,
            'label': label,
            'norm_mean': norm_mean,
            'M_results': [],
        }

        for M in args.M_list:
            blocks = split_into_blocks(r_subset, M)
            print(f"  [M={M}] splitting 32 → {[b.shape[-1] for b in blocks]}")

            block_results = []
            for block_idx, block in enumerate(blocks):
                # Each block: (n_subset, block_dim)
                block_results_block = []
                for k in KAPPA_GRID_REFINED:
                    t0 = time.time()
                    try:
                        # Each block runs MDS in its OWN block_dim space (d_mds default=32 still,
                        # but the residual block is block_dim-dim; MDS embeds in d=block_dim
                        # to avoid degeneracy)
                        d_use = block.shape[-1]  # use block_dim
                        stress = compute_true_distortion(
                            block, kappa=k, n_subset=len(block), seed=args.seed,
                            n_iter=args.n_iter, lr=args.lr, d=d_use, device=args.device,
                        )
                        elapsed = time.time() - t0
                        block_results_block.append({
                            'kappa': k, 'stress': stress, 'elapsed_sec': elapsed,
                        })
                    except Exception as e:
                        block_results_block.append({
                            'kappa': k, 'stress': float('inf'),
                            'error': str(e)[:100], 'elapsed_sec': time.time() - t0,
                        })

                valid = [x for x in block_results_block if x['stress'] == x['stress'] and x['stress'] != float('inf')]
                if valid:
                    best = min(valid, key=lambda x: x['stress'])
                    best_kappa = best['kappa']
                    best_stress = best['stress']
                    zero_stress = next((x['stress'] for x in valid if x['kappa'] == 0.0), float('inf'))
                    margin = zero_stress / best_stress if best_stress > 0 else float('inf')
                else:
                    best_kappa = None
                    best_stress = float('inf')
                    zero_stress = float('inf')
                    margin = float('inf')

                print(f"    [block {block_idx}] dim={block.shape[-1]}, best κ*={best_kappa}, "
                      f"stress(best)={best_stress:.4f}, margin={margin:.2f}×")

                block_results.append({
                    'block_idx': block_idx,
                    'block_dim': int(block.shape[-1]),
                    'kappa_grid': KAPPA_GRID_REFINED,
                    'kappa_results': block_results_block,
                    'best_kappa': best_kappa,
                    'best_stress': best_stress,
                    'zero_stress': zero_stress,
                    'margin_vs_zero': margin,
                })

            layer_data['M_results'].append({
                'M': M,
                'block_dims': [b.shape[-1] for b in blocks],
                'block_results': block_results,
            })

        layer_results.append(layer_data)

    # Aggregate
    output = {
        'task': 'task89_stage0_block_stress',
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'ckpt': args.ckpt,
        'embedding_file': args.emb,
        'num_layers': num_layers,
        'num_emb_list': model_args['num_emb_list'],
        'per_layer_curvatures': curvatures,
        'M_list': args.M_list,
        'kappa_grid': KAPPA_GRID_REFINED,
        'n_subset': args.n_subset,
        'n_iter': args.n_iter,
        'lr': args.lr,
        'd_mds': args.d_mds,
        'seed': args.seed,
        'device': args.device,
        'layers': layer_results,
        'stage0_decision_signal': (
            'IF all (layer, M, block) best κ* are 0.0 across the board → '
            'strong evidence NO-GO for Stage 1 (sub-block decomposition cannot '
            'find hyperbolic structure on Musical_Instruments). '
            'IF any block has best κ* significantly negative (≤ -0.20) → '
            'sub-block structure differs from whole, supports Stage 1 R2.'
        ),
    }
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\n[Saved] {args.output}")

    # Summary table
    print(f"\n=== Stage 0 Summary (ckpt curvatures={curvatures}) ===")
    print(f"{'Layer':<25} {'M':>3} {'Block':>5} {'Dim':>4} {'best κ*':>10} {'stress(0)':>10} {'margin':>10}")
    print("-" * 80)
    for ld in layer_results:
        for mr in ld['M_results']:
            for br in mr['block_results']:
                print(f"{ld['label'][:24]:<25} {mr['M']:>3} {br['block_idx']:>5} {br['block_dim']:>4} "
                      f"{br['best_kappa']!s:>10} {br['zero_stress']:>10.4f} {br['margin_vs_zero']:>9.2f}×")


if __name__ == '__main__':
    main()