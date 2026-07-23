#!/usr/bin/env python3
"""Task #78 — phonism RQ-VAE Gromov δ-hyperbolicity measurement.

Reuses Task #92's algorithm (`gromov_delta_4point`) with phonism RQ-VAE architecture.
Comparison: Task #78 (e_dim=32, SINKHORN last) vs Task #92 (e_dim=128, pure VQ).

Usage:
    python scripts/task79_phonism_delta_hyperbolicity.py \
        --ckpt products/task79_phonism_rqvae_768d/rqvae_ckpt/<TS>/best_loss_model.pth \
        --emb external/LLM-RecSys-ID/data/instruments/sentence-t5-768d.npy \
        --num_samples 5000 \
        --device cuda:0 \
        --output verdicts/task79_phonism_rqvae_delta_hyperbolicity.json
"""
import argparse
import json
import os
import sys
import time
import numpy as np
import torch

# Reuse Task #92 algorithm
sys.path.insert(0, '/home/wlia0047/ar57/wenyu/GeneRec/scripts')
from task92_rqvae_delta_hyperbolicity import (
    load_rqvae,
    extract_per_layer_residuals,
    gromov_delta_4point,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt', required=True)
    p.add_argument('--emb', required=True)
    p.add_argument('--num_samples', type=int, default=5000)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--output', required=True)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--batch_size', type=int, default=2048)
    return p.parse_args()


def main():
    args = parse_args()
    print(f"[Task #78] phonism RQ-VAE δ-hyperbolicity measurement")
    print(f"  ckpt: {args.ckpt}")
    print(f"  embedding: {args.emb}")
    print(f"  num_samples: {args.num_samples}")
    print(f"  device: {args.device}")

    embeddings_np = np.load(args.emb)
    print(f"  embedding shape: {embeddings_np.shape}")

    model, model_args = load_rqvae(args.ckpt, args.device)
    num_layers = len(model_args.num_emb_list)
    print(f"  num codebook layers: {num_layers}")
    print(f"  num_emb_list: {model_args.num_emb_list}")
    print(f"  e_dim: {model_args.e_dim}")
    print(f"  sk_epsilons (inferred): {[0.0 if i < num_layers - 1 else 0.003 for i in range(num_layers)]}")

    residuals, assignments = extract_per_layer_residuals(
        model, embeddings_np, args.device, args.batch_size
    )
    print(f"  residual shapes: {[r.shape for r in residuals]}")

    # Compute δ per layer
    layer_results = []
    for i, r in enumerate(residuals):
        if i == 0:
            label = "raw_encoded"
        else:
            label = f"residual_after_layer_{i-1}"
        print(f"\n[Layer {i}/{num_layers}] {label}, shape={tuple(r.shape)}, norm_mean={r.norm(dim=-1).mean().item():.4f}")
        t0 = time.time()
        result = gromov_delta_4point(r, num_samples=args.num_samples, seed=args.seed)
        elapsed = time.time() - t0

        # Compute codebook utilization (number of unique assignments)
        # For i=0 (raw encoded), no codebook assigned yet.
        # For i>=1, codebook (i-1) is the one that produced the residual.
        assignments_i = assignments[i - 1] if i >= 1 else None
        if assignments_i is not None:
            unique = torch.unique(assignments_i).numel()
            codebook_size = model_args.num_emb_list[i - 1]
            utilization = unique / codebook_size
        else:
            utilization = None
            unique = 0

        result['layer'] = i
        result['label'] = label
        result['elapsed_sec'] = elapsed
        result['residual_norm_mean'] = float(r.norm(dim=-1).mean())
        result['residual_norm_std'] = float(r.norm(dim=-1).std())
        result['codebook_size'] = model_args.num_emb_list[i - 1] if i >= 1 else None
        result['codebook_unique'] = unique
        result['codebook_utilization'] = utilization
        layer_results.append(result)
        print(f"  δ_max={result['delta_max']:.4f}, δ_95={result['delta_95']:.4f}, δ_median={result['delta_median']:.4f}, diameter={result['diameter_approx']:.4f}, codebook_util={utilization if utilization is not None else 'N/A'} ({elapsed:.1f}s)")

    output = {
        'task': 'task79_phonism_rqvae_delta_hyperbolicity',
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'comparison': {
            'task92_vanilla': {
                'e_dim': 128,
                'sk_epsilons': [0, 0, 0],
                'embedding': 'Musical_Instruments SASRec 128d',
                'delta_max_per_layer': [5.39, 4.93, 3.70, 2.74],
            },
            'task79_phonism': {
                'e_dim': 32,
                'sk_epsilons': [0, 0, 0.003],
                'embedding': 'Musical_Instruments sentence-t5 768d',
            },
        },
        'ckpt': args.ckpt,
        'embedding_file': args.emb,
        'num_samples': args.num_samples,
        'num_layers': num_layers,
        'num_emb_list': model_args.num_emb_list,
        'e_dim': model_args.e_dim,
        'seed': args.seed,
        'layers': layer_results,
    }

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\n[Task #78] Result saved to {args.output}")
    print(f"\n=== Summary ===")
    for r in layer_results:
        util = r.get('codebook_utilization')
        util_str = f", codebook_util={util:.3f}" if util is not None else ""
        print(f"  Layer {r['layer']:>2} ({r['label']:<35}): δ_max={r['delta_max']:.4f}, δ_95={r['delta_95']:.4f}{util_str}")


if __name__ == '__main__':
    main()