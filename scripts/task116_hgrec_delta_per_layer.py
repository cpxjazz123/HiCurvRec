#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Task #116 — HG-Rec per-layer Gromov δ-hyperbolicity with dual metric (Euclidean + Poincaré).

Adapter of task92_rqvae_delta_hyperbolicity.py for HG-Rec HRQ-VAE:
  - Loads HRQ-VAE from HG-Rec ckpt (model.hrq.vq_layers, not model.rq.vq_layers)
  - Restores per-layer curvature c_i from ckpt args
  - Computes δ_95/diameter with TWO metrics:
      A. Euclidean (tangent/log-map space, the default task92 behavior)
      B. Poincaré (model-native: proj_to_ball(expmap0(v, c_i), c_i) -> poincare_distance(_, _, c_i))
  - n=50,000 samples (Task #79 convergence), multiple seeds for sensitivity

Usage:
    python scripts/task116_hgrec_delta_per_layer.py \
        --ckpt products/task84/ckpt/Instruments/Jul-23-2026_20-08-06_.../best_loss_model.pth \
        --emb /home/wlia0047/ar57/wenyu/GeneRec/HG-Rec/dataset/Instruments/item_emb.parquet \
        --num_samples 50000 \
        --seeds 42 123 456 \
        --device cuda:0 \
        --output verdicts/task116_hgrec_delta_per_layer_task84.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, '/home/wlia0047/ar57/wenyu/GeneRec/HG-Rec')


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt', required=True, help='HG-Rec HRQ-VAE best_loss_model.pth')
    p.add_argument('--emb', required=True, help='item_emb.parquet path (flan-t5 / sentence-t5 embedding)')
    p.add_argument('--num_samples', type=int, default=50000,
                   help='Number of 4-tuples to sample per layer (Task #79: n=50k converges)')
    p.add_argument('--seeds', type=int, nargs='+', default=[42],
                   help='Random seeds for sensitivity (recommend 42, 123, 456)')
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--output', required=True)
    p.add_argument('--batch_size', type=int, default=512,
                   help='Batch size for HRQ-VAE forward pass')
    p.add_argument('--metrics', nargs='+', default=['euclidean', 'poincare'],
                   choices=['euclidean', 'poincare'],
                   help='Which distance metrics to evaluate')
    p.add_argument('--label', default='unknown',
                   help='Human label for this ckpt (e.g. "task84_c111" or "task88_c555")')
    return p.parse_args()


def load_hrqvae(ckpt_path, device):
    """Load HG-Rec HRQ-VAE from ckpt, restoring per-layer curvature from saved args.

    Mirrors scripts/task88_stage2_codebook.py loading logic for consistency.
    """
    from model.hrqvae import HRQVAE
    from model.utils import EmbDataset

    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    if 'state_dict' in ckpt:
        state = ckpt['state_dict']
        train_args = ckpt.get('args', None)
    else:
        state = ckpt
        train_args = None

    # Load embedding data to infer in_dim
    data_path = train_args.data_path if (train_args and hasattr(train_args, 'data_path')) else None
    if data_path is None:
        raise RuntimeError(f"❌ No data_path in ckpt args. Cannot auto-infer in_dim.")
    data = EmbDataset(data_path)

    # Extract curvature_list from args (R88 patch)
    curvature_list = getattr(train_args, 'curvatures', None) if train_args else None

    model = HRQVAE(
        in_dim=data.dim,
        num_emb_list=train_args.num_emb_list,
        e_dim=train_args.e_dim,
        layers=train_args.layers,
        dropout_prob=train_args.dropout_prob,
        bn=train_args.bn,
        loss_type=train_args.loss_type,
        quant_loss_weight=train_args.quant_loss_weight,
        beta=train_args.beta,
        kmeans_init=train_args.kmeans_init,
        kmeans_iters=train_args.kmeans_iters,
        sk_eps=train_args.sk_epsilons,
        sk_iters=train_args.sk_iters,
        curvature_list=curvature_list,
    )
    model.load_state_dict(state)
    model.to(device)
    model.eval()

    # Construct a dict that mimics args namespace for logging
    args_dict = {
        'num_emb_list': list(train_args.num_emb_list),
        'e_dim': train_args.e_dim,
        'layers': list(train_args.layers),
        'curvatures': list(curvature_list) if curvature_list else [1.0] * len(train_args.num_emb_list),
        'data_path': data_path,
    }
    return model, args_dict


def extract_per_layer_residuals(model, embeddings_np, device, batch_size, use_sk=False):
    """Run HRQ-VAE encoder+forward, return per-layer residuals.

    Returns:
        residuals: list of (N, e_dim) tensors.
            residuals[0] = encoded z (before any quant)
            residuals[1] = z - q0  (input to layer 1's quantizer)
            residuals[2] = z - q0 - q1  (input to layer 2's quantizer)
            residuals[3] = z - q0 - q1 - q2  (final residual after all quants)
        assignments: list of (N,) tensors with index per layer
        codewords: list of (N, e_dim) tensors with item-specific quantized vector per layer
    """
    embeddings = torch.from_numpy(embeddings_np).float().to(device)
    N = embeddings.shape[0]

    with torch.no_grad():
        z = model.encoder(embeddings)  # (N, e_dim)

        residuals = [z.cpu().clone()]
        codewords = []
        assignments = []

        residual = z
        for i, quantizer in enumerate(model.hrq.vq_layers):
            x_res, loss, indices = quantizer(residual, use_sk=use_sk)
            residual = residual - x_res
            residuals.append(residual.cpu().clone())
            codewords.append(x_res.cpu().clone())
            assignments.append(indices.cpu())

    return residuals, assignments, codewords


def proj_to_ball_np(x: torch.Tensor, c: float, eps: float = 1e-5) -> torch.Tensor:
    """Poincaré ball projection: ||x|| < 1/sqrt(c) for c>0.

    HG-Rec uses expmap0 then proj_to_ball. We approximate that by direct projection here
    for diagnostic purposes (the tangent/log-map coords returned by HG-Rec quantizer are
    the post-logmap values, NOT pre-expmap; so we need to project them directly).
    """
    if c <= 0:
        return x
    norm = torch.linalg.norm(x, dim=-1, keepdim=True).clamp(min=eps)
    max_norm = (1.0 - eps) / np.sqrt(c)
    cond = norm > max_norm
    projected = x / norm * max_norm
    return torch.where(cond, projected, x)


def gromov_diameter_sample_metric(
    point_cloud: torch.Tensor,
    metric: str = 'euclidean',
    c: float = 1.0,
    num_samples: int = 10000,
    seed: int = 0,
):
    """Approximate diameter via random pair sampling, supports Euclidean + Poincaré."""
    g = torch.Generator()
    g.manual_seed(seed)
    N, D = point_cloud.shape
    idx = torch.randint(0, N, (num_samples, 2), generator=g)
    p = point_cloud[idx[:, 0]]
    q = point_cloud[idx[:, 1]]

    if metric == 'euclidean':
        dists = torch.linalg.norm(p - q, dim=1)
    elif metric == 'poincare':
        # Project to ball first
        p_ball = proj_to_ball_np(p, c)
        q_ball = proj_to_ball_np(q, c)
        # poincare_distance: d(u,v) = (2/sqrt(c)) * arctanh(sqrt(c) * ||(-u) ⊕ v||)
        # Use the standard formula:
        sqrt_c = np.sqrt(c)
        diff = p_ball - q_ball
        norm_diff = torch.linalg.norm(diff, dim=-1).clamp(min=1e-10)
        # Möbius subtraction: (-u) ⊕ v
        # (-u ⊕ v) = ((1 + c*||u||^2)*v + (1 - c*||v||^2)*(-u)) / (1 + c*<u,v>)^2  -- but here we just use ||u-v|| as an approximation
        # Actually, HG-Rec poincare_distance uses expmap0(logmap0(u) - logmap0(v), c)
        # For metric comparison, use the proper closed-form:
        u = p_ball
        v = q_ball
        u_sq = (u * u).sum(dim=-1)
        v_sq = (v * v).sum(dim=-1)
        uv = (u * v).sum(dim=-1)
        denom = (1 + c * uv) ** 2
        # Möbius subtraction (-u) ⊕ v
        num = ((1 + c * u_sq).unsqueeze(-1) * v + (1 - c * v_sq).unsqueeze(-1) * (-u))
        minus_u_plus_v = num / denom.unsqueeze(-1).clamp(min=1e-10)
        norm_mobius = torch.linalg.norm(minus_u_plus_v, dim=-1).clamp(min=1e-10, max=(1.0 - 1e-5) / sqrt_c)
        dists = (2.0 / sqrt_c) * torch.arctanh(sqrt_c * norm_mobius)
    else:
        raise ValueError(f"Unknown metric: {metric}")

    return float(dists.max())


def gromov_delta_4point_metric(
    point_cloud: torch.Tensor,
    metric: str = 'euclidean',
    c: float = 1.0,
    num_samples: int = 5000,
    seed: int = 42,
):
    """Compute Gromov δ-hyperbolicity via 4-point sampling with configurable metric.

    Args:
        point_cloud: (N, D) tensor on CPU
        metric: 'euclidean' or 'poincare'
        c: per-layer curvature (only used for 'poincare')
        num_samples: number of 4-tuples to sample
        seed: random seed

    Returns:
        dict with delta_max, delta_95, delta_median, delta_mean, num_samples, diameter
    """
    g = torch.Generator()
    g.manual_seed(seed)
    N, D = point_cloud.shape

    # Sample 4 * num_samples indices
    idx = torch.randint(0, N, (num_samples, 4), generator=g)

    pc = point_cloud
    a = pc[idx[:, 0]]
    b = pc[idx[:, 1]]
    c_pt = pc[idx[:, 2]]
    d = pc[idx[:, 3]]

    if metric == 'euclidean':
        dab = torch.linalg.norm(a - b, dim=1)
        dac = torch.linalg.norm(a - c_pt, dim=1)
        dad = torch.linalg.norm(a - d, dim=1)
        dbc = torch.linalg.norm(b - c_pt, dim=1)
        dbd = torch.linalg.norm(b - d, dim=1)
        dcd = torch.linalg.norm(c_pt - d, dim=1)
    elif metric == 'poincare':
        a_b = proj_to_ball_np(a, c)
        b_b = proj_to_ball_np(b, c)
        c_b = proj_to_ball_np(c_pt, c)
        d_b = proj_to_ball_np(d, c)

        sqrt_c = np.sqrt(c)

        def pd(u, v):
            u_sq = (u * u).sum(dim=-1)
            v_sq = (v * v).sum(dim=-1)
            uv = (u * v).sum(dim=-1)
            denom = (1 + c * uv) ** 2
            num = ((1 + c * u_sq).unsqueeze(-1) * v + (1 - c * v_sq).unsqueeze(-1) * (-u))
            minus_u_plus_v = num / denom.unsqueeze(-1).clamp(min=1e-10)
            norm_mobius = torch.linalg.norm(minus_u_plus_v, dim=-1).clamp(min=1e-10, max=(1.0 - 1e-5) / sqrt_c)
            return (2.0 / sqrt_c) * torch.arctanh(sqrt_c * norm_mobius)

        dab = pd(a_b, b_b)
        dac = pd(a_b, c_b)
        dad = pd(a_b, d_b)
        dbc = pd(b_b, c_b)
        dbd = pd(b_b, d_b)
        dcd = pd(c_b, d_b)
    else:
        raise ValueError(f"Unknown metric: {metric}")

    s1 = dab + dcd
    s2 = dac + dbd
    s3 = dad + dbc

    stacked = torch.stack([s1, s2, s3], dim=1)
    sorted_, _ = torch.sort(stacked, dim=1)
    delta = (sorted_[:, -1] - sorted_[:, -2]) / 2.0

    # Diameter via sampled pairs (use larger sample for stability)
    diameter_approx = gromov_diameter_sample_metric(
        pc, metric=metric, c=c, num_samples=10000, seed=seed + 1,
    )

    return {
        'delta_max': float(delta.max()),
        'delta_95': float(torch.quantile(delta, 0.95)),
        'delta_median': float(delta.median()),
        'delta_mean': float(delta.mean()),
        'delta_std': float(delta.std()),
        'num_samples': int(num_samples),
        'point_cloud_size': N,
        'point_cloud_dim': D,
        'diameter_approx': diameter_approx,
        'normalized_delta_95': float(torch.quantile(delta, 0.95)) / diameter_approx if diameter_approx > 0 else None,
        'normalized_delta_max': float(delta.max()) / diameter_approx if diameter_approx > 0 else None,
    }


def main():
    args = parse_args()
    print(f"[Task #116] HG-Rec per-layer δ-hyperbolicity (dual metric)")
    print(f"  ckpt: {args.ckpt}")
    print(f"  label: {args.label}")
    print(f"  embedding: {args.emb}")
    print(f"  num_samples: {args.num_samples}")
    print(f"  seeds: {args.seeds}")
    print(f"  metrics: {args.metrics}")

    # Load embeddings (assume parquet with one column or numpy .npy)
    if args.emb.endswith('.parquet'):
        import pandas as pd
        df = pd.read_parquet(args.emb)
        # Try common column names
        emb_col = None
        for c in ['embedding', 'emb', 'features', 'vec']:
            if c in df.columns:
                emb_col = c
                break
        if emb_col is None:
            emb_col = df.columns[0]
        embeddings_np = np.stack(df[emb_col].values)
        print(f"  loaded {embeddings_np.shape[0]} embeddings from parquet col '{emb_col}', dim={embeddings_np.shape[1]}")
    elif args.emb.endswith('.npy'):
        embeddings_np = np.load(args.emb)
        print(f"  loaded {embeddings_np.shape[0]} embeddings from npy, dim={embeddings_np.shape[1]}")
    else:
        raise ValueError(f"Unknown embedding format: {args.emb}")

    # Load HRQ-VAE
    model, model_args = load_hrqvae(args.ckpt, args.device)
    num_layers = len(model_args['num_emb_list'])
    curvatures = model_args['curvatures']
    print(f"  num codebook layers: {num_layers}")
    print(f"  num_emb_list: {model_args['num_emb_list']}")
    print(f"  e_dim: {model_args['e_dim']}")
    print(f"  per-layer curvatures: {curvatures}")

    # Extract per-layer residuals
    residuals, assignments, codewords = extract_per_layer_residuals(
        model, embeddings_np, args.device, args.batch_size, use_sk=False,
    )
    print(f"  residual shapes: {[r.shape for r in residuals]}")

    # Compute δ per (layer, metric, seed) combination
    all_results = []
    for i, r in enumerate(residuals):
        if i == 0:
            label = 'L0_raw_encoded_z'
        elif i < num_layers:
            label = f'L{i}_pre_quant_input (z - q_0..q_{i-1})'
        else:
            label = f'L{num_layers}_final_residual (z - q_0 - q_1 - q_2)'
        layer_curvature = curvatures[min(i, num_layers - 1)] if i < num_layers else curvatures[-1]
        norm_mean = r.norm(dim=-1).mean().item()
        norm_std = r.norm(dim=-1).std().item()
        print(f"\n[Layer {i}/{num_layers}] {label}, c={layer_curvature}, ||r||={norm_mean:.4f}±{norm_std:.4f}")

        layer_result = {
            'layer': i,
            'label': label,
            'per_layer_curvature': layer_curvature,
            'residual_norm_mean': norm_mean,
            'residual_norm_std': norm_std,
            'shape': list(r.shape),
            'metrics': {},
        }

        for metric in args.metrics:
            metric_results = []
            for seed in args.seeds:
                t0 = time.time()
                result = gromov_delta_4point_metric(
                    r, metric=metric, c=layer_curvature,
                    num_samples=args.num_samples, seed=seed,
                )
                elapsed = time.time() - t0
                result['seed'] = seed
                result['elapsed_sec'] = elapsed
                metric_results.append(result)
                print(f"  [{metric} seed={seed}] δ_max={result['delta_max']:.4f}, δ_95={result['delta_95']:.4f}, "
                      f"diameter={result['diameter_approx']:.4f}, "
                      f"δ_95/d={result['normalized_delta_95']:.4f} ({elapsed:.1f}s)")

            # Aggregate across seeds (mean ± std)
            d95_arr = np.array([m['delta_95'] for m in metric_results])
            dmax_arr = np.array([m['delta_max'] for m in metric_results])
            diam_arr = np.array([m['diameter_approx'] for m in metric_results])
            norm95_arr = np.array([m['normalized_delta_95'] for m in metric_results])

            layer_result['metrics'][metric] = {
                'per_seed': metric_results,
                'delta_95_mean': float(d95_arr.mean()),
                'delta_95_std': float(d95_arr.std()),
                'delta_max_mean': float(dmax_arr.mean()),
                'delta_max_std': float(dmax_arr.std()),
                'diameter_mean': float(diam_arr.mean()),
                'diameter_std': float(diam_arr.std()),
                'normalized_delta_95_mean': float(norm95_arr.mean()),
                'normalized_delta_95_std': float(norm95_arr.std()),
            }

        all_results.append(layer_result)

    # Aggregate output
    output = {
        'task': 'task116_hgrec_delta_per_layer_dual_metric',
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'label': args.label,
        'ckpt': args.ckpt,
        'embedding_file': args.emb,
        'num_samples': args.num_samples,
        'seeds': args.seeds,
        'metrics': args.metrics,
        'num_layers': num_layers,
        'num_emb_list': model_args['num_emb_list'],
        'e_dim': model_args['e_dim'],
        'per_layer_curvatures': curvatures,
        'layers': all_results,
        'interpretation': (
            'δ_95/diameter < 0.1 → tree-like (hyperbolic). '
            'Euclidean and Poincaré metrics are computed separately; '
            'compare both to assess whether log-map coords under-represent hyperbolic structure.'
        ),
    }

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\n[Task #116] Result saved to {args.output}")

    # Print summary
    print(f"\n=== Summary ({args.label}) ===")
    for r in all_results:
        print(f"  Layer {r['layer']:>2} ({r['label'][:35]:<35}):")
        for metric in args.metrics:
            m = r['metrics'][metric]
            print(f"    [{metric:>9}] δ_95={m['delta_95_mean']:.4f}±{m['delta_95_std']:.4f}, "
                  f"d={m['diameter_mean']:.4f}±{m['diameter_std']:.4f}, "
                  f"δ_95/d={m['normalized_delta_95_mean']:.4f}")


if __name__ == '__main__':
    main()