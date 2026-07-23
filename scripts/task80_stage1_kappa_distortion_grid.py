#!/usr/bin/env python3
"""Task #80 Stage 1 — offline κ distortion grid search.

For each layer's residual point cloud, evaluate how well different κ values
fit the local geometry. We use Sarkar-style stress: pick a random subset,
compute pairwise Euclidean distances (target), then embed in hyperbolic space
under each candidate κ using stereographic projection, and measure how well
the hyperbolic pairwise distances preserve the target rank/distance structure.

Output: C group (per-layer best κ) + B group (global best κ by sum of stress)
"""
import argparse
import json
import os
import sys
import time
import numpy as np
import torch

sys.path.insert(0, '/home/wlia0047/ar57/wenyu/GeneRec/scripts')
from task92_rqvae_delta_hyperbolicity import load_rqvae, extract_per_layer_residuals


def stereographic_project(x: np.ndarray, kappa: float) -> np.ndarray:
    """Project points to Klein/Poincaré model of hyperbolic space with curvature κ.

    For κ < 0 (negative curvature, hyperbolic), use the stereographic projection:
        u = x / (1 + sqrt(1 + κ·||x||²))

    Args:
        x: (N, d) array of points
        kappa: negative curvature (κ < 0)

    Returns:
        (N, d) projected points with ||u||² < -1/κ
    """
    if kappa >= 0:
        return x
    norm_sq = np.sum(x ** 2, axis=-1, keepdims=True)
    inner = 1 + kappa * norm_sq
    if np.any(inner <= 0):
        # Some points violate Sarkar inequality; clip
        inner = np.maximum(inner, 1e-8)
    denom = 1 + np.sqrt(inner)
    return x / denom


def hyperbolic_distance(u: np.ndarray, v: np.ndarray, kappa: float) -> float:
    """Hyperbolic distance between u and v under curvature κ.

    d_κ(u, v) = (1/sqrt(-κ)) * arccosh(1 + (-2κ)·(u-v)² / ((1+κ||u||²)(1+κ||v||²)))
    """
    if kappa >= 0:
        return np.linalg.norm(u - v)
    diff_sq = np.sum((u - v) ** 2)
    norm_u_sq = np.sum(u ** 2)
    norm_v_sq = np.sum(v ** 2)
    num = -2 * kappa * diff_sq
    denom = (1 + kappa * norm_u_sq) * (1 + kappa * norm_v_sq)
    arg = 1 + num / denom
    arg = max(arg, 1.0 + 1e-7)  # numerical safety
    return float(np.arccosh(arg) / np.sqrt(-kappa))


def compute_distortion(residual: np.ndarray, kappa: float, n_subset: int = 500, seed: int = 42) -> float:
    """Sammon-style distortion: coefficient of variation of (hyperbolic_dist / euclidean_dist)
    ratio across all pairs. Under Euclidean geometry, ratio is constant; under
    hyperbolic geometry it varies with distance scale. A well-fit κ makes the ratio
    curve match the residual's intrinsic geometry (i.e., variance is minimized).

    Lower distortion = better fit.
    """
    rng = np.random.default_rng(seed)
    N = residual.shape[0]
    if N > n_subset:
        idx = rng.choice(N, size=n_subset, replace=False)
        sub = residual[idx]
    else:
        sub = residual

    n = len(sub)
    # Pairwise Euclidean distances
    eucl_d = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            eucl_d[i, j] = np.linalg.norm(sub[i] - sub[j])
            eucl_d[j, i] = eucl_d[i, j]
    iu = np.triu_indices(n, k=1)
    d_eucl = eucl_d[iu]

    # Skip degenerate (zero) pairs
    valid = d_eucl > 1e-8
    if valid.sum() < 10:
        return float('inf')
    d_eucl = d_eucl[valid]

    # Pairwise hyperbolic distances under candidate κ
    u = stereographic_project(sub, kappa)
    hyper_d = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            hyper_d[i, j] = hyperbolic_distance(u[i], u[j], kappa)
            hyper_d[j, i] = hyper_d[i, j]
    d_hyper = hyper_d[iu][valid]

    # CV of ratio
    ratios = d_hyper / d_eucl
    mean_ratio = np.mean(ratios)
    if mean_ratio < 1e-8:
        return float('inf')
    cv = float(np.std(ratios) / mean_ratio)
    return cv


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt', required=True)
    p.add_argument('--emb', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--n_subset', type=int, default=500)
    p.add_argument('--device', default='cuda:0')
    return p.parse_args()


def main():
    args = parse_args()
    print(f"[Task #80 Stage 1] κ distortion grid search")
    print(f"  ckpt: {args.ckpt}")
    print(f"  emb: {args.emb}")
    print(f"  n_subset per layer: {args.n_subset}")
    print(f"  device: {args.device}")

    # Load RQ-VAE + extract residuals
    model, model_args = load_rqvae(args.ckpt, args.device)
    embeddings_np = np.load(args.emb)
    print(f"  embedding shape: {embeddings_np.shape}")

    residuals, _ = extract_per_layer_residuals(
        model, embeddings_np, args.device, batch_size=2048
    )
    num_layers = len(residuals)
    print(f"  num codebook layers: {num_layers}")
    for i, r in enumerate(residuals):
        norm_max = r.norm(dim=-1).max().item()
        print(f"    L{i}: shape={tuple(r.shape)}, norm_max={norm_max:.4f}")

    # κ candidate grids (per δ_95 ranking constraint)
    KAPPA_GRID_L0 = [0.0, -0.3, -0.5]            # L0: mild
    KAPPA_GRID_DEEP = [-0.5, -1.0, -1.5, -2.0]   # L1/L2/L3: more negative

    layer_kappas = {
        0: KAPPA_GRID_L0,
        1: KAPPA_GRID_DEEP,
        2: KAPPA_GRID_DEEP,
        3: KAPPA_GRID_DEEP,
    }

    # Compute distortion per layer × κ
    print()
    print("=== Per-layer κ distortion grid (1 - Spearman corr) ===")
    print(f"{'Layer':<8} {'label':<32} {' | '.join(f'κ={k:>+.1f}' for k in sorted(set(KAPPA_GRID_L0 + KAPPA_GRID_DEEP)))}")
    print("-" * 90)

    results_per_layer = {}
    all_distortions = {}  # (layer, kappa) -> distortion

    for i, r in enumerate(residuals):
        r_np = r.detach().cpu().numpy()
        label = "raw_encoded" if i == 0 else f"residual_after_layer_{i-1}"
        kappas = layer_kappas[i]
        distortions = []
        for k in kappas:
            t0 = time.time()
            d = compute_distortion(r_np, k, n_subset=args.n_subset, seed=args.seed)
            elapsed = time.time() - t0
            print(f"  L{i} κ={k:+.2f}: distortion={d:.4f} ({elapsed:.1f}s)")
            distortions.append((k, d))
            all_distortions[(i, k)] = d
        results_per_layer[i] = {
            'label': label,
            'distortions': distortions,
        }
        row = " | ".join(
            f"{d:.3f}" for k, d in zip(kappas, [dist for _, dist in distortions])
        )
        print(f"L{i}      {label:<32} {row}")

    # C group (per-layer best κ)
    print()
    print("=== C group (per-layer best κ, lower distortion = better) ===")
    c_group = {}
    for i in range(num_layers):
        best_k, best_d = min(results_per_layer[i]['distortions'], key=lambda x: x[1])
        c_group[i] = {'kappa': best_k, 'distortion': best_d}
        print(f"  L{i} ({results_per_layer[i]['label']}): κ* = {best_k:+.2f}, distortion = {best_d:.4f}")

    # B group (global κ by sum of distortions across layers)
    print()
    print("=== B group (global κ, minimize sum of distortions across 4 layers) ===")
    # Use the deep grid as candidate (since L0 is the most constrained)
    all_kappas = sorted(set(KAPPA_GRID_L0 + KAPPA_GRID_DEEP))
    b_group = {}
    for k in all_kappas:
        total_d = sum(all_distortions.get((i, k), float('inf')) for i in range(num_layers))
        b_group[k] = total_d
        print(f"  κ = {k:+.2f}: sum_distortion = {total_d:.4f}")
    best_global_k = min(b_group, key=b_group.get)
    print(f"  → global κ* = {best_global_k:+.2f} (sum_distortion = {b_group[best_global_k]:.4f})")

    # Summary
    output = {
        'task': 'task80_stage1_kappa_distortion_grid',
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'ckpt': args.ckpt,
        'emb': args.emb,
        'n_subset': args.n_subset,
        'seed': args.seed,
        'kappa_grids': {
            'L0': KAPPA_GRID_L0,
            'L1/L2/L3': KAPPA_GRID_DEEP,
        },
        'num_layers': num_layers,
        'all_distortions': {f"L{i}_k{k:+.2f}": v for (i, k), v in all_distortions.items()},
        'C_group_per_layer_best': {
            f"L{i}": {'kappa': v['kappa'], 'distortion': v['distortion']}
            for i, v in c_group.items()
        },
        'B_group_global_best': {
            'kappa': best_global_k,
            'sum_distortion': b_group[best_global_k],
            'all_kappa_total': {f"k{k:+.2f}": v for k, v in b_group.items()},
        },
        'A_group_baseline': {'kappa': 0.0, 'note': 'Euclidean (current phonism)'},
    }
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\n[Stage 1] Result saved to {args.output}")


if __name__ == '__main__':
    main()