#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Task #117 — HG-Rec stress distortion grid search per-layer.

Adapter of task82_phonism_grid_refinement.py for HG-Rec HRQ-VAE:
  - Loads HRQ-VAE from HG-Rec ckpt (model.hrq.vq_layers)
  - Extracts per-layer residuals (same as task116)
  - Runs stress grid using Task #82 B refined κ grid:
      [0.0, -0.05, -0.10, -0.15, -0.20, -0.25, -0.30, -0.50, -1.00, -1.50, -2.00]
  - Reuses compute_true_distortion from task80_stage1c_true_distortion.py
  - Validates positive control first (Task #81 v3) before main run

Usage:
    python scripts/task117_hgrec_stress_grid.py \
        --ckpt products/task84/ckpt/Instruments/Jul-23-2026_20-08-06_.../best_loss_model.pth \
        --emb /home/wlia0047/ar57/wenyu/GeneRec/HG-Rec/dataset/Instruments/item_emb.parquet \
        --device cuda:0 \
        --label "task84_c111" \
        --output verdicts/task117_hgrec_stress_task84_c111.json
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
sys.path.insert(0, '/home/wlia0047/ar57/wenyu/GeneRec/scripts')

# Reuse from task80 (compute_true_distortion)
from task80_stage1c_true_distortion import compute_true_distortion

# Reuse from task116 (load + extract_per_layer_residuals)
from task116_hgrec_delta_per_layer import load_hrqvae, extract_per_layer_residuals


# Task #82 B refined κ grid
KAPPA_GRID_REFINED = [
    0.0,
    -0.05,
    -0.10,
    -0.15,
    -0.20,
    -0.25,
    -0.30,
    -0.50,
    -1.00,
    -1.50,
    -2.00,
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt', required=True, help='HG-Rec HRQ-VAE best_loss_model.pth')
    p.add_argument('--emb', required=True, help='item_emb.parquet path')
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--output', required=True)
    p.add_argument('--label', default='unknown',
                   help='Human label for this ckpt (e.g. "task84_c111" or "task88_c555")')
    p.add_argument('--n_subset', type=int, default=300,
                   help='Number of points to subsample per residual layer (task80 default)')
    p.add_argument('--n_iter', type=int, default=200,
                   help='RGD iterations (task80 default)')
    p.add_argument('--lr', type=float, default=0.005,
                   help='RGD learning rate (task80 default)')
    p.add_argument('--d_mds', type=int, default=32,
                   help='MDS output dimension (task80 default)')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--batch_size', type=int, default=512)
    p.add_argument('--kappas', type=float, nargs='+', default=KAPPA_GRID_REFINED,
                   help='κ grid (default: Task #82 B refined)')
    p.add_argument('--skip_positive_control', action='store_true',
                   help='Skip Task #81 v3 positive control validation')
    return p.parse_args()


def run_positive_control(device='cuda:0', n_subset=300, seed=42):
    """Replicate Task #81 v3 positive control: 3 synthetic 32D hyperbolic trees.

    Validates that the stress grid correctly identifies negative curvature
    direction on synthetic data. If this fails, abort main run.
    """
    print("\n=== Positive Control (Task #81 v3 replica) ===")
    print("  Generating 3 synthetic 32D hyperbolic trees...")

    def gen_hyperbolic_tree(n_nodes=200, n_branches=4, depth=4, dim=32, kappa=-1.0, seed=0):
        """Generate a synthetic hyperbolic tree by recursive branching."""
        rng = np.random.RandomState(seed)
        coords = [np.zeros(dim)]
        parents = [-1]

        for d in range(depth):
            n_cur = len(coords)
            for i in range(n_cur):
                if parents[i] != -1 and rng.random() > 0.7:
                    continue  # prune
                for _ in range(n_branches):
                    direction = rng.randn(dim)
                    direction /= np.linalg.norm(direction) + 1e-10
                    # Radial displacement scales with depth for hyperbolic
                    scale = (0.5 + d * 0.2) / np.sqrt(-kappa)
                    new_pt = coords[i] + scale * direction
                    # Project to ball
                    norm = np.linalg.norm(new_pt)
                    max_norm = 0.99 / np.sqrt(-kappa)
                    if norm > max_norm:
                        new_pt = new_pt / norm * max_norm
                    coords.append(new_pt)
                    parents.append(i)

        coords = np.array(coords[:n_nodes])
        # If we have more, truncate; if less, repeat with new seed
        if len(coords) > n_nodes:
            coords = coords[:n_nodes]
        elif len(coords) < n_nodes:
            coords = np.vstack([coords, coords[rng.randint(0, len(coords)):n_nodes - len(coords)]])
        return coords

    results = {}
    for label, ks in [('Tree-A', -1.0), ('Tree-B', -1.0), ('Tree-C', -1.0)]:
        coords = gen_hyperbolic_tree(n_nodes=300, dim=32, kappa=ks, seed=hash(label) % 10000)
        print(f"\n  [{label}] synthetic tree: shape={coords.shape}")

        kappa_results = []
        for k in KAPPA_GRID_REFINED:
            t0 = time.time()
            try:
                stress = compute_true_distortion(
                    coords, kappa=k, n_subset=min(200, len(coords)), seed=seed,
                    n_iter=100, lr=0.005, d=32, device=device,
                )
                kappa_results.append({'kappa': k, 'stress': stress, 'elapsed_sec': time.time() - t0})
            except Exception as e:
                kappa_results.append({'kappa': k, 'stress': float('inf'), 'error': str(e)[:100]})

        best = min(kappa_results, key=lambda x: x['stress'] if x['stress'] == x['stress'] else float('inf'))
        results[label] = {
            'true_kappa': ks,
            'best_kappa': best['kappa'],
            'best_stress': best['stress'],
            'all_kappas': kappa_results,
        }
        print(f"    best κ*={best['kappa']:.2f}, stress={best['stress']:.4f}")

    # Validate: at least 2/3 trees should detect negative κ direction
    neg_detected = sum(1 for r in results.values() if r['best_kappa'] < 0)
    print(f"\n  Positive control: {neg_detected}/3 trees detected negative curvature")
    if neg_detected < 2:
        print(f"  ❌ FAIL: positive control failed, aborting main run")
        return results, False
    print(f"  ✅ PASS: positive control validated")
    return results, True


def main():
    args = parse_args()
    print(f"[Task #117] HG-Rec stress distortion grid search")
    print(f"  ckpt: {args.ckpt}")
    print(f"  label: {args.label}")
    print(f"  embedding: {args.emb}")
    print(f"  κ grid: {args.kappas}")
    print(f"  n_subset={args.n_subset}, n_iter={args.n_iter}, lr={args.lr}, d={args.d_mds}")

    # Step 1: positive control validation
    pc_results = None
    if not args.skip_positive_control:
        pc_results, pc_pass = run_positive_control(
            device=args.device, n_subset=args.n_subset, seed=args.seed,
        )
        if not pc_pass:
            print(f"\n❌ Aborting: positive control failed")
            output = {
                'task': 'task117_hgrec_stress_grid',
                'label': args.label,
                'status': 'failed_positive_control',
                'positive_control': pc_results,
            }
            os.makedirs(os.path.dirname(args.output), exist_ok=True)
            with open(args.output, 'w') as f:
                json.dump(output, f, indent=2)
            return

    # Step 2: load embedding
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
    elif args.emb.endswith('.npy'):
        embeddings_np = np.load(args.emb)
        print(f"  loaded {embeddings_np.shape[0]} embeddings from npy")
    else:
        raise ValueError(f"Unknown embedding format: {args.emb}")

    # Step 3: load HRQ-VAE
    model, model_args = load_hrqvae(args.ckpt, args.device)
    num_layers = len(model_args['num_emb_list'])
    curvatures = model_args['curvatures']
    print(f"  num codebook layers: {num_layers}")
    print(f"  num_emb_list: {model_args['num_emb_list']}")
    print(f"  per-layer curvatures: {curvatures}")

    # Step 4: extract per-layer residuals
    residuals, assignments, codewords = extract_per_layer_residuals(
        model, embeddings_np, args.device, args.batch_size, use_sk=False,
    )
    print(f"  residual shapes: {[r.shape for r in residuals]}")

    # Step 5: run stress grid per layer
    layer_results = []
    for i, r in enumerate(residuals):
        if i == 0:
            label = 'L0_raw_encoded_z'
        elif i < num_layers:
            label = f'L{i}_pre_quant_input (z - q_0..q_{i-1})'
        else:
            label = f'L{num_layers}_final_residual (z - q_0 - q_1 - q_2)'
        norm_mean = r.norm(dim=-1).mean().item()
        print(f"\n[Layer {i}/{num_layers}] {label}, ||r||={norm_mean:.4f}")

        r_np = r.detach().cpu().numpy().astype(np.float32)
        if len(r_np) > args.n_subset:
            # Stratified subsample for stability
            rng = np.random.RandomState(args.seed)
            idx = rng.choice(len(r_np), size=args.n_subset, replace=False)
            r_subset = r_np[idx]
        else:
            r_subset = r_np

        kappa_results = []
        for k in args.kappas:
            t0 = time.time()
            try:
                stress = compute_true_distortion(
                    r_subset, kappa=k, n_subset=len(r_subset), seed=args.seed,
                    n_iter=args.n_iter, lr=args.lr, d=args.d_mds, device=args.device,
                )
                elapsed = time.time() - t0
                kappa_results.append({'kappa': k, 'stress': stress, 'elapsed_sec': elapsed})
                print(f"  κ={k:>6.2f}: stress={stress:.6f} ({elapsed:.1f}s)")
            except Exception as e:
                kappa_results.append({'kappa': k, 'stress': float('inf'), 'error': str(e)[:100], 'elapsed_sec': time.time() - t0})
                print(f"  κ={k:>6.2f}: ERROR ({e})")

        # Find best κ
        valid = [kr for kr in kappa_results if kr['stress'] == kr['stress'] and kr['stress'] != float('inf')]
        if valid:
            best = min(valid, key=lambda x: x['stress'])
            best_kappa = best['kappa']
            best_stress = best['stress']
            zero_stress = next((kr['stress'] for kr in valid if kr['kappa'] == 0.0), float('inf'))
            margin = zero_stress / best_stress if best_stress > 0 else float('inf')
            print(f"  → best κ*={best_kappa:.2f}, stress={best_stress:.6f}, "
                  f"stress(0)/stress(best)={margin:.2f}×")
        else:
            best_kappa = None
            best_stress = float('inf')
            zero_stress = float('inf')
            margin = float('inf')
            print(f"  → all stress values invalid (NaN/Inf)")

        layer_results.append({
            'layer': i,
            'label': label,
            'per_layer_curvature': curvatures[min(i, num_layers - 1)] if i < num_layers else curvatures[-1],
            'residual_norm_mean': norm_mean,
            'shape': list(r.shape),
            'kappa_grid': args.kappas,
            'kappa_results': kappa_results,
            'best_kappa': best_kappa,
            'best_stress': best_stress,
            'zero_stress': zero_stress,
            'margin_vs_zero': margin,
        })

    # Step 6: aggregate
    output = {
        'task': 'task117_hgrec_stress_grid',
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'label': args.label,
        'ckpt': args.ckpt,
        'embedding_file': args.emb,
        'positive_control': pc_results,
        'num_layers': num_layers,
        'num_emb_list': model_args['num_emb_list'],
        'e_dim': model_args['e_dim'],
        'per_layer_curvatures': curvatures,
        'kappa_grid': args.kappas,
        'n_subset': args.n_subset,
        'n_iter': args.n_iter,
        'lr': args.lr,
        'd_mds': args.d_mds,
        'seed': args.seed,
        'layers': layer_results,
        'interpretation': (
            'best κ* < 0 → residual space is better fit by hyperbolic geometry. '
            'margin_vs_zero = stress(κ=0)/stress(best κ*) > 2 → significant hyperbolic preference. '
            'Per-layer curvature: c_i from ckpt; if [0.5,0.5,0.5] then per-layer c is smaller (more hyperbolic).'
        ),
    }

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\n[Task #117] Result saved to {args.output}")

    # Summary
    print(f"\n=== Summary ({args.label}) ===")
    for r in layer_results:
        print(f"  Layer {r['layer']:>2} ({r['label'][:35]:<35}): "
              f"best κ*={r['best_kappa']}, "
              f"stress(best)={r['best_stress']:.6f}, "
              f"margin={r['margin_vs_zero']:.2f}×")


if __name__ == '__main__':
    main()