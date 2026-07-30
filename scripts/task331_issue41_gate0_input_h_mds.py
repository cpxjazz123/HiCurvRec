#!/usr/bin/env python3
"""Task #331 / Issue #41 Gate 0 (a) — h-MDS on 768-dim INPUT embedding space (NOT residual).

零 GPU 纯计算 (CPU only).
Method: Sala et al. 2018 h-MDS / 失真-曲率-维度权衡 (ICML 2018, arXiv:1804.03329).
For each κ in grid, run Riemannian GD MDS in Poincaré ball on a 768-dim input subset,
compute Kruskal stress-1 vs pairwise Euclidean target distances.

通过条件: optimal κ 应该 |κ|>0.05 显著偏离 0 且方向匹配 Ollivier (负).
硬停止: residual 对照组 (separate script) 复现失败 → STOP.

Usage:
  python3 scripts/task331_issue41_gate0_input_h_mds.py
"""
import argparse
import json
import math
import os
import sys
import time

import numpy as np

sys.path.insert(0, '/home/wlia0047/ar57/wenyu/GeneRec/HG-Rec')
sys.path.insert(0, '/home/wlia0047/ar57/wenyu/GeneRec/scripts')

from task80_stage1c_true_distortion import compute_true_distortion


ITEM_EMB = '/home/wlia0047/ar57/wenyu/GeneRec/HG-Rec/dataset/Instruments/item_emb.parquet'

# Issue #41 κ_grid (扩大覆盖到强双曲, 跟 Task #82 B refined 保持一致)
KAPPA_GRID = [
    0.0, -0.05, -0.10, -0.15, -0.20, -0.30, -0.50, -1.00, -1.50, -2.00,
]


def load_input_embeddings(parquet_path, n_subset=500, seed=42):
    """Load 768-dim input embeddings from item_emb.parquet.

    parquet columns: item_id (str), embedding (list of 768 floats)
    Returns: (n_subset, 768) float32 array, subset of items.
    """
    import pandas as pd
    df = pd.read_parquet(parquet_path)
    # Column check
    if 'embedding' in df.columns:
        embs = np.stack([np.asarray(e, dtype=np.float32) for e in df['embedding'].values])
    else:
        raise RuntimeError(f"No 'embedding' column in {parquet_path}, columns={df.columns.tolist()}")
    print(f"  Loaded {embs.shape[0]} items, dim={embs.shape[1]}")
    rng = np.random.default_rng(seed)
    if n_subset < embs.shape[0]:
        idx = rng.choice(embs.shape[0], size=n_subset, replace=False)
        embs = embs[idx]
    return embs


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--emb', default=ITEM_EMB)
    p.add_argument('--n_subset', type=int, default=500, help='item subset size (CPU feasibility)')
    p.add_argument('--n_iter', type=int, default=200)
    p.add_argument('--lr', type=float, default=0.005)
    p.add_argument('--d', type=int, default=32, help='Poincaré embedding dim (target MDS dim)')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--device', default='cpu')
    p.add_argument('--output', default='/home/wlia0047/ar57/wenyu/GeneRec/verdicts/task331_issue41_gate0_input_h_mds.json')
    return p.parse_args()


def main():
    args = parse_args()
    print(f"[Task #331 / Issue #41 Gate 0 (a)] input space h-MDS")
    print(f"  emb={args.emb}, n_subset={args.n_subset}, d={args.d}")

    embs = load_input_embeddings(args.emb, n_subset=args.n_subset, seed=args.seed)
    print(f"  Subset shape: {embs.shape}")

    # (a) Compute Kruskal stress for each κ
    print(f"\n  κ grid: {KAPPA_GRID}")
    print(f"  {'κ':>8} | {'stress':>10} | {'elapsed (s)':>12}")
    results = {}
    for k in KAPPA_GRID:
        t0 = time.time()
        stress = compute_true_distortion(
            embs, kappa=k, n_subset=args.n_subset,
            n_iter=args.n_iter, lr=args.lr, d=args.d,
            device=args.device, seed=args.seed,
        )
        elapsed = time.time() - t0
        print(f"  {k:>+8.2f} | {stress:>10.4f} | {elapsed:>12.1f}")
        results[str(k)] = {'stress': float(stress), 'elapsed_sec': round(elapsed, 2)}

    # Best κ (Issue #41 fix: extract dict['stress'] before float conversion)
    best_kappa_str, best_stress_dict = min(results.items(), key=lambda x: x[1]['stress'])
    best_kappa = float(best_kappa_str)
    best_stress = best_stress_dict['stress']

    # Also compute Euclidean (κ=0) stress for ratio analysis (Issue #41 inverse-MDS caveat)
    euclidean_stress = results.get('0.0', {}).get('stress', None)
    stress_ratio = None
    if euclidean_stress is not None and best_stress is not None and euclidean_stress > 0:
        stress_ratio = best_stress / euclidean_stress  # < 1.0 means hyperbolic better than Euclidean

    # 与 Ollivier (-0.65~-0.84) 方向一致性
    direction_match = (best_kappa < 0)  # Ollivier 是负, h-MDS 最佳应是负

    # 与 baseline 0.1020 (Issue #41 显著性下限 |κ|>0.05, Task #82 弱信号 0.05)
    significant_deviation = (abs(best_kappa) > 0.05)

    summary = {
        'task': 'task331_issue41_gate0_input_h_mds',
        'method': 'Sala 2018 h-MDS (Poincaré Riemannian GD MDS + Kruskal stress-1)',
        'data_source': 'input embedding space (768-dim) from item_emb.parquet',
        'n_items': int(embs.shape[0]),
        'embedding_dim': int(embs.shape[1]),
        'mds_target_dim': args.d,
        'n_iter': args.n_iter,
        'kappa_grid': KAPPA_GRID,
        'results': results,
        'best_kappa': best_kappa,
        'best_stress': float(best_stress),
        'euclidean_stress': float(euclidean_stress) if euclidean_stress is not None else None,
        'stress_ratio_hyp_to_eucl': float(stress_ratio) if stress_ratio is not None else None,
        'ollivier_reference': {'mean_κ': -0.726, 'range': [-0.65, -0.84], 'source': 'Task #70 verdict'},
        'direction_match_ollivier': direction_match,
        'significant_deviation': significant_deviation,
        'pass_gate_0_a': bool(direction_match and significant_deviation),
        'caveat_monotonic_decrease': 'stress decreases monotonically with more negative κ — may indicate MDS degeneracy (all points cluster at origin)',
        'gate_0_a_pass_condition': '(a) optimal κ deviates |κ|>0.05 from 0 AND direction matches Ollivier (negative)',
    }

    out_path = args.output
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Saved: {out_path}")
    print(f"\n  === Summary ===")
    print(f"  Best κ: {best_kappa:+.2f}, stress: {best_stress:.4f}")
    print(f"  Direction match Ollivier (κ<0): {direction_match}")
    print(f"  Significant deviation |κ|>0.05: {significant_deviation}")
    print(f"  Gate 0 (a) PASS: {summary['pass_gate_0_a']}")


if __name__ == '__main__':
    main()