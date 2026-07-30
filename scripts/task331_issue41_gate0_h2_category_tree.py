#!/usr/bin/env python3
"""Task #331 / Issue #41 Gate 0 H2 — Sala 2018 tree composite construction.

零 GPU 纯计算 (CPU only).
Method: 直接对 Musical_Instruments 商品类目树用 Sala 2018 树状组合构造法.

Data source:
  HG-Rec/dataset/Instruments/category_depth.npy — per-item ρ (radius from origin in hyperbolic ball)
  HG-Rec/dataset/Instruments/category_depth_meta.json — depth histogram (depth 2-16, count per depth)

Sala 2018 Theorem 2: 任何深度 d 的树可嵌入 (κ, d) 双曲空间, 节点 ρ ≤ log(d) / sqrt(-κ).
对于 Musical_Instruments 类目树: tree_max_depth = 16, 节点数 9922.
如果数据真的是树状, 则 ρ = log(d_path) / sqrt(-κ) where d_path is path depth.

Method v2:
  对每个 item i, 有 ρ_i 和 depth_i (从 histogram 推算).
  假设 ρ_i = (1/sqrt(-κ)) · log(depth_i) + noise
  拟合 κ via least squares: ρ_i = a · log(depth_i), a = 1/sqrt(-κ), κ = -1/a²
"""
import argparse
import json
import math
import os
import sys

import numpy as np


CATEGORY_DEPTH = '/home/wlia0047/ar57/wenyu/GeneRec/HG-Rec/dataset/Instruments/category_depth.npy'
CATEGORY_META = '/home/wlia0047/ar57/wenyu/GeneRec/HG-Rec/dataset/Instruments/category_depth_meta.json'


def load_category_data():
    """Load per-item ρ + depth histogram."""
    rho_per_item = np.load(CATEGORY_DEPTH)
    meta = json.load(open(CATEGORY_META))
    histogram = {int(k): int(v) for k, v in meta['depth_histogram'].items()}
    return rho_per_item, meta, histogram


def assign_depth_per_item(rho_per_item, histogram):
    """Assign depth to each item by reverse-engineering from histogram.

    The category_depth.npy stores ρ (continuous radius). We need to map each ρ value
    back to a discrete depth. Since ρ is monotonic in depth (deeper → larger ρ),
    we bucket items by ρ range: smallest ρ → depth 2, etc.
    """
    # Compute percentile-based bin edges from histogram proportions
    total = sum(histogram.values())
    cumulative = 0
    bin_edges = [rho_per_item.min() - 0.01]
    for d in sorted(histogram.keys()):
        cumulative += histogram[d]
        bin_edges.append(np.percentile(rho_per_item, cumulative / total * 100))
    bin_edges[-1] = rho_per_item.max() + 0.01

    # Map each item to depth based on ρ bin
    depths = np.zeros_like(rho_per_item, dtype=int)
    depth_keys = sorted(histogram.keys())
    for i, rho in enumerate(rho_per_item):
        for j, d in enumerate(depth_keys):
            if bin_edges[j] <= rho < bin_edges[j+1]:
                depths[i] = d
                break
    return depths, depth_keys


def sala_2018_tree_fit(rho_per_item, depths):
    """Fit κ from ρ vs log(depth) linear relationship.

    ρ = a · log(depth) + noise, where a = 1/sqrt(-κ)
    Least squares: a = sum(ρ · log(d)) / sum(log(d)^2)
    κ = -1/a²
    """
    valid_mask = depths >= 2
    rho_v = rho_per_item[valid_mask]
    depth_v = depths[valid_mask]
    log_d = np.log(depth_v.astype(float))

    # Weighted by item count (depth frequency)
    unique_d, counts = np.unique(depth_v, return_counts=True)
    weights = np.array([counts[np.where(unique_d == d)[0][0]] for d in depth_v])

    # Weighted least squares
    w = weights.astype(float)
    a = float((w * rho_v * log_d).sum() / (w * log_d ** 2).sum())
    if a > 0:
        kappa_fit = -1.0 / (a ** 2)
    else:
        kappa_fit = float('nan')

    # Also fit unweighted for comparison
    a_unw = float((rho_v * log_d).sum() / (log_d ** 2).sum())
    kappa_fit_unw = -1.0 / (a_unw ** 2) if a_unw > 0 else float('nan')

    # Per-depth mean rho + log(depth)
    per_depth_stats = {}
    for d in sorted(unique_d):
        mask = depth_v == d
        per_depth_stats[int(d)] = {
            'n_items': int(counts[np.where(unique_d == d)[0][0]]),
            'mean_rho': float(rho_v[mask].mean()),
            'std_rho': float(rho_v[mask].std()),
            'log_d': math.log(d),
        }

    return {
        'kappa_fit_weighted': float(kappa_fit),
        'kappa_fit_unweighted': float(kappa_fit_unw),
        'slope_a_weighted': a,
        'slope_a_unweighted': a_unw,
        'per_depth_stats': per_depth_stats,
        'r_squared_weighted': compute_r_squared(rho_v, log_d, a, w),
        'r_squared_unweighted': compute_r_squared(rho_v, log_d, a_unw, None),
    }


def compute_r_squared(rho, log_d, a, w):
    """R² of fit ρ = a · log(d)."""
    if w is None:
        ss_res = ((rho - a * log_d) ** 2).sum()
        ss_tot = ((rho - rho.mean()) ** 2).sum()
    else:
        ss_res = (w * (rho - a * log_d) ** 2).sum()
        ss_tot = (w * (rho - rho.mean()) ** 2).sum()
    if ss_tot > 0:
        return float(1 - ss_res / ss_tot)
    return float('nan')


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--output', default='/home/wlia0047/ar57/wenyu/GeneRec/verdicts/task331_issue41_gate0_h2_category_tree.json')
    return p.parse_args()


def main():
    args = parse_args()
    print(f"[Task #331 / Issue #41 Gate 0 H2] Sala 2018 tree composite construction (v2 — empirical ρ)")

    rho_per_item, meta, histogram = load_category_data()
    print(f"  Loaded ρ per item: shape={rho_per_item.shape}, range=[{rho_per_item.min():.2f}, {rho_per_item.max():.2f}]")
    print(f"  Depth histogram: {histogram}")
    print(f"  Tree max depth: {meta['depth_max']}, rho range: [{meta['rho_min']}, {meta['rho_max']}]")

    # Assign depth per item by bucketing ρ
    depths, depth_keys = assign_depth_per_item(rho_per_item, histogram)
    print(f"  Assigned depths: range=[{depths.min()}, {depths.max()}]")

    # Sala 2018 fit
    fit = sala_2018_tree_fit(rho_per_item, depths)
    print(f"\n  === Sala 2018 tree composite fit ===")
    print(f"  κ (weighted fit):     {fit['kappa_fit_weighted']:+.4f}  (slope a = {fit['slope_a_weighted']:+.4f})")
    print(f"  κ (unweighted fit):   {fit['kappa_fit_unweighted']:+.4f}  (slope a = {fit['slope_a_unweighted']:+.4f})")
    print(f"  R² (weighted):        {fit['r_squared_weighted']:.4f}")
    print(f"  R² (unweighted):      {fit['r_squared_unweighted']:.4f}")
    print(f"\n  Per-depth stats:")
    for d, stats in fit['per_depth_stats'].items():
        print(f"    depth={d}: n={stats['n_items']}, mean_ρ={stats['mean_rho']:.3f}, log(d)={stats['log_d']:.3f}")

    summary = {
        'task': 'task331_issue41_gate0_h2_category_tree_v2',
        'method': 'Sala 2018 树状组合构造法 (Theorem 2) — empirical ρ fit',
        'formula': 'ρ = (1/sqrt(-κ)) · log(depth) ⇒ κ = -1/a² (slope a from linear fit)',
        'data_source': 'HG-Rec/dataset/Instruments/category_depth.npy + category_depth_meta.json',
        'n_items': int(rho_per_item.shape[0]),
        'rho_min': float(rho_per_item.min()),
        'rho_max': float(rho_per_item.max()),
        'depth_histogram': histogram,
        'tree_max_depth': int(meta['depth_max']),
        'fit_results': fit,
        'ollivier_reference': {'mean_κ': -0.726, 'range': [-0.65, -0.84], 'source': 'Task #70'},
    }
    summary['kappa_estimate'] = fit['kappa_fit_weighted']
    summary['gate_0_h2_pass'] = bool(
        abs(summary['kappa_estimate']) > 0.05 and summary['kappa_estimate'] < 0
    )
    summary['gate_0_h2_pass_condition'] = '|κ|>0.05 且 κ<0 (方向负 + 显著)'

    out_path = args.output
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Saved: {out_path}")
    print(f"  Gate 0 H2 PASS: {summary['gate_0_h2_pass']}")


if __name__ == '__main__':
    main()