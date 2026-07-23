#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Task #90 — Phonism vs HG-Rec Codebook Decomposition Analysis.

Compare 4 RQ-VAE SID outputs on Musical_Instruments:
1. vanilla RQ-VAE + Sinkhorn (phonism baseline)
2. HG-Rec c111 (Poincaré κ=1.0)
3. HG-Rec c555 (Poincaré κ=0.5) ⭐ marginal best
4. HG-Rec free-curv A arm (Poincaré κ→0)

Metrics:
- Per-layer utilization (unique tokens / max codes)
- Per-layer entropy (token distribution uniformity)
- Per-layer top-K token concentration
- Inter-method SID overlap (Jaccard on 4-token SIDs)
- Inter-method agreement on L0/L1/L2 token assignments

No GPU required - pure numpy analysis on existing .npy files.
"""
from __future__ import annotations

import json
import os
from collections import Counter

import numpy as np


SID_FILES = {
    'vanilla (phonism)': 'HG-Rec/dataset/Instruments/Instruments_t5_hrqvae_poincare.npy',
    'HG-Rec c111 (κ=1.0)': 'HG-Rec/dataset/Instruments/Instruments_curv_1.0_1.0_1.0_t5_hrqvae_poincare.npy',
    'HG-Rec c555 (κ=0.5)': 'HG-Rec/dataset/Instruments/Instruments_curv_0.5_0.5_0.5_t5_hrqvae_poincare.npy',
    'HG-Rec free-curv (κ→0)': 'HG-Rec/dataset/Instruments/Instruments_curv_free_M1_t5_hrqvae_poincare.npy',
}

# Known downstream R@10 results (from verdicts)
DOWNSTREAM_R10 = {
    'vanilla (phonism)': 0.1058,
    'HG-Rec c111 (κ=1.0)': 0.1020,
    'HG-Rec c555 (κ=0.5)': 0.1051,
    'HG-Rec free-curv (κ→0)': 0.1015,
}

CODEBOOK_SIZES = {0: 64, 1: 128, 2: 256}


def shannon_entropy(counts: np.ndarray) -> float:
    """Shannon entropy H = -Σ p log p, max = log K for uniform."""
    total = counts.sum()
    if total == 0:
        return 0.0
    p = counts / total
    p = p[p > 0]
    return float(-(p * np.log(p)).sum())


def max_entropy(K: int) -> float:
    return float(np.log(K))


def layer_stats(tokens: np.ndarray, layer: int) -> dict:
    """Compute utilization + entropy + top-K stats for one layer."""
    K = CODEBOOK_SIZES[layer]
    counts = np.bincount(tokens, minlength=K)
    n_unique = int((counts > 0).sum())
    util = n_unique / K
    H = shannon_entropy(counts)
    H_max = max_entropy(K)
    norm_H = H / H_max if H_max > 0 else 0.0
    top5_count = int(np.sort(counts)[-5:].sum())
    top5_frac = top5_count / counts.sum()
    top1_count = int(counts.max())
    top1_frac = top1_count / counts.sum()
    return {
        'layer': layer,
        'unique_tokens': n_unique,
        'codebook_size': K,
        'utilization': util,
        'entropy': H,
        'max_entropy': H_max,
        'norm_entropy': norm_H,
        'top1_count': top1_count,
        'top1_frac': top1_frac,
        'top5_count': top5_count,
        'top5_frac': top5_frac,
    }


def jaccard_per_layer(a: np.ndarray, b: np.ndarray) -> dict:
    """Per-layer token-set Jaccard similarity."""
    out = {}
    for layer in range(3):
        sa = set(a[:, layer].tolist())
        sb = set(b[:, layer].tolist())
        if len(sa | sb) == 0:
            j = 0.0
        else:
            j = len(sa & sb) / len(sa | sb)
        out[f'L{layer}_jaccard'] = j
    # 4-token SID overlap
    sa4 = set(map(tuple, a.tolist()))
    sb4 = set(map(tuple, b.tolist()))
    if len(sa4 | sb4) == 0:
        out['4col_jaccard'] = 0.0
    else:
        out['4col_jaccard'] = len(sa4 & sb4) / len(sa4 | sb4)
    # Same SID (per-item)
    same_4col = int((a == b).all(axis=1).sum())
    out['items_with_same_4col_sid'] = same_4col
    out['total_items'] = len(a)
    return out


def main():
    print("=" * 80)
    print("Task #90 — Phonism vs HG-Rec Codebook Decomposition Analysis")
    print("=" * 80)

    # Load all SIDs
    sids = {}
    for name, path in SID_FILES.items():
        if os.path.exists(path):
            sids[name] = np.load(path)
            print(f"  Loaded {name}: shape={sids[name].shape}, "
                  f"unique 4-col={len(np.unique(sids[name], axis=0))}")
        else:
            print(f"  ⚠️ Missing: {path}")

    if not sids:
        raise RuntimeError("No SID files found")

    # Per-method per-layer stats
    print("\n" + "=" * 80)
    print("Per-Method Per-Layer Codebook Statistics")
    print("=" * 80)

    all_stats = {}
    for name, sid in sids.items():
        stats = [layer_stats(sid[:, layer], layer) for layer in range(3)]
        all_stats[name] = stats

    # Print comparison table
    print(f"\n{'Method':<30} {'Layer':<6} {'Unique':<8} {'Util':<8} {'Entropy':<10} {'NormH':<8} {'Top1%':<8} {'Top5%':<8}")
    for name, stats in all_stats.items():
        for s in stats:
            print(f"{name:<30} L{s['layer']:<5} {s['unique_tokens']:<8} {s['utilization']:<8.3f} "
                  f"{s['entropy']:<10.3f} {s['norm_entropy']:<8.3f} "
                  f"{s['top1_frac']*100:<8.2f} {s['top5_frac']*100:<8.2f}")

    # Inter-method SID overlap
    print("\n" + "=" * 80)
    print("Inter-Method SID Overlap (Jaccard)")
    print("=" * 80)
    names = list(sids.keys())
    pairs = []
    for i, n1 in enumerate(names):
        for j, n2 in enumerate(names):
            if i < j:
                pairs.append((n1, n2))

    print(f"\n{'Pair':<60} {'L0':<8} {'L1':<8} {'L2':<8} {'4-col':<8} {'Items same':<12}")
    overlap_data = {}
    for n1, n2 in pairs:
        ov = jaccard_per_layer(sids[n1], sids[n2])
        overlap_data[f'{n1}_vs_{n2}'] = ov
        print(f"{(n1 + ' vs ' + n2):<60} {ov['L0_jaccard']:<8.3f} {ov['L1_jaccard']:<8.3f} "
              f"{ov['L2_jaccard']:<8.3f} {ov['4col_jaccard']:<8.3f} "
              f"{ov['items_with_same_4col_sid']:<12}")

    # Summary: rank methods by entropy + utilization
    print("\n" + "=" * 80)
    print("Summary — Utilization + Entropy ranking (higher = more uniform = potentially better)")
    print("=" * 80)

    print(f"\n{'Method':<30} {'L0 util':<10} {'L0 normH':<10} {'L1 util':<10} {'L1 normH':<10} {'L2 util':<10} {'L2 normH':<10} {'R@10':<8}")
    ranking = []
    for name, stats in all_stats.items():
        s0, s1, s2 = stats
        avg_util = (s0['utilization'] + s1['utilization'] + s2['utilization']) / 3
        avg_normH = (s0['norm_entropy'] + s1['norm_entropy'] + s2['norm_entropy']) / 3
        r10 = DOWNSTREAM_R10.get(name, None)
        ranking.append((name, s0['utilization'], s0['norm_entropy'], s1['utilization'], s1['norm_entropy'],
                        s2['utilization'], s2['norm_entropy'], r10))
        print(f"{name:<30} {s0['utilization']:<10.3f} {s0['norm_entropy']:<10.3f} "
              f"{s1['utilization']:<10.3f} {s1['norm_entropy']:<10.3f} "
              f"{s2['utilization']:<10.3f} {s2['norm_entropy']:<10.3f} "
              f"{(r10 if r10 is not None else 'N/A'):<8}")

    # Save results
    out = {
        'task': 'Task #90 — Phonism vs HG-Rec Codebook Decomposition',
        'per_method_per_layer_stats': all_stats,
        'inter_method_overlap': overlap_data,
        'downstream_R10': DOWNSTREAM_R10,
    }
    out_path = 'verdicts/task90_codebook_decomposition.json'
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\n[Done] Results saved to {out_path}")


if __name__ == '__main__':
    main()