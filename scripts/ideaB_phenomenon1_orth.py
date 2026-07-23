#!/usr/bin/env python3
"""Idea B 现象 1: 欧氏 vs causal 内积下, 概念方向正交性差异

概念方向 v_brand = mean(x | brand=b) - mean(x)  (x ∈ R^{2048} Stage 1 embedding)
取 ≥30 items 的 brand (确保方向稳定)

Causal inner product: <a,b>_M = a^T M b
简单 M 构造 (暂时用 RQ 码本第一层 centroids 协方差, 作为对比 baseline):
    M = cov(centroids_l1)  -- 用于 Idea B 现象 3 详细比较

cos_Eucl(v1, v2) = v1.T v2 / (||v1|| ||v2||)
cos_M(v1, v2)    = v1.T M v2 / (sqrt(v1.T M v1) * sqrt(v2.T M v2))
"""

import sys
sys.path.insert(0, '/home/wlia0047/ar57/wenyu/GeneRec/GRID')

# Pre-import to break circular import
import src.utils.decorators

import os, json
import numpy as np
import torch

OUT_DIR = '/home/wlia0047/ar57/wenyu/GeneRec/GRID/result/ideaB_causal'
os.makedirs(OUT_DIR, exist_ok=True)

EMB_PATH = '/home/wlia0047/ar57/wenyu/GeneRec/GRID/logs/inference/runs/2026-07-07/18-51-46/pickle/merged_predictions_tensor.pt'
META_PATH = os.path.join(OUT_DIR, 'toys_metadata.json')
BY_BRAND_PATH = os.path.join(OUT_DIR, 'by_brand.json')

# RQ 第一层 centroids (来自 B_mmq 的码本, 用于构造 M_baseline)
# 注: 完整 M 稳定性分析在现象 3 做, 这里先用 RQ 码本作为 placeholder
RQ_CKPT_PATH = '/home/wlia0047/ar57/wenyu/GeneRec/GRID/logs/train/runs/2026-07-11/13-15-28/checkpoints/checkpoint_000_003000.ckpt'

MIN_BRAND_ITEMS = 30   # 至少 30 items 才稳定
N_BRAND_TOP = 20       # 取 top 20 brand 做配对比较


def build_brand_directions(x, by_brand, min_items, top_k):
    """v_brand = mean(x | brand=b) - mean(x)"""
    global_mean = x.mean(axis=0)
    dirs = {}
    counts = []
    for brand, ids in by_brand.items():
        if len(ids) < min_items:
            continue
        ids_arr = np.array(ids)
        # Filter valid ids (item_id 在 [0, 11924))
        valid = ids_arr[(ids_arr >= 0) & (ids_arr < x.shape[0])]
        if len(valid) < min_items:
            continue
        v = x[valid].mean(axis=0) - global_mean
        n = len(valid)
        dirs[brand] = {'v': v, 'n': n}
        counts.append((brand, n))
    counts.sort(key=lambda x: -x[1])
    top = dict(counts[:top_k])
    return {b: dirs[b] for b in top}


def cos_eucl(v1, v2):
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    return float((v1 @ v2) / (n1 * n2 + 1e-12))


def cos_causal(v1, v2, M):
    """cos_M(v1, v2) = v1.T M v2 / sqrt(v1.T M v1) sqrt(v2.T M v2)"""
    mv1 = M @ v1
    mv2 = M @ v2
    n1 = np.sqrt(max(v1 @ mv1, 1e-12))
    n2 = np.sqrt(max(v2 @ mv2, 1e-12))
    return float((v1 @ mv2) / (n1 * n2 + 1e-12))


def build_M_from_centroids(ckpt_path):
    """M = centroids.T @ centroids (centroid 协方差 proxy)."""
    ck = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    sd = ck['state_dict']
    # Try common centroid keys
    keys = [k for k in sd.keys() if 'centroid' in k.lower() or 'codebook' in k.lower() or 'embed' in k.lower()]
    print(f'  centroid-like keys in ckpt: {keys[:5]}')
    # Get L1 codebook
    for k in keys:
        v = sd[k]
        if hasattr(v, 'shape') and len(v.shape) == 2:
            # Normalize to (D, K)
            t = v.float().numpy()
            if t.shape[0] == 2048 or (t.ndim == 2 and (t.shape[0] == 2048 or t.shape[1] == 2048)):
                if t.shape[0] < t.shape[1]:
                    t = t.T
                # Now t is (D, K)
                print(f'  using {k}, shape={t.shape}')
                return t @ t.T
    # Fallback: use first 2D tensor
    for k, v in sd.items():
        if hasattr(v, 'shape') and len(v.shape) == 2:
            t = v.float().numpy()
            if t.shape[0] < t.shape[1]:
                t = t.T
            if t.shape[0] == 2048:
                print(f'  fallback using {k}, shape={t.shape}')
                return t @ t.T
    raise RuntimeError('No 2D tensor with dim 2048 found')


def main():
    print('=' * 70)
    print('Loading Stage 1 embedding + metadata')
    print('=' * 70)
    x = torch.load(EMB_PATH, map_location='cpu', weights_only=False).float().numpy()
    print(f'  x shape={x.shape}')
    by_brand = json.load(open(BY_BRAND_PATH))
    print(f'  brand groups (≥5 items): {len(by_brand)}')

    # Build brand concept directions
    print('\nBuilding brand concept directions...')
    brand_dirs = build_brand_directions(x, by_brand, MIN_BRAND_ITEMS, N_BRAND_TOP)
    brands = list(brand_dirs.keys())
    print(f'  selected brands (≥{MIN_BRAND_ITEMS} items, top {N_BRAND_TOP}):')
    for b in brands:
        print(f'    {b}: {brand_dirs[b]["n"]} items')

    # Load M (RQ 码本协方差 proxy)
    print('\nBuilding M from RQ codebook (placeholder, will be expanded in 现象 3)...')
    M = build_M_from_centroids(RQ_CKPT_PATH)
    print(f'  M shape={M.shape}, trace={np.trace(M):.2e}, '
          f'frobenius={np.linalg.norm(M):.2e}')

    # Compute pairwise cosines
    print('\n' + '=' * 70)
    print(f'Pairwise cosines for {len(brands)} brand directions')
    print('=' * 70)
    rows = []
    for i in range(len(brands)):
        for j in range(i + 1, len(brands)):
            b1, b2 = brands[i], brands[j]
            v1, v2 = brand_dirs[b1]['v'], brand_dirs[b2]['v']
            ce = cos_eucl(v1, v2)
            cm = cos_causal(v1, v2, M)
            rows.append({'brand_1': b1, 'brand_2': b2,
                         'cos_eucl': ce, 'cos_causal': cm,
                         'abs_diff': abs(ce - cm)})
    rows.sort(key=lambda r: -r['abs_diff'])

    # Stats
    diffs = np.array([r['abs_diff'] for r in rows])
    print(f'\n  |cos_Eucl - cos_causal| stats:')
    print(f'    mean={diffs.mean():.4f}, median={np.median(diffs):.4f}, '
          f'p25={np.quantile(diffs,0.25):.4f}, p75={np.quantile(diffs,0.75):.4f}, '
          f'max={diffs.max():.4f}')

    print(f'\n  Top 10 pairs by |Δcos|:')
    print(f'  {"brand_1":<25s} {"brand_2":<25s} {"cos_Eucl":>10s} {"cos_causal":>10s} {"Δ":>8s}')
    for r in rows[:10]:
        print(f'  {r["brand_1"]:<25s} {r["brand_2"]:<25s} '
              f'{r["cos_eucl"]:>10.4f} {r["cos_causal"]:>10.4f} '
              f'{r["abs_diff"]:>8.4f}')

    # Verdict
    print('\n' + '=' * 70)
    print('VERDICT — Idea B 现象 1 (causal 内积是否带来实质性差异)')
    print('=' * 70)
    p75 = np.quantile(diffs, 0.75)
    p90 = np.quantile(diffs, 0.90)
    # Kill line: |Δcos| < 0.05 对所有测试对
    n_small = (diffs < 0.05).sum()
    n_big = (diffs > 0.20).sum()
    print(f'  pairs: {len(rows)}, |Δcos|<0.05: {n_small} ({n_small/len(rows):.1%}), '
          f'|Δcos|>0.20: {n_big} ({n_big/len(rows):.1%})')
    if n_small == len(rows):
        verdict = '✗ kill (Idea B dead)'
    elif n_big > 0.3 * len(rows):
        verdict = '✓ causal 内积带来显著差异'
    else:
        verdict = '△ 部分差异 (继续看现象 2)'
    print(f'  Verdict: {verdict}')

    # Save
    out_path = os.path.join(OUT_DIR, 'ideaB_phenomenon1.json')
    with open(out_path, 'w') as f:
        json.dump({
            'n_brands': len(brands),
            'brands': brands,
            'M_source': 'RQ_L1_codebook_covariance',
            'M_trace': float(np.trace(M)),
            'M_frobenius': float(np.linalg.norm(M)),
            'pairwise': rows,
            'abs_diff_stats': {
                'mean': float(diffs.mean()),
                'median': float(np.median(diffs)),
                'p25': float(np.quantile(diffs, 0.25)),
                'p50': float(np.quantile(diffs, 0.50)),
                'p75': float(np.quantile(diffs, 0.75)),
                'p90': float(p90),
                'p99': float(np.quantile(diffs, 0.99)),
                'max': float(diffs.max()),
            },
            'kill_verdict': verdict,
            'n_pairs_total': len(rows),
            'n_pairs_diff_lt_0p05': int(n_small),
            'n_pairs_diff_gt_0p20': int(n_big),
        }, f, indent=2)
    print(f'\nSaved → {out_path}')


if __name__ == '__main__':
    main()