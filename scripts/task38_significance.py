#!/usr/bin/env python3
"""Task #38 — STRONG space 显著性检验 (subspace_0/1 + fused, n_random=50 + per-test bootstrap).

复核: task36 显示 sphere / euclid / fused 三空间 min_cos gap 达 STRONG (≥ 0.02).
本任务用 n_random=50 (vs task36 的 10) + 99% bootstrap CI 验证 gap 稳定性.
"""
import json
import pickle
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

ROOT = Path('/home/wlia0047/ar57/wenyu/GeneRec')
PHONISM_DATA = Path('/home/wlia0047/ar57/wenyu/genrec/dataset/amazon')
MCKG_EMB_PATH = ROOT / 'products/task99_mckg_rebuild/entity_embedding.pt'
OUT_DIR = ROOT / 'products/task38_significance'
OUT_DIR.mkdir(exist_ok=True)

MAX_HISTORY = 30
SCORE_BATCH = 64


def load_test():
    with open(PHONISM_DATA / 'raw/toys/datamaps.json') as f:
        dm = json.load(f)
    asin_to_strid = dm['item2id']
    with open(PHONISM_DATA / 'raw/toys/review_splits.pkl', 'rb') as f:
        splits = pickle.load(f)
    uh = defaultdict(list)
    for r in splits['train']:
        if r['asin'] in asin_to_strid:
            uh[r['reviewerID']].append(r)
    for u in uh:
        uh[u].sort(key=lambda x: x['unixReviewTime'])
    samples = []
    for r in splits['test']:
        if r['asin'] not in asin_to_strid or r['reviewerID'] not in uh:
            continue
        ha = [x['asin'] for x in uh[r['reviewerID']] if x['asin'] in asin_to_strid]
        samples.append({
            'user': r['reviewerID'],
            'history_idxs': [int(asin_to_strid[a]) - 1 for a in ha[-MAX_HISTORY:]],
            'target_idx': int(asin_to_strid[r['asin']]) - 1,
        })
    return samples


def compute_min_cos_scores(test_samples, item_emb, item_emb_norm, n_items):
    """max over history of cosine (等价 min_cos gap = max-pool)."""
    N = len(test_samples)
    max_h = max(len(s['history_idxs']) for s in test_samples)
    hist_mat = np.full((N, max_h), -1, dtype=np.int64)
    hist_mask = np.zeros((N, max_h), dtype=bool)
    valid = np.zeros(N, dtype=bool)
    for i, s in enumerate(test_samples):
        h = s['history_idxs']
        if not h:
            continue
        hist_mat[i, :len(h)] = h
        hist_mask[i, :len(h)] = True
        valid[i] = True
    scores = np.zeros((N, n_items), dtype=np.float32)
    for s in range(0, N, SCORE_BATCH):
        e = min(s + SCORE_BATCH, N)
        B = e - s
        h_b = hist_mat[s:e]
        m_b = hist_mask[s:e]
        flat = h_b.flatten()
        he = item_emb[flat].reshape(B, max_h, -1)
        he = he / (np.linalg.norm(he, axis=2, keepdims=True) + 1e-12)
        sims = he @ item_emb_norm.T
        sims = np.where(m_b[:, :, None], sims, np.float32(-2.0))
        scores[s:e] = sims.max(axis=1)
    return scores, valid


def gap_per_test(score_mat, target_arr, valid, n_random=50, seed=42):
    rng = np.random.RandomState(seed)
    N, n_items = score_mat.shape
    gaps = np.zeros(N)
    for i in range(N):
        if not valid[i]:
            continue
        row = score_mat[i]
        target = target_arr[i]
        st = row[target]
        r_idx = []
        while len(r_idx) < n_random:
            c = rng.randint(0, n_items)
            if c != target:
                r_idx.append(c)
        sr = row[r_idx].mean()
        gaps[i] = st - sr
    return gaps[valid]


def bootstrap_ci(gaps, n_boot=10000, ci=99, seed=42):
    rng = np.random.RandomState(seed)
    n = len(gaps)
    boot_means = np.zeros(n_boot)
    for b in range(n_boot):
        idx = rng.randint(0, n, size=n)
        boot_means[b] = gaps[idx].mean()
    lo = np.percentile(boot_means, (100 - ci) / 2)
    hi = np.percentile(boot_means, 100 - (100 - ci) / 2)
    return float(lo), float(hi), float(np.percentile(boot_means, 2.5)), float(np.percentile(boot_means, 97.5))


def main():
    print('========== Task #38 STRONG space 显著性检验 ==========')
    test_samples = load_test()
    targets = np.asarray([s['target_idx'] for s in test_samples])
    print(f'test samples: {len(test_samples)}')

    mckg = torch.load(MCKG_EMB_PATH, weights_only=False)
    sub_item = mckg['subspace_item'].numpy()
    fused_item = mckg['fused_item'].numpy()
    n_items = fused_item.shape[0]

    spaces = {
        'subspace_0_sphere': sub_item[0],
        'subspace_1_euclid': sub_item[1],
        'fused(64d)': fused_item,
    }

    results = {}
    for name, item_emb in spaces.items():
        norm = item_emb / (np.linalg.norm(item_emb, axis=1, keepdims=True) + 1e-12)
        score_mat, valid = compute_min_cos_scores(test_samples, item_emb, norm, n_items)
        gaps = gap_per_test(score_mat, targets, valid, n_random=50)
        ci99_lo, ci99_hi, ci95_lo, ci95_hi = bootstrap_ci(gaps, n_boot=5000, ci=99)
        results[name] = {
            'n_valid': int(valid.sum()),
            'mean_gap': float(gaps.mean()),
            'median_gap': float(np.median(gaps)),
            'gap_std': float(gaps.std()),
            'p25': float(np.percentile(gaps, 25)),
            'p75': float(np.percentile(gaps, 75)),
            'frac_positive_gap': float((gaps > 0).mean()),
            'frac_large_gap_>=0.02': float((gaps >= 0.02).mean()),
            'bootstrap_99ci': [ci99_lo, ci99_hi],
            'bootstrap_95ci': [ci95_lo, ci95_hi],
        }
        r = results[name]
        print(f'\n{name} (n_random=50, min_cos):')
        print(f'  mean gap = {r["mean_gap"]:+.4f}, median = {r["median_gap"]:+.4f}, std = {r["gap_std"]:.4f}')
        print(f'  p25/p75 = {r["p25"]:+.4f} / {r["p75"]:+.4f}')
        print(f'  99% bootstrap CI = [{r["bootstrap_99ci"][0]:+.4f}, {r["bootstrap_99ci"][1]:+.4f}]')
        print(f'  95% bootstrap CI = [{r["bootstrap_95ci"][0]:+.4f}, {r["bootstrap_95ci"][1]:+.4f}]')
        print(f'  frac positive = {r["frac_positive_gap"]:.3f}, frac ≥ 0.02 = {r["frac_large_gap_>=0.02"]:.3f}')

    # 与 T5 baseline (gap=0.0002) 对比
    t5_gap = 0.0002
    print(f'\n========== vs T5 baseline gap={t5_gap} ==========')
    for name, r in results.items():
        ci_lo, ci_hi = r['bootstrap_99ci']
        # 显著高于 T5: 99% CI 下限 > T5 gap
        sig_strong = ci_lo > 0.02
        sig_above_t5 = ci_lo > t5_gap * 10  # 10x T5 gap
        sig_above_t5_pct = (r['mean_gap'] / t5_gap) * 100 if t5_gap > 0 else 0
        print(f'{name}:')
        print(f'  99% CI [{ci_lo:+.4f}, {ci_hi:+.4f}]')
        print(f'  gap 是 T5 的 {sig_above_t5_pct:.1f}x')
        print(f'  显著高于 STRONG 阈值 0.02: {"✅" if sig_strong else "❌"}')

    out = OUT_DIR / 'task38_significance.json'
    summary = {
        'task': 'Task #38 — STRONG space significance (n_random=50 + bootstrap CI)',
        't5_baseline_gap': t5_gap,
        'results': results,
    }
    with open(out, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f'\n输出: {out}')


if __name__ == '__main__':
    main()