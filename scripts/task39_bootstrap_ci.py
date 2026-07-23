#!/usr/bin/env python3
"""Task #39 — R@K bootstrap CI (T5 baseline vs MCKG STRONG spaces).

对每个 test sample 的 R@5 indicator (1 if target in top-5 else 0) 做 bootstrap.
比较 T5 baseline (Task #132) 与 MCKG subspace_0 / subspace_1 / fused 的 R@5 提升是否统计显著.
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
OUT_DIR = ROOT / 'products/task39_bootstrap'
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


def compute_min_cos(test_samples, item_emb, item_emb_norm, n_items):
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


def per_user_r5(score_mat, target_arr, valid, k=5):
    N = score_mat.shape[0]
    r_at_k = np.zeros(N, dtype=bool)
    for s in range(0, N, 512):
        e = min(s + 512, N)
        bs = score_mat[s:e]
        bt = target_arr[s:e]
        top = np.argpartition(-bs, k, axis=1)[:, :k]
        for i in range(e - s):
            if bt[i] in top[i]:
                r_at_k[s + i] = True
    return r_at_k & valid


def bootstrap_diff_ci(t5_hits, mckg_hits, n_boot=10000, ci=99, seed=42):
    """Bootstrap CI of (mckg_R@5 - t5_R@5)."""
    rng = np.random.RandomState(seed)
    n = len(t5_hits)
    diffs = np.zeros(n_boot)
    for b in range(n_boot):
        idx = rng.randint(0, n, size=n)
        diffs[b] = mckg_hits[idx].mean() - t5_hits[idx].mean()
    lo = np.percentile(diffs, (100 - ci) / 2)
    hi = np.percentile(diffs, 100 - (100 - ci) / 2)
    return float(lo), float(hi), float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))


def main():
    print('========== Task #39 R@K bootstrap CI ==========')
    test_samples = load_test()
    targets = np.asarray([s['target_idx'] for s in test_samples])
    print(f'test samples: {len(test_samples)}')

    mckg = torch.load(MCKG_EMB_PATH, weights_only=False)
    sub_item = mckg['subspace_item'].numpy()
    fused_item = mckg['fused_item'].numpy()
    n_items = fused_item.shape[0]

    # MCKG 三个空间
    spaces = {
        'subspace_0_sphere': sub_item[0],
        'subspace_1_euclid': sub_item[1],
        'fused_64d': fused_item,
    }

    # 模拟 T5 baseline: 假设 hits 是 sample-wise R@5 indicator, baseline 命中率 = 0.00107
    # 我们没 T5 的 per-test hit indicator, 但可以从 task36 fused min_cos 的 rank 推断一个近似
    # 简化: 直接用 fused min_cos 的 hits, R@5=0.0005 (实际 task36), 并报 MCKG vs random baseline
    # 更严格: T5 baseline (R@5=0.00107, gap=0.0002) — 假设 R@5 hits 全是均匀随机抽取的,
    # 因为 t5 本身不做 dense retrieval 做 SID 检索, 没法直接 per-user indicator.
    # 这里改为: 对每个 space, bootstrap 它的 R@5 的绝对 CI (作为参考),
    # 顺便计算 space-vs-space 差值 (因全部 MCKG 用同一 test set, 配对 bootstrap).

    results = {}
    space_hits = {}
    for name, item_emb in spaces.items():
        norm = item_emb / (np.linalg.norm(item_emb, axis=1, keepdims=True) + 1e-12)
        score_mat, valid = compute_min_cos(test_samples, item_emb, norm, n_items)
        hits = per_user_r5(score_mat, targets, valid, k=5)
        space_hits[name] = hits
        # bootstrap absolute CI
        rng = np.random.RandomState(42)
        n = hits.sum()
        N = len(hits)
        boot_means = np.zeros(10000)
        for b in range(10000):
            idx = rng.randint(0, N, size=N)
            boot_means[b] = hits[idx].mean()
        ci_lo = float(np.percentile(boot_means, 0.5))
        ci_hi = float(np.percentile(boot_means, 99.5))
        results[name] = {
            'R@5': float(hits.mean()),
            'n_hits': int(hits.sum()),
            'n_valid': int(valid.sum()),
            'bootstrap_99ci_abs': [ci_lo, ci_hi],
        }
        print(f'\n{name}: R@5={results[name]["R@5"]:.4f}, '
              f'99% CI=[{ci_lo:.4f}, {ci_hi:.4f}] (n_hits={int(hits.sum())})')

    # 配对差值 (subspace_1 vs subspace_0, fused vs subspace_1)
    print('\n--- 配对 bootstrap 差值 ---')
    pairs = [
        ('subspace_1_euclid', 'subspace_0_sphere'),
        ('fused_64d', 'subspace_1_euclid'),
        ('fused_64d', 'subspace_0_sphere'),
    ]
    diff_results = {}
    for a, b in pairs:
        lo, hi, lo95, hi95 = bootstrap_diff_ci(space_hits[b], space_hits[a], n_boot=10000)
        diff_results[f'{a} - {b}'] = {
            'diff_R@5': results[a]['R@5'] - results[b]['R@5'],
            'bootstrap_99ci_diff': [lo, hi],
            'bootstrap_95ci_diff': [lo95, hi95],
        }
        print(f'\n{a} - {b}:')
        print(f'  ΔR@5 = {results[a]["R@5"] - results[b]["R@5"]:+.4f}')
        print(f'  99% CI = [{lo:+.4f}, {hi:+.4f}]')
        sig = '✅ 显著' if lo > 0 else ('❌ 不显著' if hi < 0 else '⚠️ 不确定 (CI 跨 0)')
        print(f'  {sig}')

    out = OUT_DIR / 'task39_bootstrap.json'
    summary = {
        'task': 'Task #39 — R@K bootstrap CI',
        'method': 'per-user R@5 indicator, paired bootstrap diff',
        't5_baseline_R5': 0.00107,  # Task #132 报告值, 无 per-user indicator 可对比
        'note': 'T5 baseline 没有 per-user R@5 indicator, 这里只比较 MCKG 三个 STRONG 空间的成对差',
        'absolute_results': results,
        'paired_diff_results': diff_results,
    }
    with open(out, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f'\n输出: {out}')


if __name__ == '__main__':
    main()