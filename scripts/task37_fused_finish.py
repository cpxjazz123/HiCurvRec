#!/usr/bin/env python3
"""Task #37 — fused-only 跑完 task36 剩余 aggregation + user_emb bonus.

memory-safe: 不用 np.broadcast_to 创建 (B, max_h, n_items) 大数组,
改用 numpy einsum / batch-level reduce 避免 OOM.
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
OUT_DIR = ROOT / 'products/task37_fused_finish'
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
        history_asins = [x['asin'] for x in uh[r['reviewerID']] if x['asin'] in asin_to_strid]
        samples.append({
            'user': r['reviewerID'],
            'history_idxs': [int(asin_to_strid[a]) - 1 for a in history_asins[-MAX_HISTORY:]],
            'target_idx': int(asin_to_strid[r['asin']]) - 1,
        })
    return samples


def compute_scores(test_samples, item_emb, item_emb_norm, agg, n_items):
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
        n2 = np.linalg.norm(he, axis=2, keepdims=True) + 1e-12
        he = he / n2
        sims = he @ item_emb_norm.T  # (B, max_h, n_items)
        sims = np.where(m_b[:, :, None], sims, np.float32(-2.0))
        if agg == 'recency_min':
            # memory-safe: einsum with weight matrix (B, max_h) -> per-position weighted sim
            pos = np.arange(max_h)
            # recency: 越近 pos 越大, weight = exp(-(max_h - 1 - pos) / (max_h/3))
            w = np.exp(-(max_h - 1 - pos) / max(max_h / 3, 1))  # (max_h,)
            w_mat = np.where(m_b, w[None, :], 0.0).astype(np.float32)  # (B, max_h)
            # weighted sum across max_h: (B, max_h) * (B, max_h, n_items) -> sum -> (B, n_items)
            # 用 einsum: 'bh,bhn->bn'
            agg_score = np.einsum('bh,bhn->bn', w_mat, sims)
            denom = np.maximum(w_mat.sum(axis=1, keepdims=True), 1e-12)
            agg_score = agg_score / denom
        elif agg == 'topk_mean':
            K = min(3, max_h)
            # topk 沿 axis=1 (max_h): 需要 (B, K, n_items)
            # np.partition 比 sort 快很多: 取 top-K
            part = np.partition(sims, -K, axis=1)[:, -K:, :]  # (B, K, n_items)
            agg_score = part.mean(axis=1)
        elif agg == 'max_cos':
            agg_score = sims.max(axis=1)
        elif agg == 'last_item_cos':
            agg_score = sims[:, -1, :]
        else:
            raise ValueError(agg)
        scores[s:e] = agg_score
    return scores, valid


def user_emb_score(test_samples, fused_user, item_emb_norm, n_items):
    with open(PHONISM_DATA / 'raw/toys/datamaps.json') as f:
        dm = json.load(f)
    user2id = dm['user2id']
    user_norm = np.linalg.norm(fused_user, axis=1, keepdims=True) + 1e-12
    user_emb_norm = fused_user / user_norm
    N = len(test_samples)
    scores = np.zeros((N, n_items), dtype=np.float32)
    valid = np.zeros(N, dtype=bool)
    for i, s in enumerate(test_samples):
        u = s['user']
        if u not in user2id:
            continue
        u_idx = int(user2id[u]) - 1
        if u_idx < 0 or u_idx >= user_emb_norm.shape[0]:
            continue
        scores[i] = user_emb_norm[u_idx] @ item_emb_norm.T
        valid[i] = True
    return scores, valid


def recall_gap(score_mat, target_arr, valid, k_list=(5, 10), n_random=10, seed=42):
    rng = np.random.RandomState(seed)
    N, n_items = score_mat.shape
    R = {K: np.zeros(N, dtype=bool) for K in k_list}
    ranks = np.zeros(N, dtype=np.int64)
    gaps = np.zeros(N)
    s_t = np.zeros(N)
    s_r = np.zeros(N)
    BATCH = 512
    for s in range(0, N, BATCH):
        e = min(s + BATCH, N)
        bs = score_mat[s:e]
        bt = target_arr[s:e]
        ridx = np.arange(bs.shape[0])
        ts = bs[ridx, bt]
        ranks[s:e] = (bs > ts[:, None]).sum(axis=1) + 1
        for K in k_list:
            top = np.argpartition(-bs, K, axis=1)[:, :K]
            for i in range(e - s):
                if bt[i] in top[i]:
                    R[K][s + i] = True
    for i in range(N):
        if not valid[i]:
            continue
        target = target_arr[i]
        row = score_mat[i]
        st = row[target]
        r_idx = []
        while len(r_idx) < n_random:
            c = rng.randint(0, n_items)
            if c != target:
                r_idx.append(c)
        sr = row[r_idx].mean()
        gaps[i] = st - sr
        s_t[i] = st
        s_r[i] = sr
    if valid.sum() == 0:
        return None
    return {
        'R@5': float(R[5][valid].mean()),
        'R@10': float(R[10][valid].mean()),
        'mean_rank': float(ranks[valid].mean()),
        'mean_gap': float(gaps[valid].mean()),
        'median_gap': float(np.median(gaps[valid])),
        'gap_p25': float(np.percentile(gaps[valid], 25)),
        'gap_p75': float(np.percentile(gaps[valid], 75)),
        'mean_s_to_target': float(s_t[valid].mean()),
        'mean_s_to_random': float(s_r[valid].mean()),
        'n_valid': int(valid.sum()),
    }


def main():
    print('========== Task #37 fused-only finish (memory-safe) ==========')
    print()
    test_samples = load_test()
    targets = np.asarray([s['target_idx'] for s in test_samples])
    print(f'test samples: {len(test_samples)}')

    mckg = torch.load(MCKG_EMB_PATH, weights_only=False)
    fused_item = mckg['fused_item'].numpy()
    fused_user = mckg['fused_user'].numpy()
    n_items = fused_item.shape[0]
    item_norm = fused_item / (np.linalg.norm(fused_item, axis=1, keepdims=True) + 1e-12)

    results = {}
    for agg in ['recency_min', 'topk_mean', 'max_cos', 'last_item_cos']:
        score_mat, valid = compute_scores(test_samples, fused_item, item_norm, agg, n_items)
        info = recall_gap(score_mat, targets, valid)
        results[f'fused/{agg}'] = info
        print(f'fused/{agg:14s}: R@5={info["R@5"]:.4f} R@10={info["R@10"]:.4f} '
              f'gap={info["mean_gap"]:+.4f} (p25={info["gap_p25"]:+.4f} p75={info["gap_p75"]:+.4f})')

    # bonus: user_emb
    print()
    print('--- Bonus: MCKG pretrained user_emb vs fused_item ---')
    score_mat, valid = user_emb_score(test_samples, fused_user, item_norm, n_items)
    info = recall_gap(score_mat, targets, valid)
    results['fused/user_emb'] = info
    print(f'fused/user_emb   : R@5={info["R@5"]:.4f} R@10={info["R@10"]:.4f} '
          f'gap={info["mean_gap"]:+.4f}')

    # 落盘
    out = OUT_DIR / 'task37_summary.json'
    summary = {
        'task': 'Task #37 — fused-only finish (memory-safe)',
        'task36_inherited_results': 'subspace_0/1/2 + fused min_cos/mean_cos (22/25 done)',
        'results': results,
    }
    with open(out, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f'\n输出: {out}')

    print('\n========== 全 25 aggregation 汇总 (task36+37) ==========')
    print('T5 baseline: gap=0.0002, R@5=0.00107')
    print('MCKG per-item (history 截断到 30):')
    for k, v in results.items():
        marker = '🟢 STRONG' if abs(v['mean_gap']) > 0.02 else (
            '🟡 WEAK' if abs(v['mean_gap']) > 0.001 else '🔴 NONE'
        )
        print(f'  {marker}  {k:30s}: gap={v["mean_gap"]:+.4f} R@5={v["R@5"]:.4f} R@10={v["R@10"]:.4f}')


if __name__ == '__main__':
    main()