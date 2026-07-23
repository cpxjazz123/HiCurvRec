#!/usr/bin/env python3
"""Task #36 — MCKG embedding 重复 Task #132 完整诊断 (CORRECTED PROTOCOL).

正确的 dense retrieval protocol:
  对每个 test query (user 的 history 序列), 对 corpus 中每个 candidate item c,
  score(query, c) = aggregate_{h in history} cosine(h, c)
  而非 history centroid -> target.

聚合方式 (score_aggregation):
  - 'min_cos'         : max over history (any one history item similar -> top)
  - 'mean_cos'        : 平均所有 history item 的相似度
  - 'recency_min'     : recency-weighted aggregation (近因权重高)
  - 'topk_mean'       : top-K history item similarities 的均值
  - 'max_cos'         : max cosine over history (等价 max-pool over sim)
  - 'last_item_cos'   : 只看最后一个 history item 的 cosine

Smoking gun (Task #132 同样):
  score(query, target) - mean(score(query, n_random 个 random items))
  与 T5 任务 gap=0.0002 对比.

性能约束:
  phonism Toys user history p99=88, max=416, 截断到最近 MAX_HISTORY=30 项:
    p90=27, p99=88 范围内的多数被完整覆盖, 极长尾截断 (417 -> 30) 显著降 batch 矩阵 size.
  在 batch=64, MAX_HISTORY=30 下: sim 矩阵 ~ (64, 30, 11924) ~ 92 MB float32, OK.
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
OUTPUT_DIR = ROOT / 'products/task36_mckg_smoking_gun'
OUTPUT_DIR.mkdir(exist_ok=True)

MAX_HISTORY = 30  # 截断到最近 30 个 history item (p99=88, max=416)
SCORE_BATCH = 64  # 相似度矩阵 batch 大小


def load_phonism_test_set():
    """复用 Task #131 的 test 集构造逻辑 (phonism Toys next-item target)."""
    with open(PHONISM_DATA / 'raw/toys/datamaps.json') as f:
        dm = json.load(f)
    asin_to_strid = dm['item2id']

    with open(PHONISM_DATA / 'raw/toys/review_splits.pkl', 'rb') as f:
        splits = pickle.load(f)

    user_history = defaultdict(list)
    for r in splits['train']:
        if r['asin'] in asin_to_strid:
            user_history[r['reviewerID']].append(r)
    for u in user_history:
        user_history[u].sort(key=lambda x: x['unixReviewTime'])

    test_samples = []
    for r in splits['test']:
        target_asin = r['asin']
        user = r['reviewerID']
        if target_asin not in asin_to_strid:
            continue
        if user not in user_history or len(user_history[user]) == 0:
            continue
        history_asins = [x['asin'] for x in user_history[user] if x['asin'] in asin_to_strid]
        target_idx = int(asin_to_strid[target_asin]) - 1
        # 截断到最近 MAX_HISTORY 项 (按 unixReviewTime 已排序, 末尾为最近)
        history_truncated = [int(asin_to_strid[a]) - 1 for a in history_asins[-MAX_HISTORY:]]
        test_samples.append({
            'user': user,
            'history_idxs': history_truncated,
            'target_idx': target_idx,
        })
    return test_samples


def compute_per_item_score(test_samples, item_emb, item_emb_norm, score_aggregation,
                            target_idxs, n_items):
    """Per-item protocol: 计算每 test sample 对每 candidate 的聚合分数.

    Returns:
      score_matrix: (N, n_items) - 每个 (query, item) 的聚合分数
      valid_mask: (N,) bool
    """
    N = len(test_samples)

    # 在 load_phonism_test_set 已经截断到 MAX_HISTORY, 这里取实际 max
    max_h = max((len(s['history_idxs']) for s in test_samples if len(s['history_idxs']) > 0),
                default=1)

    history_matrix = np.full((N, max_h), -1, dtype=np.int64)
    history_mask = np.zeros((N, max_h), dtype=bool)
    valid_mask = np.zeros(N, dtype=bool)

    for i, s in enumerate(test_samples):
        hist = s['history_idxs']
        n_h = len(hist)
        if n_h == 0:
            continue
        history_matrix[i, :n_h] = hist
        history_mask[i, :n_h] = True
        valid_mask[i] = True

    score_matrix = np.zeros((N, n_items), dtype=np.float32)

    for s in range(0, N, SCORE_BATCH):
        e = min(s + SCORE_BATCH, N)
        h_batch = history_matrix[s:e]  # (B, max_h)
        m_batch = history_mask[s:e]    # (B, max_h)

        flat_indices = h_batch.flatten()
        hist_embs = item_emb[flat_indices].reshape(e - s, max_h, -1)
        # L2-normalize
        norms = np.linalg.norm(hist_embs, axis=2, keepdims=True) + 1e-12
        hist_embs = hist_embs / norms

        # (B, max_h, D) @ (D, n_items) -> (B, max_h, n_items)
        sims = hist_embs @ item_emb_norm.T
        # mask 掉 padding (m_batch=False 的 sim 置为 -2)
        sims = np.where(m_batch[:, :, None], sims, np.float32(-2.0))

        if score_aggregation == 'min_cos':
            agg = sims.max(axis=1)
        elif score_aggregation == 'mean_cos':
            n_h_per_row = m_batch.sum(axis=1, keepdims=True)
            agg = (sims * m_batch[:, :, None]).sum(axis=1) / np.maximum(n_h_per_row, 1)
        elif score_aggregation == 'recency_min':
            # recency weight: 越近的 history item 权重越高
            w = np.exp(-np.arange(max_h)[::-1] / max(max_h / 3, 1))
            w = w[None, :, None]  # (1, max_h, 1)
            ww = np.where(m_batch[:, :, None], w, np.float32(0.0))  # (B, max_h, 1)
            # reduce (B, max_h, n_items) over max_h axis -> (B, n_items)
            ww_full = np.broadcast_to(ww, sims.shape)  # (B, max_h, n_items)
            numer = (sims * ww_full).sum(axis=1)  # (B, n_items)
            ww_sum = ww.sum(axis=1).squeeze(-1)  # (B,)
            agg = numer / np.maximum(ww_sum[:, None], 1e-12)  # (B, n_items)
        elif score_aggregation == 'topk_mean':
            K = min(3, max_h)
            # 取每个 (B, max_h, n_items) 沿 max_h 维度的 top-K 均值 -> (B, n_items)
            sims_sorted = np.sort(sims, axis=1)[:, -K:, :]
            agg = sims_sorted.mean(axis=1)
        elif score_aggregation == 'max_cos':
            agg = sims.max(axis=1)
        elif score_aggregation == 'last_item_cos':
            agg = sims[:, -1, :]
        else:
            raise ValueError(f'Unknown score_aggregation: {score_aggregation}')

        score_matrix[s:e] = agg

    return score_matrix, valid_mask


def cosine_recall_from_scores(score_matrix, target_idxs, k_list=(5, 10), valid_mask=None):
    """从预聚合的 (N, M) score 矩阵算 R@K + mean rank.

    标准做法: query 是 history 的某种表示, target 是 corpus 中实际下一个 item.
    R@K = fraction(test_sample, target in top-K by score).
    """
    target_arr = np.asarray(target_idxs)
    n = score_matrix.shape[0]
    R = {K: np.zeros(n, dtype=bool) for K in k_list}
    ranks = np.zeros(n, dtype=np.int64)

    BATCH = 512
    for s in range(0, n, BATCH):
        e = min(s + BATCH, n)
        batch_score = score_matrix[s:e]
        batch_target = target_arr[s:e]
        row_idx = np.arange(batch_score.shape[0])
        target_score = batch_score[row_idx, batch_target]
        ranks[s:e] = (batch_score > target_score[:, None]).sum(axis=1) + 1
        for K in k_list:
            top_idx = np.argpartition(-batch_score, K, axis=1)[:, :K]
            for i in range(e - s):
                if batch_target[i] in top_idx[i]:
                    R[K][s + i] = True

    if valid_mask is not None:
        for K in k_list:
            R[K] = R[K][valid_mask]
        ranks = ranks[valid_mask]

    recalls = {K: r.mean() for K, r in R.items()}
    mean_rank = ranks.mean() if len(ranks) > 0 else -1.0
    return recalls, mean_rank


def compute_user_emb_score(test_samples, user_emb_norm, item_emb_norm, target_idxs, n_items):
    """直接用 MCKG user_emb (fused_user) 作为单向量 query, score = cosine(user, item)."""
    with open(PHONISM_DATA / 'raw/toys/datamaps.json') as f:
        dm = json.load(f)
    user2id = dm['user2id']

    N = len(test_samples)
    score_matrix = np.zeros((N, n_items), dtype=np.float32)
    valid_mask = np.zeros(N, dtype=bool)

    for i, s in enumerate(test_samples):
        u = s['user']
        if u not in user2id:
            continue
        u_idx = int(user2id[u]) - 1
        if u_idx < 0 or u_idx >= user_emb_norm.shape[0]:
            continue
        score_matrix[i] = user_emb_norm[u_idx] @ item_emb_norm.T
        valid_mask[i] = True

    return score_matrix, valid_mask


def smoking_gun_gap_from_scores(score_matrix, target_idxs, valid_mask, n_random=10, seed=42):
    """core smoking gun, 从预聚合 score 矩阵算.

    对每个 test query:
      s_target = score(query, target)
      s_random = mean over n_random random items (排除 target)
      gap = s_target - s_random

    注: 在 per-item protocol 下, score 已经是 "history 中任一相关 item 决定查询分数".
    gap 衡量 target 是否在 history 物品的"高相似聚簇"中.
    """
    rng = np.random.RandomState(seed)
    target_arr = np.asarray(target_idxs)
    N = score_matrix.shape[0]
    n_items = score_matrix.shape[1]

    gaps = np.zeros(N)
    s_target = np.zeros(N)
    s_random = np.zeros(N)

    for i in range(N):
        if not valid_mask[i]:
            continue
        target = target_arr[i]
        row = score_matrix[i]
        st = row[target]
        random_idx = []
        while len(random_idx) < n_random:
            c = rng.randint(0, n_items)
            if c != target:
                random_idx.append(c)
        sr = row[random_idx].mean()
        gaps[i] = st - sr
        s_target[i] = st
        s_random[i] = sr

    valid = valid_mask
    if valid.sum() == 0:
        return {
            'mean_gap': 0.0, 'median_gap': 0.0,
            'mean_cos_to_target': 0.0, 'mean_cos_to_random': 0.0,
            'n_valid': 0,
        }
    return {
        'gap_per_test': gaps[valid],
        'cos_to_target_per_test': s_target[valid],
        'cos_to_random_per_test': s_random[valid],
        'mean_gap': float(gaps[valid].mean()),
        'median_gap': float(np.median(gaps[valid])),
        'mean_cos_to_target': float(s_target[valid].mean()),
        'mean_cos_to_random': float(s_random[valid].mean()),
        'n_valid': int(valid.sum()),
    }


def main():
    print('========== Task #36 MCKG smoking gun (CORRECTED PROTOCOL) ==========')
    print('Per-item cosine → aggregate (NOT history centroid → target)')
    print()

    print('\n[1] 加载 phonism Toys test set (next-item target) ...')
    test_samples = load_phonism_test_set()
    target_idxs = np.asarray([s['target_idx'] for s in test_samples])
    print(f'  test samples: {len(test_samples)}')

    print('\n[2] 加载 MCKG embedding (3 sub + fused) ...')
    mckg = torch.load(MCKG_EMB_PATH, weights_only=False)
    subspace_item = mckg['subspace_item'].numpy()
    fused_item = mckg['fused_item'].numpy()
    fused_user = mckg['fused_user'].numpy()
    kappas = mckg['kappas']
    print(f'  kappas = {kappas}')

    N_items = fused_item.shape[0]

    print('\n[3] 6 score aggregations × 4 spaces × R@5/10 + smoking gun ...')

    score_aggs = ['min_cos', 'mean_cos', 'recency_min', 'topk_mean', 'max_cos', 'last_item_cos']
    spaces = {
        'subspace_0_sphere(κ+5.05)': subspace_item[0],
        'subspace_1_euclid(κ-0.08)': subspace_item[1],
        'subspace_2_hyperbolic(κ-5.04)': subspace_item[2],
        'fused(64d)': fused_item,
    }

    results = {}
    for space_name, item_emb in spaces.items():
        print(f'\n--- Space: {space_name} (D={item_emb.shape[1]}) ---')
        norms = np.linalg.norm(item_emb, axis=1, keepdims=True) + 1e-12
        item_emb_norm = item_emb / norms

        for agg in score_aggs:
            score_mat, valid_mask = compute_per_item_score(
                test_samples, item_emb, item_emb_norm, agg, target_idxs, N_items,
            )
            recalls, mean_rank = cosine_recall_from_scores(
                score_mat, target_idxs, k_list=(5, 10), valid_mask=valid_mask)
            gap_info = smoking_gun_gap_from_scores(score_mat, target_idxs, valid_mask)
            results[f'{space_name}/{agg}'] = {
                'R@5': recalls[5],
                'R@10': recalls[10],
                'mean_rank': mean_rank,
                **gap_info,
            }
            print(f'  {agg:18s}: R@5={recalls[5]:.4f} R@10={recalls[10]:.4f} '
                  f'gap={gap_info["mean_gap"]:+.4f} '
                  f'(s_t={gap_info["mean_cos_to_target"]:.4f}, '
                  f's_r={gap_info["mean_cos_to_random"]:.4f})')

        if 'fused' in space_name:
            print(f'\n  --- Bonus: MCKG pretrained user_emb (vs fused_item) ---')
            user_norms = np.linalg.norm(fused_user, axis=1, keepdims=True) + 1e-12
            user_emb_norm = fused_user / user_norms
            score_mat_u, valid_u = compute_user_emb_score(
                test_samples, user_emb_norm, item_emb_norm, target_idxs, N_items)
            recalls, mean_rank = cosine_recall_from_scores(
                score_mat_u, target_idxs, k_list=(5, 10), valid_mask=valid_u)
            gap_info = smoking_gun_gap_from_scores(score_mat_u, target_idxs, valid_u)
            results[f'{space_name}/user_emb'] = {
                'R@5': recalls[5],
                'R@10': recalls[10],
                'mean_rank': mean_rank,
                **gap_info,
            }
            print(f'  {"user_emb":18s}: R@5={recalls[5]:.4f} R@10={recalls[10]:.4f} '
                  f'gap={gap_info["mean_gap"]:+.4f} '
                  f'(s_t={gap_info["mean_cos_to_target"]:.4f}, '
                  f's_r={gap_info["mean_cos_to_random"]:.4f})')

    print('\n[4] T5 baseline contrast (Task #132 已报 gap=0.0002, R@5=0.00107)')

    summary = {
        'task': 'Task #36 — MCKG smoking gun (per-item corrected protocol)',
        'protocol_note': 'per-item cosine + aggregate (not history centroid -> target)',
        'mckg_geometry': {
            'kappas': kappas,
            'subspace_dims': int(subspace_item.shape[2]),
            'fused_dim': int(fused_item.shape[1]),
        },
        'task132_t5_baseline': {
            'mean_cos_to_target': 0.9182,
            'mean_cos_to_random': 0.9180,
            'mean_gap': 0.0002,
            'R@5_pre_quant': 0.00107,
        },
        'results': {},
    }
    for k, v in results.items():
        summary['results'][k] = {
            'R@5': float(v['R@5']),
            'R@10': float(v['R@10']),
            'mean_rank': float(v['mean_rank']),
            'mean_gap': float(v['mean_gap']),
            'median_gap': float(v['median_gap']),
            'mean_s_to_target': float(v['mean_cos_to_target']),
            'mean_s_to_random': float(v['mean_cos_to_random']),
            'n_valid': int(v['n_valid']),
        }

    out_json = OUTPUT_DIR / 'task36_summary.json'
    with open(out_json, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f'\n输出: {out_json}')

    print('\n========== Summary ==========')
    print(f'T5 baseline (Task #132):  gap=0.0002, R@5=0.00107')
    print(f'\nMCKG per-item results:')
    for space_name in spaces.keys():
        print(f'\n{space_name}:')
        for agg in score_aggs:
            r = results[f'{space_name}/{agg}']
            marker = '🟢 STRONG' if abs(r['mean_gap']) > 0.02 else \
                     ('🟡 WEAK' if abs(r['mean_gap']) > 0.001 else '🔴 NONE')
            print(f'  {marker}  {agg:18s}: gap={r["mean_gap"]:+.4f}, R@5={r["R@5"]:.4f}, R@10={r["R@10"]:.4f}')


if __name__ == '__main__':
    main()