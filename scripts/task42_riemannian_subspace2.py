#!/usr/bin/env python3
"""Task #42 — 用 stereographic Riemannian 距离重测 subspace_2_hyperbolic 的 per-item dense retrieval.

目的: 验证 Task #36 报 subspace_2 gap=+0.0160 (WEAK) 是 metric 不匹配 (Euclidean cosine 测双曲空间不准)
      还是 subspace_2 本身信号弱。

MCKG 用 stereographic/Poincaré 距离 (公式 7): d_κ(u, v) = 2 · tan⁻¹_κ(||(-u) ⊕κ v||)
  - κ > 0 (sphere): tan⁻¹_κ(x) = arctan(√κ · x) / √κ
  - κ < 0 (hyperbolic): tan⁻¹_κ(x) = arctanh(√|κ| · x) / √|κ|
  - κ = 0 (euclidean): tan⁻¹_κ(x) = x

Protocol 与 Task #36 完全一致,只把 cosine 换成 dist_kappa。

agg | score_i = -dist(query_i, target)   (距离越小分数越高)
score_per_user = agg over history items
"""
import json
import pickle
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

ROOT = Path('/home/wlia0047/ar57/wenyu/GeneRec')
PHONISM_DATA = Path('/home/wlia0047/ar57/wenyu/genrec/dataset/amazon')
MCKG_EMB_PATH = ROOT / 'products/task99_mckg_rebuild/entity_embedding.pt'
MCKG_STEREO = ROOT / 'task_artifacts/scripts/mckg_model/stereographic.py'
OUT_DIR = ROOT / 'products/task42_riemannian'
OUT_DIR.mkdir(exist_ok=True)

MAX_HISTORY = 30
SCORE_BATCH = 64
N_RANDOM = 50  # for smoking gun gap baseline
SEED = 42


def load_stereographic():
    """动态 import MCKG 的 stereographic 模块 (在 task_artifacts 下)."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("stereographic", str(MCKG_STEREO))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_test():
    """复用 Task #36 协议: phonism TIGER Toys test set, history 截断到最近 30."""
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


def compute_riemannian_scores(test_samples, item_emb, stereo_mod, kappa):
    """Per-item Riemannian protocol.

    item_emb: (N, D) numpy array
    stereo_mod.dist_kappa(u, v, kappa) -> distance
    返回: scores (N_test, N_items) — 越大越相关,包含 mask (history 长度)
    """
    N = len(test_samples)
    n_items = item_emb.shape[0]
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

    # 转为 torch (因为 dist_kappa 是 torch op)
    item_emb_t = torch.from_numpy(item_emb).float()

    scores = np.zeros((N, n_items), dtype=np.float32)
    for s in range(0, N, SCORE_BATCH):
        e = min(s + SCORE_BATCH, N)
        B = e - s
        h_b = hist_mat[s:e]
        m_b = hist_mask[s:e]

        flat = h_b.flatten()
        he_t = item_emb_t[flat].reshape(B, max_h, -1)  # (B, max_h, D)
        # Compute distance: for each (b, h, j) -> dist(he_t[b,h,:], item_emb_t[j,:])
        # 用 einsum 拆分: 先扩成 (B, max_h, N, D) 再算 pairwise dist (内存太大)
        # 替代: 对每个 j 单独算 (B, max_h, D) -> (B, max_h) dist, 拼成 (B, max_h, N)
        # 但这样要 N 次循环
        # 用 chunked: 把 N 拆成 chunks
        n_chunks = max(1, n_items // 256)
        chunk_size = (n_items + n_chunks - 1) // n_chunks
        dist_bh_n = np.full((B, max_h, n_items), np.inf, dtype=np.float32)
        for j_start in range(0, n_items, chunk_size):
            j_end = min(j_start + chunk_size, n_items)
            cand_t = item_emb_t[j_start:j_end]  # (chunk, D)
            # he_t: (B, max_h, D) -> (B, max_h, 1, D), cand_t: (1, 1, chunk, D)
            he_exp = he_t.unsqueeze(2)  # (B, max_h, 1, D)
            cand_exp = cand_t.unsqueeze(0).unsqueeze(0)  # (1, 1, chunk, D)
            # dist_kappa 接受 (..., D) -> 标量; 重塑成 (B*max_h*chunk, D) 再 reshape
            he_flat = he_exp.expand(B, max_h, j_end - j_start, -1).reshape(-1, he_t.shape[-1])
            cand_flat = cand_exp.expand(B, max_h, j_end - j_start, -1).reshape(-1, cand_t.shape[-1])
            d_flat = stereo_mod.dist_kappa(he_flat, cand_flat, kappa)  # (B*max_h*chunk,)
            d_bh = d_flat.reshape(B, max_h, j_end - j_start).numpy()
            dist_bh_n[:, :, j_start:j_end] = d_bh
        # distance -> score: score = -distance, mask 用 -inf
        # sims (B, max_h, N), dist 越小越好
        sims_bh_n = -dist_bh_n  # (B, max_h, N)
        # mask invalid history positions
        sims_bh_n = np.where(m_b[:, :, None], sims_bh_n, np.float32(-1e10))
        # 4 aggregations
        # 1) min_cos (= max over history of similarity) → argmax over h
        # 2) max_cos 同上 (注意 dist_kappa 单调)
        # 3) mean_cos
        # 4) topk_mean (K=3)
        # 但我们这里 score = -dist, 所以:
        # - min_cos (history 内最强 signal) = max over h of score = max over h of -dist = -min over h of dist
        # - mean_cos = mean over h of -dist (但需 mask)
        # - topk_mean: top-K max score

        # 这里我们直接返回 per-item score mat, 后面再算 aggregations
        # 但为了节省内存, 我们直接在 batch 内聚合
        # 因为要存到 scores[s:e], 用 sims.max axis=1 (min dist) 作为聚合结果存
        # 然后 mean_cos / topk_mean 单独跑

        # min_cos / max_cos: same when only one history item, otherwise min_cos picks closest
        # 等价于 sims.max(axis=1)
        scores[s:e] = sims_bh_n.max(axis=1)  # 默认 min_cos 聚合

    return scores, valid, hist_mat, hist_mask, item_emb_t, n_items


def aggregate_scores(test_samples, item_emb_t, kappa, stereo_mod, agg_name, K_top=3):
    """对每个 test sample 计算 4 种 aggregation 的 per-sample score mat."""
    N = len(test_samples)
    n_items = item_emb_t.shape[0]
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

    agg_scores = np.zeros((N, n_items), dtype=np.float32)
    for s in range(0, N, SCORE_BATCH):
        e = min(s + SCORE_BATCH, N)
        B = e - s
        h_b = hist_mat[s:e]
        m_b = hist_mask[s:e]

        flat = h_b.flatten()
        he_t = item_emb_t[flat].reshape(B, max_h, -1)  # (B, max_h, D)

        n_chunks = max(1, n_items // 256)
        chunk_size = (n_items + n_chunks - 1) // n_chunks
        sims_bh_n = np.full((B, max_h, n_items), np.float32(-1e10), dtype=np.float32)
        for j_start in range(0, n_items, chunk_size):
            j_end = min(j_start + chunk_size, n_items)
            cand_t = item_emb_t[j_start:j_end]
            he_exp = he_t.unsqueeze(2)
            cand_exp = cand_t.unsqueeze(0).unsqueeze(0)
            he_flat = he_exp.expand(B, max_h, j_end - j_start, -1).reshape(-1, he_t.shape[-1])
            cand_flat = cand_exp.expand(B, max_h, j_end - j_start, -1).reshape(-1, cand_t.shape[-1])
            d_flat = stereo_mod.dist_kappa(he_flat, cand_flat, kappa)
            d_bh = d_flat.reshape(B, max_h, j_end - j_start).numpy()
            sims_chunk = -d_bh  # score = -dist
            sims_bh_n[:, :, j_start:j_end] = np.where(m_b[:, :, None], sims_chunk, np.float32(-1e10))

        # aggregations
        if agg_name == 'min_cos':  # history 内最强匹配 (min dist, max score)
            agg = sims_bh_n.max(axis=1)
        elif agg_name == 'mean_cos':
            # mean over h (with mask) — but mask value is -1e10, so divide by valid count
            m_3d = m_b[:, :, None]
            # Replace -1e10 with 0 for sum
            sims_for_sum = np.where(m_3d, sims_bh_n, 0.0)
            sums = sims_for_sum.sum(axis=1)
            counts = m_b.sum(axis=1, keepdims=True).clip(min=1)
            agg = sums / counts
        elif agg_name == 'topk_mean':
            K = min(K_top, max_h)
            part = np.partition(sims_bh_n, -K, axis=1)[:, -K:, :]
            agg = part.mean(axis=1)
        elif agg_name == 'last_item_cos':
            # 取最后一个 history item (history 是按时间排序的, 最后一个 = 最新)
            agg = sims_bh_n[:, -1, :]
        else:
            raise ValueError(f'Unknown agg: {agg_name}')
        agg_scores[s:e] = agg

    return agg_scores, valid


def per_user_rk(score_mat, target_arr, valid, k=5):
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


def smoking_gun_gap(test_samples, item_emb_t, kappa, stereo_mod, agg_name, n_random=N_RANDOM, seed=SEED):
    """Smoking gun gap: mean(score(query, target)) - mean(score(query, n_random random items))."""
    rng = np.random.RandomState(seed)
    N = len(test_samples)
    targets = np.asarray([s['target_idx'] for s in test_samples])

    # Step 1: 算 per-test mean cosine(target vs target) (smoking gun 直接量)
    # 但我们改用 score(query, target) - score(query, random) — query = history items (per-item protocol)
    # 实际: 对每个 test sample, target_emb = item_emb[target], 算 score(target_emb, item_emb) - 它的 random baseline

    # 更直接: 用 task36 的方式 — score(query=target, candidate) vs score(query=target, random)
    # 但 task36 的是 query=history items
    # 这里保持 task36 一致: query=history items, candidate=target vs random
    # 即对每个 test sample, 算 score_per_item(history_i, target) - score_per_item(history_i, random_j)
    # 然后 aggregate

    # 简化: 直接算 history items 对 target 的 score 与对 random items 的 score
    n_items = item_emb_t.shape[0]
    target_emb = item_emb_t[targets]  # (N, D)
    random_idx = rng.randint(0, n_items, size=(N, n_random))  # (N, n_random)
    random_emb = item_emb_t[random_idx.flatten()].reshape(N, n_random, -1)  # (N, n_random, D)

    # 对每个 test, history items 与 target/random 算 distance, 然后 score aggregation
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

    # score(query_i = history_i, candidate_j) per aggregation
    # min_cos / max_cos = max over h of -dist(hist_emb[h], candidate)
    # mean_cos = mean over h
    # topk_mean = mean of top K

    def score_for_candidates(hist_emb_bh, candidate_bk_d, mask_bh, agg_name, K_top=3):
        """hist_emb_bh: (B, max_h, D), candidate_bk_d: (B, K, D), mask_bh: (B, max_h)."""
        B, max_h, D = hist_emb_bh.shape
        K = candidate_bk_d.shape[1]
        he = hist_emb_bh.unsqueeze(2)  # (B, max_h, 1, D)
        cand = candidate_bk_d.unsqueeze(1)  # (B, 1, K, D)
        he_f = he.expand(B, max_h, K, D).reshape(-1, D)
        cand_f = cand.expand(B, max_h, K, D).reshape(-1, D)
        d = stereo_mod.dist_kappa(he_f, cand_f, kappa).reshape(B, max_h, K)
        sims = -d
        sims = np.where(mask_bh[:, :, None], sims.numpy(), np.float32(-1e10))
        if agg_name == 'min_cos':
            return sims.max(axis=1)  # (B, K)
        elif agg_name == 'mean_cos':
            m3 = mask_bh[:, :, None]
            sims_sum = np.where(m3, sims, 0.0).sum(axis=1)
            cnt = mask_bh.sum(axis=1, keepdims=True).clip(min=1)
            return sims_sum / cnt
        elif agg_name == 'topk_mean':
            Kt = min(K_top, max_h)
            part = np.partition(sims, -Kt, axis=1)[:, -Kt:, :]
            return part.mean(axis=1)
        elif agg_name == 'last_item_cos':
            return sims[:, -1, :]
        else:
            raise ValueError(agg_name)

    # 分批
    target_scores = np.zeros(N, dtype=np.float32)
    random_scores = np.zeros((N, n_random), dtype=np.float32)
    for s in range(0, N, SCORE_BATCH):
        e = min(s + SCORE_BATCH, N)
        B = e - s
        h_b = hist_mat[s:e]
        m_b = hist_mask[s:e]
        flat = h_b.flatten()
        he = item_emb_t[flat].reshape(B, max_h, -1)
        te = target_emb[s:e].unsqueeze(1)  # (B, 1, D)
        re = random_emb[s:e]  # (B, n_random, D)

        target_scores[s:e] = score_for_candidates(he, te, m_b, agg_name)[:, 0]
        random_scores[s:e] = score_for_candidates(he, re, m_b, agg_name)

    return float((target_scores[valid] - random_scores[valid].mean(axis=1)).mean())


def main():
    print('========== Task #42 Riemannian 重测 subspace_2_hyperbolic ==========')
    stereo = load_stereographic()
    print(f'Loaded stereographic.dist_kappa')

    test_samples = load_test()
    targets = np.asarray([s['target_idx'] for s in test_samples])
    print(f'test samples: {len(test_samples)}')

    mckg = torch.load(MCKG_EMB_PATH, weights_only=False)
    sub_item = mckg['subspace_item'].numpy()
    fused_item = mckg['fused_item'].numpy()
    kappas = mckg['kappas']  # [5.05, -0.08, -5.04]
    n_items = sub_item.shape[1]
    print(f'\nMCKG: {n_items} items, kappas={kappas}')

    # 验证 dist_kappa 的 sanity (κ=-5.04 hyperbolic)
    kappa_h = kappas[2]
    sub_h_t = torch.from_numpy(sub_item[2]).float()
    print(f'\nSanity check dist_kappa(κ={kappa_h:.2f}):')
    u = sub_h_t[0:1]
    v_same = sub_h_t[1:2]
    v_diff = sub_h_t[100:101]
    d_same = stereo.dist_kappa(u, v_same, kappa_h).item()
    d_diff = stereo.dist_kappa(u, v_diff, kappa_h).item()
    print(f'  dist(u[0], u[1]) = {d_same:.4f}')
    print(f'  dist(u[0], u[100]) = {d_diff:.4f}')
    print(f'  ratio = {d_diff/d_same:.2f}x (期望 >1)')

    results = {}
    aggs = ['min_cos', 'mean_cos', 'topk_mean', 'last_item_cos']

    # Task #36 subspace_2 hyperbolic 的 4 aggregations (Euclidean cosine baseline)
    task36_baseline = {
        'min_cos': 0.0160,
        'mean_cos': -0.0003,
        'topk_mean': 0.0067,
        'last_item_cos': -0.0002,
    }

    for agg in aggs:
        print(f'\n--- subspace_2_hyperbolic κ={kappa_h:.2f}, agg={agg} (Riemannian) ---')
        agg_scores, valid = aggregate_scores(test_samples, sub_h_t, kappa_h, stereo, agg)

        # R@5 / R@10
        hits5 = per_user_rk(agg_scores, targets, valid, k=5)
        hits10 = per_user_rk(agg_scores, targets, valid, k=10)
        r5 = float(hits5.mean())
        r10 = float(hits10.mean())

        # smoking gun gap (Riemannian 版)
        gap = smoking_gun_gap(test_samples, sub_h_t, kappa_h, stereo, agg)

        # Task #36 baseline (Euclidean cosine)
        base_gap = task36_baseline[agg]
        delta = gap - base_gap

        results[agg] = {
            'Riemannian_gap': gap,
            'Euclidean_gap_task36': base_gap,
            'delta': delta,
            'R@5': r5,
            'R@10': r10,
        }
        print(f'  R@5={r5:.4f}, R@10={r10:.4f}')
        print(f'  Riemannian gap = {gap:+.4f} (vs Euclidean {base_gap:+.4f}, Δ={delta:+.4f})')

    # 同时作为 sanity check, 用 sphere (κ=+5.05) 也跑一遍 min_cos, 验证 metric 在 κ>0 时仍工作
    print(f'\n--- sanity: subspace_0_sphere κ={kappas[0]:.2f}, agg=min_cos ---')
    sub_s_t = torch.from_numpy(sub_item[0]).float()
    agg_scores_s, valid_s = aggregate_scores(test_samples, sub_s_t, kappas[0], stereo, 'min_cos')
    hits5_s = per_user_rk(agg_scores_s, targets, valid_s, k=5)
    gap_s = smoking_gun_gap(test_samples, sub_s_t, kappas[0], stereo, 'min_cos')
    print(f'  sphere min_cos: R@5={hits5_s.mean():.4f}, Riemannian gap={gap_s:+.4f} (Euclidean task36=+0.0237)')

    results['sanity_sphere_min_cos'] = {
        'Riemannian_gap': gap_s,
        'Euclidean_gap_task36': 0.0237,
        'R@5': float(hits5_s.mean()),
    }

    out = OUT_DIR / 'task42_riemannian.json'
    summary = {
        'task': 'Task #42 — Riemannian 重测 subspace_2_hyperbolic',
        'protocol': 'per-item Riemannian distance (MCKG stereographic.dist_kappa), 4 aggregations, MAX_HISTORY=30',
        'metric': 'd_κ(u, v) = 2 · tan⁻¹_κ(||(-u) ⊕κ v||)',
        'kappa_subspace_2': float(kappa_h),
        'results': results,
        'interpretation': {
            'high_distinct_sphere': 'sanity: sphere Riemannian gap ≈ Euclidean (expected, sphere also wrong metric but consistent)',
            'subspace_2_delta': 'Riemannian - Euclidean: positive → cosine underestimated, negative → cosine overestimated',
        },
    }
    with open(out, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f'\n输出: {out}')

    # 总结
    print('\n========== 总结 ==========')
    print(f'subspace_2_hyperbolic (κ={kappa_h:.2f}) Riemannian vs Euclidean:')
    for agg in aggs:
        r = results[agg]
        verdict = ''
        if r['delta'] >= 0.005:
            verdict = '🟢 强提升 — 双曲 metric 显著高于 cosine,说明 cosine 测错'
        elif r['delta'] >= -0.005:
            verdict = '🟡 metric 不是核心问题 — subspace_2 本身就是 WEAK'
        else:
            verdict = '🔴 Riemannian 比 cosine 更弱 — subspace_2 在 metric 上没有额外信息'
        print(f'  {agg:15s}: Δ={r["delta"]:+.4f}  {verdict}')


if __name__ == '__main__':
    main()
