#!/usr/bin/env python3
"""Task #42 — Riemannian 重测 subspace_2 (GPU 加速版).

先前 CPU 版本跑了 4 小时未完成。改用 GPU + 向量化,预期 30 秒。
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
N_RANDOM = 50
SEED = 42
DEVICE = 'cuda:0'  # 用 GPU


def load_stereographic():
    import importlib.util
    spec = importlib.util.spec_from_file_location("stereographic", str(MCKG_STEREO))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


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


def compute_per_item_score_gpu(hist_emb, cand_emb, mask, stereo_mod, kappa, agg_name, K_top=3):
    """GPU 向量化算 per-item 分数.
    hist_emb: (B, max_h, D) torch
    cand_emb: (B, N, D) torch — 候选 item embeddings
    mask: (B, max_h) bool
    返回: (B, N) per-user score mat
    """
    B, max_h, D = hist_emb.shape
    N = cand_emb.shape[1]
    # 距离: dist(hist_emb[b,h,:], cand_emb[b,j,:]) for all b,h,j
    # 用 broadcast: (B, max_h, 1, D) vs (B, 1, N, D) -> (B, max_h, N, D) 然后 dist
    # 内存太大 (B=64, max_h=30, N=11924, D=64) = 64*30*11924*64*4 = 5.8 GB
    # 分 chunk
    chunk = 1024  # 一次算 1024 个 candidate
    out = torch.zeros(B, N, device=hist_emb.device, dtype=torch.float32)
    he_flat = hist_emb  # (B, max_h, D)
    for j_start in range(0, N, chunk):
        j_end = min(j_start + chunk, N)
        # he_flat: (B, max_h, D) -> (B, max_h, 1, D)
        # cand: (B, j_end-j_start, D) -> (B, 1, j_end-j_start, D)
        he = he_flat.unsqueeze(2).expand(B, max_h, j_end - j_start, D)
        cand = cand_emb[:, j_start:j_end].unsqueeze(1).expand(B, max_h, j_end - j_start, D)
        he_2d = he.reshape(-1, D)
        cand_2d = cand.reshape(-1, D)
        d = stereo_mod.dist_kappa(he_2d, cand_2d, kappa).reshape(B, max_h, j_end - j_start)
        sims = -d  # score
        sims = torch.where(mask.unsqueeze(2), sims, torch.tensor(-1e10, device=sims.device))

        if agg_name == 'min_cos':
            out[:, j_start:j_end] = sims.max(dim=1).values
        elif agg_name == 'mean_cos':
            sims_for_sum = torch.where(mask.unsqueeze(2), sims, torch.tensor(0.0, device=sims.device))
            sums = sims_for_sum.sum(dim=1)
            cnt = mask.sum(dim=1, keepdim=True).clamp(min=1).float()
            out[:, j_start:j_end] = sums / cnt
        elif agg_name == 'topk_mean':
            Kt = min(K_top, max_h)
            topk_vals, _ = sims.topk(Kt, dim=1)
            out[:, j_start:j_end] = topk_vals.mean(dim=1)
        elif agg_name == 'last_item_cos':
            out[:, j_start:j_end] = sims[:, -1, :]
    return out


def per_user_rk(score_mat, target_arr, valid, k=5):
    top = torch.topk(score_mat, k, dim=1).indices.cpu().numpy()  # (B, k)
    hits = np.zeros(len(target_arr), dtype=bool)
    for i in range(len(target_arr)):
        if valid[i] and target_arr[i] in top[i]:
            hits[i] = True
    return hits


def main():
    print('========== Task #42 Riemannian 重测 subspace_2 (GPU) ==========')
    device = torch.device(DEVICE)
    print(f'Device: {device}')

    stereo = load_stereographic()

    test_samples = load_test()
    targets = np.asarray([s['target_idx'] for s in test_samples])
    print(f'test samples: {len(test_samples)}')

    mckg = torch.load(MCKG_EMB_PATH, weights_only=False)
    sub_item = mckg['subspace_item'].numpy()
    fused_item = mckg['fused_item'].numpy()
    kappas = mckg['kappas']  # [+5.05, -0.08, -5.04]
    n_items = sub_item.shape[1]

    # 构造 history 矩阵
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

    # 转 GPU tensor
    sub_h = torch.from_numpy(sub_item[2]).float().to(device)  # subspace_2 hyperbolic
    sub_s = torch.from_numpy(sub_item[0]).float().to(device)  # subspace_0 sphere (sanity)
    fused_t = torch.from_numpy(fused_item).float().to(device)
    hist_mask_t = torch.from_numpy(hist_mask).to(device)

    kappa_h = float(kappas[2])
    kappa_s = float(kappas[0])
    print(f'\nsubspace_2 κ={kappa_h:.2f}, subspace_0 κ={kappa_s:.2f}')

    # Sanity: dist_kappa 的输出范围
    u = sub_h[0:1]
    v_same = sub_h[1:2]
    v_diff = sub_h[100:101]
    print(f'Sanity: dist(u[0], u[1])={stereo.dist_kappa(u, v_same, kappa_h).item():.4f}, '
          f'dist(u[0], u[100])={stereo.dist_kappa(u, v_diff, kappa_h).item():.4f}')

    # 构造 hist_emb 张量 (N, max_h, D) GPU — 直接索引
    # 为节省内存, 一次性把所有 hist_emb 加载到 GPU
    flat_idx = hist_mat.flatten()  # (N*max_h,)
    hist_emb_h = sub_h[flat_idx].reshape(N, max_h, -1)  # (N, max_h, D) GPU
    hist_emb_s = sub_s[flat_idx].reshape(N, max_h, -1)
    hist_emb_f = fused_t[flat_idx].reshape(N, max_h, -1)

    # 候选: 所有 items (N_items, D) GPU
    cand_h = sub_h  # (n_items, D)
    cand_s = sub_s
    cand_f = fused_t

    # 把 (N, max_h, D) 和 (N_items, D) 改成 (N, 1, D) 和 (N_items, D) 的 batched 模式
    # 但 compute_per_item_score_gpu 假设 cand 是 (B, N, D), 所以要 expand
    # 实际上 cand 对每个 sample 都是一样的 (全物品候选)
    # 改成全局候选: cand_emb: (1, n_items, D), expand 到 (N, n_items, D) — 但内存大
    # 更高效: 让 cand 是 (n_items, D), 在函数内部 broadcast
    # 简化: 直接用 chunk 方式, cand 作为 (n_items, D) 传入, 不做 B 维度
    # 重新写一个无 B 的版本

    def compute_scores_global(hist_emb, cand_emb, mask, stereo_mod, kappa, agg_name, K_top=3):
        """hist_emb: (N, max_h, D), cand_emb: (n_items, D), mask: (N, max_h)."""
        N_, max_h_, D_ = hist_emb.shape
        N_items_ = cand_emb.shape[0]
        # 更小的 chunk + 分 N batch (避免 OOM)
        sample_batch = 1024  # 一次处理 1024 个 sample
        chunk = 256  # 一次算 256 个 candidate
        out = torch.zeros(N_, N_items_, device=hist_emb.device, dtype=torch.float32)

        # 把 hist_emb 按 N 维度分批处理
        for s_start in range(0, N_, sample_batch):
            s_end = min(s_start + sample_batch, N_)
            he_b = hist_emb[s_start:s_end]
            mask_b = mask[s_start:s_end]
            NB = s_end - s_start

            for j_start in range(0, N_items_, chunk):
                j_end = min(j_start + chunk, N_items_)
                # he: (NB, max_h_, 1, D_), cand: (NB, 1, j_end-j_start, D_)
                he = he_b.unsqueeze(2).expand(NB, max_h_, j_end - j_start, D_)
                cand = cand_emb[j_start:j_end].unsqueeze(0).unsqueeze(0).expand(NB, max_h_, j_end - j_start, D_)
                he_2d = he.reshape(-1, D_)
                cand_2d = cand.reshape(-1, D_)
                d = stereo_mod.dist_kappa(he_2d, cand_2d, kappa).reshape(NB, max_h_, j_end - j_start)
                sims = -d
                sims = torch.where(mask_b.unsqueeze(2), sims, torch.tensor(-1e10, device=sims.device))

                if agg_name == 'min_cos':
                    out[s_start:s_end, j_start:j_end] = sims.max(dim=1).values
                elif agg_name == 'mean_cos':
                    sims_for_sum = torch.where(mask_b.unsqueeze(2), sims, torch.tensor(0.0, device=sims.device))
                    sums = sims_for_sum.sum(dim=1)
                    cnt = mask_b.sum(dim=1, keepdim=True).clamp(min=1).float()
                    out[s_start:s_end, j_start:j_end] = sums / cnt
                elif agg_name == 'topk_mean':
                    Kt = min(K_top, max_h_)
                    topk_vals, _ = sims.topk(Kt, dim=1)
                    out[s_start:s_end, j_start:j_end] = topk_vals.mean(dim=1)
                elif agg_name == 'last_item_cos':
                    out[s_start:s_end, j_start:j_end] = sims[:, -1, :]
            del he_b, mask_b
            torch.cuda.empty_cache()
        return out

    # 把 hist_emb 放到 GPU
    hist_emb_h = hist_emb_h.to(device)
    hist_emb_s = hist_emb_s.to(device)
    hist_emb_f = hist_emb_f.to(device)
    hist_mask_t = hist_mask_t.to(device)

    aggs = ['min_cos', 'mean_cos', 'topk_mean', 'last_item_cos']
    task36_baseline = {
        'min_cos': 0.0160,
        'mean_cos': -0.0003,
        'topk_mean': 0.0067,
        'last_item_cos': -0.0002,
    }

    results = {}

    # ============ subspace_2 hyperbolic (Riemannian) ============
    print('\n--- subspace_2_hyperbolic (Riemannian) ---')
    for agg in aggs:
        scores = compute_scores_global(hist_emb_h, cand_h, hist_mask_t, stereo, kappa_h, agg)  # (N, n_items)
        # R@5 / R@10
        top5 = torch.topk(scores, 5, dim=1).indices.cpu().numpy()
        top10 = torch.topk(scores, 10, dim=1).indices.cpu().numpy()
        hits5 = np.array([valid[i] and targets[i] in top5[i] for i in range(N)])
        hits10 = np.array([valid[i] and targets[i] in top10[i] for i in range(N)])
        r5 = float(hits5.mean())
        r10 = float(hits10.mean())

        # smoking gun gap: score(query=target, target) - mean(score(query=target, n_random))
        # 简化: 直接对每个 valid sample, 算 (score_at_target - mean(score_at_random))
        scores_at_target = scores[torch.arange(N, device=device), torch.from_numpy(targets).to(device)]  # (N,)
        rng = np.random.RandomState(SEED)
        rand_idx = torch.from_numpy(rng.randint(0, n_items, size=(N, N_RANDOM))).to(device)
        scores_at_random = scores.gather(1, rand_idx)  # (N, N_RANDOM)
        gap = float(((scores_at_target[valid] - scores_at_random[valid].mean(dim=1))).mean())

        base = task36_baseline[agg]
        delta = gap - base
        results[f'subspace_2_{agg}'] = {
            'space': 'subspace_2_hyperbolic',
            'agg': agg,
            'metric': 'Riemannian',
            'gap': gap,
            'euclidean_gap': base,
            'delta': delta,
            'R@5': r5,
            'R@10': r10,
        }
        print(f'  {agg:15s}: R@5={r5:.4f}, R@10={r10:.4f}, gap={gap:+.4f} (vs Euclidean {base:+.4f}, Δ={delta:+.4f})')

    # ============ sanity: subspace_0 sphere ============
    print('\n--- sanity: subspace_0_sphere (Riemannian) ---')
    for agg in ['min_cos']:
        scores = compute_scores_global(hist_emb_s, cand_s, hist_mask_t, stereo, kappa_s, agg)
        top5 = torch.topk(scores, 5, dim=1).indices.cpu().numpy()
        hits5 = np.array([valid[i] and targets[i] in top5[i] for i in range(N)])
        scores_at_target = scores[torch.arange(N, device=device), torch.from_numpy(targets).to(device)]
        rng = np.random.RandomState(SEED)
        rand_idx = torch.from_numpy(rng.randint(0, n_items, size=(N, N_RANDOM))).to(device)
        scores_at_random = scores.gather(1, rand_idx)
        gap = float(((scores_at_target[valid] - scores_at_random[valid].mean(dim=1))).mean())
        results[f'subspace_0_{agg}_sanity'] = {
            'space': 'subspace_0_sphere',
            'agg': agg,
            'metric': 'Riemannian',
            'gap': gap,
            'euclidean_gap': 0.0237,
            'delta': gap - 0.0237,
            'R@5': float(hits5.mean()),
        }
        print(f'  {agg:15s}: R@5={hits5.mean():.4f}, gap={gap:+.4f} (vs Euclidean 0.0237, Δ={gap-0.0237:+.4f})')

    # ============ sanity: fused_64d (euclidean) — 用 dist_kappa(κ=0) = L2 ============
    print('\n--- sanity: fused_64d (Riemannian κ=0 ≈ Euclidean) ---')
    for agg in ['min_cos']:
        scores = compute_scores_global(hist_emb_f, cand_f, hist_mask_t, stereo, 0.0, agg)
        top5 = torch.topk(scores, 5, dim=1).indices.cpu().numpy()
        hits5 = np.array([valid[i] and targets[i] in top5[i] for i in range(N)])
        scores_at_target = scores[torch.arange(N, device=device), torch.from_numpy(targets).to(device)]
        rng = np.random.RandomState(SEED)
        rand_idx = torch.from_numpy(rng.randint(0, n_items, size=(N, N_RANDOM))).to(device)
        scores_at_random = scores.gather(1, rand_idx)
        gap = float(((scores_at_target[valid] - scores_at_random[valid].mean(dim=1))).mean())
        results[f'fused_{agg}_sanity'] = {
            'space': 'fused_64d',
            'agg': agg,
            'metric': 'Riemannian (κ=0)',
            'gap': gap,
            'euclidean_gap': 0.0298,
            'delta': gap - 0.0298,
            'R@5': float(hits5.mean()),
        }
        print(f'  {agg:15s}: R@5={hits5.mean():.4f}, gap={gap:+.4f} (vs Euclidean 0.0298, Δ={gap-0.0298:+.4f})')

    # 输出
    out = OUT_DIR / 'task42_riemannian.json'
    summary = {
        'task': 'Task #42 — Riemannian 重测 subspace_2_hyperbolic (GPU)',
        'protocol': 'per-item Riemannian distance (MCKG stereographic.dist_kappa), 4 aggregations, MAX_HISTORY=30, GPU',
        'metric': 'd_κ(u, v) = 2 · tan⁻¹_κ(||(-u) ⊕κ v||)',
        'kappa_subspace_2': float(kappa_h),
        'results': results,
    }
    with open(out, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f'\n输出: {out}')

    # 总结
    print('\n========== 总结 ==========')
    for k, r in results.items():
        delta = r['delta']
        if abs(delta) < 0.005:
            verdict = '🟡 metric 不是核心问题'
        elif delta > 0:
            verdict = '🟢 Riemannian > Euclidean (cosine 测错)'
        else:
            verdict = '🔴 Riemannian < Euclidean'
        print(f'  {k:30s} Δ={delta:+.4f}  {verdict}')


if __name__ == '__main__':
    main()
