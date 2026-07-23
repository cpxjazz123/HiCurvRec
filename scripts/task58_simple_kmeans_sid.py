#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
task58_simple_kmeans_sid.py — Simple KMeans 3-layer SID 推断 (Task #58)

任务: Task #58 — Simple KMeans SID + TIGER 端到端验证 (绕过 Task #53 RQ-VAE 架构缺陷)
承接: Task #53-#56 verdict — Simple KMeans 在 log1p S4 AE 64d 上 cov0=1.000 vs
      Neural RQ-VAE cov0=0.109. 跳过 RQ-VAE 的 encoder/decoder 神经压缩,
      直接用 sklearn MiniBatchKMeans 做 3-layer residual quantization.

本脚本只做 Stage 2 推断 (Stage 2.1 是"训练 codebook", 2.2 是"推断 code"), 我们把两步合为一步:
  - 用 k-means++ 初始化 + 多轮迭代 fit (相当于 Stage 2.1 训练)
  - 用 predict 做 (Stage 2.2 推断)

最终输出 (与 rqvae_inference_flat 兼容格式):
  cluster_ids.pt: shape (num_hierarchies, N) int64 — 3 layer code + 1 dedup digit
  merged_predictions_tensor.pt: shape (4, N) int64 (与 Task #53 一致)

执行:
  source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh
  conda activate /home/wlia0047/ar57_scratch/wenyu/grid_toys

  python3 scripts/task58_simple_kmeans_sid.py \
      --input products/task53_log1p_s4_ae/entity_embedding.pt \
      --output-dir logs/task58_s2_infer \
      --k 256 --layers 3 --dedup-digit 1
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path('/home/wlia0047/ar57/wenyu/GeneRec')
sys.path.insert(0, str(ROOT))


def _kmeans_fit(x: np.ndarray, K: int, n_iters: int, seed: int):
    """MiniBatchKMeans 简化版 (sklearn-like API) — 适配纯 numpy GPU/CPU 切换

    Returns: (centroids (K, D), codes (N,))
    """
    rng = np.random.default_rng(seed)
    N, D = x.shape

    # k-means++ 初始化 (确定性)
    init_idx = [rng.integers(N)]
    for _ in range(1, K):
        dists = np.min(np.sum((x - x[init_idx[-1]]) ** 2, axis=1))
        closest_sq_dist = np.zeros(N)
        for j, ix in enumerate(init_idx):
            d = np.sum((x - x[ix]) ** 2, axis=1)
            if j == 0:
                closest_sq_dist = d
            else:
                closest_sq_dist = np.minimum(closest_sq_dist, d)
        probs = closest_sq_dist / (closest_sq_dist.sum() + 1e-9)
        next_idx = rng.choice(N, p=probs)
        init_idx.append(int(next_idx))
    centroids = x[init_idx].copy()  # (K, D)

    # Lloyd's iterations
    for it in range(n_iters):
        # assign
        # 用 ||x||^2 - 2 x·c + ||c||^2 算距离 (N, K)
        dist_sq = (x ** 2).sum(axis=1, keepdims=True) - 2 * x @ centroids.T
        dist_sq += (centroids ** 2).sum(axis=1, keepdims=True).T
        codes = np.argmin(dist_sq, axis=1)
        # update
        new_centroids = np.zeros_like(centroids)
        counts = np.zeros(K, dtype=np.int64)
        for k in range(K):
            mask = (codes == k)
            if mask.any():
                new_centroids[k] = x[mask].mean(axis=0)
                counts[k] = mask.sum()
            else:
                new_centroids[k] = x[rng.choice(N)]
                counts[k] = 1
        shift = np.linalg.norm(new_centroids - centroids)
        centroids = new_centroids
        if it > 0 and shift < 1e-5:
            break

    return centroids, codes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True, help='input embedding .pt')
    parser.add_argument('--output-dir', required=True, help='output dir (saved cluster_ids.pt)')
    parser.add_argument('--k', type=int, default=256, help='codebook size per layer')
    parser.add_argument('--layers', type=int, default=3, help='number of RQ layers')
    parser.add_argument('--dedup-digit', type=int, default=1,
                        help='appended dedup digit (1 = same as baseline)')
    parser.add_argument('--iters', type=int, default=300, help='k-means iterations per layer')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--summary', default=None, help='optional JSON summary path')
    args = parser.parse_args()

    input_path = ROOT / args.input
    output_dir = ROOT / args.output_dir

    if not input_path.exists():
        raise FileNotFoundError(f'输入 embedding 不存在: {input_path}')

    output_dir.mkdir(parents=True, exist_ok=True)

    # 加载 (同 task53 逻辑)
    emb = torch.load(input_path, weights_only=False, map_location='cpu')
    if isinstance(emb, dict):
        if 'fused_item' in emb:
            emb = emb['fused_item']
        else:
            raise KeyError(f'输入 dict 缺 fused_item: keys={list(emb.keys())}')
    emb = emb.float()
    print(f'[task58] 输入: {input_path}, shape={tuple(emb.shape)}')

    x = emb.numpy().astype(np.float32)
    N, D = x.shape

    # 3-layer RQ (sequential residual)
    all_codes = []
    all_centroids = []
    all_recon_mse = []

    residual = x.copy()
    print(f'[task58] 启动 {args.layers}-layer RQ, K={args.k}, iters={args.iters}')

    for layer in range(args.layers):
        t0 = time.time()
        centroids, codes = _kmeans_fit(residual, K=args.k, n_iters=args.iters, seed=args.seed + layer)
        # residual update
        new_residual = residual - centroids[codes]
        mse = float(np.mean(new_residual ** 2))
        cov = float(len(np.unique(codes))) / float(args.k)
        t_l = time.time() - t0
        print(f'[task58]   layer {layer}: cov={cov:.3f} mse={mse:.5f} ({t_l:.1f}s)')
        all_codes.append(codes)
        all_centroids.append(centroids)
        all_recon_mse.append(mse)
        residual = new_residual

    # 拼成 (layers, N) int64
    codes_stack = np.stack(all_codes, axis=0).astype(np.int64)  # (3, N)

    # 追加 dedup digit (与 Task #87 baseline 一致, 给 1 = unique digit)
    if args.dedup_digit > 0:
        dedup = np.full((1, N), args.dedup_digit, dtype=np.int64)
        cluster_ids = np.concatenate([codes_stack, dedup], axis=0)  # (4, N)
    else:
        cluster_ids = codes_stack

    # 保存 (与 rqvae_inference_flat 输出一致格式: pickle/cluster_ids.pt)
    pickle_dir = output_dir / 'pickle'
    pickle_dir.mkdir(parents=True, exist_ok=True)

    cluster_ids_t = torch.from_numpy(cluster_ids)
    cluster_ids_pt = pickle_dir / 'cluster_ids.pt'
    torch.save(cluster_ids_t, cluster_ids_pt)
    print(f'[task58] ✅ cluster_ids.pt → {cluster_ids_pt}, shape={tuple(cluster_ids_t.shape)}')

    # merged_predictions_tensor.pt (Task #53 推断用这个)
    merged_pt = pickle_dir / 'merged_predictions_tensor.pt'
    torch.save(cluster_ids_t, merged_pt)
    print(f'[task58] ✅ merged_predictions_tensor.pt → {merged_pt}')

    # 综合指标
    total_recon = sum(all_recon_mse)
    print('')
    print('=' * 60)
    print('Task #58 Stage 2 (simple KMeans SID 推断) 完成')
    print('=' * 60)
    print(f'   input shape: {tuple(emb.shape)}')
    print(f'   K = {args.k}, layers = {args.layers}')
    for layer in range(args.layers):
        print(f'   layer {layer}: cov={float(len(np.unique(all_codes[layer])))/float(args.k):.3f}')
    print(f'   total residual MSE = {total_recon:.5f}')
    print(f'   cluster_ids shape = {tuple(cluster_ids_t.shape)}')
    print('')

    summary = {
        'input': str(input_path),
        'output_dir': str(output_dir),
        'shape': list(emb.shape),
        'k': args.k,
        'layers': args.layers,
        'dedup_digit': args.dedup_digit,
        'cov_per_layer': [float(len(np.unique(c))) / float(args.k) for c in all_codes],
        'mse_per_layer': all_recon_mse,
        'mse_total': total_recon,
        'cluster_ids_path': str(cluster_ids_pt),
        'merged_predictions_path': str(merged_pt),
        'cluster_ids_shape': list(cluster_ids_t.shape),
        'seed': args.seed,
    }
    if args.summary:
        summary_path = ROOT / args.summary
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
        print(f'[task58] summary → {summary_path}')


if __name__ == '__main__':
    main()
