#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
task55_opq_train.py — ITQ (Iterative Quantization) — OPQ 闭式解版
====================================================================

任务: Task #55 OPQ 变体 (在 RQ 前对 embedding 做正交旋转, 减小 RQ 失真)
承接: Task #53 R@5=0.002 负结果, 根因 layer-0 RQ cov0=0.11 码本坍缩

算法: ITQ (Iterative Quantization, Gong & Lazebnik 2011)
  - 比端到端 OPQ 更稳定 (闭式 SVD 求解, 无 SGD 不稳定)
  - 等价于 OPQ 的交替优化收敛版本
  - 输入: (N, D) embedding, 目标 bit 数 b (D=b)
  - 输出: (N, D) rotated embedding, R ∈ R^{D × D} 正交

  步骤:
    1. PCA whitening: x_whitened = S^{-1/2} U^T (x - mean)
    2. 初始化 R = random orthogonal (D, D)
    3. 交替 (50 次):
       a. 固定 R, 用 k-means (256 桶) 量化 R @ x_whitened → codes
       b. 固定 codes, 求解 R = argmin ||RC - X||^2 s.t. R^T R = I
          R = (V U^T) where X C^T = U S V^T (SVD)
    4. 输出: rotated = x @ R (un-whiten if needed)

对比基线: 直接 RQ (无 OPQ), RQ recon = MSE

验证:
  RQ recon baseline (无 OPQ): 直接 RQ-VAE 训练 500 steps 取 MSE
  RQ recon OPQ:               用 ITQ rotate 后 RQ-VAE 训练 500 steps 取 MSE
  如果 OPQ_recon < 0.9 × RQ_recon_baseline → ✅ OPQ 显著有效

执行:
  source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh
  conda activate /home/wlia0047/ar57_scratch/wenyu/grid_toys
  cd /home/wlia0047/ar57/wenyu/GeneRec
  python3 scripts/task55_opq_train.py \
      --input products/task53_log1p_s4_ae/entity_embedding.pt \
      --output products/task55_opq_s4_ae/entity_embedding.pt \
      --codebook-size 256 --iters 50 --rq-steps 1500

注: RQ 阶段复用 snap-research/GRID MiniBatchKMeans 但简化训练 (无 encoder/decoder),
     直接对 ITQ-rotated embedding 做量化.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path('/home/wlia0047/ar57/wenyu/GeneRec')

# 把 GRID src 加进 path 以复用 RQ 组件
sys.path.insert(0, str(ROOT))


def itq_transform(x: torch.Tensor, n_iter: int = 50, seed: int = 42):
    """ITQ (Iterative Quantization) - 闭式求正交旋转矩阵 R

    Args:
        x: (N, D) 输入 embedding (CPU tensor, float32)
        n_iter: 交替优化次数
        seed: 随机种子

    Returns:
        R: (D, D) 正交旋转矩阵
        x_rot: (N, D) 旋转后的 embedding
    """
    rng = np.random.default_rng(seed)
    N, D = x.shape

    x_np = x.cpu().numpy().astype(np.float32)

    # Step 1: PCA + whitening
    mean = x_np.mean(axis=0, keepdims=True)
    x_centered = x_np - mean
    U, S, Vt = np.linalg.svd(x_centered, full_matrices=False)
    # 保留 D 维 (白化投影到原始空间)
    V = Vt.T  # (D, D)
    S_inv_sqrt = np.diag(1.0 / np.sqrt(S + 1e-9))  # (D, D)
    x_whitened = x_centered @ V @ S_inv_sqrt  # (N, D)
    # ❌ 注意: 不需要真正"白化", OPQ 目的是旋转不是缩放
    # 改正: x_whitened = x_centered @ V (无 S_inv_sqrt)
    x_pca = x_centered @ V  # 投影到主成分空间但保留 scale

    # Step 2: 初始化 R 为随机正交 (PCA 域 → ITQ 域)
    R = rng.standard_normal((D, D)).astype(np.float32)
    Q, _ = np.linalg.qr(R)  # QR 正交化
    R = Q

    # Step 3: 交替优化 codes ↔ R
    for it in range(n_iter):
        # a. 固定 R, 量化 R^T @ x_pca  (注: Gong 论文中 x 是 whitened, R 是变元)
        x_rot = x_pca @ R  # (N, D)
        # 用 k-means 简单量化
        codes = _simple_kmeans(x_rot, K=256, n_iter=20, seed=seed)
        codebook = _compute_centroids(x_rot, codes, K=256)  # (K, D)

        # b. 固定 codes, 求解 R = argmin_R ||X_pca R - codebook[codes]||^2
        #    等价于 R = argmin ||B @ R^T - X||^2 s.t. R^T R = I
        #    其中 B[codes] (N, D) = codebook 选择向量
        B = codebook[codes]  # (N, D)
        # SVD: B^T X_pca = U S V^T  →  R = V U^T
        # 我们要 R 使 X_pca @ R ≈ B
        # 即 R = argmin_R ||X_pca R - B||^2 s.t. R^T R = I
        # 解法: X_pca^T @ B = U S V^T → R = V U^T
        M = x_pca.T @ B  # (D, D)
        U_svd, _, Vt_svd = np.linalg.svd(M)
        R_new = U_svd @ Vt_svd  # (D, D) 正交

        # 检查 R 收敛
        diff = np.linalg.norm(R_new - R)
        R = R_new
        if it > 0 and diff < 1e-6:
            print(f'[ITQ] 第 {it} 轮收敛 (R 变化 {diff:.2e})')
            break

    # 旋转后的 embedding (在原 embedding 域)
    x_final = x_np @ (V @ R)  # = x_pca @ R
    R_total = V @ R  # (D, D) 从原 embedding 域到 ITQ 域的变换

    return torch.from_numpy(R_total), torch.from_numpy(x_final.astype(np.float32))


def _simple_kmeans(x: np.ndarray, K: int = 256, n_iter: int = 20, seed: int = 42):
    """简单 k-means 量化 (返回 codes)"""
    rng = np.random.default_rng(seed)
    N, D = x.shape
    # K-means++ init
    init_idx = rng.choice(N, size=K, replace=False)
    centroids = x[init_idx].copy()  # (K, D)
    codes = np.zeros(N, dtype=np.int32)

    for it in range(n_iter):
        # assign
        # dist (N, K)
        dist = np.sum(x ** 2, axis=1, keepdims=True) - 2 * x @ centroids.T
        codes = np.argmin(dist, axis=1)
        # update
        new_centroids = np.zeros_like(centroids)
        for k in range(K):
            mask = (codes == k)
            if mask.any():
                new_centroids[k] = x[mask].mean(axis=0)
            else:
                new_centroids[k] = x[rng.choice(N)]
        diff = np.linalg.norm(new_centroids - centroids)
        centroids = new_centroids
        if diff < 1e-5:
            break

    return codes


def _compute_centroids(x: np.ndarray, codes: np.ndarray, K: int = 256):
    """计算 codes 对应的 centroids"""
    centroids = np.zeros((K, x.shape[1]), dtype=np.float32)
    for k in range(K):
        mask = (codes == k)
        if mask.any():
            centroids[k] = x[mask].mean(axis=0)
        else:
            centroids[k] = np.zeros(x.shape[1], dtype=np.float32)
    return centroids


def rq_baseline(emb: torch.Tensor, codebook_size: int = 256, n_steps: int = 1500,
                seed: int = 42) -> float:
    """简单 RQ baseline: 第一层 K-means + 第二层 K-means (残差)

    Args:
        emb: (N, D) 输入 embedding
        codebook_size: K
        n_steps: k-means 迭代次数
        seed: 种子

    Returns:
        reconstruction MSE
    """
    x = emb.cpu().numpy().astype(np.float32)
    N, D = x.shape

    # Layer 0
    codes_0 = _simple_kmeans(x, K=codebook_size, n_iter=n_steps, seed=seed)
    centroid_0 = _compute_centroids(x, codes_0, K=codebook_size)
    residual_1 = x - centroid_0[codes_0]
    mse_0 = float(np.mean(residual_1 ** 2))

    # Layer 1
    codes_1 = _simple_kmeans(residual_1, K=codebook_size, n_iter=n_steps, seed=seed + 1)
    centroid_1 = _compute_centroids(residual_1, codes_1, K=codebook_size)
    residual_2 = residual_1 - centroid_1[codes_1]
    mse_1 = float(np.mean(residual_2 ** 2))

    # Layer 2
    codes_2 = _simple_kmeans(residual_2, K=codebook_size, n_iter=n_steps // 2, seed=seed + 2)
    centroid_2 = _compute_centroids(residual_2, codes_2, K=codebook_size)
    residual_3 = residual_2 - centroid_2[codes_2]
    mse_2 = float(np.mean(residual_3 ** 2))

    total_mse = mse_0 + mse_1 + mse_2
    cov_0 = _coverage(codes_0, codebook_size)
    cov_1 = _coverage(codes_1, codebook_size)
    cov_2 = _coverage(codes_2, codebook_size)

    return {
        'mse_layer_0': mse_0,
        'mse_layer_1': mse_1,
        'mse_layer_2': mse_2,
        'mse_total': total_mse,
        'cov_layer_0': cov_0,
        'cov_layer_1': cov_1,
        'cov_layer_2': cov_2,
    }


def _coverage(codes: np.ndarray, K: int) -> float:
    return float(len(np.unique(codes))) / float(K)


def main():
    parser = argparse.ArgumentParser(description='Task #55: OPQ via ITQ before RQ-VAE')
    parser.add_argument('--input', required=True, help='input embedding .pt')
    parser.add_argument('--output', required=True, help='output rotated embedding .pt')
    parser.add_argument('--codebook-size', type=int, default=256, help='RQ codebook size K')
    parser.add_argument('--iters', type=int, default=50, help='ITQ iterations')
    parser.add_argument('--rq-steps', type=int, default=1500, help='RQ k-means iterations')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--summary', default=None, help='optional JSON summary path')
    args = parser.parse_args()

    input_path = ROOT / args.input
    output_path = ROOT / args.output

    if not input_path.exists():
        raise FileNotFoundError(f'输入 embedding 不存在: {input_path}')

    print(f'[task55_opq_train] 输入: {input_path}')
    print(f'[task55_opq_train] 输出: {output_path}')

    # 加载 embedding (同 task53 逻辑)
    emb = torch.load(input_path, weights_only=False, map_location='cpu')
    if isinstance(emb, dict):
        if 'fused_item' in emb:
            emb = emb['fused_item']
        else:
            raise KeyError(f'输入 dict 缺 fused_item: keys={list(emb.keys())}')
    if not torch.is_tensor(emb):
        raise TypeError(f'期望 torch.Tensor, 实际 {type(emb)}')
    emb = emb.float()
    print(f'[task55_opq_train] 输入 shape={tuple(emb.shape)}, dtype={emb.dtype}')

    # Step 1: RQ baseline (无 OPQ)
    print('[task55_opq_train] Step 1: RQ baseline (无 OPQ) ...')
    t0 = time.time()
    rq_baseline_metrics = rq_baseline(emb, codebook_size=args.codebook_size,
                                      n_steps=args.rq_steps, seed=args.seed)
    t_baseline = time.time() - t0
    print(f'[task55_opq_train] RQ baseline: MSE={rq_baseline_metrics["mse_total"]:.4f} '
          f'cov0={rq_baseline_metrics["cov_layer_0"]:.3f} '
          f'cov1={rq_baseline_metrics["cov_layer_1"]:.3f} '
          f'cov2={rq_baseline_metrics["cov_layer_2"]:.3f} '
          f'({t_baseline:.1f}s)')

    # Step 2: ITQ → RQ
    print(f'[task55_opq_train] Step 2: ITQ ({args.iters} iters) ...')
    t0 = time.time()
    R_total, x_rot = itq_transform(emb, n_iter=args.iters, seed=args.seed)
    t_itq = time.time() - t0
    print(f'[task55_opq_train] ITQ done ({t_itq:.1f}s)')
    print(f'[task55_opq_train] x_rot shape={tuple(x_rot.shape)}')

    print('[task55_opq_train] Step 3: RQ on OPQ-rotated embedding ...')
    t0 = time.time()
    rq_opq_metrics = rq_baseline(x_rot, codebook_size=args.codebook_size,
                                 n_steps=args.rq_steps, seed=args.seed)
    t_opq = time.time() - t0
    print(f'[task55_opq_train] RQ on OPQ: MSE={rq_opq_metrics["mse_total"]:.4f} '
          f'cov0={rq_opq_metrics["cov_layer_0"]:.3f} '
          f'cov1={rq_opq_metrics["cov_layer_1"]:.3f} '
          f'cov2={rq_opq_metrics["cov_layer_2"]:.3f} '
          f'({t_opq:.1f}s)')

    # 决策
    if rq_baseline_metrics['mse_total'] > 0:
        ratio = rq_opq_metrics['mse_total'] / rq_baseline_metrics['mse_total']
    else:
        ratio = 1.0
    coverage_improvement = (rq_opq_metrics['cov_layer_0'] -
                            rq_baseline_metrics['cov_layer_0'])

    print('')
    print('=' * 60)
    print('Task #55 决策汇总')
    print('=' * 60)
    print(f'RQ baseline MSE (no OPQ): {rq_baseline_metrics["mse_total"]:.4f}')
    print(f'RQ + OPQ  MSE (ITQ 旋转): {rq_opq_metrics["mse_total"]:.4f}')
    print(f'MSE ratio:               {ratio:.3f} (>1 = 退化, <1 = 改进)')
    print(f'layer-0 cov 改进:        {rq_baseline_metrics["cov_layer_0"]:.3f} → '
          f'{rq_opq_metrics["cov_layer_0"]:.3f} '
          f'(+{coverage_improvement:+.3f})')
    print('')

    if ratio < 0.9 and coverage_improvement > 0.05:
        decision = '✅ OPQ 显著有效 (MSE < 0.9x 且 cov0 +0.05), 建议加入 Stage 2 流水线'
        valid = True
        # 保存 rotated embedding
        output_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(x_rot, output_path)
        print(f'[task55_opq_train] ✅ 已保存 ITQ-rotated embedding: {output_path}')
    elif ratio < 1.0:
        decision = '⚠️ OPQ 边际改进, 可选'
        valid = False
        # 仍保存以便后续使用
        output_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(x_rot, output_path)
        print(f'[task55_opq_train] ⚠️ 边际, 也保存以备用: {output_path}')
    else:
        decision = '❌ OPQ 无效 (orientation 已对齐, 或 ITQ 退化), 跳过'
        valid = False
        print(f'[task55_opq_train] ❌ 不保存 (ITQ-rotated 没用)')

    # Summary
    summary = {
        'input': str(input_path),
        'output': str(output_path) if (ratio < 1.0) else None,
        'shape': list(emb.shape),
        'seed': args.seed,
        'itq_iters': args.iters,
        'rq_steps': args.rq_steps,
        'rq_baseline': rq_baseline_metrics,
        'rq_opq': rq_opq_metrics,
        'mse_ratio': ratio,
        'cov0_improvement': coverage_improvement,
        'decision': decision,
        'valid': valid,
        't_baseline': t_baseline,
        't_itq': t_itq,
        't_opq_rq': t_opq,
    }
    if args.summary:
        summary_path = ROOT / args.summary
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
        print(f'[task55_opq_train] ✅ summary → {summary_path}')

    print('')
    print(decision)


if __name__ == '__main__':
    main()
