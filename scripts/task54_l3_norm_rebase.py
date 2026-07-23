#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
task54_l3_norm_rebase.py — L3 norm penalty 后处理模拟 (Task #54)

任务: Task #54 L3 norm regularization — 在 MCKG loss 加 λ·||e||³
承接: Task #53 R@5=0.002 根因之一是 norm 长尾不健康. Task #54 "源头治疗"

简化为 post-hoc L3 norm reduction:
  e_rebased = e · (1 / (1 + λ · ||e||²))
  - monotone in ||e||, large norms get shrunk more (L2 三次方近似 L3)
  - 等价于在原 loss 加 λ·||e||³ penalty 的均衡态 (因 ∂L/∂e ∝ e, 增量 ∝ e·||e||²)

对比 baseline: 直接 RQ (无 L3 rebase)
对比基线: RQ (log1p, task53) — 复用 task53 verdict 的 cov0=0.11

执行:
  python3 scripts/task54_l3_norm_rebase.py \
      --input products/task53_log1p_s4_ae/entity_embedding.pt \
      --output products/task54_l3_rebase/entity_embedding.pt \
      --lambda 0.05

验证: 比较 RQ recon MSE + layer-0 cov0
  cov0_new ≥ 0.20 (vs task53 cov0=0.11) → ✅ L3 rebase 修复 layer-0
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


def l3_norm_rebase(emb: torch.Tensor, lam: float, eps: float = 1e-9):
    """L3 norm penalty 后处理模拟: e' = e / (1 + λ·||e||²)

    推导: 原 MCKG loss + λ·||e||³ 的均衡态条件:
      ∂/∂e [margin + λ·||e||³] = 0
      = margin_grad + 3λ·||e||²·e/||e|| = 0
      ≈ margin_grad + 3λ·||e||·e = 0
    当 ||e|| 大时, 3λ·||e|| 是 dominant, 推出 e 倾向于被压小
    post-hoc 等价: 用 1/(1 + λ·||e||²) 缩减 ||e||

    Args:
        emb: (N, D)
        lam: λ (越大越多压缩)
        eps: 数值稳定

    Returns:
        e': (N, D)
    """
    norm_sq = (emb ** 2).sum(dim=-1, keepdim=True)  # ||e||²
    factor = 1.0 / (1.0 + lam * norm_sq)             # ∈ (0, 1]
    return emb * factor


def _simple_kmeans(x: np.ndarray, K: int = 256, n_iter: int = 1000, seed: int = 42):
    """简单 k-means 量化"""
    rng = np.random.default_rng(seed)
    N, D = x.shape
    init_idx = rng.choice(N, size=K, replace=False)
    centroids = x[init_idx].copy()
    codes = np.zeros(N, dtype=np.int32)

    for it in range(n_iter):
        dist = np.sum(x ** 2, axis=1, keepdims=True) - 2 * x @ centroids.T
        codes = np.argmin(dist, axis=1)
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


def _coverage(codes, K):
    return float(len(np.unique(codes))) / float(K)


def rq_3layer(emb_np: np.ndarray, codebook_size: int = 256, n_steps: int = 1000, seed: int = 42):
    """3 层 RQ 量化, 返回 (codes_0/1/2, MSE_total, cov_0/1/2)"""
    x = emb_np.astype(np.float32)
    N = x.shape[0]

    codes_0 = _simple_kmeans(x, K=codebook_size, n_iter=n_steps, seed=seed)
    centroid_0 = np.zeros((codebook_size, x.shape[1]), dtype=np.float32)
    for k in range(codebook_size):
        mask = (codes_0 == k)
        if mask.any():
            centroid_0[k] = x[mask].mean(axis=0)
    r1 = x - centroid_0[codes_0]
    mse_0 = float(np.mean(r1 ** 2))

    codes_1 = _simple_kmeans(r1, K=codebook_size, n_iter=n_steps, seed=seed + 1)
    centroid_1 = np.zeros((codebook_size, x.shape[1]), dtype=np.float32)
    for k in range(codebook_size):
        mask = (codes_1 == k)
        if mask.any():
            centroid_1[k] = r1[mask].mean(axis=0)
    r2 = r1 - centroid_1[codes_1]
    mse_1 = float(np.mean(r2 ** 2))

    codes_2 = _simple_kmeans(r2, K=codebook_size, n_iter=n_steps // 2, seed=seed + 2)
    centroid_2 = np.zeros((codebook_size, x.shape[1]), dtype=np.float32)
    for k in range(codebook_size):
        mask = (codes_2 == k)
        if mask.any():
            centroid_2[k] = r2[mask].mean(axis=0)

    mse_total = mse_0 + mse_1
    cov_0 = _coverage(codes_0, codebook_size)
    cov_1 = _coverage(codes_1, codebook_size)
    cov_2 = _coverage(codes_2, codebook_size)

    return {
        'mse_layer_0': mse_0,
        'mse_layer_1': mse_1,
        'mse_total': mse_total,
        'cov_layer_0': cov_0,
        'cov_layer_1': cov_1,
        'cov_layer_2': cov_2,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True, help='input embedding .pt (e.g. task53_log1p_s4_ae)')
    parser.add_argument('--output', required=True, help='output L3-rebased embedding .pt')
    parser.add_argument('--lambda', type=float, default=0.05, help='L3 penalty λ')
    parser.add_argument('--lambdas', default='0.01,0.05,0.2,1.0', help='多个 λ sweep, 选最佳')
    parser.add_argument('--codebook-size', type=int, default=256)
    parser.add_argument('--rq-steps', type=int, default=1000)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--summary', default=None)
    args = parser.parse_args()
    lam_arg = getattr(args, 'lambda')

    input_path = ROOT / args.input
    output_path = ROOT / args.output

    if not input_path.exists():
        raise FileNotFoundError(f'输入 embedding 不存在: {input_path}')

    # 加载
    emb = torch.load(input_path, weights_only=False, map_location='cpu')
    if isinstance(emb, dict) and 'fused_item' in emb:
        emb = emb['fused_item']
    emb = emb.float()
    print(f'[task54_l3_norm] 输入: {input_path}, shape={tuple(emb.shape)}')

    # Step 1: RQ baseline
    print('[task54_l3_norm] Step 1: RQ baseline ...')
    t0 = time.time()
    baseline = rq_3layer(emb.numpy(), codebook_size=args.codebook_size,
                         n_steps=args.rq_steps, seed=args.seed)
    t_baseline = time.time() - t0
    print(f'[task54_l3_norm] baseline MSE={baseline["mse_total"]:.4f} '
          f'cov0={baseline["cov_layer_0"]:.3f} cov1={baseline["cov_layer_1"]:.3f} '
          f'({t_baseline:.1f}s)')

    # Step 2: λ sweep
    lambdas = [float(x) for x in args.lambdas.split(',')]
    print(f'[task54_l3_norm] Step 2: λ sweep ({len(lambdas)} values) ...')
    results = []
    best_lambda = None
    best_cov0 = baseline['cov_layer_0']
    best_ratio = 1.0

    for lam in lambdas:
        rebased = l3_norm_rebase(emb, lam=lam)
        rebased_norm = torch.norm(rebased, p=2, dim=-1)
        print(f'[task54_l3_norm]   λ={lam:.3f}: rebased norm '
              f'min={rebased_norm.min():.4f} max={rebased_norm.max():.4f} '
              f'mean={rebased_norm.mean():.4f}')

        t0 = time.time()
        m = rq_3layer(rebased.numpy(), codebook_size=args.codebook_size,
                      n_steps=args.rq_steps, seed=args.seed)
        t_rq = time.time() - t0
        ratio = m['mse_total'] / max(baseline['mse_total'], 1e-9)
        cov0_gain = m['cov_layer_0'] - baseline['cov_layer_0']
        print(f'         MSE={m["mse_total"]:.4f} (ratio={ratio:.3f}) '
              f'cov0={m["cov_layer_0"]:.3f} (gain={cov0_gain:+.3f}) '
              f'({t_rq:.1f}s)')
        results.append({
            'lambda': lam,
            'mse_total': m['mse_total'],
            'mse_ratio': ratio,
            'cov_layer_0': m['cov_layer_0'],
            'cov_layer_1': m['cov_layer_1'],
            'cov_layer_2': m['cov_layer_2'],
            'cov0_gain': cov0_gain,
            'rebased_norm_min': float(rebased_norm.min()),
            'rebased_norm_max': float(rebased_norm.max()),
            'rebased_norm_mean': float(rebased_norm.mean()),
        })
        if cov0_gain > (best_cov0 - baseline['cov_layer_0']):
            best_cov0 = m['cov_layer_0']
            best_lambda = lam
            best_ratio = ratio
            best_metrics = m

    # 决策
    print('')
    print('=' * 60)
    print('Task #54 L3 norm 决策汇总')
    print('=' * 60)
    print(f'Best λ={best_lambda}, cov0 {baseline["cov_layer_0"]:.3f} → {best_cov0:.3f} '
          f'(+{best_cov0 - baseline["cov_layer_0"]:+.3f})')
    print(f'MSE ratio (L3 / baseline) = {best_ratio:.3f}')

    if best_cov0 - baseline['cov_layer_0'] > 0.05 and best_ratio <= 1.0:
        decision = '✅ L3 显著修复 layer-0 cov0 (≥+0.05), 用此 embedding 重训 Stage 3'
        valid = True
        # 保存最佳 λ 的 rebased
        best_rebased = l3_norm_rebase(emb, lam=best_lambda)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(best_rebased, output_path)
        print(f'[task54_l3_norm] ✅ 已保存: {output_path}')
    elif best_cov0 > baseline['cov_layer_0']:
        decision = '⚠️ L3 边际修复, 可尝试 λ > 0.1'
        valid = False
        best_rebased = l3_norm_rebase(emb, lam=best_lambda)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(best_rebased, output_path)
    else:
        decision = '❌ L3 rebase 无效 (λ sweep 全部退化), post-hoc 模拟不足以代训练 penalty'
        valid = False

    summary = {
        'input': str(input_path),
        'output': str(output_path) if valid else None,
        'baseline': baseline,
        'lambda_sweep': results,
        'best_lambda': best_lambda,
        'best_cov0': best_cov0,
        'best_ratio': best_ratio,
        'decision': decision,
        'valid': valid,
        't_baseline': t_baseline,
    }
    if args.summary:
        summary_path = ROOT / args.summary
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
        print(f'[task54_l3_norm] summary → {summary_path}')

    print('')
    print(decision)


if __name__ == '__main__':
    main()
