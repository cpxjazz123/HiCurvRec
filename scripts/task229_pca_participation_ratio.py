#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Task #229 PCA Participation Ratio — codeword 有效维度诊断 (CPU only, no GPU).

目的 (用户 2026-07-27 决策): 在 Gate 1 启动前, 用 PCA participation ratio 测每层
codeword embedding 的"有效维度" $D_{\\text{eff}}$, 验证论文 Eq11/12 的核心动机
(L2 codebook 256 个 32-d codeword 在欧氏空间太挤, 双曲空间能解决) 是否成立.

公式: $D_{\\text{eff}} = \\frac{(\\sum_i \\lambda_i)^2}{\\sum_i \\lambda_i^2}$
其中 $\\lambda_i$ 是 codeword 矩阵的 PCA 特征值 (per layer).

决策规则 (用户):
  - $D_{\\text{eff}} > 8$  → L2 拥挤论证在数学上站不住, 曲率解决的不是这个问题, 论文动机失效
  - $D_{\\text{eff}} \\approx 6\\sim 8$ → 临界, 双曲能买到约 1.5 倍间隔, 值得做
  - $D_{\\text{eff}} < 6$  → 拥挤是真问题, 双曲增益缩到 1.0~1.2 倍, 收益有限

数据源:
  - baseline c=1: products/task84/ckpt/Instruments/.../best_collision_model.pth (R@10=0.1020)
  - c=10 ep1: products/m_arm/m_radius_spread_step3_50ep_wdiv100_c10_jul-27-2026_*/best_collision_model.pth
              (R@10 训练中, 预期 < 0.07)

输出: descriptions/task229_pca_participation_ratio.json
"""
from __future__ import annotations

import json
import sys

import numpy as np
import torch

sys.path.insert(0, '/home/wlia0047/ar57/wenyu/GeneRec/HG-Rec')

from model.hrqvae import HRQVAE
from model.utils import EmbDataset


def load_codebooks_from_ckpt(ckpt_path: str):
    """Load codebook embeddings from ckpt (CPU only)."""
    ckpt = torch.load(ckpt_path, weights_only=False, map_location=torch.device('cpu'))
    ckpt_args = vars(ckpt['args']) if not isinstance(ckpt['args'], dict) else ckpt['args']
    state_dict = ckpt['state_dict']

    data = EmbDataset(ckpt_args['data_path'])

    # Build model skeleton (no need to load state_dict — only codebook weights)
    r_target_list_raw = ckpt_args.get('r_target_list', None)
    if isinstance(r_target_list_raw, str) and r_target_list_raw:
        r_target_list = [float(x) for x in r_target_list_raw.split(',')]
    else:
        r_target_list = r_target_list_raw

    model = HRQVAE(
        in_dim=data.dim,
        num_emb_list=ckpt_args['num_emb_list'],
        e_dim=ckpt_args['e_dim'],
        layers=ckpt_args['layers'],
        dropout_prob=ckpt_args.get('dropout_prob', 0.0),
        bn=ckpt_args.get('bn', False),
        loss_type=ckpt_args['loss_type'],
        quant_loss_weight=ckpt_args.get('quant_loss_weight', 1.0),
        beta=ckpt_args['beta'],
        kmeans_init=ckpt_args.get('kmeans_init', False),
        kmeans_iters=ckpt_args.get('kmeans_iters', 100),
        sk_eps=ckpt_args['sk_epsilons'],
        sk_iters=ckpt_args['sk_iters'],
        product_manifold=ckpt_args.get('product_manifold', False),
        angular_dim=ckpt_args.get('angular_dim', None),
        radial_dim=ckpt_args.get('radial_dim', None),
        kappa_mode=ckpt_args.get('kappa_mode', 'fixed'),
        theta_init=ckpt_args.get('theta_init', 0.0),
        r_target_list=r_target_list,
    )

    # Extract only vq_layers.embedding weights (codebook)
    codebooks = []
    for li, vq in enumerate(model.hrq.vq_layers):
        # Find matching key in state_dict
        keys = [k for k in state_dict.keys() if f'vq_layers.{li}.' in k and 'embedding' in k]
        if keys:
            w = state_dict[keys[0]].detach().cpu().numpy()
        else:
            w = vq.embedding.weight.detach().cpu().numpy()
        codebooks.append(w)
    return codebooks, ckpt, ckpt_args


def participation_ratio(X: np.ndarray, center: bool = True) -> dict:
    """PCA participation ratio (effective dimensionality).

    X: (n_samples, n_features) — typically (n_codewords, e_dim=32)
    $D_{\\text{eff}} = \\frac{(\sum \\lambda_i)^2}{\\sum \\lambda_i^2}$
    """
    if center:
        X_centered = X - X.mean(axis=0, keepdims=True)
    else:
        X_centered = X
    # Covariance eigenvalues (n_features=32)
    cov = np.cov(X_centered, rowvar=False)  # (n_features, n_features)
    eigvals = np.linalg.eigvalsh(cov)  # ascending order
    eigvals = np.maximum(eigvals, 0)  # numerical stability

    sum_lambda = eigvals.sum()
    sum_lambda_sq = (eigvals ** 2).sum()
    D_eff = (sum_lambda ** 2) / sum_lambda_sq if sum_lambda_sq > 0 else 0.0

    # Explained variance ratio (top-k / total)
    sorted_eigvals = eigvals[::-1]  # descending
    total_var = sorted_eigvals.sum()
    explained_ratio = (sorted_eigvals.cumsum() / total_var) if total_var > 0 else sorted_eigvals

    # Top eigenvalue share
    top_share = sorted_eigvals[0] / total_var if total_var > 0 else 0.0

    # Effective rank (Roy & Vetterli 2007) — same formula but typically computed differently
    # Use classical participation ratio (Gao & Erik 2017)
    return {
        'D_eff': float(D_eff),
        'n_codewords': int(X.shape[0]),
        'e_dim': int(X.shape[1]),
        'sum_lambda': float(sum_lambda),
        'sum_lambda_sq': float(sum_lambda_sq),
        'top_eigenvalue': float(sorted_eigvals[0]),
        'top_eigenvalue_share': float(top_share),
        'explained_var_top1': float(explained_ratio[0]) if len(explained_ratio) > 0 else 0.0,
        'explained_var_top2': float(explained_ratio[1]) if len(explained_ratio) > 1 else 0.0,
        'explained_var_top3': float(explained_ratio[2]) if len(explained_ratio) > 2 else 0.0,
        'explained_var_top5': float(explained_ratio[4]) if len(explained_ratio) > 4 else 0.0,
        'explained_var_top8': float(explained_ratio[7]) if len(explained_ratio) > 7 else 0.0,
        'explained_var_top16': float(explained_ratio[15]) if len(explained_ratio) > 15 else 0.0,
        'explained_var_top32': float(explained_ratio[-1]) if len(explained_ratio) > 0 else 0.0,
        'eigenvalues_descending': sorted_eigvals.tolist(),
    }


def nearest_neighbor_stats(X: np.ndarray, topk: int = 5) -> dict:
    """Compute nearest-neighbor distance statistics for crowd assessment.

    Returns average nearest-neighbor distance and its std (proxy for crowding).
    """
    from scipy.spatial.distance import cdist
    dists = cdist(X, X, metric='euclidean')  # (n, n)
    np.fill_diagonal(dists, np.inf)
    nn_dists = dists.min(axis=1)
    sorted_nn = np.sort(nn_dists)

    # Top-k mean distances (vs all other codewords)
    # dists[i, j] = 0 for j == i (diagonal), after fill_diagonal it's inf
    # so np.sort ascending [:, 0] is the nearest neighbor (excluding self), [:, topk-1] is the topk-th nearest
    if topk < X.shape[0] - 1:
        sorted_dists = np.sort(dists, axis=1)
        topk_mean_dist = float(sorted_dists[:, topk - 1].mean())
    else:
        topk_mean_dist = None

    return {
        'mean_nn_dist': float(nn_dists.mean()),
        'std_nn_dist': float(nn_dists.std()),
        'min_nn_dist': float(nn_dists.min()),
        'p10_nn_dist': float(np.percentile(nn_dists, 10)),
        'p50_nn_dist': float(np.percentile(nn_dists, 50)),
        'p90_nn_dist': float(np.percentile(nn_dists, 90)),
        'max_nn_dist': float(nn_dists.max()),
        f'top{topk}_mean_dist': topk_mean_dist,
    }


def main():
    ckpts = {
        'baseline_c1': '/home/wlia0047/ar57/wenyu/GeneRec/products/task84/ckpt/Instruments/Jul-23-2026_20-08-06_beta_1.000_codebook_[64,128,256]_sk_0.000/best_collision_model.pth',
        'c10_ep1': '/home/wlia0047/ar57/wenyu/GeneRec/products/m_arm/m_radius_spread_step3_50ep_wdiv100_c10_jul-27-2026_16-18-09/Jul-27-2026_16-18-16_beta_0.500_codebook_[64,128,256]_sk_0.000/best_collision_model.pth',
    }

    all_results = {}

    for arm_name, ckpt_path in ckpts.items():
        print(f"\n=== [{arm_name}] Loading ckpt: {ckpt_path} ===")
        codebooks, ckpt, ckpt_args = load_codebooks_from_ckpt(ckpt_path)

        print(f"  ckpt epoch: {ckpt.get('epoch', '?')}, best_loss: {ckpt.get('best_loss', '?')}, "
              f"best_collision_rate: {ckpt.get('best_collision_rate', '?')}")
        print(f"  num_emb_list={ckpt_args['num_emb_list']}, e_dim={ckpt_args['e_dim']}, "
              f"r_target_list={ckpt_args.get('r_target_list', 'None')}")

        all_results[arm_name] = {
            'ckpt_path': ckpt_path,
            'ckpt_epoch': int(ckpt.get('epoch', -1)),
            'best_loss': float(ckpt.get('best_loss', -1)),
            'best_collision_rate': float(ckpt.get('best_collision_rate', -1)),
            'num_emb_list': ckpt_args['num_emb_list'],
            'e_dim': ckpt_args['e_dim'],
            'per_layer': {},
        }

        for li, cb in enumerate(codebooks):
            print(f"\n  Layer {li} (n={cb.shape[0]}, d={cb.shape[1]}):")

            # 1. Participation ratio (effective dimensionality)
            pr = participation_ratio(cb)
            print(f"    D_eff (participation ratio) = {pr['D_eff']:.3f}")
            print(f"    Top-1 eigenvalue share = {pr['top_eigenvalue_share']:.3f}")
            print(f"    Explained var: top1={pr['explained_var_top1']:.3f}, "
                  f"top2={pr['explained_var_top2']:.3f}, "
                  f"top3={pr['explained_var_top3']:.3f}, "
                  f"top5={pr['explained_var_top5']:.3f}, "
                  f"top8={pr['explained_var_top8']:.3f}, "
                  f"top16={pr['explained_var_top16']:.3f}, "
                  f"top32={pr['explained_var_top32']:.3f}")

            # 2. Nearest-neighbor distance stats
            nn = nearest_neighbor_stats(cb, topk=5)
            print(f"    NN distance: mean={nn['mean_nn_dist']:.4f}, "
                  f"std={nn['std_nn_dist']:.4f}, "
                  f"min={nn['min_nn_dist']:.4f}, "
                  f"p10={nn['p10_nn_dist']:.4f}, "
                  f"p50={nn['p50_nn_dist']:.4f}, "
                  f"p90={nn['p90_nn_dist']:.4f}")
            print(f"    Top-5 mean distance: {nn['top5_mean_dist']:.4f}")

            # Decision per user rule
            d_eff = pr['D_eff']
            if d_eff > 8:
                verdict = 'paper_motivation_fails'  # L2 不挤, 曲率不是 bottleneck
                interp = f"D_eff={d_eff:.2f} > 8 → L2 拥挤论证在数学上站不住, 论文 Eq11/12 动机失效"
            elif d_eff >= 6:
                verdict = 'critical_1.5x_worth'  # 临界, 双曲 ~1.5× 间隔, 值得做
                interp = f"D_eff={d_eff:.2f} ∈ [6,8] → 临界, 双曲能买到 ~1.5× 间隔, Gate 1 值得做"
            else:
                verdict = 'crowded_but_gain_limited'  # 真挤, 但双曲增益 1.0-1.2×, 收益有限
                interp = f"D_eff={d_eff:.2f} < 6 → 拥挤是真问题, 但双曲增益仅 1.0~1.2×, Gate 1 收益有限"

            print(f"    >>> {verdict}: {interp}")

            all_results[arm_name]['per_layer'][f'layer_{li}'] = {
                'participation_ratio': pr,
                'nearest_neighbor': nn,
                'verdict': verdict,
                'interpretation': interp,
                'd_eff': d_eff,
            }

    # Summary table
    print("\n" + "=" * 80)
    print("Summary: PCA participation ratio per layer")
    print("=" * 80)
    print(f"{'Arm':<15} {'Layer':<6} {'n':<5} {'D_eff':<8} {'Top1 share':<12} {'Verdict':<28}")
    print("-" * 80)
    for arm_name, arm_data in all_results.items():
        for li_key, layer_data in arm_data['per_layer'].items():
            li = int(li_key.split('_')[1])
            d_eff = layer_data['d_eff']
            top1 = layer_data['participation_ratio']['top_eigenvalue_share']
            verdict_short = layer_data['verdict'][:28]
            print(f"{arm_name:<15} L{li:<5} {layer_data['participation_ratio']['n_codewords']:<5} "
                  f"{d_eff:<8.3f} {top1:<12.4f} {verdict_short:<28}")

    output_path = '/home/wlia0047/ar57/wenyu/GeneRec/descriptions/task229_pca_participation_ratio.json'
    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\n✅ Saved: {output_path}")


if __name__ == '__main__':
    main()