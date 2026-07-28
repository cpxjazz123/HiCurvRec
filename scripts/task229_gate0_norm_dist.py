#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Task #229 Gate 0 — codeword ‖e‖ 分布 + max_c2 预测 (zero training, no GPU).

目的 (用户 2026-07-27 提议):
  测量 baseline c=1 RQ-VAE ckpt 的 codeword 范数分布 (per layer p50/p90/max/min),
  验证 identity $c \cdot \|e\|^2 = 1.0$ → max_c2 = (max/median)^2,
  并预测 c=[13.45, 96.21, 199.78] proposal 下的 max_c2 是否 < 0.5 警戒线.

输出:
  - descriptions/task229_gate0_norm_dist.json (per-layer distribution + max_c2 predictions)
  - 同时检查 c=10 ep1 ckpt 做对照

不依赖 GPU (CPU-only).
"""
from __future__ import annotations

import argparse
import json
import sys

import numpy as np
import torch

sys.path.insert(0, '/home/wlia0047/ar57/wenyu/GeneRec/HG-Rec')

from model.hrqvae import HRQVAE
from model.utils import EmbDataset


def load_hrqvae_from_ckpt(ckpt_path: str, device: str = 'cpu'):
    """Load HRQVAE model + codebook embeddings from ckpt."""
    ckpt = torch.load(ckpt_path, weights_only=False, map_location=torch.device('cpu'))
    ckpt_args = vars(ckpt['args']) if not isinstance(ckpt['args'], dict) else ckpt['args']
    state_dict = ckpt['state_dict']

    data = EmbDataset(ckpt_args['data_path'])

    # Build model with same kwargs (without loading state_dict yet)
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
    ).to(device)

    # Only load VQ-related weights (skip encoder/decoder to avoid shape mismatch across tasks)
    # We only need vq_layers.embedding.weight which is the codebook
    cb_state = {}
    for k, v in state_dict.items():
        if 'vq_layers' in k and 'embedding' in k:
            cb_state[k] = v
    if not cb_state:
        # Fallback: load full state_dict and see if it works
        model.load_state_dict(state_dict, strict=False)
        cb_state = {k: v for k, v in state_dict.items() if 'vq_layers' in k and 'embedding' in k}

    # Manually extract codebook weights
    codebooks = []
    for li, vq in enumerate(model.hrq.vq_layers):
        # Find the matching state_dict key for this layer
        keys = [k for k in cb_state.keys() if f'vq_layers.{li}.' in k]
        if keys:
            w = cb_state[keys[0]].detach().cpu()
        else:
            # Use the model's actual embedding
            w = vq.embedding.weight.detach().cpu()
        codebooks.append(w)
    return model, codebooks, ckpt, ckpt_args


def compute_norm_stats(codebooks):
    """For each layer's codebook, compute ‖e‖ distribution."""
    stats = []
    for li, cb in enumerate(codebooks):
        # cb shape: (num_emb, e_dim)
        norms = cb.norm(dim=-1).numpy()  # (num_emb,) — this is切空间 Euclidean norm
        norms_sq = norms ** 2
        stats.append({
            'layer': li,
            'num_emb': int(cb.shape[0]),
            'e_dim': int(cb.shape[1]),
            'norm_min': float(norms.min()),
            'norm_p10': float(np.percentile(norms, 10)),
            'norm_p25': float(np.percentile(norms, 25)),
            'norm_p50': float(np.percentile(norms, 50)),
            'norm_p75': float(np.percentile(norms, 75)),
            'norm_p90': float(np.percentile(norms, 90)),
            'norm_p95': float(np.percentile(norms, 95)),
            'norm_p99': float(np.percentile(norms, 99)),
            'norm_max': float(norms.max()),
            'norm_mean': float(norms.mean()),
            'norm_std': float(norms.std()),
            'norm_sq_min': float(norms_sq.min()),
            'norm_sq_p50': float(np.percentile(norms_sq, 50)),
            'norm_sq_p90': float(np.percentile(norms_sq, 90)),
            'norm_sq_p95': float(np.percentile(norms_sq, 95)),
            'norm_sq_p99': float(np.percentile(norms_sq, 99)),
            'norm_sq_max': float(norms_sq.max()),
        })
    return stats


def predict_max_c2(stats, c_list):
    """Given norm stats and c proposals, predict max_c2 = max_i c·‖e_i‖²."""
    results = []
    for li_s in stats:
        li_results = []
        for c in c_list:
            max_c2 = c * li_s['norm_sq_max']
            p50_c2 = c * li_s['norm_sq_p50']
            p90_c2 = c * li_s['norm_sq_p90']
            max_c2_via_p50_p99 = c * li_s['norm_sq_p99']
            li_results.append({
                'c': c,
                'max_c2': max_c2,
                'p50_c2': p50_c2,
                'p90_c2': p90_c2,
                'p99_c2': max_c2_via_p50_p99,
                'ratio_max_over_p50': float(li_s['norm_sq_max'] / li_s['norm_sq_p50']),
                'ratio_max_over_p90': float(li_s['norm_sq_max'] / li_s['norm_sq_p90']),
                'passes_warning_line_0.5': max_c2 < 0.5,
            })
        results.append({'layer': li_s['layer'], 'predictions': li_results})
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline_ckpt", type=str,
                        default="/home/wlia0047/ar57/wenyu/GeneRec/products/task84/ckpt/Instruments/Jul-23-2026_20-08-06_beta_1.000_codebook_[64,128,256]_sk_0.000/best_collision_model.pth")
    parser.add_argument("--c10_ckpt", type=str,
                        default="/home/wlia0047/ar57/wenyu/GeneRec/products/m_arm/m_radius_spread_step3_50ep_wdiv100_c10_jul-27-2026_16-18-09/Jul-27-2026_16-18-16_beta_0.500_codebook_[64,128,256]_sk_0.000/best_collision_model.pth")
    parser.add_argument("--output", type=str,
                        default="/home/wlia0047/ar57/wenyu/GeneRec/descriptions/task229_gate0_norm_dist.json")
    parser.add_argument("--c_proposals", type=float, nargs='+',
                        default=[1.0, 10.0, 13.45, 63.7, 96.21, 100.0, 199.78])
    args = parser.parse_args()

    print(f"=== Task #229 Gate 0 — codeword ‖e‖ 分布测量 ===\n")

    all_results = {}

    # 1. Baseline c=1 ckpt (Task #84)
    print(f"[1/2] Loading baseline c=1 ckpt: {args.baseline_ckpt}")
    model_b, codebooks_b, ckpt_b, ckpt_args_b = load_hrqvae_from_ckpt(args.baseline_ckpt)
    stats_b = compute_norm_stats(codebooks_b)
    print(f"  Per-layer ‖e‖ distribution (切空间 Euclidean norm):")
    for s in stats_b:
        print(f"    L{s['layer']} (n={s['num_emb']}): "
              f"min={s['norm_min']:.4f}, p50={s['norm_p50']:.4f}, "
              f"p90={s['norm_p90']:.4f}, max={s['norm_max']:.4f}, "
              f"mean={s['norm_mean']:.4f}±{s['norm_std']:.4f}")
    pred_b = predict_max_c2(stats_b, args.c_proposals)
    print(f"  max_c2 predictions (using切空间 ‖e‖²):")
    for li_p in pred_b:
        for pr in li_p['predictions']:
            warn = "✓" if pr['passes_warning_line_0.5'] else "✗ FAIL"
            print(f"    L{li_p['layer']} c={pr['c']:>7.2f}: "
                  f"max_c2={pr['max_c2']:.4f}, p50_c2={pr['p50_c2']:.4f}, "
                  f"max/p50={pr['ratio_max_over_p50']:.2f}× {warn}")

    all_results['baseline_c1'] = {
        'ckpt_path': args.baseline_ckpt,
        'best_loss': float(ckpt_b.get('best_loss', -1)),
        'best_collision_rate': float(ckpt_b.get('best_collision_rate', -1)),
        'epoch': int(ckpt_b.get('epoch', -1)),
        'num_emb_list': ckpt_args_b['num_emb_list'],
        'e_dim': ckpt_args_b['e_dim'],
        'norm_stats_per_layer': stats_b,
        'max_c2_predictions': pred_b,
    }

    # 2. c=10 ckpt (Task #233 训练产物, 方向 I) — 对照
    print(f"\n[2/2] Loading c=10 ep1 ckpt: {args.c10_ckpt}")
    model_c10, codebooks_c10, ckpt_c10, ckpt_args_c10 = load_hrqvae_from_ckpt(args.c10_ckpt)
    stats_c10 = compute_norm_stats(codebooks_c10)
    print(f"  Per-layer ‖e‖ distribution:")
    for s in stats_c10:
        print(f"    L{s['layer']} (n={s['num_emb']}): "
              f"min={s['norm_min']:.4f}, p50={s['norm_p50']:.4f}, "
              f"p90={s['norm_p90']:.4f}, max={s['norm_max']:.4f}")
    pred_c10 = predict_max_c2(stats_c10, args.c_proposals)
    print(f"  max_c2 predictions:")
    for li_p in pred_c10:
        for pr in li_p['predictions']:
            warn = "✓" if pr['passes_warning_line_0.5'] else "✗ FAIL"
            print(f"    L{li_p['layer']} c={pr['c']:>7.2f}: "
                  f"max_c2={pr['max_c2']:.4f} {warn}")

    all_results['c10_ep1'] = {
        'ckpt_path': args.c10_ckpt,
        'best_loss': float(ckpt_c10.get('best_loss', -1)),
        'best_collision_rate': float(ckpt_c10.get('best_collision_rate', -1)),
        'epoch': int(ckpt_c10.get('epoch', -1)),
        'num_emb_list': ckpt_args_c10['num_emb_list'],
        'e_dim': ckpt_args_c10['e_dim'],
        'norm_stats_per_layer': stats_c10,
        'max_c2_predictions': pred_c10,
    }

    # Critical analysis
    print("\n=== 关键诊断: c·‖e‖²_p50 = 1.0 calibration ===")
    print("(用户身份前提: c = 1/‖e‖²_p50, 此时 p50_c2=1.0; max_c2 = max/p50 倍率)")
    print()
    for label, res in all_results.items():
        print(f"[{label}]")
        for s in res['norm_stats_per_layer']:
            implied_c = 1.0 / s['norm_sq_p50']
            implied_max_c2 = s['norm_sq_max'] / s['norm_sq_p50']
            print(f"  L{s['layer']}: ‖e‖²_p50={s['norm_sq_p50']:.4f} → "
                  f"implied c={implied_c:.2f}, expected max_c2={implied_max_c2:.4f}, "
                  f"< 0.5? {implied_max_c2 < 0.5}")

    with open(args.output, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n✅ Saved: {args.output}")


if __name__ == '__main__':
    main()