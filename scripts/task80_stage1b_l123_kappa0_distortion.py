#!/usr/bin/env python3
"""Task #80 Stage 1b — 补测 L1/L2/L3 在 κ=0 (Euclidean) 时的 distortion.

L0 @ κ=0 distortion=0 是数学必然 (Sarkar hyperbolic distance 在 κ=0 退化为 Euclidean,
ratio = 1.0 for all pairs, CV = 0). 这个不是发现, 是 trivial.

真正有信息量的是 L1/L2/L3 在 κ=0 vs κ=-0.5 谁更小:
- 如果 L1@κ=0 比 L1@κ=-0.5 小: L1 用 κ=0 (欧氏几何拟合更好)
- 如果 L1@κ=-0.5 比 L1@κ=0 小: L1 用 κ=-0.5 (双曲几何拟合更好)
"""
import argparse
import json
import os
import sys
import time
import numpy as np
import torch

sys.path.insert(0, '/home/wlia0047/ar57/wenyu/GeneRec/scripts')
from task92_rqvae_delta_hyperbolicity import load_rqvae, extract_per_layer_residuals
from task80_stage1_kappa_distortion_grid import compute_distortion


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt', required=True)
    p.add_argument('--emb', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--n_subset', type=int, default=300)
    p.add_argument('--device', default='cuda:0')
    return p.parse_args()


def main():
    args = parse_args()
    print(f"[Task #80 Stage 1b] L1/L2/L3 @ κ=0 distortion (Euclidean reference)")

    model, model_args = load_rqvae(args.ckpt, args.device)
    embeddings_np = np.load(args.emb)
    print(f"  embedding shape: {embeddings_np.shape}")

    residuals, _ = extract_per_layer_residuals(
        model, embeddings_np, args.device, batch_size=2048
    )
    num_layers = len(residuals)
    print(f"  num layers: {num_layers}")
    for i, r in enumerate(residuals):
        norm_max = r.norm(dim=-1).max().item()
        print(f"    L{i}: shape={tuple(r.shape)}, norm_max={norm_max:.4f}")

    # 测试 κ=0 (Euclidean) 对 L0/L1/L2/L3 全部
    print()
    print(f"{'Layer':<8} {'label':<32} {'CV@κ=0':<12} {'CV@κ=-0.5':<12} {'Winner':<10}")
    print("-" * 75)

    results = {}
    for i, r in enumerate(residuals):
        r_np = r.detach().cpu().numpy()
        label = "raw_encoded" if i == 0 else f"residual_after_layer_{i-1}"
        cv_k0 = compute_distortion(r_np, 0.0, n_subset=args.n_subset, seed=args.seed)
        cv_k_neg = compute_distortion(r_np, -0.5, n_subset=args.n_subset, seed=args.seed)
        # 对于 L0, κ=0 是 trivial 0 (Sarkar 退化), 不是对比依据
        if i == 0:
            winner = "κ=0 (trivial)"
        else:
            winner = "κ=0 (Euclidean)" if cv_k0 < cv_k_neg else "κ=-0.5 (hyperbolic)"
        print(f"L{i}      {label:<32} {cv_k0:<12.6f} {cv_k_neg:<12.6f} {winner:<10}")
        results[f"L{i}"] = {
            'cv_kappa_0': cv_k0,
            'cv_kappa_neg0.5': cv_k_neg,
            'winner': winner,
            'label': label,
        }

    # 决策: C 组 per-layer κ 选择
    print()
    print("=== C 组 (per-layer κ) 重新决策 ===")
    c_group = {}
    for i in range(num_layers):
        r = results[f"L{i}"]
        if i == 0:
            chosen_k = 0.0  # L0 trivial, 用 κ=0
            chosen_d = r['cv_kappa_0']
        else:
            if r['cv_kappa_0'] <= r['cv_kappa_neg0.5']:
                chosen_k = 0.0
                chosen_d = r['cv_kappa_0']
            else:
                chosen_k = -0.5
                chosen_d = r['cv_kappa_neg0.5']
        c_group[i] = {'kappa': chosen_k, 'distortion': chosen_d}
        print(f"  L{i}: κ* = {chosen_k:+.2f}, distortion = {chosen_d:.6f}")

    output = {
        'task': 'task80_stage1b_l123_kappa0_distortion',
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'ckpt': args.ckpt,
        'emb': args.emb,
        'n_subset': args.n_subset,
        'seed': args.seed,
        'all_results': results,
        'C_group_redecided': {
            f"L{i}": {'kappa': v['kappa'], 'distortion': v['distortion']}
            for i, v in c_group.items()
        },
    }
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\n[Stage 1b] Result saved to {args.output}")


if __name__ == '__main__':
    main()
