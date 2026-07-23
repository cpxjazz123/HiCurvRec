#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Task #80 Stage 1c (v2) — Phase 2 true distortion metric 重跑 (含 κ=0 baseline).

对比 Stage 1 / Stage 1b:
  - Stage 1 (旧): 用 CV-of-ratio 作为 metric. κ=0 → ratio=1.0 → CV=0 → trivial.
  - Stage 1b: 在 Stage 1 基础上补测 L1/L2/L3 κ=0, 发现 trivial.
  - Stage 1c (Phase 2): 用 Riemannian GD MDS, normalized stress (Kruskal stress-1).
      κ=0 时 Riemannian GD 退化为 Euclidean MDS, 不再 trivial — stress 应 > 0.

本脚本 (v2):
  - κ grid: L0 ∈ {0, -0.3, -0.5, -1.0, -2.0}; L1/L2/L3 ∈ {0, -0.5, -1.0, -1.5, -2.0}
    全部含 κ=0 (跟 Stage 1b 漏了相反, 现在全包含).
  - n_subset=300, n_iter=200 (Phase 2 默认).
  - 4 层 × 5 个 κ = 20 个 distortion 值.
  - 输出 C 组 (per-layer best κ) + B 组 (global κ by sum).

预期:
  - κ=0 不再 trivial (stress > 0).
  - 真正的最佳 κ 可能不是 0.
  - C 组 per-layer κ 可能不再全部 = 0.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Dict, List

import numpy as np
import torch

# Phase 2 metric: Riemannian GD MDS
sys.path.insert(0, '/home/wlia0047/ar57/wenyu/GeneRec/scripts')
from task80_stage1c_true_distortion import compute_true_distortion  # noqa: E402

# RQ-VAE 加载 + per-layer residual 抽取 (复用 Task #92 的实现)
sys.path.insert(0, '/home/wlia0047/ar57/wenyu/GeneRec/scripts')
from task92_rqvae_delta_hyperbolicity import load_rqvae, extract_per_layer_residuals  # noqa: E402


# ==============================================================================
# κ grids (per-layer, 必须包含 κ=0)
# ==============================================================================
KAPPA_GRID_L0 = [0.0, -0.3, -0.5, -1.0, -2.0]      # 5 个 κ, 含 0
KAPPA_GRID_DEEP = [0.0, -0.5, -1.0, -1.5, -2.0]    # 5 个 κ, 含 0

LAYER_KAPPAS = {
    0: KAPPA_GRID_L0,
    1: KAPPA_GRID_DEEP,
    2: KAPPA_GRID_DEEP,
    3: KAPPA_GRID_DEEP,
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt', required=True)
    p.add_argument('--emb', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--n_subset', type=int, default=300)
    p.add_argument('--n_iter', type=int, default=200)
    p.add_argument('--lr', type=float, default=0.005)
    p.add_argument('--d', type=int, default=32)
    p.add_argument('--device', default='cuda:1',
                   help='cuda:1 或 cuda:3 是空闲 GPU (R7: 不抢已占卡)')
    return p.parse_args()


def main() -> None:
    args = parse_args()
    print(f"[Task #80 Stage 1c v2] Phase 2 true distortion grid search (含 κ=0)")
    print(f"  ckpt:           {args.ckpt}")
    print(f"  emb:            {args.emb}")
    print(f"  n_subset:       {args.n_subset}")
    print(f"  n_iter:         {args.n_iter}")
    print(f"  lr:             {args.lr}")
    print(f"  d:              {args.d}")
    print(f"  device:         {args.device}")
    print(f"  κ grids:")
    print(f"    L0:           {KAPPA_GRID_L0}")
    print(f"    L1/L2/L3:     {KAPPA_GRID_DEEP}")

    # 加载 RQ-VAE + 抽取 per-layer residuals
    model, model_args = load_rqvae(args.ckpt, args.device)
    embeddings_np = np.load(args.emb)
    print(f"  embedding shape: {embeddings_np.shape}")
    print(f"  num codebook layers: {len(model_args.num_emb_list)}")

    residuals, _ = extract_per_layer_residuals(
        model, embeddings_np, args.device, batch_size=2048
    )
    num_layers = len(residuals)
    print(f"  residual shapes: {[tuple(r.shape) for r in residuals]}")

    # =====================================================================
    # 4 层 × 5 κ = 20 个 distortion 值
    # =====================================================================
    print()
    print("=" * 90)
    print("Per-layer × κ 矩阵 (normalized Kruskal stress-1, Phase 2 Riemannian GD MDS)")
    print("=" * 90)

    all_distortions: Dict[int, Dict[float, float]] = {}  # layer -> {kappa -> stress}
    all_elapsed: Dict[int, Dict[float, float]] = {}       # layer -> {kappa -> seconds}

    for i, r in enumerate(residuals):
        r_np = r.detach().cpu().numpy()
        label = "raw_encoded" if i == 0 else f"residual_after_layer_{i-1}"
        kappas = LAYER_KAPPAS[i]

        print(f"\n[L{i}] {label}, shape={tuple(r.shape)}, norm_mean={float(r.norm(dim=-1).mean()):.4f}")
        print(f"  {'κ':>8} | {'stress':>10} | {'elapsed (s)':>12}")
        print(f"  {'-'*8}-+-{'-'*10}-+-{'-'*12}")

        layer_dist: Dict[float, float] = {}
        layer_time: Dict[float, float] = {}
        for k in kappas:
            t0 = time.time()
            stress = compute_true_distortion(
                r_np, kappa=k, n_subset=args.n_subset,
                seed=args.seed, n_iter=args.n_iter, lr=args.lr,
                d=args.d, device=args.device,
            )
            elapsed = time.time() - t0
            print(f"  {k:>+8.2f} | {stress:>10.4f} | {elapsed:>12.1f}")
            layer_dist[k] = stress
            layer_time[k] = elapsed

        all_distortions[i] = layer_dist
        all_elapsed[i] = layer_time

    # =====================================================================
    # C 组 (per-layer best κ)
    # =====================================================================
    print()
    print("=" * 90)
    print("C 组 (per-layer best κ, 最低 stress = 拟合最佳)")
    print("=" * 90)

    c_group: Dict[int, Dict] = {}
    for i in range(num_layers):
        best_k, best_stress = min(all_distortions[i].items(), key=lambda x: x[1])
        label = "raw_encoded" if i == 0 else f"residual_after_layer_{i-1}"
        c_group[i] = {'kappa': best_k, 'stress': best_stress, 'label': label}
        print(f"  L{i} ({label:<32}): κ* = {best_k:>+.2f}, stress = {best_stress:.4f}")

    # =====================================================================
    # B 组 (global κ by sum across 4 layers)
    # =====================================================================
    print()
    print("=" * 90)
    print("B 组 (global κ, 最低 sum of stresses across 4 layers)")
    print("=" * 90)

    all_kappas = sorted(set(KAPPA_GRID_L0 + KAPPA_GRID_DEEP))
    b_group: Dict[float, float] = {}
    for k in all_kappas:
        total = sum(
            all_distortions[i].get(k, float('inf'))
            for i in range(num_layers)
        )
        b_group[k] = total

    best_global_k = min(b_group, key=b_group.get)
    print(f"  {'κ':>8} | {'sum_stress':>12}")
    print(f"  {'-'*8}-+-{'-'*12}")
    for k in sorted(b_group):
        marker = " ← global κ*" if k == best_global_k else ""
        print(f"  {k:>+8.2f} | {b_group[k]:>12.4f}{marker}")

    # =====================================================================
    # 对比 Stage 1 / Stage 1b (trivial κ=0)
    # =====================================================================
    print()
    print("=" * 90)
    print("对比 Stage 1 (CV-ratio, κ=0 trivial=0) vs Phase 2 (RGD MDS, κ=0 > 0)")
    print("=" * 90)

    print(f"  κ=0 stress per layer (本脚本 Phase 2 metric):")
    for i in range(num_layers):
        s0 = all_distortions[i][0.0]
        print(f"    L{i}: stress@κ=0 = {s0:.4f}  (Phase 2 metric, 非 trivial)")

    # κ=0 是不是真正的最佳? 对比 B 组
    is_zero_best_global = (best_global_k == 0.0)
    zero_in_c_group = sum(1 for i in range(num_layers) if c_group[i]['kappa'] == 0.0)
    print()
    print(f"  κ=0 在 B 组 global 是不是最佳?  {'是 (trivial 仍是 trivial)' if is_zero_best_global else '否 (真正的最佳 κ 不是 0)'}")
    print(f"  κ=0 在 C 组 per-layer 出现次数: {zero_in_c_group}/{num_layers}")

    # =====================================================================
    # 输出 JSON
    # =====================================================================
    output = {
        'task': 'task80_stage1c_kappa_distortion_v2',
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'method': 'Phase 2 Riemannian Gradient Descent MDS (Kruskal stress-1)',
        'method_references': ['Sarkar 2011', 'Nickel & Kiela 2017', 'Gu 2019'],
        'config': {
            'ckpt': args.ckpt,
            'emb': args.emb,
            'n_subset': args.n_subset,
            'n_iter': args.n_iter,
            'lr': args.lr,
            'd': args.d,
            'seed': args.seed,
            'device': args.device,
        },
        'kappa_grids': {
            'L0': KAPPA_GRID_L0,
            'L1/L2/L3': KAPPA_GRID_DEEP,
        },
        'num_layers': num_layers,
        # 4 层 × 5 κ 矩阵
        'stress_matrix': {
            f"L{i}": {
                f"k{k:>+.2f}": all_distortions[i][k] for k in sorted(all_distortions[i])
            }
            for i in range(num_layers)
        },
        'elapsed_seconds': {
            f"L{i}": {
                f"k{k:>+.2f}": all_elapsed[i][k] for k in sorted(all_elapsed[i])
            }
            for i in range(num_layers)
        },
        # C 组 per-layer best
        'C_group_per_layer_best': {
            f"L{i}": {
                'kappa': v['kappa'],
                'stress': v['stress'],
                'label': v['label'],
            }
            for i, v in c_group.items()
        },
        # B 组 global best
        'B_group_global_best': {
            'kappa': best_global_k,
            'sum_stress': b_group[best_global_k],
            'all_kappa_sum': {f"k{k:>+.2f}": v for k, v in sorted(b_group.items())},
        },
        # 跟 Stage 1 / Stage 1b 对比
        'comparison_to_stage1': {
            'note': (
                'Stage 1 用 CV-of-ratio, κ=0 → CV=0 (trivial 数学必然). '
                '本脚本 Phase 2 metric 用 normalized Kruskal stress, κ=0 也 > 0, '
                '因 Riemannian GD 在 κ=0 时退化为 Euclidean MDS, 而欧氏 MDS 对 '
                '高维点云也有 stress (只有完美 1D/2D 等特殊情况下 stress = 0).'
            ),
            'is_zero_best_global': is_zero_best_global,
            'zero_count_in_C_group': zero_in_c_group,
            'zero_count_in_C_group_max': num_layers,
        },
    }

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    # 替换 Python 的 Infinity (json.dump 默认允许但不严格) 为 None, 避免下游解析失败
    output_json = json.dumps(output, indent=2).replace('Infinity', 'null').replace('NaN', 'null')
    with open(args.output, 'w') as f:
        f.write(output_json)
    print(f"\n[Stage 1c v2] Result saved to {args.output}")


if __name__ == '__main__':
    main()