#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Task #82 Step B — Phonism 真实数据网格加密 (Grid Refinement).

承接 Task #81 v3 + Task #82 Step A:
- 当前 Stage 1c grid = {0, -0.3, -0.5, -1.0, -1.5, -2.0}
- 0 和 -0.3 之间是**空的** — 如果真实最优点在 -0.1~-0.25, 网格根本测不到
- Step A 显示 κ=-0.05 已经在 32D 数据点显著优于 κ=0 (stress 0.019 vs 0.067, 3.5-4×)
- 必须加密网格排除"phonism 真实 κ 落在网格空隙"的可能性

本脚本:
  - 复用 task80_stage1c_kappa_distortion_v2.py 的 metric (RGD MDS + Kruskal stress-1)
  - 但加密 κ grid: 在 0 和 -0.5 之间补 {-0.05, -0.1, -0.15, -0.2, -0.25}
  - 完整 grid: {0, -0.05, -0.1, -0.15, -0.2, -0.25, -0.3, -0.5, -1.0, -1.5, -2.0}
  - 跑 4 层 (L0/L1/L2/L3) × 11 κ = 44 个 stress 值
  - 输出 best κ per layer + 综合判定
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Dict

import numpy as np
import torch

sys.path.insert(0, '/home/wlia0047/ar57/wenyu/GeneRec/scripts')
from task80_stage1c_true_distortion import compute_true_distortion  # noqa: E402
from task92_rqvae_delta_hyperbolicity import load_rqvae, extract_per_layer_residuals  # noqa: E402


# 加密 κ grid (在 0 和 -0.3 之间补 5 个细粒度点)
KAPPA_GRID_REFINED = [
    0.0,    # 欧氏
    -0.05,  # 弱双曲 1
    -0.10,  # 弱双曲 2
    -0.15,  # 弱双曲 3
    -0.20,  # 弱双曲 4
    -0.25,  # 弱双曲 5
    -0.30,  # 原 grid 第二点
    -0.50,  # 原 grid 第三点
    -1.00,
    -1.50,
    -2.00,
]

# 四层都用同一加密 grid (v2 之前 L0 和 L1/L2/L3 不同)
LAYER_KAPPAS = {i: KAPPA_GRID_REFINED for i in range(4)}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt', required=True, help='Phonism RQ-VAE checkpoint')
    p.add_argument('--emb', required=True, help='sentence-t5 768d embedding .npy')
    p.add_argument('--output', required=True, help='Output JSON path')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--n_subset', type=int, default=300)
    p.add_argument('--n_iter', type=int, default=200)
    p.add_argument('--lr', type=float, default=0.005)
    p.add_argument('--d', type=int, default=32)
    p.add_argument('--device', default='cuda:1',
                   help='cuda:1 是空闲 GPU (R7: 不抢已占卡)')
    return p.parse_args()


def main() -> None:
    args = parse_args()
    print(f"[Task #82 Step B] Phonism 真实数据网格加密 (11 κ × 4 层)")
    print(f"  ckpt:           {args.ckpt}")
    print(f"  emb:            {args.emb}")
    print(f"  κ grid:         {KAPPA_GRID_REFINED}")
    print(f"  device:         {args.device}")
    print(f"  n_subset:       {args.n_subset}, n_iter: {args.n_iter}")

    # 加载 phonism RQ-VAE + 抽取 per-layer residuals
    model, model_args = load_rqvae(args.ckpt, args.device)
    embeddings_np = np.load(args.emb)
    print(f"  embedding shape: {embeddings_np.shape}")
    print(f"  num codebook layers: {len(model_args.num_emb_list)}")

    residuals, _ = extract_per_layer_residuals(
        model, embeddings_np, args.device, batch_size=2048
    )
    num_layers = len(residuals)
    print(f"  residual shapes: {[tuple(r.shape) for r in residuals]}")

    # 跑 4 层 × 11 κ = 44 个 stress 值
    print()
    print("=" * 90)
    print("Per-layer × κ 矩阵 (加密 grid, 11 κ × 4 层, Phase 2 RGD MDS Kruskal stress-1)")
    print("=" * 90)

    all_distortions: Dict[int, Dict[float, float]] = {}
    all_elapsed: Dict[int, Dict[float, float]] = {}

    for i, r in enumerate(residuals):
        r_np = r.detach().cpu().numpy()
        label = "raw_encoded" if i == 0 else f"residual_after_layer_{i-1}"
        kappas = LAYER_KAPPAS[i]

        print(f"\n[L{i}] {label}, shape={tuple(r.shape)}, "
              f"norm_mean={float(r.norm(dim=-1).mean()):.4f}")
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

    # C 组 (per-layer best κ)
    print()
    print("=" * 90)
    print("C 组 (per-layer best κ, 加密 grid)")
    print("=" * 90)

    c_group: Dict[int, Dict] = {}
    for i in range(num_layers):
        best_k, best_stress = min(all_distortions[i].items(), key=lambda x: x[1])
        label = "raw_encoded" if i == 0 else f"residual_after_layer_{i-1}"
        c_group[i] = {'kappa': best_k, 'stress': best_stress, 'label': label}
        marker = " ⭐ 弱信号命中" if (-0.30 < best_k < 0) else ""
        print(f"  L{i} ({label:<32}): κ* = {best_k:>+.3f}, stress = {best_stress:.4f}{marker}")

    # 关键判定: best κ 落入弱信号区间 (-0.05 ~ -0.25)?
    print()
    print("=" * 90)
    print("判定: phonism 真实数据 best κ 是否落入弱信号区间 [-0.05, -0.25]?")
    print("=" * 90)

    weak_signal_layers = []
    for i in range(num_layers):
        best_k = c_group[i]['kappa']
        is_weak_signal = (-0.25 <= best_k <= -0.05)
        is_zero = abs(best_k) < 0.01
        is_strong = best_k <= -0.3
        if is_weak_signal:
            verdict_label = "⚠️ 弱双曲信号命中"
        elif is_zero:
            verdict_label = "✅ 欧氏最优"
        else:
            verdict_label = "🔵 强双曲信号"
        print(f"  L{i}: best κ = {best_k:>+.3f}  → {verdict_label}")
        if is_weak_signal:
            weak_signal_layers.append(i)

    if len(weak_signal_layers) >= 3:
        verdict = "B2 (phonism 是弱双曲 — 真实 κ ∈ [-0.05, -0.25])"
        decision_note = (
            f"≥ 3 层 best κ 落入弱信号区间 ({weak_signal_layers}). "
            f"phonism 残差几何**不是欧氏**, 而是**弱双曲 (κ ≈ -0.05 ~ -0.25)**. "
            f"之前因网格跳跃 (-0.3 起步) 漏掉了真实最优点, 误判为欧氏."
        )
    elif len(weak_signal_layers) >= 1:
        verdict = "B2-partial (部分层弱双曲)"
        decision_note = (
            f"{len(weak_signal_layers)}/4 层 best κ 落入弱信号区间 "
            f"({weak_signal_layers}). 部分层是弱双曲, 不是欧氏."
        )
    elif all(abs(c_group[i]['kappa']) < 0.01 for i in range(num_layers)):
        verdict = "B1 (κ=0 维持 — 加最密的网格后仍是欧氏最优)"
        decision_note = (
            "4 层 best κ 仍是 0, 加最密的 -0.05/-0.1/-0.15/-0.2/-0.25 都不优. "
            "phonism 真实几何匹配欧氏的结论站得住 (排除'网格空隙'解释)."
        )
    else:
        verdict = "B3 (best κ 仍是 -0.5 或更深)"
        decision_note = (
            f"加密网格不影响结果, best κ ≤ -0.5 ({c_group}). "
            f"phonism 残差几何是**强双曲**而非欧氏. (与原 v2 结论一致)"
        )

    print(f"\n  判定 = {verdict}")
    print(f"  决策 = {decision_note}")

    # 输出 JSON
    output = {
        'task': 'task82_step_b_phonism_grid_refinement',
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
        'kappa_grid_refined': KAPPA_GRID_REFINED,
        'num_layers': num_layers,
        'stress_matrix': {
            f"L{i}": {f"k{k:>+.2f}": all_distortions[i][k] for k in sorted(all_distortions[i])}
            for i in range(num_layers)
        },
        'C_group_per_layer_best': {
            f"L{i}": {'kappa': v['kappa'], 'stress': v['stress'], 'label': v['label']}
            for i, v in c_group.items()
        },
        'verdict': verdict,
        'decision_note': decision_note,
        'weak_signal_layers': weak_signal_layers,
        'comparison_to_v2': {
            'note': (
                '原 v2 grid = {0, -0.3, -0.5, -1.0, -1.5, -2.0}. '
                '加密 grid = {0, -0.05, -0.1, -0.15, -0.2, -0.25, -0.3, -0.5, -1.0, -1.5, -2.0}. '
                '在 0 和 -0.3 之间补 5 个细粒度点 (v2 完全跳过这段).'
            ),
        },
    }

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    output_json = json.dumps(output, indent=2).replace('Infinity', 'null').replace('NaN', 'null')
    with open(args.output, 'w') as f:
        f.write(output_json)
    print(f"\n[Task #82 Step B] Result saved to {args.output}")


if __name__ == '__main__':
    main()