#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Task #81 v3 — Stage 1c Metric 阳性对照 v3 (target = 真实 Poincaré distance @ κ=-1).

v1/v2 失败根因:
  - v1 (2D + Euclidean pairwise): Euclidean MDS 完美拟合 (stress ≈ 0), κ=0 trivial
  - v2 (32D + graph shortest path): scale 不匹配, κ=0 仍最优 (0.23-0.35 vs κ=-1 7-14)
  - 共同根因: target 不是数据点之间的"真实双曲距离"

v3 设计 (最严格 sanity check):
  - target = 数据点之间的 Poincaré distance @ κ=-1 (i.e., 数据点的"真实"双曲距离)
  - 数据点本身嵌入到 Poincaré ball κ=-1
  - 对每个 κ 候选, metric 让 hyperbolic_dist 拟合这个 target
  - 预期: κ=-1 应该 stress 最低 (因为 target 就是 κ=-1 hyperbolic distance)

如果 v3 失败 (κ=-1 不最优), 说明 metric 实现本身有 bug (Riemannian GD 公式 / Poincaré distance 公式错了).
如果 v3 成功 (κ=-1 最优), 说明 metric 实现正确, v1/v2 失败是因为 metric 设计选择 (target 选取) 的偏差.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Dict, List, Tuple

import numpy as np
import torch

sys.path.insert(0, '/home/wlia0047/ar57/wenyu/GeneRec/scripts')
from task80_stage1c_true_distortion import (
    riemannian_grad_descent_mds,
    stereographic_project,
    project_to_poincare_ball,
    hyperbolic_distance_pairwise,
)  # noqa: E402

from task81_positive_control_v2 import (
    mobius_exp_map,
    project_to_ball_np,
    generate_hyperbolic_tree_32d,
    KAPPA_CANDIDATES,
)  # noqa: E402


def compute_poincare_distance_matrix(coords: np.ndarray, kappa: float) -> np.ndarray:
    """计算数据点之间的 Poincaré distance matrix (κ 双曲距离).

    Returns:
        dist: (N, N) Poincaré distance
    """
    coords_t = torch.from_numpy(coords.astype(np.float32))
    dist_t = hyperbolic_distance_pairwise(coords_t, coords_t, kappa)
    return dist_t.numpy()


def compute_distortion_with_poincare_target(
    coords: np.ndarray,
    target_dist: np.ndarray,
    kappa: float,
    n_iter: int = 200,
    lr: float = 0.005,
    d: int = 32,
    device: str = 'cpu',
    seed: int = 42,
    grad_clip: float = 1.0,
) -> float:
    """Stage 1c metric, target = Poincaré pairwise distance (κ=-1 时的真实距离)."""
    n = coords.shape[0]
    d_use = min(d, coords.shape[1])

    target_t = torch.from_numpy(target_dist.astype(np.float32)).to(device=device)

    # init
    init_x = stereographic_project(torch.from_numpy(coords).to(device), kappa)
    perturb_rng = torch.Generator(device='cpu').manual_seed(seed + 7)
    perturb = torch.randn(init_x.shape, generator=perturb_rng) * 0.1
    init_x = init_x + perturb.to(device=device)
    if kappa < 0:
        init_x = project_to_poincare_ball(init_x, kappa)

    _, final_stress = riemannian_grad_descent_mds(
        target_d_eucl=target_t,
        kappa=kappa,
        n_iter=n_iter,
        lr=lr,
        init_x=init_x,
        n=n,
        d=d_use,
        seed=seed,
        verbose=False,
        grad_clip=grad_clip,
    )

    return final_stress


def run_metric(coords: np.ndarray, target_dist: np.ndarray, label: str,
               device: str = 'cpu') -> Dict:
    """对单棵树跑 7 κ 候选."""
    print(f"\n[{label}] Running metric on {coords.shape[0]} points (dim={coords.shape[1]})")
    print(f"  target = Poincaré distance @ κ=-1 "
          f"(max={target_dist.max():.2f}, mean={target_dist[target_dist > 0].mean():.2f})")
    print(f"  {'κ':>8} | {'stress':>10} | {'elapsed (s)':>12}")
    print(f"  {'-'*8}-+-{'-'*10}-+-{'-'*12}")

    stresses: Dict[float, float] = {}
    for k in KAPPA_CANDIDATES:
        t0 = time.time()
        try:
            stress = compute_distortion_with_poincare_target(
                coords.astype(np.float32),
                target_dist=target_dist,
                kappa=k,
                n_iter=200,
                lr=0.005,
                d=coords.shape[1],
                device=device,
            )
            if not np.isfinite(stress):
                stress = float('inf')
        except Exception as e:
            print(f"  κ={k:+.2f}: ERROR {e}")
            stress = float('inf')
        elapsed = time.time() - t0
        print(f"  {k:>+8.2f} | {stress:>10.4f} | {elapsed:>12.1f}")
        stresses[k] = stress

    finite = {k: v for k, v in stresses.items() if np.isfinite(v)}
    if finite:
        best_k, best_s = min(finite.items(), key=lambda x: x[1])
    else:
        best_k, best_s = 0.0, float('inf')

    return {
        'label': label,
        'num_points': coords.shape[0],
        'kappa_generated': -1.0,
        'kappa_target': -1.0,
        'stress_per_kappa': stresses,
        'best_kappa': best_k,
        'best_stress': best_s,
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--output', required=True)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--device', default='cpu')
    args = p.parse_args()

    print("=" * 90)
    print("[Task #81 v3] Stage 1c Metric Positive Control — target = Poincaré dist @ κ=-1")
    print("=" * 90)

    # 1. 生成 32D 合成树 (norm 较小, 给 RGD 优化空间)
    trees_spec = [
        ('Tree-A', {'n_levels': 4, 'branching': 3, 'kappa': -1.0, 'noise_sigma': 0.03,
                    'step_per_level': [0.0, 0.6, 0.45, 0.35]}),  # norm max ~0.7
        ('Tree-B', {'n_levels': 5, 'branching': 2, 'kappa': -1.0, 'noise_sigma': 0.03,
                    'step_per_level': [0.0, 0.5, 0.4, 0.3, 0.25]}),  # norm max ~0.6
        ('Tree-C', {'n_levels': 4, 'branching': 4, 'kappa': -1.0, 'noise_sigma': 0.03,
                    'step_per_level': [0.0, 0.55, 0.4, 0.3]}),  # norm max ~0.65
    ]

    all_results = []
    for label, spec in trees_spec:
        print(f"\n[{label}] Generating 32D tree with spec {spec}...")
        t0 = time.time()
        coords, edges = generate_hyperbolic_tree_32d(seed=args.seed, dim=32, **spec)
        norms = np.linalg.norm(coords, axis=1)
        print(f"  Generated {coords.shape[0]} nodes in {time.time()-t0:.1f}s")
        print(f"  norm max = {norms.max():.3f}, norm mean = {norms.mean():.3f}")

        # target = 数据点之间的 Poincaré distance @ κ=-1
        t0 = time.time()
        target_dist = compute_poincare_distance_matrix(coords, kappa=-1.0)
        print(f"  Poincaré distance computed in {time.time()-t0:.1f}s "
              f"(max={target_dist.max():.2f}, mean={target_dist[target_dist > 0].mean():.2f})")

        result = run_metric(coords, target_dist, label, device=args.device)
        result['spec'] = spec
        result['norm_max'] = float(norms.max())
        result['norm_mean'] = float(norms.mean())
        result['target_max_dist'] = float(target_dist.max())
        result['target_mean_dist'] = float(target_dist[target_dist > 0].mean())
        all_results.append(result)

    # 2. 判定
    print()
    print("=" * 90)
    print("判定: target = Poincaré distance @ κ=-1, 能否识别 κ=-1?")
    print("=" * 90)

    for r in all_results:
        marker = "✓" if r['best_kappa'] <= -0.5 else "✗"
        print(f"  [{r['label']}] best κ = {r['best_kappa']:>+.2f}, "
              f"stress = {r['best_stress']:.4f}  {marker} "
              f"{'识别负曲率' if r['best_kappa'] <= -0.5 else '未识别负曲率'}")

    n_negative = sum(1 for r in all_results if r['best_kappa'] <= -0.5)
    n_zero = sum(1 for r in all_results if abs(r['best_kappa']) < 0.01)

    if n_negative >= 2:
        verdict = "A (sanity check 通过 — metric 可信)"
        decision_note = (
            f"≥ 2/3 合成树识别出 κ ≤ -0.5. 当 target = 真实 κ=-1 Poincaré distance 时, "
            f"metric 能正确识别 κ=-1. 说明 metric 实现正确, v1/v2 失败是因为 target 选择."
        )
    elif n_negative == 1:
        verdict = "B (sanity check 部分通过)"
        decision_note = "仅 1/3 树识别负曲率, metric 部分可信, 需诊断."
    elif n_zero == len(all_results):
        verdict = "C (sanity check 失败 — metric 实现有 bug)"
        decision_note = (
            f"3/3 树 best κ = 0. 即使 target 是数据点的真实 κ=-1 Poincaré distance, "
            f"metric 仍选 κ=0. 说明 Riemannian GD 或 Poincaré distance 公式实现有 bug."
        )
    else:
        verdict = "D (识别更负)"
        decision_note = "metric 偏向过负曲率, 也有 bias."

    print(f"\n  判定 = {verdict}")
    print(f"  决策 = {decision_note}")

    output = {
        'task': 'task81_positive_control_v3',
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'method': 'Stage 1c RGD MDS, target=Poincaré distance @ κ=-1 (true hyperbolic dist)',
        'method_references': ['Sarkar 2011', 'Nickel & Kiela 2017', 'Gu 2019'],
        'kappa_candidates': KAPPA_CANDIDATES,
        'trees': all_results,
        'verdict': verdict,
        'decision_note': decision_note,
        'summary': {
            'n_negative_recognized': n_negative,
            'n_zero_recognized': n_zero,
            'n_total': len(all_results),
        },
        'v1_v2_failure_summary': (
            'v1 (2D + Euclidean pairwise): κ=0 trivial stress ≈ 0. '
            'v2 (32D + graph shortest path): κ=0 stress 0.23-0.35 vs κ=-1 7-14. '
            '共同根因: target 不是数据点的真实双曲距离. v3 用 Poincaré dist @ κ=-1 作为 target.'
        ),
    }

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    output_json = json.dumps(output, indent=2, default=lambda o: float(o) if isinstance(o, np.floating) else o)
    output_json = output_json.replace('Infinity', 'null').replace('NaN', 'null')
    with open(args.output, 'w') as f:
        f.write(output_json)
    print(f"\n[Task #81 v3] Result saved to {args.output}")


if __name__ == '__main__':
    main()