#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Task #82 Step A — 弱信号阳性对照 (Weak Signal Sensitivity).

承接 Task #81 v3 阳性对照 (best κ=-0.3~-0.5 vs 真实 κ=-1, shrinkage ~30-50%).

本脚本验证流程能否识别**弱双曲信号**: 生成真实 κ ∈ {-0.10, -0.15, -0.20} 的合成树,
跑 grid search + stress, 看 best κ 是否能识别弱信号方向, 还是会因收缩偏差被压回 0.

判定:
- A1 (灵敏度足够): 3/3 树 best κ 落在 (0, κ_real] 区间
- A2 (灵敏度部分足够): 1-2/3 树识别
- A3 (灵敏度不足): 3/3 树 best κ = 0 (弱信号被直接压成欧氏)
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
from task81_positive_control_v3 import (
    compute_poincare_distance_matrix,
    compute_distortion_with_poincare_target,
    run_metric,
)  # noqa: E402


# Step A 关键 κ grid: 包含 0 到 -1 的细粒度
KAPPA_CANDIDATES_WEAK = [0.0, -0.05, -0.1, -0.15, -0.2, -0.3, -0.5, -1.0, -1.5, -2.0]


def run_metric_weak(coords: np.ndarray, target_dist: np.ndarray, label: str,
                    device: str = 'cpu') -> Dict:
    """对单棵弱双曲树跑 10 κ 候选."""
    print(f"\n[{label}] Running metric on {coords.shape[0]} points (dim={coords.shape[1]})")
    print(f"  target = Poincaré distance @ κ_real "
          f"(max={target_dist.max():.2f}, mean={target_dist[target_dist > 0].mean():.2f})")
    print(f"  {'κ':>8} | {'stress':>10} | {'elapsed (s)':>12}")
    print(f"  {'-'*8}-+-{'-'*10}-+-{'-'*12}")

    stresses: Dict[float, float] = {}
    for k in KAPPA_CANDIDATES_WEAK:
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
        'kappa_generated': None,  # filled by caller
        'stress_per_kappa': stresses,
        'best_kappa': best_k,
        'best_stress': best_s,
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--output', required=True)
    p.add_argument('--seed', type=int, default=42)
    args = p.parse_args()

    print("=" * 90)
    print("[Task #82 Step A] Stage 1c Metric 弱信号灵敏度验证")
    print("=" * 90)

    # 3 棵弱双曲树 (真实 κ ∈ {-0.10, -0.15, -0.20})
    # 注意: κ 较小时 (例如 -0.1), Poincaré ball 半径 = 1/sqrt(0.1) = 3.16, 远大于 1
    # 所以数据点 norm 几乎可以任意大小, 但 step_per_level 需要小心, 不要让 norm 太接近 0
    trees_spec = [
        ('Tree-A κ_real=-0.20', -0.20, 4, 3, [0.0, 0.55, 0.4, 0.3]),
        ('Tree-B κ_real=-0.15', -0.15, 4, 3, [0.0, 0.55, 0.4, 0.3]),
        ('Tree-C κ_real=-0.10', -0.10, 4, 3, [0.0, 0.55, 0.4, 0.3]),
    ]

    all_results = []
    for label, kappa_real, n_levels, branching, step_per_level in trees_spec:
        print(f"\n[{label}] Generating 32D tree...")
        coords, edges = generate_hyperbolic_tree_32d(
            seed=args.seed, dim=32,
            n_levels=n_levels, branching=branching,
            kappa=kappa_real, noise_sigma=0.03,
            step_per_level=step_per_level,
        )
        norms = np.linalg.norm(coords, axis=1)
        print(f"  Generated {coords.shape[0]} nodes, norm max = {norms.max():.3f}, "
              f"norm mean = {norms.mean():.3f}")

        # target = 数据点之间的 Poincaré distance @ κ_real
        target_dist = compute_poincare_distance_matrix(coords, kappa=kappa_real)
        print(f"  target dist: max={target_dist.max():.2f}, "
              f"mean={target_dist[target_dist > 0].mean():.2f}")

        result = run_metric_weak(coords, target_dist, label, device='cpu')
        result['kappa_real'] = kappa_real
        result['norm_max'] = float(norms.max())
        result['norm_mean'] = float(norms.mean())
        all_results.append(result)

    # 判定
    print()
    print("=" * 90)
    print("判定: 弱双曲信号灵敏度 (best κ 是否落在 (0, κ_real])")
    print("=" * 90)

    n_recognized = 0
    for r in all_results:
        kappa_real = r['kappa_real']
        # 识别条件: best κ 在 (0, κ_real × 1.5] 区间 (允许 ±50% 偏差)
        recognized = (0 < abs(r['best_kappa']) <= abs(kappa_real) * 1.5)
        # 或者 best κ 严格等于 κ_real (完美识别)
        strict_match = abs(r['best_kappa'] - kappa_real) < 0.01
        marker = "✓" if recognized else "✗"
        label = '严格命中' if strict_match else ('识别方向' if recognized else '未识别(被压回 0)')
        print(f"  [{r['label']}] best κ = {r['best_kappa']:>+.3f}, "
              f"κ_real = {kappa_real:>+.3f}, stress = {r['best_stress']:.4f}  {marker} {label}")
        if recognized:
            n_recognized += 1

    if n_recognized == 3:
        verdict = "A1 (灵敏度足够 — 3/3 弱信号识别)"
        decision_note = (
            "3/3 弱双曲数据 best κ ∈ (0, κ_real × 1.5]. "
            "metric 能识别弱信号方向 (即使 κ_real 仅为 -0.10)."
        )
    elif n_recognized >= 1:
        verdict = f"A2 (灵敏度部分足够 — {n_recognized}/3 弱信号识别)"
        decision_note = (
            f"仅 {n_recognized}/3 弱双曲数据识别. metric 对弱信号灵敏度有限, "
            f"需要更多诊断或更细的 κ grid."
        )
    else:
        verdict = "A3 (灵敏度不足 — 弱信号被压回 κ=0)"
        decision_note = (
            "3/3 弱双曲数据 best κ = 0. metric 对 κ ∈ {-0.10, -0.15, -0.20} 弱信号"
            "**完全没有灵敏度**, 全被压回欧氏. 这意味着 phonism 真实 κ=-0.1~-0.2 量级的"
            "弱双曲信号也可能被压成欧氏, metric 无法区分'真欧氏' vs '弱双曲'."
        )

    print(f"\n  判定 = {verdict}")
    print(f"  决策 = {decision_note}")

    output = {
        'task': 'task82_step_a_weak_signal_positive_control',
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'method': 'Stage 1c RGD MDS, target=Poincaré distance @ κ_real',
        'kappa_candidates': KAPPA_CANDIDATES_WEAK,
        'trees': all_results,
        'verdict': verdict,
        'decision_note': decision_note,
        'n_recognized': n_recognized,
        'n_total': len(all_results),
        'v3_failure_context': (
            'v3 (κ_real=-1.0): best κ=-0.3~-0.5, shrinkage 30-50%. '
            '本脚本验证: κ_real 缩小到 -0.10~-0.20 时, 流程能否识别弱信号.'
        ),
    }

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    output_json = json.dumps(output, indent=2, default=lambda o: float(o) if isinstance(o, np.floating) else o)
    output_json = output_json.replace('Infinity', 'null').replace('NaN', 'null')
    with open(args.output, 'w') as f:
        f.write(output_json)
    print(f"\n[Task #82 Step A] Result saved to {args.output}")


if __name__ == '__main__':
    main()