#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Task #81 v2 — Stage 1c Metric 阳性对照 v2 (32D 嵌入 + graph shortest path target).

v1 失败根因:
  - 用了 euclidean pairwise distance 作为 target
  - 2D Poincaré ball 嵌入 + 数据点 norm 小 → euclidean distance ≈ Poincaré distance
  - κ=0 Euclidean MDS 完美拟合 (stress ≈ 0), κ<0 受 Poincaré ball 边界约束 stress > 0

v2 修复:
  - target = graph shortest path distance (而非 euclidean pairwise)
    这是 Sarkar 2011 / Nickel & Kiela 2017 论文里的标准 metric
  - 32D 嵌入 (与 phonism RQ-VAE 残差 dim 一致)
  - 嵌入维度足够高, 让 κ=0 Euclidean MDS 不能完美拟合 hyperbolic pairwise

设计:
  1. 生成 500 节点的树 (4 层 branching=5 + 5 层 branching=4)
  2. 在 32D Poincaré ball κ=-1 嵌入 (Möbius exp map, step 大到 norm 接近边界)
  3. target = graph shortest path distance (Floyd-Warshall, 树边 = 1)
  4. 对每个 κ 候选, Riemannian GD 在 Poincaré ball 拟合 hyperbolic_dist → target
  5. 判定 best κ 是否 ≈ -1
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

# 复用 Stage 1c 的 Riemannian GD MDS 实现
sys.path.insert(0, '/home/wlia0047/ar57/wenyu/GeneRec/scripts')
from task80_stage1c_true_distortion import (
    riemannian_grad_descent_mds,
    stereographic_project,
    project_to_poincare_ball,
)  # noqa: E402


# ==============================================================================
# 32D 双曲树嵌入 (Poincaré ball, κ=-1)
# ==============================================================================

def mobius_exp_map(x: np.ndarray, v: np.ndarray, kappa: float) -> np.ndarray:
    """Möbius exponential map at x for tangent vector v (Poincaré ball).

    Ganea 2018 Eq. 7.
    """
    if kappa >= 0:
        return x + v
    sqrt_kappa = np.sqrt(-kappa)
    x_norm_sq = np.dot(x, x)
    lambda_x = 2.0 / (1.0 - kappa * x_norm_sq)
    v_norm = np.linalg.norm(v)
    if v_norm < 1e-12:
        return x
    t = np.tanh(sqrt_kappa * lambda_x * v_norm / 2.0)
    return x + (t / (sqrt_kappa * v_norm)) * v


def project_to_ball_np(x: np.ndarray, kappa: float, eps: float = 0.05) -> np.ndarray:
    """numpy 版的 project_to_poincare_ball: 约束点 ||x||² < -1/κ - eps."""
    if kappa >= 0:
        return x
    max_norm = np.sqrt(-1.0 / kappa) - eps
    norm = np.linalg.norm(x)
    if norm >= max_norm:
        if norm < 1e-12:
            return x
        return x * (max_norm / norm)
    return x


def generate_hyperbolic_tree_32d(
    n_levels: int,
    branching: int,
    dim: int = 32,
    kappa: float = -1.0,
    step_per_level: List[float] | None = None,
    noise_sigma: float = 0.05,
    seed: int = 42,
) -> Tuple[np.ndarray, List[Tuple[int, int]]]:
    """在 32D Poincaré ball (κ=-1) 嵌入一棵树.

    Args:
        n_levels: 层数 (含根)
        branching: 每节点分支数
        dim: 嵌入维度 (默认 32)
        kappa: 双曲曲率
        step_per_level: 每层 step 大小
        noise_sigma: 切空间噪声
        seed: 随机种子

    Returns:
        coords: (N, dim) Poincaré ball 点
        edges: list of (parent_idx, child_idx)
    """
    rng = np.random.default_rng(seed)

    if step_per_level is None:
        # 让数据点 norm 接近边界 (|x| ≈ 0.8), 这样 Poincaré ball 几何特征显著
        # step 大: 0, 2.0, 1.5, 1.2, 1.0, 0.8 (与 dim=32 匹配, step 大让 norm 快速接近边界)
        step_per_level = [0.0] + [2.0 / (1.0 + 0.3 * l) for l in range(1, n_levels)]

    coords: List[np.ndarray] = [np.zeros(dim)]  # root at origin
    edges: List[Tuple[int, int]] = []

    for level in range(1, n_levels):
        n_at_prev_level = branching ** (level - 1)
        parent_start = len(coords) - n_at_prev_level

        for parent_idx in range(parent_start, len(coords)):
            parent = coords[parent_idx]
            r = step_per_level[level]

            for _ in range(branching):
                # 随机切空间方向 (32D 随机单位向量)
                direction = rng.standard_normal(dim)
                direction = direction / np.linalg.norm(direction)
                v = direction * r

                # Möbius exp map
                child = mobius_exp_map(parent, v, kappa)
                child = project_to_ball_np(child, kappa, eps=0.05)

                # 加切空间噪声
                if noise_sigma > 0:
                    noise = rng.normal(0, noise_sigma, size=dim)
                    inner_cn = np.dot(child, noise)
                    proj_noise = noise + kappa * inner_cn * child
                    proj_noise_norm = np.linalg.norm(proj_noise)
                    if proj_noise_norm > 1e-12:
                        child = mobius_exp_map(child, proj_noise, kappa)
                        child = project_to_ball_np(child, kappa, eps=0.05)

                coords.append(child)
                edges.append((parent_idx, len(coords) - 1))

    return np.array(coords), edges


# ==============================================================================
# Graph shortest path (Floyd-Warshall / BFS for tree)
# ==============================================================================

def compute_graph_shortest_path(n: int, edges: List[Tuple[int, int]]) -> np.ndarray:
    """计算树的 graph shortest path distance matrix.

    树结构, 用 BFS 算所有对最短路径 (树边权重 = 1).
    """
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)

    dist = np.full((n, n), np.inf, dtype=np.float32)
    for start in range(n):
        # BFS
        dist[start, start] = 0
        visited = np.zeros(n, dtype=bool)
        visited[start] = True
        queue = [start]
        depth = 0
        frontier = [start]
        while frontier:
            next_frontier = []
            for u in frontier:
                for v in adj[u]:
                    if not visited[v]:
                        visited[v] = True
                        dist[start, v] = depth + 1
                        next_frontier.append(v)
            frontier = next_frontier
            depth += 1
    return dist


# ==============================================================================
# Stage 1c metric 但 target = graph shortest path
# ==============================================================================

def compute_distortion_with_graph_target(
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
    """Stage 1c metric 但 target = graph shortest path (而非 euclidean pairwise).

    Returns:
        stress: normalized Kruskal stress-1
    """
    n = coords.shape[0]
    d_use = min(d, coords.shape[1])

    # target = graph shortest path
    target_t = torch.from_numpy(target_dist.astype(np.float32)).to(device=device)

    # init: stereographic + 扰动
    init_x = stereographic_project(torch.from_numpy(coords).to(device), kappa)
    perturb_rng = torch.Generator(device='cpu').manual_seed(seed + 7)
    perturb = torch.randn(init_x.shape, generator=perturb_rng) * 0.3
    init_x = init_x + perturb.to(device=device)
    if kappa < 0:
        init_x = project_to_poincare_ball(init_x, kappa)

    _, final_stress = riemannian_grad_descent_mds(
        target_d_eucl=target_t,  # 这里 "eucl" 是名, 实际是 graph distance
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


KAPPA_CANDIDATES = [-2.0, -1.5, -1.0, -0.7, -0.5, -0.3, 0.0]


def run_metric(coords: np.ndarray, target_dist: np.ndarray, label: str,
               device: str = 'cpu') -> Dict:
    """对单棵树跑 7 κ 候选."""
    print(f"\n[{label}] Running metric on {coords.shape[0]} points (dim={coords.shape[1]})")
    print(f"  target = graph shortest path (max={np.nanmax(target_dist):.1f}, "
          f"mean={np.nanmean(target_dist[target_dist < np.inf]):.2f})")
    print(f"  {'κ':>8} | {'stress':>10} | {'elapsed (s)':>12}")
    print(f"  {'-'*8}-+-{'-'*10}-+-{'-'*12}")

    stresses: Dict[float, float] = {}
    for k in KAPPA_CANDIDATES:
        t0 = time.time()
        try:
            stress = compute_distortion_with_graph_target(
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
    print("[Task #81 v2] Stage 1c Metric Positive Control — 32D + graph shortest path target")
    print("=" * 90)

    # ============================================================
    # 1. 生成 32D 合成树
    # ============================================================
    # Tree-A: 4 层 branching=4 → 1+4+16+64 = 85 节点
    # Tree-B: 5 层 branching=3 → 1+3+9+27+81 = 121 节点
    # Tree-C: 4 层 branching=5 → 1+5+25+125 = 156 节点
    trees_spec = [
        ('Tree-A', {'n_levels': 4, 'branching': 4, 'kappa': -1.0, 'noise_sigma': 0.05}),
        ('Tree-B', {'n_levels': 5, 'branching': 3, 'kappa': -1.0, 'noise_sigma': 0.05}),
        ('Tree-C', {'n_levels': 4, 'branching': 5, 'kappa': -1.0, 'noise_sigma': 0.05}),
    ]

    all_results = []
    for label, spec in trees_spec:
        print(f"\n[{label}] Generating 32D tree with spec {spec}...")
        t0 = time.time()
        coords, edges = generate_hyperbolic_tree_32d(seed=args.seed, dim=32, **spec)
        norms = np.linalg.norm(coords, axis=1)
        print(f"  Generated {coords.shape[0]} nodes in {time.time()-t0:.1f}s")
        print(f"  norm max = {norms.max():.3f}, norm mean = {norms.mean():.3f}, "
              f"norm p95 = {np.percentile(norms, 95):.3f}")

        # Graph shortest path
        t0 = time.time()
        target_dist = compute_graph_shortest_path(coords.shape[0], edges)
        print(f"  graph shortest path computed in {time.time()-t0:.1f}s "
              f"(max={np.nanmax(target_dist):.1f})")

        result = run_metric(coords, target_dist, label, device=args.device)
        result['spec'] = spec
        result['norm_max'] = float(norms.max())
        result['norm_mean'] = float(norms.mean())
        result['target_max_dist'] = float(np.nanmax(target_dist))
        all_results.append(result)

    # ============================================================
    # 2. 判定
    # ============================================================
    print()
    print("=" * 90)
    print("判定: 32D + graph target 能否识别 κ ≈ -1?")
    print("=" * 90)

    for r in all_results:
        marker = "✓" if r['best_kappa'] <= -0.5 else "✗"
        print(f"  [{r['label']}] best κ = {r['best_kappa']:>+.2f}, "
              f"stress = {r['best_stress']:.4f}  {marker} "
              f"{'识别负曲率' if r['best_kappa'] <= -0.5 else '未识别负曲率'}")

    n_negative = sum(1 for r in all_results if r['best_kappa'] <= -0.5)
    n_zero = sum(1 for r in all_results if abs(r['best_kappa']) < 0.01)

    if n_negative >= 2:
        verdict = "A (流程可信)"
        decision_note = (
            f"≥ 2/3 合成树识别出 κ ≤ -0.5 (32D + graph shortest path target). "
            f"Stage 1c metric **可信**. phonism RQ-VAE 真实数据 κ=0 全层最优结论维持."
        )
    elif n_negative == 1:
        verdict = "B (部分可信)"
        decision_note = (
            f"仅 1/3 合成树识别负曲率. 流程部分可信, 需进一步诊断."
        )
    elif n_zero == len(all_results):
        verdict = "C (流程有 bug)"
        decision_note = (
            f"3/3 合成树 best κ = 0. 即使 32D + graph shortest path target, "
            f"metric **仍有 κ=0 bias**. 流程需要更深入修改."
        )
    else:
        verdict = "D (识别更负)"
        decision_note = (
            f"≥ 2/3 合成树 best κ ≤ -1.5. metric 偏向过负曲率."
        )

    print(f"\n  判定 = {verdict}")
    print(f"  决策 = {decision_note}")

    # ============================================================
    # 3. 输出 JSON
    # ============================================================
    output = {
        'task': 'task81_positive_control_v2',
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'method': 'Stage 1c RGD MDS + Kruskal stress-1, target=graph shortest path',
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
        'v1_failure_note': (
            'v1 用 2D 嵌入 + euclidean pairwise target, 3/3 树 best κ=0. '
            '根因: 2D Poincaré ball 嵌入数据点 norm 小时, euclidean ≈ Poincaré distance, '
            'κ=0 Euclidean MDS 完美拟合 (stress ≈ 0).'
        ),
    }

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    output_json = json.dumps(output, indent=2, default=lambda o: float(o) if isinstance(o, np.floating) else o)
    output_json = output_json.replace('Infinity', 'null').replace('NaN', 'null')
    with open(args.output, 'w') as f:
        f.write(output_json)
    print(f"\n[Task #81 v2] Result saved to {args.output}")


if __name__ == '__main__':
    main()