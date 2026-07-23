#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Task #81 — Stage 1c Metric 阳性对照 (Positive Control).

用户 (2026-07-23) 质疑 Task #80 Stage 1c v2 metric 是否可信: 如果流程在已知是双曲结构
(κ=-1 嵌入) 的合成数据上**都不能**识别出负曲率, 那 "phonism 数据是欧氏" 结论站不住脚.

本脚本生成 3 棵已知是双曲结构的合成树, 喂进 Stage 1c metric (Riemannian GD MDS +
Kruskal stress-1), 看能否正确识别 κ ≈ -1.

流程:
  1. 在 Poincaré ball κ=-1 嵌入 3 棵不同结构的树 (Tree-A/B/C, 21/40/31 节点)
  2. 对每棵树, 跑 compute_true_distortion 在 7 个 κ 候选
  3. 输出 best κ per tree + 全局判定
  4. 输出 JSON 到 verdicts/task81_positive_control_phonism_metric.json
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

# 复用 Task #80 Stage 1c metric
sys.path.insert(0, '/home/wlia0047/ar57/wenyu/GeneRec/scripts')
from task80_stage1c_true_distortion import compute_true_distortion  # noqa: E402


# ==============================================================================
# 双曲树嵌入 (Nickel & Kiela 2017 / Ganea 2018 Eq. 7 风格)
# ==============================================================================

def mobius_exp_map(x: np.ndarray, v: np.ndarray, kappa: float) -> np.ndarray:
    """Möbius exponential map at x for tangent vector v (Poincaré ball).

    Ganea 2018 Eq. 7:
        exp_x(v) = x ⊕ (tanh(sqrt(|κ|)·λ_x·||v||/2) · v / (sqrt(|κ|)·||v||))
    where λ_x = 2 / (1 - κ||x||²) 是 conformal factor.

    Args:
        x: (d,) 当前点 (Poincaré ball)
        v: (d,) 切空间向量
        kappa: 曲率 (κ < 0)

    Returns:
        x_new: (d,) 新点
    """
    if kappa >= 0:
        return x + v  # 欧氏
    sqrt_kappa = np.sqrt(-kappa)
    x_norm_sq = np.dot(x, x)
    lambda_x = 2.0 / (1.0 - kappa * x_norm_sq)
    v_norm = np.linalg.norm(v)
    if v_norm < 1e-12:
        return x
    t = np.tanh(sqrt_kappa * lambda_x * v_norm / 2.0)
    return x + (t / (sqrt_kappa * v_norm)) * v


def project_to_ball(x: np.ndarray, kappa: float, eps: float = 1e-4) -> np.ndarray:
    """约束点到 Poincaré ball 内 (||x||² < -1/κ - eps)."""
    if kappa >= 0:
        return x
    max_norm = np.sqrt(-1.0 / kappa) - eps
    norm = np.linalg.norm(x)
    if norm >= max_norm:
        return x * (max_norm / norm)
    return x


def generate_hyperbolic_tree(
    n_levels: int,
    branching: int,
    kappa: float = -1.0,
    step_per_level: List[float] | None = None,
    noise_sigma: float = 0.02,
    seed: int = 42,
    dim: int = 2,
) -> Tuple[np.ndarray, List[Tuple[int, int]]]:
    """在 Poincaré ball κ 双曲空间嵌入一棵树.

    Args:
        n_levels: 层数 (含根, 所以 n_levels=3 表示 root + 2 层子节点)
        branching: 每节点分支数
        kappa: 双曲曲率 (κ<0)
        step_per_level: 每层 step 大小 list (默认 [0, 0.40, 0.25, 0.15])
        noise_sigma: 切空间噪声 σ
        seed: 随机种子
        dim: 嵌入维度 (默认 2D)

    Returns:
        coords: (N, dim) Poincaré ball 点
        edges: list of (parent_idx, child_idx)
    """
    rng = np.random.default_rng(seed)

    if step_per_level is None:
        # 默认: root=0, level 1=0.40, level 2=0.25, level 3=0.15 (越深 step 越小)
        step_per_level = [0.0] + [0.40 / (1.0 + 0.5 * l) for l in range(1, n_levels)]

    coords: List[np.ndarray] = [np.zeros(dim)]  # root at origin
    edges: List[Tuple[int, int]] = []

    for level in range(1, n_levels):
        # 当前层从上一层的所有节点扩展
        n_at_prev_level = branching ** (level - 1)
        parent_start = len(coords) - n_at_prev_level

        for parent_idx in range(parent_start, len(coords)):
            parent = coords[parent_idx]
            r = step_per_level[level]

            for _ in range(branching):
                # 随机切空间方向
                theta = rng.uniform(0, 2 * np.pi)
                direction = np.array([np.cos(theta), np.sin(theta)])[:dim]
                v = direction * r

                # Möbius exp map 到双曲空间
                child = mobius_exp_map(parent, v, kappa)
                child = project_to_ball(child, kappa)

                # 加切空间高斯噪声
                if noise_sigma > 0:
                    noise = rng.normal(0, noise_sigma, size=dim)
                    # 投影到切空间 at child
                    inner_cn = np.dot(child, noise)
                    proj_noise = noise + kappa * inner_cn * child
                    child = mobius_exp_map(child, proj_noise, kappa)
                    child = project_to_ball(child, kappa)

                coords.append(child)
                edges.append((parent_idx, len(coords) - 1))

    return np.array(coords), edges


# ==============================================================================
# Stage 1c metric 跑合成数据
# ==============================================================================

KAPPA_CANDIDATES = [-2.0, -1.5, -1.0, -0.7, -0.5, -0.3, 0.0]


def run_metric_on_tree(coords: np.ndarray, tree_label: str, device: str = 'cpu') -> Dict:
    """对单棵树跑 7 κ 候选, 返回 stress 表 + best κ."""
    print(f"\n[{tree_label}] Running metric on {coords.shape[0]} points (dim={coords.shape[1]})")
    print(f"  {'κ':>8} | {'stress':>10} | {'elapsed (s)':>12}")
    print(f"  {'-'*8}-+-{'-'*10}-+-{'-'*12}")

    stresses: Dict[float, float] = {}
    for k in KAPPA_CANDIDATES:
        t0 = time.time()
        try:
            stress = compute_true_distortion(
                coords.astype(np.float32),
                kappa=k,
                n_subset=coords.shape[0],  # 全用, 合成数据 < 300
                seed=42,
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

    # Best κ (lowest stress)
    finite = {k: v for k, v in stresses.items() if np.isfinite(v)}
    if finite:
        best_k, best_s = min(finite.items(), key=lambda x: x[1])
    else:
        best_k, best_s = 0.0, float('inf')

    return {
        'tree_label': tree_label,
        'num_points': coords.shape[0],
        'kappa_generated': -1.0,
        'stress_per_kappa': stresses,
        'best_kappa': best_k,
        'best_stress': best_s,
        'all_finite': len(finite) == len(stresses),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--output', required=True, help='Output JSON path')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--device', default='cpu')
    args = p.parse_args()

    print("=" * 90)
    print("[Task #81] Stage 1c Metric Positive Control — 3 棵已知 κ=-1 合成树")
    print("=" * 90)

    # ============================================================
    # 1. 生成 3 棵合成树
    # ============================================================
    trees_spec = [
        ('Tree-A', {'n_levels': 3, 'branching': 4, 'kappa': -1.0, 'noise_sigma': 0.02}),
        ('Tree-B', {'n_levels': 4, 'branching': 3, 'kappa': -1.0, 'noise_sigma': 0.02}),
        ('Tree-C', {'n_levels': 3, 'branching': 5, 'kappa': -1.0, 'noise_sigma': 0.02}),
    ]

    all_results = []
    all_coords = {}
    for label, spec in trees_spec:
        coords, edges = generate_hyperbolic_tree(seed=args.seed, dim=2, **spec)
        all_coords[label] = coords
        print(f"\n[{label}] Generated: {coords.shape[0]} nodes, "
              f"norm max = {np.linalg.norm(coords, axis=1).max():.3f}, "
              f"norm mean = {np.linalg.norm(coords, axis=1).mean():.3f}")

    # ============================================================
    # 2. 跑 Stage 1c metric on 每棵树
    # ============================================================
    for label, spec in trees_spec:
        coords = all_coords[label]
        result = run_metric_on_tree(coords, label, device=args.device)
        result['config'] = spec
        all_results.append(result)

    # ============================================================
    # 3. 决策判定
    # ============================================================
    print()
    print("=" * 90)
    print("判定: Stage 1c metric 对 3 棵合成树能否识别 κ ≈ -1?")
    print("=" * 90)

    for r in all_results:
        marker = "✓" if r['best_kappa'] <= -0.5 else "✗"
        print(f"  [{r['tree_label']}] best κ = {r['best_kappa']:>+.2f}, "
              f"stress = {r['best_stress']:.4f}  {marker} "
              f"{'识别负曲率' if r['best_kappa'] <= -0.5 else '未识别负曲率'}")

    # 统计
    n_negative = sum(1 for r in all_results if r['best_kappa'] <= -0.5)
    n_zero = sum(1 for r in all_results if abs(r['best_kappa']) < 0.01)

    print()
    if n_negative >= 2:
        verdict = "A (流程可信)"
        decision_note = (
            f"≥ 2/3 合成树识别出 κ ≤ -0.5. Stage 1c metric **可信**. "
            f"phonism RQ-VAE 真实数据 κ=0 全层最优结论维持 (task80 verdict §9)."
        )
    elif n_negative == 1:
        verdict = "B (部分可信)"
        decision_note = (
            f"仅 1/3 合成树识别负曲率. 流程部分可信, 需进一步加跑更多合成配置."
        )
    elif n_zero == len(all_results):
        verdict = "C (流程有 bug)"
        decision_note = (
            f"3/3 合成树 best κ = 0. Stage 1c metric **有 bug**. "
            f"task80 verdict §9 (phonism κ=0 最优) 失效, 需先修流程, 不能据此下结论."
        )
    else:
        verdict = "D (识别更负曲率)"
        decision_note = (
            f"≥ 2/3 合成树 best κ ≤ -1.5. Stage 1c metric 偏向过负曲率, "
            f"流程也有 bias. task80 verdict §9 失效."
        )

    print(f"\n  判定 = {verdict}")
    print(f"  决策 = {decision_note}")

    # ============================================================
    # 4. 输出 JSON
    # ============================================================
    output = {
        'task': 'task81_positive_control_phonism_metric',
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'method': 'Stage 1c Riemannian GD MDS + Kruskal stress-1 (re-used)',
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
    }

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    output_json = json.dumps(output, indent=2, default=lambda o: float(o) if isinstance(o, np.floating) else o)
    output_json = output_json.replace('Infinity', 'null').replace('NaN', 'null')
    with open(args.output, 'w') as f:
        f.write(output_json)
    print(f"\n[Task #81] Result saved to {args.output}")


if __name__ == '__main__':
    main()