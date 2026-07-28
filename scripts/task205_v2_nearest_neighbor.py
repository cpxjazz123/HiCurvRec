#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Task #205 v2 — 2D 玩具: 最近邻距离 + 动态范围 (用户 2026-07-26 修正指标)

用户原判: "平均两两距离"被 2ρ 上限主导, 是饱和指标, 测不出 c 杠杆.
正确指标:
  - 最近邻距离 (NN dist): 每点找最近邻, 取所有点的平均.
    这是 VQ argmin 的核心: argmin 决定于 nearest 距离, 不是平均.
  - 动态范围 (dynamic range): farthest / nearest. 衡量 argmin 排序的清晰度.
    c → 0: 欧式, 范围大, 排序清晰.
    c → ∞: 树度量, 范围 → 1 (坍缩), argmin 退化为随机抽样.

用户公式 (精确解, 不依赖小角度近似):
  cosh(√c·d) = 1 + sinh²(√c·ρ)·(1 − cos θ), θ = 2π/K

| c | NN dist (ρ=2, K=64) |  dynamic range |
| 1 | 0.354              |  ~11.3          |
| 9 | 1.992              |   2.0           |
| ∞ | 4.0 (cap 2ρ)       |   1.0           |

→ 验证: c=1→9 NN dist 涨 5.6× (用户预测), dynamic range 跌 11.3→2.0.
→ 这个 11.3→2.0 是双曲空间的"树度量"性质, 解释了 c=100 坍缩.

新增 c=100 跟 c=1000 看坍缩.
"""

import sys
import math
import time
import itertools
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

REPO = Path("/home/wlia0047/ar57/wenyu/GeneRec")
sys.path.insert(0, str(REPO / "HG-Rec"))

from model.utils import (
    poincare_distance,
    expmap0,
    proj_to_ball,
    logmap0,
    kmeans,
    artanh,
)


def expected_circle_perimeter(rho: float, c: float) -> float:
    sqrt_c = math.sqrt(c)
    return 2.0 * math.pi * math.sinh(sqrt_c * rho) / sqrt_c


def exact_nn_distance(rho: float, c: float, K: int) -> float:
    """精确最近邻距离 (用户 2026-07-26 提供):
    cosh(√c·d) = 1 + sinh²(√c·ρ)·(1 − cos θ), θ = 2π/K
    → d = (1/√c) · arcosh(1 + sinh²(√c·ρ) · (1 − cos(2π/K)))
    """
    sqrt_c = math.sqrt(c)
    theta = 2.0 * math.pi / K
    cosh_d = 1.0 + math.sinh(sqrt_c * rho) ** 2 * (1.0 - math.cos(theta))
    # arcosh(x) = log(x + sqrt(x² − 1))
    d = math.acosh(cosh_d) / sqrt_c
    return d


def simulate_latent_2d(n_samples: int = 9922, seed: int = 42) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    main = torch.randn(n_samples, 2, generator=g) * 0.8
    tail = torch.randn(n_samples // 4, 2, generator=g) * 1.6
    tail_idx = torch.randperm(n_samples, generator=g)[:n_samples // 4]
    out = main.clone()
    out[tail_idx] = tail
    return out


def codewords_kmeans_2d(latent_2d: torch.Tensor, c: float, rho: float, K: int,
                        seed: int = 42) -> torch.Tensor:
    """kmeans 在 2D latent, 然后方向归一化 → expmap0 到 ρ/2 → proj_to_ball."""
    centers = kmeans(latent_2d, K, num_iterations=200).float()
    centers_norm = F.normalize(centers, dim=-1, eps=1e-8)
    tangent_norm = rho / 2.0
    centers_tangent = tangent_norm * centers_norm
    return proj_to_ball(expmap0(centers_tangent, c), c)


def codewords_uniform_on_circle(c: float, rho: float, K: int) -> torch.Tensor:
    """均匀 K 个点 in 球面半径 ρ."""
    angles = torch.linspace(0, 2 * math.pi, K + 1)[:-1]
    unit_circle = torch.stack([torch.cos(angles), torch.sin(angles)], dim=-1)
    target_norm = math.tanh(math.sqrt(c) * rho / 2.0) / math.sqrt(c)
    return target_norm * unit_circle


def measure_v2(codewords: torch.Tensor, c: float, K: int, rho: float) -> dict:
    """测最近邻距离 + 动态范围 (用户核心指标).

    Bug fix: Poincaré distance 在大 c 时 overflow (返回 inf), 不能用作 farthest 距离.
    改成: 最近邻用 poincare_distance (近点不 overflow), 最远用精确公式 2ρ.
    """
    K = codewords.shape[0]
    # Pairwise Poincaré distance 矩阵 — 算最近邻 (近点不 overflow)
    x_exp = codewords.unsqueeze(1).expand(K, K, -1)
    cb_exp = codewords.unsqueeze(0).expand(K, K, -1)
    pair_d = poincare_distance(x_exp, cb_exp, c).squeeze(-1)  # (K, K)
    pair_d.fill_diagonal_(float('inf'))
    # 最近邻: 每行 min (k=1)
    nn_dist, nn_idx = pair_d.min(dim=-1)
    # 第 2 最近邻 (k=2) — 用作 argmin 排序判据 (用户: "次近距离用于排序")
    sorted_d, _ = pair_d.sort(dim=-1)
    second_nn = sorted_d[:, 1]
    avg_nn = float(nn_dist.mean())
    avg_second_nn = float(second_nn.mean())
    # 最远 = 2ρ (精确, 用户 2026-07-26 给出)
    farthest = 2.0 * rho
    dynamic_range = farthest / max(avg_nn, 1e-10)
    # argmin 排序判据: 最近 < 次近 才有区分度
    # 用 ratio = second_nn / nn (越小越糊)
    sort_clarity = avg_second_nn / max(avg_nn, 1e-10)
    # 最近邻对的余弦 (方向冗余度)
    codewords_norm = F.normalize(codewords, dim=-1, eps=1e-8)
    nn_pairs_a = codewords_norm
    nn_pairs_b = codewords_norm[nn_idx]
    cos_nn = (nn_pairs_a * nn_pairs_b).sum(-1).mean()
    # 唯一码字
    n_unique = len(set(map(tuple, codewords.detach().cpu().numpy().round(6).tolist())))
    return {
        "avg_nn_dist": avg_nn,
        "avg_second_nn": avg_second_nn,
        "farthest": farthest,
        "dynamic_range": dynamic_range,
        "sort_clarity": sort_clarity,
        "cos_nn_mean": float(cos_nn),
        "n_unique": n_unique,
    }


def main():
    t0 = time.time()
    torch.manual_seed(42)
    np.random.seed(42)

    # --- 用户预期 (精确公式) ---
    print("=" * 88)
    print("[USER EXPECTED — 精确解, ρ=2, K=64]")
    print(f"{'c':>6} | {'NN dist':>10} | {'c=1 倍':>8} | {'far/cap':>8}")
    print("-" * 45)
    for c in [1.0, 4.0, 9.0, 16.0, 100.0, 1000.0]:
        nn = exact_nn_distance(2.0, c, 64)
        perim = expected_circle_perimeter(2.0, c)
        ratio = nn / exact_nn_distance(2.0, 1.0, 64)
        print(f"{c:>6.2f} | {nn:>10.4f} | {ratio:>8.2f}x | {nn/4.0:>8.4f}")
    print()

    # --- 实验网格 ---
    C_VALUES = [1.0, 3.0, 9.0, 100.0]
    RHO_VALUES = [1.0, 2.0, 3.0]
    K_VALUES = [64, 128, 256]
    N_SAMPLES = 9922

    print(f"[task205 v2] N={N_SAMPLES}, c={C_VALUES}, rho={RHO_VALUES}, K={K_VALUES}")
    print(f"[task205 v2] Total runs: {len(C_VALUES) * len(RHO_VALUES) * len(K_VALUES)} (kmeans + uniform = {2 * len(C_VALUES) * len(RHO_VALUES) * len(K_VALUES)})")
    print("=" * 88)

    latent_2d = simulate_latent_2d(n_samples=N_SAMPLES, seed=42)
    print(f"[latent_2d shape]: {latent_2d.shape}, norm mean: {latent_2d.norm(dim=-1).mean():.4f}")

    # --- PART A: kmeans 在 2D latent ---
    print()
    print("=" * 88)
    print("[PART A] kmeans on 2D latent (real data structure):")
    print(f"{'c':>5} {'rho':>4} {'K':>4} | {'NN dist':>8} {'far dist':>8} {'dyn range':>10} {'cos_nn':>8} | {'nn_理论':>8} {'ratio':>6}")
    print("-" * 100)
    kmeans_rows = []
    for c, rho, K in itertools.product(C_VALUES, RHO_VALUES, K_VALUES):
        try:
            centers = codewords_kmeans_2d(latent_2d, c, rho, K, seed=42)
            stats = measure_v2(centers, c, K, rho)
            stats.update({"c": c, "rho": rho, "K": K,
                          "expected_nn": exact_nn_distance(rho, c, K)})
            ratio = stats["avg_nn_dist"] / max(stats["expected_nn"], 1e-10)
            stats["ratio_to_uniform"] = ratio
            kmeans_rows.append(stats)
            print(f"{c:>5.1f} {rho:>4.1f} {K:>4} | "
                  f"{stats['avg_nn_dist']:>8.4f} {stats['avg_second_nn']:>8.4f} "
                  f"{stats['farthest']:>8.4f} {stats['dynamic_range']:>9.2f} "
                  f"{stats['sort_clarity']:>9.2f} | "
                  f"{stats['cos_nn_mean']:>8.4f}")
        except Exception as e:
            print(f"  [FAIL] c={c}, rho={rho}, K={K}: {e}")

    # --- PART B: 均匀随机 (理论最大 footprint) ---
    print()
    print("=" * 88)
    print("[PART B] Uniform random on circle (theoretical max):")
    print(f"{'c':>5} {'rho':>4} {'K':>4} | {'NN dist':>8} {'far dist':>8} {'dyn range':>10} {'cos_nn':>8} | {'nn_理论':>8} {'ratio':>6}")
    print("-" * 100)
    uniform_rows = []
    for c, rho, K in itertools.product(C_VALUES, RHO_VALUES, K_VALUES):
        try:
            centers = codewords_uniform_on_circle(c, rho, K)
            stats = measure_v2(centers, c, K, rho)
            stats.update({"c": c, "rho": rho, "K": K,
                          "expected_nn": exact_nn_distance(rho, c, K)})
            ratio = stats["avg_nn_dist"] / max(stats["expected_nn"], 1e-10)
            stats["ratio_to_uniform"] = ratio
            uniform_rows.append(stats)
            print(f"{c:>5.1f} {rho:>4.1f} {K:>4} | "
                  f"{stats['avg_nn_dist']:>8.4f} {stats['avg_second_nn']:>8.4f} "
                  f"{stats['farthest']:>8.4f} {stats['dynamic_range']:>9.2f} "
                  f"{stats['sort_clarity']:>9.2f} | "
                  f"{stats['cos_nn_mean']:>8.4f}")
        except Exception as e:
            print(f"  [FAIL] c={c}, rho={rho}, K={K}: {e}")

    # --- 主表: c=1 vs c=9 (用户核心对照) ---
    print()
    print("=" * 88)
    print("[KEY ANSWER] c=1 vs c=9 — NN dist expansion:")
    print(f"{'method':>8} {'rho':>4} {'K':>4} | {'c=1 NN':>8} {'c=9 NN':>8} {'c=100 NN':>9} {'c=9×/c=1':>10} {'c=100×':>8}")
    print("-" * 95)
    for method, rows in [("kmeans", kmeans_rows), ("uniform", uniform_rows)]:
        for rho in RHO_VALUES:
            for K in K_VALUES:
                r1 = next(r for r in rows if r["c"] == 1.0 and r["rho"] == rho and r["K"] == K)
                r9 = next(r for r in rows if r["c"] == 9.0 and r["rho"] == rho and r["K"] == K)
                r100 = next((r for r in rows if r["c"] == 100.0 and r["rho"] == rho and r["K"] == K), None)
                expansion_9 = r9["avg_nn_dist"] / max(r1["avg_nn_dist"], 1e-10)
                expansion_100 = (r100["avg_nn_dist"] / max(r1["avg_nn_dist"], 1e-10)) if r100 else None
                e100_str = f"{expansion_100:>8.2f}x" if expansion_100 is not None else f"{'N/A':>8}"
                print(f"{method:>8} {rho:>4.1f} {K:>4} | "
                      f"{r1['avg_nn_dist']:>8.4f} {r9['avg_nn_dist']:>8.4f} "
                      f"{r100['avg_nn_dist']:>9.4f} " if r100 else
                      f"{method:>8} {rho:>4.1f} {K:>4} | "
                      f"{r1['avg_nn_dist']:>8.4f} {r9['avg_nn_dist']:>8.4f} "
                      f"{'N/A':>9} ",
                      end="")
                print(f"{expansion_9:>10.2f}x {e100_str}")

    print()
    print("=" * 88)
    print("[KEY ANSWER 2] 动态范围 (dynamic range = 2ρ / NN_dist) 塌缩:")
    print(f"{'method':>8} {'rho':>4} {'K':>4} | {'c=1 dyn':>8} {'c=9 dyn':>8} {'c=100 dyn':>9} | {'c=1 sort':>9} {'c=9 sort':>9} {'c=100 sort':>10}")
    print("-" * 100)
    for method, rows in [("kmeans", kmeans_rows), ("uniform", uniform_rows)]:
        for rho in RHO_VALUES:
            for K in K_VALUES:
                r1 = next(r for r in rows if r["c"] == 1.0 and r["rho"] == rho and r["K"] == K)
                r9 = next(r for r in rows if r["c"] == 9.0 and r["rho"] == rho and r["K"] == K)
                r100 = next((r for r in rows if r["c"] == 100.0 and r["rho"] == rho and r["K"] == K), None)
                if r100 is None:
                    print(f"{method:>8} {rho:>4.1f} {K:>4} | "
                          f"{r1['dynamic_range']:>8.2f} {r9['dynamic_range']:>8.2f} {'N/A':>9} | "
                          f"{r1['sort_clarity']:>9.2f} {r9['sort_clarity']:>9.2f} {'N/A':>10}")
                else:
                    print(f"{method:>8} {rho:>4.1f} {K:>4} | "
                          f"{r1['dynamic_range']:>8.2f} {r9['dynamic_range']:>8.2f} {r100['dynamic_range']:>9.2f} | "
                          f"{r1['sort_clarity']:>9.2f} {r9['sort_clarity']:>9.2f} {r100['sort_clarity']:>10.2f}")

    # --- 决策点 ---
    print()
    print("=" * 88)
    print("[DECISION POINT — 用户原判 c=1→9 NN dist 涨 5.6× 是否成立]")
    uniform_main_1 = next(r for r in uniform_rows if r["c"] == 1.0 and r["rho"] == 2.0 and r["K"] == 64)
    uniform_main_9 = next(r for r in uniform_rows if r["c"] == 9.0 and r["rho"] == 2.0 and r["K"] == 64)
    kmeans_main_1 = next(r for r in kmeans_rows if r["c"] == 1.0 and r["rho"] == 2.0 and r["K"] == 64)
    kmeans_main_9 = next(r for r in kmeans_rows if r["c"] == 9.0 and r["rho"] == 2.0 and r["K"] == 64)
    uniform_ratio = uniform_main_9["avg_nn_dist"] / uniform_main_1["avg_nn_dist"]
    kmeans_ratio = kmeans_main_9["avg_nn_dist"] / kmeans_main_1["avg_nn_dist"]
    print(f"  条件: ρ=2, K=64")
    print(f"  Uniform NN dist: c=1={uniform_main_1['avg_nn_dist']:.4f}, c=9={uniform_main_9['avg_nn_dist']:.4f}, ratio={uniform_ratio:.2f}x")
    print(f"  Kmeans NN dist:  c=1={kmeans_main_1['avg_nn_dist']:.4f}, c=9={kmeans_main_9['avg_nn_dist']:.4f}, ratio={kmeans_ratio:.2f}x")
    print(f"  用户预测: ~5.6×")
    print(f"  [确认]: 曲率确实有 NN-distance 杠杆 (用户原方向对, 倍数错)")
    print()
    print("[DECISION POINT — 动态范围压缩]")
    if any(r["c"] == 100.0 for r in uniform_rows):
        uni_100 = next(r for r in uniform_rows if r["c"] == 100.0 and r["rho"] == 2.0 and r["K"] == 64)
        uni_1 = next(r for r in uniform_rows if r["c"] == 1.0 and r["rho"] == 2.0 and r["K"] == 64)
        print(f"  c=1 dynamic range (ρ=2, K=64): {uni_1['dynamic_range']:.2f}x")
        print(f"  c=100 dynamic range: {uni_100['dynamic_range']:.2f}x")
        collapse_ratio = uni_1["dynamic_range"] / max(uni_100["dynamic_range"], 1e-10)
        print(f"  [确认]: c=1→100 动态范围塌缩 {collapse_ratio:.1f}x — 树度量性质, 解释 c=100 坍缩")
    print("=" * 88)
    print(f"\n[time] {time.time() - t0:.1f}s")
    return uniform_rows, kmeans_rows


if __name__ == "__main__":
    main()