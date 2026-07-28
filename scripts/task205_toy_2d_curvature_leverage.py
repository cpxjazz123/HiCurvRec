#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Task #205 — 2D hyperbolic toy: does c:1→9 really expand codeword footprint ×18?

用户 2026-07-26 根本诊断: 32D 太空了, 几何没约束力. 核心主路: 降维到 2D 双曲.
本脚本验证前提: 在 2D Poincaré ball 里, c:1→9 是否真的让码字地盘 (码字间平均 Poincaré 距离)
扩大 18×.

设计:
- 变量: 曲率 c ∈ {1, 3, 9}, 半径 ρ ∈ {1.0, 2.0, 3.0}, 码字数 K ∈ {64, 128, 256}
- 不变: 2D, 数据 (模拟 encoder latent 在 2D 投影)
- 测量:
    1. avg_pair_poincare_dist  — 用户表里"码字间距"
    2. avg_pair_cosine  — cos 大 → 挤; cos 小 → 空
    3. avg_norm  — 码字平均 ‖x‖_E (期望 ≈ 半径 bound (1-1e-5)/√c 但实际会小, 因 kmeans 在切空间)
    4. n_unique  — 唯一码字数 (理论 = K)
- 输出: 一张表 (c × ρ × K) + 用户的 ×18 假设验证

不占 GPU (纯 CPU).
"""

import sys
import math
import time
import itertools
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

# 复用 HG-Rec utils
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
    """圆周长 2D hyperbolic 半径 ρ 处: 2π · sinh(√c · ρ) / √c.
    用户 2026-07-26 关键公式. 码字地盘 = perimeter / K (近似, 假设均匀分布).
    """
    sqrt_c = math.sqrt(c)
    return 2.0 * math.pi * math.sinh(sqrt_c * rho) / sqrt_c


def expected_footprint(rho: float, c: float, K: int) -> float:
    """Per-codeword footprint ≈ perimeter / K. 用户预期: c=1 vs c=9 (ρ=2) → ×18."""
    return expected_circle_perimeter(rho, c) / K


def simulate_latent_2d(n_samples: int = 9922, seed: int = 42) -> torch.Tensor:
    """模拟 9922 个 item 的 2D encoder latent (跟 HG-Real 9.9k item 数一致).
    HG-Rec 32D latent norm 大概 0.5-2.0, 这里 2D 同尺度.
    """
    g = torch.Generator().manual_seed(seed)
    # Mix: 80% cluster around origin (most items), 20% tail (rare items)
    main = torch.randn(n_samples, 2, generator=g) * 0.8
    tail = torch.randn(n_samples // 4, 2, generator=g) * 1.6
    tail_idx = torch.randperm(n_samples, generator=g)[:n_samples // 4]
    out = main.clone()
    out[tail_idx] = tail
    return out


def codewords_in_ball_2d(latent_2d: torch.Tensor, c: float, rho: float, K: int,
                         seed: int = 42) -> torch.Tensor:
    """Kmeans in tangent space (logmap0 from default rho=1), then map to Poincaré ball at radius rho/2.

    用户设计: 码字方向编码语义, 半径编码层级.
    这里用 tangent space 的 logmap0 / expmap0 在 ρ=1 球面上做 kmeans, 然后 rescale 到 ρ/2 再 expmap0.
    复用现有 utils 全部函数.

    Returns:
        codebook (K, 2) on Poincaré ball, ‖x‖_E ≈ tanh(ρ/2) (c=1)
    """
    # Step 1: 投射 latent 到 ρ=1 球面 (tangent-equivalent normalize).
    # 我们直接在 Euclidean plane 上做 kmeans (因为 2D tangent ≈ Euclidean 小邻域),
    # 然后通过 expmap0 把码字放到 ρ/2 球面上.
    centers = kmeans(latent_2d, K, num_iterations=200).float()  # (K, 2) raw
    # Step 2: 方向归一化 (取码字方向)
    centers_norm = F.normalize(centers, dim=-1, eps=1e-8)
    # Step 3: 投射到 Poincaré ball 半径 ρ/2 (tangent norm = ρ/2 → expmap0 后球内径 = tanh(ρ/2 / (1+sqrt(1+ρ²/4))) 简化成 tanh(ρ/2))
    # 用户 ρ 是 Poincaré 半径, 对应 tangent norm = sinh(ρ) (小角度) ≈ ρ. 我们用 tangent_norm = ρ/2 作缩放.
    tangent_norm = rho / 2.0
    centers_tangent = tangent_norm * centers_norm
    centers_hyp = proj_to_ball(expmap0(centers_tangent, c), c)
    return centers_hyp


def measure_distribution(codewords: torch.Tensor, c: float) -> dict:
    """测码字分布.
    Args:
        codewords: (K, 2) on Poincaré ball.
        c: 曲率.
    Returns:
        dict with avg_pair_d, avg_pair_cos, avg_norm, n_unique, min_norm, max_norm.
    """
    K = codewords.shape[0]
    norms = codewords.norm(dim=-1)
    # Pairwise Poincaré distance
    x_exp = codewords.unsqueeze(1).expand(K, K, -1)  # (K, K, 2)
    cb_exp = codewords.unsqueeze(0).expand(K, K, -1)
    pair_d = poincare_distance(x_exp, cb_exp, c).squeeze(-1)  # (K, K)
    # 取上三角 (排除 i==j)
    triu_mask = torch.triu(torch.ones(K, K, dtype=torch.bool), diagonal=1)
    pair_d_offdiag = pair_d[triu_mask]
    # Pairwise cosine (raw Euclidean — 在 Poincaré ball 内 cos 还是衡方向的量)
    codewords_norm = F.normalize(codewords, dim=-1, eps=1e-8)
    pair_cos = (codewords_norm.unsqueeze(1) * codewords_norm.unsqueeze(0)).sum(-1)  # (K, K)
    pair_cos_offdiag = pair_cos[triu_mask]
    # 唯一码字
    n_unique = len(set(map(tuple, codewords.detach().cpu().numpy().round(6).tolist())))
    return {
        "avg_pair_d": float(pair_d_offdiag.mean()),
        "min_pair_d": float(pair_d_offdiag.min()),
        "max_pair_d": float(pair_d_offdiag.max()),
        "std_pair_d": float(pair_d_offdiag.std()),
        "avg_pair_cos": float(pair_cos_offdiag.mean()),
        "max_pair_cos": float(pair_cos_offdiag.max()),
        "avg_norm": float(norms.mean()),
        "min_norm": float(norms.min()),
        "max_norm": float(norms.max()),
        "n_unique": n_unique,
    }


def main():
    t0 = time.time()
    torch.manual_seed(42)
    np.random.seed(42)

    # --- 实验网格 ---
    C_VALUES = [1.0, 3.0, 9.0]
    RHO_VALUES = [1.0, 2.0, 3.0]
    K_VALUES = [64, 128, 256]
    N_SAMPLES = 9922  # 跟 HG-Real item 数对齐 (5-core 后 Musical_Instruments 9922)

    print(f"[task205] 2D hyperbolic curvature leverage toy")
    print(f"[task205] N={N_SAMPLES}, c={C_VALUES}, rho={RHO_VALUES}, K={K_VALUES}")
    print(f"[task205] Total runs: {len(C_VALUES) * len(RHO_VALUES) * len(K_VALUES)}")
    print("=" * 88)

    # --- 一次性生成 latent ---
    latent_2d = simulate_latent_2d(n_samples=N_SAMPLES, seed=42)
    print(f"[task205] latent_2d shape: {latent_2d.shape}, norm mean: {latent_2d.norm(dim=-1).mean():.4f}")
    print()

    # --- 用户预期表 (perimeter / 64 码字间距) ---
    print("[User expected table — codeword spacing = perimeter(K=64, rho=2):")
    print(f"{'c':>6} | {'perimeter(ρ=2)':>15} | {'64 码字间距':>12}")
    print("-" * 40)
    for c in [0.25, 1.0, 4.0, 9.0]:
        perim = expected_circle_perimeter(2.0, c)
        print(f"{c:>6.2f} | {perim:>15.2f} | {perim/64:>12.3f}")
    print()
    print("[Expected footprint expansion at ρ=2, K=64 (perimeter/c=1 vs c=9):")
    print(f"  c=1 footprint: {expected_footprint(2.0, 1.0, 64):.3f}")
    print(f"  c=9 footprint: {expected_footprint(2.0, 9.0, 64):.3f}")
    print(f"  ratio: {expected_footprint(2.0, 9.0, 64) / expected_footprint(2.0, 1.0, 64):.2f}x")
    print("=" * 88)
    print()

    # --- CONTROL 测试: 均匀随机码字在圆上 (理论最大) ---
    # 用户假设: c:1→9 → 码字地盘 ×18. 这个假设的前提是码字均匀分布在圆上.
    # 但 kmeans 找的是数据聚类中心, 不是均匀分布. 这里对照测一下.
    print("=" * 88)
    print("[CONTROL TEST] 均匀随机码字 (uniform on circle, 理论最大 footprint):")
    print(f"{'c':>4} {'rho':>4} {'K':>4} | {'avg_d':>8} {'min_d':>8} {'max_d':>8} {'std_d':>8} | {'footprint':>10} {'L/2':>10}")
    print("-" * 80)
    uniform_rows = []
    for c, rho, K in itertools.product(C_VALUES, RHO_VALUES, K_VALUES):
        # 均匀 K 个点 on 单位圆, 然后 radius 缩放到 ρ (球面), 然后 expmap0
        angles = torch.linspace(0, 2 * math.pi, K + 1)[:-1]  # K 个均匀角度
        unit_circle = torch.stack([torch.cos(angles), torch.sin(angles)], dim=-1)  # (K, 2)
        # Scale to hyperbolic radius ρ: 投到球面 ‖x‖_E = tanh(√c·ρ/2)/√c
        target_norm = math.tanh(math.sqrt(c) * rho / 2.0) / math.sqrt(c)
        scaled = target_norm * unit_circle
        stats = measure_distribution(scaled, c)
        stats.update({"c": c, "rho": rho, "K": K,
                      "footprint_expected": expected_footprint(rho, c, K),
                      "L/2_expected": expected_circle_perimeter(rho, c) / 2.0})
        uniform_rows.append(stats)
        print(f"{c:>4.1f} {rho:>4.1f} {K:>4} | "
              f"{stats['avg_pair_d']:>8.4f} {stats['min_pair_d']:>8.4f} {stats['max_pair_d']:>8.4f} {stats['std_pair_d']:>8.4f} | "
              f"{stats['footprint_expected']:>10.3f} {stats['L/2_expected']:>10.3f}")
    print()

    # --- 主表 (c × ρ × K) ---
    rows = []
    for c, rho, K in itertools.product(C_VALUES, RHO_VALUES, K_VALUES):
        try:
            centers_hyp = codewords_in_ball_2d(latent_2d, c, rho, K, seed=42)
            stats = measure_distribution(centers_hyp, c)
            stats.update({"c": c, "rho": rho, "K": K,
                          "footprint_expected": expected_footprint(rho, c, K)})
            rows.append(stats)
        except Exception as e:
            print(f"  [FAIL] c={c}, rho={rho}, K={K}: {e}")

    # --- 输出表 1: avg_pair_d (用户表里"码字间距"主指标) ---
    print("\n[TABLE 1] avg_pair_d (码字间平均 Poincaré 距离, 用户表里'码字间距'主指标):")
    print(f"{'c':>4} {'rho':>4} {'K':>4} | {'avg_d':>8} {'min_d':>8} {'max_d':>8} {'std_d':>8} | {'footprint':>10} {'ratio':>8}")
    print("-" * 80)
    for r in rows:
        # ratio vs (c=1, same rho, K=64) baseline
        baseline = next(b for b in rows if b["c"] == 1.0 and b["rho"] == r["rho"] and b["K"] == r["K"])
        ratio = r["avg_pair_d"] / baseline["avg_pair_d"]
        print(f"{r['c']:>4.1f} {r['rho']:>4.1f} {r['K']:>4} | "
              f"{r['avg_pair_d']:>8.4f} {r['min_pair_d']:>8.4f} {r['max_pair_d']:>8.4f} {r['std_pair_d']:>8.4f} | "
              f"{r['footprint_expected']:>10.3f} {ratio:>8.2f}x")
    print()

    # --- 输出表 2: 关键杠杆判定 (用户假设: c=1→9 给 ×18 在 ρ=2, K=64) ---
    print("\n[TABLE 2] KEY ANSWER: c=1 vs c=9 footprint expansion (用户 ×18 假设验证):")
    print(f"{'rho':>4} {'K':>4} | {'c=1 avg_d':>10} {'c=9 avg_d':>10} {'实际 ratio':>10} {'footprint ratio':>14}")
    print("-" * 65)
    for rho in RHO_VALUES:
        for K in K_VALUES:
            r1 = next(r for r in rows if r["c"] == 1.0 and r["rho"] == rho and r["K"] == K)
            r9 = next(r for r in rows if r["c"] == 9.0 and r["rho"] == rho and r["K"] == K)
            actual_ratio = r9["avg_pair_d"] / r1["avg_pair_d"]
            footprint_ratio = r9["footprint_expected"] / r1["footprint_expected"]
            print(f"{rho:>4.1f} {K:>4} | "
                  f"{r1['avg_pair_d']:>10.4f} {r9['avg_pair_d']:>10.4f} "
                  f"{actual_ratio:>10.2f}x {footprint_ratio:>14.2f}x")
    print()

    # --- 输出表 3: avg_pair_cos (空 vs 挤 — cos 小 → 空, cos 大 → 挤) ---
    print("\n[TABLE 3] avg_pair_cos (cos 小 → 空 / cos 大 → 挤; 用户'32D 太松'诊断对应):")
    print(f"{'c':>4} {'rho':>4} {'K':>4} | {'avg_cos':>8} {'max_cos':>8} | {'avg_norm':>8} {'min_norm':>8} {'max_norm':>8} | {'n_unique':>8}")
    print("-" * 80)
    for r in rows:
        print(f"{r['c']:>4.1f} {r['rho']:>4.1f} {r['K']:>4} | "
              f"{r['avg_pair_cos']:>8.4f} {r['max_pair_cos']:>8.4f} | "
              f"{r['avg_norm']:>8.4f} {r['min_norm']:>8.4f} {r['max_norm']:>8.4f} | "
              f"{r['n_unique']:>8}")
    print()

    # --- 决策树答案 ---
    print("=" * 88)
    print("[DECISION TREE — 用户假设验证]")
    # 主要看 ρ=2 K=64 c=1 vs c=9
    main_1 = next(r for r in rows if r["c"] == 1.0 and r["rho"] == 2.0 and r["K"] == 64)
    main_9 = next(r for r in rows if r["c"] == 9.0 and r["rho"] == 2.0 and r["K"] == 64)
    main_ratio = main_9["avg_pair_d"] / main_1["avg_pair_d"]
    footprint_ratio_2_64 = expected_footprint(2.0, 9.0, 64) / expected_footprint(2.0, 1.0, 64)
    print(f"  条件: ρ=2, K=64, c=1 vs c=9")
    print(f"  实际 avg_pair_d ratio: {main_ratio:.2f}x")
    print(f"  理论 footprint ratio (用户预期): {footprint_ratio_2_64:.2f}x")
    if main_ratio >= 10.0:
        verdict = "✅ PASS — 降维前提成立 (c:1→9 实际扩张 ≥ 10×). 建议进入 Phase 2: e_dim=32=2+30 拆分."
    elif main_ratio >= 5.0:
        verdict = "⚠️ WEAK — 2D 优势不够大. 建议改 4D 双曲 + 加大 ρ."
    else:
        verdict = "❌ FAIL — 用户几何理论有问题. 暂停重设计, 跟用户对话."
    print(f"  判定: {verdict}")
    print("=" * 88)

    print(f"\n[time] {time.time() - t0:.1f}s")
    return rows, main_ratio, footprint_ratio_2_64


if __name__ == "__main__":
    main()