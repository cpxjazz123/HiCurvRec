#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
task63_sinkhorn_l1.py — G1 D0: L1 Sinkhorn-balanced k-means vs vanilla k-means

在半离散 OT 框架下用 Sinkhorn 算法让每码字质量 m_k ≈ 1/K,
对比 vanilla Lloyd k-means 的 PPL (perplexity) 与 D_rel.

PPL = exp(-Σ p_k log p_k), p_k = fraction of items in codebook k.
Perfect balance → p_k = 1/K → PPL = K.
Vanilla k-means 通常 PPL << K (大量 m_k → 0).

GO 条件: PPL ≥ 0.95K 且 D_rel 劣化 < 10%.

执行:
  python3 scripts/task63_sinkhorn_l1.py \
      --input_pt logs/task59_s1/runs/2026-07-20/04-05-08/pickle/merged_predictions_tensor.pt \
      --output_dir logs/task63_s2_d0/pickle \
      --K 256 --sinkhorn_eps 0.05 --num_iters 50 --seed 42
"""
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch


def simple_kmeans_vanilla(emb: torch.Tensor, K: int, n_iters: int = 50, seed: int = 42) -> tuple:
    """标准 Lloyd k-means (Voronoi 分配, 无质量约束)."""
    g = torch.Generator(device=emb.device).manual_seed(seed)
    n, d = emb.shape
    # k-means++
    init_idx = torch.randint(0, n, (1,), generator=g, device=emb.device).item()
    centers = [emb[init_idx].clone()]
    for _ in range(1, K):
        dists = torch.stack([torch.sum((emb - c) ** 2, dim=1) for c in centers]).min(dim=0).values
        probs = dists / dists.sum()
        idx = torch.multinomial(probs, 1, generator=g).item()
        centers.append(emb[idx].clone())
    centers = torch.stack(centers)

    for it in range(n_iters):
        dists = torch.cdist(emb, centers)
        idx = dists.argmin(dim=1)
        new_centers = torch.zeros_like(centers)
        counts = torch.zeros(K, device=emb.device)
        new_centers.index_add_(0, idx, emb)
        counts.index_add_(0, idx, torch.ones(n, device=emb.device))
        mask = counts > 0
        new_centers[mask] = new_centers[mask] / counts[mask].unsqueeze(1)
        # 重置空 cluster
        empty = ~mask
        if empty.any():
            for j in empty.nonzero(as_tuple=True)[0].tolist():
                new_centers[j] = emb[torch.randint(0, n, (1,), generator=g, device=emb.device).item()]
        if torch.allclose(new_centers, centers, atol=1e-6):
            centers = new_centers
            break
        centers = new_centers

    final_idx = torch.cdist(emb, centers).argmin(dim=1)
    return final_idx.cpu().numpy(), centers.cpu().numpy()


def simple_kmeans_sinkhorn_balanced(emb: torch.Tensor, K: int, n_iters: int = 50,
                                     sinkhorn_eps: float = 0.05, seed: int = 42) -> tuple:
    """Sinkhorn-balanced k-means: 交替 (Sinkhorn 分配 → Lloyd 更新中心).

    每步:
      1. 算 pairwise cost C[i,k] = ||e_i - c_k||^2
      2. Sinkhorn: T = exp(-C/eps), 反复行/列归一化直到边际接近 (1/N, 1/K)
      3. 硬分配: idx_i = argmax_k T[i,k]
      4. 更新中心: c_k = mean(emb[idx==k])
    """
    g = torch.Generator(device=emb.device).manual_seed(seed)
    n, d = emb.shape
    # k-means++ init (同 vanilla)
    init_idx = torch.randint(0, n, (1,), generator=g, device=emb.device).item()
    centers = [emb[init_idx].clone()]
    for _ in range(1, K):
        dists = torch.stack([torch.sum((emb - c) ** 2, dim=1) for c in centers]).min(dim=0).values
        probs = dists / dists.sum()
        idx = torch.multinomial(probs, 1, generator=g).item()
        centers.append(emb[idx].clone())
    centers = torch.stack(centers)

    for it in range(n_iters):
        # Sinkhorn 分配
        C = torch.cdist(emb, centers) ** 2  # (n, K)
        T = torch.exp(-C / sinkhorn_eps)
        # 行/列归一化 (目标 marginal: 1/n, 1/K)
        for _ in range(20):
            T = T / T.sum(dim=1, keepdim=True).clamp(min=1e-30) * 1.0  # row marginal 1/n → 实际乘以 1
            T = T / T.sum(dim=0, keepdim=True).clamp(min=1e-30) * (n / K)  # col marginal n/K

        idx = T.argmax(dim=1)
        # 更新中心 (与 vanilla 相同)
        new_centers = torch.zeros_like(centers)
        counts = torch.zeros(K, device=emb.device)
        new_centers.index_add_(0, idx, emb)
        counts.index_add_(0, idx, torch.ones(n, device=emb.device))
        mask = counts > 0
        new_centers[mask] = new_centers[mask] / counts[mask].unsqueeze(1)
        # 重置空 cluster
        empty = ~mask
        if empty.any():
            for j in empty.nonzero(as_tuple=True)[0].tolist():
                new_centers[j] = emb[torch.randint(0, n, (1,), generator=g, device=emb.device).item()]
        if torch.allclose(new_centers, centers, atol=1e-6):
            centers = new_centers
            break
        centers = new_centers

    final_idx = torch.cdist(emb, centers).argmin(dim=1)
    return final_idx.cpu().numpy(), centers.cpu().numpy()


def compute_metrics(idx: np.ndarray, centers: np.ndarray, emb_np: np.ndarray) -> dict:
    n, d = emb_np.shape
    K = centers.shape[0]
    counts = np.bincount(idx, minlength=K)
    p_k = counts / n
    # PPL = exp(-Σ p_k log p_k), p_k=0 时该项 = 0
    nonzero = p_k > 0
    ppl = float(np.exp(-np.sum(p_k[nonzero] * np.log(p_k[nonzero]))))
    # D_rel = mean ||e - c||^2 / ||e||^2
    recon = centers[idx]
    d_rel = float(np.mean(np.sum((emb_np - recon) ** 2, axis=1) / np.sum(emb_np ** 2, axis=1)))
    # 利用率: 至少 1 个 item 的码字数 / K
    utilization = float((counts > 0).sum() / K)
    return {"PPL": ppl, "D_rel": d_rel, "utilization": utilization,
            "min_count": int(counts.min()), "max_count": int(counts.max())}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_pt", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--K", type=int, default=256)
    parser.add_argument("--sinkhorn_eps", type=float, default=0.05)
    parser.add_argument("--num_iters", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[task63 D0] loading {args.input_pt}")
    emb_t = torch.load(args.input_pt, map_location="cuda" if torch.cuda.is_available() else "cpu",
                       weights_only=False).float()
    emb_np = emb_t.cpu().numpy()
    n, d = emb_np.shape
    print(f"  shape: {n} × {d}, K={args.K}, sinkhorn_eps={args.sinkhorn_eps}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    emb = emb_t.to(device)

    # Vanilla
    print(f"\n[task63 D0] vanilla k-means (Lloyd)...")
    idx_v, centers_v = simple_kmeans_vanilla(emb, args.K, n_iters=args.num_iters, seed=args.seed)
    m_v = compute_metrics(idx_v, centers_v, emb_np)
    print(f"  PPL={m_v['PPL']:.2f}  D_rel={m_v['D_rel']:.6f}  util={m_v['utilization']:.4f}")

    # Sinkhorn-balanced
    print(f"\n[task63 D0] Sinkhorn-balanced k-means (eps={args.sinkhorn_eps})...")
    idx_s, centers_s = simple_kmeans_sinkhorn_balanced(emb, args.K, n_iters=args.num_iters,
                                                        sinkhorn_eps=args.sinkhorn_eps, seed=args.seed)
    m_s = compute_metrics(idx_s, centers_s, emb_np)
    print(f"  PPL={m_s['PPL']:.2f}  D_rel={m_s['D_rel']:.6f}  util={m_s['utilization']:.4f}")

    # 对比
    d_rel_increase = (m_s['D_rel'] - m_v['D_rel']) / m_v['D_rel']
    ppl_increase = (m_s['PPL'] - m_v['PPL']) / m_v['PPL']

    # 保存
    np.save(out_dir / "cluster_centers_vanilla.npy", centers_v)
    np.save(out_dir / "cluster_centers_sinkhorn.npy", centers_s)
    np.save(out_dir / "cluster_idx_vanilla.npy", idx_v)
    np.save(out_dir / "cluster_idx_sinkhorn.npy", idx_s)

    summary = {
        "input_pt": args.input_pt,
        "K": args.K,
        "sinkhorn_eps": args.sinkhorn_eps,
        "vanilla": m_v,
        "sinkhorn_balanced": m_s,
        "D_rel_increase_pct": d_rel_increase * 100,
        "PPL_increase_pct": ppl_increase * 100,
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # GO/NO-GO 判定
    print("\n=== G1 D0 GO/NO-GO 判定 ===")
    print(f"  Sinkhorn PPL ({m_s['PPL']:.1f}) vs K ({args.K}): ratio = {m_s['PPL']/args.K:.4f}")
    print(f"  Sinkhorn utilization: {m_s['utilization']:.4f}")
    print(f"  D_rel increase: {d_rel_increase*100:+.2f}%")
    print(f"  PPL increase:   {ppl_increase*100:+.2f}%")

    if m_s['PPL'] / args.K >= 0.95 and d_rel_increase < 0.10:
        print("  ✅ GO: PPL 接近 K 且 D_rel 劣化 < 10%")
    elif d_rel_increase >= 0.30:
        print("  ❌ NO-GO: D_rel 劣化过大 (>= 30%)")
    else:
        print(f"  ⚠️ PARTIAL: PPL/利用率达标但失真税 {d_rel_increase*100:.1f}% 偏高, 调 sinkhorn_eps")

    print(f"\n[task63 D0] saved → {out_dir}")


if __name__ == "__main__":
    main()