#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
task63_p1_rq_sinkhorn.py — Task #63 (G1) P1: 完整 RQ 级联 Sinkhorn-balanced Laguerre 分配

对 embedding (11924, 2048) 跑 3 层 RQ 级联, 每层 Sinkhorn-balanced k-means (eps=0.05).
测逐层 PPL, D_rel, utilization, min_count/max_count, 残差统计 (方差缩减比, 类间 δ_t).
最后产出 Stage 3 可用的 SID tensor (4, 11924) = [3 hierarchy + 1 dedup digit (去重 depth)].

预期对照 (vanilla Lloyd): 与 D0 一致 (min_count=1, max_count=220)

启动:
  CUDA_VISIBLE_DEVICES=0 python3 scripts/task63_p1_rq_sinkhorn.py \
      --input_pt /home/wlia0047/ar57/wenyu/GeneRec/logs/task59_s1/runs/2026-07-20/04-05-08/pickle/merged_predictions_tensor.pt \
      --output_dir /home/wlia0047/ar57/wenyu/GeneRec/logs/task63_p1/pickle \
      --K 256 --num_hierarchies 3 --sinkhorn_eps 0.05 --num_iters 30 --seed 42
"""
from __future__ import annotations
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

# 重用 task63 D0 的方法
import sys
sys.path.insert(0, str(Path(__file__).parent))
from task63_sinkhorn_l1 import (
    simple_kmeans_vanilla,
    simple_kmeans_sinkhorn_balanced,
    compute_metrics,
)


def compute_residual_stats(residual: np.ndarray, layer_idx: int) -> dict:
    """残差统计 — proxy for G1-H2 (残差平稳性)."""
    n, d = residual.shape
    var_total = float(np.var(residual))
    norm_per_dim = float(np.linalg.norm(residual, axis=1).mean())
    # 各维度方差分布 (depthwise variance) — 用于 detect 是否"近各向同性噪声"
    var_per_dim = np.var(residual, axis=0)  # (d,)
    cv = float(np.std(var_per_dim) / (np.mean(var_per_dim) + 1e-12))  # 变异系数
    # NN-based local variance (5-NN) — 衡量残差的"局部一致性"
    # 用近似: 残差第 i 行与近邻的方差比, 抽样 200 个 item
    rng = np.random.default_rng(42)
    sample_idx = rng.choice(n, size=min(200, n), replace=False)
    # 残差对子空间投影后的最近邻计算 (用 cKDTree 取 5-NN 局部方差比)
    from scipy.spatial import cKDTree
    tree = cKDTree(residual[sample_idx])
    local_vars = []
    for i in sample_idx:
        d_i = float(np.var(residual[i]))
        nn_dists, _ = tree.query(residual[i], k=8)
        # 5 近邻局部方差
        nn5_vars = []
        nn_idx = tree.query(residual[i], k=6)[1]
        for j in nn_idx:
            nn5_vars.append(float(np.var(residual[sample_idx[j]])))
        if nn5_vars and d_i > 0:
            local_vars.append(np.mean(nn5_vars) / (d_i + 1e-12))
    local_var_ratio = float(np.mean(local_vars)) if local_vars else 1.0

    return {
        "layer": layer_idx,
        "var_total": var_total,
        "norm_per_item_mean": norm_per_dim,
        "depthwise_cv": cv,
        "local_var_ratio_nn5": local_var_ratio,
        "residual_norm_p50": float(np.median(np.linalg.norm(residual, axis=1))),
        "residual_norm_p90": float(np.percentile(np.linalg.norm(residual, axis=1), 90)),
    }


def append_dedup_column(cluster_ids_per_layer: np.ndarray) -> np.ndarray:
    """在 RQ 级联输出的 (num_hierarchies, N) 后追加 1 列, 表示每 item 真实需要的非冗余层数.

    例如: item_i 被分配 (5, 12, 0) → 前缀 5/12 有效, 0 时停止 → dedup=2.
    即 dedup = 第一个 0 (或末尾+1 防止全非零) 的层号.
    """
    n_layers, n = cluster_ids_per_layer.shape
    dedup = np.ones(n, dtype=np.int64) * n_layers
    for i in range(n):
        for l in range(n_layers):
            if cluster_ids_per_layer[l, i] == 0:
                dedup[i] = l
                break
    return dedup


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_pt", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--K", type=int, default=256)
    parser.add_argument("--num_hierarchies", type=int, default=3)
    parser.add_argument("--sinkhorn_eps", type=float, default=0.05)
    parser.add_argument("--num_iters", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--variant", choices=["vanilla", "sinkhorn_balanced"], default="sinkhorn_balanced")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[task63 P1] loading {args.input_pt}")
    emb_t = torch.load(args.input_pt, map_location="cuda", weights_only=False).float()
    n, d = emb_t.shape
    print(f"  shape: {n} × {d}, K={args.K}, num_hierarchies={args.num_hierarchies}, eps={args.sinkhorn_eps}, variant={args.variant}")

    device = "cuda"
    emb = emb_t.to(device)

    # ============ 完整 RQ 级联 ============
    residual = emb.clone()
    layer_centers = []
    layer_idx = []
    layer_metrics = []
    layer_stats = []

    for l in range(args.num_hierarchies):
        t0 = time.time()
        print(f"\n[task63 P1] layer {l}: residual shape {residual.shape}")

        if args.variant == "vanilla":
            idx_l, c_l = simple_kmeans_vanilla(residual, args.K, n_iters=args.num_iters, seed=args.seed)
        else:
            idx_l, c_l = simple_kmeans_sinkhorn_balanced(
                residual, args.K, n_iters=args.num_iters,
                sinkhorn_eps=args.sinkhorn_eps, seed=args.seed
            )

        idx_t = torch.from_numpy(idx_l).to(device)
        c_t = torch.from_numpy(c_l).to(device)
        # 残差更新: r <- r - c[idx]
        recon_l = c_t[idx_t]
        new_residual = residual - recon_l

        # 当前层指标 (在原始输入空间) & 残差统计
        m = compute_metrics(idx_l, c_l, residual.cpu().numpy())
        s = compute_residual_stats(new_residual.cpu().numpy(), l)
        m["time_sec"] = time.time() - t0
        m["var_before"] = float(np.var(residual.cpu().numpy()))
        m["var_after"] = float(np.var(new_residual.cpu().numpy()))
        m["variance_reduction_ratio"] = (
            1.0 - m["var_after"] / (m["var_before"] + 1e-12)
        )

        layer_idx.append(idx_t)
        layer_centers.append(c_t)
        layer_metrics.append(m)
        layer_stats.append(s)

        print(f"  PPL={m['PPL']:.2f}, D_rel={m['D_rel']:.6f}, util={m['utilization']:.4f}, min_count={m['min_count']}, max_count={m['max_count']}")
        print(f"  var_before={m['var_before']:.4f}, var_after={m['var_after']:.4f}, var_red={m['variance_reduction_ratio']*100:.2f}%")
        print(f"  local_var_ratio={s['local_var_ratio_nn5']:.3f}, depthwise_cv={s['depthwise_cv']:.3f}")
        print(f"  time={m['time_sec']:.1f}s")

        residual = new_residual

    # ============ 拼接 SID + dedup digit ============
    cluster_ids_tensor = torch.stack(layer_idx, dim=0).cpu()  # (num_hierarchies, n)
    cluster_ids_np = cluster_ids_tensor.numpy()

    # dedup digit
    dedup = append_dedup_column(cluster_ids_np)  # (n,)
    sid_full = np.concatenate([cluster_ids_np, dedup[None, :]], axis=0)  # (num_hierarchies+1, n)
    sid_tensor = torch.from_numpy(sid_full).long()

    # ============ 保存 ============
    for l, c in enumerate(layer_centers):
        np.save(out_dir / f"cluster_centers_l{l}_{args.variant}.npy", c.cpu().numpy())
        np.save(out_dir / f"cluster_idx_l{l}_{args.variant}.npy", layer_idx[l].cpu().numpy())
    torch.save(sid_tensor, out_dir / f"sid_{args.variant}.pt")  # Stage 3 输入
    np.save(out_dir / f"sid_{args.variant}_dedup.npy", dedup)

    # ============ G1-H2 V-information profile proxy ============
    # 测逐层 ε_layer: 每 item 在层 l 的相对失真 (作为 V-information proxy)
    # (近似: ε_layer_l_i = ||r^(l) - c^(l)||^2 / ||e||^2)
    orig_norm_sq = np.sum(emb_t.cpu().numpy() ** 2, axis=1)  # (n,)
    eps_per_item_per_layer = []
    recon_cur = np.zeros_like(emb_t.cpu().numpy())
    for l in range(args.num_hierarchies):
        c_l = layer_centers[l].cpu().numpy()
        recon_cur = recon_cur + c_l[layer_idx[l].cpu().numpy()]
        r_l = emb_t.cpu().numpy() - recon_cur
        eps_l = np.sum(r_l ** 2, axis=1) / (orig_norm_sq + 1e-12)
        eps_per_item_per_layer.append(eps_l)
    eps_tensor = np.stack(eps_per_item_per_layer, axis=0)  # (num_hierarchies, n)

    # 残差层间相关性 (跨层 ε 的 Pearson) — G1-H2: Sinkhorn vs Vanilla 应让深层 ε 更难预测浅层 (de-concentration)
    eps_corr_matrix = np.corrcoef(eps_tensor)
    print(f"\n=== G1-H2 V-information proxy (逐层 ε 相关矩阵) ===")
    print(f"shape: {eps_corr_matrix.shape}")
    print(f"ε_layer_corr:\n{eps_corr_matrix}")

    # ============ 汇总 ============
    summary = {
        "input_pt": args.input_pt,
        "n_items": n, "d": d,
        "K": args.K, "num_hierarchies": args.num_hierarchies,
        "sinkhorn_eps": args.sinkhorn_eps,
        "variant": args.variant,
        "layer_metrics": layer_metrics,
        "layer_residual_stats": layer_stats,
        "eps_per_layer": {
            f"layer_{l}": {
                "p10": float(np.percentile(eps_tensor[l], 10)),
                "p50": float(np.percentile(eps_tensor[l], 50)),
                "p90": float(np.percentile(eps_tensor[l], 90)),
                "mean": float(np.mean(eps_tensor[l])),
            }
            for l in range(args.num_hierarchies)
        },
        "epsilon_layer_correlation": eps_corr_matrix.tolist(),
        "dedup_distribution": {
            "min": int(dedup.min()), "max": int(dedup.max()),
            "mean": float(dedup.mean()), "median": float(np.median(dedup)),
            "hist": np.bincount(dedup, minlength=args.num_hierarchies + 1).tolist(),
        },
    }
    with open(out_dir / f"summary_{args.variant}.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n=== TASK #63 P1 SUMMARY ({args.variant}) ===")
    print(f"  per-layer PPL: {[m['PPL'] for m in layer_metrics]}")
    print(f"  per-layer D_rel: {[m['D_rel'] for m in layer_metrics]}")
    print(f"  per-layer min_count: {[m['min_count'] for m in layer_metrics]}")
    print(f"  per-layer var_reduction_ratio: {[m['variance_reduction_ratio'] for m in layer_metrics]}")
    print(f"  dedup histogram: {summary['dedup_distribution']['hist']}")
    print(f"  saved → {out_dir}")


if __name__ == "__main__":
    main()
