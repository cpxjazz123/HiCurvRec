#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
task59_simple_kmeans_2048d.py — Task #59 Stage 2: flan-t5 2048d + Simple KMeans SID (GPU 优化)

继承 Task #58 simple kmeans 思路, 改用 PyTorch GPU 加速避免 Python for-loop 慢问题.

执行:
  CUDA_VISIBLE_DEVICES=0 python3 scripts/task59_simple_kmeans_2048d.py \
      --input_pt logs/task59_s1/runs/<date>/<time>/pickle/merged_predictions_tensor.pt \
      --output_dir logs/task59_s2_infer/pickle \
      --K 256 --num_hierarchies 3 --max_iter 50 --seed 42
"""
from __future__ import annotations
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path("/home/wlia0047/ar57/wenyu/GeneRec")


def _kmeans_fit_gpu(X: torch.Tensor, k: int, max_iter: int, seed: int, device: str = "cuda") -> tuple[torch.Tensor, torch.Tensor]:
    """KMeans++ init + Lloyd's iterations on GPU.

    Args:
        X: (n, d) GPU tensor (float32)
        k: 码本大小
        max_iter: Lloyd 最大迭代次数
        seed: 随机种子
        device: 设备

    Returns:
        codes: (n,) int64 cluster assignments
        centers: (k, d) float32 centers
    """
    n, d = X.shape
    g = torch.Generator(device=device).manual_seed(seed)

    # k-means++ init
    idx0 = int(torch.randint(0, n, (1,), generator=g, device=device).item())
    centers = X[idx0:idx0+1].clone()
    closest_sq = torch.sum((X - centers[0]) ** 2, dim=1)
    for i in range(1, k):
        probs = closest_sq / (closest_sq.sum() + 1e-12)
        cumprobs = torch.cumsum(probs, dim=0)
        r = torch.rand(1, generator=g, device=device).item()
        idx = int(torch.searchsorted(cumprobs, torch.tensor([r], device=device)).item())
        idx = min(idx, n - 1)
        new_center = X[idx:idx+1]
        centers = torch.cat([centers, new_center], dim=0)
        new_sq = torch.sum((X - new_center[0]) ** 2, dim=1)
        closest_sq = torch.minimum(closest_sq, new_sq)

    # Lloyd's iterations on GPU
    codes = torch.zeros(n, dtype=torch.int64, device=device)
    for it in range(max_iter):
        # assignment: cdist on GPU
        dist = torch.cdist(X, centers)  # (n, k)
        codes = torch.argmin(dist, dim=1)

        # update centers with scatter_add
        new_centers = torch.zeros_like(centers)
        new_centers.index_add_(0, codes, X)
        ones = torch.ones(n, dtype=torch.int64, device=device)
        counts = torch.zeros(k, dtype=torch.int64, device=device)
        counts.index_add_(0, codes, ones)

        # handle empty clusters: reassign to a random point
        empty_mask = counts == 0
        n_empty = int(empty_mask.sum().item())
        if n_empty > 0:
            rand_idx = torch.randint(0, n, (n_empty,), generator=g, device=device)
            new_centers[empty_mask] = X[rand_idx]
            counts[empty_mask] = 1

        new_centers = new_centers / counts.unsqueeze(1).float()

        shift = float(torch.norm(new_centers - centers).item())
        centers = new_centers
        if (it + 1) % 5 == 0 or shift < 1e-5:
            print(f"  kmeans iter {it+1}, shift={shift:.6f}, empty_clusters={n_empty}", flush=True)
        if shift < 1e-5:
            print(f"  kmeans converged at iter {it+1}", flush=True)
            break

    return codes, centers


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_pt", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--K", type=int, default=256)
    parser.add_argument("--num_hierarchies", type=int, default=3)
    parser.add_argument("--max_iter", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = args.device
    print(f"[task59_kmeans] using device: {device}", flush=True)

    print(f"[task59_kmeans] loading {args.input_pt}", flush=True)
    emb = torch.load(args.input_pt, map_location="cpu", weights_only=False)
    print(f"[task59_kmeans] input shape: {tuple(emb.shape)} dtype={emb.dtype}", flush=True)
    if emb.dim() != 2:
        raise ValueError(f"expected 2D (n, d), got shape {tuple(emb.shape)}")

    n, d = emb.shape
    X_gpu = emb.float().to(device)

    # mean-center
    global_mean = X_gpu.mean(dim=0, keepdim=True)
    X_gpu = X_gpu - global_mean

    cluster_ids = torch.zeros((args.num_hierarchies, n), dtype=torch.int64, device=device)
    centers_list = []
    total_mse = 0.0

    for layer in range(args.num_hierarchies):
        print(f"[task59_kmeans] === layer {layer} ===", flush=True)
        print(f"  residual norm mean: {torch.norm(X_gpu, dim=1).mean().item():.4f}", flush=True)
        t0 = time.time()
        codes, centers = _kmeans_fit_gpu(X_gpu, args.K, args.max_iter, args.seed + layer, device)
        elapsed = time.time() - t0
        cluster_ids[layer] = codes
        centers_list.append(centers.cpu().numpy())

        X_res_rec = centers[codes]
        diff = X_gpu - X_res_rec
        mse = float((diff ** 2).mean().item())
        total_mse += mse
        cov = float(len(torch.unique(codes))) / args.K
        print(f"[task59_kmeans] layer {layer} cov={cov:.4f} ({int(len(torch.unique(codes)))}/{args.K}) MSE={mse:.6f} time={elapsed:.1f}s", flush=True)
        X_gpu = diff

    # append dedup digit
    print(f"[task59_kmeans] appending dedup digit (column {args.num_hierarchies})", flush=True)
    dedup = torch.zeros(n, dtype=torch.int64, device=device)
    cluster_ids_full = torch.cat([cluster_ids, dedup.unsqueeze(0)], dim=0)  # (4, n)

    cluster_ids_cpu = cluster_ids_full.cpu().long()
    merged_pt = cluster_ids_cpu.clone()

    torch.save(cluster_ids_cpu, out_dir / "cluster_ids.pt")
    torch.save(merged_pt, out_dir / "merged_predictions_tensor.pt")
    np.save(out_dir / "cluster_centers_layer0.npy", centers_list[0])
    np.save(out_dir / "cluster_centers_layer1.npy", centers_list[1])
    np.save(out_dir / "cluster_centers_layer2.npy", centers_list[2])

    coverage = [(layer, len(torch.unique(cluster_ids[layer])) / args.K) for layer in range(args.num_hierarchies)]

    summary = {
        "input_pt": str(args.input_pt),
        "output_dir": str(out_dir),
        "K": args.K,
        "num_hierarchies": args.num_hierarchies,
        "max_iter": args.max_iter,
        "seed": args.seed,
        "device": device,
        "n_items": n,
        "input_dim": d,
        "cov_per_layer": coverage,
        "total_residual_mse": total_mse,
        "final_residual_norm_mean": float(torch.norm(X_gpu, dim=1).mean().item()),
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"[task59_kmeans] === DONE ===", flush=True)
    print(f"  cluster_ids.pt: shape {tuple(cluster_ids_cpu.shape)}", flush=True)
    print(f"  merged_predictions_tensor.pt: shape {tuple(merged_pt.shape)}", flush=True)
    print(f"  cov: {[f'{c:.3f}' for _, c in coverage]}", flush=True)
    print(f"  total residual MSE: {total_mse:.6f}", flush=True)


if __name__ == "__main__":
    main()