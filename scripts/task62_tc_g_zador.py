#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
task62_tc_g_zador.py — G4 D0: Gaussian 总相关 TC_G + Zador 指数

对每份 embedding 计算:
- 全维协方差 Σ (d × d)
- 块对角协方差 Σ_s (按 S 段分, 这里用 axis-分块, S=2)
- TC_G = 0.5 * log(det(ΠΣ_s) / det(Σ))
- Zador 指数拟合: log D vs log K 斜率 → d_eff = -2/s
- L2 norm 分布直方图 + tail ratio (p99/p50)

执行:
  python3 scripts/task62_tc_g_zador.py \
      --embeddings logs/task59_s1/runs/2026-07-20/04-05-08/pickle/merged_predictions_tensor.pt \
                   logs/task60_s1/runs/2026-07-20/05-53-06/pickle/merged_predictions_tensor.pt \
                   logs/task61_s1/merged_predictions_2816d.pt \
      --labels flan-t5 sentence-t5 hybrid2816 \
      --out_csv verdicts/task62_tc_zador.csv
"""
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch


def compute_tc_g(emb: np.ndarray, n_segments: int = 2, jitter: float = 1e-3) -> float:
    """TC_G = 0.5 * log(det(ΠΣ_s) / det(Σ)).

    把 d 维 embedding 按 axis 切成 n_segments 段, 每段独立拟合高斯.
    Σ 是全维协方差, Σ_s 是各段协方差. TC_G 衡量分段独立编码 vs 全维编码的损失税.
    """
    n, d = emb.shape
    seg_size = d // n_segments
    eps = jitter * np.trace(np.cov(emb.T)) / d * np.eye(d)

    sigma_full = np.cov(emb.T) + eps
    sign_full, logdet_full = np.linalg.slogdet(sigma_full)

    # 分段: 每段独立协方差 + 块对角 det = Π det(Σ_s)
    seg_logdets = []
    for s in range(n_segments):
        start = s * seg_size
        end = (s + 1) * seg_size if s < n_segments - 1 else d
        seg = emb[:, start:end]
        sigma_s = np.cov(seg.T) + jitter * np.eye(end - start)
        sign_s, logdet_s = np.linalg.slogdet(sigma_s)
        if sign_s <= 0:
            return float('inf')
        seg_logdets.append(logdet_s)

    logdet_diag = sum(seg_logdets)
    tc_g = 0.5 * (logdet_diag - logdet_full)
    return tc_g


def compute_zador_exponent(emb: np.ndarray, ks: list = None) -> dict:
    """拟合 D(K) ~ K^(-2/d_eff), 估计 d_eff.

    用 Simple KMeans 在不同 K 上跑, 量 D_rel = mean ||e - c||^2 / ||e||^2.
    """
    if ks is None:
        ks = [16, 32, 64, 128, 256]

    results = []
    for k in ks:
        # GPU KMeans via k-means++ + Lloyd
        idx, centers = simple_kmeans_gpu(emb, k, n_iters=20, seed=42)
        recon = centers[idx]
        d_rel = float(np.mean(np.sum((emb - recon) ** 2, axis=1) / np.sum(emb ** 2, axis=1)))
        results.append((k, d_rel))

    log_ks = np.log([k for k, _ in results])
    log_ds = np.log([d for _, d in results])
    slope, _ = np.polyfit(log_ks, log_ds, 1)
    d_eff = -2.0 / slope
    return {"ks": results, "slope": float(slope), "d_eff": float(d_eff)}


def simple_kmeans_gpu(emb_np: np.ndarray, k: int, n_iters: int = 20, seed: int = 42) -> tuple:
    """Simple KMeans via torch on GPU."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    emb = torch.from_numpy(emb_np).float().to(device)
    n, d = emb.shape

    g = torch.Generator(device=device).manual_seed(seed)
    # k-means++
    init_idx = torch.randint(0, n, (1,), generator=g, device=device).item()
    centers = [emb[init_idx]]
    for _ in range(1, k):
        dists = torch.stack([torch.sum((emb - c) ** 2, dim=1) for c in centers]).min(dim=0).values
        probs = dists / dists.sum()
        idx = torch.multinomial(probs, 1, generator=g).item()
        centers.append(emb[idx])
    centers = torch.stack(centers)

    for _ in range(n_iters):
        dists = torch.cdist(emb, centers)
        idx = dists.argmin(dim=1)
        new_centers = torch.zeros_like(centers)
        counts = torch.zeros(k, device=device)
        new_centers.index_add_(0, idx, emb)
        counts.index_add_(0, idx, torch.ones(n, device=device))
        mask = counts > 0
        new_centers[mask] = new_centers[mask] / counts[mask].unsqueeze(1)
        # 重新初始化空 cluster
        empty = ~mask
        if empty.any():
            for j in empty.nonzero(as_tuple=True)[0].tolist():
                new_centers[j] = emb[torch.randint(0, n, (1,), generator=g).item()]
        if torch.allclose(new_centers, centers, atol=1e-6):
            centers = new_centers
            break
        centers = new_centers

    final_idx = torch.cdist(emb, centers).argmin(dim=1).cpu().numpy()
    return final_idx, centers.cpu().numpy()


def compute_norm_tail(emb: np.ndarray) -> dict:
    """norm 分布: p50, p99, tail_ratio = p99 / p50."""
    norms = np.linalg.norm(emb, axis=1)
    return {
        "p50": float(np.percentile(norms, 50)),
        "p99": float(np.percentile(norms, 99)),
        "max": float(norms.max()),
        "tail_ratio_p99_p50": float(np.percentile(norms, 99) / np.percentile(norms, 50)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--embeddings", nargs="+", required=True)
    parser.add_argument("--labels", nargs="+", default=None)
    parser.add_argument("--n_segments", type=int, default=2)
    parser.add_argument("--out_csv", required=True)
    args = parser.parse_args()

    n_emb = len(args.embeddings)
    labels = args.labels or [f"emb{i}" for i in range(n_emb)]

    rows = []
    for label, path in zip(labels, args.embeddings):
        print(f"[task62] processing {label} from {path}")
        emb_t = torch.load(path, map_location="cpu", weights_only=False).float()
        emb_np = emb_t.numpy()
        n, d = emb_np.shape
        print(f"  shape: {n} × {d}")

        # TC_G
        tc_g = compute_tc_g(emb_np, n_segments=args.n_segments)
        print(f"  TC_G (S={args.n_segments}): {tc_g:.6f}")

        # Zador 指数 (在 d ≤ 1024 上跑得动; d=2816 时 K=256 较慢但可接受)
        if d <= 1024:
            zador = compute_zador_exponent(emb_np, ks=[16, 32, 64, 128, 256])
            print(f"  Zador d_eff: {zador['d_eff']:.2f}")
        else:
            zador = {"ks": [], "slope": float('nan'), "d_eff": float('nan')}
            print(f"  Zador skipped (d={d} > 1024)")

        # norm tail
        norm_stats = compute_norm_tail(emb_np)
        print(f"  norm tail ratio p99/p50: {norm_stats['tail_ratio_p99_p50']:.4f}")

        rows.append({
            "label": label,
            "path": path,
            "n": n,
            "d": d,
            "TC_G": tc_g,
            "Zador_d_eff": zador['d_eff'],
            "Zador_slope": zador['slope'],
            "norm_p50": norm_stats['p50'],
            "norm_p99": norm_stats['p99'],
            "norm_max": norm_stats['max'],
            "norm_tail_ratio": norm_stats['tail_ratio_p99_p50'],
        })

    # 写 CSV
    out = Path(args.out_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    import csv
    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"[task62] saved → {out}")

    # 总结 GO/NO-GO 决策 (按 G4-P1: TC_G 与 SCR 单调相关)
    print("\n=== G4-P1 GO/NO-GO 判定 ===")
    print(f"{'label':<15} {'d':<6} {'TC_G':<10} {'norm_tail':<10}")
    for r in rows:
        print(f"{r['label']:<15} {r['d']:<6} {r['TC_G']:<10.4f} {r['norm_tail_ratio']:<10.4f}")
    print("\n预期: SCR 高 (病态) → TC_G 高 + norm_tail 高; SCR 低 (健康) → TC_G 低 + norm_tail 低")
    print("SCR 参考: T5=0.95x (健康), MCKG=4.22x (病态). 当前缺 MCKG 实测, 但可用 norm_tail_ratio 代理.")


if __name__ == "__main__":
    main()