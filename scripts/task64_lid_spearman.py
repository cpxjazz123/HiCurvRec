#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
task64_lid_spearman.py — G2 D0: 逐点 LID vs 每 item 量化误差 Spearman 相关

LID (Levina-Bickel MLE): d_k(x) = [(1/(k-1)) Σ_{j=1}^{k-1} log(r_k(x)/r_j(x))]^{-1}
ε_i = ||e_i - recon_i||^2 / ||e_i||^2

GO 条件: |ρ(Spearman)| > 0.3 且 p < 0.01 (item 级 n≈12K).

执行:
  python3 scripts/task64_lid_spearman.py \
      --embeddings logs/task59_s1/runs/2026-07-20/04-05-08/pickle/merged_predictions_tensor.pt \
      --rq_outputs logs/task61_s2_infer/pickle \
      --out_json verdicts/task64_lid_spearman.json
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from scipy.spatial import cKDTree
from scipy.stats import spearmanr


def compute_lid_per_point(emb_np: np.ndarray, k: int = 20) -> np.ndarray:
    """对每个点算 LID (Levina-Bickel MLE).

    r_j(x) = 距离到第 j 近邻 (j=1,...,k). d = [(1/(k-1)) Σ log(r_k/r_j)]^{-1}.
    """
    tree = cKDTree(emb_np)
    # query k+1 nearest (含 self), 跳过 j=0
    dists, _ = tree.query(emb_np, k=k + 1)
    dists = dists[:, 1:]  # (n, k) — 跳过 self
    r_k = dists[:, -1]   # 距离第 k 近邻
    log_ratios = np.log(r_k[:, None] / dists[:, :-1])  # (n, k-1)
    lids = (k - 1) / np.sum(log_ratios, axis=1)
    return lids


def compute_quantization_error(emb_np: np.ndarray, centers: np.ndarray) -> np.ndarray:
    """ε_i = ||e_i - c_{idx_i}||^2 / ||e_i||^2."""
    tree = cKDTree(centers)
    _, idx = tree.query(emb_np, k=1)
    recon = centers[idx]
    eps = np.sum((emb_np - recon) ** 2, axis=1) / np.sum(emb_np ** 2, axis=1)
    return eps


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--embeddings", nargs="+", required=True, help="一个或多个 embedding 路径")
    parser.add_argument("--rq_outputs", nargs="+", required=True,
                        help="每个 embedding 对应的 RQ 输出目录 (含 cluster_centers_layer*.npy)")
    parser.add_argument("--labels", nargs="+", default=None)
    parser.add_argument("--k_neighbors", type=int, default=20)
    parser.add_argument("--out_json", required=True)
    args = parser.parse_args()

    n_emb = len(args.embeddings)
    assert len(args.rq_outputs) == n_emb
    labels = args.labels or [f"emb{i}" for i in range(n_emb)]

    results = []
    for label, emb_path, rq_dir in zip(labels, args.embeddings, args.rq_outputs):
        print(f"\n[task64 D0] processing {label}")
        emb_t = torch.load(emb_path, map_location="cpu", weights_only=False).float()
        emb_np = emb_t.numpy()
        n, d = emb_np.shape
        print(f"  embedding: {n} × {d}")

        # 加载 RQ 各层中心
        rq_dir = Path(rq_dir)
        centers_per_layer = []
        for layer in range(3):
            c = np.load(rq_dir / f"cluster_centers_layer{layer}.npy")
            centers_per_layer.append(c)
            print(f"  layer {layer} centers: {c.shape}")

        # 总 ε_i: 用所有层中心量化后重构的总误差
        # 简单近似: 用 L1 中心量化, 看最近距离 (对应 L1 残差 → 用 L2 中心量化)
        # 这里用第 0 层中心做粗量化, 误差近似为 ε_i_layer0
        eps_0 = compute_quantization_error(emb_np, centers_per_layer[0])

        # LID
        print(f"  computing LID (k={args.k_neighbors})...")
        lids = compute_lid_per_point(emb_np, k=args.k_neighbors)
        print(f"  LID stats: mean={lids.mean():.2f}, std={lids.std():.2f}, "
              f"min={lids.min():.2f}, max={lids.max():.2f}")

        # Spearman: LID vs ε_i_layer0
        rho, p = spearmanr(lids, eps_0)
        print(f"  Spearman(LID, ε_i_layer0): ρ={rho:.4f}, p={p:.2e}")

        # 顺便: LID 分位数分层的 Recall 预测能力 (需要下游指标, 此处先报告分布)
        quantiles = np.percentile(lids, [10, 50, 90])
        print(f"  LID percentiles (10/50/90): {quantiles}")

        # 顺便: ε 分位数分层
        eps_quantiles = np.percentile(eps_0, [10, 50, 90])
        print(f"  ε_layer0 percentiles (10/50/90): {eps_quantiles}")

        results.append({
            "label": label,
            "embedding_path": emb_path,
            "n": n,
            "d": d,
            "spearman_rho_LID_vs_eps0": float(rho),
            "spearman_p_LID_vs_eps0": float(p),
            "lid_mean": float(lids.mean()),
            "lid_std": float(lids.std()),
            "lid_p10_p50_p90": [float(x) for x in quantiles],
            "eps0_p10_p50_p90": [float(x) for x in eps_quantiles],
        })

    out = Path(args.out_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump({"results": results}, f, indent=2)

    # GO/NO-GO 判定
    print("\n=== G2 D0 GO/NO-GO 判定 ===")
    for r in results:
        rho = abs(r["spearman_rho_LID_vs_eps0"])
        p = r["spearman_p_LID_vs_eps0"]
        if rho > 0.3 and p < 0.01:
            print(f"  ✅ {r['label']}: |ρ|={rho:.3f} > 0.3, p={p:.2e} < 0.01 → GO")
        elif rho > 0.15:
            print(f"  ⚠️ {r['label']}: |ρ|={rho:.3f} 在 0.15-0.3 → PARTIAL (调 k 近邻数)")
        else:
            print(f"  ❌ {r['label']}: |ρ|={rho:.3f} < 0.15 → NO-GO")

    print(f"\n[task64 D0] saved → {out}")


if __name__ == "__main__":
    main()