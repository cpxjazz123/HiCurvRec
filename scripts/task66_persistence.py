#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
task66_persistence.py — G3 D0: 子样 H_0/H_1 persistence diagram

对 embedding 与 RQ 重构后的 embedding 各采样 n_subsample 点,
算 Vietoris-Rips persistence diagram (H_0 + H_1),
用 bottleneck distance d_B 衡量拓扑破坏.

GO 条件: d_B > 自举噪声带 (5 次 bootstrap 标准差).

依赖:
  pip install ripser

执行:
  python3 scripts/task66_persistence.py \
      --embeddings logs/task59_s1/.../merged_predictions_tensor.pt \
                   logs/task61_s1/merged_predictions_2816d.pt \
      --rq_outputs logs/task59_s2_infer/pickle logs/task61_s2_infer/pickle \
      --n_subsample 2000 --n_bootstrap 5 --max_dim 1 \
      --out_json verdicts/task66_persistence.json
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np
import torch


def subsample_maxmin(emb_np: np.ndarray, n: int, seed: int = 42) -> np.ndarray:
    """maxmin sampling: 选 n 个点, 最大化最小距离."""
    rng = np.random.default_rng(seed)
    n_total = emb_np.shape[0]
    if n >= n_total:
        return np.arange(n_total)
    # init: 随机第一个
    selected = [rng.integers(0, n_total)]
    # 后续: 每步选距离已选点最小距离最大的点
    while len(selected) < n:
        dists = np.linalg.norm(emb_np[:, None] - emb_np[selected], axis=2).min(axis=1)
        # 排除已选
        dists[selected] = -1
        next_idx = int(np.argmax(dists))
        selected.append(next_idx)
    return np.array(selected)


def load_or_compute_rq_recon(emb_t: torch.Tensor, rq_dir: Path, n: int) -> np.ndarray:
    """用 RQ 各层中心重构: recon = Σ c^{(l)}_{idx^{(l)}(e_i)}.

    如果 cluster_ids.pt 不存在或形状不对, 用最近中心量化 (1 层近似).
    """
    # 加载每层中心
    centers = []
    for layer in range(3):
        c = np.load(rq_dir / f"cluster_centers_layer{layer}.npy")
        centers.append(torch.from_numpy(c).float())

    # 加载 cluster_ids (如果有)
    cluster_ids_path = rq_dir / "cluster_ids.pt"
    if cluster_ids_path.exists():
        ids = torch.load(cluster_ids_path, map_location="cpu", weights_only=False).long()
        # ids shape: (4, N) 或 (N, 4) — 按 (N, 4) 重塑
        if ids.shape[0] == 4:
            ids = ids.T  # (N, 4)
        recon = torch.zeros_like(emb_t)
        for layer in range(min(3, ids.shape[1])):
            layer_ids = ids[:, layer]
            layer_centers = centers[layer]
            recon += layer_centers[layer_ids]
        return recon.numpy()
    else:
        # fallback: 仅用 L1 中心量化
        centers_l0 = centers[0]
        dists = torch.cdist(emb_t, centers_l0)
        idx = dists.argmin(dim=1)
        return centers_l0[idx].numpy()


def compute_persistence(emb_np: np.ndarray, max_dim: int = 1) -> dict:
    """用 Ripser 算 persistence diagram."""
    try:
        from ripser import ripser
    except ImportError:
        raise ImportError("Ripser not installed. Run: pip install ripser")

    result = ripser(emb_np, maxdim=max_dim)
    # result['dgms'] is list of arrays; dgms[0] = H_0, dgms[1] = H_1, ...
    dgms = result["dgms"]
    out = {}
    for dim in range(min(len(dgms), max_dim + 1)):
        dgm = dgms[dim]
        # 过滤掉 inf (H_0 通常有一个 inf 表示无穷远合并)
        finite = dgm[np.isfinite(dgm).all(axis=1)]
        if len(finite) > 0:
            persistence = finite[:, 1] - finite[:, 0]
            out[f"H{dim}_n_features"] = int(len(finite))
            out[f"H{dim}_total_persistence"] = float(persistence.sum())
            out[f"H{dim}_max_persistence"] = float(persistence.max())
            out[f"H{dim}_mean_persistence"] = float(persistence.mean())
            out[f"H{dim}_std_persistence"] = float(persistence.std())
        else:
            out[f"H{dim}_n_features"] = 0
            out[f"H{dim}_total_persistence"] = 0.0
    return out


def bottleneck_distance(dgm1: np.ndarray, dgm2: np.ndarray) -> float:
    """Bottleneck distance d_B(Dgm1, Dgm2).

    用简单 Hungarian 匹配 (scipy.optimize.linear_sum_assignment) 近似.
    """
    from scipy.optimize import linear_sum_assignment
    from scipy.spatial.distance import cdist

    # 过滤 inf
    d1 = dgm1[np.isfinite(dgm1).all(axis=1)] if len(dgm1) > 0 else np.empty((0, 2))
    d2 = dgm2[np.isfinite(dgm2).all(axis=1)] if len(dgm2) > 0 else np.empty((0, 2))

    if len(d1) == 0 and len(d2) == 0:
        return 0.0
    if len(d1) == 0 or len(d2) == 0:
        # 一边空: 用另一边最大持久度
        only = d1 if len(d1) > 0 else d2
        pers = only[:, 1] - only[:, 0]
        return float(pers.max())

    # 对角线加 padding (经典 Bottleneck 匹配 trick: 加 (b,b) 点使两边点数相等)
    n_pad = abs(len(d1) - len(d2))
    if len(d1) < len(d2):
        # 在 d1 加 (∞, ∞) 占位 — 用 d2 的最大持久度点
        max_pers = (d2[:, 1] - d2[:, 0]).max()
        d1 = np.vstack([d1, np.tile([max_pers, max_pers], (n_pad, 1))])
    elif len(d2) < len(d1):
        max_pers = (d1[:, 1] - d1[:, 0]).max()
        d2 = np.vstack([d2, np.tile([max_pers, max_pers], (n_pad, 1))])

    # 代价矩阵: L_inf (max of |Δb|, |Δd|)
    cost = cdist(d1, d2, metric="chebyshev")
    row, col = linear_sum_assignment(cost)
    return float(cost[row, col].max())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--embeddings", nargs="+", required=True)
    parser.add_argument("--rq_outputs", nargs="+", required=True)
    parser.add_argument("--labels", nargs="+", default=None)
    parser.add_argument("--n_subsample", type=int, default=2000)
    parser.add_argument("--n_bootstrap", type=int, default=5)
    parser.add_argument("--max_dim", type=int, default=1)
    parser.add_argument("--out_json", required=True)
    args = parser.parse_args()

    n_emb = len(args.embeddings)
    assert len(args.rq_outputs) == n_emb
    labels = args.labels or [f"emb{i}" for i in range(n_emb)]

    results = []
    for label, emb_path, rq_dir in zip(labels, args.embeddings, args.rq_outputs):
        print(f"\n[task66 D0] processing {label}")
        emb_t = torch.load(emb_path, map_location="cpu", weights_only=False).float()
        emb_np = emb_t.numpy()
        n_total, d = emb_np.shape
        print(f"  shape: {n_total} × {d}")

        # 子样 + 5 次 bootstrap
        bootstrap_d_b_h0 = []
        bootstrap_d_b_h1 = []
        for b in range(args.n_bootstrap):
            print(f"  bootstrap {b+1}/{args.n_bootstrap}...")
            idx = subsample_maxmin(emb_np, args.n_subsample, seed=42 + b)
            sub_emb = emb_np[idx]
            # RQ 重构
            rq_dir_p = Path(rq_dir)
            try:
                sub_recon = load_or_compute_rq_recon(torch.from_numpy(sub_emb).float(), rq_dir_p, args.n_subsample)
            except Exception as e:
                print(f"  ⚠️ RQ recon failed: {e}; fallback to L1 only")
                centers_l0 = np.load(rq_dir_p / "cluster_centers_layer0.npy")
                from scipy.spatial import cKDTree
                tree = cKDTree(centers_l0)
                _, ci = tree.query(sub_emb, k=1)
                sub_recon = centers_l0[ci]

            # Persistence
            try:
                dgm_emb_h0 = compute_persistence(sub_emb, max_dim=0).get("H0_dgms", np.empty((0, 2)))
                dgm_emb_h1 = compute_persistence(sub_emb, max_dim=1).get("H1_dgms", np.empty((0, 2)))
                dgm_rec_h0 = compute_persistence(sub_recon, max_dim=0).get("H0_dgms", np.empty((0, 2)))
                dgm_rec_h1 = compute_persistence(sub_recon, max_dim=1).get("H1_dgms", np.empty((0, 2)))
            except Exception as e:
                print(f"  ⚠️ Ripser failed: {e}; using simplified diagram (just H_0 from MST)")
                # Simplified fallback: H_0 = 单链接合并树 (用 scipy)
                from scipy.cluster.hierarchy import linkage, fcluster
                from scipy.spatial.distance import pdist
                Z = linkage(pdist(sub_emb[:500]), method="single")
                # 简化为前 k 个合并的高度作为 H_0 diagram
                heights = Z[:50, 2]  # 前 50 个合并
                dgm_emb_h0 = np.column_stack([np.zeros(50), heights])
                dgm_rec_h0 = np.column_stack([np.zeros(50), heights])
                dgm_emb_h1 = np.empty((0, 2))
                dgm_rec_h1 = np.empty((0, 2))

            d_b_h0 = bottleneck_distance(dgm_emb_h0, dgm_rec_h0)
            d_b_h1 = bottleneck_distance(dgm_emb_h1, dgm_rec_h1)
            bootstrap_d_b_h0.append(d_b_h0)
            bootstrap_d_b_h1.append(d_b_h1)

        d_b_h0_mean = float(np.mean(bootstrap_d_b_h0))
        d_b_h0_std = float(np.std(bootstrap_d_b_h0))
        d_b_h1_mean = float(np.mean(bootstrap_d_b_h1))
        d_b_h1_std = float(np.std(bootstrap_d_b_h1))

        print(f"  d_B(H_0): mean={d_b_h0_mean:.4f}, std={d_b_h0_std:.4f}")
        print(f"  d_B(H_1): mean={d_b_h1_mean:.4f}, std={d_b_h1_std:.4f}")

        # 噪声带判定: mean > 2*std → 信号显著
        h0_signal = d_b_h0_mean > 2 * d_b_h0_std if d_b_h0_std > 0 else d_b_h0_mean > 0.01
        h1_signal = d_b_h1_mean > 2 * d_b_h1_std if d_b_h1_std > 0 else d_b_h1_mean > 0.01

        results.append({
            "label": label,
            "embedding_path": emb_path,
            "n_total": n_total,
            "d": d,
            "d_B_H0_mean": d_b_h0_mean,
            "d_B_H0_std": d_b_h0_std,
            "d_B_H1_mean": d_b_h1_mean,
            "d_B_H1_std": d_b_h1_std,
            "H0_signal_above_noise": bool(h0_signal),
            "H1_signal_above_noise": bool(h1_signal),
        })

    out = Path(args.out_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump({"results": results}, f, indent=2)

    # GO/NO-GO 判定
    print("\n=== G3 D0 GO/NO-GO 判定 ===")
    for r in results:
        h0 = r["H0_signal_above_noise"]
        h1 = r["H1_signal_above_noise"]
        if h0 or h1:
            print(f"  ✅ {r['label']}: d_B 信号 > 噪声带 (H_0={h0}, H_1={h1}) → GO")
        else:
            print(f"  ❌ {r['label']}: d_B ≤ 噪声带 (H_0={h0}, H_1={h1}) → NO-GO")

    print(f"\n[task66 D0] saved → {out}")


if __name__ == "__main__":
    main()