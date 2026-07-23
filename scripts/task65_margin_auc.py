#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
task65_margin_auc.py — G5 D0: 边界间隙场 γ̃_i + 翻码 AUC

γ_i^{(l)} = d_{(2)}(r_i^{(l)}) - d_{(1)}(r_i^{(l)})
γ̃_i^{(l)} = γ_i^{(l)} / ||r_i^{(l)}||

翻码定义: 用两个不同 seed 跑 Simple KMeans, 看 item 是否分到不同码字.
AUC = γ̃ 预测翻码的 AUC (越小 = 越靠近边界 = 越容易翻码).

GO 条件: AUC > 0.7.

执行:
  python3 scripts/task65_margin_auc.py \
      --embeddings logs/task59_s1/.../merged_predictions_tensor.pt \
                   logs/task61_s1/merged_predictions_2816d.pt \
      --seeds 42 1234 \
      --K 256 --num_hierarchies 3 \
      --out_json verdicts/task65_margin_auc.json
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score


def simple_kmeans(emb: torch.Tensor, K: int, n_iters: int = 30, seed: int = 42) -> tuple:
    g = torch.Generator(device=emb.device).manual_seed(seed)
    n, d = emb.shape
    init_idx = torch.randint(0, n, (1,), generator=g, device=emb.device).item()
    centers = [emb[init_idx].clone()]
    for _ in range(1, K):
        dists = torch.stack([torch.sum((emb - c) ** 2, dim=1) for c in centers]).min(dim=0).values
        probs = dists / dists.sum()
        idx = torch.multinomial(probs, 1, generator=g).item()
        centers.append(emb[idx].clone())
    centers = torch.stack(centers)

    for _ in range(n_iters):
        dists = torch.cdist(emb, centers)
        idx = dists.argmin(dim=1)
        new_centers = torch.zeros_like(centers)
        counts = torch.zeros(K, device=emb.device)
        new_centers.index_add_(0, idx, emb)
        counts.index_add_(0, idx, torch.ones(n, device=emb.device))
        mask = counts > 0
        new_centers[mask] = new_centers[mask] / counts[mask].unsqueeze(1)
        empty = ~mask
        if empty.any():
            for j in empty.nonzero(as_tuple=True)[0].tolist():
                new_centers[j] = emb[torch.randint(0, n, (1,), generator=g, device=emb.device).item()]
        if torch.allclose(new_centers, centers, atol=1e-6):
            centers = new_centers
            break
        centers = new_centers

    final_idx = torch.cdist(emb, centers).argmin(dim=1)
    return final_idx.cpu().numpy(), centers


def compute_margin(emb_t: torch.Tensor, centers: torch.Tensor) -> np.ndarray:
    """γ_i = d_{(2)}(e_i) - d_{(1)}(e_i), 归一化 γ̃ = γ / ||e_i||."""
    dists = torch.cdist(emb_t, centers)  # (n, K)
    sorted_dists, _ = dists.sort(dim=1)
    gamma = (sorted_dists[:, 1] - sorted_dists[:, 0]).cpu().numpy()
    emb_norm = emb_t.norm(dim=1).cpu().numpy().clip(min=1e-10)
    gamma_tilde = gamma / emb_norm
    return gamma, gamma_tilde


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--embeddings", nargs="+", required=True)
    parser.add_argument("--labels", nargs="+", default=None)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 1234])
    parser.add_argument("--K", type=int, default=256)
    parser.add_argument("--n_iters", type=int, default=30)
    parser.add_argument("--out_json", required=True)
    args = parser.parse_args()

    n_emb = len(args.embeddings)
    labels = args.labels or [f"emb{i}" for i in range(n_emb)]
    assert len(args.seeds) >= 2, "need ≥ 2 seeds to compute flip"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[task65 D0] device={device}, K={args.K}, seeds={args.seeds}")

    results = []
    for label, emb_path in zip(labels, args.embeddings):
        print(f"\n[task65 D0] processing {label} from {emb_path}")
        emb_t = torch.load(emb_path, map_location=device, weights_only=False).float()
        n, d = emb_t.shape
        print(f"  shape: {n} × {d}")

        # 对每个 seed 跑 KMeans, 算 idx + 边界间隙
        seed_data = {}
        for seed in args.seeds:
            idx, centers = simple_kmeans(emb_t, args.K, n_iters=args.n_iters, seed=seed)
            gamma, gamma_tilde = compute_margin(emb_t, centers)
            seed_data[seed] = {"idx": idx, "gamma": gamma, "gamma_tilde": gamma_tilde,
                               "centers": centers.cpu().numpy()}

        # 翻码定义: 用 seed_a vs seed_b, 看 idx 是否不同
        # 取第一对 seed (42 vs 1234) 作为主对照
        seed_a, seed_b = args.seeds[0], args.seeds[1]
        flip = (seed_data[seed_a]["idx"] != seed_data[seed_b]["idx"]).astype(int)
        flip_rate = flip.mean()
        print(f"  flip rate (seed {seed_a} vs {seed_b}): {flip_rate:.4f}")

        # γ̃ 越小 → 越靠近边界 → 越可能翻码 (负相关)
        # AUC: 用 1-γ̃ 当作"翻码概率预测", AUC 越高 = γ̃ 预测越准
        gamma_tilde_a = seed_data[seed_a]["gamma_tilde"]
        auc = roc_auc_score(flip, -gamma_tilde_a)  # 用 -γ̃ 当 score (越小越可能翻码 → score 越高)
        print(f"  AUC (γ̃ → flip): {auc:.4f}")

        # 顺便: 用原始 γ (未归一化) 算 AUC
        gamma_a = seed_data[seed_a]["gamma"]
        auc_gamma = roc_auc_score(flip, -gamma_a)
        print(f"  AUC (γ  → flip): {auc_gamma:.4f}")

        # γ̃ 分布统计
        gt_stats = {
            "mean": float(gamma_tilde_a.mean()),
            "std": float(gamma_tilde_a.std()),
            "p1": float(np.percentile(gamma_tilde_a, 1)),
            "p10": float(np.percentile(gamma_tilde_a, 10)),
            "p50": float(np.percentile(gamma_tilde_a, 50)),
        }
        print(f"  γ̃ stats: mean={gt_stats['mean']:.4f}, p1={gt_stats['p1']:.4f}, "
              f"p10={gt_stats['p10']:.4f}, p50={gt_stats['p50']:.4f}")

        results.append({
            "label": label,
            "embedding_path": emb_path,
            "n": n,
            "d": d,
            "flip_rate": float(flip_rate),
            "AUC_gamma_tilde": float(auc),
            "AUC_gamma": float(auc_gamma),
            "gamma_tilde_stats": gt_stats,
        })

    out = Path(args.out_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump({"results": results}, f, indent=2)

    # GO/NO-GO 判定
    print("\n=== G5 D0 GO/NO-GO 判定 ===")
    for r in results:
        auc = r["AUC_gamma_tilde"]
        if auc > 0.7:
            print(f"  ✅ {r['label']}: AUC={auc:.4f} > 0.7 → GO")
        elif auc > 0.6:
            print(f"  ⚠️ {r['label']}: AUC={auc:.4f} 在 0.6-0.7 → PARTIAL")
        else:
            print(f"  ❌ {r['label']}: AUC={auc:.4f} < 0.6 → NO-GO")

    print(f"\n[task65 D0] saved → {out}")


if __name__ == "__main__":
    main()