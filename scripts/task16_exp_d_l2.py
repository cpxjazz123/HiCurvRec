"""
Task #71 — Exp D: 三残差 L2 量化对比

用 r_E/r_H/r_S 三个残差在同一个 L2 码本（欧氏 K-Means on L2 残差）上做 argmin，
对比：
  - q_L2 选择一致率（top-1, top-3）
  - L2 重建 MSE
  - q_L2 选择差异模式（哪些样本不同）

启动:
  python scripts/task6_exp_d_l2.py
"""

import argparse
import json
import os
import sys

import numpy as np
import torch
from sklearn.cluster import MiniBatchKMeans


def project_to_poincare(x):
    norm = x.norm(dim=-1, keepdim=True).clamp(min=1e-7)
    return torch.tanh(norm) * (x / norm)


def project_to_sphere(x):
    norm = x.norm(dim=-1, keepdim=True).clamp(min=1e-7)
    return x / norm


def poincare_log_map(q, r):
    q_sqnorm = (q ** 2).sum(dim=-1, keepdim=True).clamp(min=1e-7, max=1 - 1e-5)
    r_sqnorm = (r ** 2).sum(dim=-1, keepdim=True).clamp(min=1e-7, max=1 - 1e-5)
    inner = (q * r).sum(dim=-1, keepdim=True)
    num = (1 + 2 * inner + r_sqnorm) * q + (1 - q_sqnorm) * r
    denom = (1 + 2 * inner + q_sqnorm * r_sqnorm).clamp(min=1e-7)
    u = num / denom
    u_norm = u.norm(dim=-1, keepdim=True).clamp(min=1e-7)
    diff_sq = ((r - q) ** 2).sum(dim=-1, keepdim=True)
    arg = (1 + 2 * diff_sq / ((1 - q_sqnorm) * (1 - r_sqnorm))).clamp(min=1 + 1e-7)
    d = torch.acosh(arg)
    lambda_q = 2 / (1 - q_sqnorm)
    return (d / (lambda_q * u_norm)) * u


def spherical_log_map(q, r):
    q_norm = q.norm(dim=-1, keepdim=True).clamp(min=1e-7)
    r_norm = r.norm(dim=-1, keepdim=True).clamp(min=1e-7)
    cos_sim = (q * r).sum(dim=-1, keepdim=True) / (q_norm * r_norm)
    cos_sim = cos_sim.clamp(-1 + 1e-7, 1 - 1e-7)
    d = torch.arccos(cos_sim)
    inner = (q * r).sum(dim=-1, keepdim=True)
    proj = r - (inner / (q_norm ** 2)) * q
    proj_norm = proj.norm(dim=-1, keepdim=True).clamp(min=1e-7)
    return (d / proj_norm) * proj


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--embeddings_pt", default="products/task16/from_task62/item_embeddings.pt")
    parser.add_argument("--out_json", default="products/task16/from_task71/exp_d_l2.json")
    parser.add_argument("--n_clusters_l1", type=int, default=256)
    parser.add_argument("--n_clusters_l2", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print("=" * 60)
    print("Task #71 — Exp D: 三残差 L2 量化对比")
    print("=" * 60)

    # 加载 embeddings
    print(f"\n[1/6] Loading embeddings: {args.embeddings_pt}")
    embeddings = torch.load(args.embeddings_pt, map_location="cpu", weights_only=False)
    print(f"  shape: {tuple(embeddings.shape)}")

    # 训练 L1 码本
    print(f"\n[2/6] Training L1 MiniBatchKMeans (n_clusters={args.n_clusters_l1})...")
    emb_np = embeddings.numpy()
    km_l1 = MiniBatchKMeans(
        n_clusters=args.n_clusters_l1, random_state=args.seed,
        batch_size=1024, n_init=3, max_iter=100,
    )
    km_l1.fit(emb_np)
    q_l1_idx = km_l1.predict(emb_np)
    c_l1 = torch.tensor(km_l1.cluster_centers_, dtype=torch.float32)
    q_l1 = c_l1[torch.tensor(q_l1_idx)]
    r_1 = embeddings - q_l1
    print(f"  r_1 norm: {r_1.norm(dim=-1).mean():.4f}")

    # 训练 L2 码本（在 L1 残差上）
    print(f"\n[3/6] Training L2 MiniBatchKMeans (n_clusters={args.n_clusters_l2})...")
    r1_np = r_1.numpy()
    km_l2 = MiniBatchKMeans(
        n_clusters=args.n_clusters_l2, random_state=args.seed + 1,
        batch_size=1024, n_init=3, max_iter=100,
    )
    km_l2.fit(r1_np)
    c_l2 = torch.tensor(km_l2.cluster_centers_, dtype=torch.float32)
    print(f"  c_l2 shape: {tuple(c_l2.shape)}, norm: {c_l2.norm(dim=-1).mean():.4f}")

    # 计算三残差
    print(f"\n[4/6] Computing 3 residuals...")
    r_E = r_1
    r_H = poincare_log_map(project_to_poincare(q_l1), project_to_poincare(r_1))
    r_S = spherical_log_map(project_to_sphere(q_l1), project_to_sphere(r_1))

    # 对每个残差，用欧氏距离在 L2 码本上做 argmin
    print(f"\n[5/6] Computing L2 argmin on each residual...")
    d_E = torch.cdist(r_E, c_l2, p=2)
    d_H = torch.cdist(r_H, c_l2, p=2)
    d_S = torch.cdist(r_S, c_l2, p=2)

    q_idx_E = d_E.argmin(dim=1)
    q_idx_H = d_H.argmin(dim=1)
    q_idx_S = d_S.argmin(dim=1)

    print(f"  unique codewords used:")
    print(f"    E: {len(set(q_idx_E.tolist()))}/{args.n_clusters_l2}")
    print(f"    H: {len(set(q_idx_H.tolist()))}/{args.n_clusters_l2}")
    print(f"    S: {len(set(q_idx_S.tolist()))}/{args.n_clusters_l2}")

    # 计算 L2 重建 MSE
    q_l2_E = c_l2[q_idx_E]
    q_l2_H = c_l2[q_idx_H]
    q_l2_S = c_l2[q_idx_S]
    mse_E = ((r_E - q_l2_E) ** 2).mean().item()
    mse_H = ((r_H - q_l2_H) ** 2).mean().item()
    mse_S = ((r_S - q_l2_S) ** 2).mean().item()
    print(f"\n  L2 Reconstruction MSE:")
    print(f"    E: {mse_E:.6f}")
    print(f"    H: {mse_H:.6f}")
    print(f"    S: {mse_S:.6f}")

    # 计算 MSE 差异百分比
    mse_diff_EH = abs(mse_E - mse_H) / mse_E * 100
    mse_diff_ES = abs(mse_E - mse_S) / mse_E * 100
    mse_diff_HS = abs(mse_H - mse_S) / mse_H * 100
    print(f"\n  MSE diff:")
    print(f"    |E-H|/E: {mse_diff_EH:.2f}%")
    print(f"    |E-S|/E: {mse_diff_ES:.2f}%")
    print(f"    |H-S|/H: {mse_diff_HS:.2f}%")

    # 一致率分析
    print(f"\n[6/6] Comparing q_L2 selections...")
    match_EH = (q_idx_E == q_idx_H).float().mean().item()
    match_ES = (q_idx_E == q_idx_S).float().mean().item()
    match_HS = (q_idx_H == q_idx_S).float().mean().item()
    print(f"  q_L2 match rates:")
    print(f"    E == H: {match_EH * 100:.2f}%")
    print(f"    E == S: {match_ES * 100:.2f}%")
    print(f"    H == S: {match_HS * 100:.2f}%")

    # top-3 一致率
    top3_E = d_E.argsort(dim=1)[:, :3]
    top3_H = d_H.argsort(dim=1)[:, :3]
    top3_S = d_S.argsort(dim=1)[:, :3]
    match3_EH = (top3_E == top3_H).all(dim=1).float().mean().item()
    match3_ES = (top3_E == top3_S).all(dim=1).float().mean().item()
    match3_HS = (top3_H == top3_S).all(dim=1).float().mean().item()
    print(f"\n  top-3 match rates:")
    print(f"    E == H: {match3_EH * 100:.2f}%")
    print(f"    E == S: {match3_ES * 100:.2f}%")
    print(f"    H == S: {match3_HS * 100:.2f}%")

    # 决策
    print(f"\n{'='*60}")
    print("Decision (R4: 三残差 L2 量化是否不同?)")
    print(f"{'='*60}")
    print(f"  Match rates: EH={match_EH*100:.2f}%, ES={match_ES*100:.2f}%, HS={match_HS*100:.2f}%")
    print(f"  MSE diff:    EH={mse_diff_EH:.2f}%, ES={mse_diff_ES:.2f}%, HS={mse_diff_HS:.2f}%")

    if match_EH > 0.99 and match_ES > 0.99 and mse_diff_EH < 1 and mse_diff_ES < 1:
        decision = "R4_REJECTED (q_L2 完全相同 + MSE 差异 < 1% → 关闭 D 路线)"
    elif match_EH < 0.95 or match_ES < 0.95 or mse_diff_EH > 5 or mse_diff_ES > 5:
        decision = "R4_CONFIRMED (q_L2 差异 > 5% 或 MSE 差异 > 5% → 继续 E)"
    else:
        decision = "R4_MARGINAL (中等差异 → 继续 E 验证)"

    print(f"  Decision: {decision}")

    # 保存
    out = {
        "task": "#71 Exp D — L2 quantization on 3 residuals",
        "n_items": int(embeddings.shape[0]),
        "n_clusters_l1": args.n_clusters_l1,
        "n_clusters_l2": args.n_clusters_l2,
        "mse": {"r_E": mse_E, "r_H": mse_H, "r_S": mse_S},
        "mse_diff_pct": {"EH": mse_diff_EH, "ES": mse_diff_ES, "HS": mse_diff_HS},
        "top1_match_rate": {"EH": match_EH, "ES": match_ES, "HS": match_HS},
        "top3_match_rate": {"EH": match3_EH, "ES": match3_ES, "HS": match3_HS},
        "decision": decision,
    }
    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n  Saved: {args.out_json}")


if __name__ == "__main__":
    main()