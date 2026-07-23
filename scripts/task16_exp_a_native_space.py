"""
Task #67 — Exp A: 未归一化原生空间下三距离等价性测试

验证 Task #62 的"距离等价性"是 L2 归一化假象，还是根本性的。
- 加载 Stage 2 RQ-VAE ckpt
- 提取 L1 残差 r_1（不归一化）
- 3 个版本对比：A1 欧氏 / A2 球面 / A3 双曲
- 计算 pairwise 最近邻一致性 + 每版本 MSE

输入: Task 62 已处理的 768-dim unit-norm embeddings
  (Stage 1 原始 2048-dim 需先经过 GRID 编码器才能用，Task 62 已封装)

启动:
  python scripts/task6_exp_a_native_space.py \
      --embeddings_pt products/task16/from_task62/item_embeddings.pt \
      --rqvae_ckpt products/task16/from_task62/rqvae_ckpt.pt \
      --out_json products/task16/from_task67/exp_a_native_space.json
"""

import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F


# ============== 距离函数 ==============

def dist_euclidean(x, c):
    """x: (B, D), c: (K, D) → (B, K)"""
    return torch.cdist(x, c, p=2)


def dist_spherical(x, c):
    """球面距离: arccos(<x, c>)，x 和 c 已在单位球上"""
    # 防止数值误差
    x = F.normalize(x, dim=-1)
    c = F.normalize(c, dim=-1)
    cos_sim = (x @ c.T).clamp(-1.0 + 1e-7, 1.0 - 1e-7)
    return torch.arccos(cos_sim)


def dist_poincare(x, c):
    """Poincaré 球距离: d(x, y) = arcosh(1 + 2||x-y||² / ((1-||x||²)(1-||y||²)))"""
    x_sqnorm = (x ** 2).sum(dim=-1, keepdim=True)  # (B, 1)
    c_sqnorm = (c ** 2).sum(dim=-1, keepdim=True).T  # (1, K)

    # diff² = ||x - c||²
    diff_sq = ((x.unsqueeze(1) - c.unsqueeze(0)) ** 2).sum(dim=-1)  # (B, K)

    # arcosh argument
    arg = 1.0 + 2.0 * diff_sq / ((1.0 - x_sqnorm) * (1.0 - c_sqnorm))
    arg = arg.clamp(min=1.0 + 1e-7)
    return torch.acosh(arg)


def nn_consistency(idx_a, idx_b):
    """两个最近邻索引张量 (B,) → 一致率 (%)"""
    assert idx_a.shape == idx_b.shape
    return (idx_a == idx_b).float().mean().item()


def compute_mse(x, indices, codebook):
    """x → 量化重建后的 MSE"""
    x_q = codebook[indices]
    return F.mse_loss(x_q, x).item()


# ============== 主实验 ==============

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--embeddings_pt", required=True,
                        help="768-dim unit-norm embeddings (Task 62 item_embeddings.pt)")
    parser.add_argument("--rqvae_ckpt", required=True)
    parser.add_argument("--out_json", required=True)
    parser.add_argument("--max_items", type=int, default=11924,
                        help="采样 item 数（默认 11924 = Toys 全集）")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print("=" * 60)
    print("Task #67 — Exp A: native space equivalence")
    print("=" * 60)

    # ============== 加载 RQ-VAE ==============
    print(f"\n[1/4] Loading RQ-VAE: {args.rqvae_ckpt}")
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from task62_train_rqvae import ResidualQuantization

    ckpt = torch.load(args.rqvae_ckpt, map_location="cpu", weights_only=False)
    model = ResidualQuantization(
        n_features=ckpt["n_features"],
        n_layers=ckpt["n_layers"],
        n_clusters=ckpt["n_clusters"],
    )
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    print(f"  layers={ckpt['n_layers']}, clusters={ckpt['n_clusters']}, "
          f"features={ckpt['n_features']}")

    # ============== 加载 embeddings ==============
    print(f"\n[2/4] Loading embeddings: {args.embeddings_pt}")
    embeddings = torch.load(args.embeddings_pt, map_location="cpu", weights_only=False)
    if embeddings.dim() > 2:
        embeddings = embeddings.squeeze()
    print(f"  raw shape: {tuple(embeddings.shape)}")
    if embeddings.shape[0] > args.max_items:
        idx = torch.randperm(embeddings.shape[0])[:args.max_items]
        embeddings = embeddings[idx]
        print(f"  sampled to: {tuple(embeddings.shape)}")

    # ============== 提取 L1 残差 ==============
    print(f"\n[3/4] Extracting L1 residual (native, not normalized)...")
    with torch.no_grad():
        # Task 62 model: forward(x) returns (z_q_list, indices_list, residuals_list, loss)
        # residuals_list[i] = residual AFTER layer i quantization
        # so residuals_list[0] = r_1 (after L1 quantization)
        _, _, residuals_list, _ = model(embeddings, update_codebook=False)
        r_1 = residuals_list[0]
    print(f"  r_1 shape: {tuple(r_1.shape)}")
    print(f"  r_1 norm stats: mean={r_1.norm(dim=-1).mean():.4f}, "
          f"std={r_1.norm(dim=-1).std():.4f}, "
          f"min={r_1.norm(dim=-1).min():.4f}, "
          f"max={r_1.norm(dim=-1).max():.4f}")

    # ============== 三版本距离 + 最近邻 + MSE ==============
    print(f"\n[4/4] Computing 3 versions of distances...")
    codebooks = [q.codebook for q in model.quantizers]  # 3 layers, each (256, 768)
    c_E = codebooks[0]  # 欧氏码本

    # ===== 版本 A1: 欧氏空间（r 不归一化）=====
    print("  [A1] Euclidean native space...")
    d_A1 = dist_euclidean(r_1, c_E)
    idx_A1 = d_A1.argmin(dim=1)
    mse_A1 = compute_mse(r_1, idx_A1, c_E)

    # ===== 版本 A2: 球面空间（r 归一化到 S^{d-1}，码本也归一化）=====
    print("  [A2] Spherical native space...")
    r_S = F.normalize(r_1, dim=-1)
    c_S = F.normalize(c_E, dim=-1)
    d_A2 = dist_spherical(r_S, c_S)
    idx_A2 = d_A2.argmin(dim=1)
    # MSE on spherical: ||r_S - c[idx]||²
    mse_A2 = compute_mse(r_S, idx_A2, c_S)

    # ===== 版本 A3: 双曲空间（r 投影到 Poincaré 球，码本也投影）=====
    print("  [A3] Poincaré native space...")
    # 投影: tanh(||x||) * x / ||x|| — 把欧氏向量压到 Poincaré 球内
    r_norm = r_1.norm(dim=-1, keepdim=True).clamp(min=1e-7)
    r_H = torch.tanh(r_norm) * (r_1 / r_norm)

    c_norm = c_E.norm(dim=-1, keepdim=True).clamp(min=1e-7)
    c_H = torch.tanh(c_norm) * (c_E / c_norm)

    d_A3 = dist_poincare(r_H, c_H)
    idx_A3 = d_A3.argmin(dim=1)
    mse_A3 = compute_mse(r_H, idx_A3, c_H)

    # ============== 一致性 + 决策 ==============
    cons_A1_A2 = nn_consistency(idx_A1, idx_A2)
    cons_A1_A3 = nn_consistency(idx_A1, idx_A3)
    cons_A2_A3 = nn_consistency(idx_A2, idx_A3)
    cons_mean = (cons_A1_A2 + cons_A1_A3 + cons_A2_A3) / 3.0

    print("\n" + "=" * 60)
    print("Exp A Results Summary")
    print("=" * 60)
    print(f"  N items:                  {r_1.shape[0]}")
    print(f"  r_1 dim:                  {r_1.shape[1]}")
    print(f"  ---")
    print(f"  A1 (Euclidean) MSE:       {mse_A1:.6f}")
    print(f"  A2 (Spherical)  MSE:      {mse_A2:.6f}")
    print(f"  A3 (Poincaré)   MSE:      {mse_A3:.6f}")
    print(f"  ---")
    print(f"  Consistency A1↔A2:        {cons_A1_A2 * 100:.2f}%")
    print(f"  Consistency A1↔A3:        {cons_A1_A3 * 100:.2f}%")
    print(f"  Consistency A2↔A3:        {cons_A2_A3 * 100:.2f}%")
    print(f"  Mean consistency:         {cons_mean * 100:.2f}%")
    print(f"  ---")
    print(f"  Decision threshold: < 70% = 打破等价")
    print(f"  Decision threshold: ≥ 70% = 仍等价")

    # Decision
    decision_equiv = "BROKEN" if cons_mean < 0.70 else "HELD"
    decision_mse = "DIFFERENT" if abs(mse_A1 - mse_A2) / max(mse_A1, 1e-10) > 0.05 else "SAME"
    print(f"  >>> DECISION (consistency): {decision_equiv}")
    print(f"  >>> DECISION (MSE):         {decision_mse}")

    # ============== 保存 ==============
    out = {
        "task": "#67 Exp A",
        "n_items": int(r_1.shape[0]),
        "r_1_dim": int(r_1.shape[1]),
        "r_1_norm_stats": {
            "mean": float(r_1.norm(dim=-1).mean()),
            "std": float(r_1.norm(dim=-1).std()),
            "min": float(r_1.norm(dim=-1).min()),
            "max": float(r_1.norm(dim=-1).max()),
        },
        "mse": {
            "A1_euclidean": mse_A1,
            "A2_spherical": mse_A2,
            "A3_poincare": mse_A3,
        },
        "consistency": {
            "A1_vs_A2": cons_A1_A2,
            "A1_vs_A3": cons_A1_A3,
            "A2_vs_A3": cons_A2_A3,
            "mean": cons_mean,
        },
        "decisions": {
            "equiv_threshold_70": decision_equiv,
            "mse_threshold_5pct": decision_mse,
        },
    }

    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n  Saved: {args.out_json}")


if __name__ == "__main__":
    main()