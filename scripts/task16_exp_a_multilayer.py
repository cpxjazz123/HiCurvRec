"""
Task #69 — Exp A 扩展：L1/L2/L3 三层残差都测

扩展 Task #67 Exp A：把"只测 L1"扩展到"测 L1, L2, L3"，
验证 d_E ≈ d_H ≈ d_S 等价性是否在所有层都成立。

启动:
  python scripts/task6_exp_a_multilayer.py \
      --embeddings_pt products/task16/from_task62/item_embeddings.pt \
      --rqvae_ckpt products/task16/from_task62/rqvae_ckpt.pt \
      --out_json products/task16/from_task69/exp_a_multilayer.json
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
    return torch.cdist(x, c, p=2)


def dist_spherical(x, c):
    x = F.normalize(x, dim=-1)
    c = F.normalize(c, dim=-1)
    cos_sim = (x @ c.T).clamp(-1.0 + 1e-7, 1.0 - 1e-7)
    return torch.arccos(cos_sim)


def dist_poincare(x, c):
    x_sqnorm = (x ** 2).sum(dim=-1, keepdim=True)
    c_sqnorm = (c ** 2).sum(dim=-1, keepdim=True).T
    diff_sq = ((x.unsqueeze(1) - c.unsqueeze(0)) ** 2).sum(dim=-1)
    arg = 1.0 + 2.0 * diff_sq / ((1.0 - x_sqnorm) * (1.0 - c_sqnorm))
    arg = arg.clamp(min=1.0 + 1e-7)
    return torch.acosh(arg)


def nn_consistency(idx_a, idx_b):
    assert idx_a.shape == idx_b.shape
    return (idx_a == idx_b).float().mean().item()


def compute_mse(x, indices, codebook):
    x_q = codebook[indices]
    return F.mse_loss(x_q, x).item()


def evaluate_layer(r, c_E, layer_name):
    """对单层残差 r 评估 A1/A2/A3"""
    print(f"\n  --- {layer_name} (r shape={tuple(r.shape)}, "
          f"norm={r.norm(dim=-1).mean():.4f}) ---")

    # A1: 欧氏
    d_A1 = dist_euclidean(r, c_E)
    idx_A1 = d_A1.argmin(dim=1)
    mse_A1 = compute_mse(r, idx_A1, c_E)

    # A2: 球面
    r_S = F.normalize(r, dim=-1)
    c_S = F.normalize(c_E, dim=-1)
    d_A2 = dist_spherical(r_S, c_S)
    idx_A2 = d_A2.argmin(dim=1)
    mse_A2 = compute_mse(r_S, idx_A2, c_S)

    # A3: Poincaré
    r_norm = r.norm(dim=-1, keepdim=True).clamp(min=1e-7)
    r_H = torch.tanh(r_norm) * (r / r_norm)
    c_norm = c_E.norm(dim=-1, keepdim=True).clamp(min=1e-7)
    c_H = torch.tanh(c_norm) * (c_E / c_norm)
    d_A3 = dist_poincare(r_H, c_H)
    idx_A3 = d_A3.argmin(dim=1)
    mse_A3 = compute_mse(r_H, idx_A3, c_H)

    cons_A1_A2 = nn_consistency(idx_A1, idx_A2)
    cons_A1_A3 = nn_consistency(idx_A1, idx_A3)
    cons_A2_A3 = nn_consistency(idx_A2, idx_A3)
    cons_mean = (cons_A1_A2 + cons_A1_A3 + cons_A2_A3) / 3.0

    print(f"    MSE:       A1={mse_A1:.6f}  A2={mse_A2:.6f}  A3={mse_A3:.6f}")
    print(f"    Consist:   A1↔A2={cons_A1_A2*100:5.2f}%  "
          f"A1↔A3={cons_A1_A3*100:5.2f}%  A2↔A3={cons_A2_A3*100:5.2f}%  "
          f"mean={cons_mean*100:5.2f}%")

    return {
        "mse_A1": mse_A1,
        "mse_A2": mse_A2,
        "mse_A3": mse_A3,
        "cons_A1_A2": cons_A1_A2,
        "cons_A1_A3": cons_A1_A3,
        "cons_A2_A3": cons_A2_A3,
        "cons_mean": cons_mean,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--embeddings_pt", required=True)
    parser.add_argument("--rqvae_ckpt", required=True)
    parser.add_argument("--out_json", required=True)
    parser.add_argument("--max_items", type=int, default=11924)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print("=" * 60)
    print("Task #69 — Exp A 扩展：L1/L2/L3 三层残差")
    print("=" * 60)

    # 加载 RQ-VAE
    print(f"\n[1/3] Loading RQ-VAE: {args.rqvae_ckpt}")
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

    # 加载 embeddings
    print(f"\n[2/3] Loading embeddings: {args.embeddings_pt}")
    embeddings = torch.load(args.embeddings_pt, map_location="cpu", weights_only=False)
    if embeddings.dim() > 2:
        embeddings = embeddings.squeeze()
    if embeddings.shape[0] > args.max_items:
        idx = torch.randperm(embeddings.shape[0])[:args.max_items]
        embeddings = embeddings[idx]
    print(f"  shape: {tuple(embeddings.shape)}")

    # 提取 L1/L2/L3 残差
    print(f"\n[3/3] Extracting L1/L2/L3 residuals...")
    with torch.no_grad():
        _, _, residuals_list, _ = model(embeddings, update_codebook=False)
    print(f"  residuals_list has {len(residuals_list)} layers")

    # 评估每层
    print(f"\n{'='*60}")
    print("Multi-layer consistency analysis")
    print(f"{'='*60}")

    codebook_l1 = model.quantizers[0].codebook  # 用 L1 的码本
    results = {}
    for i, r in enumerate(residuals_list, start=1):
        results[f"L{i}"] = evaluate_layer(r, codebook_l1, f"L{i} residual")

    # 汇总
    print(f"\n{'='*60}")
    print("Summary Table")
    print(f"{'='*60}")
    print(f"{'Layer':<8}{'MSE A1':<12}{'MSE A2':<12}{'MSE A3':<12}"
          f"{'A1↔A2':<10}{'A1↔A3':<10}{'mean':<10}")
    print("-" * 70)
    for layer_name, r in results.items():
        print(f"{layer_name:<8}{r['mse_A1']:<12.6f}{r['mse_A2']:<12.6f}"
              f"{r['mse_A3']:<12.6f}{r['cons_A1_A2']*100:<10.2f}"
              f"{r['cons_A1_A3']*100:<10.2f}{r['cons_mean']*100:<10.2f}")

    # 决策
    print(f"\n{'='*60}")
    print("Decision Analysis")
    print(f"{'='*60}")
    all_broken = all(results[L]["cons_mean"] < 0.70 for L in ["L1", "L2", "L3"])
    all_held = all(results[L]["cons_mean"] >= 0.70 for L in ["L1", "L2", "L3"])
    if all_broken:
        decision = "ALL_LAYERS_BROKEN (等价性在每层都不成立)"
    elif all_held:
        decision = "ALL_LAYERS_HELD (等价性是 RQ-VAE 几何的基本特性)"
    else:
        decision = "MIXED (只在某些层成立)"
    print(f"  Decision: {decision}")

    # 保存
    out = {
        "task": "#69 Exp A multi-layer",
        "n_items": int(embeddings.shape[0]),
        "layers": list(results.keys()),
        "results": results,
        "decision": decision,
    }
    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n  Saved: {args.out_json}")


if __name__ == "__main__":
    main()