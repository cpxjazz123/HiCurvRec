"""
Task #70 — 多几何根因诊断包：D1 + D4 + D5

D1: 距离函数本身的单调性分析（纯数学，与 RQ-VAE 无关）
D4: 实际 RQ-VAE 残差上 d_E vs d_H 的全排序 vs top1 一致率
D5: 距离数值差异分布 Δ = |d_E - d_H|

启动:
  python scripts/task6_d1_d4_d5.py
"""

import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F


# ============== Exp D1：函数单调性 ==============

def d_euclidean_unit(cos_val):
    """假设 ||x||=||y||=1: d_E = √(2 - 2·cos)"""
    return np.sqrt(np.maximum(2 - 2 * cos_val, 0))


def d_poincare_unit(cos_val, eps=1e-7):
    """Poincaré 距离，假设 ||x||=||y||=r
    d_H = arcosh(1 + 2(1-cos)·r² / ((1-r²)))
    当 r=||x||=||y||=1 时，公式简化为 arcosh(1 + (1-cos)²·2/(0))  → 无穷大
    所以必须假设 r < 1，这里取 r=sin(arccos(cos_val)) 即 cos=⟨x,y⟩/||x||·||y||
    实际：当 ||x||=||y||=r, <x,y>=cos·r², <x,y>/r²=cos, 1-||x||²=1-r²
    d_H = arcosh(1 + 2||x-y||² / ((1-||x||²)(1-||y||²)))
        = arcosh(1 + 2·r²·(1-cos)·2 / ((1-r²)²))  -- 错
        ||x-y||² = ||x||² + ||y||² - 2<x,y> = 2r² - 2r²·cos = 2r²(1-cos)
    d_H = arcosh(1 + 2·2r²(1-cos) / (1-r²)²)
    """
    r = 0.9  # 假设单位球内 (避免 1-||x||²=0)
    return np.arccosh(1 + 2 * 2 * r**2 * (1 - cos_val) / (1 - r**2)**2)


def exp_d1():
    print("=" * 60)
    print("Exp D1: distance function monotonicity analysis")
    print("=" * 60)

    cos_vals = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.99])
    d_E = d_euclidean_unit(cos_vals)
    d_H = d_poincare_unit(cos_vals)

    print(f"\n  cos_val     d_E         d_H")
    print(f"  --------  ----------  ----------")
    for c, e, h in zip(cos_vals, d_E, d_H):
        print(f"  {c:.2f}      {e:.6f}    {h:.6f}")

    # 单调性检查
    E_monotonic_dec = all(d_E[i] > d_E[i+1] for i in range(len(d_E)-1))
    H_monotonic_dec = all(d_H[i] > d_H[i+1] for i in range(len(d_H)-1))
    print(f"\n  d_E monotonic decreasing: {E_monotonic_dec}")
    print(f"  d_H monotonic decreasing: {H_monotonic_dec}")

    # 排序一致检查（按 cos 升序 → d 应降序）
    # 如果两者都单调降序，则排序与 cos 排序一致 → 等价
    sort_E = np.argsort(-d_E)  # 距离降序（最大值在前）
    sort_H = np.argsort(-d_H)
    rank_match = np.array_equal(sort_E, sort_H)
    print(f"  Sort by d_E: {sort_E.tolist()}")
    print(f"  Sort by d_H: {sort_H.tolist()}")
    print(f"  Rank identical: {rank_match}")

    # Pearson correlation
    pearson = np.corrcoef(d_E, d_H)[0, 1]
    print(f"  Pearson correlation d_E vs d_H: {pearson:.6f}")

    return {
        "cos_vals": cos_vals.tolist(),
        "d_E": d_E.tolist(),
        "d_H": d_H.tolist(),
        "E_monotonic_dec": bool(E_monotonic_dec),
        "H_monotonic_dec": bool(H_monotonic_dec),
        "rank_identical": bool(rank_match),
        "pearson": float(pearson),
    }


# ============== Exp D4 + D5：实际残差数据 ==============

def dist_euclidean_torch(x, c):
    return torch.cdist(x, c, p=2)


def dist_poincare_torch(x, c):
    x_sqnorm = (x ** 2).sum(dim=-1, keepdim=True)
    c_sqnorm = (c ** 2).sum(dim=-1, keepdim=True).T
    diff_sq = ((x.unsqueeze(1) - c.unsqueeze(0)) ** 2).sum(dim=-1)
    arg = 1.0 + 2.0 * diff_sq / ((1.0 - x_sqnorm) * (1.0 - c_sqnorm))
    arg = arg.clamp(min=1.0 + 1e-7)
    return torch.acosh(arg)


def project_to_poincare(x):
    """tanh(||x||) · x/||x|| 投影到 Poincaré 球"""
    norm = x.norm(dim=-1, keepdim=True).clamp(min=1e-7)
    return torch.tanh(norm) * (x / norm)


def exp_d4_d5():
    print(f"\n{'='*60}")
    print("Exp D4 + D5: real RQ-VAE residual analysis")
    print(f"{'='*60}")

    # 加载 RQ-VAE
    print(f"\n  Loading Task 62 RQ-VAE...")
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from task62_train_rqvae import ResidualQuantization

    ckpt = torch.load("products/task16/from_task62/rqvae_ckpt.pt",
                       map_location="cpu", weights_only=False)
    model = ResidualQuantization(
        n_features=ckpt["n_features"],
        n_layers=ckpt["n_layers"],
        n_clusters=ckpt["n_clusters"],
    )
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    # 加载 embeddings
    embeddings = torch.load("products/task16/from_task62/item_embeddings.pt",
                             map_location="cpu", weights_only=False)
    print(f"  embeddings: {tuple(embeddings.shape)}")

    # 提取 L1 残差
    with torch.no_grad():
        _, _, residuals_list, _ = model(embeddings, update_codebook=False)
    r_1 = residuals_list[0]
    print(f"  r_1: {tuple(r_1.shape)}, norm={r_1.norm(dim=-1).mean():.4f}")

    # 用 L1 码本（与 Task #67 一致）
    c_E = model.quantizers[0].codebook

    # Poincaré 投影
    r_H = project_to_poincare(r_1)
    c_H = project_to_poincare(c_E)

    print(f"  Computing distance matrices...")
    d_E = dist_euclidean_torch(r_1, c_E)   # (11924, 256)
    d_H = dist_poincare_torch(r_H, c_H)    # (11924, 256)
    print(f"  d_E shape: {tuple(d_E.shape)}")
    print(f"  d_H shape: {tuple(d_H.shape)}")

    # ===== Exp D4: 排序一致 =====
    # 修复：原代码用 argsort()[:, 0] 在 ties 时不可靠（PyTorch argsort 不稳定）
    # 改用 argmin 保证与 Task #67 一致
    print(f"\n  --- Exp D4: sort consistency ---")
    idx_E = d_E.argmin(dim=1)  # (B,) 最近邻索引（argmin ties-stable）
    idx_H = d_H.argmin(dim=1)
    top1_match = (idx_E == idx_H)  # (B,) bool

    # 全排序一致用 stable sort
    sort_E = torch.argsort(d_E, dim=1, stable=True)
    sort_H = torch.argsort(d_H, dim=1, stable=True)
    full_match = (sort_E == sort_H).all(dim=1)  # (B,) bool
    top3_match = (sort_E[:, :3] == sort_H[:, :3]).all(dim=1)  # (B,) bool

    full_match_rate = full_match.float().mean().item()
    top1_match_rate = top1_match.float().mean().item()
    top3_match_rate = top3_match.float().mean().item()
    print(f"    Full sort match rate:    {full_match_rate * 100:.2f}%")
    print(f"    Top-1 only match rate:   {top1_match_rate * 100:.2f}%")
    print(f"    Top-3 only match rate:   {top3_match_rate * 100:.2f}%")

    # ===== Exp D5: 距离数值差异 =====
    print(f"\n  --- Exp D5: distance value distribution ---")
    Delta = (d_E - d_H).abs()  # (B, K)
    delta_mean = Delta.mean().item()
    delta_std = Delta.std().item()
    delta_max = Delta.max().item()
    delta_min = Delta.min().item()
    print(f"    Δ mean:  {delta_mean:.6f}")
    print(f"    Δ std:   {delta_std:.6f}")
    print(f"    Δ max:   {delta_max:.6f}")
    print(f"    Δ min:   {delta_min:.6f}")

    # 最近邻处的 Δ（argmin ties-stable）
    nn_idx_E = idx_E  # 复用上面的结果
    nn_delta = Delta[torch.arange(Delta.shape[0]), nn_idx_E]
    print(f"    Δ at top-1 (Euclidean): mean={nn_delta.mean():.6f}, "
          f"max={nn_delta.max():.6f}")

    return {
        "d4": {
            "full_match_rate": full_match_rate,
            "top1_match_rate": top1_match_rate,
            "top3_match_rate": top3_match_rate,
        },
        "d5": {
            "delta_mean": delta_mean,
            "delta_std": delta_std,
            "delta_max": delta_max,
            "delta_min": delta_min,
            "nn_delta_mean": float(nn_delta.mean()),
            "nn_delta_max": float(nn_delta.max()),
        },
    }


# ============== 主函数 ==============

def main():
    out_dir = "products/task16/from_task70"
    os.makedirs(out_dir, exist_ok=True)

    d1_result = exp_d1()
    d4_d5_result = exp_d4_d5()

    # 综合决策
    print(f"\n{'='*60}")
    print("Final Decision")
    print(f"{'='*60}")
    d1_rank_ok = d1_result["rank_identical"]
    d1_pearson = d1_result["pearson"]
    d4_top1 = d4_d5_result["d4"]["top1_match_rate"]
    d4_full = d4_d5_result["d4"]["full_match_rate"]
    d5_max = d4_d5_result["d5"]["delta_max"]

    print(f"  D1 (function rank identical):    {d1_rank_ok} (Pearson={d1_pearson:.4f})")
    print(f"  D4 (top-1 match rate):          {d4_top1 * 100:.2f}%")
    print(f"  D4 (full sort match rate):      {d4_full * 100:.2f}%")
    print(f"  D5 (delta_max):                 {d5_max:.6f}")

    # 情景判定
    if d4_top1 == 1.0 and d5_max < 0.001:
        scenario = "A (等价性是根本数学性质)"
    elif d4_top1 == 1.0 and d5_max > 0.1:
        scenario = "B (排序保持但距离实际不同)"
    elif d4_top1 < 1.0:
        scenario = "C (测试方式有问题，top1 都不一致)"
    else:
        scenario = "D (中间态，需进一步分析)"
    print(f"  >>> Scenario: {scenario}")

    # 保存
    out = {
        "task": "#70 root cause diagnosis",
        "D1_function": d1_result,
        "D4_real_data": d4_d5_result["d4"],
        "D5_value_distribution": d4_d5_result["d5"],
        "scenario": scenario,
    }
    with open(os.path.join(out_dir, "d1_d4_d5_results.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n  Saved: products/task16/from_task70/d1_d4_d5_results.json")


if __name__ == "__main__":
    main()