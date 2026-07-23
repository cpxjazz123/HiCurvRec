"""
Task 62 — Exp3: 多几何距离替换实验
比较 4 种 settings (A/B/C/D) 下 RQ-VAE 的量化损失和码本利用率。

用法: python scripts/task6_evaluate_distance.py [--device cuda:0]

Settings:
  A (baseline): L1=d_E, L2=d_E, L3=d_E  (全欧氏)
  B (HRQ):      L1=d_H, L2=d_H, L3=d_H  (全双曲)
  C (多几何v1):  L1=d_E, L2=d_H, L3=d_S
  D (多几何v2):  L1=d_E, L2=d_H, L3=d_E
"""

import os, sys, argparse, json, math
import torch
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def euclidean_dist(x, y):
    """d_E(x, y) = ||x - y||₂"""
    return torch.norm(x - y, dim=1)


def poincare_dist(x, y, eps=1e-8):
    """
    Poincaré ball 距离 (双曲).
    x, y: (D,) tensors
    投影到单位球内: x_p = tanh(||x||) * x / ||x||
    """
    # 投影到单位球内
    x_norm = torch.norm(x).clamp(min=eps)
    y_norm = torch.norm(y).clamp(min=eps)
    x_p = torch.tanh(x_norm) * x / x_norm
    y_p = torch.tanh(y_norm) * y / y_norm

    diff_norm = torch.norm(x_p - y_p)
    x_norm_sq = 1 - torch.norm(x_p) ** 2
    y_norm_sq = 1 - torch.norm(y_p) ** 2

    # d_H = arcosh(1 + 2 * ||x - y||² / ((1-||x||²)(1-||y||²)))
    arg = 1 + 2 * diff_norm ** 2 / (x_norm_sq * y_norm_sq).clamp(min=eps)
    # 数值稳定: arcosh(t) = log(t + sqrt(t² - 1))
    arg = arg.clamp(min=1.0)
    d = torch.log(arg + torch.sqrt(arg ** 2 - 1))
    return d


def spherical_dist(x, y, eps=1e-8):
    """
    球面距离 (测地线).
    x, y: (D,) tensors
    归一化到单位球: x_s = x / ||x||
    """
    x_norm = torch.norm(x).clamp(min=eps)
    y_norm = torch.norm(y).clamp(min=eps)
    x_s = x / x_norm
    y_s = y / y_norm

    cos_sim = (x_s * y_s).sum().clamp(-1 + eps, 1 - eps)
    return torch.acos(cos_sim)


def create_distance_function(name):
    """返回距离函数 (codebook_vector, input_vector) -> scalar"""
    if name == "E":
        return euclidean_dist
    elif name == "H":
        return poincare_dist
    elif name == "S":
        return spherical_dist
    else:
        raise ValueError(f"Unknown distance: {name}")


def quantize_with_distance(residual, codebook, dist_fn):
    """
    用指定距离函数做量化。
    residual: (D,) tensor
    codebook: (K, D) tensor
    returns: (z_q, index)
    """
    distances = dist_fn(codebook, residual.unsqueeze(0).expand_as(codebook))
    # 当 dist_fn 返回 scalar 时需要特殊处理
    # 改为 vectorized
    pass


def quantize_with_distance_batch(batch, codebook, dist_name, device='cuda:0'):
    """
    batch: (B, D) tensor
    codebook: (K, D) tensor
    dist_name: 'E', 'H', or 'S'
    returns: indices (B,)
    """
    B, D = batch.shape
    K = codebook.shape[0]

    if dist_name == 'E':
        # 欧氏距离: cdist 支持批量
        dist = torch.cdist(batch, codebook, p=2)  # (B, K)
        return dist.argmin(dim=1)

    elif dist_name == 'H':
        # Poincaré 距离: 逐点计算
        # 投影到单位球内
        batch_norm = torch.norm(batch, dim=1, keepdim=True).clamp(min=1e-8)  # (B, 1)
        codebook_norm = torch.norm(codebook, dim=1, keepdim=True).clamp(min=1e-8)  # (K, 1)

        batch_p = torch.tanh(batch_norm) * batch / batch_norm  # (B, D)
        codebook_p = torch.tanh(codebook_norm) * codebook / codebook_norm  # (K, D)

        # d_H: (B, K) 矩阵
        # diff = batch_p.unsqueeze(1) - codebook_p.unsqueeze(0)  # (B, K, D)
        batch_exp = batch_p.unsqueeze(1)  # (B, 1, D)
        codebook_exp = codebook_p.unsqueeze(0)  # (1, K, D)
        diff_sq = (batch_exp - codebook_exp).norm(dim=2) ** 2  # (B, K)

        x_norm_sq = (1 - (batch_p ** 2).sum(dim=1, keepdim=True)).clamp(min=1e-8)  # (B, 1)
        y_norm_sq = (1 - (codebook_p ** 2).sum(dim=1, keepdim=True)).clamp(min=1e-8)  # (K, 1)
        y_norm_sq = y_norm_sq.T  # (1, K) for broadcasting with (B, 1)

        arg = 1 + 2 * diff_sq / (x_norm_sq * y_norm_sq)  # (B, K)
        arg = arg.clamp(min=1.0)
        dist = torch.log(arg + torch.sqrt(arg ** 2 - 1))
        return dist.argmin(dim=1)

    elif dist_name == 'S':
        # 球面距离: 余弦相似度
        batch_norm = torch.norm(batch, dim=1, keepdim=True).clamp(min=1e-8)
        codebook_norm = torch.norm(codebook, dim=1, keepdim=True).clamp(min=1e-8)

        batch_s = batch / batch_norm  # (B, D)
        codebook_s = codebook / codebook_norm  # (K, D)

        # 余弦相似度越大 = 角度越小 = 距离越小
        cos_sim = batch_s @ codebook_s.T  # (B, K)
        cos_sim = cos_sim.clamp(-1 + 1e-8, 1 - 1e-8)
        dist = torch.acos(cos_sim)  # (B, K)
        return dist.argmin(dim=1)

    else:
        raise ValueError(f"Unknown distance: {dist_name}")


def evaluate_setting(model, embeddings, setting, device='cuda:0'):
    """
    Evaluate one setting.
    setting: list of 3 strings, e.g. ['E', 'E', 'E']
    returns: dict of metrics
    """
    names = {'E': 'd_E', 'H': 'd_H', 'S': 'd_S'}
    print(f"\n  Setting: L1={names[setting[0]]}, L2={names[setting[1]]}, L3={names[setting[2]]}")

    model.eval()
    model = model.to(device)
    data = embeddings.to(device)
    N, D = data.shape

    # 逐层量化（使用指定距离函数）
    total_recon_error = 0.0
    all_indices = []
    residuals_list = []

    residual = data
    with torch.no_grad():
        for layer_idx in range(model.n_layers):
            codebook = model.quantizers[layer_idx].codebook  # (K, D)
            dist_name = setting[layer_idx]

            # 批量量化
            indices = quantize_with_distance_batch(residual, codebook, dist_name, device)
            z_q = codebook[indices]
            r = residual - z_q

            # 记录
            all_indices.append(indices)
            residuals_list.append(r)

            # 分层重建误差
            layer_mse = (r ** 2).mean().item()
            print(f"    Layer {layer_idx + 1}: {names[dist_name]} | residual MSE = {layer_mse:.8f}")

            total_recon_error += layer_mse
            residual = r

    # 总重建误差 (输入 → 最终残差的 MSE)
    total_mse = (residual ** 2).mean().item()

    # 码本利用率
    usage_stats = []
    for layer_idx in range(model.n_layers):
        indices = all_indices[layer_idx]
        unique = indices.unique().numel()
        total = model.quantizers[layer_idx].n_clusters
        usage_stats.append(float(unique) / total * 100)

    # 熵 (分配分布的信息量)
    entropy_stats = []
    for layer_idx in range(model.n_layers):
        indices = all_indices[layer_idx].cpu().numpy()
        counts = np.bincount(indices, minlength=model.quantizers[layer_idx].n_clusters)
        probs = counts / counts.sum()
        # 熵
        ent = -np.sum(probs * np.log(probs.clip(min=1e-12))) / np.log(model.quantizers[layer_idx].n_clusters)
        entropy_stats.append(float(ent))

    result = {
        "setting": setting,
        "setting_name": f"({names[setting[0]]},{names[setting[1]]},{names[setting[2]]})",
        "layer_recon_mse": [float(r.pow(2).mean().item()) for r in residuals_list],
        "total_recon_mse": total_mse,
        "codebook_usage_pct": usage_stats,
        "codebook_entropy": entropy_stats,
        "normalized_entropy": entropy_stats,
    }

    print(f"    Total recon MSE = {total_mse:.8f}")
    print(f"    Codebook usage: {[f'{u:.1f}%' for u in usage_stats]}")
    print(f"    Codebook entropy: {[f'{e:.3f}' for e in entropy_stats]}")

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rqvae_ckpt", default="products/task16/from_task62/rqvae_ckpt.pt")
    parser.add_argument("--embeddings", default="products/task16/from_task62/item_embeddings.pt")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    print("=" * 60)
    print("Task 62 — Exp3: 多几何距离替换实验")
    print("=" * 60)

    if not os.path.exists(args.rqvae_ckpt):
        print(f"ERROR: {args.rqvae_ckpt} not found.")
        sys.exit(1)

    # Load
    print(f"\n[1/3] Loading model...")
    ckpt = torch.load(args.rqvae_ckpt, map_location="cpu")
    from scripts.task62_train_rqvae import ResidualQuantization
    model = ResidualQuantization(
        n_features=ckpt["n_features"],
        n_layers=ckpt["n_layers"],
        n_clusters=ckpt["n_clusters"],
    )
    model.load_state_dict(ckpt["state_dict"])

    print(f"[2/3] Loading embeddings...")
    embeddings = torch.load(args.embeddings, map_location="cpu")

    device = args.device if torch.cuda.is_available() else "cpu"

    # Settings
    settings = [
        ['E', 'E', 'E'],  # A: baseline
        ['H', 'H', 'H'],  # B: HRQ
        ['E', 'H', 'S'],  # C: multi-geometry v1
        ['E', 'H', 'E'],  # D: multi-geometry v2
    ]

    print(f"\n[3/3] Running {len(settings)} settings...")
    results = {}
    for sidx, setting in enumerate(settings):
        key = chr(ord('A') + sidx)
        print(f"\n--- Setting {key} ---")
        results[key] = evaluate_setting(model, embeddings, setting, device)

    # Summary table
    print("\n" + "=" * 60)
    print("Summary Table")
    print("=" * 60)
    print(f"{'Setting':>10} {'d(L1,L2,L3)':>20} {'Recon MSE':>12} {'Usage':>16} {'Entropy':>10}")
    print("-" * 70)
    for key in ['A', 'B', 'C', 'D']:
        r = results[key]
        usage_str = "/".join([f"{u:.0f}%" for u in r["codebook_usage_pct"]])
        entropy_str = "/".join([f"{e:.2f}" for e in r["normalized_entropy"]])
        print(f"  {key:>8} {r['setting_name']:>20} {r['total_recon_mse']:>12.8f} {usage_str:>16} {entropy_str:>10}")

    # Save
    out_dir = "products/task16/from_task62"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "exp3_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved: {out_path}")


if __name__ == "__main__":
    main()
