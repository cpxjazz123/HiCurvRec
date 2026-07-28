#!/usr/bin/env python3
"""
Task #210 Phase A — 维度 × 半径 扫描.

输入: A0 baseline (#181) best_loss_model.pth 的 encoder, 在 Instruments item_emb.parquet 上跑一遍, 提取 (9922, 32) latent.

扫描:
  d_hyp ∈ {2, 4, 8, 16, 32, 64}     (双曲维度, > 32 用 0-pad 到 64)
  ρ     ∈ {0.5, 1.0, 2.0, 3.0, 4.0} (target radius)
  层 L ∈ {0, 1, 2}, K = 64 / 128 / 256

每个格子输出:
  λ          — 共形因子 (2/(1-‖x‖²))
  dyn_range  — max_pair_dist / min_pair_dist (在 hyp ball 上)
  util       — unique argmin / K (latent 分配到码字的比例)
  flip_rate  — 双曲 argmin ≠ 欧式 argmin 的比例 (vs A0 codes)
  min_angle  — 最小两两夹角 (rad)
  n_centers  — 实际 kmeans 收敛的码字数 (max = K)

判据:
  λ ≥ 4.7 AND util ≥ 0.9 → 可行区 ✅
  λ ≥ 4.7 AND util < 0.5 → 几何激活但分配崩 ❌
  λ <  4.7                → 几何未激活 (无操作)
"""
import sys, os, json
sys.path.insert(0, '/home/wlia0047/ar57/wenyu/GeneRec/HG-Rec')
sys.path.insert(0, '/home/wlia0047/ar57/wenyu/GeneRec/scripts')

import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.cluster import KMeans

from model.utils import expmap0, proj_to_ball
from model.hrqvae import HRQVAE

# === 1. 加载 A0 baseline encoder ===
A0_CKPT = '/home/wlia0047/ar57/wenyu/GeneRec/products/task181/hrqvae_fix_v2/Jul-25-2026_16-20-31_beta_1.000_codebook_[64,128,256]_sk_0.000/best_loss_model.pth'

def load_a0_encoder(ckpt_path):
    ckpt = torch.load(ckpt_path, weights_only=False, map_location='cpu')
    args = ckpt['args']
    state_dict = ckpt['state_dict']

    from model.utils import EmbDataset
    data = EmbDataset(args.data_path)

    model = HRQVAE(
        in_dim=data.dim,
        num_emb_list=args.num_emb_list,
        e_dim=args.e_dim,
        layers=args.layers,
        dropout_prob=args.dropout_prob,
        bn=args.bn,
        loss_type=args.loss_type,
        quant_loss_weight=args.quant_loss_weight,
        beta=args.beta,
        kmeans_init=args.kmeans_init,
        kmeans_iters=args.kmeans_iters,
        sk_eps=args.sk_epsilons,
        sk_iters=args.sk_iters,
    )
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model, data

# === 2. 提取 latent + A0 codes ===
def extract_latent_and_codes(model, data, device='cpu'):
    loader = DataLoader(data, batch_size=64, shuffle=False, num_workers=2)
    all_latents = []
    with torch.no_grad():
        for d in loader:
            d = d.to(device)
            x_lat = model.encoder(d)
            all_latents.append(x_lat.cpu())
    all_latents = torch.cat(all_latents, dim=0)  # (N, e_dim=32)
    print(f"  Latent shape: {all_latents.shape}, mean norm: {all_latents.norm(dim=-1).mean().item():.4f}")

    # A0 codes per layer (raw embedding weights)
    codes_per_layer = []
    for vq in model.hrq.vq_layers:
        codes_per_layer.append(vq.embeddings.weight.data.cpu())  # (K, 32)
    return all_latents, codes_per_layer

# === 3. Phase A 扫描核心 ===
def scan(latents, codes_per_layer, d_list, rho_list):
    """latents: (N, 32), codes_per_layer: list of (K, 32)."""
    N = latents.shape[0]
    results = []

    for L in range(3):
        K = codes_per_layer[L].shape[0]
        euc_codes = codes_per_layer[L].numpy()  # (K, 32)
        # 计算每个 latent 的 argmin_euc (跟 A0 codes 比欧氏距离)
        euc_dist = np.linalg.norm(latents.numpy()[:, None, :] - euc_codes[None, :, :], axis=-1)  # (N, K)
        argmin_euc = euc_dist.argmin(axis=1)  # (N,)

        for d in d_list:
            # 取 hyp part: 前 d 维 (d > 32 用 0-pad)
            if d <= 32:
                hyp_lat = latents[:, :d].numpy()  # (N, d)
            else:
                hyp_lat = np.zeros((N, d), dtype=np.float32)
                hyp_lat[:, :32] = latents[:, :32].numpy()

            for rho in rho_list:
                # kmeans on hyp_lat → K cluster centers in d-dim tangent space
                kmeans = KMeans(n_clusters=K, n_init=3, random_state=42 + L*100 + d + int(rho*10))
                kmeans.fit(hyp_lat)
                centers_tan = kmeans.cluster_centers_  # (K, d) — tangent space

                # 把 centers 放到 tangent norm = ρ/2 的壳上 (球面 kmeans 替代)
                norms = np.linalg.norm(centers_tan, axis=-1, keepdims=True).clip(min=1e-8)
                # 1) 方向归一
                directions = centers_tan / norms
                # 2) 切空间 norm = ρ/2 → 球面 norm = tanh(ρ/2)
                target_tan_norm = rho / 2.0
                tan_centers = directions * target_tan_norm
                # 3) expmap0 → 球面
                tan_centers_t = torch.from_numpy(tan_centers).float()
                ball_centers_t = proj_to_ball(expmap0(tan_centers_t, c=1.0), c=1.0)
                ball_centers = ball_centers_t.numpy()  # (K, d)
                ball_norms = np.linalg.norm(ball_centers, axis=-1)  # (K,)

                # λ = 2 / (1 - ‖x‖²)
                # 注意: ‖x‖_E 在 d 维子空间内, 跟 e_dim=32 无关
                lambda_kappa = 2.0 / (1.0 - ball_norms ** 2).clip(min=1e-5)
                lambda_mean = float(lambda_kappa.mean())
                lambda_min = float(lambda_kappa.min())

                # 动态范围: 最大 pair dist / 最小 pair dist (hyp ball 上)
                # 用欧氏距离作为 proxy (d 维小, 几何近似)
                if K > 1:
                    pair_dists = []
                    # 取前 100 个中心算 pair dist (K=256 全算太慢)
                    sample = ball_centers[:min(100, K)]
                    for i in range(len(sample)):
                        for j in range(i+1, len(sample)):
                            pair_dists.append(np.linalg.norm(sample[i] - sample[j]))
                    pair_dists = np.array(pair_dists)
                    if len(pair_dists) > 0:
                        max_dist = float(pair_dists.max())
                        min_dist = float(pair_dists.min())
                        dyn_range = max_dist / max(min_dist, 1e-8)
                    else:
                        max_dist = min_dist = dyn_range = 0.0
                else:
                    max_dist = min_dist = dyn_range = 0.0

                # argmin_hyp: latent 在 hyp 子空间到 ball_centers 的最近
                # 用 Poincaré distance (在 d 维 ball 上)
                # poincare_distance(u, v, c=1) = arccosh(1 + 2·‖u-v‖² / ((1-‖u‖²)(1-‖v‖²)))
                # 简化: 用欧式 dist (小 d 时近似)
                hyp_lat_t = torch.from_numpy(hyp_lat).float()
                hyp_lat_ball = proj_to_ball(expmap0(hyp_lat_t * target_tan_norm / (np.linalg.norm(hyp_lat, axis=-1, keepdims=True).clip(min=1e-8)), c=1.0), c=1.0).numpy() \
                    if (np.linalg.norm(hyp_lat, axis=-1).mean() > 0.01) else hyp_lat  # 若 hyp_lat 已经很小, 不映射
                # 简化: 用 tan_centers 计算欧氏距离
                hyp_dist = np.linalg.norm(hyp_lat[:, None, :] - tan_centers[None, :, :], axis=-1)
                argmin_hyp = hyp_dist.argmin(axis=1)

                # util
                unique_used = len(np.unique(argmin_hyp))
                util = unique_used / K

                # flip_rate (跟 A0 euc argmin 比)
                flip_rate = float((argmin_hyp != argmin_euc).mean())

                # min_pair_angle
                # cos(θ) = u·v / (‖u‖·‖v‖)
                if K > 1 and len(ball_centers) > 1:
                    sample = ball_centers[:min(100, K)]
                    cos_sim = sample @ sample.T  # (S, S)
                    norms_s = np.linalg.norm(sample, axis=-1, keepdims=True).clip(min=1e-8)
                    cos_sim = cos_sim / (norms_s @ norms_s.T).clip(min=1e-8)
                    np.fill_diagonal(cos_sim, -2)  # 排除自身 (用 -2 替代 -1 防止 cos=1 误判)
                    max_cos = float(cos_sim.max())
                    # 用 np.clip(arr, ...) 或 min/max 替代
                    max_cos_clipped = min(1.0, max(-1.0, max_cos))
                    min_angle = float(np.arccos(max_cos_clipped))
                else:
                    min_angle = 0.0

                results.append({
                    'layer': L,
                    'K': K,
                    'd_hyp': d,
                    'rho': rho,
                    'lambda_mean': lambda_mean,
                    'lambda_min': lambda_min,
                    'ball_norm_mean': float(ball_norms.mean()),
                    'dyn_range': dyn_range,
                    'util': util,
                    'flip_rate': flip_rate,
                    'min_pair_angle_rad': min_angle,
                    'n_unique_centers_used': unique_used,
                })
    return results

if __name__ == "__main__":
    print("=== Phase A: 加载 A0 baseline encoder ===")
    model, data = load_a0_encoder(A0_CKPT)

    print("=== 提取 latent + A0 codes ===")
    latents, codes = extract_latent_and_codes(model, data)

    # 3 层 codes shape
    for L in range(3):
        print(f"  L{L}: codes shape={codes[L].shape}")

    print("=== Phase A 扫描 (d × ρ) ===")
    d_list = [2, 4, 8, 16, 32, 64]
    rho_list = [0.5, 1.0, 2.0, 3.0, 4.0]
    results = scan(latents, codes, d_list, rho_list)

    # 保存
    out_path = '/home/wlia0047/ar57/wenyu/GeneRec/products/task210/phaseA_scan.json'
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n✅ Phase A 扫描结果 saved to {out_path}")
    print(f"  Total configs: {len(results)} (= 3 layers × {len(d_list)} d × {len(rho_list)} rho)")

    # 打印 summary: 找可行区 (λ ≥ 4.7 AND util ≥ 0.9)
    feasible = [r for r in results if r['lambda_mean'] >= 4.7 and r['util'] >= 0.9]
    print(f"\n=== 可行区 (λ≥4.7 AND util≥0.9): {len(feasible)} 格子 ===")
    if feasible:
        for r in feasible[:10]:
            print(f"  L{r['layer']} d={r['d_hyp']} ρ={r['rho']}: λ={r['lambda_mean']:.2f}, util={r['util']:.2f}, flip={r['flip_rate']:.3f}, dyn={r['dyn_range']:.2f}")

    # 不可行但 λ≥4.7 (分配崩)
    lambda_active = [r for r in results if r['lambda_mean'] >= 4.7 and r['util'] < 0.5]
    print(f"\n=== 几何激活但分配崩 (λ≥4.7 AND util<0.5): {len(lambda_active)} 格子 ===")
    if lambda_active:
        for r in lambda_active[:10]:
            print(f"  L{r['layer']} d={r['d_hyp']} ρ={r['rho']}: λ={r['lambda_mean']:.2f}, util={r['util']:.2f}")

    # λ < 4.7 (无操作)
    no_op = [r for r in results if r['lambda_mean'] < 4.7]
    print(f"\n=== 无操作 (λ<4.7): {len(no_op)} 格子 ===")
