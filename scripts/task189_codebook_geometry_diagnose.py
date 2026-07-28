#!/usr/bin/env python3
"""
Task #189 — Stage 1 codebook 几何诊断 (8 项).
读最低碰撞率 ckpt + 最终 ckpt 各一遍, 输出分位数.
"""
import argparse
import collections
import json
import logging
import sys
import os
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "/HG-Rec")
sys.path.insert(0, os.getcwd())
import numpy as np
import torch
import glob
from torch.utils.data import DataLoader
from tqdm import tqdm

from model.utils import *
from model.hrqvae import HRQVAE
print(f"[task189] sys.path = {os.getcwd()}")


def quantile(x, q):
    return float(np.quantile(np.asarray(x, dtype=np.float64), q))


def summarize(name, x):
    """报告 p05 / p50 / p95 / max, 绝对不报均值."""
    arr = np.asarray(x, dtype=np.float64)
    if arr.size == 0:
        return f"  {name}: EMPTY"
    return (
        f"  {name}: p05={quantile(arr, 0.05):.6f}  p50={quantile(arr, 0.5):.6f}  "
        f"p95={quantile(arr, 0.95):.6f}  max={arr.max():.6f}  (n={arr.size})"
    )


def get_indices_raw(model, data, device, e_dim):
    """只用 argmin (no Sinkhorn), 返回 latents + indices + 残差."""
    model.eval()
    by_layer = {0: [], 1: [], 2: []}
    with torch.no_grad():
        for d in DataLoader(data, batch_size=128, shuffle=False):
            d = d.to(device)
            z = model.encoder(d)  # (B, e_dim)
            x = proj_to_ball(expmap0(z, model.hrq.vq_layers[0].c), model.hrq.vq_layers[0].c)
            residual = x
            for li, layer in enumerate(model.hrq.vq_layers):
                # 不走 vq_layers[0].forward 全路径, 走简化版本
                cb = layer.embeddings.weight
                cb_h = proj_to_ball(expmap0(cb, layer.c), layer.c)
                # 用最后一次 argmin
                # 计算距离
                latent_h = residual
                B = latent_h.shape[0]
                K = cb_h.shape[0]
                x_exp = latent_h.unsqueeze(1).expand(B, K, -1)
                cb_exp = cb_h.unsqueeze(0).expand(B, K, -1)
                d = poincare_distance(x_exp, cb_exp, layer.c).squeeze(-1)
                idx = torch.argmin(d, dim=-1)
                x_q = cb.index_select(0, idx)
                by_layer[li].append(idx.cpu().numpy())
                residual = residual - proj_to_ball(expmap0(x_q, layer.c), layer.c)
    out = {li: np.concatenate(v) for li, v in by_layer.items()}
    return out


def encode_all(model, data, device):
    """收集所有 latent (z = encoder(x))."""
    zs = []
    model.eval()
    with torch.no_grad():
        for d in DataLoader(data, batch_size=128, shuffle=False):
            d = d.to(device)
            z = model.encoder(d)
            zs.append(z.cpu().numpy())
    return np.concatenate(zs, axis=0)  # (N, e_dim)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_path", type=str, required=True)
    parser.add_argument("--label", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--include_mid", action="store_true")
    args_cli = parser.parse_args()

    print(f"\n\n{'='*80}")
    print(f"Task #189 — {args_cli.label}")
    print(f"ckpt: {args_cli.ckpt_path}")
    print(f"{'='*80}")

    ckpt = torch.load(args_cli.ckpt_path, weights_only=False, map_location="cpu")
    train_args = ckpt["args"]
    state_dict = ckpt["state_dict"]
    device = torch.device(args_cli.device)

    data = EmbDataset(train_args.data_path)
    model = HRQVAE(
        in_dim=data.dim,
        num_emb_list=train_args.num_emb_list,
        e_dim=train_args.e_dim,
        layers=train_args.layers,
        dropout_prob=train_args.dropout_prob,
        bn=train_args.bn,
        loss_type=train_args.loss_type,
        quant_loss_weight=train_args.quant_loss_weight,
        beta=train_args.beta,
        kmeans_init=train_args.kmeans_init,
        kmeans_iters=train_args.kmeans_iters,
        sk_eps=train_args.sk_epsilons,
        sk_iters=train_args.sk_iters,
    )
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    print(f"\ntrain_args: e_dim={train_args.e_dim}, num_emb_list={train_args.num_emb_list}, "
          f"curvatures={[l.c for l in model.hrq.vq_layers]}")

    n_layers = len(model.hrq.vq_layers)

    # === Item 1: 每层码字的半径 (3 口径) ===
    print("\n[Item 1] 每层码字半径 ‖e‖ (切空间) / ‖x‖ = tanh(√c·‖e‖) (球内) / ρ = 2·artanh(√c·‖x‖) (测地)")
    for li, layer in enumerate(model.hrq.vq_layers):
        e = layer.embeddings.weight.detach()  # (K, e_dim)
        c = layer.c
        sqrt_c = np.sqrt(c)
        e_norm = e.norm(dim=-1).cpu().numpy()
        x_norm = np.tanh(sqrt_c * e_norm)
        rho = 2 * np.arctanh(np.clip(sqrt_c * x_norm, 1e-9, 1 - 1e-9))
        print(f"  L{li} (K={e.shape[0]}, c={c:.4f}):")
        print(summarize(f"L{li}.‖e‖", e_norm))
        print(summarize(f"L{li}.‖x‖", x_norm))
        print(summarize(f"L{li}.ρ    ", rho))

    # === Item 2: 累加和的半径 (decoder-step x_q = Σ_{k≤ℓ} x_k) ===
    # 真实运行时 x_q = Σ x_l_k 是逐 item 不同. 因此这里重新跑 forward, 对每个
    # item 的 encoder 输出逐层量化, 累加 L0, L0+L1, L0+L1+L2 三种情形.
    print("\n[Item 2] decoder 累加 x_q = Σ_{k≤ℓ} x_k (球内范数 + ρ)")
    c0 = model.hrq.vq_layers[0].c
    sum_x_norms = {0: [], 1: [], 2: []}
    model.eval()
    with torch.no_grad():
        for d in DataLoader(data, batch_size=128, shuffle=False):
            d = d.to(device)
            z = model.encoder(d)
            x0 = proj_to_ball(expmap0(z, c0), c0)
            cum_x = torch.zeros_like(x0)
            for li, layer in enumerate(model.hrq.vq_layers):
                c = layer.c
                cb = layer.embeddings.weight.detach()
                cb_h = proj_to_ball(expmap0(cb, c), c)
                B = cum_x.shape[0]
                K = cb_h.shape[0]
                x_exp = cum_x.unsqueeze(1).expand(B, K, -1)
                cb_exp = cb_h.unsqueeze(0).expand(B, K, -1)
                dist = poincare_distance(x_exp, cb_exp, c).squeeze(-1)
                idx = torch.argmin(dist, dim=-1)
                x_q = cb.index_select(0, idx)
                x_q_h = proj_to_ball(expmap0(x_q, c), c)
                cum_x = cum_x + x_q_h
                sum_x_norms[li].append(cum_x.norm(dim=-1).cpu().numpy())
    for li in range(n_layers):
        arr = np.concatenate(sum_x_norms[li])
        c = model.hrq.vq_layers[li].c
        sqrt_c = np.sqrt(c)
        x_sum = arr  # 这已经是球内范数
        rho = 2 * np.arctanh(np.clip(sqrt_c * x_sum, 1e-9, 1 - 1e-9))
        print(f"  Σ_{{k≤{li}}} (n={arr.size}):")
        print(summarize(f"  ‖x‖  ", x_sum))
        print(summarize(f"  ρ    ", rho))

    # === Item 3: λ 分位数 (绝对不均值) ===
    print("\n[Item 3] λ = 2/(1 - c‖x‖²)  (绝对不均值, 重尾)")
    for li, layer in enumerate(model.hrq.vq_layers):
        c = layer.c
        e = layer.embeddings.weight.detach()
        # 把码字投到球内, 然后算 λ
        cb_h = proj_to_ball(expmap0(e, c), c)
        x_norm_sq = (cb_h * cb_h).sum(dim=-1).detach().cpu().numpy()
        lam = 2.0 / np.clip(1.0 - c * x_norm_sq, 1e-6, None)
        print(f"  L{li}:")
        print(f"    λ p50={quantile(lam, 0.5):.4f}  p90={quantile(lam, 0.9):.4f}  "
              f"p99={quantile(lam, 0.99):.4f}  max={lam.max():.4f}")

    # === Item 4: 边界安全性 ===
    print("\n[Item 4] 边界安全性")
    for li, layer in enumerate(model.hrq.vq_layers):
        c = layer.c
        e = layer.embeddings.weight.detach()
        cb_h = proj_to_ball(expmap0(e, c), c)
        x_norm_sq = (cb_h * cb_h).sum(dim=-1).detach().cpu().numpy()
        cnrm2 = c * x_norm_sq
        boundary = 1.0 - 1e-4
        near = (cnrm2 > boundary).sum()
        print(f"  L{li}: max(c‖x‖²) = {cnrm2.max():.6e}   "
              f"near-boundary (>1-1e-4) = {near}/{cnrm2.size} = {100*near/cnrm2.size:.4f}%")

    # === Item 5: encoder 输出范数 ===
    print("\n[Item 5] encoder 输出 ‖z‖ (切空间, 先于 expmap)")
    zs = encode_all(model, data, device)
    z_norm = np.linalg.norm(zs, axis=-1)
    print(summarize("‖z‖", z_norm))

    # === Item 6: 每层残差范数 ===
    print("\n[Item 6] 每层残差 ‖r_ℓ‖ (球内范数)")
    # 先跑一次 model.get_indices 但同时收集残差
    model.eval()
    residual_norms = {li: [] for li in range(n_layers)}
    latents_for_residual = []
    with torch.no_grad():
        for d in DataLoader(data, batch_size=128, shuffle=False):
            d = d.to(device)
            z = model.encoder(d)  # (B, e_dim)
            x = proj_to_ball(expmap0(z, model.hrq.vq_layers[0].c), model.hrq.vq_layers[0].c)
            residual = x
            for li, layer in enumerate(model.hrq.vq_layers):
                c = layer.c
                residual_norms[li].append(residual.norm(dim=-1).cpu().numpy())
                cb = layer.embeddings.weight
                cb_h = proj_to_ball(expmap0(cb, c), c)
                B = residual.shape[0]
                K = cb_h.shape[0]
                x_exp = residual.unsqueeze(1).expand(B, K, -1)
                cb_exp = cb_h.unsqueeze(0).expand(B, K, -1)
                d = poincare_distance(x_exp, cb_exp, c).squeeze(-1)
                idx = torch.argmin(d, dim=-1)
                x_q = cb.index_select(0, idx)
                x_q_h = proj_to_ball(expmap0(x_q, c), c)
                residual = residual - x_q_h
    for li in range(n_layers):
        arr = np.concatenate(residual_norms[li])
        print(f"  L{li}: p50={quantile(arr, 0.5):.6f}  "
              f"p05={quantile(arr, 0.05):.6f}  p95={quantile(arr, 0.95):.6f}  max={arr.max():.6f}")

    # === Item 7: 真实码本利用率 (Sinkhorn 前, 去重前) ===
    print("\n[Item 7] 真实码本利用率 (argmin raw, no Sinkhorn)")
    by_layer = get_indices_raw(model, data, device, train_args.e_dim)
    for li in range(n_layers):
        idx = by_layer[li]
        K = model.hrq.vq_layers[li].embeddings.weight.shape[0]
        used = len(set(idx.tolist()))
        dead = K - used
        print(f"  L{li}: K={K}, used={used} ({100*used/K:.2f}%), dead={dead}")

    # === Item 8: collision rate 训练轨迹 (静态信息, 从 ckpt 路径推断) ===
    print("\n[Item 8] collision rate 训练轨迹 (静态)")
    # ckpt 文件名包含 epoch 和 collision
    ckpt_name = os.path.basename(args_cli.ckpt_path)
    if "best_collision" in ckpt_name:
        print(f"  这是一个 best_collision ckpt (不带 epoch 编号)")
    else:
        import re
        m = re.search(r"epoch_(\d+)_collision_([\d.]+)", ckpt_name)
        if m:
            ep = int(m.group(1))
            coll = float(m.group(2))
            print(f"  这是 epoch={ep}, collision={coll:.4f}")

    print(f"\n{'='*80}")
    print(f"DONE — {args_cli.label}")
    print(f"{'='*80}\n")
