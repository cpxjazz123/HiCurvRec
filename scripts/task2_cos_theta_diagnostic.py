#!/usr/bin/env python3
"""task16 量 X: cos θ — 量化方向匹配度（独立诊断量）

定义：cos θ_l = r̂_l · q̂_l ，其中 r̂ = r/‖r‖, q̂ = q/‖q‖
- 值域 [-1, 1]
- cos θ = 1 → 完全同向（量化方向完全匹配）
- cos θ = 0 → 完全正交（方向信息丢失）
- cos θ = -1 → 完全反向（量化方向完全错乱）

为什么 f_radial 不够：
- f_radial 把误差按"幅度 vs 方向"拆，但 f_radial 的"幅度"分量里隐含 cos θ
- 用户的 f_radial 定义 A = (‖r‖ - ‖q‖)²/‖e‖²，定义 B = (e·r̂)²/‖e‖²
- 定义 B 在 cos θ → 0 时数学上自动趋近 ‖r‖²/(‖r‖² + ‖q‖²) ≈ 1（与 cos θ 退化时数值偏置）
- 因此 f_radial 在 L2/L3 高数值可能只是 cos θ ≈ 0 的副产品，不是真正"方向对齐"

cos θ 直接测"方向匹配度"，无偏置地刻画量化器的方向信息保留能力。
"""

import sys
sys.path.insert(0, '/home/wlia0047/ar57/wenyu/GeneRec/GRID')

# Pre-import to break circular import
import src.utils.decorators  # noqa: F401

import json
import os
import numpy as np
import torch


def forward_residual(x, codebooks, gains=None):
    r_lst = [x.clone()]
    q_lst, idx_lst = [], []
    L = len(codebooks)
    for l in range(L):
        r = r_lst[-1]
        C = codebooks[l]
        if gains is not None and gains[l] is not None:
            eff_C = C * gains[l].unsqueeze(-1)
        else:
            eff_C = C
        r_norm2 = (r ** 2).sum(-1, keepdim=True)
        C_norm2 = (eff_C ** 2).sum(-1)
        d2 = r_norm2 - 2 * (r @ eff_C.T) + C_norm2
        idx = d2.argmin(dim=1)
        q = eff_C[idx]
        q_lst.append(q)
        idx_lst.append(idx)
        r_lst.append(r - q)
    return r_lst, q_lst, idx_lst


def compute_cos_theta(r_lst, q_lst, idx_lst):
    """Per-layer cos θ distribution."""
    L = len(r_lst) - 1
    out = []
    for l in range(L):
        r = r_lst[l]
        q = q_lst[l]
        r_norm = r.norm(dim=-1)
        q_norm = q.norm(dim=-1)
        # Avoid division by near-zero norms (degenerate residuals)
        denom = (r_norm * q_norm).clamp(min=1e-9)
        cos_theta = ((r * q).sum(dim=-1) / denom).clamp(-1, 1)
        cos_theta_np = cos_theta.cpu().numpy()

        # Bin distribution
        bins = {
            '<-0.5':  float((cos_theta < -0.5).float().mean().item()),
            '[-0.5, -0.1)': float(((cos_theta >= -0.5) & (cos_theta < -0.1)).float().mean().item()),
            '[-0.1, 0.1)':  float(((cos_theta >= -0.1) & (cos_theta < 0.1)).float().mean().item()),
            '[0.1, 0.5)':   float(((cos_theta >= 0.1) & (cos_theta < 0.5)).float().mean().item()),
            '[0.5, 0.9)':   float(((cos_theta >= 0.5) & (cos_theta < 0.9)).float().mean().item()),
            '≥0.9':   float((cos_theta >= 0.9).float().mean().item()),
        }

        out.append({
            'l': l + 1,
            'cos_theta_mean':   float(cos_theta.mean().item()),
            'cos_theta_median': float(np.median(cos_theta_np)),
            'cos_theta_std':    float(cos_theta.std().item()),
            'cos_theta_p10':    float(np.quantile(cos_theta_np, 0.10)),
            'cos_theta_p25':    float(np.quantile(cos_theta_np, 0.25)),
            'cos_theta_p75':    float(np.quantile(cos_theta_np, 0.75)),
            'cos_theta_p90':    float(np.quantile(cos_theta_np, 0.90)),
            'frac_aligned_high':  float((cos_theta > 0.5).float().mean().item()),  # well-aligned
            'frac_orthogonal':    float((cos_theta.abs() < 0.1).float().mean().item()),
            'frac_anti_aligned':  float((cos_theta < -0.5).float().mean().item()),
            'distribution_bins':  bins,
        })
    return out


def load_codebooks(ckpt_path):
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    sd = ckpt.get('state_dict', ckpt)
    L = sum(1 for k in sd.keys()
            if k.startswith('quantization_layer_list.') and k.endswith('.centroids'))
    codebooks = [sd[f'quantization_layer_list.{l}.centroids'].float() for l in range(L)]
    gains = []
    has_gains = False
    for l in range(L):
        gk = f'quantization_layer_list.{l}.cluster_gains'
        if gk in sd:
            has_gains = True
            gains.append(sd[gk].float())
        else:
            gains.append(None)
    return codebooks, gains, has_gains


def main(algorithms, emb_path, out_dir):
    print(f"=== Loading embedding: {emb_path}")
    X = torch.load(emb_path, map_location='cpu', weights_only=False)
    print(f"  shape={tuple(X.shape)}, dtype={X.dtype}")

    all_results = {}
    for name, ckpt_path in algorithms.items():
        print(f"\n=== {name}: {ckpt_path} ===")
        if not os.path.exists(ckpt_path):
            print(f"  ckpt NOT FOUND, skipping")
            all_results[name] = {'error': f'ckpt missing: {ckpt_path}'}
            continue
        codebooks, gains, has_gains = load_codebooks(ckpt_path)
        print(f"  has_gains={has_gains}, codebook shapes={[list(c.shape) for c in codebooks]}")
        r_lst, q_lst, idx_lst = forward_residual(X, codebooks, gains if has_gains else None)
        layers = compute_cos_theta(r_lst, q_lst, idx_lst)
        result = {
            'algorithm': name,
            'ckpt': ckpt_path,
            'has_gains': has_gains,
            'layers': layers,
        }
        all_results[name] = result
        print(f"  {'L':<3} {'mean':<8} {'median':<8} {'std':<8} {'p25':<8} {'p75':<8} {'frac>0.5':<10} {'frac<0.1':<10} {'frac<-0.5':<10}")
        for lyr in layers:
            print(f"  {lyr['l']:<3} {lyr['cos_theta_mean']:<8.4f} {lyr['cos_theta_median']:<8.4f} "
                  f"{lyr['cos_theta_std']:<8.4f} {lyr['cos_theta_p25']:<8.4f} {lyr['cos_theta_p75']:<8.4f} "
                  f"{lyr['frac_aligned_high']:<10.4f} {lyr['frac_orthogonal']:<10.4f} {lyr['frac_anti_aligned']:<10.4f}")

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'task4_cos_theta.json')
    with open(out_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\n=== Saved → {out_path} ===")


if __name__ == '__main__':
    EMB = '/home/wlia0047/ar57/wenyu/GeneRec/GRID/logs/inference/runs/2026-07-07/18-51-46/pickle/merged_predictions_tensor.pt'
    OUT = '/home/wlia0047/ar57/wenyu/GeneRec/GRID/result/task14_unified_rerun'

    ALGOS = {
        'A_baseline': '/home/wlia0047/ar57/wenyu/GeneRec/GRID/logs/train/runs/task3_group_a_s21/checkpoints/checkpoint_000_003000.ckpt',
        'B_mmq':      '/home/wlia0047/ar57/wenyu/GeneRec/GRID/logs/train/runs/2026-07-11/13-15-28/checkpoints/checkpoint_000_003000.ckpt',
        'C_gsrq':     '/home/wlia0047/ar57/wenyu/GeneRec/GRID/logs/train/runs/task3_group_c_s21/checkpoints/checkpoint_000_003000.ckpt',
    }

    main(ALGOS, EMB, OUT)