#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Task #230 P2 — 单 ckpt 钉死全套基础测量 (CPU only, no GPU).

目的 (用户 2026-07-28 P2 设计):
  一个 ckpt (记 sha256), 一次性测全所有基础量. 后续所有计算只引用这一份.

测量项 (per layer, 全部 3 层 L0/L1/L2):
  1. 切空间 ‖e‖:min / p10 / p50 / p90 / max
  2. 球空间 ‖e^H‖ = expmap0(e):min / p10 / p50 / p90 / max
  3. 双曲边长 ρ:用 Eq1 直接算 d_B(origin, exp(e)) — **不**从范数推
  4. 码字两两**夹角**(cos):min / p1 / p5 / p50 — **重点, 从没测过**
  5. D_eff (PCA participation ratio)
  6. max_c2 (明确定义为 max c·‖e‖², 平方范数口径)

数据源: baseline c=1 ckpt (Task #84 best_collision_model.pth epoch 59, sha256=24d25501...)
        + 同目录 epoch_79 / best_loss_model.pth (sha256=59a38fa3...) 作为 drift 对照
        + 后续还可选 c=10 ep1 (Task #233) 作 product_manifold 对照

输出: descriptions/task230_p2_codebook_geometry.json + 控制台汇总

夹角那项最重要 — 它决定天花板 θ/2, 进而决定临界曲率.
目前用的 59.3°/56.6°/53.9° 是随机放置估计, 实测只会更差, 临界曲率只会更高.
"""
from __future__ import annotations

import hashlib
import json
import sys

import numpy as np
import torch

sys.path.insert(0, '/home/wlia0047/ar57/wenyu/GeneRec/HG-Rec')

from model.hrqvae import HRQVAE
from model.utils import EmbDataset


def sha256_of_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def md5_of_file(path: str) -> str:
    h = hashlib.md5()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def load_codebook_weights(ckpt_path: str):
    """Load only codebook embedding weights from ckpt."""
    ckpt = torch.load(ckpt_path, weights_only=False, map_location=torch.device('cpu'))
    ckpt_args = vars(ckpt['args']) if not isinstance(ckpt['args'], dict) else ckpt['args']
    state_dict = ckpt['state_dict']

    # Build model skeleton (no need to load state_dict beyond codebook)
    r_target_list_raw = ckpt_args.get('r_target_list', None)
    if isinstance(r_target_list_raw, str) and r_target_list_raw:
        r_target_list = [float(x) for x in r_target_list_raw.split(',')]
    else:
        r_target_list = r_target_list_raw

    model = HRQVAE(
        in_dim=ckpt_args.get('in_dim', 768),
        num_emb_list=ckpt_args['num_emb_list'],
        e_dim=ckpt_args['e_dim'],
        layers=ckpt_args['layers'],
        dropout_prob=ckpt_args.get('dropout_prob', 0.0),
        bn=ckpt_args.get('bn', False),
        loss_type=ckpt_args['loss_type'],
        quant_loss_weight=ckpt_args.get('quant_loss_weight', 1.0),
        beta=ckpt_args['beta'],
        kmeans_init=ckpt_args.get('kmeans_init', False),
        kmeans_iters=ckpt_args.get('kmeans_iters', 100),
        sk_eps=ckpt_args['sk_epsilons'],
        sk_iters=ckpt_args['sk_iters'],
        product_manifold=ckpt_args.get('product_manifold', False),
        angular_dim=ckpt_args.get('angular_dim', None),
        radial_dim=ckpt_args.get('radial_dim', None),
        kappa_mode=ckpt_args.get('kappa_mode', 'fixed'),
        theta_init=ckpt_args.get('theta_init', 0.0),
        r_target_list=r_target_list,
    )

    # Extract codebook weights (only vq embedding)
    codebooks = []
    for li, vq in enumerate(model.hrq.vq_layers):
        keys = [k for k in state_dict.keys() if f'vq_layers.{li}.' in k and 'embedding' in k]
        if keys:
            w = state_dict[keys[0]].detach().cpu().numpy()
        else:
            w = vq.embedding.weight.detach().cpu().numpy()
        codebooks.append(w)
    return codebooks, ckpt, ckpt_args


def poincare_expmap0(v: np.ndarray, c: float = 1.0) -> np.ndarray:
    """Exponential map at origin (HG-Rec Nickel-Kiela 公式, NOT /2):
    e^H = tanh(√c · ‖v‖) · v / (√c · ‖v‖)
    For c=1: ‖e^H‖ = tanh(‖v‖).
    """
    sqrt_c = np.sqrt(c)
    v_norm = np.linalg.norm(v, axis=-1, keepdims=True)
    safe_norm = np.where(v_norm < 1e-12, 1e-12, v_norm)
    factor = np.tanh(sqrt_c * safe_norm) / (sqrt_c * safe_norm)
    return factor * v


def poincare_distance(x: np.ndarray, y: np.ndarray, c: float = 1.0) -> np.ndarray:
    """Poincaré distance d(x, y) = (2/√c) arctanh(√c · ‖(-x) ⊕ y‖).
    For origin: d(origin, y) = 2 arctanh(√c · ‖y‖) / √c.
    For c=1: d(origin, y) = 2 arctanh(‖y‖).
    """
    sqrt_c = np.sqrt(c)
    y_norm = np.linalg.norm(y, axis=-1)
    safe_norm = np.clip(sqrt_c * y_norm, 0, 1.0 - 1e-9)
    return 2.0 * np.arctanh(safe_norm) / sqrt_c


def hyperbolic_radius_from_expmap(e_norm_h: np.ndarray, c: float = 1.0) -> np.ndarray:
    """Inverse: from球空间 ‖e^H‖ to ρ.
    ρ = 2 arctanh(√c · ‖e^H‖) / √c. For c=1: ρ = 2 arctanh(‖e^H‖).
    """
    sqrt_c = np.sqrt(c)
    safe_arg = np.clip(sqrt_c * e_norm_h, 0, 1.0 - 1e-9)
    return 2.0 * np.arctanh(safe_arg) / sqrt_c


def pairwise_cosines(X: np.ndarray) -> np.ndarray:
    """All pairwise cosines (excluding diagonal)."""
    norms = np.linalg.norm(X, axis=-1, keepdims=True)
    safe_norms = np.where(norms < 1e-12, 1e-12, norms)
    X_normalized = X / safe_norms
    sim = X_normalized @ X_normalized.T  # (n, n)
    # Exclude diagonal
    n = sim.shape[0]
    mask = ~np.eye(n, dtype=bool)
    return sim[mask]


def participation_ratio(X: np.ndarray) -> float:
    """PCA participation ratio D_eff = (Σ λ)² / Σ λ²."""
    X_centered = X - X.mean(axis=0, keepdims=True)
    cov = np.cov(X_centered, rowvar=False)
    eigvals = np.maximum(np.linalg.eigvalsh(cov), 0)
    sum_lambda = eigvals.sum()
    sum_lambda_sq = (eigvals ** 2).sum()
    return float((sum_lambda ** 2) / sum_lambda_sq) if sum_lambda_sq > 0 else 0.0


def per_layer_full_geometry(cb_e: np.ndarray, c: float = 1.0) -> dict:
    """Full geometry diagnostic for a single layer's切空间 codebook."""
    n, d = cb_e.shape

    # 1. 切空间 ‖e‖
    e_norm = np.linalg.norm(cb_e, axis=-1)  # (n,)

    # 2. 球空间 ‖e^H‖ via expmap0
    e_h = poincare_expmap0(cb_e, c=c)
    e_h_norm = np.linalg.norm(e_h, axis=-1)  # = tanh(√c ‖e‖/2) / √c

    # 3. 双曲边长 ρ (从球空间 ‖e^H‖ 反推, 不是从切空间 ‖e‖ 反推)
    rho = hyperbolic_radius_from_expmap(e_h_norm, c=c)

    # 4. 码字两两夹角 (cos, 切空间)
    cosines = pairwise_cosines(cb_e)  # (n*(n-1),)

    # 5. D_eff
    d_eff = participation_ratio(cb_e)

    # 6. max_c2 = max c · ‖e_i‖² (平方范数口径)
    max_c2 = float(c * np.max(e_norm ** 2))
    p50_c2 = float(c * np.percentile(e_norm ** 2, 50))
    max_c2_over_p50 = max_c2 / p50_c2 if p50_c2 > 0 else 0

    # Eq1 直接算 d_B(origin, exp(e)) 验证 ρ 跟 d_B 一致
    origin = np.zeros_like(cb_e[:1])  # (1, d)
    e_h_first = e_h[:1]  # first codeword 球空间
    # 我们算全部 codeword 到 origin 的 d_B, 应该等于 ρ
    # 但这里只验证 first few
    d_b_first = poincare_distance(np.broadcast_to(origin, e_h.shape), e_h, c=c)[:5]
    rho_first = rho[:5]

    # 转换 cos → angle (度)
    angles_deg = np.degrees(np.arccos(np.clip(cosines, -1, 1)))

    # 临界曲率 (per codeword ceiling θ/2): 码字对半开
    # 半开角度 = ceil(p50 angle) — 邻居必须 ≤ 这角度
    angle_ceiling_p50 = float(np.percentile(angles_deg, 50))

    # 用户 2026-07-28 两因子公式: c = (4/ρ²) × [arcsinh(cot(θ/2))]²
    rho_p50_val = float(np.percentile(rho, 50))
    theta_min_deg = float(angles_deg.min())
    theta_p1_deg = float(np.percentile(angles_deg, 1))
    theta_p5_deg = float(np.percentile(angles_deg, 5))
    theta_p50_deg = float(np.percentile(angles_deg, 50))

    def critical_c(rho_val, theta_deg):
        if theta_deg <= 0 or rho_val <= 0:
            return float('inf')
        return float((4.0 / rho_val ** 2) * (np.arcsinh(1.0 / np.tan(np.radians(theta_deg) / 2.0))) ** 2)

    c_at_min = critical_c(rho_p50_val, theta_min_deg)
    c_at_p1 = critical_c(rho_p50_val, theta_p1_deg)
    c_at_p5 = critical_c(rho_p50_val, theta_p5_deg)
    c_at_p50 = critical_c(rho_p50_val, theta_p50_deg)

    def grad_remain_pct(t):
        return float(4.0 * t / (1.0 + t) ** 2 * 100.0)

    cnorm_at_c_min = c_at_min * p50_c2
    cnorm_at_c_p1 = c_at_p1 * p50_c2
    cnorm_at_c_p5 = c_at_p5 * p50_c2
    cnorm_at_c_p50 = c_at_p50 * p50_c2

    safe_c_from_max_c2 = 0.5 / max_c2 if max_c2 > 0 else float('inf')

    return {
        'n_codewords': int(n),
        'e_dim': int(d),
        '切空间_‖e‖': {
            'min': float(e_norm.min()),
            'p10': float(np.percentile(e_norm, 10)),
            'p50': float(np.percentile(e_norm, 50)),
            'p90': float(np.percentile(e_norm, 90)),
            'max': float(e_norm.max()),
            'mean': float(e_norm.mean()),
            'std': float(e_norm.std()),
        },
        '球空间_‖e^H‖': {
            'min': float(e_h_norm.min()),
            'p10': float(np.percentile(e_h_norm, 10)),
            'p50': float(np.percentile(e_h_norm, 50)),
            'p90': float(np.percentile(e_h_norm, 90)),
            'max': float(e_h_norm.max()),
            'mean': float(e_h_norm.mean()),
            'std': float(e_h_norm.std()),
        },
        '双曲边长_ρ': {
            'min': float(rho.min()),
            'p10': float(np.percentile(rho, 10)),
            'p50': float(np.percentile(rho, 50)),
            'p90': float(np.percentile(rho, 90)),
            'max': float(rho.max()),
            'mean': float(rho.mean()),
            'std': float(rho.std()),
        },
        '码字两两夹角_cos': {
            'min': float(cosines.min()),
            'p0_1': float(np.percentile(cosines, 0.1)),
            'p1': float(np.percentile(cosines, 1)),
            'p5': float(np.percentile(cosines, 5)),
            'p50': float(np.percentile(cosines, 50)),
            'max': float(cosines.max()),
            'mean': float(cosines.mean()),
        },
        '码字两两夹角_角度_度': {
            'min': float(angles_deg.min()),
            'p0_1': float(np.percentile(angles_deg, 0.1)),
            'p1': float(np.percentile(angles_deg, 1)),
            'p5': float(np.percentile(angles_deg, 5)),
            'p50': float(np.percentile(angles_deg, 50)),
            'max': float(angles_deg.max()),
            'mean': float(angles_deg.mean()),
            'ceiling_theta_over_2_p50_deg': angle_ceiling_p50,
        },
        'D_eff_participation_ratio': float(d_eff),
        'max_c2': {
            'definition': 'max(c · ‖e‖²), 平方范数口径',
            'value': max_c2,
            'p50': p50_c2,
            'max_over_p50': max_c2_over_p50,
            'safe_c_from_max_c2_0_5': safe_c_from_max_c2,
        },
        'Eq1_vs_ρ_验证': {
            '说明': '前 5 个 codeword: d_B(origin, exp(e)) 应等于 ρ',
            'rho_first5': rho_first.tolist(),
            'd_b_origin_to_exp_e_first5': d_b_first.tolist(),
            'max_abs_diff': float(np.max(np.abs(d_b_first - rho_first))),
        },
        'critical_c_两因子分解': {
            '公式': 'c = (4/ρ²) × [arcsinh(cot(θ/2))]²',
            'rho_p50_used': rho_p50_val,
            'theta_min_deg': theta_min_deg,
            'theta_p1_deg': theta_p1_deg,
            'theta_p5_deg': theta_p5_deg,
            'theta_p50_deg': theta_p50_deg,
            'c_critical_θ_min': c_at_min,
            'c_critical_θ_p1': c_at_p1,
            'c_critical_θ_p5': c_at_p5,
            'c_critical_θ_p50': c_at_p50,
            'c_times_‖e‖²_p50_at_θ_min': cnorm_at_c_min,
            'c_times_‖e‖²_p50_at_θ_p1': cnorm_at_c_p1,
            'c_times_‖e‖²_p50_at_θ_p5': cnorm_at_c_p5,
            'c_times_‖e‖²_p50_at_θ_p50': cnorm_at_c_p50,
            'gradient_remaining_pct_at_θ_min': grad_remain_pct(cnorm_at_c_min),
            'gradient_remaining_pct_at_θ_p1': grad_remain_pct(cnorm_at_c_p1),
            'gradient_remaining_pct_at_θ_p5': grad_remain_pct(cnorm_at_c_p5),
            'gradient_remaining_pct_at_θ_p50': grad_remain_pct(cnorm_at_c_p50),
            'safe_c_from_max_c2_0_5_警戒线': safe_c_from_max_c2,
            '需求超上限倍数_at_θ_p1': c_at_p1 / safe_c_from_max_c2 if safe_c_from_max_c2 > 0 else float('inf'),
            '需求超上限倍数_at_θ_min': c_at_min / safe_c_from_max_c2 if safe_c_from_max_c2 > 0 else float('inf'),
        },
    }


def main():
    ckpts = {
        'baseline_c1_ep59': '/home/wlia0047/ar57/wenyu/GeneRec/products/task84/ckpt/Instruments/Jul-23-2026_20-08-06_beta_1.000_codebook_[64,128,256]_sk_0.000/best_collision_model.pth',
        'baseline_c1_ep79_best_loss': '/home/wlia0047/ar57/wenyu/GeneRec/products/task84/ckpt/Instruments/Jul-23-2026_20-08-06_beta_1.000_codebook_[64,128,256]_sk_0.000/best_loss_model.pth',
        'c10_ep1_dirI': '/home/wlia0047/ar57/wenyu/GeneRec/products/m_arm/m_radius_spread_step3_50ep_wdiv100_c10_jul-27-2026_16-18-09/Jul-27-2026_16-18-16_beta_0.500_codebook_[64,128,256]_sk_0.000/best_collision_model.pth',
    }

    # Step 1: SHA256 钉死 (P2 强制要求)
    print("=" * 80)
    print("P2 Step 1 — SHA256 + MD5 钉死 (后续计算只引用这一份)")
    print("=" * 80)
    ckpt_hashes = {}
    for name, path in ckpts.items():
        if not __import__('os').path.exists(path):
            print(f"  ⚠️  {name}: NOT FOUND, skip")
            continue
        sha = sha256_of_file(path)
        md5 = md5_of_file(path)
        ckpt_hashes[name] = {'path': path, 'sha256': sha, 'md5': md5}
        print(f"  [{name}]")
        print(f"    path: {path}")
        print(f"    sha256: {sha}")
        print(f"    md5: {md5}")

    # Step 2: 全套几何测量
    print()
    print("=" * 80)
    print("P2 Step 2 — 全套几何测量 (3 ckpt × 3 层 = 9 套)")
    print("=" * 80)

    all_results = {'ckpt_hashes': ckpt_hashes, 'per_ckpt': {}}

    for arm_name, path in ckpts.items():
        if arm_name not in ckpt_hashes:
            continue
        print(f"\n>>> [{arm_name}]")
        # 默认 c=1 baseline, c=10 ep1 用 c=10
        c = 10.0 if 'c10' in arm_name else 1.0
        print(f"  c = {c}")

        codebooks, ckpt, ckpt_args = load_codebook_weights(path)
        print(f"  ckpt epoch = {ckpt.get('epoch', '?')}, "
              f"best_loss = {ckpt.get('best_loss', '?')}, "
              f"best_collision = {ckpt.get('best_collision_rate', '?')}")
        print(f"  num_emb_list = {ckpt_args['num_emb_list']}, "
              f"e_dim = {ckpt_args['e_dim']}, "
              f"r_target_list = {ckpt_args.get('r_target_list')}, "
              f"w_div = {ckpt_args.get('w_div')}, "
              f"w_angular = {ckpt_args.get('w_angular')}, "
              f"kappa_mode = {ckpt_args.get('kappa_mode')}")

        per_layer_data = {}
        for li, cb in enumerate(codebooks):
            print(f"\n  Layer {li} (n={cb.shape[0]}, d={cb.shape[1]}):")
            layer_geom = per_layer_full_geometry(cb, c=c)

            print(f"    切空间 ‖e‖: min={layer_geom['切空间_‖e‖']['min']:.4f}, "
                  f"p50={layer_geom['切空间_‖e‖']['p50']:.4f}, "
                  f"max={layer_geom['切空间_‖e‖']['max']:.4f}")
            print(f"    球空间 ‖e^H‖: min={layer_geom['球空间_‖e^H‖']['min']:.4f}, "
                  f"p50={layer_geom['球空间_‖e^H‖']['p50']:.4f}, "
                  f"max={layer_geom['球空间_‖e^H‖']['max']:.4f}")
            print(f"    双曲边长 ρ: min={layer_geom['双曲边长_ρ']['min']:.4f}, "
                  f"p50={layer_geom['双曲边长_ρ']['p50']:.4f}, "
                  f"max={layer_geom['双曲边长_ρ']['max']:.4f}")
            print(f"    码字两两 cos: min={layer_geom['码字两两夹角_cos']['min']:.4f}, "
                  f"p1={layer_geom['码字两两夹角_cos']['p1']:.4f}, "
                  f"p5={layer_geom['码字两两夹角_cos']['p5']:.4f}, "
                  f"p50={layer_geom['码字两两夹角_cos']['p50']:.4f}")
            print(f"    码字两两 角度(°): min={layer_geom['码字两两夹角_角度_度']['min']:.2f}, "
                  f"p1={layer_geom['码字两两夹角_角度_度']['p1']:.2f}, "
                  f"p5={layer_geom['码字两两夹角_角度_度']['p5']:.2f}, "
                  f"p50={layer_geom['码字两两夹角_角度_度']['p50']:.2f}, "
                  f"max={layer_geom['码字两两夹角_角度_度']['max']:.2f}")
            print(f"    D_eff = {layer_geom['D_eff_participation_ratio']:.3f}")
            print(f"    max_c2 = {layer_geom['max_c2']['value']:.4f} "
                  f"(c·max‖e‖²), max/p50 = {layer_geom['max_c2']['max_over_p50']:.3f}")
            print(f"    Eq1 验证: max|ρ - d_B(origin, exp(e))| = "
                  f"{layer_geom['Eq1_vs_ρ_验证']['max_abs_diff']:.6e} "
                  f"({'✓ 一致' if layer_geom['Eq1_vs_ρ_验证']['max_abs_diff'] < 1e-5 else '✗ 不一致'})")

            per_layer_data[f'layer_{li}'] = layer_geom

        all_results['per_ckpt'][arm_name] = {
            'ckpt_path': path,
            'sha256': ckpt_hashes[arm_name]['sha256'],
            'ckpt_epoch': int(ckpt.get('epoch', -1)),
            'best_loss': float(ckpt.get('best_loss', -1)),
            'best_collision_rate': float(ckpt.get('best_collision_rate', -1)),
            'c_value': c,
            'per_layer': per_layer_data,
        }

    # Summary
    print()
    print("=" * 80)
    print("P2 Summary — 夹角 (临界曲率天花板) 对比")
    print("=" * 80)
    print(f"{'ckpt':<28} {'L':<3} {'n':<5} {'angle_p1(°)':<12} {'angle_p5(°)':<12} "
          f"{'angle_p50(°)':<12} {'D_eff':<8} {'max_c2':<10}")
    print("-" * 95)
    for arm_name, arm_data in all_results['per_ckpt'].items():
        for li_key, layer_data in arm_data['per_layer'].items():
            li = int(li_key.split('_')[1])
            angles = layer_data['码字两两夹角_角度_度']
            d_eff = layer_data['D_eff_participation_ratio']
            max_c2 = layer_data['max_c2']['value']
            print(f"{arm_name:<28} L{li:<2} {layer_data['n_codewords']:<5} "
                  f"{angles['p1']:<12.2f} {angles['p5']:<12.2f} "
                  f"{angles['p50']:<12.2f} {d_eff:<8.3f} {max_c2:<10.4f}")

    # Save
    output_path = '/home/wlia0047/ar57/wenyu/GeneRec/descriptions/task230_p2_codebook_geometry.json'
    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\n✅ Saved: {output_path}")


if __name__ == '__main__':
    main()