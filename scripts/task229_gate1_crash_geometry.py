#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Task #229 Gate 1 — 6-ckpt codebook 几何全套测量 (CPU only).

目的: 验证 verdict #230 预测的崩溃机制 — 高 c → ‖e‖ → 0 (L0 范数最大最先崩)
+ max_c2 (c·‖e‖²) 跨过饱和阈值 → argmin 退化成"只看范数".

数据源 (全部 200 epoch, 干净 config: kappa_mode=fixed, r_target=None, w_div=w_angular=0):
  - c=1:    products/task229/gate1_c1_baseline/.../best_collision_model.pth
  - c=3:    products/task229/gate1_c3/.../best_collision_model.pth
  - c=10:   products/task229/gate1_c10/.../best_collision_model.pth
  - c=30:   products/task229/gate1_c30/.../best_collision_model.pth
  - c=100:  products/task229/gate1_c100/.../best_collision_model.pth
  - c=300:  products/task229/gate1_c300/.../best_collision_model.pth

输出: descriptions/task229_gate1_crash_geometry.json
"""
from __future__ import annotations

import json
import sys

import numpy as np
import torch

sys.path.insert(0, '/home/wlia0047/ar57/wenyu/GeneRec/HG-Rec')

from model.hrqvae import HRQVAE
from model.utils import EmbDataset


def load_codebooks_from_ckpt(ckpt_path: str):
    ckpt = torch.load(ckpt_path, weights_only=False, map_location=torch.device('cpu'))
    ckpt_args = vars(ckpt['args']) if not isinstance(ckpt['args'], dict) else ckpt['args']
    state_dict = ckpt['state_dict']

    data = EmbDataset(ckpt_args['data_path'])

    r_target_list_raw = ckpt_args.get('r_target_list', None)
    if isinstance(r_target_list_raw, str) and r_target_list_raw:
        r_target_list = [float(x) for x in r_target_list_raw.split(',')]
    else:
        r_target_list = r_target_list_raw

    model = HRQVAE(
        in_dim=data.dim,
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

    codebooks = []
    for li, vq in enumerate(model.hrq.vq_layers):
        keys = [k for k in state_dict.keys() if f'vq_layers.{li}.' in k and 'embedding' in k]
        if keys:
            w = state_dict[keys[0]].detach().cpu().numpy()
        else:
            w = vq.embedding.weight.detach().cpu().numpy()
        codebooks.append(w)
    return codebooks, ckpt, ckpt_args


def per_layer_geometry(codebooks, c_value):
    """对每个 codebook 测 ‖e‖ 切空间分布 + max_c2."""
    out = {}
    for li, cb in enumerate(codebooks):
        norms = np.linalg.norm(cb, axis=1)  # 切空间范数
        norms_sq = norms ** 2
        out[f'L{li}'] = {
            'n': int(cb.shape[0]),
            'd': int(cb.shape[1]),
            'norm_p10': float(np.percentile(norms, 10)),
            'norm_p50': float(np.percentile(norms, 50)),
            'norm_p90': float(np.percentile(norms, 90)),
            'norm_max': float(norms.max()),
            'norm_min': float(norms.min()),
            'norm_mean': float(norms.mean()),
            'norm_std': float(norms.std()),
            'norm_sq_max': float(norms_sq.max()),
            'norm_sq_mean': float(norms_sq.mean()),
            'cnorm_max': float(c_value * norms_sq.max()),
            'cnorm_mean': float(c_value * norms_sq.mean()),
            # c·‖e‖² > 4 → argmin 退化成"只看范数" (verdict #230 预测)
            'saturation_pct': float((c_value * norms_sq > 4.0).mean() * 100),
            'gradient_remaining_pct': float(
                (4.0 * np.sqrt(c_value * norms_sq) / (1.0 + c_value * norms_sq) ** 2).mean() * 100
            ),
        }
    return out


def main():
    ckpts = {
        'c1':   '/home/wlia0047/ar57/wenyu/GeneRec/products/task229/gate1_c1_baseline/Jul-28-2026_14-27-14_beta_0.500_codebook_[64,128,256]_sk_0.000/best_collision_model.pth',
        'c3':   '/home/wlia0047/ar57/wenyu/GeneRec/products/task229/gate1_c3/Jul-28-2026_14-27-14_beta_0.500_codebook_[64,128,256]_sk_0.000/best_collision_model.pth',
        'c10':  '/home/wlia0047/ar57/wenyu/GeneRec/products/task229/gate1_c10/Jul-28-2026_14-27-14_beta_0.500_codebook_[64,128,256]_sk_0.000/best_collision_model.pth',
        'c30':  '/home/wlia0047/ar57/wenyu/GeneRec/products/task229/gate1_c30/Jul-28-2026_14-27-14_beta_0.500_codebook_[64,128,256]_sk_0.000/best_collision_model.pth',
        'c100': '/home/wlia0047/ar57/wenyu/GeneRec/products/task229/gate1_c100/Jul-28-2026_14-29-37_beta_0.500_codebook_[64,128,256]_sk_0.000/best_collision_model.pth',
        'c300': '/home/wlia0047/ar57/wenyu/GeneRec/products/task229/gate1_c300/Jul-28-2026_14-29-37_beta_0.500_codebook_[64,128,256]_sk_0.000/best_collision_model.pth',
    }

    c_values = {'c1': 1.0, 'c3': 3.0, 'c10': 10.0, 'c30': 30.0, 'c100': 100.0, 'c300': 300.0}

    all_results = {}

    for arm_name, ckpt_path in ckpts.items():
        c_val = c_values[arm_name]
        print(f"\n=== [{arm_name}] c={c_val} Loading ckpt: {ckpt_path.split('/')[-3]} ===")
        codebooks, ckpt, ckpt_args = load_codebooks_from_ckpt(ckpt_path)

        epoch = ckpt.get('epoch', '?')
        best_loss = ckpt.get('best_loss', '?')
        best_coll = ckpt.get('best_collision_rate', '?')

        print(f"  epoch={epoch} best_loss={best_loss:.4f} best_collision={best_coll:.4f}")

        geom = per_layer_geometry(codebooks, c_val)

        all_results[arm_name] = {
            'c_value': c_val,
            'ckpt_path': ckpt_path,
            'epoch': int(epoch) if epoch != '?' else None,
            'best_loss': float(best_loss),
            'best_collision_rate': float(best_coll),
            'per_layer': geom,
        }

        for li_key, g in geom.items():
            print(f"  {li_key} (n={g['n']}): ‖e‖ p50={g['norm_p50']:.4f} max={g['norm_max']:.4f}, "
                  f"c·‖e‖²_max={g['cnorm_max']:.3f}, "
                  f"saturation%={g['saturation_pct']:.1f}, "
                  f"grad_remain%={g['gradient_remaining_pct']:.2f}")

    # Summary table
    print("\n" + "=" * 100)
    print(f"{'Arm':<6} {'c':<6} {'best_coll':<10} {'L0‖e‖_max':<10} {'L0 c·‖e‖²_max':<14} {'L0 sat%':<8} {'L1 c·‖e‖²_max':<14} {'L1 sat%':<8} {'L2 c·‖e‖²_max':<14} {'L2 sat%':<8}")
    print("-" * 100)
    for arm_name, r in all_results.items():
        l0 = r['per_layer']['L0']
        l1 = r['per_layer']['L1']
        l2 = r['per_layer']['L2']
        print(f"{arm_name:<6} {r['c_value']:<6.0f} {r['best_collision_rate']:<10.4f} "
              f"{l0['norm_max']:<10.4f} {l0['cnorm_max']:<14.3f} {l0['saturation_pct']:<8.1f} "
              f"{l1['cnorm_max']:<14.3f} {l1['saturation_pct']:<8.1f} "
              f"{l2['cnorm_max']:<14.3f} {l2['saturation_pct']:<8.1f}")

    output_path = '/home/wlia0047/ar57/wenyu/GeneRec/descriptions/task229_gate1_crash_geometry.json'
    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\n✅ Saved: {output_path}")


if __name__ == '__main__':
    main()