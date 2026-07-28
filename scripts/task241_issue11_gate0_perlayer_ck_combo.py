#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Task #241 / Issue #11 Gate 0 — Per-layer c_k range 组合性验证.

配置: Task #84 baseline ckpt (frozen) + 逐层 c_k 区间:
  - L0: PC κ, c_k ~ U(1, 5)        (task220 最优)
  - L1: PC κ, c_k ~ U(0.5, 20)     (task231 最优)
  - L2: PC κ, c_k ~ U(0.5, 20)     (task231 最优)

3 seeds (42/43/44), 期望三层一致率 82.68 / 67.15 / 75.69.
通过: 三层全 OPEN (60-90%) + 全 < 0.90.
硬停止: 任一层不在 OPEN → H1 证伪 → STOP.
"""
import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
log = logging.getLogger("task241_issue11_gate0")

sys.path.insert(0, '/home/wlia0047/ar57/wenyu/GeneRec/HG-Rec')


def load_baseline_ckpt(ckpt_path, device='cpu'):
    from model.hrqvae import HRQVAE
    from model.utils import EmbDataset
    log.info(f"Loading ckpt: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=torch.device('cpu'), weights_only=False)
    args = ckpt['args']
    log.info(f"  num_emb_list={args.num_emb_list}, e_dim={args.e_dim}, "
             f"curvature={getattr(args, 'curvature', 1.0)}, "
             f"product_manifold={getattr(args, 'product_manifold', False)}")
    data = EmbDataset(args.data_path)
    log.info(f"  Data: {len(data)} items, dim={data.dim}")
    model = HRQVAE(
        in_dim=data.dim,
        num_emb_list=args.num_emb_list,
        e_dim=args.e_dim,
        layers=getattr(args, 'layers', [512, 256, 128, args.e_dim]),
        dropout_prob=getattr(args, 'dropout_prob', 0.0),
        bn=getattr(args, 'bn', False),
        loss_type=args.loss_type,
        quant_loss_weight=getattr(args, 'quant_loss_weight', 1.0),
        beta=args.beta,
        kmeans_init=False,
        kmeans_iters=100,
        sk_eps=args.sk_epsilons,
        sk_iters=args.sk_iters,
    )
    model.load_state_dict(ckpt['state_dict'], strict=False)
    model = model.to(device).eval()
    return model, data, args


@torch.no_grad()
def encode_all(model, data, batch_size=256, device='cpu'):
    from torch.utils.data import DataLoader
    loader = DataLoader(data, batch_size=batch_size, shuffle=False, num_workers=2)
    latents = []
    for batch in loader:
        batch = batch.to(device)
        z = model.encoder(batch)
        latents.append(z.cpu())
    return torch.cat(latents, dim=0)


def euc_dist(z, e):
    z_sq = (z ** 2).sum(dim=-1, keepdim=True)
    e_sq = (e ** 2).sum(dim=-1, keepdim=True).t()
    ze = z @ e.t()
    return z_sq + e_sq - 2 * ze


def per_codeword_kappa_stereographic(z, e, c_k):
    """Per-codeword κ-Stereographic distance matrix (N, K)."""
    eps = 1e-5
    z_norm = z.norm(dim=-1, keepdim=True).clamp(min=1e-12, max=1 - eps)
    e_norm = e.norm(dim=-1, keepdim=True).clamp(min=1e-12, max=1 - eps)
    z_in = z / z.norm(dim=-1, keepdim=True).clamp(min=1e-12) * z_norm
    e_in = e / e.norm(dim=-1, keepdim=True).clamp(min=1e-12) * e_norm
    z_sq = (z_in ** 2).sum(dim=-1, keepdim=True)
    e_sq = (e_in ** 2).sum(dim=-1, keepdim=True).t()
    ze = z_in @ e_in.t()
    eucl_d_sq = z_sq + e_sq - 2 * ze
    c_k_z_sq = c_k.view(1, -1) * z_sq
    c_k_e_sq = c_k.view(1, -1) * e_sq
    one_minus_cz = (1 - c_k_z_sq).clamp(min=1e-6)
    one_minus_ce = (1 - c_k_e_sq).clamp(min=1e-6)
    inner = 1 + 2 * c_k.view(1, -1) * eucl_d_sq / (one_minus_cz * one_minus_ce)
    inner = inner.clamp(min=1.0 + 1e-12)
    inv_sqrt_c = 1.0 / c_k.sqrt().clamp(min=1e-6)
    return inv_sqrt_c.view(1, -1) * torch.acosh(inner)


def per_layer_criterion(z_layer, e, c_k_range, n_seeds=3, seed_base=42):
    """Per-layer per-codeword argmin vs Euclidean argmin agreement."""
    N, e_dim = z_layer.shape
    K = e.shape[0]
    eps = 1e-5
    z_norm = z_layer.norm(dim=-1, keepdim=True).clamp(min=1e-12, max=1 - eps)
    z_in = z_layer / z_layer.norm(dim=-1, keepdim=True).clamp(min=1e-12) * z_norm
    e_norm = e.norm(dim=-1, keepdim=True).clamp(min=1e-12, max=1 - eps)
    e_in = e / e.norm(dim=-1, keepdim=True).clamp(min=1e-12) * e_norm

    d_euc = euc_dist(z_in, e_in)
    argmin_euc = d_euc.argmin(dim=-1)

    per_seed = []
    for s in range(n_seeds):
        torch.manual_seed(seed_base + s)
        np.random.seed(seed_base + s)
        c_k = torch.from_numpy(
            np.random.uniform(c_k_range[0], c_k_range[1], size=K)
        ).float()
        score = per_codeword_kappa_stereographic(z_in, e_in, c_k)
        argmin_pc = score.argmin(dim=-1)
        agreement = (argmin_pc == argmin_euc).float().mean().item()
        per_seed.append({
            'seed': seed_base + s,
            'c_k_min': float(c_k.min()),
            'c_k_max': float(c_k.max()),
            'c_k_mean': float(c_k.mean()),
            'agreement': agreement,
        })
        log.info(f"  seed {seed_base+s}: c_k=[{c_k.min():.3f},{c_k.max():.3f}] mean={c_k.mean():.3f}  agreement={agreement*100:.2f}%")

    mean_a = float(np.mean([p['agreement'] for p in per_seed]))
    std_a = float(np.std([p['agreement'] for p in per_seed]))
    return {
        'K': int(K),
        'c_k_range': list(c_k_range),
        'mean_agreement': mean_a,
        'std_agreement': std_a,
        'per_seed': per_seed,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_path", type=str,
                        default="/home/wlia0047/ar57/wenyu/GeneRec/products/task84/ckpt/Instruments/Jul-23-2026_20-08-06_beta_1.000_codebook_[64,128,256]_sk_0.000/best_loss_model.pth")
    parser.add_argument("--data_path", type=str,
                        default="/home/wlia0047/ar57/wenyu/GeneRec/HG-Rec/dataset/Instruments/item_emb.parquet")
    parser.add_argument("--n_seeds", type=int, default=3)
    parser.add_argument("--output_json", type=str,
                        default="/home/wlia0047/.claude/jobs/04ccf474/tmp/task241_issue11_gate0_results.json")
    args = parser.parse_args()

    log.info("=" * 70)
    log.info("Task #241 / Issue #11 Gate 0 — per-layer c_k 组合性")
    log.info("=" * 70)
    log.info("L0: c_k ~ U(1, 5) | L1: c_k ~ U(0.5, 20) | L2: c_k ~ U(0.5, 20)")
    log.info("期望: 82.68 / 67.15 / 75.69 (task231 独立测量)")
    log.info("通过: 三层全 OPEN (60-90%) + 全 < 0.90")

    # Per-layer c_k ranges per Issue #11 H1
    per_layer_ranges = [
        (1.0, 5.0),    # L0: task220 best
        (0.5, 20.0),   # L1: task231 best
        (0.5, 20.0),   # L2: task231 best
    ]
    expected = [82.68, 67.15, 75.69]

    model, data, ckpt_args = load_baseline_ckpt(args.ckpt_path)
    log.info("\n=== 编码所有 items ===")
    t0 = time.time()
    z_all = encode_all(model, data)
    log.info(f"  Latent shape: {z_all.shape}, encode time: {time.time()-t0:.1f}s")

    n_e_list = ckpt_args.num_emb_list
    log.info(f"\n=== Per-layer PC κ argmin 组合检查 (num_emb_list={n_e_list}) ===")
    results = {'per_layer': [], 'expected': expected, 'ranges': per_layer_ranges}
    for li in range(len(n_e_list)):
        if li == 0:
            z_layer = z_all
        else:
            # Use residual for layer l
            from torch.utils.data import DataLoader
            loader = DataLoader(data, batch_size=256, shuffle=False, num_workers=2)
            cur = z_all.clone()
            for lj in range(li):
                cb = model.hrq.vq_layers[lj].embeddings.weight.detach().cpu()
                d = euc_dist(cur, cb)
                idx = d.argmin(dim=-1)
                z_q = cb[idx]
                cur = cur - z_q
            eps = 1e-5
            cur_norm = cur.norm(dim=-1, keepdim=True).clamp(min=1e-12, max=1 - eps)
            z_layer = cur / cur.norm(dim=-1, keepdim=True).clamp(min=1e-12) * cur_norm

        cb = model.hrq.vq_layers[li].embeddings.weight.detach().cpu()
        K = cb.shape[0]
        log.info(f"\n--- L{li}: K={K}, c_k range = {per_layer_ranges[li]} ---")
        r = per_layer_criterion(z_layer, cb, per_layer_ranges[li], n_seeds=args.n_seeds)
        r['layer'] = li
        r['expected_pct'] = expected[li]
        r['delta_from_expected'] = r['mean_agreement']*100 - expected[li]
        results['per_layer'].append(r)

    # Summary
    log.info("\n" + "=" * 70)
    log.info("Issue #11 Gate 0 决策表")
    log.info("=" * 70)
    log.info(f"{'L':<3} {'K':<5} {'c_k_range':<14} {'agreement':<18} {'expected':<10} {'Δ':<7} {'verdict':<30}")
    gate0_passed = True
    all_in_open = True
    all_under_90 = True
    for r in results['per_layer']:
        m_a = r['mean_agreement']
        delta = r['delta_from_expected']
        in_open = 0.6 <= m_a <= 0.9
        under_90 = m_a < 0.9
        if not in_open:
            all_in_open = False
        if not under_90:
            all_under_90 = False
        if in_open and under_90:
            verdict = 'OPEN (in 60-90% + <0.90)'
        elif m_a > 0.95:
            verdict = 'FAIL (>95% euc-consistent)'
        elif m_a < 0.6:
            verdict = 'TOO_STRONG (<60%)'
        else:
            verdict = 'check'
        log.info(f"L{r['layer']:<2} {r['K']:<5} {str(r['c_k_range']):<14} "
                 f"{m_a*100:5.2f}% ± {r['std_agreement']*100:.2f}%   "
                 f"{r['expected_pct']:5.2f}%   {delta:+5.2f}pp  {verdict}")

    # Issue #11 hard stop check
    if not all_in_open:
        log.info("\n❌ Issue #11 Gate 0 FAIL: ∃ layer not in OPEN band (60-90%)")
        log.info("   → H1 REFUTED, 逐层 c_k 区间不可组合")
        results['gate0_passed'] = False
    elif not all_under_90:
        log.info("\n❌ Issue #11 Gate 0 FAIL: ∃ layer >= 0.90 (违反 §6.7.4 safety)")
        results['gate0_passed'] = False
    else:
        log.info("\n✅ Issue #11 Gate 0 PASS: 三层全 OPEN + 全 < 0.90")
        log.info("   → H1 维持, 进入 Gate 1 (Stage 1 40-epoch 训练)")
        results['gate0_passed'] = True

    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_json, 'w') as f:
        json.dump(results, f, indent=2)
    log.info(f"\nOK 结果落盘: {args.output_json}")


if __name__ == "__main__":
    main()
