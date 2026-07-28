#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Task #232 — Issue #7 调参版: Gromov product argmax weight/entropy/spread-norm sweep.

用户 2026-07-28 Issue #7 提议:
  三个候选 fix:
  (A) weight decay: score = ρ_e · weight_l − d(z,e), sweep weight_l ∈ {0.3, 0.5, 0.7}
  (B) entropy reg: 乘以 exp(-H)/K 抑制深层码字过度选用
  (C) spread norm: 用 |ρ_e − ρ_mean| 归一化

Goal: L1/L2 进 60-90% open band (current 50.21%/40.48% TOO_STRONG), L0 保持 (79.65%).

CPU only, 不占卡. 用 HG-Rec baseline (#84) ckpt.
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
log = logging.getLogger("task232_gromov_sweep")

sys.path.insert(0, '/home/wlia0047/ar57/wenyu/GeneRec/HG-Rec')


def load_baseline_ckpt(ckpt_path, device='cpu'):
    """Load HG-Rec baseline ckpt."""
    from model.hrqvae import HRQVAE
    from model.utils import EmbDataset

    log.info(f"Loading ckpt: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=torch.device('cpu'), weights_only=False)
    args = ckpt['args']
    log.info(f"  num_emb_list={args.num_emb_list}, e_dim={args.e_dim}, "
             f"curvature={getattr(args, 'curvature', 1.0)}")

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
    """Encoder forward pass."""
    from torch.utils.data import DataLoader
    loader = DataLoader(data, batch_size=batch_size, shuffle=False, num_workers=2)
    latents = []
    for batch in loader:
        batch = batch.to(device)
        z = model.encoder(batch)
        latents.append(z.cpu())
    return torch.cat(latents, dim=0)


def euc_dist(z, e):
    """欧式距离矩阵 (N, K) = ||z - e||^2."""
    z_sq = (z ** 2).sum(dim=-1, keepdim=True)
    e_sq = (e ** 2).sum(dim=-1, keepdim=True).t()
    ze = z @ e.t()
    return z_sq + e_sq - 2 * ze


def poincare_dist_matrix(z, e, c=1.0):
    """Poincaré 距离矩阵 (N, K)."""
    eps = 1e-5
    z_norm = z.norm(dim=-1, keepdim=True).clamp(min=1e-12, max=1 - eps)
    e_norm = e.norm(dim=-1, keepdim=True).clamp(min=1e-12, max=1 - eps)
    z_in = z / z.norm(dim=-1, keepdim=True).clamp(min=1e-12) * z_norm
    e_in = e / e.norm(dim=-1, keepdim=True).clamp(min=1e-12) * e_norm
    z_sq = (z_in ** 2).sum(dim=-1, keepdim=True)
    e_sq = (e_in ** 2).sum(dim=-1, keepdim=True)
    ze = z_in @ e_in.t()
    eucl_d_sq = z_sq + e_sq.t() - 2 * ze
    denom = (1 - z_sq) * (1 - e_sq.t())
    x = 1 + 2 * eucl_d_sq / denom.clamp(min=1e-12)
    x = x.clamp(min=1.0 + 1e-12)
    return torch.acosh(x)


def gromov_score_matrix(z, e, c=1.0, variant='baseline', weight_l=1.0):
    """Gromov product 矩阵 (N, K).

    variants:
      baseline:  score = ½ (d(0,z) + d(0,e) - d(z,e)) ∝ ρ_e - d(z,e)  (default weight_l=1.0)
      (A) weight: score = ρ_e · weight_l − d(z,e)
      (B) entropy: score = (ρ_e − d(z,e)) · exp(−H)/K
      (C) spread:  score = |ρ_e − ρ_mean| − d(z,e)
    """
    eps = 1e-5
    z_norm = z.norm(dim=-1, keepdim=True).clamp(min=1e-12, max=1 - eps)
    e_norm = e.norm(dim=-1, keepdim=True).clamp(min=1e-12, max=1 - eps)
    z_in = z / z.norm(dim=-1, keepdim=True).clamp(min=1e-12) * z_norm
    e_in = e / e.norm(dim=-1, keepdim=True).clamp(min=1e-12) * e_norm

    rho_z = 2 * torch.atanh(z_in.norm(dim=-1).clamp(max=1 - 1e-6))  # (N,)
    rho_e = 2 * torch.atanh(e_in.norm(dim=-1).clamp(max=1 - 1e-6))  # (K,)

    d_ze = poincare_dist_matrix(z_in, e_in, c=c)  # (N, K)

    if variant == 'baseline':
        # score = ρ_e - d(z,e)  (argmax)
        score = rho_e.view(1, -1) - d_ze
    elif variant == 'weight':
        # (A) score = ρ_e · weight_l − d(z,e)
        score = weight_l * rho_e.view(1, -1) - d_ze
    elif variant == 'entropy':
        # (B) score = (ρ_e − d(z,e)) · exp(−H)/K
        # 熵 H = -Σ p_k log p_k, p_k = softmax(score_unnorm)
        # 取 ρ_e − d(z,e) 作 unnorm, normalize 后乘 exp(-H)/K
        score_unnorm = rho_e.view(1, -1) - d_ze
        p = torch.softmax(score_unnorm, dim=-1)  # (N, K)
        H = -(p * torch.log(p.clamp(min=1e-12))).sum(dim=-1)  # (N,)
        K_size = float(score_unnorm.shape[1])
        score = score_unnorm * torch.exp(-H).view(-1, 1) / K_size
    elif variant == 'spread':
        # (C) score = |ρ_e − ρ_mean| − d(z,e)
        rho_mean = rho_e.mean()
        score = (rho_e - rho_mean).abs().view(1, -1) - d_ze
    else:
        raise ValueError(f"Unknown variant: {variant}")

    return score


def criterion_check(z_layer, e, variant='baseline', weight_l=1.0):
    """Gromov argmax vs 欧式 argmin 一致率."""
    N, e_dim = z_layer.shape
    K = e.shape[0]
    log.info(f"  Latent shape: {z_layer.shape}, codebook: {e.shape}, variant={variant}, weight_l={weight_l}")

    eps = 1e-5
    z_norm = z_layer.norm(dim=-1, keepdim=True).clamp(min=1e-12, max=1 - eps)
    z_in = z_layer / z_layer.norm(dim=-1, keepdim=True).clamp(min=1e-12) * z_norm
    e_norm = e.norm(dim=-1, keepdim=True).clamp(min=1e-12, max=1 - eps)
    e_in = e / e.norm(dim=-1, keepdim=True).clamp(min=1e-12) * e_norm

    d_euc = euc_dist(z_in, e_in)
    argmin_euc = d_euc.argmin(dim=-1)

    score = gromov_score_matrix(z_in, e_in, c=1.0, variant=variant, weight_l=weight_l)
    argmax_gromov = score.argmax(dim=-1)

    agreement = (argmax_gromov == argmin_euc).float().mean().item()
    log.info(f"  >>> Gromov argmax vs 欧式 argmin 一致率: {agreement*100:.2f}%")

    verdict = (
        'FAIL (>95% 一致, Gromov 影响被淹没)' if agreement > 0.95
        else 'OPEN (60-95% 一致, Gromov 有区分力) OK' if agreement > 0.60
        else 'TOO_STRONG (<60% 一致, 几何主导过头)'
    )

    return {
        'N': int(N),
        'K': int(K),
        'e_dim': int(e_dim),
        'variant': variant,
        'weight_l': weight_l,
        'agreement_gromov_vs_euc': agreement,
        'verdict_escape_two': verdict,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_path", type=str,
                        default="/home/wlia0047/ar57/wenyu/GeneRec/products/task84/ckpt/Instruments/Jul-23-2026_20-08-06_beta_1.000_codebook_[64,128,256]_sk_0.000/best_loss_model.pth")
    parser.add_argument("--data_path", type=str,
                        default="/home/wlia0047/ar57/wenyu/GeneRec/HG-Rec/dataset/Instruments/item_emb.parquet")
    parser.add_argument("--output_json", type=str,
                        default="/home/wlia0047/.claude/jobs/04ccf474/tmp/task232_gromov_phase0_results.json")
    args = parser.parse_args()

    log.info("=" * 70)
    log.info("Task #232 - Issue #7 调参版 (Gromov weight/entropy/spread-norm sweep)")
    log.info("=" * 70)
    log.info("Variants: baseline / weight(0.3/0.5/0.7) / entropy / spread")

    model, data, ckpt_args = load_baseline_ckpt(args.ckpt_path)
    log.info("\n=== 编码所有 items ===")
    t0 = time.time()
    z_all = encode_all(model, data)
    log.info(f"  Latent shape: {z_all.shape}, encode time: {time.time()-t0:.1f}s")

    n_e_list = ckpt_args.num_emb_list
    log.info(f"\n=== Per-layer Gromov sweep (num_emb_list={n_e_list}) ===")

    all_results = {
        "task": "task232_gromov_phase0_weight_sweep",
        "ckpt": args.ckpt_path,
        "n_items": int(z_all.shape[0]),
        "e_dim": int(z_all.shape[1]),
        "n_e_list": list(n_e_list),
        "variants": {},
    }

    eps = 1e-5
    z_norm = z_all.norm(dim=-1, keepdim=True).clamp(min=1e-12, max=1 - eps)
    z_proj = z_all / z_all.norm(dim=-1, keepdim=True).clamp(min=1e-12) * z_norm

    # variants: baseline (复现 task219) + Issue #7 提议的 (A)/(B)/(C)
    variants_to_test = [
        ("baseline", 1.0),
        ("weight", 0.3),
        ("weight", 0.5),
        ("weight", 0.7),
        ("entropy", 1.0),
        ("spread", 1.0),
    ]

    for li in range(len(n_e_list)):
        K = n_e_list[li]
        e = model.hrq.vq_layers[li].embeddings.weight.detach().cpu()

        if li == 0:
            z_layer = z_proj
        else:
            from torch.utils.data import DataLoader
            loader = DataLoader(data, batch_size=256, shuffle=False, num_workers=2)
            cur = z_all.clone()
            for lj in range(li):
                cb = model.hrq.vq_layers[lj].embeddings.weight.detach().cpu()
                d = euc_dist(cur, cb)
                idx = d.argmin(dim=-1)
                z_q = cb[idx]
                cur = cur - z_q
            cur_norm = cur.norm(dim=-1, keepdim=True).clamp(min=1e-12, max=1 - eps)
            z_layer = cur / cur.norm(dim=-1, keepdim=True).clamp(min=1e-12) * cur_norm
            log.info(f"  Layer {li} residual norm range (after proj): [{z_layer.norm(dim=-1).min().item():.3f}, {z_layer.norm(dim=-1).max().item():.3f}]")

        e_norm = e.norm(dim=-1, keepdim=True).clamp(min=1e-12, max=1 - eps)
        e_proj = e / e.norm(dim=-1, keepdim=True).clamp(min=1e-12) * e_norm
        log.info(f"\n  --- Layer {li} (K={K}) ---")

        for variant, weight_l in variants_to_test:
            log.info(f"\n  >>> variant={variant}, weight_l={weight_l}")
            layer_results = criterion_check(
                z_layer, e_proj,
                variant=variant,
                weight_l=weight_l,
            )
            var_key = f"{variant}_w{weight_l}" if variant == "weight" else variant
            if str(li) not in all_results["variants"]:
                all_results["variants"][str(li)] = {}
            all_results["variants"][str(li)][var_key] = layer_results

    log.info("\n" + "=" * 70)
    log.info("汇总决策 (Gromov weight/entropy/spread-norm sweep):")
    log.info("=" * 70)
    header = f"{'Layer':<6} {'K':<5} {'variant':<20} {'agreement':<12} {'verdict':<35}"
    log.info(header)
    for li in range(len(n_e_list)):
        for variant, weight_l in variants_to_test:
            var_key = f"{variant}_w{weight_l}" if variant == "weight" else variant
            r = all_results["variants"][str(li)][var_key]
            log.info(f"{li:<6} {n_e_list[li]:<5} {var_key:<20} {r['agreement_gromov_vs_euc']*100:5.2f}%        {r['verdict_escape_two']}")

    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_json, "w") as f:
        json.dump(all_results, f, indent=2)
    log.info(f"\nOK 结果落盘: {args.output_json}")


if __name__ == "__main__":
    main()
