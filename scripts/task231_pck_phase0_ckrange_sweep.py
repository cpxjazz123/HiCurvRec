#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Task #231 — Issue #6 调参版: Per-Codeword κ c_k range sweep.

用户 2026-07-28 Issue #6 提议:
  Re-run Task #218's Phase 0 script with c_k ~ Uniform(0.5,5) and Uniform(1,5),
  3 seeds each. Goal: L0 进 60-90% open band (current 28.90% TOO_STRONG with
  Uniform(0.5,20)), L1/L2 保持.

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
log = logging.getLogger("task231_pck_sweep")

sys.path.insert(0, '/home/wlia0047/ar57/wenyu/GeneRec/HG-Rec')


def load_baseline_ckpt(ckpt_path, device='cpu'):
    """Load HG-Rec baseline ckpt, instantiate HRQVAE, load weights."""
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
    """Encoder forward pass: items → latent."""
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


def per_codeword_kappa_stereographic(z, e, c_k):
    """Per-codeword κ-Stereographic distance matrix (N, K).

    score_k = (1/sqrt(c_k)) · arccosh( 1 + 2*c_k*||z-e_k||^2 / [(1-c_k*||z||^2)(1-c_k*||e_k||^2)] )
    """
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
    score = inv_sqrt_c.view(1, -1) * torch.acosh(inner)
    return score


def criterion_check(z_layer, e, c_k_range, n_seeds=3, seed_base=42):
    """Per-codeword argmin vs 欧式 argmin 一致率 (3 seeds 平均)."""
    N, e_dim = z_layer.shape
    K = e.shape[0]
    log.info(f"  Latent shape: {z_layer.shape}, codebook: {e.shape}, e_dim={e_dim}")
    log.info(f"  c_k_range = {c_k_range}, n_seeds = {n_seeds}")

    eps = 1e-5
    z_norm = z_layer.norm(dim=-1, keepdim=True).clamp(min=1e-12, max=1 - eps)
    z_in = z_layer / z_layer.norm(dim=-1, keepdim=True).clamp(min=1e-12) * z_norm
    e_norm = e.norm(dim=-1, keepdim=True).clamp(min=1e-12, max=1 - eps)
    e_in = e / e.norm(dim=-1, keepdim=True).clamp(min=1e-12) * e_norm

    d_euc = euc_dist(z_in, e_in)
    argmin_euc = d_euc.argmin(dim=-1)

    per_seed_rates = []
    per_seed_details = []
    for s in range(n_seeds):
        torch.manual_seed(seed_base + s)
        np.random.seed(seed_base + s)
        c_k = torch.from_numpy(
            np.random.uniform(c_k_range[0], c_k_range[1], size=K)
        ).float()
        score = per_codeword_kappa_stereographic(z_in, e_in, c_k)
        argmin_pc = score.argmin(dim=-1)
        agreement = (argmin_pc == argmin_euc).float().mean().item()
        per_seed_rates.append(agreement)
        log.info(f"  Seed {seed_base + s}: 一致率 = {agreement*100:.2f}% "
                 f"(c_k range [{c_k.min().item():.3f}, {c_k.max().item():.3f}], "
                 f"mean={c_k.mean().item():.3f})")
        per_seed_details.append({
            'seed': seed_base + s,
            'c_k_min': float(c_k.min()),
            'c_k_max': float(c_k.max()),
            'c_k_mean': float(c_k.mean()),
            'agreement': agreement,
        })

    mean_agreement = float(np.mean(per_seed_rates))
    std_agreement = float(np.std(per_seed_rates))
    log.info(f"  >>> 一致率 mean = {mean_agreement*100:.2f}% +/- {std_agreement*100:.2f}%")

    verdict_escape_one = (
        'FAIL (>95% 一致, per-codeword κ 影响被淹没)' if mean_agreement > 0.95
        else 'OPEN (60-95% 一致, per-codeword κ 有区分力) OK' if mean_agreement > 0.60
        else 'TOO_STRONG (<60% 一致, 几何主导过头, 可能破坏分配)'
    )

    return {
        'N': int(N),
        'K': int(K),
        'e_dim': int(e_dim),
        'c_k_range': list(c_k_range),
        'n_seeds': n_seeds,
        'mean_agreement': mean_agreement,
        'std_agreement': std_agreement,
        'per_seed': per_seed_details,
        'verdict_escape_one': verdict_escape_one,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_path", type=str,
                        default="/home/wlia0047/ar57/wenyu/GeneRec/products/task84/ckpt/Instruments/Jul-23-2026_20-08-06_beta_1.000_codebook_[64,128,256]_sk_0.000/best_loss_model.pth")
    parser.add_argument("--data_path", type=str,
                        default="/home/wlia0047/ar57/wenyu/GeneRec/HG-Rec/dataset/Instruments/item_emb.parquet")
    parser.add_argument("--n_seeds", type=int, default=3,
                        help="随机 c_k 抽样次数 per range")
    parser.add_argument("--output_json", type=str,
                        default="/home/wlia0047/.claude/jobs/04ccf474/tmp/task231_pck_phase0_results.json")
    args = parser.parse_args()

    log.info("=" * 70)
    log.info("Task #231 - Issue #6 调参版 (Per-Codeword κ c_k range sweep)")
    log.info("=" * 70)
    log.info("Sweep 3 ranges: Uniform(0.5,20) baseline / Uniform(0.5,5) / Uniform(1,5)")

    model, data, ckpt_args = load_baseline_ckpt(args.ckpt_path)
    log.info("\n=== 编码所有 items ===")
    t0 = time.time()
    z_all = encode_all(model, data)
    log.info(f"  Latent shape: {z_all.shape}, encode time: {time.time()-t0:.1f}s")

    n_e_list = ckpt_args.num_emb_list
    log.info(f"\n=== Per-layer 判据检查 (num_emb_list={n_e_list}) ===")

    all_results = {
        "task": "task231_pck_phase0_ckrange_sweep",
        "ckpt": args.ckpt_path,
        "n_items": int(z_all.shape[0]),
        "e_dim": int(z_all.shape[1]),
        "n_e_list": list(n_e_list),
        "n_seeds": args.n_seeds,
        "ranges": {},
    }

    eps = 1e-5
    z_norm = z_all.norm(dim=-1, keepdim=True).clamp(min=1e-12, max=1 - eps)
    z_proj = z_all / z_all.norm(dim=-1, keepdim=True).clamp(min=1e-12) * z_norm

    ranges = [
        ("Uniform(0.5,20)", 0.5, 20.0),
        ("Uniform(0.5,5)", 0.5, 5.0),
        ("Uniform(1,5)", 1.0, 5.0),
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

        for range_name, c_low, c_high in ranges:
            log.info(f"\n  >>> Range: {range_name}")
            layer_results = criterion_check(
                z_layer, e_proj,
                c_k_range=(c_low, c_high),
                n_seeds=args.n_seeds,
            )
            if str(li) not in all_results["ranges"]:
                all_results["ranges"][str(li)] = {}
            all_results["ranges"][str(li)][range_name] = layer_results

    log.info("\n" + "=" * 70)
    log.info("汇总决策 (Per-Codeword κ c_k range sweep):")
    log.info("=" * 70)
    header = f"{'Layer':<6} {'K':<5} {'Range':<18} {'agreement':<25} {'verdict':<35}"
    log.info(header)
    for li in range(len(n_e_list)):
        for range_name, _, _ in ranges:
            r = all_results["ranges"][str(li)][range_name]
            log.info(f"{li:<6} {n_e_list[li]:<5} {range_name:<18} {r['mean_agreement']*100:5.2f}% +/- {r['std_agreement']*100:5.2f}%      {r['verdict_escape_one']}")

    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_json, "w") as f:
        json.dump(all_results, f, indent=2)
    log.info(f"\nOK 结果落盘: {args.output_json}")


if __name__ == "__main__":
    main()
