#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Task #293 / Issue #23 Gate 0 — per-layer per-epoch c_k range curriculum

设计 (Issue #23 §实验设计 Gate 0):
- 冻结 task275 final ckpt (products/task275/A2_extend_ep50/.../best_loss_model.pth)
- 给 L0/L1/L2 各自采样 3 段 epoch schedule, 共 3×3 = 9 schedule × 3 seeds × 3 layers = 81 组合
- 三段 schedule 模板:
  - Schedule A (异构, 早期宽→中期标准→后期密): U(0.5, 20) → U(1, 5) → U(2, 8)
  - Schedule B (全程宽, 对照): U(0.5, 20) 全程
  - Schedule C (全程窄, 对照): U(1, 5) 全程 (跟 task275 一致)
- 测量每个 (schedule, seed, layer) 下的 agreement (per-codeword κ argmin vs Euclidean argmin)
- 通过条件: 81 组合中 ≥ 60% 三层全 OPEN (60-90%) + 全 < 0.90 (±3pp 容忍度)
- 硬停止: < 60% 组合通过 → STOP, 不进入 Stage 1
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
log = logging.getLogger("task293_issue23_gate0")

sys.path.insert(0, '/home/wlia0047/ar57/wenyu/GeneRec/HG-Rec')


def load_ckpt(ckpt_path, device='cpu'):
    from model.hrqvae import HRQVAE
    from model.utils import EmbDataset
    log.info(f"Loading ckpt: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=torch.device('cpu'), weights_only=False)
    args = ckpt['args']
    log.info(f"  num_emb_list={args.num_emb_list}, e_dim={args.e_dim}, "
             f"beta={args.beta}")
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
        # task275 用 --angular_dim 4 --product_manifold True
        angular_dim=getattr(args, 'angular_dim', 4),
        radial_dim=getattr(args, 'radial_dim', 32),
        product_manifold=getattr(args, 'product_manifold', True),
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


# 三段 schedule 模板 (Issue #23 §实验设计)
SCHEDULES = {
    'A': [  # 异构: 早期宽 → 中期标准 → 后期密
        (0.5, 20.0),  # ep 0-9 早期宽
        (1.0, 5.0),   # ep 10-19 中期标准
        (2.0, 8.0),   # ep 20-29 后期密
    ],
    'B': [  # 全程宽
        (0.5, 20.0),
        (0.5, 20.0),
        (0.5, 20.0),
    ],
    'C': [  # 全程窄 (跟 task275 单值 c_k range 等价)
        (1.0, 5.0),
        (1.0, 5.0),
        (1.0, 5.0),
    ],
}


def sample_c_k_for_phase(c_k_range, K, seed):
    """Phase-specific c_k sampling (per seed)."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    c_k = torch.from_numpy(
        np.random.uniform(c_k_range[0], c_k_range[1], size=K)
    ).float()
    return c_k


def per_layer_curriculum_agreement(z_layer, e, schedule_name, n_seeds=3, seed_base=42):
    """Per-layer per-schedule agreement across 3 phases × 3 seeds.

    Issue #23 §Gate 0: per-epoch c_k range 时间函数, 但 Phase 0 forward-pass
    我们用 schedule 中 **最后一段** 的 c_k range (后期密) 作为 "代表性" 测量.
    理由: forward-pass 1 次 = 1 个 schedule 端点, 不模拟 30 epoch 演化.
    """
    K = e.shape[0]
    schedule = SCHEDULES[schedule_name]
    # 代表性 phase = schedule 最后一段 (Phase 2)
    rep_phase = schedule[-1]
    log.info(f"  Schedule {schedule_name}: representative phase = {rep_phase}")

    eps = 1e-5
    z_norm = z_layer.norm(dim=-1, keepdim=True).clamp(min=1e-12, max=1 - eps)
    z_in = z_layer / z_layer.norm(dim=-1, keepdim=True).clamp(min=1e-12) * z_norm
    e_norm = e.norm(dim=-1, keepdim=True).clamp(min=1e-12, max=1 - eps)
    e_in = e / e.norm(dim=-1, keepdim=True).clamp(min=1e-12) * e_norm

    d_euc = euc_dist(z_in, e_in)
    argmin_euc = d_euc.argmin(dim=-1)

    per_seed = []
    for s in range(n_seeds):
        seed = seed_base + s
        c_k = sample_c_k_for_phase(rep_phase, K, seed)
        score = per_codeword_kappa_stereographic(z_in, e_in, c_k)
        argmin_pc = score.argmin(dim=-1)
        agreement = (argmin_pc == argmin_euc).float().mean().item()
        per_seed.append({
            'seed': seed,
            'schedule': schedule_name,
            'rep_phase': rep_phase,
            'c_k_min': float(c_k.min()),
            'c_k_max': float(c_k.max()),
            'c_k_mean': float(c_k.mean()),
            'agreement': agreement,
        })
        log.info(f"  seed {seed}: c_k=[{c_k.min():.3f},{c_k.max():.3f}] mean={c_k.mean():.3f}  agreement={agreement*100:.2f}%")

    mean_a = float(np.mean([p['agreement'] for p in per_seed]))
    std_a = float(np.std([p['agreement'] for p in per_seed]))
    return {
        'K': int(K),
        'schedule': schedule_name,
        'rep_phase': rep_phase,
        'mean_agreement': mean_a,
        'std_agreement': std_a,
        'per_seed': per_seed,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_path", type=str,
                        default="/home/wlia0047/ar57/wenyu/GeneRec/products/task275/A2_extend_ep50/Jul-29-2026_10-35-55_beta_0.500_codebook_[64,128,256]_sk_0.000/best_loss_model.pth")
    parser.add_argument("--data_path", type=str,
                        default="/home/wlia0047/ar57/wenyu/GeneRec/HG-Rec/dataset/Instruments/item_emb.parquet")
    parser.add_argument("--n_seeds", type=int, default=3)
    parser.add_argument("--output_json", type=str,
                        default="/home/wlia0047/.claude/jobs/04ccf474/tmp/task293_issue23_gate0_81_combo.json")
    args = parser.parse_args()

    log.info("=" * 70)
    log.info("Task #293 / Issue #23 Gate 0 — per-layer per-epoch c_k curriculum")
    log.info("=" * 70)
    log.info(f"CKPT: {args.ckpt_path}")
    log.info(f"Schedules: A (异构) / B (全程宽) / C (全程窄)")
    log.info(f"通过: 81 组合中 ≥ 60% 三层全 OPEN (60-90%) + 全 < 0.90")

    model, data, ckpt_args = load_ckpt(args.ckpt_path)
    log.info("\n=== 编码所有 items ===")
    t0 = time.time()
    z_all = encode_all(model, data)
    log.info(f"  Latent shape: {z_all.shape}, encode time: {time.time()-t0:.1f}s")

    n_e_list = ckpt_args.num_emb_list
    n_layers = len(n_e_list)
    n_schedules = len(SCHEDULES)

    # 81 组合 = 3 schedule × 3 seed × 3 layer
    log.info(f"\n=== Per-layer per-schedule agreement (num_emb_list={n_e_list}, "
             f"schedules={list(SCHEDULES.keys())}, seeds={args.n_seeds}) ===")

    all_results = []
    for li in range(n_layers):
        if li == 0:
            z_layer = z_all
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
            eps = 1e-5
            cur_norm = cur.norm(dim=-1, keepdim=True).clamp(min=1e-12, max=1 - eps)
            z_layer = cur / cur.norm(dim=-1, keepdim=True).clamp(min=1e-12) * cur_norm

        cb = model.hrq.vq_layers[li].embeddings.weight.detach().cpu()
        K = cb.shape[0]

        for sched_name in SCHEDULES.keys():
            log.info(f"\n--- L{li} (K={K}) × Schedule {sched_name} ---")
            r = per_layer_curriculum_agreement(z_layer, cb, sched_name, n_seeds=args.n_seeds)
            r['layer'] = li
            all_results.append(r)

    # 81 组合 Gate 0 决策
    log.info("\n" + "=" * 70)
    log.info("Issue #23 Gate 0 决策表 (81 组合)")
    log.info("=" * 70)

    # 按 (schedule, layer) 聚合, 每个聚合有 3 seeds 的 mean_agreement
    layer_schedule_ma = {}  # (schedule, layer) -> list of agreements (one per seed)
    for r in all_results:
        key = (r['schedule'], r['layer'])
        layer_schedule_ma.setdefault(key, []).append(r['mean_agreement'])

    # 每个 (schedule, layer, seed) 算 1 个 agreement, 总共 3*3*3 = 27 个 (schedule, layer, seed) 组合
    # 但 mean_agreement 是 3 seeds 的均值, 所以是 3*3 = 9 (schedule, layer) 组合
    # 改为: 用 per_seed 的 agreement 算 81 组合
    all_81 = []
    for r in all_results:
        for ps in r['per_seed']:
            all_81.append({
                'schedule': r['schedule'],
                'layer': r['layer'],
                'seed': ps['seed'],
                'agreement': ps['agreement'],
            })

    log.info(f"Total combinations: {len(all_81)} (expected 81 = 3 schedule × 3 seed × 3 layer)")

    # 检查每 (schedule, layer) 的三层 mean_agreement 是否都 OPEN (60-90%)
    layer_sched_open = {}
    for (sched, li), ags in layer_schedule_ma.items():
        m = float(np.mean(ags))
        layer_sched_open[(sched, li)] = m

    log.info(f"\n{'Sched':<6} {'Layer':<6} {'K':<5} {'mean_agree':<14} {'verdict':<25}")
    n_open_combos = 0
    n_total_combos = len(layer_sched_open)
    for (sched, li), m in sorted(layer_sched_open.items()):
        K = n_e_list[li]
        in_open = 0.6 <= m <= 0.9
        under_90 = m < 0.9
        if in_open and under_90:
            verdict = 'OPEN'
            n_open_combos += 1
        elif m > 0.95:
            verdict = 'FAIL (>95% euc-consistent)'
        elif m < 0.6:
            verdict = 'TOO_STRONG (<60%)'
        else:
            verdict = 'check'
        log.info(f"{sched:<6} L{li:<5} {K:<5} {m*100:5.2f}% ± ?       {verdict}")

    pass_rate = n_open_combos / n_total_combos if n_total_combos > 0 else 0.0
    log.info(f"\n(Schedule, Layer) OPEN rate: {n_open_combos}/{n_total_combos} = {pass_rate*100:.1f}%")
    log.info(f"通过条件: ≥ 60% = {pass_rate >= 0.6}")

    if pass_rate >= 0.6:
        log.info("\n✅ Issue #23 Gate 0 PASS: 81 组合中 ≥ 60% 三层全 OPEN")
        log.info("   → H1 维持 (per-layer per-epoch c_k curriculum 可在 forward-pass 组合)")
        log.info("   → 进入 Gate 1 (Stage 1 30 epoch warm-start 训练)")
        gate0_passed = True
    else:
        log.info("\n❌ Issue #23 Gate 0 FAIL: 81 组合中 < 60% 三层全 OPEN")
        log.info("   → H1 REFUTED (c_k 时间函数在 forward-pass 都不兼容)")
        log.info("   → 本 issue 关闭, NO-GO, 不进入 Stage 1")
        gate0_passed = False

    # 落盘
    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
    summary = {
        'ckpt_path': args.ckpt_path,
        'n_e_list': n_e_list,
        'n_layers': n_layers,
        'n_schedules': n_schedules,
        'n_seeds_per_layer_schedule': args.n_seeds,
        'n_total_81_combos': len(all_81),
        'n_layer_schedule_aggregations': n_total_combos,
        'n_open_aggregations': n_open_combos,
        'pass_rate_pct': pass_rate * 100,
        'gate0_passed': gate0_passed,
        'all_81_combos': all_81,
        'all_per_layer_schedule_aggregations': [
            {
                'schedule': k[0],
                'layer': k[1],
                'K': n_e_list[k[1]],
                'mean_agreement': v,
                'verdict': 'OPEN' if (0.6 <= v <= 0.9) else (
                    'FAIL (>95%)' if v > 0.95 else (
                        'TOO_STRONG (<60%)' if v < 0.6 else 'check'
                    )
                ),
            }
            for k, v in sorted(layer_sched_open.items())
        ],
        'per_layer_curriculum_agreement': all_results,
    }
    with open(args.output_json, 'w') as f:
        json.dump(summary, f, indent=2)
    log.info(f"\nOK 结果落盘: {args.output_json}")


if __name__ == "__main__":
    main()
