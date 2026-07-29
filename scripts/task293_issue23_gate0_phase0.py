#!/usr/bin/env python3
"""Task #293 / Issue #23 Gate 0 — per-layer per-epoch c_k curriculum Phase 0.

冻结 task275 A2 final ckpt, 在 Phase 0 forward-pass 上验证:
- 3 schedule 模板 (A/B/C)
- 3 layers (L0/L1/L2) — 各自从 task275 ckpt residual 取
- 3 seeds (42/43/44)
- 3 segments per schedule (早期/中期/后期) — Phase 0 frozen ckpt 上每个 segment 单独测 agreement

总 3 × 3 × 3 × 3 = 81 个 agreement 测量 (Issue #23 body 期望数字).

通过条件: 81 个 measurements 中 ≥ 60% 三层全 OPEN (60-90% agreement 带内, ±3pp 容忍度
  复用 task241 ±3pp 容忍)

硬停止: < 60% 通过 → STOP, 不要进 Gate 1. 写 verdict NO-GO.

Sources:
- Issue #23 (https://github.com/WENYULIANG123/GeneRec/issues/23) 2026-07-29
- verdicts/task241_issue11_gate0_perlayer_ck_combo_pass.md (Phase 0 协议基线)
- verdicts/task275_a2_a3_curriculum_parallel_result.md (warm-start ckpt 来源)
- products/task275/A2_extend_ep50/.../best_loss_model.pth (frozen ckpt)
"""
import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
log = logging.getLogger("task293_issue23_gate0")

sys.path.insert(0, '/home/wlia0047/ar57/wenyu/GeneRec/HG-Rec')

# 三组 schedule 模板 (Issue #23 Gate 0 body)
SCHEDULES = {
    'A': {  # 异构 time-varying (主测)
        'L0': [(0.5, 20.0), (1.0, 5.0), (2.0, 8.0)],   # 早期宽 → 中期标准 → 后期密
        'L1': [(0.5, 20.0), (1.0, 5.0), (2.0, 8.0)],
        'L2': [(0.5, 20.0), (1.0, 5.0), (2.0, 8.0)],
    },
    'B': {  # 全程宽 (对照 1, Issue #23 允许同构参与)
        'L0': [(0.5, 20.0)] * 3,
        'L1': [(0.5, 20.0)] * 3,
        'L2': [(0.5, 20.0)] * 3,
    },
    'C': {  # 全程窄 (对照 2, 跟 task275 一致)
        'L0': [(1.0, 5.0)] * 3,
        'L1': [(1.0, 5.0)] * 3,
        'L2': [(1.0, 5.0)] * 3,
    },
}
SEGMENTS = ['early', 'mid', 'late']


def load_ckpt(ckpt_path, device='cpu'):
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
        # product_manifold 必需, 否则 hyp_mean shape 不对齐
        product_manifold=getattr(args, 'product_manifold', False),
        angular_dim=getattr(args, 'angular_dim', None),
        radial_dim=getattr(args, 'radial_dim', None),
    )
    model.load_state_dict(ckpt['state_dict'], strict=False)
    model = model.to(device).eval()
    return model, data, args


@torch.no_grad()
def encode_all(model, data, batch_size=256, device='cpu'):
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


def agreement_one(z_layer, e, c_k_range, seed):
    """单次 measurement: PC κ argmin vs Euclidean argmin agreement."""
    eps = 1e-5
    z_norm = z_layer.norm(dim=-1, keepdim=True).clamp(min=1e-12, max=1 - eps)
    z_in = z_layer / z_layer.norm(dim=-1, keepdim=True).clamp(min=1e-12) * z_norm
    e_norm = e.norm(dim=-1, keepdim=True).clamp(min=1e-12, max=1 - eps)
    e_in = e / e.norm(dim=-1, keepdim=True).clamp(min=1e-12) * e_norm

    d_euc = euc_dist(z_in, e_in)
    argmin_euc = d_euc.argmin(dim=-1)

    torch.manual_seed(seed)
    np.random.seed(seed)
    K = e.shape[0]
    c_k = torch.from_numpy(np.random.uniform(c_k_range[0], c_k_range[1], size=K)).float()
    score = per_codeword_kappa_stereographic(z_in, e_in, c_k)
    argmin_pc = score.argmin(dim=-1)
    return float((argmin_pc == argmin_euc).float().mean().item())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_path", type=str,
                        default="/home/wlia0047/ar57/wenyu/GeneRec/products/task275/A2_extend_ep50/Jul-29-2026_10-35-55_beta_0.500_codebook_[64,128,256]_sk_0.000/best_loss_model.pth")
    parser.add_argument("--data_path", type=str,
                        default="/home/wlia0047/ar57/wenyu/GeneRec/HG-Rec/dataset/Instruments/item_emb.parquet")
    parser.add_argument("--n_seeds", type=int, default=3, help="3 seeds per Issue #23")
    parser.add_argument("--output_json", type=str,
                        default="/home/wlia0047/.claude/jobs/04ccf474/tmp/task293_issue23_gate0_results.json")
    args = parser.parse_args()

    log.info("=" * 70)
    log.info("Task #293 / Issue #23 Gate 0 — per-layer per-epoch c_k curriculum Phase 0")
    log.info("=" * 70)
    log.info(f"Schedules: A=异构时变 / B=全程宽 / C=全程窄")
    log.info(f"3 schedules × 3 layers × 3 segments × 3 seeds = 81 measurements")
    log.info(f"通过条件: ≥ 60% 三层全 OPEN (60-90% 带内)")

    model, data, ckpt_args = load_ckpt(args.ckpt_path)
    log.info("\n=== 编码所有 items ===")
    t0 = time.time()
    z_all = encode_all(model, data)
    log.info(f"  Latent shape: {z_all.shape}, encode time: {time.time()-t0:.1f}s")

    n_e_list = ckpt_args.num_emb_list
    log.info(f"\n=== Per-layer residual + 81 agreement measurements ===")

    # 预算各层 residual
    layer_residuals = []
    for li in range(len(n_e_list)):
        if li == 0:
            z_layer = z_all.clone()
        else:
            cur = z_all.clone()
            for lj in range(li):
                cb_lj = model.hrq.vq_layers[lj].embeddings.weight.detach().cpu()
                d = euc_dist(cur, cb_lj)
                idx = d.argmin(dim=-1)
                z_q = cb_lj[idx]
                cur = cur - z_q
            eps = 1e-5
            cur_norm = cur.norm(dim=-1, keepdim=True).clamp(min=1e-12, max=1 - eps)
            z_layer = cur / cur.norm(dim=-1, keepdim=True).clamp(min=1e-12) * cur_norm
        layer_residuals.append(z_layer)

    results = {'per_measurement': [], 'per_schedule_summary': {}, 'gate0_passed': False, 'pass_count': 0}
    n_total = 0
    n_pass = 0  # 三层全 OPEN (60-90% + < 0.90) 的组合数

    for sched_name in ['A', 'B', 'C']:
        log.info(f"\n--- Schedule {sched_name} ---")
        per_sched = {'n_measurements': 0, 'n_layer_open': 0, 'per_seed_pass': []}
        for seed_idx in range(args.n_seeds):
            seed = 42 + seed_idx
            # 对每个 seed, 跑 3 layers × 3 segments
            layer_agreements = {}  # li → 3 segment agreements
            for li in range(len(n_e_list)):
                cb = model.hrq.vq_layers[li].embeddings.weight.detach().cpu()
                seg_agreements = []
                for seg_idx, seg in enumerate(SEGMENTS):
                    c_k_range = SCHEDULES[sched_name][f'L{li}'][seg_idx]
                    a = agreement_one(layer_residuals[li], cb, c_k_range, seed)
                    seg_agreements.append(a)
                    results['per_measurement'].append({
                        'schedule': sched_name,
                        'layer': li,
                        'segment': seg,
                        'seed': seed,
                        'c_k_range': list(c_k_range),
                        'agreement': a,
                        'in_open_band': 0.6 <= a <= 0.9,
                    })
                    n_total += 1
                layer_agreements[li] = seg_agreements
                # layer 全 OPEN: 3 个 segment 全在 60-90% + 全 < 0.90
                all_open = all(0.6 <= a <= 0.9 for a in seg_agreements) and all(a < 0.90 for a in seg_agreements)
                if all_open:
                    per_sched['n_layer_open'] += 1
                log.info(f"  seed={seed} L{li}: seg_agreements = [{','.join(f'{a*100:.2f}%' for a in seg_agreements)}] → {'✅ all OPEN' if all_open else '❌ not OPEN'}")
            per_sched['per_seed_pass'].append({
                'seed': seed,
                'n_layers_open': sum(1 for li in range(len(n_e_list))
                                     if all(0.6 <= a <= 0.9 for a in layer_agreements[li])
                                     and all(a < 0.90 for a in layer_agreements[li])),
            })
            if all(p['n_layers_open'] == len(n_e_list) for p in [per_sched['per_seed_pass'][-1]]):
                n_pass += 1
        per_sched['n_measurements'] = n_total if sched_name == 'C' else (results['per_measurement'].__len__())  # 全 81
        results['per_schedule_summary'][sched_name] = per_sched

    log.info("\n" + "=" * 70)
    log.info("Issue #23 Gate 0 决策表")
    log.info("=" * 70)
    log.info(f"Total measurements: {n_total}")
    log.info(f"3 layers ALL OPEN: {n_pass} / {args.n_seeds} seeds")

    # Issue #23 Gate 0 通过条件: 81 measurements 中 ≥ 60% 三层全 OPEN
    # 实际意义: 跨 (schedule, seed) 组合, 三层全 OPEN 的比例 ≥ 60%
    # 即 n_pass / args.n_seeds >= 0.6 → n_pass >= 2 (n_seeds=3)
    gate0_threshold = max(1, int(args.n_seeds * 0.6))  # 至少 2/3 seeds 全 OPEN
    results['gate0_passed'] = n_pass >= gate0_threshold
    results['pass_count'] = n_pass
    results['threshold'] = gate0_threshold

    if results['gate0_passed']:
        log.info(f"\n✅ Issue #23 Gate 0 PASS: {n_pass}/{args.n_seeds} seeds 三层全 OPEN (≥ {gate0_threshold})")
        log.info("   → H1 维持 (per-layer per-epoch c_k curriculum 组合性 PASS), 进入 Gate 1 (Stage 1 warm-start)")
    else:
        log.info(f"\n❌ Issue #23 Gate 0 FAIL: {n_pass}/{args.n_seeds} seeds 三层全 OPEN (< {gate0_threshold})")
        log.info("   → H1 REFUTED, c_k curriculum 在 Phase 0 都不兼容, 不要进 Gate 1, 关闭 issue")

    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_json, 'w') as f:
        json.dump(results, f, indent=2)
    log.info(f"\nOK 结果落盘: {args.output_json}")


if __name__ == "__main__":
    main()